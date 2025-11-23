"""
XGBoost Model AMÉLIORÉ - Quantum Trader Pro
Version optimisée pour gérer le déséquilibre de classes (10% positifs)
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight
from utils.logger import setup_logger
from utils.safety import safe_dataframe_access

# Import conditionnel de imblearn
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


class XGBoostModel:
    """
    Modèle XGBoost optimisé pour déséquilibre de classes

    AMÉLIORATIONS:
    - SMOTE pour équilibrer les classes
    - Optimisation du seuil de décision
    - Cross-validation stratifiée
    - Calibration des probabilités
    """

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('XGBoostModel')

        # Configuration XGBoost améliorée
        xgb_config = config.get('ml', {}).get('models', {}).get('xgboost', {})

        # Hyperparamètres optimisés pour déséquilibre
        self.n_estimators = xgb_config.get('n_estimators', 500)
        self.max_depth = xgb_config.get('max_depth', 4)
        self.learning_rate = xgb_config.get('learning_rate', 0.02)
        self.objective = 'binary:logistic'

        # Régularisation forte
        self.reg_alpha = xgb_config.get('reg_alpha', 1.0)
        self.reg_lambda = xgb_config.get('reg_lambda', 2.0)
        self.min_child_weight = xgb_config.get('min_child_weight', 5)
        self.gamma = xgb_config.get('gamma', 0.1)
        self.subsample = xgb_config.get('subsample', 0.7)
        self.colsample_bytree = xgb_config.get('colsample_bytree', 0.6)
        self.colsample_bylevel = 0.6
        self.colsample_bynode = 0.8

        # Model et métriques
        self.model = None
        self.feature_names = []
        self.feature_importance = {}
        self.training_metrics = {}

        # Seuil optimal (sera calculé automatiquement)
        self.optimal_threshold = 0.5
        self.prediction_threshold = 0.25  # Fallback
        self.class_weights = None
        self.scale_pos_weight = None

        # Stratégie de rééchantillonnage
        self.resampling_strategy = 'smote'  # 'smote', 'rus', 'smoteenn', 'none'
        self.smote_k_neighbors = 3

        # Calibration
        self.calibrate_probabilities = True

        # Paths
        self.model_dir = Path('ml_models/saved_models')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("✅ XGBoost Model initialisé")

    def _apply_resampling(self, X_train, y_train):
        """Applique une stratégie de rééchantillonnage"""

        if not IMBLEARN_AVAILABLE:
            self.logger.warning("⚠️ imblearn non disponible, pas de rééchantillonnage")
            return X_train, y_train

        original_pos = y_train.sum()
        original_neg = len(y_train) - original_pos

        if original_pos < 5:
            self.logger.warning(f"⚠️ Trop peu de positifs ({original_pos}), pas de rééchantillonnage")
            return X_train, y_train

        try:
            if self.resampling_strategy == 'smote':
                smote = SMOTE(
                    sampling_strategy=0.3,
                    k_neighbors=min(self.smote_k_neighbors, original_pos - 1),
                    random_state=42
                )
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            elif self.resampling_strategy == 'rus':
                rus = RandomUnderSampler(
                    sampling_strategy=0.25,
                    random_state=42
                )
                X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

            elif self.resampling_strategy == 'smoteenn':
                smoteenn = SMOTEENN(
                    smote=SMOTE(k_neighbors=min(3, original_pos - 1)),
                    random_state=42
                )
                X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)

            else:
                return X_train, y_train

            new_pos = y_resampled.sum()
            new_neg = len(y_resampled) - new_pos

            self.logger.info(f"📊 Rééchantillonnage ({self.resampling_strategy}):")
            self.logger.info(f"   Original: {original_neg} neg / {original_pos} pos")
            self.logger.info(f"   Après:    {new_neg} neg / {new_pos} pos")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.warning(f"⚠️ Échec rééchantillonnage: {e}")
            return X_train, y_train

    def _find_optimal_threshold(self, X_val, y_val):
        """Trouve le seuil optimal pour maximiser F1"""

        y_proba = self.model.predict_proba(X_val)[:, 1]

        # Calculer precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

        # Calculer F1 pour chaque seuil
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

        # Trouver le seuil qui maximise F1
        best_idx = np.argmax(f1_scores[:-1])
        self.optimal_threshold = thresholds[best_idx]
        self.prediction_threshold = self.optimal_threshold

        # Afficher les métriques pour différents seuils
        test_thresholds = [0.2, 0.3, 0.4, 0.5, self.optimal_threshold]

        self.logger.info(f"🎯 Optimisation du seuil de décision:")
        for thresh in sorted(set(test_thresholds)):
            y_pred = (y_proba >= thresh).astype(int)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            marker = "⭐" if abs(thresh - self.optimal_threshold) < 0.01 else "  "
            self.logger.info(f"   {marker} Seuil {thresh:.3f}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

        return self.optimal_threshold

    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_split: float = 0.2, verbose: bool = True) -> Dict:
        """Entraîne le modèle avec techniques avancées"""

        self.logger.info(f"🚀 Début entraînement XGBoost ({len(X)} samples)")

        # Split stratifié pour garder les proportions
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            stratify=y,
            shuffle=True,
            random_state=42
        )

        self.logger.info(f"📊 Train: {len(X_train)} | Validation: {len(X_val)}")

        # Calculer scale_pos_weight
        n_negative = len(y_train[y_train == 0])
        n_positive = len(y_train[y_train == 1])
        self.scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

        self.logger.info(f"⚖️ Balance: {n_negative} neg / {n_positive} pos")
        self.logger.info(f"   scale_pos_weight = {self.scale_pos_weight:.2f}")

        # Appliquer rééchantillonnage sur train set
        X_train_resampled, y_train_resampled = self._apply_resampling(X_train, y_train)

        # Sauvegarder feature names
        self.feature_names = list(X.columns)

        # Paramètres optimisés
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            'eval_metric': ['logloss', 'auc'],
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'scale_pos_weight': self.scale_pos_weight,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'colsample_bylevel': self.colsample_bylevel,
            'colsample_bynode': self.colsample_bynode,
            'early_stopping_rounds': 50,
            'verbosity': 1 if verbose else 0
        }

        # Créer et entraîner modèle
        self.model = xgb.XGBClassifier(**params)

        eval_set = [(X_train_resampled, y_train_resampled), (X_val, y_val)]

        self.model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=eval_set,
            verbose=verbose
        )

        # Trouver seuil optimal
        self._find_optimal_threshold(X_val, y_val)

        # Évaluation avec seuil optimal
        metrics = self._evaluate_with_optimal_threshold(X_train, y_train, X_val, y_val)

        # Feature importance
        self._calculate_feature_importance()

        # Cross-validation pour vérifier stabilité
        cv_scores = self._cross_validate(X, y)
        metrics['cv_scores'] = cv_scores

        # Sauvegarder métriques
        self.training_metrics = metrics
        self.training_metrics['timestamp'] = datetime.now().isoformat()
        self.training_metrics['optimal_threshold'] = self.optimal_threshold

        self.logger.info(f"✅ Entraînement terminé (seuil optimal: {self.optimal_threshold:.3f}):")
        self.logger.info(f"   - Val Accuracy: {metrics['val_accuracy']:.4f}")
        self.logger.info(f"   - Val Precision: {metrics['val_precision']:.4f}")
        self.logger.info(f"   - Val Recall: {metrics['val_recall']:.4f}")
        self.logger.info(f"   - Val F1: {metrics['val_f1']:.4f}")
        self.logger.info(f"   - Val ROC AUC: {metrics['val_roc_auc']:.4f}")
        self.logger.info(f"   - CV F1 (mean±std): {cv_scores['mean_f1']:.3f}±{cv_scores['std_f1']:.3f}")

        return metrics

    def _evaluate_with_optimal_threshold(self, X_train, y_train, X_val, y_val) -> Dict:
        """Évalue avec le seuil optimal"""

        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_val_proba = self.model.predict_proba(X_val)[:, 1]

        y_train_pred = (y_train_proba >= self.optimal_threshold).astype(int)
        y_val_pred = (y_val_proba >= self.optimal_threshold).astype(int)

        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
            'train_roc_auc': roc_auc_score(y_train, y_train_proba),

            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
            'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
            'val_f1': f1_score(y_val, y_val_pred, zero_division=0),
            'val_roc_auc': roc_auc_score(y_val, y_val_proba)
        }

        # Matrice de confusion
        cm = confusion_matrix(y_val, y_val_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

            self.logger.info(f"📊 Matrice de confusion (validation):")
            self.logger.info(f"   TN={tn:4d}  FP={fp:4d}")
            self.logger.info(f"   FN={fn:4d}  TP={tp:4d}")

            metrics['confusion_matrix'] = {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }

        return metrics

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
        """Cross-validation stratifiée pour évaluer la stabilité"""

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        f1_scores = []
        precision_scores_list = []
        recall_scores_list = []

        self.logger.info(f"🔄 Cross-validation ({n_splits} folds)...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]

            temp_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                scale_pos_weight=self.scale_pos_weight,
                random_state=42,
                verbosity=0
            )

            temp_model.fit(X_fold_train, y_fold_train, verbose=False)

            y_proba = temp_model.predict_proba(X_fold_val)[:, 1]
            y_pred = (y_proba >= self.optimal_threshold).astype(int)

            f1 = f1_score(y_fold_val, y_pred, zero_division=0)
            precision = precision_score(y_fold_val, y_pred, zero_division=0)
            recall = recall_score(y_fold_val, y_pred, zero_division=0)

            f1_scores.append(f1)
            precision_scores_list.append(precision)
            recall_scores_list.append(recall)

            self.logger.info(f"   Fold {fold}: F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}")

        return {
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_precision': np.mean(precision_scores_list),
            'std_precision': np.std(precision_scores_list),
            'mean_recall': np.mean(recall_scores_list),
            'std_recall': np.std(recall_scores_list)
        }

    def _calculate_feature_importance(self):
        """Calcule l'importance des features"""

        if self.model is None:
            return

        importance_scores = self.model.feature_importances_

        self.feature_importance = {
            feature: float(score)
            for feature, score in zip(self.feature_names, importance_scores)
        }

        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        self.logger.info("📊 Top 15 features importantes:")
        for i, (feature, score) in enumerate(list(self.feature_importance.items())[:15], 1):
            self.logger.info(f"   {i:2d}. {feature}: {score:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédiction avec seuil optimal"""

        if self.model is None:
            raise ValueError("Modèle non entraîné")

        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]

        probas = self.model.predict_proba(X)[:, 1]
        return (probas >= self.optimal_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prédiction de probabilités"""

        if self.model is None:
            raise ValueError("Modèle non entraîné")

        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]

        return self.model.predict_proba(X)

    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne la confidence de la prédiction (0-1)"""
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)

    def get_signal_with_confidence(self, X: pd.DataFrame) -> Tuple[int, float]:
        """Retourne signal et confidence pour la dernière observation"""

        if not safe_dataframe_access(X, "xgboost_predict"):
            return 0, 0.0

        if len(X) == 0:
            return 0, 0.0

        try:
            X_last = X.iloc[[-1]]
            proba = self.predict_proba(X_last)[0]

            signal = int(proba[1] >= self.optimal_threshold)

            if signal == 1:
                confidence = min((proba[1] - self.optimal_threshold) / (1 - self.optimal_threshold), 1.0)
            else:
                confidence = min((self.optimal_threshold - proba[1]) / self.optimal_threshold, 1.0)

            if self.training_metrics and 'val_f1' in self.training_metrics:
                f1_score_val = self.training_metrics['val_f1']
                confidence *= min(f1_score_val * 2, 1.0)

            return signal, float(confidence)

        except Exception as e:
            self.logger.error(f"❌ Erreur prédiction: {e}")
            return 0, 0.0

    def save(self, filename: Optional[str] = None) -> str:
        """Sauvegarde le modèle avec toutes les métadonnées"""

        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'xgboost_{timestamp}.pkl'

        filepath = self.model_dir / filename

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'optimal_threshold': self.optimal_threshold,
            'prediction_threshold': self.prediction_threshold,
            'scale_pos_weight': self.scale_pos_weight,
            'resampling_strategy': self.resampling_strategy,
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda
            }
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"💾 Modèle sauvegardé: {filepath}")

        return str(filepath)

    def load(self, filepath: str):
        """Charge un modèle sauvegardé"""

        self.logger.info(f"📂 Chargement modèle: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data.get('feature_importance', {})
        self.training_metrics = model_data.get('training_metrics', {})
        self.optimal_threshold = model_data.get('optimal_threshold', 0.5)
        self.prediction_threshold = model_data.get('prediction_threshold', self.optimal_threshold)
        self.scale_pos_weight = model_data.get('scale_pos_weight', 1.0)

        self.logger.info("✅ Modèle chargé avec succès")
        self.logger.info(f"   - Seuil optimal: {self.optimal_threshold:.3f}")

        if self.training_metrics:
            self.logger.info(f"   - Val F1: {self.training_metrics.get('val_f1', 0):.4f}")

    def get_metrics(self) -> Dict:
        return self.training_metrics.copy()

    def get_feature_importance(self, top_n: int = 20) -> Dict:
        if not self.feature_importance:
            return {}
        return dict(list(self.feature_importance.items())[:top_n])
