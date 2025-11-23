"""
LightGBM Model AMÉLIORÉ - Quantum Trader Pro
Version optimisée pour classes déséquilibrées avec techniques avancées
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from utils.logger import setup_logger
from utils.safety import safe_dataframe_access

# Import conditionnel de imblearn
try:
    from imblearn.over_sampling import ADASYN, BorderlineSMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


class LightGBMModel:
    """
    LightGBM optimisé pour déséquilibre de classes extrême

    AMÉLIORATIONS MAJEURES:
    - ADASYN pour génération synthétique adaptative
    - Optimisation du seuil via F2-score (favorise recall)
    - Ensemble voting avec plusieurs modèles
    - Normalisation des features
    - Cross-validation stratifiée
    """

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('LightGBMModel')

        # Configuration LightGBM améliorée
        lgb_config = config.get('ml', {}).get('models', {}).get('lightgbm', {})

        # Hyperparamètres équilibrés (pas trop restrictifs)
        self.n_estimators = lgb_config.get('n_estimators', 500)
        self.max_depth = lgb_config.get('max_depth', 6)
        self.learning_rate = lgb_config.get('learning_rate', 0.05)
        self.num_leaves = lgb_config.get('num_leaves', 31)
        self.min_child_samples = lgb_config.get('min_child_samples', 20)
        self.min_split_gain = 0.001

        # Régularisation modérée (pas trop forte)
        self.reg_alpha = lgb_config.get('reg_alpha', 0.5)
        self.reg_lambda = lgb_config.get('reg_lambda', 0.5)
        self.max_bin = 255

        # Sous-échantillonnage
        self.subsample = lgb_config.get('subsample', 0.8)
        self.subsample_freq = 5
        self.colsample_bytree = lgb_config.get('colsample_bytree', 0.8)
        self.colsample_bynode = 0.8

        # Paramètres pour déséquilibre
        self.path_smooth = 1

        # Models et métriques (ensemble)
        self.models = []
        self.n_models = 3
        self.model = None  # Pour compatibilité
        self.feature_names = []
        self.feature_importance = {}
        self.training_metrics = {}

        # Seuils optimaux
        self.optimal_thresholds = []
        self.ensemble_threshold = 0.5
        self.prediction_threshold = 0.35  # Fallback

        # Stratégie de rééchantillonnage
        # 'none' = utilise is_unbalance (plus stable que SMOTE/ADASYN)
        self.resampling_strategy = 'none'  # 'adasyn', 'borderline', 'none'

        # Normalisation des features
        self.scaler = StandardScaler()
        self.normalize_features = True

        # Focal loss (désactivé par défaut car complexe)
        self.use_focal_loss = False
        self.focal_gamma = 2.0
        self.focal_alpha = 0.25

        # Paths
        self.model_dir = Path('ml_models/saved_models')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("✅ LightGBM Model initialisé")
        self.logger.info(f"   - Ensemble de {self.n_models} modèles")

    def _apply_advanced_resampling(self, X_train, y_train):
        """Rééchantillonnage avancé avec ADASYN ou BorderlineSMOTE"""

        if not IMBLEARN_AVAILABLE:
            self.logger.warning("⚠️ imblearn non disponible, pas de rééchantillonnage")
            return X_train, y_train

        original_pos = y_train.sum()
        original_neg = len(y_train) - original_pos

        if original_pos < 5:
            self.logger.warning(f"⚠️ Trop peu de positifs ({original_pos}), pas de rééchantillonnage")
            return X_train, y_train

        try:
            if self.resampling_strategy == 'adasyn':
                resampler = ADASYN(
                    sampling_strategy=0.25,
                    n_neighbors=min(5, original_pos - 1),
                    random_state=42
                )
            elif self.resampling_strategy == 'borderline':
                resampler = BorderlineSMOTE(
                    sampling_strategy=0.3,
                    k_neighbors=min(5, original_pos - 1),
                    kind='borderline-1',
                    random_state=42
                )
            else:
                return X_train, y_train

            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

            new_pos = y_resampled.sum()
            new_neg = len(y_resampled) - new_pos

            self.logger.info(f"📊 Rééchantillonnage ({self.resampling_strategy}):")
            self.logger.info(f"   Original: {original_neg} neg / {original_pos} pos")
            self.logger.info(f"   Après:    {new_neg} neg / {new_pos} pos")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.warning(f"⚠️ Échec rééchantillonnage: {e}")
            return X_train, y_train

    def _find_optimal_threshold_f2(self, X_val, y_val, model):
        """Optimisation du seuil pour maximiser F2-score (favorise recall)"""

        y_proba = model.predict_proba(X_val)[:, 1]

        def f2_score(threshold):
            y_pred = (y_proba >= threshold).astype(int)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)

            if precision + recall == 0:
                return 0

            beta = 2.0
            f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

            # Pénalité si trop peu de positifs prédits
            n_positive_pred = y_pred.sum()
            if n_positive_pred < len(y_val) * 0.02:
                f_beta *= 0.5

            return f_beta

        thresholds = np.linspace(0.1, 0.9, 50)
        scores = [f2_score(t) for t in thresholds]

        best_idx = np.argmax(scores)
        optimal_threshold = thresholds[best_idx]

        self.logger.info(f"🎯 Optimisation du seuil (F2-score):")
        for thresh in sorted(set([0.2, 0.3, 0.4, 0.5, optimal_threshold])):
            y_pred = (y_proba >= thresh).astype(int)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            marker = "⭐" if abs(thresh - optimal_threshold) < 0.02 else "  "
            self.logger.info(f"   {marker} Seuil {thresh:.3f}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

        return optimal_threshold

    def _train_single_model(self, X_train, y_train, X_val, y_val, model_idx: int):
        """Entraîne un seul modèle LightGBM"""

        np.random.seed(42 + model_idx)

        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves + np.random.randint(-2, 3),
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate * np.random.uniform(0.8, 1.2),
            'n_estimators': self.n_estimators,
            'min_child_samples': self.min_child_samples,
            'min_split_gain': self.min_split_gain,
            'is_unbalance': True,
            'reg_alpha': self.reg_alpha * np.random.uniform(0.8, 1.2),
            'reg_lambda': self.reg_lambda * np.random.uniform(0.8, 1.2),
            'max_bin': self.max_bin,
            'path_smooth': self.path_smooth,
            'subsample': self.subsample,
            'subsample_freq': self.subsample_freq,
            'colsample_bytree': self.colsample_bytree,
            'colsample_bynode': self.colsample_bynode,
            'random_state': 42 + model_idx,
            'n_jobs': -1,
            'verbose': -1,
            'force_col_wise': True
        }

        # Utiliser moins d'estimateurs sans early stopping
        # (early stopping ne fonctionne pas bien avec données rééchantillonnées)
        params['n_estimators'] = 200

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(period=50)]  # Log toutes les 50 itérations
        )

        optimal_threshold = self._find_optimal_threshold_f2(X_val, y_val, model)

        self.logger.info(f"   Modèle {model_idx + 1}: Seuil optimal = {optimal_threshold:.3f}")

        return model, optimal_threshold

    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_split: float = 0.2, verbose: bool = True) -> Dict:
        """Entraîne un ensemble de modèles LightGBM"""

        self.logger.info(f"🚀 Début entraînement LightGBM Ensemble ({len(X)} samples)")

        # Split stratifié
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            stratify=y,
            shuffle=True,
            random_state=42
        )

        self.logger.info(f"📊 Train: {len(X_train)} | Validation: {len(X_val)}")

        # Analyser déséquilibre
        n_negative = len(y_train[y_train == 0])
        n_positive = len(y_train[y_train == 1])
        imbalance_ratio = n_negative / n_positive if n_positive > 0 else 1.0

        self.logger.info(f"⚖️ Balance: {n_negative} neg / {n_positive} pos (ratio: {imbalance_ratio:.2f})")

        # Normalisation des features
        if self.normalize_features:
            X_train_norm = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X.columns,
                index=X_train.index
            )
            X_val_norm = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X.columns,
                index=X_val.index
            )
        else:
            X_train_norm = X_train
            X_val_norm = X_val

        # Rééchantillonnage avancé
        X_train_resampled, y_train_resampled = self._apply_advanced_resampling(X_train_norm, y_train)

        # Sauvegarder feature names
        self.feature_names = list(X.columns)

        # Entraîner plusieurs modèles pour ensemble
        self.models = []
        self.optimal_thresholds = []

        self.logger.info(f"🎯 Entraînement de {self.n_models} modèles...")

        for i in range(self.n_models):
            self.logger.info(f"\n📊 Modèle {i + 1}/{self.n_models}:")

            # Bootstrap sampling pour diversité
            if i > 0:
                sample_idx = np.random.choice(
                    len(X_train_resampled),
                    size=int(len(X_train_resampled) * 0.8),
                    replace=False
                )
                if isinstance(X_train_resampled, pd.DataFrame):
                    X_train_i = X_train_resampled.iloc[sample_idx]
                    y_train_i = y_train_resampled.iloc[sample_idx] if hasattr(y_train_resampled, 'iloc') else y_train_resampled[sample_idx]
                else:
                    X_train_i = X_train_resampled[sample_idx]
                    y_train_i = y_train_resampled[sample_idx]
            else:
                X_train_i = X_train_resampled
                y_train_i = y_train_resampled

            model, threshold = self._train_single_model(
                X_train_i, y_train_i, X_val_norm, y_val, i
            )

            self.models.append(model)
            self.optimal_thresholds.append(threshold)

        # Garder le premier modèle pour compatibilité
        self.model = self.models[0] if self.models else None

        # Calculer seuil optimal pour ensemble
        self._optimize_ensemble_threshold(X_val_norm, y_val)

        # Évaluation complète
        metrics = self._evaluate_ensemble(X_train_norm, y_train, X_val_norm, y_val)

        # Feature importance
        self._calculate_ensemble_feature_importance()

        # Cross-validation
        cv_scores = self._cross_validate_ensemble(X, y)
        metrics['cv_scores'] = cv_scores

        # Sauvegarder métriques
        self.training_metrics = metrics
        self.training_metrics['timestamp'] = datetime.now().isoformat()
        self.training_metrics['ensemble_threshold'] = self.ensemble_threshold

        self.logger.info(f"\n✅ Entraînement terminé (Ensemble de {self.n_models} modèles):")
        self.logger.info(f"   - Val Accuracy: {metrics['val_accuracy']:.4f}")
        self.logger.info(f"   - Val Precision: {metrics['val_precision']:.4f}")
        self.logger.info(f"   - Val Recall: {metrics['val_recall']:.4f}")
        self.logger.info(f"   - Val F1: {metrics['val_f1']:.4f}")
        self.logger.info(f"   - Val ROC AUC: {metrics['val_roc_auc']:.4f}")
        self.logger.info(f"   - CV F1 (mean±std): {cv_scores['mean_f1']:.3f}±{cv_scores['std_f1']:.3f}")

        return metrics

    def _optimize_ensemble_threshold(self, X_val, y_val):
        """Optimise le seuil pour l'ensemble"""

        ensemble_proba = self._get_ensemble_probabilities(X_val)

        def ensemble_metric(threshold):
            y_pred = (ensemble_proba >= threshold).astype(int)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)

            if precision + recall == 0:
                return 0

            beta = 2.0
            f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
            return f_beta

        thresholds = np.linspace(0.15, 0.85, 70)
        scores = [ensemble_metric(t) for t in thresholds]

        best_idx = np.argmax(scores)
        self.ensemble_threshold = thresholds[best_idx]
        self.prediction_threshold = self.ensemble_threshold

        self.logger.info(f"🎯 Seuil optimal pour ensemble: {self.ensemble_threshold:.3f}")

    def _get_ensemble_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Obtient les probabilités moyennes de l'ensemble"""

        if self.normalize_features and hasattr(self.scaler, 'mean_'):
            X_norm = pd.DataFrame(
                self.scaler.transform(X),
                columns=self.feature_names if self.feature_names else X.columns,
                index=X.index
            )
        else:
            X_norm = X

        probas = []
        for model in self.models:
            proba = model.predict_proba(X_norm)[:, 1]
            probas.append(proba)

        return np.mean(probas, axis=0)

    def _evaluate_ensemble(self, X_train, y_train, X_val, y_val) -> Dict:
        """Évalue l'ensemble de modèles"""

        train_proba = self._get_ensemble_probabilities(X_train)
        val_proba = self._get_ensemble_probabilities(X_val)

        y_train_pred = (train_proba >= self.ensemble_threshold).astype(int)
        y_val_pred = (val_proba >= self.ensemble_threshold).astype(int)

        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
            'train_roc_auc': roc_auc_score(y_train, train_proba),

            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
            'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
            'val_f1': f1_score(y_val, y_val_pred, zero_division=0),
            'val_roc_auc': roc_auc_score(y_val, val_proba),
            'val_avg_precision': average_precision_score(y_val, val_proba)
        }

        # Matrice de confusion
        cm = confusion_matrix(y_val, y_val_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

            self.logger.info(f"\n📊 Matrice de confusion (validation):")
            self.logger.info(f"   TN={tn:4d}  FP={fp:4d}")
            self.logger.info(f"   FN={fn:4d}  TP={tp:4d}")

            metrics['confusion_matrix'] = {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }

        return metrics

    def _calculate_ensemble_feature_importance(self):
        """Calcule l'importance moyenne des features sur l'ensemble"""

        all_importances = []

        for model in self.models:
            importance = model.feature_importances_
            all_importances.append(importance)

        mean_importance = np.mean(all_importances, axis=0)

        self.feature_importance = {
            feature: float(score)
            for feature, score in zip(self.feature_names, mean_importance)
        }

        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        self.logger.info("\n📊 Top 15 features (ensemble):")
        for i, (feature, score) in enumerate(list(self.feature_importance.items())[:15], 1):
            self.logger.info(f"   {i:2d}. {feature}: {score:.1f}")

    def _cross_validate_ensemble(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> Dict:
        """Cross-validation de l'ensemble (rapide)"""

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        f1_scores = []
        precision_scores_list = []
        recall_scores_list = []

        self.logger.info(f"\n🔄 Cross-validation ({n_splits} folds)...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]

            simple_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=self.max_depth,
                learning_rate=0.1,
                is_unbalance=True,
                random_state=42,
                verbose=-1
            )

            simple_model.fit(X_fold_train, y_fold_train, callbacks=[lgb.log_evaluation(0)])

            y_proba = simple_model.predict_proba(X_fold_val)[:, 1]
            y_pred = (y_proba >= 0.3).astype(int)

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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédiction binaire avec ensemble et seuil optimal"""

        if not self.models:
            raise ValueError("Modèle non entraîné")

        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]

        ensemble_proba = self._get_ensemble_probabilities(X)
        return (ensemble_proba >= self.ensemble_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilités de l'ensemble"""

        if not self.models:
            raise ValueError("Modèle non entraîné")

        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]

        ensemble_proba = self._get_ensemble_probabilities(X)
        return np.column_stack([1 - ensemble_proba, ensemble_proba])

    def get_signal_with_confidence(self, X: pd.DataFrame) -> Tuple[int, float]:
        """Signal avec confidence ajustée"""

        if not safe_dataframe_access(X, "lightgbm_predict"):
            return 0, 0.0

        if len(X) == 0:
            return 0, 0.0

        try:
            X_last = X.iloc[[-1]]

            if list(X_last.columns) != self.feature_names:
                X_last = X_last[self.feature_names]

            # Obtenir probabilités de chaque modèle
            if self.normalize_features and hasattr(self.scaler, 'mean_'):
                X_norm = pd.DataFrame(
                    self.scaler.transform(X_last),
                    columns=self.feature_names,
                    index=X_last.index
                )
            else:
                X_norm = X_last

            probas = []
            for model in self.models:
                proba = model.predict_proba(X_norm)[0, 1]
                probas.append(proba)

            mean_proba = np.mean(probas)
            std_proba = np.std(probas)

            signal = int(mean_proba >= self.ensemble_threshold)

            if signal == 1:
                base_confidence = (mean_proba - self.ensemble_threshold) / (1 - self.ensemble_threshold)
            else:
                base_confidence = (self.ensemble_threshold - mean_proba) / self.ensemble_threshold

            # Pénaliser si modèles en désaccord
            uncertainty_penalty = 1.0 - min(std_proba * 2, 0.5)

            if self.training_metrics and 'val_f1' in self.training_metrics:
                performance_factor = min(self.training_metrics['val_f1'] * 1.5, 1.0)
            else:
                performance_factor = 0.5

            confidence = base_confidence * uncertainty_penalty * performance_factor
            confidence = max(0.0, min(1.0, confidence))

            return signal, float(confidence)

        except Exception as e:
            self.logger.error(f"❌ Erreur prédiction: {e}")
            return 0, 0.0

    def save(self, filename: Optional[str] = None) -> str:
        """Sauvegarde l'ensemble de modèles"""

        if not self.models:
            raise ValueError("Aucun modèle à sauvegarder")

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'lightgbm_{timestamp}.pkl'

        filepath = self.model_dir / filename

        model_data = {
            'models': self.models,
            'n_models': self.n_models,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'optimal_thresholds': self.optimal_thresholds,
            'ensemble_threshold': self.ensemble_threshold,
            'prediction_threshold': self.prediction_threshold,
            'scaler': self.scaler if self.normalize_features else None,
            'normalize_features': self.normalize_features,
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'resampling_strategy': self.resampling_strategy
            }
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"💾 Ensemble sauvegardé: {filepath}")

        return str(filepath)

    def load(self, filepath: str):
        """Charge un ensemble sauvegardé"""

        self.logger.info(f"📂 Chargement ensemble: {filepath}")

        model_data = joblib.load(filepath)

        self.models = model_data['models']
        self.n_models = model_data.get('n_models', len(self.models))
        self.model = self.models[0] if self.models else None
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data.get('feature_importance', {})
        self.training_metrics = model_data.get('training_metrics', {})
        self.optimal_thresholds = model_data.get('optimal_thresholds', [])
        self.ensemble_threshold = model_data.get('ensemble_threshold', 0.5)
        self.prediction_threshold = model_data.get('prediction_threshold', self.ensemble_threshold)
        self.scaler = model_data.get('scaler')
        self.normalize_features = model_data.get('normalize_features', False)

        self.logger.info(f"✅ Ensemble chargé ({self.n_models} modèles)")
        self.logger.info(f"   - Seuil ensemble: {self.ensemble_threshold:.3f}")

        if self.training_metrics:
            self.logger.info(f"   - Val F1: {self.training_metrics.get('val_f1', 0):.4f}")

    def get_metrics(self) -> Dict:
        return self.training_metrics.copy()

    def get_feature_importance(self, top_n: int = 20) -> Dict:
        if not self.feature_importance:
            return {}
        return dict(list(self.feature_importance.items())[:top_n])
