"""
LightGBM Model - Quantum Trader Pro
Modèle de gradient boosting rapide et efficace (remplace LSTM)
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils.logger import setup_logger
from utils.safety import safe_dataframe_access


class LightGBMModel:
    """
    Modèle LightGBM pour prédiction de direction du marché

    AVANTAGES vs LSTM :
    - Plus rapide à entraîner (10x)
    - Meilleure généralisation sur petits datasets
    - Pas besoin de GPU
    - Interprétable (feature importance)
    - Moins d'overfitting

    UTILISATION :
    - Comme filtre de confirmation pour les signaux
    - Pas comme générateur de signaux
    """

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('LightGBMModel')

        # Configuration LightGBM
        lgb_config = config.get('ml', {}).get('models', {}).get('lightgbm', {})

        self.n_estimators = lgb_config.get('n_estimators', 200)
        self.max_depth = lgb_config.get('max_depth', 6)
        self.learning_rate = lgb_config.get('learning_rate', 0.05)
        self.num_leaves = lgb_config.get('num_leaves', 31)
        self.min_child_samples = lgb_config.get('min_child_samples', 20)
        self.reg_alpha = lgb_config.get('reg_alpha', 0.1)
        self.reg_lambda = lgb_config.get('reg_lambda', 0.1)
        self.subsample = lgb_config.get('subsample', 0.8)
        self.colsample_bytree = lgb_config.get('colsample_bytree', 0.8)

        # Model
        self.model = None
        self.feature_names = []
        self.feature_importance = {}
        self.training_metrics = {}

        # Seuil de classification ajustable (0.3 au lieu de 0.5 pour classe déséquilibrée)
        self.prediction_threshold = 0.35

        # Paths
        self.model_dir = Path('ml_models/saved_models')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("✅ LightGBM Model initialisé")

    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_split: float = 0.2, verbose: bool = True) -> Dict:
        """Entraîne le modèle LightGBM"""

        self.logger.info(f"🚀 Début entraînement LightGBM ({len(X)} samples)")

        # Split train/validation (garder ordre temporel)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )

        self.logger.info(f"📊 Train: {len(X_train)} | Validation: {len(X_val)}")

        # Sauvegarder feature names
        self.feature_names = list(X.columns)

        # Calculer le déséquilibre de classes
        n_negative = len(y_train[y_train == 0])
        n_positive = len(y_train[y_train == 1])
        imbalance_ratio = n_negative / n_positive if n_positive > 0 else 1.0

        self.logger.info(f"⚖️ Balance: {n_negative} neg / {n_positive} pos (ratio: {imbalance_ratio:.2f})")

        # Paramètres du modèle - optimisés pour déséquilibre
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            # Class imbalance - utiliser SEULEMENT is_unbalance (pas les deux)
            'is_unbalance': True,
            # Regularization
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree
        }

        # Créer et entraîner
        self.model = lgb.LGBMClassifier(**params)

        # Early stopping callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=verbose),
            lgb.log_evaluation(period=50 if verbose else 0)
        ]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )

        # Évaluation
        metrics = self._evaluate(X_train, y_train, X_val, y_val)

        # Feature importance
        self._calculate_feature_importance()

        # Sauvegarder métriques
        self.training_metrics = metrics
        self.training_metrics['timestamp'] = datetime.now().isoformat()

        self.logger.info(f"✅ Entraînement terminé:")
        self.logger.info(f"   - Accuracy (val): {metrics['val_accuracy']:.4f}")
        self.logger.info(f"   - F1 Score (val): {metrics['val_f1']:.4f}")
        self.logger.info(f"   - ROC AUC (val): {metrics['val_roc_auc']:.4f}")

        return metrics

    def _evaluate(self, X_train, y_train, X_val, y_val) -> Dict:
        """Évalue le modèle"""

        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_val_proba = self.model.predict_proba(X_val)[:, 1]

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

        # Check overfitting
        accuracy_diff = metrics['train_accuracy'] - metrics['val_accuracy']
        if accuracy_diff > 0.1:
            self.logger.warning(f"⚠️ Possible overfitting: diff = {accuracy_diff:.4f}")

        return metrics

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

        self.logger.info("📊 Top 10 features:")
        for i, (feature, score) in enumerate(list(self.feature_importance.items())[:10], 1):
            self.logger.info(f"   {i}. {feature}: {score:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédiction binaire avec seuil ajusté"""

        if self.model is None:
            raise ValueError("Modèle non entraîné")

        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]

        # Utiliser seuil personnalisé au lieu de 0.5
        probas = self.model.predict_proba(X)[:, 1]
        return (probas >= self.prediction_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prédiction de probabilités"""

        if self.model is None:
            raise ValueError("Modèle non entraîné")

        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]

        return self.model.predict_proba(X)

    def get_signal_with_confidence(self, X: pd.DataFrame) -> Tuple[int, float]:
        """Retourne signal et confidence pour la dernière observation"""

        if not safe_dataframe_access(X, "lightgbm_predict"):
            return 0, 0.0

        if len(X) == 0:
            return 0, 0.0

        try:
            X_last = X.iloc[[-1]]
            signal = self.predict(X_last)[0]
            proba = self.predict_proba(X_last)[0]
            confidence = proba[signal]

            return int(signal), float(confidence)
        except Exception as e:
            self.logger.error(f"❌ Erreur prédiction: {e}")
            return 0, 0.0

    def save(self, filename: Optional[str] = None) -> str:
        """Sauvegarde le modèle"""

        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'lightgbm_{timestamp}.pkl'

        filepath = self.model_dir / filename

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate
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

        self.logger.info("✅ Modèle chargé avec succès")

        if self.training_metrics:
            self.logger.info(f"   - Accuracy (val): {self.training_metrics.get('val_accuracy', 0):.4f}")

    def get_metrics(self) -> Dict:
        return self.training_metrics.copy()

    def get_feature_importance(self, top_n: int = 20) -> Dict:
        if not self.feature_importance:
            return {}
        return dict(list(self.feature_importance.items())[:top_n])
