"""
XGBoost Model - Quantum Trader Pro
Modèle de gradient boosting pour prédiction directionnelle
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils.logger import setup_logger
from utils.safety import safe_dataframe_access, safe_iloc

class XGBoostModel:
    """
    Modèle XGBoost pour prédiction de direction du marché:
    - Classification binaire (UP/DOWN)
    - Feature importance
    - Hyperparameter tuning
    - Model persistence
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le modèle XGBoost
        
        Args:
            config: Configuration complète du bot
        """
        self.config = config
        self.logger = setup_logger('XGBoostModel')
        
        # Configuration XGBoost
        xgb_config = config.get('ml', {}).get('models', {}).get('xgboost', {})
        
        self.n_estimators = xgb_config.get('n_estimators', 200)
        self.max_depth = xgb_config.get('max_depth', 6)
        self.learning_rate = xgb_config.get('learning_rate', 0.1)
        self.objective = xgb_config.get('objective', 'binary:logistic')
        
        # Model
        self.model = None
        self.feature_names = []
        self.feature_importance = {}
        self.training_metrics = {}
        
        # Paths
        self.model_dir = Path('ml_models/saved_models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ XGBoost Model initialisé")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict:
        """
        Entraîne le modèle XGBoost
        
        Args:
            X: Features DataFrame
            y: Target Series
            validation_split: Proportion pour validation
            verbose: Afficher progression
        
        Returns:
            Dict avec métriques d'entraînement
        """
        
        self.logger.info(f"🚀 Début entraînement XGBoost ({len(X)} samples)")
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            shuffle=False  # Garder ordre temporel
        )
        
        self.logger.info(f"📊 Train: {len(X_train)} | Validation: {len(X_val)}")
        
        # Sauvegarder feature names
        self.feature_names = list(X.columns)
        
        # Paramètres du modèle
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        # Créer et entraîner modèle
        self.model = xgb.XGBClassifier(**params)
        
        # Entraînement avec early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
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
    
    def _evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """Évalue le modèle sur train et validation"""
        
        # Prédictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Métriques
        metrics = {
            # Train
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'train_roc_auc': roc_auc_score(y_train, y_train_proba),
            
            # Validation
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred),
            'val_recall': recall_score(y_val, y_val_pred),
            'val_f1': f1_score(y_val, y_val_pred),
            'val_roc_auc': roc_auc_score(y_val, y_val_proba)
        }
        
        # Check overfitting
        accuracy_diff = metrics['train_accuracy'] - metrics['val_accuracy']
        if accuracy_diff > 0.1:
            self.logger.warning(
                f"⚠️ Possible overfitting: Train acc - Val acc = {accuracy_diff:.4f}"
            )
        
        return metrics
    
    def _calculate_feature_importance(self):
        """Calcule l'importance des features"""
        
        if self.model is None:
            return
        
        # Get importance scores
        importance_scores = self.model.feature_importances_
        
        # Créer dict
        self.feature_importance = {
            feature: float(score)
            for feature, score in zip(self.feature_names, importance_scores)
        }
        
        # Trier par importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Log top 10
        self.logger.info("📊 Top 10 features importantes:")
        for i, (feature, score) in enumerate(list(self.feature_importance.items())[:10], 1):
            self.logger.info(f"   {i}. {feature}: {score:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédiction binaire (0 ou 1)
        
        Args:
            X: Features DataFrame
        
        Returns:
            Array de prédictions (0=DOWN, 1=UP)
        """
        
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        
        # Vérifier features
        if list(X.columns) != self.feature_names:
            self.logger.warning("⚠️ Features différentes, réordonnancement")
            X = X[self.feature_names]
        
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédiction de probabilités
        
        Args:
            X: Features DataFrame
        
        Returns:
            Array de probabilités [P(DOWN), P(UP)]
        """
        
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        
        # Vérifier features
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]
        
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retourne la confidence de la prédiction (0-1)
        
        Args:
            X: Features DataFrame
        
        Returns:
            Array de confidence scores
        """
        
        proba = self.predict_proba(X)
        
        # Confidence = max probability
        confidence = np.max(proba, axis=1)
        
        return confidence
    
    def get_signal_with_confidence(self, X: pd.DataFrame) -> Tuple[int, float]:
        """
        Retourne signal et confidence pour la dernière observation

        Args:
            X: Features DataFrame (dernière ligne sera utilisée)

        Returns:
            (signal, confidence) où signal = 1 (UP) ou 0 (DOWN)
        """

        # Validation sécurité
        if not safe_dataframe_access(X, "xgboost_predict"):
            self.logger.warning("⚠️ DataFrame invalide pour prédiction")
            return 0, 0.0

        if len(X) == 0:
            return 0, 0.0

        try:
            # Prendre dernière ligne (safe_iloc retourne une Series, on veut un DataFrame)
            X_last = X.iloc[[-1]]

            # Prédiction
            signal = self.predict(X_last)[0]
            proba = self.predict_proba(X_last)[0]

            # Confidence = probabilité de la classe prédite
            confidence = proba[signal]

            return int(signal), float(confidence)
        except Exception as e:
            self.logger.error(f"❌ Erreur prédiction XGBoost: {e}")
            return 0, 0.0
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Sauvegarde le modèle
        
        Args:
            filename: Nom du fichier (optionnel)
        
        Returns:
            Chemin du fichier sauvegardé
        """
        
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'xgboost_{timestamp}.pkl'
        
        filepath = self.model_dir / filename
        
        # Sauvegarder tout
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
        """
        Charge un modèle sauvegardé
        
        Args:
            filepath: Chemin vers le fichier
        """
        
        self.logger.info(f"📂 Chargement modèle: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data.get('feature_importance', {})
        self.training_metrics = model_data.get('training_metrics', {})
        
        # Restore config
        config = model_data.get('config', {})
        self.n_estimators = config.get('n_estimators', self.n_estimators)
        self.max_depth = config.get('max_depth', self.max_depth)
        self.learning_rate = config.get('learning_rate', self.learning_rate)
        
        self.logger.info("✅ Modèle chargé avec succès")
        
        if self.training_metrics:
            self.logger.info(
                f"   - Accuracy (val): {self.training_metrics.get('val_accuracy', 0):.4f}"
            )
    
    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50
    ) -> Dict:
        """
        Optimise les hyperparamètres avec Optuna
        
        Args:
            X: Features DataFrame
            y: Target Series
            n_trials: Nombre d'essais
        
        Returns:
            Meilleurs paramètres trouvés
        """
        
        try:
            import optuna
        except ImportError:
            self.logger.error("❌ Optuna non installé: pip install optuna")
            return {}
        
        self.logger.info(f"🔍 Optimisation hyperparamètres ({n_trials} trials)")
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        def objective(trial):
            """Fonction objectif pour Optuna"""
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            return accuracy
        
        # Optimisation
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_score = study.best_value
        
        self.logger.info(f"✅ Meilleur score: {best_score:.4f}")
        self.logger.info(f"   Paramètres: {best_params}")
        
        # Update config
        self.n_estimators = best_params['n_estimators']
        self.max_depth = best_params['max_depth']
        self.learning_rate = best_params['learning_rate']
        
        return best_params
    
    def get_metrics(self) -> Dict:
        """Retourne les métriques d'entraînement"""
        return self.training_metrics.copy()
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """
        Retourne les N features les plus importantes
        
        Args:
            top_n: Nombre de features à retourner
        
        Returns:
            Dict {feature: importance}
        """
        
        if not self.feature_importance:
            return {}
        
        return dict(list(self.feature_importance.items())[:top_n])
