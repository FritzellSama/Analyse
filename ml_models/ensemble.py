"""
Ensemble Model - Quantum Trader Pro
Combine les prédictions de plusieurs modèles ML pour améliorer la robustesse
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from utils.logger import setup_logger
from utils.safety import safe_dataframe_access
from utils.safe_math import safe_divide
from utils.config_helpers import get_nested_config

class EnsembleModel:
    """
    Modèle ensemble qui combine:
    - XGBoost (gradient boosting)
    - LSTM (deep learning)
    - Autres modèles custom
    
    Méthodes de combinaison:
    - Voting (majorité)
    - Weighted average (pondération)
    - Stacking (méta-modèle)
    """
    
    def __init__(self, config: Dict):
        """
        Initialise l'ensemble de modèles
        
        Args:
            config: Configuration complète du bot
        """
        self.config = config
        self.logger = setup_logger('EnsembleModel')
        
        # Configuration ensemble
        ensemble_config = config.get('ml', {}).get('models', {}).get('ensemble', {})
        
        self.method = ensemble_config.get('method', 'voting')  # voting, weighted, stacking
        self.weights = ensemble_config.get('weights', [0.4, 0.4, 0.2])  # XGBoost, LSTM, autres
        
        # Modèles
        self.models = {}
        self.model_names = []
        
        self.logger.info(f"✅ Ensemble Model initialisé (méthode: {self.method})")
    
    def add_model(self, name: str, model):
        """
        Ajoute un modèle à l'ensemble
        
        Args:
            name: Nom du modèle (ex: 'xgboost', 'lstm')
            model: Instance du modèle avec méthode predict_proba()
        """
        
        self.models[name] = model
        self.model_names.append(name)
        
        self.logger.info(f"➕ Modèle ajouté: {name}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédiction binaire avec ensemble
        
        Args:
            X: Features DataFrame
        
        Returns:
            Array de prédictions (0=DOWN, 1=UP)
        """
        
        if not self.models:
            raise ValueError("Aucun modèle dans l'ensemble")
        
        # Prédictions de chaque modèle
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur prédiction {name}: {e}")
                continue
        
        if not predictions:
            return np.array([])
        
        # Combiner selon méthode
        if self.method == 'voting':
            ensemble_pred = self._voting(predictions)
        elif self.method == 'weighted':
            probas = self._get_probabilities(X)
            ensemble_pred = self._weighted_average(probas)
        else:
            # Default: voting
            ensemble_pred = self._voting(predictions)
        
        return ensemble_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédiction de probabilités avec ensemble
        
        Args:
            X: Features DataFrame
        
        Returns:
            Array de probabilités [P(DOWN), P(UP)]
        """
        
        if not self.models:
            raise ValueError("Aucun modèle dans l'ensemble")
        
        # Probabilités de chaque modèle
        probas = self._get_probabilities(X)
        
        if not probas:
            return np.array([])
        
        # Combiner selon méthode
        if self.method == 'weighted':
            ensemble_proba = self._weighted_average(probas)
        else:
            # Moyenne simple
            ensemble_proba = np.mean(list(probas.values()), axis=0)
        
        return ensemble_proba
    
    def _get_probabilities(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Récupère les probabilités de chaque modèle"""
        
        probas = {}
        
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X)
                probas[name] = proba
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur proba {name}: {e}")
                continue
        
        return probas
    
    def _voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Vote majoritaire
        
        Args:
            predictions: Dict {model_name: predictions}
        
        Returns:
            Prédictions ensemble
        """
        
        # Stack predictions
        pred_array = np.array(list(predictions.values()))
        
        # Vote majoritaire
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=pred_array
        )
        
        return ensemble_pred
    
    def _weighted_average(self, probas: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Moyenne pondérée des probabilités
        
        Args:
            probas: Dict {model_name: probabilities}
        
        Returns:
            Probabilités ensemble
        """
        
        # Normaliser weights si nécessaire
        total_weight = sum(self.weights[:len(probas)])
        if total_weight == 0:
            total_weight = 1.0  # Avoid division by zero
        normalized_weights = [safe_divide(w, total_weight, default=1.0 / len(probas)) for w in self.weights[:len(probas)]]
        
        # Moyenne pondérée
        ensemble_proba = np.zeros_like(list(probas.values())[0])
        
        for i, (name, proba) in enumerate(probas.items()):
            weight = normalized_weights[i] if i < len(normalized_weights) else 1.0 / len(probas)
            ensemble_proba += weight * proba
        
        return ensemble_proba
    
    def get_signal_with_confidence(self, X: pd.DataFrame) -> Tuple[int, float, Dict]:
        """
        Retourne signal, confidence et détails de chaque modèle
        
        Args:
            X: Features DataFrame
        
        Returns:
            (signal, confidence, model_details)
        """
        
        if not self.models:
            return 0, 0.0, {}
        
        # Prédictions de chaque modèle
        model_details = {}
        
        for name, model in self.models.items():
            try:
                signal, conf = model.get_signal_with_confidence(X)
                model_details[name] = {
                    'signal': int(signal),
                    'confidence': float(conf)
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur {name}: {e}")
                continue
        
        if not model_details:
            return 0, 0.0, {}
        
        # Ensemble prediction
        proba = self.predict_proba(X)
        
        if len(proba) == 0:
            return 0, 0.0, model_details
        
        # Dernière prédiction
        proba_last = proba[-1]
        signal = 1 if proba_last[1] > 0.5 else 0
        confidence = proba_last[signal]
        
        return int(signal), float(confidence), model_details
    
    def get_agreement_score(self, X: pd.DataFrame) -> float:
        """
        Calcule le score d'accord entre les modèles (0-1)
        
        Args:
            X: Features DataFrame
        
        Returns:
            Score d'accord (1 = tous d'accord, 0 = tous en désaccord)
        """
        
        if len(self.models) < 2:
            return 1.0
        
        # Prédictions de chaque modèle
        predictions = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                if len(pred) > 0:
                    predictions.append(pred[-1])  # Dernière prédiction
            except Exception as e:
                self.logger.warning(f"Erreur prédiction modèle {name}: {e}")
                continue
        
        if len(predictions) < 2:
            return 1.0
        
        # Calculer accord
        # Si tous identiques → 1.0
        # Si moitié-moitié → 0.0
        predictions = np.array(predictions)
        agreement = np.mean(predictions == predictions[0])
        
        return float(agreement)
    
    def should_trade(
        self,
        X: pd.DataFrame,
        min_confidence: float = 0.7,
        min_agreement: float = 0.6
    ) -> Tuple[bool, int, float]:
        """
        Décide si on doit trader basé sur ensemble
        
        Args:
            X: Features DataFrame
            min_confidence: Confidence minimum requise
            min_agreement: Accord minimum entre modèles
        
        Returns:
            (should_trade, signal, confidence)
        """
        
        # Prédiction ensemble
        signal, confidence, details = self.get_signal_with_confidence(X)
        
        # Score d'accord
        agreement = self.get_agreement_score(X)
        
        # Décision
        should_trade = (confidence >= min_confidence) and (agreement >= min_agreement)
        
        if not should_trade:
            self.logger.debug(
                f"🚫 Pas de trade: confidence={confidence:.2f} (min={min_confidence}), "
                f"agreement={agreement:.2f} (min={min_agreement})"
            )
        else:
            self.logger.info(
                f"✅ Signal validé: {['DOWN', 'UP'][signal]} | "
                f"Confidence: {confidence:.2%} | Agreement: {agreement:.2%}"
            )
        
        return should_trade, signal, confidence
    
    def get_model_contributions(self, X: pd.DataFrame) -> Dict:
        """
        Analyse la contribution de chaque modèle à la décision
        
        Args:
            X: Features DataFrame
        
        Returns:
            Dict avec contributions
        """
        
        signal, confidence, details = self.get_signal_with_confidence(X)
        
        # Calculer contribution de chaque modèle
        contributions = {}
        
        for name, info in details.items():
            # Contribution = confidence × agreement with ensemble
            model_signal = info['signal']
            model_conf = info['confidence']
            
            agrees_with_ensemble = (model_signal == signal)
            contribution = model_conf if agrees_with_ensemble else (1 - model_conf)
            
            contributions[name] = {
                'signal': info['signal'],
                'confidence': info['confidence'],
                'agrees_with_ensemble': agrees_with_ensemble,
                'contribution': contribution
            }
        
        return contributions
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Évalue l'ensemble sur des données de test
        
        Args:
            X: Features DataFrame
            y: Target Series
        
        Returns:
            Dict avec métriques
        """
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Prédictions
        y_pred = self.predict(X)
        
        # Align lengths (LSTM peut retourner moins de prédictions)
        min_len = min(len(y), len(y_pred))
        y = y.iloc[-min_len:]
        y_pred = y_pred[-min_len:]
        
        # Métriques
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
        
        # Évaluation de chaque modèle
        individual_metrics = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                
                # Align
                min_len_model = min(len(y), len(pred))
                y_model = y.iloc[-min_len_model:]
                pred_model = pred[-min_len_model:]
                
                individual_metrics[name] = {
                    'accuracy': accuracy_score(y_model, pred_model),
                    'f1': f1_score(y_model, pred_model)
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur évaluation {name}: {e}")
        
        metrics['individual'] = individual_metrics
        
        # Log
        self.logger.info("📊 Évaluation Ensemble:")
        self.logger.info(f"   - Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"   - F1 Score: {metrics['f1']:.4f}")
        
        return metrics
    
    def get_summary(self) -> Dict:
        """Retourne un résumé de l'ensemble"""
        
        return {
            'method': self.method,
            'n_models': len(self.models),
            'model_names': self.model_names,
            'weights': self.weights[:len(self.models)]
        }
