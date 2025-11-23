"""
ML Signal Filter - Quantum Trader Pro
Utilise XGBoost + LightGBM pour CONFIRMER les signaux (pas générer)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from utils.logger import setup_logger
from utils.safe_math import safe_divide
from ml_models.xgboost_model import XGBoostModel
from ml_models.lightgbm_model import LightGBMModel
from ml_models.feature_engineering import FeatureEngineer


class MLSignalFilter:
    """
    Filtre ML pour valider les signaux des stratégies

    RÔLE :
    - NE génère PAS de signaux
    - CONFIRME ou REJETTE les signaux des stratégies
    - Combine XGBoost (60%) + LightGBM (40%)

    UTILISATION :
    1. Stratégie génère un signal (BUY/SELL)
    2. MLFilter analyse le contexte de marché
    3. Si ML confidence > seuil → signal validé
    4. Sinon → signal rejeté
    """

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('MLSignalFilter')

        # Configuration
        ml_config = config.get('ml', {})
        filter_config = ml_config.get('signal_filter', {})

        self.min_confidence = filter_config.get('min_confidence', 0.65)
        self.xgboost_weight = filter_config.get('xgboost_weight', 0.6)
        self.lightgbm_weight = filter_config.get('lightgbm_weight', 0.4)

        # Feature engineering
        self.feature_engineer = FeatureEngineer(config)

        # Modèles
        self.xgboost = XGBoostModel(config)
        self.lightgbm = LightGBMModel(config)

        # État
        self.is_ready = False
        self.models_loaded = {'xgboost': False, 'lightgbm': False}

        # Charger les modèles existants
        self._load_models()

        self.logger.info(f"✅ ML Signal Filter initialisé (XGB: {self.xgboost_weight:.0%}, LGB: {self.lightgbm_weight:.0%})")

    def _load_models(self):
        """Charge les modèles pré-entraînés"""

        # Chemin absolu vers le dossier des modèles (racine du projet)
        project_root = Path(__file__).parent.parent
        model_dir = project_root / 'ml_models' / 'saved_models'

        if not model_dir.exists():
            self.logger.warning(f"⚠️ Dossier modèles introuvable: {model_dir}")
            return

        # Charger XGBoost
        xgb_files = sorted(model_dir.glob('xgboost_*.pkl'))
        if xgb_files:
            try:
                self.xgboost.load(str(xgb_files[-1]))
                self.models_loaded['xgboost'] = True
                self.logger.info(f"✅ XGBoost chargé: {xgb_files[-1].name}")
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur chargement XGBoost: {e}")

        # Charger LightGBM
        lgb_files = sorted(model_dir.glob('lightgbm_*.pkl'))
        if lgb_files:
            try:
                self.lightgbm.load(str(lgb_files[-1]))
                self.models_loaded['lightgbm'] = True
                self.logger.info(f"✅ LightGBM chargé: {lgb_files[-1].name}")
            except Exception as e:
                self.logger.warning(f"⚠️ Erreur chargement LightGBM: {e}")

        # Vérifier si au moins un modèle est prêt
        self.is_ready = any(self.models_loaded.values())

        if not self.is_ready:
            self.logger.warning("⚠️ Aucun modèle ML chargé - filtre désactivé")

    def validate_signal(self, signal: Dict, market_data: Dict) -> Dict:
        """
        Valide un signal avec les modèles ML

        Args:
            signal: Le signal à valider (de la stratégie)
            market_data: Données de marché actuelles

        Returns:
            Dict avec:
            - validated: bool
            - confidence: float (0-1)
            - xgboost_conf: float
            - lightgbm_conf: float
            - recommendation: str
        """

        result = {
            'validated': False,
            'confidence': 0.0,
            'xgboost_conf': 0.0,
            'lightgbm_conf': 0.0,
            'recommendation': 'REJECT'
        }

        if not self.is_ready:
            # Pas de modèle = valider par défaut
            result['validated'] = True
            result['confidence'] = signal.get('confidence', 0.5)
            result['recommendation'] = 'PASS_THROUGH'
            return result

        try:
            # Préparer les features
            df = market_data.get('5m') or market_data.get('1m')
            if df is None or len(df) < 200:
                result['validated'] = True
                result['confidence'] = signal.get('confidence', 0.5)
                return result

            # Feature engineering
            df_features = self.feature_engineer.generate_features(df)
            if df_features.empty:
                result['validated'] = True
                result['confidence'] = signal.get('confidence', 0.5)
                return result

            # Obtenir les features
            feature_names = self.feature_engineer.get_feature_names(df_features)
            X = df_features[feature_names].iloc[[-1]]  # Dernière ligne

            # Prédictions des modèles
            signal_direction = 1 if signal['side'] == 'BUY' else 0

            # XGBoost
            xgb_conf = 0.0
            if self.models_loaded['xgboost']:
                try:
                    xgb_signal, xgb_raw_conf = self.xgboost.get_signal_with_confidence(X)
                    # Si le modèle prédit la même direction que le signal
                    if xgb_signal == signal_direction:
                        xgb_conf = xgb_raw_conf
                    else:
                        xgb_conf = 1 - xgb_raw_conf
                except Exception as e:
                    self.logger.debug(f"XGBoost error: {e}")
                    xgb_conf = 0.5

            # LightGBM
            lgb_conf = 0.0
            if self.models_loaded['lightgbm']:
                try:
                    lgb_signal, lgb_raw_conf = self.lightgbm.get_signal_with_confidence(X)
                    if lgb_signal == signal_direction:
                        lgb_conf = lgb_raw_conf
                    else:
                        lgb_conf = 1 - lgb_raw_conf
                except Exception as e:
                    self.logger.debug(f"LightGBM error: {e}")
                    lgb_conf = 0.5

            # Combiner les confidences
            total_weight = 0
            combined_conf = 0

            if self.models_loaded['xgboost']:
                combined_conf += xgb_conf * self.xgboost_weight
                total_weight += self.xgboost_weight

            if self.models_loaded['lightgbm']:
                combined_conf += lgb_conf * self.lightgbm_weight
                total_weight += self.lightgbm_weight

            if total_weight > 0:
                combined_conf = combined_conf / total_weight

            # Décision
            validated = combined_conf >= self.min_confidence

            result = {
                'validated': validated,
                'confidence': combined_conf,
                'xgboost_conf': xgb_conf,
                'lightgbm_conf': lgb_conf,
                'recommendation': 'ACCEPT' if validated else 'REJECT',
                'signal_direction': signal['side'],
                'ml_direction': 'BUY' if combined_conf > 0.5 else 'SELL'
            }

            self.logger.debug(
                f"ML Filter: {signal['strategy']} {signal['side']} | "
                f"XGB: {xgb_conf:.1%} | LGB: {lgb_conf:.1%} | "
                f"Combined: {combined_conf:.1%} | {result['recommendation']}"
            )

            return result

        except Exception as e:
            self.logger.error(f"❌ Erreur validation ML: {e}")
            # En cas d'erreur, laisser passer le signal
            result['validated'] = True
            result['confidence'] = signal.get('confidence', 0.5)
            return result

    def train_models(self, df: pd.DataFrame) -> Dict:
        """Entraîne les deux modèles"""

        self.logger.info("🔄 Entraînement des modèles ML...")

        # Feature engineering
        df_features = self.feature_engineer.generate_features(df)
        df_features = self.feature_engineer.create_target(df_features, horizon=5, threshold=0.003)
        df_features = df_features.dropna()

        if len(df_features) < 500:
            self.logger.error(f"❌ Pas assez de données: {len(df_features)}")
            return {}

        feature_names = self.feature_engineer.get_feature_names(df_features)
        X = df_features[feature_names]
        y = df_features['target']

        results = {}

        # Entraîner XGBoost
        self.logger.info("📊 Entraînement XGBoost...")
        try:
            xgb_metrics = self.xgboost.train(X, y)
            self.xgboost.save()
            results['xgboost'] = xgb_metrics
            self.models_loaded['xgboost'] = True
        except Exception as e:
            self.logger.error(f"❌ Erreur XGBoost: {e}")
            results['xgboost'] = {'error': str(e)}

        # Entraîner LightGBM
        self.logger.info("📊 Entraînement LightGBM...")
        try:
            lgb_metrics = self.lightgbm.train(X, y)
            self.lightgbm.save()
            results['lightgbm'] = lgb_metrics
            self.models_loaded['lightgbm'] = True
        except Exception as e:
            self.logger.error(f"❌ Erreur LightGBM: {e}")
            results['lightgbm'] = {'error': str(e)}

        self.is_ready = any(self.models_loaded.values())

        self.logger.info("✅ Entraînement terminé")
        return results

    def get_status(self) -> Dict:
        """Retourne le statut du filtre"""

        return {
            'is_ready': self.is_ready,
            'models_loaded': self.models_loaded,
            'min_confidence': self.min_confidence,
            'weights': {
                'xgboost': self.xgboost_weight,
                'lightgbm': self.lightgbm_weight
            },
            'xgboost_metrics': self.xgboost.get_metrics() if self.models_loaded['xgboost'] else {},
            'lightgbm_metrics': self.lightgbm.get_metrics() if self.models_loaded['lightgbm'] else {}
        }
