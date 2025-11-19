"""
ML Strategy - Quantum Trader Pro
Stratégie basée 100% sur Machine Learning (XGBoost + LSTM + Ensemble)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from strategies.base_strategy import BaseStrategy, Signal
from ml_models.ensemble import EnsembleModel
from ml_models.feature_engineering import FeatureEngineer
from ml_models.xgboost_model import XGBoostModel
from utils.logger import setup_logger
from utils.calculations import calculate_atr, calculate_rsi, calculate_macd, calculate_bollinger_bands
from utils.safety import ensure_minimum_data, safe_last_value

class MLStrategy(BaseStrategy):
    """
    Stratégie ML qui:
    1. Génère features techniques via FeatureEngineering
    2. Utilise EnsembleModel pour prédictions
    3. Génère signaux basés sur ML uniquement
    """
    
    def __init__(self, config: Dict):
        """
        Initialise la stratégie ML
        
        Args:
            config: Configuration du bot
        """
        super().__init__('ml_strategy', config)
        
        self.logger = setup_logger('MLStrategy')
        
        # ML Config
        ml_config = config.get('ml', {})
        
        # Feature Engineering
        self.feature_engineering = FeatureEngineer(config)
        
        # Ensemble Model (sera chargé avec modèles entraînés)
        self.ensemble = EnsembleModel(config)
        
        # Seuils de décision
        self.min_confidence = ml_config.get('min_confidence', 0.7)
        self.min_agreement = ml_config.get('min_agreement', 0.6)
        
        # État
        self.models_loaded = False
        
        self.logger.info("✅ ML Strategy initialisée")
    
    def load_models(self, models_path: str):
        """
        Charge les modèles ML entraînés

        Args:
            models_path: Chemin vers les modèles sauvegardés
        """
        try:
            models_dir = Path(models_path)

            if not models_dir.exists():
                self.logger.warning(f"⚠️ Dossier modèles non trouvé: {models_path}")
                self.models_loaded = False
                return

            # Charger XGBoost Model
            xgb_path = models_dir / 'xgboost_model.joblib'
            if xgb_path.exists():
                xgb_model = XGBoostModel(self.config)
                xgb_model.load(str(xgb_path))
                self.ensemble.add_model('xgboost', xgb_model)
                self.logger.info(f"✅ XGBoost chargé depuis {xgb_path}")
            else:
                self.logger.warning(f"⚠️ XGBoost non trouvé: {xgb_path}")

            # Charger LSTM Model (optionnel, plus lourd)
            lstm_path = models_dir / 'lstm_model.h5'
            if lstm_path.exists():
                try:
                    from ml_models.lstm_model import LSTMModel
                    lstm_model = LSTMModel(self.config)
                    lstm_model.load(str(lstm_path))
                    self.ensemble.add_model('lstm', lstm_model)
                    self.logger.info(f"✅ LSTM chargé depuis {lstm_path}")
                except ImportError as ie:
                    self.logger.warning(f"⚠️ TensorFlow non disponible pour LSTM: {ie}")
            else:
                self.logger.debug(f"ℹ️ LSTM non trouvé (optionnel): {lstm_path}")

            # Vérifier qu'au moins un modèle est chargé
            if len(self.ensemble.models) > 0:
                self.models_loaded = True
                self.logger.info(f"✅ {len(self.ensemble.models)} modèle(s) ML chargé(s)")
            else:
                self.logger.error("❌ Aucun modèle ML chargé")
                self.models_loaded = False

        except Exception as e:
            self.logger.error(f"❌ Erreur chargement modèles: {e}")
            import traceback
            traceback.print_exc()
            self.models_loaded = False
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Génère signaux basés sur ML
        
        Args:
            data: Dict timeframe → DataFrame OHLCV
            
        Returns:
            Liste de signaux ML
        """
        
        if not self.models_loaded:
            self.logger.debug("⚠️ Modèles ML non chargés - pas de signaux")
            return []

        # Utiliser la timeframe principale
        main_tf = self.timeframes.get('signal', '1h') if self.timeframes else '1h'

        if main_tf not in data or len(data[main_tf]) < 100:
            return []
        
        df = data[main_tf].copy()
        
        # Calculer indicateurs
        df = self.calculate_indicators(df)
        
        # Générer features ML
        try:
            features_df = self.feature_engineering.generate_features(df)
            
            if len(features_df) < 50:
                return []
            
            # Prédiction ensemble
            should_trade, signal_direction, confidence = self.ensemble.should_trade(
                features_df,
                min_confidence=self.min_confidence,
                min_agreement=self.min_agreement
            )
            
            if not should_trade:
                return []

            # Vérifier données suffisantes
            if not ensure_minimum_data(df, 1, "ml_strategy_signal"):
                return []

            # Créer signal
            current_price = df['close'].iloc[-1]
            current_time = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
            
            action = 'BUY' if signal_direction == 1 else 'SELL'
            
            # Calculer stop loss et take profit basés sur ML
            atr = self._calculate_atr(df)
            
            if action == 'BUY':
                stop_loss = current_price - (atr * 2.0)
                take_profits = [
                    (current_price + (atr * 2.0), 0.5),  # 50% à 2R
                    (current_price + (atr * 3.5), 0.3),  # 30% à 3.5R
                    (current_price + (atr * 5.0), 0.2),  # 20% à 5R
                ]
            else:  # SELL
                stop_loss = current_price + (atr * 2.0)
                take_profits = [
                    (current_price - (atr * 2.0), 0.5),
                    (current_price - (atr * 3.5), 0.3),
                    (current_price - (atr * 5.0), 0.2),
                ]
            
            signal = Signal(
                timestamp=current_time,
                action=action,
                symbol=self.symbol,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profits,
                reason=[f'ml_ensemble_{signal_direction}', f'confidence_{confidence:.2f}'],
                metadata={
                    'model': 'ensemble',
                    'agreement': self.ensemble.get_agreement_score(features_df),
                    'atr': atr
                }
            )
            
            self.logger.info(
                f"🤖 Signal ML: {action} @ ${current_price:.2f} | "
                f"Confidence: {confidence:.2%} | SL: ${stop_loss:.2f}"
            )
            
            return [signal]
            
        except Exception as e:
            self.logger.error(f"❌ Erreur génération signal ML: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs de base (utilise fonctions centralisées)

        Args:
            df: DataFrame OHLCV

        Returns:
            DataFrame avec indicateurs
        """
        from utils.calculations import calculate_sma, calculate_ema

        # SMA (centralisé)
        df['sma_20'] = calculate_sma(df['close'], 20)
        df['sma_50'] = calculate_sma(df['close'], 50)

        # EMA (centralisé)
        df['ema_12'] = calculate_ema(df['close'], 12)
        df['ema_26'] = calculate_ema(df['close'], 26)

        # MACD (centralisé)
        macd_line, macd_signal, _ = calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal

        # RSI (centralisé)
        df['rsi'] = calculate_rsi(df['close'], 14)

        # Bollinger Bands (centralisé)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower

        # ATR (centralisé)
        df['atr'] = calculate_atr(df, 14)

        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcule ATR (utilise fonction centralisée)"""
        if not ensure_minimum_data(df, 1, "ml_strategy_atr"):
            return 0.0

        try:
            if 'atr' in df.columns and len(df) > 0:
                atr_val = safe_last_value(df['atr'], default=None)
                if atr_val is not None and not pd.isna(atr_val):
                    return float(atr_val)

            # Utiliser fonction centralisée
            atr_series = calculate_atr(df, period)
            atr_val = safe_last_value(atr_series, default=None)
            if atr_val is not None and not pd.isna(atr_val):
                return float(atr_val)
            else:
                close_val = safe_last_value(df['close'], default=0.0)
                return float(close_val * 0.02) if close_val > 0 else 0.0
        except Exception as e:
            self.logger.warning(f"Erreur calcul ATR: {e}")
            close_val = safe_last_value(df['close'], default=0.0) if len(df) > 0 else 0.0
            return float(close_val * 0.02) if close_val > 0 else 0.0
    
    def get_model_info(self) -> Dict:
        """Retourne infos sur les modèles ML"""
        
        return {
            'models_loaded': self.models_loaded,
            'ensemble_summary': self.ensemble.get_summary() if self.models_loaded else {},
            'min_confidence': self.min_confidence,
            'min_agreement': self.min_agreement
        }

__all__ = ['MLStrategy']
