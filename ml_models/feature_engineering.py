"""
Feature Engineering - Quantum Trader Pro
Extraction et transformation des features pour les modèles ML
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from utils.logger import setup_logger
from utils.calculations import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_atr, calculate_sma, calculate_ema
)
from utils.safety import safe_dataframe_access, ensure_minimum_data
from utils.safe_math import safe_divide
from utils.config_helpers import get_nested_config


class FeatureEngineer:
    """
    Ingénieur de features qui génère:
    - Technical indicators (20+)
    - Market microstructure features
    - Sentiment features
    - Time-based features
    - Price patterns
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le feature engineer
        
        Args:
            config: Configuration complète du bot
        """
        self.config = config
        self.logger = setup_logger('FeatureEngineer')
        
        # Configuration ML
        ml_config = config.get('ml', {})
        features_config = ml_config.get('features', {})
        
        self.technical_indicators = features_config.get('technical_indicators', [
            'rsi', 'macd', 'bollinger_bands', 'atr', 'adx', 
            'stochastic', 'cci', 'mfi'
        ])
        
        self.market_features = features_config.get('market_microstructure', [
            'order_book_imbalance', 'bid_ask_spread', 
            'volume_profile', 'trade_intensity'
        ])
        
        self.sentiment_features = features_config.get('sentiment', [
            'fear_greed_index', 'funding_rate', 'open_interest'
        ])
        
        self.time_features = features_config.get('time_features', [
            'hour_of_day', 'day_of_week', 'month', 'is_weekend'
        ])
        
        self.logger.info("✅ Feature Engineer initialisé")
    
    def generate_features(
        self,
        df: pd.DataFrame,
        orderbook_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Génère toutes les features pour le ML

        Args:
            df: DataFrame OHLCV
            orderbook_data: Données orderbook (optionnel)

        Returns:
            DataFrame avec toutes les features
        """

        # Validation sécurité
        if not safe_dataframe_access(df, "feature_engineering"):
            self.logger.error("❌ DataFrame invalide pour feature engineering")
            return pd.DataFrame()

        if not ensure_minimum_data(df, 200, "feature_engineering"):
            self.logger.warning("⚠️ Données insuffisantes pour feature engineering (min 200)")
            return pd.DataFrame()

        self.logger.info(f"🔨 Génération features sur {len(df)} bougies")

        # Créer copie pour éviter modifications
        df_features = df.copy()

        # 1. Technical indicators
        df_features = self._add_technical_indicators(df_features)

        # 2. Price patterns
        df_features = self._add_price_patterns(df_features)

        # 3. Volume features
        df_features = self._add_volume_features(df_features)

        # 4. Market microstructure (si orderbook disponible)
        if orderbook_data:
            df_features = self._add_market_microstructure(df_features, orderbook_data)

        # 5. Time features
        df_features = self._add_time_features(df_features)

        # 6. Lag features
        df_features = self._add_lag_features(df_features)

        # 7. Rolling statistics
        df_features = self._add_rolling_stats(df_features)

        # Supprimer NaN
        df_features = df_features.dropna()

        num_features = len([col for col in df_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
        self.logger.info(f"✅ {num_features} features générées")

        return df_features
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les indicateurs techniques (utilise fonctions centralisées)"""

        # RSI - utilise utils.calculations.calculate_rsi
        if 'rsi' in self.technical_indicators:
            df['rsi'] = calculate_rsi(df['close'], period=14)
            df['rsi_7'] = calculate_rsi(df['close'], period=7)

        # MACD - utilise utils.calculations.calculate_macd
        if 'macd' in self.technical_indicators:
            macd_line, signal_line, histogram = calculate_macd(df['close'])
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_histogram'] = histogram

        # Bollinger Bands - utilise utils.calculations.calculate_bollinger_bands
        if 'bollinger_bands' in self.technical_indicators:
            upper, middle, lower = calculate_bollinger_bands(df['close'])
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = safe_divide(upper - lower, middle, default=0.0)
            df['bb_position'] = safe_divide(df['close'] - lower, upper - lower, default=0.5)

        # ATR - utilise utils.calculations.calculate_atr
        if 'atr' in self.technical_indicators:
            df['atr'] = calculate_atr(df, period=14)
            df['atr_percent'] = safe_divide(df['atr'], df['close'], default=0.0)

        # ADX (calcul spécifique, pas dans utils)
        if 'adx' in self.technical_indicators:
            df = self._calculate_adx(df)

        # Stochastic
        if 'stochastic' in self.technical_indicators:
            df = self._calculate_stochastic(df)

        # CCI
        if 'cci' in self.technical_indicators:
            df = self._calculate_cci(df)

        # MFI
        if 'mfi' in self.technical_indicators:
            df = self._calculate_mfi(df)

        # Moving Averages - utilise utils.calculations
        for period in [7, 20, 50, 200]:
            df[f'sma_{period}'] = calculate_sma(df['close'], period)
            df[f'ema_{period}'] = calculate_ema(df['close'], period)

        # Distance from MAs
        df['dist_sma_20'] = safe_divide(df['close'] - df['sma_20'], df['sma_20'], default=0.0)
        df['dist_sma_50'] = safe_divide(df['close'] - df['sma_50'], df['sma_50'], default=0.0)

        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX"""
        # Plus DM et Minus DM
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = -df['low'].diff()
        
        df['plus_dm'] = np.where(
            (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
            df['high_diff'], 0
        )
        df['minus_dm'] = np.where(
            (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
            df['low_diff'], 0
        )

        # ATR pour normalisation
        if 'atr' not in df.columns:
            df['atr'] = calculate_atr(df, period)

        # DI+ et DI-
        df['plus_di'] = 100 * safe_divide(df['plus_dm'].rolling(period).mean(), df['atr'], default=0.0)
        df['minus_di'] = 100 * safe_divide(df['minus_dm'].rolling(period).mean(), df['atr'], default=0.0)

        # DX et ADX
        df['dx'] = 100 * safe_divide(np.abs(df['plus_di'] - df['minus_di']), df['plus_di'] + df['minus_di'], default=0.0)
        df['adx'] = df['dx'].rolling(period).mean()
        
        # Cleanup
        df = df.drop(['high_diff', 'low_diff', 'plus_dm', 'minus_dm'], axis=1)
        
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        df['stoch_lowest'] = df['low'].rolling(period).min()
        df['stoch_highest'] = df['high'].rolling(period).max()

        df['stoch_k'] = 100 * safe_divide(df['close'] - df['stoch_lowest'], df['stoch_highest'] - df['stoch_lowest'], default=50.0)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        df = df.drop(['stoch_lowest', 'stoch_highest'], axis=1)

        return df
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())

        df['cci'] = safe_divide(tp - sma_tp, 0.015 * mad, default=0.0)

        return df
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Money Flow Index"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']

        mf_positive = np.where(tp > tp.shift(1), mf, 0)
        mf_negative = np.where(tp < tp.shift(1), mf, 0)

        mf_positive_sum = pd.Series(mf_positive).rolling(period).sum()
        mf_negative_sum = pd.Series(mf_negative).rolling(period).sum()

        # Protection contre division par zéro
        money_ratio = np.where(
            mf_negative_sum != 0,
            mf_positive_sum / mf_negative_sum,
            100.0  # Si pas de flux négatif, MFI = 100
        )
        df['mfi'] = 100 - (100 / (1 + money_ratio))

        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les patterns de prix"""
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low range
        df['hl_range'] = safe_divide(df['high'] - df['low'], df['close'], default=0.0)

        # Close position dans la bougie
        df['close_position'] = safe_divide(df['close'] - df['low'], df['high'] - df['low'], default=0.5)

        # Body size
        df['body_size'] = safe_divide(np.abs(df['close'] - df['open']), df['close'], default=0.0)

        # Upper/Lower shadows
        df['upper_shadow'] = safe_divide(df['high'] - np.maximum(df['open'], df['close']), df['close'], default=0.0)
        df['lower_shadow'] = safe_divide(np.minimum(df['open'], df['close']) - df['low'], df['close'], default=0.0)
        
        # Candle type
        df['is_green'] = (df['close'] > df['open']).astype(int)
        
        # Gap
        df['gap'] = safe_divide(df['open'] - df['close'].shift(1), df['close'].shift(1), default=0.0)

        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            df[f'roc_{period}'] = safe_divide(df['close'] - df['close'].shift(period), df['close'].shift(period), default=0.0) * 100
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features de volume"""
        
        # Volume moving averages
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = safe_divide(df['volume'], df['volume_ma_20'], default=1.0)
        
        # Volume trend
        df['volume_trend'] = df['volume'].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ma'] = df['obv'].rolling(20).mean()
        
        # Volume-Price Trend
        df['vpt'] = (df['volume'] * df['returns']).fillna(0).cumsum()
        
        return df
    
    def _add_market_microstructure(self, df: pd.DataFrame, orderbook_data: Dict) -> pd.DataFrame:
        """Ajoute les features de microstructure"""
        
        # Ces features sont constantes pour tout le DataFrame (snapshot)
        # Dans une implémentation temps réel, ces valeurs changeraient
        
        if 'order_book_imbalance' in self.market_features:
            df['orderbook_imbalance'] = orderbook_data.get('imbalance', 0)
        
        if 'bid_ask_spread' in self.market_features:
            df['bid_ask_spread'] = orderbook_data.get('spread', 0)
        
        if 'trade_intensity' in self.market_features:
            # Simulé par le volume normalisé
            df['trade_intensity'] = df['volume_ratio']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features temporelles"""
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        if 'hour_of_day' in self.time_features:
            df['hour'] = df.index.hour
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'day_of_week' in self.time_features:
            df['day_of_week'] = df.index.dayofweek
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        if 'month' in self.time_features:
            df['month'] = df.index.month
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        if 'is_weekend' in self.time_features:
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features de lag (valeurs passées)"""
        
        # Lags des prix
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Lags des indicateurs importants
        if 'rsi' in df.columns:
            for lag in [1, 2, 5]:
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        
        if 'macd' in df.columns:
            for lag in [1, 2]:
                df[f'macd_lag_{lag}'] = df['macd'].shift(lag)
        
        return df
    
    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les statistiques rolling"""
        
        # Volatilité rolling
        for window in [10, 20, 50]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'volatility_{window}_pct'] = safe_divide(df[f'volatility_{window}'], df['returns'].rolling(window).mean(), default=0.0)

        # Skewness et Kurtosis
        df['skewness_20'] = df['returns'].rolling(20).skew()
        df['kurtosis_20'] = df['returns'].rolling(20).kurt()

        # Drawdown
        df['cum_max'] = df['close'].cummax()
        df['drawdown'] = safe_divide(df['close'] - df['cum_max'], df['cum_max'], default=0.0)
        
        return df
    
    def create_target(self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.003) -> pd.DataFrame:
        """
        Crée la variable target pour le ML

        Args:
            df: DataFrame avec features
            horizon: Nombre de bougies à prédire
            threshold: Seuil pour classification (0.3% par défaut - mouvement significatif)

        Returns:
            DataFrame avec colonne 'target'
        """
        
        # Future return
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Classification binaire
        # 1 = UP (achat profitable)
        # 0 = DOWN (vente ou neutre)
        df['target'] = (df['future_return'] > threshold).astype(int)
        
        # Regression target (optionnel)
        df['target_regression'] = df['future_return']
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Retourne la liste des noms de features (sans OHLCV et target)"""
        
        exclude = ['open', 'high', 'low', 'close', 'volume', 
                   'target', 'target_regression', 'future_return']
        
        features = [col for col in df.columns if col not in exclude]
        
        return features
    
    def scale_features(self, df: pd.DataFrame, scaler=None) -> Tuple[pd.DataFrame, object]:
        """
        Scale les features (normalisation)
        
        Args:
            df: DataFrame avec features
            scaler: Scaler sklearn (si None, crée nouveau)
        
        Returns:
            (DataFrame scaled, scaler)
        """
        
        from sklearn.preprocessing import StandardScaler
        
        feature_names = self.get_feature_names(df)
        
        if scaler is None:
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(df[feature_names])
        else:
            scaled_values = scaler.transform(df[feature_names])
        
        # Créer nouveau DataFrame
        df_scaled = df.copy()
        df_scaled[feature_names] = scaled_values
        
        return df_scaled, scaler
