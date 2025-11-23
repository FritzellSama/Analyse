"""
Ichimoku Scalping Strategy - Quantum Trader Pro
Stratégie Ichimoku H1 + RSI/BB M5
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List
from datetime import datetime
from strategies.base_strategy import BaseStrategy, Signal
from utils.logger import setup_logger
from utils.safety import ensure_minimum_data
from utils.safe_math import safe_divide
from utils.config_helpers import get_nested_config

class IchimokuScalpingStrategy(BaseStrategy):
    """Stratégie Ichimoku + RSI/BB"""
    
    def __init__(self, config: Dict):
        super().__init__('IchimokuScalping', config)
        self.logger = setup_logger('IchimokuScalping')

        # Config Ichimoku - avec fallbacks sécurisés
        ichi_cfg = config.get('strategies', {}).get('ichimoku_scalping', {}).get('ichimoku', {})
        self.tenkan_period = ichi_cfg.get('tenkan_period', 9)
        self.kijun_period = ichi_cfg.get('kijun_period', 26)
        self.senkou_b_period = ichi_cfg.get('senkou_span_b_period', 52)
        self.displacement = ichi_cfg.get('displacement', 26)

        # Config Scalping - avec fallbacks sécurisés
        scalp_cfg = config.get('strategies', {}).get('ichimoku_scalping', {}).get('scalping', {})
        rsi_cfg = scalp_cfg.get('rsi', {})
        self.rsi_period = rsi_cfg.get('period', 14)
        self.rsi_oversold = rsi_cfg.get('oversold', 30)
        self.rsi_overbought = rsi_cfg.get('overbought', 70)

        bb_cfg = scalp_cfg.get('bollinger', {})
        self.bb_period = bb_cfg.get('period', 20)
        self.bb_std = bb_cfg.get('std_dev', 2.0)

        volume_cfg = scalp_cfg.get('volume', {})
        self.volume_ma_period = volume_cfg.get('ma_period', 20)
        self.min_volume_ratio = volume_cfg.get('min_ratio', 1.0)

        # ATR pour stops - avec fallbacks sécurisés
        stop_loss_cfg = config.get('risk', {}).get('stop_loss', {})
        self.atr_period = stop_loss_cfg.get('atr_multiplier', 1.5) * 10
        self.sl_atr_mult = stop_loss_cfg.get('atr_multiplier', 1.5)
        tp_cfg = config.get('risk', {}).get('take_profit', {})
        tp_levels = tp_cfg.get('levels', [{}])
        self.tp_atr_mult = tp_levels[0].get('multiplier', 2.5) if tp_levels else 2.5
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule indicateurs"""
        df = df.copy()
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Ichimoku
        tenkan_sen = (pd.Series(high).rolling(self.tenkan_period).max() + 
                     pd.Series(low).rolling(self.tenkan_period).min()) / 2
        kijun_sen = (pd.Series(high).rolling(self.kijun_period).max() + 
                    pd.Series(low).rolling(self.kijun_period).min()) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.displacement)
        senkou_span_b = ((pd.Series(high).rolling(self.senkou_b_period).max() + 
                         pd.Series(low).rolling(self.senkou_b_period).min()) / 2).shift(self.displacement)
        
        chikou_span = pd.Series(close).shift(-self.displacement)
        
        df['tenkan_sen'] = tenkan_sen
        df['kijun_sen'] = kijun_sen
        df['senkou_span_a'] = senkou_span_a
        df['senkou_span_b'] = senkou_span_b
        df['chikou_span'] = chikou_span
        
        # Cloud
        df['cloud_top'] = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        df['cloud_bottom'] = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        df['cloud_thickness'] = safe_divide(df['cloud_top'] - df['cloud_bottom'], df['close'], default=0.0)
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=self.rsi_period)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            close, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # Volume
        df['volume_ma'] = talib.SMA(volume, timeperiod=self.volume_ma_period)
        df['volume_ratio'] = volume / (df['volume_ma'] + 1e-8)
        
        # ATR
        df['atr'] = talib.ATR(high, low, close, timeperiod=self.atr_period)
        
        return df
    
    def get_ichimoku_direction(self, df: pd.DataFrame) -> str:
        """Détermine direction Ichimoku"""
        min_required = self.senkou_b_period + self.displacement
        if not ensure_minimum_data(df, min_required, "ichimoku_direction"):
            return 'NEUTRAL'

        last = df.iloc[-1]
        
        # Vérifier NaN
        required_cols = ['tenkan_sen', 'kijun_sen', 'cloud_top', 'cloud_bottom', 'chikou_span']
        if last[required_cols].isna().any():
            return 'NEUTRAL'
        
        close = last['close']
        tenkan = last['tenkan_sen']
        kijun = last['kijun_sen']
        cloud_top = last['cloud_top']
        cloud_bottom = last['cloud_bottom']
        
        # Conditions BULLISH
        bullish_score = 0
        if close > cloud_top:
            bullish_score += 2
        elif close > cloud_bottom:  # Inside cloud but above bottom = weak bullish
            bullish_score += 1
        if tenkan > kijun:
            bullish_score += 1
        if last['chikou_span'] > close:
            bullish_score += 1

        # Conditions BEARISH
        bearish_score = 0
        if close < cloud_bottom:
            bearish_score += 2
        elif close < cloud_top:  # Inside cloud but below top = weak bearish
            bearish_score += 1
        if tenkan < kijun:
            bearish_score += 1
        if last['chikou_span'] < close:
            bearish_score += 1

        # STRICT: Require 3+ conditions for high win rate (was 2)
        if bullish_score >= 3:
            return 'BULLISH'
        elif bearish_score >= 3:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Génère signaux de trading"""
        signals = []
        
        # Data H1 et M5
        df_trend = data.get(self.timeframes['trend'])
        df_signal = data.get(self.timeframes['signal'])
        
        if df_trend is None or df_signal is None:
            return signals
        
        # Calculer indicateurs
        df_trend = self.calculate_indicators(df_trend)
        df_signal = self.calculate_indicators(df_signal)

        min_required = max(self.bb_period, self.volume_ma_period, 2) + 1
        if not ensure_minimum_data(df_signal, min_required, "ichimoku_generate_signals"):
            return signals

        # Direction Ichimoku
        direction = self.get_ichimoku_direction(df_trend)

        if direction == 'NEUTRAL':
            return signals

        # Analyser dernières bougies M5 - sécurisé car min_required >= 3
        try:
            last = df_signal.iloc[-1]
            prev = df_signal.iloc[-2]
        except IndexError:
            self.logger.warning("⚠️ Données insuffisantes pour analyse Ichimoku")
            return signals
        
        close = last['close']
        rsi = last['rsi']
        bb_upper = last['bb_upper']
        bb_lower = last['bb_lower']
        bb_position = last['bb_position']
        volume_ratio = last['volume_ratio']
        atr = last['atr']

        # Vérifier NaN - utiliser pd.isna() qui gère les scalaires
        if any(pd.isna(x) for x in [rsi, bb_upper, bb_lower, volume_ratio, atr]):
            return signals
        
        # SIGNAL LONG
        if direction == 'BULLISH' and not self.position_open:
            conditions = []
            confidence = 0.0
            
            # RSI survendu
            if rsi < self.rsi_oversold:
                conditions.append('rsi_oversold')
                confidence += 0.35
            elif rsi < 40:
                conditions.append('rsi_low')
                confidence += 0.2
            
            # Prix touche BB lower
            if close <= bb_lower * 1.002:
                conditions.append('bb_lower_touch')
                confidence += 0.35
            elif bb_position < 0.2:
                conditions.append('bb_lower_near')
                confidence += 0.2
            
            # Volume
            if volume_ratio >= self.min_volume_ratio:
                conditions.append('volume_surge')
                confidence += 0.2
            
            # RSI reversal
            if rsi > prev['rsi']:
                conditions.append('rsi_reversal')
                confidence += 0.1

            # Signal valide - STRICT: 3+ conditions AND 0.70 confidence for high win rate
            if len(conditions) >= 3 and confidence >= 0.70:
                stop_loss = close - (atr * self.sl_atr_mult)
                tp_distance = atr * self.tp_atr_mult

                signal = Signal(
                    timestamp=last.name if hasattr(last, 'name') else datetime.now(),
                    action='BUY',
                    symbol=self.symbol,
                    confidence=min(confidence, 0.95),
                    entry_price=close,
                    stop_loss=stop_loss,
                    take_profit=[
                        (close + tp_distance * 0.6, 0.5),  # TP1: 50% @ 1.5x
                        (close + tp_distance * 1.0, 0.3),  # TP2: 30% @ 2.5x
                        (close + tp_distance * 1.6, 0.2),  # TP3: 20% @ 4x
                    ],
                    reason=conditions,
                    metadata={
                        'rsi': float(rsi),
                        'bb_position': float(bb_position),
                        'volume_ratio': float(volume_ratio),
                        'atr': float(atr),
                        'ichimoku_direction': direction
                    }
                )
                signals.append(signal)
                self.logger.info(f"📊 Signal LONG: conf={confidence:.2f} reasons={conditions}")
        
        # SIGNAL SHORT
        elif direction == 'BEARISH' and not self.position_open:
            conditions = []
            confidence = 0.0
            
            # RSI suracheté
            if rsi > self.rsi_overbought:
                conditions.append('rsi_overbought')
                confidence += 0.35
            elif rsi > 60:
                conditions.append('rsi_high')
                confidence += 0.2
            
            # Prix touche BB upper
            if close >= bb_upper * 0.998:
                conditions.append('bb_upper_touch')
                confidence += 0.35
            elif bb_position > 0.8:
                conditions.append('bb_upper_near')
                confidence += 0.2
            
            # Volume
            if volume_ratio >= self.min_volume_ratio:
                conditions.append('volume_surge')
                confidence += 0.2
            
            # RSI reversal
            if rsi < prev['rsi']:
                conditions.append('rsi_reversal')
                confidence += 0.1
            
            # Signal valide - STRICT: 3+ conditions AND 0.70 confidence for high win rate
            if len(conditions) >= 3 and confidence >= 0.70:
                stop_loss = close + (atr * self.sl_atr_mult)
                tp_distance = atr * self.tp_atr_mult

                signal = Signal(
                    timestamp=last.name if hasattr(last, 'name') else datetime.now(),
                    action='SELL',
                    symbol=self.symbol,
                    confidence=min(confidence, 0.95),
                    entry_price=close,
                    stop_loss=stop_loss,
                    take_profit=[
                        (close - tp_distance * 0.6, 0.5),
                        (close - tp_distance * 1.0, 0.3),
                        (close - tp_distance * 1.6, 0.2),
                    ],
                    reason=conditions,
                    metadata={
                        'rsi': float(rsi),
                        'bb_position': float(bb_position),
                        'volume_ratio': float(volume_ratio),
                        'atr': float(atr),
                        'ichimoku_direction': direction
                    }
                )
                signals.append(signal)
                self.logger.info(f"📊 Signal SHORT: conf={confidence:.2f} reasons={conditions}")
        
        return signals

__all__ = ['IchimokuScalpingStrategy']
