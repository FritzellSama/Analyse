"""
Grid Trading Strategy - Quantum Trader Pro
Stratégie de grille d'ordres
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from strategies.base_strategy import BaseStrategy, Signal
from utils.logger import setup_logger
from utils.safety import ensure_minimum_data
from utils.safe_math import safe_divide
from utils.config_helpers import get_nested_config

class GridTradingStrategy(BaseStrategy):
    """Stratégie Grid Trading"""

    def __init__(self, config: Dict):
        super().__init__('GridTrading', config)
        self.logger = setup_logger('GridTrading')

        # Config Grid - avec fallbacks sécurisés
        grid_cfg = get_nested_config(config, 'strategies', 'grid_trading', 'grid', default={})
        self.grid_type = grid_cfg.get('type', 'arithmetic')
        self.range_percent = grid_cfg.get('range_percent', 5.0)
        self.num_levels = max(2, grid_cfg.get('num_levels', 10))  # Minimum 2 niveaux pour éviter division par zéro
        self.profit_per_grid = grid_cfg.get('profit_per_grid', 0.5)
        
        # State
        self.grid_levels = []
        self.current_price = None
        self.grid_center = None
        self.grid_initialized = False
        self.open_grid_orders = {}  # {level: {'side': 'buy/sell', 'price': float}}
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule indicateurs pour range detection"""
        df = df.copy()
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # ATR pour volatilité
        import talib
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Bollinger pour range
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Volatilité
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * 100
        
        return df
    
    def detect_range_market(self, df: pd.DataFrame) -> bool:
        """Détecte si le marché est en range"""
        if not ensure_minimum_data(df, 50, "grid_detect_range"):
            return False

        last = df.iloc[-1]

        # Vérifier NaN
        if pd.isna(last.get('volatility', np.nan)):
            return False

        # Critère 1: Volatilité modérée (increased from 2% to 4% for crypto)
        if last['volatility'] < 4.0:
            # Critère 2: Prix dans BB middle 60% du temps (relaxed from 80%)
            recent = df.tail(20)
            bb_range = recent['bb_upper'] - recent['bb_lower']
            price_in_middle = ((recent['close'] > recent['bb_lower'] + bb_range * 0.15) &
                              (recent['close'] < recent['bb_upper'] - bb_range * 0.15))

            if price_in_middle.sum() / len(recent) > 0.6:  # Relaxed from 0.8
                return True

        return False
    
    def initialize_grid(self, current_price: float):
        """Initialise la grille d'ordres"""
        self.grid_center = current_price
        self.current_price = current_price

        # Calculer range
        range_amount = current_price * (self.range_percent / 100)
        upper_bound = current_price + range_amount
        lower_bound = current_price - range_amount

        # Validation: minimum 2 niveaux pour éviter division par zéro
        if self.num_levels < 2:
            self.logger.error(f"num_levels doit être >= 2, got {self.num_levels}. Setting to 2.")
            self.num_levels = 2

        if self.grid_type == 'arithmetic':
            # Grille arithmétique (espacement égal)
            # Protection division par zéro: num_levels - 1 >= 1 maintenant garanti
            step = safe_divide(upper_bound - lower_bound, self.num_levels - 1, default=0.0)
            self.grid_levels = [lower_bound + i * step for i in range(self.num_levels)]

        elif self.grid_type == 'geometric':
            # Grille géométrique (ratio constant)
            # Protection division par zéro et lower_bound = 0
            if lower_bound <= 0:
                self.logger.warning("lower_bound <= 0, using arithmetic grid instead")
                step = safe_divide(upper_bound - lower_bound, self.num_levels - 1, default=0.0)
                self.grid_levels = [lower_bound + i * step for i in range(self.num_levels)]
            else:
                exponent = safe_divide(1.0, self.num_levels - 1, default=1.0)
                ratio = (upper_bound / lower_bound) ** exponent
                self.grid_levels = [lower_bound * (ratio ** i) for i in range(self.num_levels)]
        else:
            # Fallback to arithmetic
            self.logger.warning(f"Unknown grid type '{self.grid_type}', using arithmetic")
            step = safe_divide(upper_bound - lower_bound, self.num_levels - 1, default=0.0)
            self.grid_levels = [lower_bound + i * step for i in range(self.num_levels)]

        self.grid_initialized = True

        self.logger.info(
            f"📊 Grid initialisée: center={current_price:.2f} "
            f"range=[{lower_bound:.2f}, {upper_bound:.2f}] "
            f"levels={self.num_levels}"
        )
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Génère signaux grid"""
        signals = []
        
        df = data.get(self.timeframes['signal'])
        if df is None or not ensure_minimum_data(df, 50, "grid_generate_signals"):
            return signals

        # Calculer indicateurs
        df = self.calculate_indicators(df)
        
        # Détecter range market
        is_range = self.detect_range_market(df)
        
        if not is_range:
            self.logger.debug("⚠️  Marché trending, pas de grid")
            self.grid_initialized = False
            return signals
        
        last = df.iloc[-1]
        current_price = last['close']
        atr = last['atr']
        
        # Initialiser grid si besoin
        if not self.grid_initialized:
            self.initialize_grid(current_price)
            return signals
        
        # Vérifier si prix sort du range
        if current_price < self.grid_levels[0] or current_price > self.grid_levels[-1]:
            self.logger.warning("⚠️  Prix hors grille, réinitialisation")
            self.initialize_grid(current_price)
            return signals
        
        # Trouver niveau le plus proche
        closest_level_idx = min(range(len(self.grid_levels)), 
                               key=lambda i: abs(self.grid_levels[i] - current_price))
        closest_level = self.grid_levels[closest_level_idx]
        
        # Générer signaux pour niveaux proches
        tolerance = atr * 0.5  # Tolérance pour trigger
        
        for i, level in enumerate(self.grid_levels):
            # Skip si ordre déjà placé
            if i in self.open_grid_orders:
                continue
            
            # BUY si prix proche d'un niveau bas
            if i < len(self.grid_levels) // 2:
                if abs(current_price - level) < tolerance and current_price <= level:
                    # Calculer TP (niveau supérieur)
                    if i < len(self.grid_levels) - 1:
                        tp_level = self.grid_levels[i + 1]
                        
                        signal = Signal(
                            timestamp=last.name if hasattr(last, 'name') else datetime.now(),
                            action='BUY',
                            symbol=self.symbol,
                            confidence=0.75,
                            entry_price=level,
                            stop_loss=self.grid_levels[0] - atr,  # Stop sous grille
                            take_profit=[(tp_level, 1.0)],  # 100% au niveau suivant
                            reason=['grid_level_buy', f'level_{i}'],
                            metadata={
                                'grid_level': i,
                                'grid_type': self.grid_type,
                                'expected_profit': safe_divide(tp_level - level, level, default=0.0) * 100
                            }
                        )
                        signals.append(signal)
                        self.open_grid_orders[i] = {'side': 'buy', 'price': level}
            
            # SELL si prix proche d'un niveau haut
            elif i >= len(self.grid_levels) // 2:
                if abs(current_price - level) < tolerance and current_price >= level:
                    # Calculer TP (niveau inférieur)
                    if i > 0:
                        tp_level = self.grid_levels[i - 1]
                        
                        signal = Signal(
                            timestamp=last.name if hasattr(last, 'name') else datetime.now(),
                            action='SELL',
                            symbol=self.symbol,
                            confidence=0.75,
                            entry_price=level,
                            stop_loss=self.grid_levels[-1] + atr,  # Stop au-dessus grille
                            take_profit=[(tp_level, 1.0)],
                            reason=['grid_level_sell', f'level_{i}'],
                            metadata={
                                'grid_level': i,
                                'grid_type': self.grid_type,
                                'expected_profit': safe_divide(level - tp_level, level, default=0.0) * 100
                            }
                        )
                        signals.append(signal)
                        self.open_grid_orders[i] = {'side': 'sell', 'price': level}
        
        if signals:
            self.logger.info(f"📊 {len(signals)} signaux grid générés")
        
        return signals
    
    def on_order_filled(self, level: int):
        """Callback quand ordre rempli"""
        if level in self.open_grid_orders:
            del self.open_grid_orders[level]
    
    def reset_grid(self):
        """Reset la grille"""
        self.grid_levels = []
        self.grid_initialized = False
        self.open_grid_orders = {}
        self.logger.info("🔄 Grid reset")

__all__ = ['GridTradingStrategy']
