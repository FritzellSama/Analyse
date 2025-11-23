"""
Strategies Module - Quantum Trader Pro
"""

from .base_strategy import BaseStrategy, Signal
from .ichimoku_scalping import IchimokuScalpingStrategy
from .ml_strategy import MLStrategy
from .strategy_manager import StrategyManager

# HF Strategies (nouveau système)
try:
    from .high_frequency.hf_manager import HFManager
    from .high_frequency.rsi_extreme_bounce import RSIExtremeBounceStrategy
    from .high_frequency.vwap_deviation import VWAPDeviationStrategy
    from .high_frequency.liquidity_sweep import LiquiditySweepStrategy
    from .high_frequency.order_block_sniper import OrderBlockSniperStrategy
    from .high_frequency.golden_cross_sniper import GoldenCrossSniperStrategy
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

__all__ = [
    'BaseStrategy',
    'Signal',
    'IchimokuScalpingStrategy',
    'MLStrategy',
    'StrategyManager',
    'HFManager',
    'RSIExtremeBounceStrategy',
    'VWAPDeviationStrategy',
    'LiquiditySweepStrategy',
    'OrderBlockSniperStrategy',
    'GoldenCrossSniperStrategy',
]
