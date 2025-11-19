"""
Base Strategy - Quantum Trader Pro
Classe abstraite pour toutes les stratégies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from utils.validators import validate_price as _validate_price_util

@dataclass
class Signal:
    """Signal de trading"""
    timestamp: datetime
    action: str  # 'BUY', 'SELL', 'CLOSE'
    symbol: str
    confidence: float  # 0.0 to 1.0
    entry_price: float
    strategy: str = ""  # Nom de la stratégie qui a généré le signal
    stop_loss: Optional[float] = None
    take_profit: Optional[List[Tuple[float, float]]] = None  # [(price, size_pct), ...]
    size: Optional[float] = None
    reason: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def __repr__(self):
        return (f"Signal({self.action} {self.symbol} @ {self.entry_price:.2f} "
                f"conf={self.confidence:.2f} sl={self.stop_loss} tp={self.take_profit})")

class BaseStrategy(ABC):
    """Classe de base abstraite pour stratégies"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.symbol = config.get('symbols', {}).get('primary', 'BTC/USDT')
        timeframes_cfg = config.get('timeframes', {})
        self.timeframes = {
            'trend': timeframes_cfg.get('trend', '1h'),
            'signal': timeframes_cfg.get('signal', '5m'),
            'micro': timeframes_cfg.get('micro', '1m')
        }

        # Signal validation parameters - configurable
        self.min_confidence = config.get('strategies', {}).get('min_signal_confidence', 0.5)

        # Performance tracking
        self.total_signals = 0
        self.winning_signals = 0
        self.losing_signals = 0
        self.total_pnl = 0.0
        
        # State
        self.last_signal = None
        self.position_open = False
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Génère des signaux de trading
        
        Args:
            data: Dict avec clé=timeframe, value=DataFrame OHLCV
        
        Returns:
            Liste de signaux
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques
        
        Args:
            df: DataFrame OHLCV
        
        Returns:
            DataFrame avec indicateurs ajoutés
        """
        pass
    
    def validate_signal(self, signal: Signal) -> bool:
        """Valide un signal avant exécution"""
        # Vérifier confiance minimale (configurable)
        if signal.confidence < self.min_confidence:
            return False
        
        # Valider les prix
        if not self._validate_price(signal.entry_price):
            return False
        
        # Vérifier stop loss
        if signal.stop_loss is None:
            return False
        
        if not self._validate_price(signal.stop_loss):
            return False
        
        # Vérifier que SL est dans le bon sens
        if signal.action == 'BUY' and signal.stop_loss >= signal.entry_price:
            return False
        if signal.action == 'SELL' and signal.stop_loss <= signal.entry_price:
            return False
        
        # Valider take profit si présent
        if signal.take_profit:
            for tp_price, _ in signal.take_profit:
                if not self._validate_price(tp_price):
                    return False
                # Vérifier cohérence TP
                if signal.action == 'BUY' and tp_price <= signal.entry_price:
                    return False
                if signal.action == 'SELL' and tp_price >= signal.entry_price:
                    return False
        
        return True
    
    def _validate_price(self, price: float) -> bool:
        """Valide qu'un prix est raisonnable (utilise fonction centralisée)"""
        return _validate_price_util(price, "price")
    
    def update_performance(self, pnl: float):
        """Met à jour les stats de performance"""
        self.total_signals += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_signals += 1
        else:
            self.losing_signals += 1
    
    def get_performance_stats(self) -> Dict:
        """Retourne stats de performance"""
        win_rate = (self.winning_signals / self.total_signals * 100
                   if self.total_signals > 0 else 0)

        return {
            'name': self.name,
            'total_signals': self.total_signals,
            'winning_signals': self.winning_signals,
            'losing_signals': self.losing_signals,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'avg_pnl_per_signal': (self.total_pnl / self.total_signals
                                   if self.total_signals > 0 else 0)
        }

    def reset_performance(self):
        """Reset all performance tracking"""
        self.total_signals = 0
        self.winning_signals = 0
        self.losing_signals = 0
        self.total_pnl = 0.0

__all__ = ['BaseStrategy', 'Signal']
