"""
DCA Bot Strategy - Quantum Trader Pro
Dollar Cost Averaging avec détection de dips
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from strategies.base_strategy import BaseStrategy, Signal
from utils.logger import setup_logger
from utils.safety import ensure_minimum_data
from utils.safe_math import safe_divide

class DCABotStrategy(BaseStrategy):
    """Stratégie DCA (Dollar Cost Averaging)"""
    
    def __init__(self, config: Dict):
        super().__init__('DCABot', config)
        self.logger = setup_logger('DCABot')

        # Config DCA - avec fallbacks sécurisés
        dca_cfg = config.get('strategies', {}).get('dca_bot', {}).get('dca', {})
        self.interval_hours = dca_cfg.get('interval_hours', 1)
        self.amount_per_interval = dca_cfg.get('amount_per_interval', 100.0)
        self.drop_trigger_percent = dca_cfg.get('drop_trigger_percent', -10.0)
        self.accumulation_target = dca_cfg.get('accumulation_target', 1.0)
        # Stop loss de sécurité (protection contre crash majeur, très large pour DCA)
        self.emergency_stop_loss_percent = dca_cfg.get('emergency_stop_loss_percent', 20.0)

        # State
        self.last_buy_time = None
        self.total_accumulated = 0.0
        self.average_entry_price = 0.0
        self.total_invested = 0.0
        self.buy_history = []
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule indicateurs"""
        df = df.copy()
        
        close = df['close'].values
        
        import talib
        
        # SMA pour trend
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['sma_200'] = talib.SMA(close, timeperiod=200)
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # Drawdown from high
        df['highest_20'] = df['close'].rolling(20).max()
        df['drawdown'] = (df['close'] - df['highest_20']) / df['highest_20'] * 100
        
        return df
    
    def should_buy_time_based(self) -> bool:
        """Vérifier si c'est le moment d'acheter (temps)"""
        if self.last_buy_time is None:
            return True
        
        time_since_last = datetime.now() - self.last_buy_time
        return time_since_last >= timedelta(hours=self.interval_hours)
    
    def detect_dip(self, df: pd.DataFrame) -> Dict:
        """Détecte un dip significatif"""
        if not ensure_minimum_data(df, 50, "dca_detect_dip"):
            return {'is_dip': False, 'severity': 0, 'reasons': []}

        last = df.iloc[-1]
        
        reasons = []
        severity = 0  # 0-3
        
        # Dip 1: Drawdown significatif
        if last['drawdown'] <= -self.drop_trigger_percent:
            reasons.append(f"drawdown_{abs(last['drawdown']):.1f}%")
            severity += 1
        
        # Dip 2: RSI oversold
        if last['rsi'] < 30:
            reasons.append(f"rsi_oversold_{last['rsi']:.1f}")
            severity += 1
        
        # Dip 3: Prix sous SMA200
        if 'sma_200' in df.columns and not pd.isna(last['sma_200']):
            if last['close'] < last['sma_200'] * 0.95:  # 5% sous SMA200
                reasons.append("below_sma200")
                severity += 1
        
        return {
            'is_dip': severity > 0,
            'severity': severity,
            'reasons': reasons
        }
    
    def calculate_buy_amount(self, current_price: float, dip_severity: int) -> float:
        """Calcule montant à acheter (augmente si dip)"""
        base_amount = self.amount_per_interval
        
        # Multiplier par sévérité du dip
        multiplier = 1.0 + (dip_severity * 0.5)  # +50% par niveau

        amount_usd = base_amount * multiplier
        amount_asset = safe_divide(amount_usd, current_price, default=0.0)

        return amount_asset
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Génère signaux DCA"""
        signals = []
        
        # Check si target atteint
        if self.total_accumulated >= self.accumulation_target:
            self.logger.info(f"🎯 Target atteint: {self.total_accumulated:.4f}")
            return signals
        
        df = data.get(self.timeframes['trend'])  # Utiliser H1
        if df is None or not ensure_minimum_data(df, 50, "dca_generate_signals"):
            return signals

        # Calculer indicateurs
        df = self.calculate_indicators(df)

        last = df.iloc[-1]
        current_price = last['close']
        
        # Vérifier conditions d'achat
        should_buy = False
        buy_reasons = []
        dip_severity = 0
        
        # Condition 1: Temps écoulé
        if self.should_buy_time_based():
            should_buy = True
            buy_reasons.append('time_interval')
        
        # Condition 2: Dip détecté (prioritaire)
        dip_info = self.detect_dip(df)
        if dip_info['is_dip']:
            should_buy = True
            dip_severity = dip_info['severity']
            buy_reasons.extend(dip_info['reasons'])
            self.logger.info(f"📉 Dip détecté! Severity={dip_severity} Reasons={dip_info['reasons']}")
        
        if not should_buy:
            return signals
        
        # Calculer montant
        amount = self.calculate_buy_amount(current_price, dip_severity)

        # Calculer stop_loss de sécurité (très large pour DCA long-terme)
        # Protection contre crash majeur uniquement
        emergency_stop_loss = current_price * (1 - self.emergency_stop_loss_percent / 100)

        # Créer signal
        signal = Signal(
            timestamp=last.name if hasattr(last, 'name') else datetime.now(),
            action='BUY',
            symbol=self.symbol,
            confidence=0.8 + (dip_severity * 0.05),  # Plus confiant si dip
            entry_price=current_price,
            size=amount,
            stop_loss=emergency_stop_loss,  # SL large de sécurité (protection crash)
            take_profit=None,  # Hold long terme
            reason=buy_reasons,
            metadata={
                'dca_type': 'time_based' if 'time_interval' in buy_reasons else 'dip_based',
                'dip_severity': dip_severity,
                'amount_usd': amount * current_price,
                'total_accumulated': self.total_accumulated,
                'target': self.accumulation_target,
                'progress': safe_divide(self.total_accumulated, self.accumulation_target, default=0.0) * 100
            }
        )
        
        signals.append(signal)
        
        # Mettre à jour state
        self.last_buy_time = datetime.now()
        self.total_accumulated += amount
        self.total_invested += (amount * current_price)
        self.buy_history.append({
            'timestamp': datetime.now(),
            'price': current_price,
            'amount': amount,
            'amount_usd': amount * current_price,
            'reasons': buy_reasons
        })
        
        # Recalculer average entry
        if self.total_accumulated > 0:
            self.average_entry_price = safe_divide(self.total_invested, self.total_accumulated, default=self.average_entry_price)
        
        self.logger.info(
            f"💰 Signal DCA: {amount:.6f} @ ${current_price:.2f} "
            f"(${amount*current_price:.2f}) - "
            f"Total: {self.total_accumulated:.6f}/{self.accumulation_target:.6f} "
            f"Avg: ${self.average_entry_price:.2f}"
        )
        
        return signals
    
    def get_portfolio_stats(self) -> Dict:
        """Stats du portfolio DCA"""
        if self.total_accumulated == 0:
            return {}
        
        return {
            'total_accumulated': self.total_accumulated,
            'target': self.accumulation_target,
            'progress_percent': safe_divide(self.total_accumulated, self.accumulation_target, default=0.0) * 100,
            'total_invested_usd': self.total_invested,
            'average_entry_price': self.average_entry_price,
            'num_buys': len(self.buy_history),
            'last_buy': self.buy_history[-1] if self.buy_history else None
        }
    
    def reset(self):
        """Reset le bot DCA"""
        self.last_buy_time = None
        self.total_accumulated = 0.0
        self.average_entry_price = 0.0
        self.total_invested = 0.0
        self.buy_history = []
        self.logger.info("🔄 DCA Bot reset")

__all__ = ['DCABotStrategy']
