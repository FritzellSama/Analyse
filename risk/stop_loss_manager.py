"""
Stop Loss Manager - Quantum Trader Pro
Gestion avancée des stop loss avec trailing
"""

from typing import Dict, Optional
from dataclasses import dataclass
from threading import RLock
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
from utils.thread_sync import synchronized

@dataclass
class StopLossState:
    """État du stop loss"""
    initial_sl: float
    current_sl: float
    entry_price: float
    side: str  # 'long' or 'short'
    atr: float
    is_trailing: bool = False
    breakeven_triggered: bool = False
    highest_price: Optional[float] = None  # Pour long
    lowest_price: Optional[float] = None   # Pour short

class StopLossManager:
    """Gestionnaire de stop loss"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('StopLossManager')
        self._lock = RLock()  # Thread safety

        # Config stop loss - accès sécurisé
        sl_cfg = get_nested_config(config, 'risk', 'stop_loss', default={})
        self.method = sl_cfg.get('method', 'atr_based')
        self.atr_multiplier = sl_cfg.get('atr_multiplier', 1.5)
        self.fixed_percent = sl_cfg.get('fixed_percent', 2.0)

        # Config trailing - accès sécurisé
        trailing_cfg = get_nested_config(config, 'risk', 'stop_loss', 'trailing', default={})
        self.trailing_enabled = trailing_cfg.get('enabled', False)
        self.breakeven_trigger_atr = trailing_cfg.get('breakeven_atr_trigger', 1.5)
        self.trail_distance_atr = trailing_cfg.get('trail_distance_atr', 1.0)
        self.trail_step_atr = trailing_cfg.get('step_atr', 0.5)

        # Active stops
        self.active_stops = {}  # position_id → StopLossState
    
    @synchronized()
    def create_stop_loss(
        self,
        position_id: str,
        entry_price: float,
        side: str,
        atr: float,
        custom_sl: Optional[float] = None
    ) -> float:
        """
        Crée stop loss initial
        
        Args:
            position_id: ID position
            entry_price: Prix d'entrée
            side: 'long' ou 'short'
            atr: ATR actuel
            custom_sl: SL custom (optionnel)
        
        Returns:
            Prix du stop loss
        """
        if custom_sl is not None:
            initial_sl = custom_sl
        else:
            if self.method == 'atr_based':
                initial_sl = self._calculate_atr_sl(entry_price, side, atr)
            elif self.method == 'fixed_percent':
                initial_sl = self._calculate_fixed_sl(entry_price, side)
            else:
                # Default à ATR
                initial_sl = self._calculate_atr_sl(entry_price, side, atr)
        
        # Créer state
        normalized_side = 'long' if side.upper() in ['BUY', 'LONG'] else 'short'

        state = StopLossState(
            initial_sl=initial_sl,
            current_sl=initial_sl,
            entry_price=entry_price,
            side=normalized_side,
            atr=atr,
            highest_price=entry_price if normalized_side == 'long' else None,
            lowest_price=entry_price if normalized_side == 'short' else None
        )
        
        self.active_stops[position_id] = state
        
        self.logger.info(
            f"🛡️  Stop loss créé: {position_id} {side} @ {initial_sl:.2f} "
            f"(distance: {abs(entry_price - initial_sl):.2f})"
        )
        
        return initial_sl
    

    def _calculate_atr_sl(self, entry: float, side: str, atr: float) -> float:
        """Calcule SL basé sur ATR"""
        distance = atr * self.atr_multiplier

        # Convertir BUY/SELL en long/short
        normalized_side = 'long' if side.upper() in ['BUY', 'LONG'] else 'short'

        if normalized_side == 'long':
            return entry - distance
        else:
            return entry + distance

    def _calculate_fixed_sl(self, entry: float, side: str) -> float:
        """Calcule SL basé sur % fixe"""
        distance = entry * (self.fixed_percent / 100)

        # Convertir BUY/SELL en long/short
        normalized_side = 'long' if side.upper() in ['BUY', 'LONG'] else 'short'

        if normalized_side == 'long':
            return entry - distance
        else:
            return entry + distance
    
    @synchronized()
    def update(
        self,
        position_id: str,
        current_price: float,
        atr: Optional[float] = None
    ) -> Dict:
        """
        Met à jour stop loss (trailing)
        
        Args:
            position_id: ID position
            current_price: Prix actuel
            atr: ATR actuel (optionnel)
        
        Returns:
            Dict avec info: {
                'sl_updated': bool,
                'new_sl': float,
                'sl_hit': bool,
                'breakeven_triggered': bool
            }
        """
        if position_id not in self.active_stops:
            return {'sl_updated': False, 'sl_hit': False}
        
        state = self.active_stops[position_id]
        
        # Update ATR si fourni
        if atr is not None:
            state.atr = atr
        
        result = {
            'sl_updated': False,
            'new_sl': state.current_sl,
            'sl_hit': False,
            'breakeven_triggered': False
        }
        
        # Vérifier si SL hit
        if state.side == 'long' and current_price <= state.current_sl:
            result['sl_hit'] = True
            self.logger.warning(f"🛑 Stop Loss HIT: {position_id} @ {current_price:.2f}")
            return result
        
        elif state.side == 'short' and current_price >= state.current_sl:
            result['sl_hit'] = True
            self.logger.warning(f"🛑 Stop Loss HIT: {position_id} @ {current_price:.2f}")
            return result
        
        # Trailing logic
        if not self.trailing_enabled:
            return result
        
        if state.side == 'long':
            # Update highest
            if state.highest_price is None or current_price > state.highest_price:
                state.highest_price = current_price
            
            # Check breakeven trigger
            if not state.breakeven_triggered:
                profit = state.highest_price - state.entry_price
                if profit >= state.atr * self.breakeven_trigger_atr:
                    # Move SL to breakeven
                    state.current_sl = state.entry_price
                    state.breakeven_triggered = True
                    state.is_trailing = True
                    result['sl_updated'] = True
                    result['new_sl'] = state.current_sl
                    result['breakeven_triggered'] = True
                    self.logger.info(f"⚖️  Breakeven triggered: {position_id}")
            
            # Trailing stop
            if state.is_trailing:
                new_sl = state.highest_price - (state.atr * self.trail_distance_atr)
                
                # Update si nouveau SL > ancien SL
                if new_sl > state.current_sl:
                    state.current_sl = new_sl
                    result['sl_updated'] = True
                    result['new_sl'] = new_sl
                    self.logger.info(f"📈 Trailing SL updated: {position_id} → {new_sl:.2f}")
        
        else:  # short
            # Update lowest
            if state.lowest_price is None or current_price < state.lowest_price:
                state.lowest_price = current_price
            
            # Check breakeven
            if not state.breakeven_triggered:
                profit = state.entry_price - state.lowest_price
                if profit >= state.atr * self.breakeven_trigger_atr:
                    state.current_sl = state.entry_price
                    state.breakeven_triggered = True
                    state.is_trailing = True
                    result['sl_updated'] = True
                    result['new_sl'] = state.current_sl
                    result['breakeven_triggered'] = True
                    self.logger.info(f"⚖️  Breakeven triggered: {position_id}")
            
            # Trailing
            if state.is_trailing:
                new_sl = state.lowest_price + (state.atr * self.trail_distance_atr)
                
                if new_sl < state.current_sl:
                    state.current_sl = new_sl
                    result['sl_updated'] = True
                    result['new_sl'] = new_sl
                    self.logger.info(f"📉 Trailing SL updated: {position_id} → {new_sl:.2f}")
        
        return result
    
    @synchronized()
    def get_current_sl(self, position_id: str) -> Optional[float]:
        """Récupère SL actuel"""
        state = self.active_stops.get(position_id)
        return state.current_sl if state else None

    @synchronized()
    def remove_stop(self, position_id: str):
        """Supprime stop loss"""
        if position_id in self.active_stops:
            del self.active_stops[position_id]
            self.logger.debug(f"🗑️  Stop removed: {position_id}")

    @synchronized()
    def get_all_stops(self) -> Dict[str, float]:
        """Récupère tous les SL actifs"""
        return {
            pos_id: state.current_sl
            for pos_id, state in self.active_stops.items()
        }

__all__ = ['StopLossManager', 'StopLossState']
