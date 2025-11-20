"""
Take Profit Manager - Quantum Trader Pro
Gestion multi-niveaux des take profits
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from threading import RLock
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
from utils.safe_math import safe_divide

@dataclass
class TakeProfitLevel:
    """Niveau de take profit"""
    price: float
    size_percent: float  # % de position à closer
    filled: bool = False

@dataclass
class TakeProfitState:
    """État du take profit"""
    position_id: str
    entry_price: float
    side: str
    initial_size: float
    remaining_size: float
    levels: List[TakeProfitLevel]
    filled_count: int = 0

class TakeProfitManager:
    """Gestionnaire de take profit multi-niveaux"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('TakeProfitManager')
        self._lock = RLock()  # Thread safety

        # Config TP - accès sécurisé
        tp_cfg = get_nested_config(config, 'risk', 'take_profit', default={})
        self.method = tp_cfg.get('method', 'multi_level')

        # Multi-level config
        if 'levels' in tp_cfg:
            self.tp_levels_config = tp_cfg['levels']
        else:
            # Default 3 niveaux
            self.tp_levels_config = [
                {'percent': 50, 'multiplier': 1.5},
                {'percent': 30, 'multiplier': 2.5},
                {'percent': 20, 'multiplier': 4.0},
            ]

        # Active TPs
        self.active_tps = {}  # position_id → TakeProfitState

    def create_take_profits(
        self,
        position_id: str,
        entry_price: float,
        side: str,
        size: float,
        stop_loss: float,
        atr: Optional[float] = None,
        custom_tps: Optional[List[Tuple[float, float]]] = None
    ) -> List[TakeProfitLevel]:
        """
        Crée niveaux de take profit

        Args:
            position_id: ID position
            entry_price: Prix d'entrée
            side: 'long' ou 'short'
            size: Taille position
            stop_loss: Prix SL (pour calculer risk/reward)
            atr: ATR (optionnel)
            custom_tps: TPs custom [(price, size_pct), ...]

        Returns:
            Liste de TakeProfitLevel
        """
        levels = []

        if custom_tps:
            # Utiliser TPs custom
            for price, size_pct in custom_tps:
                levels.append(TakeProfitLevel(price=price, size_percent=size_pct))

        else:
            # Générer TPs selon config
            risk = abs(entry_price - stop_loss)

            # Protection si risk = 0 (SL = Entry)
            if risk < 0.00000001:
                self.logger.warning(
                    f"Risk proche de zéro (SL={stop_loss}, Entry={entry_price}). "
                    f"Utilisation de 2% de l'entry comme risk par défaut."
                )
                risk = entry_price * 0.02  # 2% comme fallback

            for level_cfg in self.tp_levels_config:
                multiplier = level_cfg['multiplier']
                size_pct = level_cfg['percent'] / 100.0

                # Normaliser side
                normalized_side = 'long' if side.upper() in ['BUY', 'LONG'] else 'short'

                if normalized_side == 'long':
                    tp_price = entry_price + (risk * multiplier)
                else:
                    tp_price = entry_price - (risk * multiplier)

                levels.append(
                    TakeProfitLevel(price=tp_price, size_percent=size_pct)
                )

        # Créer state
        normalized_side = 'long' if side.upper() in ['BUY', 'LONG'] else 'short'

        state = TakeProfitState(
            position_id=position_id,
            entry_price=entry_price,
            side=normalized_side,
            initial_size=size,
            remaining_size=size,
            levels=levels
        )

        self.active_tps[position_id] = state

        self.logger.info(
            f"🎯 Take profits créés: {position_id} - {len(levels)} niveaux"
        )
        for i, level in enumerate(levels, 1):
            reward = abs(level.price - entry_price)
            rr_ratio = safe_divide(reward, risk, default=0.0)
            self.logger.info(
                f"   TP{i}: {level.price:.8f} ({level.size_percent:.0%} position) "
                f"R/R={rr_ratio:.2f}"
            )

        return levels

    def check_take_profits(
        self,
        position_id: str,
        current_price: float
    ) -> List[Dict]:
        """
        Vérifie si des TPs sont hit

        Args:
            position_id: ID position
            current_price: Prix actuel

        Returns:
            Liste de TPs hit: [{'level': int, 'price': float, 'size': float}, ...]
        """
        if position_id not in self.active_tps:
            return []

        state = self.active_tps[position_id]
        hits = []

        for i, level in enumerate(state.levels):
            if level.filled:
                continue

            # Check si hit
            hit = False
            if state.side == 'long' and current_price >= level.price:
                hit = True
            elif state.side == 'short' and current_price <= level.price:
                hit = True

            if hit:
                # Calculer taille à closer
                size_to_close = state.initial_size * level.size_percent

                # Marquer comme filled
                level.filled = True
                state.filled_count += 1
                state.remaining_size -= size_to_close

                hits.append({
                    'level': i + 1,
                    'price': level.price,
                    'size': size_to_close,
                    'size_percent': level.size_percent,
                    'remaining_size': state.remaining_size
                })

                # Calculer profit
                if state.side == 'long':
                    profit = (level.price - state.entry_price) * size_to_close
                else:
                    profit = (state.entry_price - level.price) * size_to_close

                # Protection division par zéro
                if state.entry_price > 0.00000001:
                    profit_pct = ((level.price - state.entry_price) / state.entry_price * 100
                                 if state.side == 'long' else
                                 (state.entry_price - level.price) / state.entry_price * 100)
                else:
                    profit_pct = 0.0

                self.logger.info(
                    f"🎯 TP{i+1} HIT: {position_id} @ {level.price:.2f} - "
                    f"Closed {size_to_close:.4f} ({level.size_percent:.0%}) - "
                    f"Profit: ${profit:.2f} ({profit_pct:+.2f}%)"
                )

        # Si tous les TPs hit, supprimer
        if state.filled_count == len(state.levels):
            self.logger.info(f"✅ Tous les TPs hit pour {position_id}")
            del self.active_tps[position_id]

        return hits

    def get_next_tp(self, position_id: str) -> Optional[float]:
        """Récupère le prochain TP non hit"""
        state = self.active_tps.get(position_id)
        if not state:
            return None

        for level in state.levels:
            if not level.filled:
                return level.price

        return None

    def get_all_tps(self, position_id: str) -> List[float]:
        """Récupère tous les TPs (hit ou non)"""
        state = self.active_tps.get(position_id)
        if not state:
            return []

        return [level.price for level in state.levels]

    def get_remaining_size(self, position_id: str) -> Optional[float]:
        """Récupère la taille restante"""
        state = self.active_tps.get(position_id)
        return state.remaining_size if state else None

    def remove_take_profits(self, position_id: str):
        """Supprime TPs"""
        if position_id in self.active_tps:
            del self.active_tps[position_id]
            self.logger.debug(f"🗑️  TPs removed: {position_id}")

    def transfer_take_profits(self, old_id: str, new_id: str):
        """Transfère les take profits d'un ID vers un autre"""
        if old_id in self.active_tps:
            self.active_tps[new_id] = self.active_tps[old_id]
            del self.active_tps[old_id]
            self.logger.debug(f"🔄 TPs transférés: {old_id} → {new_id}")

    def get_stats(self, position_id: str) -> Optional[Dict]:
        """Stats des TPs"""
        state = self.active_tps.get(position_id)
        if not state:
            return None

        return {
            'position_id': position_id,
            'total_levels': len(state.levels),
            'filled_levels': state.filled_count,
            'remaining_levels': len(state.levels) - state.filled_count,
            'initial_size': state.initial_size,
            'remaining_size': state.remaining_size,
            'closed_percent': safe_divide(
                state.initial_size - state.remaining_size,
                state.initial_size,
                default=0.0
            ) * 100
        }

__all__ = ['TakeProfitManager', 'TakeProfitLevel', 'TakeProfitState']
