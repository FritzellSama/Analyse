"""
Position Manager - Quantum Trader Pro
Gestion complète des positions ouvertes avec tracking et monitoring
Thread-safe implementation with RLock protection
"""

from typing import Dict, List, Optional
from datetime import datetime
from threading import RLock
from utils.logger import setup_trading_logger
from models.position import Position, PositionSide, PositionStatus
from utils.thread_sync import synchronized
from utils.safe_math import safe_divide


class PositionManager:
    """
    Gestionnaire de positions avec:
    - Tracking positions ouvertes
    - Calcul PnL en temps réel
    - Monitoring stop-loss/take-profit
    - Statistiques et reporting
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le gestionnaire de positions
        
        Args:
            config: Configuration complète du bot
        """
        self.config = config
        self.logger = setup_trading_logger('PositionManager')

        # Thread safety lock
        self._lock = RLock()

        # Positions (protected by _lock)
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

        # Limites
        risk_config = config.get('risk', {})
        self.max_positions = risk_config.get('max_positions_simultaneous', 3)
        self.max_same_direction = risk_config.get('max_positions_same_direction', 2)

        # Statistiques (protected by _lock)
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

        self.logger.info(f"✅ Position Manager initialisé (thread-safe)")
        self.logger.info(f"📊 Max positions simultanées: {self.max_positions}")
        self.logger.info(f"📊 Max positions même direction: {self.max_same_direction}")
    
    @synchronized()
    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profits: Optional[List[float]] = None,
        strategy: str = "",
        order_id: str = ""
    ) -> Optional[Position]:
        """
        Ouvre une nouvelle position (THREAD-SAFE)

        Args:
            symbol: Paire de trading
            side: 'long' ou 'short'
            entry_price: Prix d'entrée
            size: Taille de la position
            stop_loss: Niveau de stop loss
            take_profits: Liste des niveaux de TP (prix)
            strategy: Nom de la stratégie
            order_id: ID de l'ordre d'entrée

        Returns:
            Position créée ou None si impossible
        """

        # Vérifier limites (called within lock context)
        if not self._can_open_position_unsafe(side):
            self.logger.warning(
                f"⚠️ Impossible d'ouvrir position: limites atteintes"
            )
            return None

        # Créer position
        position_id = f"{symbol}_{side}_{int(datetime.now().timestamp() * 1000)}"

        try:
            position = Position(
                id=position_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                size=size,
                stop_loss=stop_loss,
                take_profits=take_profits if take_profits else [],
                strategy=strategy,
                entry_order_id=order_id
            )
        except ValueError as e:
            self.logger.error(f"❌ Invalid position parameters: {e}")
            return None

        # Ajouter aux positions ouvertes
        self.open_positions[position_id] = position

        self.logger.trade_opened(
            symbol=symbol,
            side=side,
            size=size,
            price=entry_price,
            order_id=order_id
        )

        return position
    
    @synchronized()
    def close_position(
        self,
        position_id: str,
        close_price: float,
        reason: str = "manual",
        order_id: str = ""
    ) -> Optional[Position]:
        """
        Ferme une position (THREAD-SAFE)

        Args:
            position_id: ID de la position
            close_price: Prix de sortie
            reason: Raison de fermeture
            order_id: ID de l'ordre de sortie

        Returns:
            Position fermée ou None
        """

        if position_id not in self.open_positions:
            self.logger.warning(f"⚠️ Position {position_id} introuvable")
            return None

        position = self.open_positions[position_id]

        # Fermer position
        position.close(close_price)
        position.exit_order_ids.append(order_id)

        # Calculer PnL brut
        pnl_brut = getattr(position, 'pnl', position.realized_pnl)

        # Déduire les frais de trading (0.1% à l'entrée + 0.1% à la sortie = 0.2% total)
        entry_fees = position.entry_price * position.size * 0.001  # 0.1% sur l'achat
        exit_fees = close_price * position.size * 0.001             # 0.1% sur la vente
        total_fees = entry_fees + exit_fees

        # PnL net après frais
        pnl = pnl_brut - total_fees

        # Déplacer vers closed
        del self.open_positions[position_id]
        self.closed_positions.append(position)

        # MAJ statistiques avec PnL net
        self.total_trades += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Log
        pnl_percent = safe_divide(pnl, position.entry_price * position.size, default=0.0) * 100
        if pnl > 0:
            self.logger.trade_closed(
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                exit_price=close_price,
                pnl=pnl,
                pnl_percent=pnl_percent
            )
            self.logger.info(f"💰 Frais déduits: ${total_fees:.2f}")
        else:
            self.logger.error(
                f"📉 Position fermée à perte: {position.symbol} "
                f"{position.side.upper()} | "
                f"PnL: ${pnl:.2f} ({pnl_percent:.2f}%) [Frais: ${total_fees:.2f}]"
            )

        # Limiter historique (garder 500 dernières)
        if len(self.closed_positions) > 500:
            self.closed_positions = self.closed_positions[-500:]

        return position
    
    @synchronized()
    def partial_close_position(
        self,
        position_id: str,
        size_to_close: float,
        close_price: float,
        reason: str = "take_profit",
        order_id: str = ""
    ) -> Optional[Position]:
        """
        Ferme partiellement une position (THREAD-SAFE)

        Args:
            position_id: ID de la position
            size_to_close: Taille à fermer
            close_price: Prix de sortie
            reason: Raison (ex: 'take_profit_1')
            order_id: ID de l'ordre

        Returns:
            Position modifiée ou None
        """

        if position_id not in self.open_positions:
            return None

        position = self.open_positions[position_id]

        # Fermeture partielle - PnL brut
        pnl_brut = position.partial_close(size_to_close, close_price)

        # Déduire les frais pour cette partie fermée
        entry_fees = position.entry_price * size_to_close * 0.001
        exit_fees = close_price * size_to_close * 0.001
        total_fees = entry_fees + exit_fees

        # PnL net pour cette fermeture partielle
        pnl_partial = pnl_brut - total_fees

        position.exit_order_ids.append(order_id)

        self.logger.info(
            f"📊 Position partiellement fermée: {position.symbol} "
            f"({size_to_close}/{position.initial_size}) | "
            f"Raison: {reason} | "
            f"PnL partiel: ${pnl_partial:.2f} [Frais: ${total_fees:.2f}]"
        )

        # Si complètement fermée, déplacer vers closed
        if position.status == 'closed':
            del self.open_positions[position_id]
            self.closed_positions.append(position)
            self.total_trades += 1

            # Calculer PnL total net (realized_pnl contient déjà toutes les fermetures partielles)
            pnl_total = getattr(position, 'pnl', position.realized_pnl)
            self.total_pnl += pnl_total

            if pnl_total > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

        return position
    
    @synchronized()
    def update_position_price(self, position_id: str, new_price: float):
        """
        Met à jour le prix d'une position (THREAD-SAFE)

        Args:
            position_id: ID de la position
            new_price: Nouveau prix
        """
        if position_id in self.open_positions:
            self.open_positions[position_id].update_price(new_price)

    def update_all_positions(self, prices: Dict[str, float]):
        """
        Met à jour toutes les positions avec nouveaux prix (THREAD-SAFE)

        Args:
            prices: Dict {symbol: price}
        """
        # Update all positions under single lock to avoid race conditions
        with self._lock:
            for pos_id, pos in list(self.open_positions.items()):
                if pos.symbol in prices:
                    pos.update_price(prices[pos.symbol])

    @synchronized()
    def get_position(self, position_id: str) -> Optional[Position]:
        """Récupère une position par ID (THREAD-SAFE)"""
        return self.open_positions.get(position_id)

    @synchronized()
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Récupère toutes les positions pour un symbole (THREAD-SAFE)"""
        return [
            pos for pos in self.open_positions.values()
            if pos.symbol == symbol
        ]

    @synchronized()
    def get_all_open_positions(self) -> List[Position]:
        """Retourne toutes les positions ouvertes (THREAD-SAFE)"""
        return list(self.open_positions.values())

    @synchronized()
    def get_positions_count(self) -> Dict[str, int]:
        """Compte les positions par direction (THREAD-SAFE)"""
        long_count = sum(1 for p in self.open_positions.values() if p.side == 'long')
        short_count = sum(1 for p in self.open_positions.values() if p.side == 'short')

        return {
            'total': len(self.open_positions),
            'long': long_count,
            'short': short_count
        }

    def _can_open_position_unsafe(self, side: str) -> bool:
        """
        Vérifie si on peut ouvrir une nouvelle position (NON THREAD-SAFE)
        Doit être appelée depuis un contexte déjà verrouillé.
        """
        long_count = sum(1 for p in self.open_positions.values() if p.side == 'long')
        short_count = sum(1 for p in self.open_positions.values() if p.side == 'short')
        total = len(self.open_positions)

        # Vérifier limite totale
        if total >= self.max_positions:
            return False

        # Vérifier limite par direction
        if side == 'long' and long_count >= self.max_same_direction:
            return False
        if side == 'short' and short_count >= self.max_same_direction:
            return False

        return True

    def _can_open_position(self, side: str) -> bool:
        """Vérifie si on peut ouvrir une nouvelle position (THREAD-SAFE)"""
        with self._lock:
            return self._can_open_position_unsafe(side)
    
    def get_total_exposure(self) -> Dict[str, float]:
        """Calcule l'exposition totale"""
        
        total_long = sum(
            p.size * p.current_price
            for p in self.open_positions.values()
            if p.side == 'long'
        )
        
        total_short = sum(
            p.size * p.current_price
            for p in self.open_positions.values()
            if p.side == 'short'
        )
        
        return {
            'long': total_long,
            'short': total_short,
            'net': total_long - total_short,
            'gross': total_long + total_short
        }
    
    def get_unrealized_pnl(self) -> float:
        """Calcule le PnL non réalisé total"""
        return sum(
            p.unrealized_pnl
            for p in self.open_positions.values()
        )
    
    def get_realized_pnl(self) -> float:
        """Calcule le PnL réalisé total (incluant positions fermées)"""
        return sum(
            p.realized_pnl
            for p in self.closed_positions
        )
    
    def get_statistics(self) -> Dict:
        """
        Retourne les statistiques complètes
        
        Returns:
            Dict avec toutes les stats
        """
        
        # Win rate
        win_rate = safe_divide(self.winning_trades, self.total_trades, default=0.0) * 100

        # Moyennes
        avg_win = 0
        avg_loss = 0

        if self.winning_trades > 0:
            winning_pnls = [p.pnl for p in self.closed_positions if p.pnl > 0]
            avg_win = safe_divide(sum(winning_pnls), len(winning_pnls), default=0.0) if winning_pnls else 0

        if self.losing_trades > 0:
            losing_pnls = [abs(p.pnl) for p in self.closed_positions if p.pnl < 0]
            avg_loss = safe_divide(sum(losing_pnls), len(losing_pnls), default=0.0) if losing_pnls else 0

        # Profit factor
        profit_factor = 0
        total_wins = sum(p.pnl for p in self.closed_positions if p.pnl > 0)
        total_losses = abs(sum(p.pnl for p in self.closed_positions if p.pnl < 0))

        if total_losses > 0:
            profit_factor = safe_divide(total_wins, total_losses, default=0.0)

        # Exposure
        exposure = self.get_total_exposure()

        # Durée moyenne
        avg_duration = 0
        if self.closed_positions:
            avg_duration = safe_divide(sum(p.duration_minutes for p in self.closed_positions), len(self.closed_positions), default=0.0)
        
        return {
            'open_positions': len(self.open_positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': self.get_unrealized_pnl(),
            'realized_pnl': self.get_realized_pnl(),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_duration_minutes': avg_duration,
            'exposure_long': exposure['long'],
            'exposure_short': exposure['short'],
            'exposure_net': exposure['net'],
            'exposure_gross': exposure['gross']
        }
    
    def get_positions_summary(self) -> List[Dict]:
        """Retourne un résumé des positions ouvertes"""
        return [pos.to_dict() for pos in self.open_positions.values()]
    
    def close_all_positions(self, close_price_map: Dict[str, float], reason: str = "emergency"):
        """
        Ferme toutes les positions (urgence)
        
        Args:
            close_price_map: Dict {symbol: price}
            reason: Raison de fermeture
        """
        self.logger.warning(f"⚠️ Fermeture de toutes les positions: {reason}")
        
        position_ids = list(self.open_positions.keys())
        
        for position_id in position_ids:
            position = self.open_positions[position_id]
            close_price = close_price_map.get(position.symbol)
            
            if close_price:
                self.close_position(position_id, close_price, reason)
