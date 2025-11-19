"""
Portfolio Manager - Quantum Trader Pro
Gestion du portfolio et des positions
"""

from typing import Dict, List, Optional
from datetime import datetime
from threading import RLock
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
from utils.safe_math import safe_divide
from utils.thread_sync import synchronized
from models.position import Position, ClosedTrade
import pandas as pd


class PortfolioManager:
    """Gestionnaire de portfolio"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('PortfolioManager')
        self._lock = RLock()  # Thread safety

        # Config - accès sécurisé
        risk_cfg = get_nested_config(config, 'risk', default={})
        self.max_positions = risk_cfg.get('max_positions_simultaneous', risk_cfg.get('max_positions_simultaneous', 3))
        self.max_positions_same_dir = risk_cfg.get('max_positions_same_direction', 2)
        self.max_correlation = risk_cfg.get('max_correlation', 0.7)

        # Portfolio state
        self.open_positions = {}  # position_id → Position
        self.closed_trades = []   # List[ClosedTrade]

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
    
    @synchronized()
    def can_open_position(
        self,
        symbol: str,
        side: str,
        current_balance: float
    ) -> tuple[bool, Optional[str]]:
        """
        Vérifie si peut ouvrir position
        
        Args:
            symbol: Symbol
            side: 'long' ou 'short'
            current_balance: Balance actuelle
        
        Returns:
            (can_open, reason_if_not)
        """
        # Check 1: Max positions
        if len(self.open_positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) atteint"
        
        # Check 2: Max positions même direction
        same_dir_count = sum(
            1 for p in self.open_positions.values() 
            if p.side.lower() == side.lower()
        )
        if same_dir_count >= self.max_positions_same_dir:
            return False, f"Max positions {side} ({self.max_positions_same_dir}) atteint"
        
        # Check 3: Position déjà ouverte sur ce symbol
        for pos in self.open_positions.values():
            if pos.symbol == symbol:
                return False, f"Position déjà ouverte sur {symbol}"
        
        return True, None
    
    @synchronized()
    def open_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profits: List[float],
        strategy: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Ouvre une position

        Returns:
            True si succès
        """
        # Vérifier si possible
        can_open, reason = self.can_open_position(symbol, side, 0)  # Balance check ailleurs

        if not can_open:
            self.logger.warning(f"⚠️  Cannot open position: {reason}")
            return False

        # Créer position (utilise la Position unifiée de models.position)
        position = Position(
            id=position_id,
            symbol=symbol,
            side=side.lower(),
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profits=take_profits,
            strategy=strategy,
            metadata=metadata or {}
        )

        self.open_positions[position_id] = position

        self.logger.info(
            f"🟢 Position opened: {position_id} | {side} {size:.4f} {symbol} @ ${entry_price:.2f} | "
            f"SL: ${stop_loss:.2f} | Strategy: {strategy}"
        )

        return True
    
    @synchronized()
    def update_position(
        self,
        position_id: str,
        current_price: float,
        stop_loss: Optional[float] = None
    ):
        """Met à jour une position"""
        if position_id not in self.open_positions:
            return

        position = self.open_positions[position_id]

        # Update SL si fourni
        if stop_loss is not None:
            position.stop_loss = stop_loss

        # Utilise la méthode update_price de Position unifiée
        position.update_price(current_price)
    
    @synchronized()
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str = 'manual',
        size: Optional[float] = None
    ) -> Optional[ClosedTrade]:
        """
        Ferme position (totale ou partielle)

        Args:
            position_id: ID position
            exit_price: Prix de sortie
            exit_reason: Raison ('tp', 'sl', 'manual', 'signal')
            size: Taille à fermer (None = tout)

        Returns:
            ClosedTrade si fermé, None sinon
        """
        if position_id not in self.open_positions:
            self.logger.warning(f"⚠️  Position {position_id} not found")
            return None

        position = self.open_positions[position_id]

        # Taille à fermer
        close_size = size if size is not None else position.remaining_size
        close_size = min(close_size, position.remaining_size)

        # Utiliser partial_close de Position unifiée pour calculer PnL
        pnl = position.partial_close(close_size, exit_price)
        pnl_pct = safe_divide(pnl, position.entry_price * close_size, default=0.0) * 100

        # Créer trade (utilise ClosedTrade unifié)
        trade = ClosedTrade(
            position_id=position_id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=close_size,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_percent=pnl_pct,
            strategy=position.strategy,
            exit_reason=exit_reason,
            duration_minutes=position.duration_minutes,
            metadata=position.metadata
        )

        self.closed_trades.append(trade)

        # Update stats
        self.total_trades += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Si complètement fermé, supprimer
        if position.status == 'closed':
            del self.open_positions[position_id]
            emoji = "💰" if pnl > 0 else "💸"
            self.logger.info(
                f"{emoji} Position CLOSED: {position_id} | "
                f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) | "
                f"Duration: {position.duration_minutes:.1f}min | Reason: {exit_reason}"
            )
        else:
            self.logger.info(
                f"📊 Position PARTIAL CLOSE: {position_id} | "
                f"Closed: {close_size:.4f} | Remaining: {position.remaining_size:.4f} | "
                f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)"
            )

        return trade
    
    @synchronized()
    def get_position(self, position_id: str) -> Optional[Position]:
        """Récupère position"""
        return self.open_positions.get(position_id)

    @synchronized()
    def get_all_positions(self) -> List[Position]:
        """Récupère toutes les positions"""
        return list(self.open_positions.values())

    @synchronized()
    def get_open_positions_count(self) -> int:
        """Nombre de positions ouvertes"""
        return len(self.open_positions)

    @synchronized()
    def get_total_exposure(self, current_prices: Dict[str, float]) -> float:
        """Calcule exposition totale"""
        total = 0.0

        for pos in self.open_positions.values():
            price = current_prices.get(pos.symbol, pos.current_price)
            total += price * pos.remaining_size

        return total

    @synchronized()
    def get_unrealized_pnl(self) -> float:
        """PnL non réalisé total"""
        return sum(pos.unrealized_pnl for pos in self.open_positions.values())
    
    def calculate_stats(self, current_balance: float) -> Dict:
        """Calcule stats de performance"""
        # Update peak et drawdown
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        self.current_drawdown = safe_divide(
            self.peak_balance - current_balance,
            self.peak_balance,
            default=0.0
        ) * 100

        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown

        # Win rate
        win_rate = safe_divide(self.winning_trades, self.total_trades, default=0.0) * 100

        # Avg win/loss
        wins = [t.pnl for t in self.closed_trades if t.pnl > 0]
        losses = [t.pnl for t in self.closed_trades if t.pnl < 0]

        avg_win = safe_divide(sum(wins), len(wins), default=0.0) if wins else 0
        avg_loss = abs(safe_divide(sum(losses), len(losses), default=0.0)) if losses else 0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = safe_divide(total_wins, total_losses, default=0.0)

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'open_positions': len(self.open_positions),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'peak_balance': self.peak_balance,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
        }
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Retourne DataFrame des trades"""
        if not self.closed_trades:
            return pd.DataFrame()

        data = []
        for trade in self.closed_trades:
            data.append({
                'position_id': trade.position_id,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_percent,
                'strategy': trade.strategy,
                'exit_reason': trade.exit_reason,
                'duration_minutes': trade.duration_minutes,
            })

        return pd.DataFrame(data)


__all__ = ['PortfolioManager']
