"""
Position Model - Quantum Trader Pro
Classe Position unifiée pour tout le système
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from utils.calculations import calculate_pnl
from utils.safe_math import safe_divide


class PositionSide(Enum):
    """Côté de la position"""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Statuts de position"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class Position:
    """
    Représente une position de trading unifiée.
    Utilisée par tous les modules : PositionManager, PortfolioManager, TradeExecutor, etc.
    """

    # Identifiants
    id: str
    symbol: str
    side: str  # 'long' ou 'short'

    # Prix et taille
    entry_price: float
    size: float
    current_price: float = 0.0
    initial_size: float = 0.0
    remaining_size: float = 0.0

    # Niveaux de risque
    stop_loss: Optional[float] = None
    take_profits: List[float] = field(default_factory=list)  # Multi-level TP

    # Métriques P&L
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    realized_pnl: float = 0.0

    # Timing
    entry_time: datetime = field(default_factory=lambda: datetime.now())
    close_time: Optional[datetime] = None
    duration_minutes: float = 0.0

    # Statut
    status: str = "open"

    # Ordres associés
    entry_order_id: str = ""
    exit_order_ids: List[str] = field(default_factory=list)

    # Stratégie
    strategy: str = ""

    # Trailing stop
    trailing_stop_active: bool = False
    highest_price: float = 0.0
    lowest_price: float = float('inf')

    # Métadonnées supplémentaires
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialisation après création avec validation complète"""
        # =====================================================================
        # VALIDATION DES CHAMPS OBLIGATOIRES
        # =====================================================================

        # Valider ID
        if not self.id or not isinstance(self.id, str) or len(self.id.strip()) == 0:
            raise ValueError(f"Invalid position ID: {self.id}. Must be non-empty string.")

        # Valider symbol
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError(f"Invalid symbol: {self.symbol}. Must be non-empty string.")
        if '/' not in self.symbol:
            raise ValueError(f"Invalid symbol format: {self.symbol}. Expected BASE/QUOTE format.")

        # Normaliser et valider side
        if isinstance(self.side, str):
            side_upper = self.side.upper()
            if side_upper in ['BUY', 'LONG']:
                self.side = 'long'
            elif side_upper in ['SELL', 'SHORT']:
                self.side = 'short'
            elif self.side not in ['long', 'short']:
                raise ValueError(f"Invalid side: {self.side}. Must be 'long', 'short', 'buy', or 'sell'.")
        else:
            raise ValueError(f"Side must be string, got: {type(self.side)}")

        # Valider entry_price
        if not isinstance(self.entry_price, (int, float)):
            raise ValueError(f"entry_price must be numeric, got: {type(self.entry_price)}")
        if self.entry_price <= 0:
            raise ValueError(f"entry_price must be positive: {self.entry_price}")

        # Valider size
        if not isinstance(self.size, (int, float)):
            raise ValueError(f"size must be numeric, got: {type(self.size)}")
        if self.size <= 0:
            raise ValueError(f"size must be positive: {self.size}")

        # =====================================================================
        # VALIDATION DES NIVEAUX DE RISQUE
        # =====================================================================

        # Valider stop_loss si défini
        if self.stop_loss is not None:
            if not isinstance(self.stop_loss, (int, float)):
                raise ValueError(f"stop_loss must be numeric, got: {type(self.stop_loss)}")
            if self.stop_loss <= 0:
                raise ValueError(f"stop_loss must be positive: {self.stop_loss}")

            # Vérifier cohérence avec side
            if self.side == 'long' and self.stop_loss >= self.entry_price:
                raise ValueError(
                    f"For LONG position, stop_loss ({self.stop_loss}) must be < entry_price ({self.entry_price})"
                )
            if self.side == 'short' and self.stop_loss <= self.entry_price:
                raise ValueError(
                    f"For SHORT position, stop_loss ({self.stop_loss}) must be > entry_price ({self.entry_price})"
                )

        # Valider take_profits si définis
        if self.take_profits:
            for i, tp in enumerate(self.take_profits):
                if not isinstance(tp, (int, float)):
                    raise ValueError(f"take_profit[{i}] must be numeric, got: {type(tp)}")
                if tp <= 0:
                    raise ValueError(f"take_profit[{i}] must be positive: {tp}")

                # Vérifier cohérence avec side
                if self.side == 'long' and tp <= self.entry_price:
                    raise ValueError(
                        f"For LONG position, take_profit[{i}] ({tp}) must be > entry_price ({self.entry_price})"
                    )
                if self.side == 'short' and tp >= self.entry_price:
                    raise ValueError(
                        f"For SHORT position, take_profit[{i}] ({tp}) must be < entry_price ({self.entry_price})"
                    )

        # =====================================================================
        # INITIALISATION DES VALEURS PAR DÉFAUT
        # =====================================================================

        # Initialiser remaining_size si non défini
        if self.remaining_size == 0.0:
            self.remaining_size = self.size

        # Initialiser initial_size si non défini
        if self.initial_size == 0.0:
            self.initial_size = self.size

        # Initialiser current_price si non défini
        if self.current_price == 0.0:
            self.current_price = self.entry_price

        # Initialiser highest/lowest pour trailing
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price

        if self.lowest_price == float('inf'):
            self.lowest_price = self.entry_price

        # Valider current_price est positif
        if self.current_price <= 0:
            self.current_price = self.entry_price

    def update_price(self, new_price: float) -> None:
        """
        Met à jour le prix courant et recalcule les métriques

        Args:
            new_price: Nouveau prix de marché
        """
        self.current_price = new_price

        # MAJ highest/lowest pour trailing stop
        if new_price > self.highest_price:
            self.highest_price = new_price
        if new_price < self.lowest_price:
            self.lowest_price = new_price

        # Calculer PnL
        self._recalculate_pnl()

        # Calculer durée
        self._update_duration()

    def _recalculate_pnl(self) -> None:
        """Recalcule le P&L basé sur le prix actuel"""
        if self.remaining_size > 0:
            pnl, pnl_pct = calculate_pnl(
                self.entry_price,
                self.current_price,
                self.remaining_size,
                self.side
            )
            self.unrealized_pnl = pnl
            self.unrealized_pnl_percent = pnl_pct

    def _update_duration(self) -> None:
        """Met à jour la durée de la position"""
        if self.status == "open":
            delta = datetime.now() - self.entry_time
            self.duration_minutes = delta.total_seconds() / 60.0
        elif self.close_time:
            delta = self.close_time - self.entry_time
            self.duration_minutes = delta.total_seconds() / 60.0

    def partial_close(self, close_size: float, close_price: float) -> float:
        """
        Ferme partiellement la position

        Args:
            close_size: Taille à fermer
            close_price: Prix de fermeture

        Returns:
            P&L réalisé sur cette portion
        """
        if close_size > self.remaining_size:
            close_size = self.remaining_size

        # Calculer P&L réalisé
        pnl, _ = calculate_pnl(
            self.entry_price,
            close_price,
            close_size,
            self.side
        )

        # Mettre à jour
        self.remaining_size -= close_size
        self.realized_pnl += pnl

        # Mettre à jour statut
        if self.remaining_size <= 0:
            self.status = "closed"
            self.close_time = datetime.now()
            self.remaining_size = 0.0
        else:
            self.status = "partial"

        # Recalculer unrealized PnL
        self._recalculate_pnl()
        self._update_duration()

        return pnl

    def close(self, close_price: float) -> float:
        """
        Ferme complètement la position

        Args:
            close_price: Prix de fermeture

        Returns:
            P&L total réalisé
        """
        return self.partial_close(self.remaining_size, close_price)

    def is_stop_loss_hit(self) -> bool:
        """
        Vérifie si le stop loss est touché

        Returns:
            True si SL touché
        """
        if self.stop_loss is None:
            return False

        if self.side == 'long':
            return self.current_price <= self.stop_loss
        else:  # short
            return self.current_price >= self.stop_loss

    def check_take_profits(self) -> List[int]:
        """
        Vérifie quels niveaux de take profit sont atteints

        Returns:
            Liste des indices des TP atteints
        """
        hit_indices = []

        for i, tp_price in enumerate(self.take_profits):
            if self.side == 'long' and self.current_price >= tp_price:
                hit_indices.append(i)
            elif self.side == 'short' and self.current_price <= tp_price:
                hit_indices.append(i)

        return hit_indices

    def get_risk_distance(self) -> float:
        """
        Calcule la distance au stop loss en %

        Returns:
            Distance en pourcentage
        """
        if self.stop_loss is None:
            return 0.0

        distance = safe_divide(abs(self.entry_price - self.stop_loss), self.entry_price, default=0.0) * 100
        return distance

    def get_reward_distance(self, tp_index: int = 0) -> float:
        """
        Calcule la distance au take profit en %

        Args:
            tp_index: Index du TP (si multi-level)

        Returns:
            Distance en pourcentage
        """
        if not self.take_profits or tp_index >= len(self.take_profits):
            return 0.0

        tp_price = self.take_profits[tp_index]
        distance = safe_divide(abs(tp_price - self.entry_price), self.entry_price, default=0.0) * 100
        return distance

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la position en dictionnaire

        Returns:
            Dict représentant la position
        """
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'size': self.size,
            'initial_size': self.initial_size,
            'remaining_size': self.remaining_size,
            'stop_loss': self.stop_loss,
            'take_profits': self.take_profits,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_percent': self.unrealized_pnl_percent,
            'realized_pnl': self.realized_pnl,
            'entry_time': self.entry_time.isoformat(),
            'close_time': self.close_time.isoformat() if self.close_time else None,
            'duration_minutes': self.duration_minutes,
            'status': self.status,
            'strategy': self.strategy,
            'trailing_stop_active': self.trailing_stop_active,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """
        Crée une Position depuis un dictionnaire

        Args:
            data: Dictionnaire source

        Returns:
            Instance Position
        """
        # Parser les dates
        entry_time = data.get('entry_time', datetime.now())
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)

        close_time = data.get('close_time')
        if isinstance(close_time, str):
            close_time = datetime.fromisoformat(close_time)

        return cls(
            id=data['id'],
            symbol=data['symbol'],
            side=data['side'],
            entry_price=data['entry_price'],
            size=data.get('size', 0),
            current_price=data.get('current_price', data['entry_price']),
            initial_size=data.get('initial_size', data.get('size', 0)),
            remaining_size=data.get('remaining_size', data.get('size', 0)),
            stop_loss=data.get('stop_loss'),
            take_profits=data.get('take_profits', []),
            unrealized_pnl=data.get('unrealized_pnl', 0.0),
            unrealized_pnl_percent=data.get('unrealized_pnl_percent', 0.0),
            realized_pnl=data.get('realized_pnl', 0.0),
            entry_time=entry_time,
            close_time=close_time,
            duration_minutes=data.get('duration_minutes', 0.0),
            status=data.get('status', 'open'),
            entry_order_id=data.get('entry_order_id', ''),
            exit_order_ids=data.get('exit_order_ids', []),
            strategy=data.get('strategy', ''),
            trailing_stop_active=data.get('trailing_stop_active', False),
            highest_price=data.get('highest_price', data['entry_price']),
            lowest_price=data.get('lowest_price', data['entry_price']),
            metadata=data.get('metadata', {})
        )

    def __str__(self) -> str:
        """Représentation string"""
        return (
            f"Position({self.id}): {self.side.upper()} {self.remaining_size:.6f} {self.symbol} "
            f"@ {self.entry_price:.2f} | PnL: ${self.unrealized_pnl:.2f} ({self.unrealized_pnl_percent:.2f}%)"
        )


@dataclass
class ClosedTrade:
    """Trade fermé pour historique"""
    position_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    strategy: str
    exit_reason: str  # 'tp', 'sl', 'manual', 'signal'
    duration_minutes: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_position(cls, position: Position, exit_price: float, exit_reason: str) -> 'ClosedTrade':
        """
        Crée un ClosedTrade depuis une Position

        Args:
            position: Position à fermer
            exit_price: Prix de sortie
            exit_reason: Raison de sortie

        Returns:
            ClosedTrade
        """
        pnl, pnl_pct = calculate_pnl(
            position.entry_price,
            exit_price,
            position.initial_size,
            position.side
        )

        return cls(
            position_id=position.id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.initial_size,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_percent=pnl_pct,
            strategy=position.strategy,
            exit_reason=exit_reason,
            duration_minutes=position.duration_minutes,
            metadata=position.metadata
        )


__all__ = ['Position', 'PositionSide', 'PositionStatus', 'ClosedTrade']
