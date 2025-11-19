"""
Execution Module - Quantum Trader Pro
Gestion complète de l'exécution des trades et des positions
"""

from execution.order_executor import OrderExecutor, OrderType, OrderSide, OrderStatus
from execution.position_manager import PositionManager
from execution.trade_executor import TradeExecutor
# Re-export Position from unified models module for backward compatibility
from models.position import Position, PositionSide, PositionStatus, ClosedTrade

__all__ = [
    'OrderExecutor',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'PositionManager',
    'Position',
    'PositionSide',
    'PositionStatus',
    'ClosedTrade',
    'TradeExecutor'
]
