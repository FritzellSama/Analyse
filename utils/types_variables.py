"""
Type Definitions - Quantum Trader Pro
Centralized type hints and type aliases for better code clarity
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Callable,
    TypeVar,
    Protocol,
    Literal
)
from datetime import datetime
import pandas as pd

# ============================================================================
# BASIC TYPE ALIASES
# ============================================================================

# Price and monetary values
Price = float
Size = float
Volume = float
Percentage = float  # 0.0 to 100.0 or 0.0 to 1.0 depending on context
Amount = float

# Identifiers
OrderId = str
PositionId = str
StrategyId = str
Symbol = str  # e.g., "BTC/USDT"

# Time-related
Timestamp = Union[datetime, pd.Timestamp, int, float]
Timeframe = Literal['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

# Trading
Side = Literal['long', 'short', 'buy', 'sell', 'BUY', 'SELL', 'LONG', 'SHORT']
OrderType = Literal['market', 'limit', 'stop_limit', 'stop_market']
OrderStatus = Literal['open', 'closed', 'canceled', 'expired', 'pending', 'filled', 'partially_filled']
PositionStatus = Literal['open', 'closed', 'partial']

# ============================================================================
# COMPLEX TYPE ALIASES
# ============================================================================

# Configuration
Config = Dict[str, Any]
ExchangeConfig = Dict[str, Any]
RiskConfig = Dict[str, Any]
StrategyConfig = Dict[str, Any]

# Market Data
OHLCV = List[List[Union[int, float]]]  # [[timestamp, open, high, low, close, volume], ...]
OHLCVRow = Tuple[int, float, float, float, float, float]
Ticker = Dict[str, Union[str, float, datetime]]
OrderBook = Dict[str, Union[List[List[float]], datetime]]  # {'bids': [[price, size], ...], 'asks': [...]}
MarketData = Dict[str, Union[pd.DataFrame, Dict[str, Any]]]

# Trading Signals
TakeProfitLevel = Tuple[Price, Percentage]  # (price, size_percentage)
TakeProfitLevels = List[TakeProfitLevel]
SignalReason = List[str]
SignalMetadata = Dict[str, Any]

# Order Results
OrderResult = Dict[str, Any]
TradeResult = Dict[str, Any]

# Performance Metrics
PerformanceMetrics = Dict[str, float]
BacktestResults = Dict[str, Any]

# ============================================================================
# PROTOCOL DEFINITIONS (Structural Subtyping)
# ============================================================================

class ExchangeClientProtocol(Protocol):
    """Protocol for exchange client implementations"""

    symbol: str
    base: str
    quote: str
    testnet: bool
    is_connected: bool

    def get_ticker(self, symbol: Optional[str] = None) -> Ticker:
        ...

    def get_balance(self, currency: Optional[str] = None) -> Dict[str, Any]:
        ...

    def fetch_ohlcv(
        self,
        timeframe: str = '5m',
        limit: int = 500,
        since: Optional[int] = None
    ) -> OHLCV:
        ...

    def create_order(
        self,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> OrderResult:
        ...

    def cancel_order(self, order_id: str) -> bool:
        ...

    def test_connectivity(self) -> bool:
        ...

    def reconnect(self) -> bool:
        ...


class StrategyProtocol(Protocol):
    """Protocol for strategy implementations"""

    name: str
    symbol: str
    enabled: bool

    def generate_signals(self, market_data: MarketData) -> List[Any]:
        ...

    def validate_signal(self, signal: Any) -> bool:
        ...


class LoggerProtocol(Protocol):
    """Protocol for logger implementations"""

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        ...

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        ...

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        ...

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        ...


# ============================================================================
# GENERIC TYPE VARIABLES
# ============================================================================

T = TypeVar('T')
ConfigT = TypeVar('ConfigT', bound=Dict[str, Any])
DataFrameT = TypeVar('DataFrameT', bound=pd.DataFrame)

# ============================================================================
# CALLBACK TYPES
# ============================================================================

SignalCallback = Callable[[Any], None]
OrderCallback = Callable[[OrderResult], None]
ErrorCallback = Callable[[Exception], None]
PriceUpdateCallback = Callable[[Price], None]

# ============================================================================
# RESULT TYPES (for error handling)
# ============================================================================

Success = Tuple[Literal[True], T]
Failure = Tuple[Literal[False], str]
Result = Union[Success[T], Failure]


def success(value: T) -> Success[T]:
    """Create a success result"""
    return (True, value)


def failure(error: str) -> Failure:
    """Create a failure result"""
    return (False, error)


def is_success(result: Result) -> bool:
    """Check if result is successful"""
    return result[0] is True


def unwrap(result: Result[T]) -> T:
    """Unwrap a result, raising ValueError on failure"""
    if result[0]:
        return result[1]  # type: ignore
    raise ValueError(f"Unwrap called on failure: {result[1]}")


__all__ = [
    # Basic types
    'Price', 'Size', 'Volume', 'Percentage', 'Amount',
    'OrderId', 'PositionId', 'StrategyId', 'Symbol',
    'Timestamp', 'Timeframe',
    'Side', 'OrderType', 'OrderStatus', 'PositionStatus',

    # Complex types
    'Config', 'ExchangeConfig', 'RiskConfig', 'StrategyConfig',
    'OHLCV', 'OHLCVRow', 'Ticker', 'OrderBook', 'MarketData',
    'TakeProfitLevel', 'TakeProfitLevels', 'SignalReason', 'SignalMetadata',
    'OrderResult', 'TradeResult',
    'PerformanceMetrics', 'BacktestResults',

    # Protocols
    'ExchangeClientProtocol', 'StrategyProtocol', 'LoggerProtocol',

    # Generic
    'T', 'ConfigT', 'DataFrameT',

    # Callbacks
    'SignalCallback', 'OrderCallback', 'ErrorCallback', 'PriceUpdateCallback',

    # Result types
    'Result', 'Success', 'Failure',
    'success', 'failure', 'is_success', 'unwrap'
]
