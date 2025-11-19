"""
Custom Exceptions - Quantum Trader Pro
Centralized exception hierarchy for better error handling
"""

from typing import Optional, Any, Dict


# ============================================================================
# BASE EXCEPTIONS
# ============================================================================

class QuantumTraderError(Exception):
    """Base exception for all Quantum Trader errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(QuantumTraderError):
    """Error in configuration"""
    pass


class ValidationError(QuantumTraderError):
    """Validation error for inputs or data"""
    pass


# ============================================================================
# EXCHANGE/API ERRORS
# ============================================================================

class ExchangeError(QuantumTraderError):
    """Base exception for exchange-related errors"""
    pass


class ConnectionError(ExchangeError):
    """Connection to exchange failed"""
    pass


class AuthenticationError(ExchangeError):
    """API key authentication failed"""
    pass


class RateLimitError(ExchangeError):
    """Rate limit exceeded"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        **kwargs
    ):
        self.retry_after = retry_after
        details = kwargs.get('details', {})
        if retry_after:
            details['retry_after_seconds'] = retry_after
        super().__init__(message, details)


class InsufficientFundsError(ExchangeError):
    """Insufficient balance for operation"""

    def __init__(
        self,
        message: str,
        required: Optional[float] = None,
        available: Optional[float] = None,
        **kwargs
    ):
        self.required = required
        self.available = available
        details = kwargs.get('details', {})
        if required is not None:
            details['required'] = required
        if available is not None:
            details['available'] = available
        super().__init__(message, details)


class OrderError(ExchangeError):
    """Error during order execution"""
    pass


class OrderNotFoundError(OrderError):
    """Order not found"""

    def __init__(self, order_id: str, **kwargs):
        self.order_id = order_id
        super().__init__(f"Order not found: {order_id}", **kwargs)


class OrderCancelError(OrderError):
    """Error canceling order"""
    pass


class OrderTimeoutError(OrderError):
    """Order execution timed out"""

    def __init__(self, order_id: str, timeout: float, **kwargs):
        self.order_id = order_id
        self.timeout = timeout
        details = kwargs.get('details', {})
        details['timeout_seconds'] = timeout
        super().__init__(f"Order {order_id} timed out after {timeout}s", details)


# ============================================================================
# TRADING LOGIC ERRORS
# ============================================================================

class TradingError(QuantumTraderError):
    """Base exception for trading logic errors"""
    pass


class PositionError(TradingError):
    """Error related to position management"""
    pass


class PositionNotFoundError(PositionError):
    """Position not found"""

    def __init__(self, position_id: str, **kwargs):
        self.position_id = position_id
        super().__init__(f"Position not found: {position_id}", **kwargs)


class PositionLimitExceededError(PositionError):
    """Maximum positions limit reached"""

    def __init__(
        self,
        current: int,
        maximum: int,
        **kwargs
    ):
        self.current = current
        self.maximum = maximum
        details = kwargs.get('details', {})
        details['current_positions'] = current
        details['max_positions'] = maximum
        super().__init__(
            f"Position limit exceeded: {current}/{maximum}",
            details
        )


class SignalError(TradingError):
    """Error in signal generation or validation"""
    pass


class InvalidSignalError(SignalError):
    """Signal failed validation"""

    def __init__(self, reason: str, signal_data: Optional[Dict] = None, **kwargs):
        self.reason = reason
        details = kwargs.get('details', {})
        if signal_data:
            details['signal'] = signal_data
        super().__init__(f"Invalid signal: {reason}", details)


class StrategyError(TradingError):
    """Error in strategy execution"""
    pass


# ============================================================================
# RISK MANAGEMENT ERRORS
# ============================================================================

class RiskError(QuantumTraderError):
    """Base exception for risk management errors"""
    pass


class RiskLimitExceededError(RiskError):
    """Risk limit breached"""

    def __init__(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        **kwargs
    ):
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value
        details = kwargs.get('details', {})
        details['limit_type'] = limit_type
        details['current'] = current_value
        details['limit'] = limit_value
        super().__init__(
            f"Risk limit exceeded: {limit_type} ({current_value} > {limit_value})",
            details
        )


class CircuitBreakerError(RiskError):
    """Circuit breaker is active"""

    def __init__(self, reason: str, cooldown_remaining: Optional[float] = None, **kwargs):
        self.reason = reason
        self.cooldown_remaining = cooldown_remaining
        details = kwargs.get('details', {})
        details['reason'] = reason
        if cooldown_remaining:
            details['cooldown_remaining'] = cooldown_remaining
        super().__init__(f"Circuit breaker active: {reason}", details)


class MaxDrawdownError(RiskError):
    """Maximum drawdown reached"""

    def __init__(self, current_dd: float, max_dd: float, **kwargs):
        self.current_dd = current_dd
        self.max_dd = max_dd
        details = kwargs.get('details', {})
        details['current_drawdown'] = current_dd
        details['max_allowed'] = max_dd
        super().__init__(
            f"Max drawdown exceeded: {current_dd:.2%} > {max_dd:.2%}",
            details
        )


# ============================================================================
# DATA ERRORS
# ============================================================================

class DataError(QuantumTraderError):
    """Base exception for data-related errors"""
    pass


class InsufficientDataError(DataError):
    """Not enough data for operation"""

    def __init__(self, required: int, available: int, **kwargs):
        self.required = required
        self.available = available
        details = kwargs.get('details', {})
        details['required_bars'] = required
        details['available_bars'] = available
        super().__init__(
            f"Insufficient data: need {required} bars, have {available}",
            details
        )


class DataValidationError(DataError):
    """Data failed validation"""
    pass


class StaleDataError(DataError):
    """Data is too old"""

    def __init__(self, age_seconds: float, max_age: float, **kwargs):
        self.age_seconds = age_seconds
        self.max_age = max_age
        details = kwargs.get('details', {})
        details['age_seconds'] = age_seconds
        details['max_age_seconds'] = max_age
        super().__init__(
            f"Stale data: {age_seconds:.0f}s old (max: {max_age:.0f}s)",
            details
        )


# ============================================================================
# ML ERRORS
# ============================================================================

class MLError(QuantumTraderError):
    """Base exception for ML-related errors"""
    pass


class ModelNotFoundError(MLError):
    """ML model not found"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        super().__init__(f"Model not found: {model_name}", **kwargs)


class ModelTrainingError(MLError):
    """Error during model training"""
    pass


class PredictionError(MLError):
    """Error during prediction"""
    pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def wrap_exception(
    exc: Exception,
    wrapper_class: type = QuantumTraderError,
    additional_context: Optional[Dict[str, Any]] = None
) -> QuantumTraderError:
    """
    Wrap a standard exception in a Quantum Trader exception

    Args:
        exc: Original exception
        wrapper_class: Exception class to wrap with
        additional_context: Additional context to add

    Returns:
        Wrapped exception
    """
    details = {
        'original_exception': type(exc).__name__,
        'original_message': str(exc)
    }
    if additional_context:
        details.update(additional_context)

    return wrapper_class(str(exc), details)


__all__ = [
    # Base
    'QuantumTraderError',
    'ConfigurationError',
    'ValidationError',

    # Exchange
    'ExchangeError',
    'ConnectionError',
    'AuthenticationError',
    'RateLimitError',
    'InsufficientFundsError',
    'OrderError',
    'OrderNotFoundError',
    'OrderCancelError',
    'OrderTimeoutError',

    # Trading
    'TradingError',
    'PositionError',
    'PositionNotFoundError',
    'PositionLimitExceededError',
    'SignalError',
    'InvalidSignalError',
    'StrategyError',

    # Risk
    'RiskError',
    'RiskLimitExceededError',
    'CircuitBreakerError',
    'MaxDrawdownError',

    # Data
    'DataError',
    'InsufficientDataError',
    'DataValidationError',
    'StaleDataError',

    # ML
    'MLError',
    'ModelNotFoundError',
    'ModelTrainingError',
    'PredictionError',

    # Helpers
    'wrap_exception'
]
