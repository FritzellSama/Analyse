"""
Safe mathematical operations for Quantum Trader Pro.
Prevents division by zero, NaN/INF propagation.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional
from loguru import logger

# Type aliases
Numeric = Union[int, float, np.number]
ArrayLike = Union[pd.Series, np.ndarray, list]


def safe_divide(
    numerator: Union[Numeric, ArrayLike],
    denominator: Union[Numeric, ArrayLike],
    default: float = 0.0,
    epsilon: float = 1e-10
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safe division that handles zero and near-zero denominators.

    Args:
        numerator: Value(s) to divide
        denominator: Value(s) to divide by
        default: Default value when denominator is zero
        epsilon: Threshold for considering denominator as zero

    Returns:
        Result of division or default value

    Examples:
        safe_divide(10, 0) -> 0.0
        safe_divide(10, 2) -> 5.0
        safe_divide(pd.Series([10, 20]), pd.Series([2, 0])) -> pd.Series([5.0, 0.0])
    """
    # Handle pandas Series
    if isinstance(denominator, pd.Series):
        # Replace zeros and very small values with NaN
        safe_denom = denominator.replace(0, np.nan)
        safe_denom = safe_denom.where(safe_denom.abs() > epsilon, np.nan)
        result = numerator / safe_denom
        return result.fillna(default)

    # Handle numpy arrays
    if isinstance(denominator, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(
                np.abs(denominator) > epsilon,
                numerator / denominator,
                default
            )
        # Replace any remaining inf/nan
        result = np.where(np.isfinite(result), result, default)
        return result

    # Handle scalars
    if abs(denominator) < epsilon:
        logger.debug(f"Division by near-zero: {numerator}/{denominator}, returning {default}")
        return default

    try:
        result = numerator / denominator
        if not np.isfinite(result):
            return default
        return result
    except (ZeroDivisionError, FloatingPointError):
        return default


def safe_log(
    value: Union[Numeric, ArrayLike],
    epsilon: float = 1e-10
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safe logarithm that handles zero and negative values.

    Args:
        value: Value(s) to take log of
        epsilon: Minimum value to use

    Returns:
        Log of value, clamped to avoid -inf
    """
    if isinstance(value, pd.Series):
        return np.log(value.clip(lower=epsilon))

    if isinstance(value, np.ndarray):
        return np.log(np.maximum(value, epsilon))

    return np.log(max(value, epsilon))


def safe_sqrt(
    value: Union[Numeric, ArrayLike],
    epsilon: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safe square root that handles negative values.
    """
    if isinstance(value, pd.Series):
        return np.sqrt(value.clip(lower=epsilon))

    if isinstance(value, np.ndarray):
        return np.sqrt(np.maximum(value, epsilon))

    return np.sqrt(max(value, epsilon))


def safe_percentage(
    part: Union[Numeric, ArrayLike],
    whole: Union[Numeric, ArrayLike],
    default: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate percentage safely.

    Args:
        part: Numerator
        whole: Denominator (total)
        default: Default if whole is zero

    Returns:
        Percentage (0-100 scale)
    """
    return safe_divide(part, whole, default) * 100


def safe_ratio(
    a: Union[Numeric, ArrayLike],
    b: Union[Numeric, ArrayLike],
    default: float = 1.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate ratio safely.
    """
    return safe_divide(a, b, default)


def clean_series(
    series: pd.Series,
    fill_method: str = 'ffill',
    fill_value: float = 0.0
) -> pd.Series:
    """
    Clean a pandas Series by handling NaN and INF values.

    Args:
        series: Input series
        fill_method: Method to fill NaN ('ffill', 'bfill', 'value')
        fill_value: Value to use if fill_method is 'value'

    Returns:
        Cleaned series
    """
    # Replace inf with NaN
    series = series.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values
    if fill_method == 'ffill':
        series = series.ffill().fillna(fill_value)
    elif fill_method == 'bfill':
        series = series.bfill().fillna(fill_value)
    else:
        series = series.fillna(fill_value)

    return series


def clean_dataframe(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """
    Clean entire DataFrame by handling NaN and INF values.
    """
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Forward fill, then fill remaining with fill_value
    df = df.ffill().fillna(fill_value)

    return df


def validate_numeric(
    value: Numeric,
    name: str = "value",
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_zero: bool = True,
    allow_negative: bool = True
) -> bool:
    """
    Validate that a numeric value is within acceptable bounds.

    Returns:
        True if valid, False otherwise
    """
    # Check for NaN/INF
    if not np.isfinite(value):
        logger.warning(f"{name} is not finite: {value}")
        return False

    # Check zero
    if not allow_zero and value == 0:
        logger.warning(f"{name} cannot be zero")
        return False

    # Check negative
    if not allow_negative and value < 0:
        logger.warning(f"{name} cannot be negative: {value}")
        return False

    # Check bounds
    if min_val is not None and value < min_val:
        logger.warning(f"{name} below minimum: {value} < {min_val}")
        return False

    if max_val is not None and value > max_val:
        logger.warning(f"{name} above maximum: {value} > {max_val}")
        return False

    return True


def clamp(
    value: Union[Numeric, ArrayLike],
    min_val: float,
    max_val: float
) -> Union[float, np.ndarray, pd.Series]:
    """
    Clamp value(s) to range [min_val, max_val].
    """
    if isinstance(value, pd.Series):
        return value.clip(lower=min_val, upper=max_val)

    if isinstance(value, np.ndarray):
        return np.clip(value, min_val, max_val)

    return max(min_val, min(value, max_val))


def safe_mean(values: ArrayLike, default: float = 0.0) -> float:
    """
    Calculate mean safely, handling empty arrays and NaN values.
    """
    if isinstance(values, pd.Series):
        if values.empty or values.isna().all():
            return default
        return float(values.mean())

    if isinstance(values, np.ndarray):
        if len(values) == 0 or np.all(np.isnan(values)):
            return default
        return float(np.nanmean(values))

    if isinstance(values, list):
        if not values:
            return default
        return sum(values) / len(values)

    return default


def safe_std(values: ArrayLike, default: float = 0.0) -> float:
    """
    Calculate standard deviation safely.
    """
    if isinstance(values, pd.Series):
        if values.empty or len(values) < 2:
            return default
        return float(values.std())

    if isinstance(values, np.ndarray):
        if len(values) < 2:
            return default
        return float(np.nanstd(values))

    if isinstance(values, list):
        if len(values) < 2:
            return default
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    return default


__all__ = [
    'safe_divide',
    'safe_log',
    'safe_sqrt',
    'safe_percentage',
    'safe_ratio',
    'clean_series',
    'clean_dataframe',
    'validate_numeric',
    'clamp',
    'safe_mean',
    'safe_std'
]
