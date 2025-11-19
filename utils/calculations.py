"""
Calculations - Quantum Trader Pro
Module centralisé pour tous les calculs techniques communs
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from utils.logger import setup_logger
from utils.safe_math import safe_divide

logger = setup_logger('Calculations')


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcule l'Average True Range (ATR)

    Args:
        df: DataFrame avec colonnes 'high', 'low', 'close'
        period: Période de calcul (défaut 14)

    Returns:
        Series avec les valeurs ATR
    """
    if len(df) < period:
        logger.warning(f"Données insuffisantes pour ATR: {len(df)} < {period}")
        return pd.Series([0.0] * len(df), index=df.index)

    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR = EMA du True Range
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def calculate_atr_value(high: float, low: float, close: float, prev_close: float) -> float:
    """
    Calcule le True Range pour une seule bougie

    Args:
        high: Plus haut
        low: Plus bas
        close: Clôture
        prev_close: Clôture précédente

    Returns:
        True Range
    """
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    return max(tr1, tr2, tr3)


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average

    Args:
        data: Données source
        period: Période

    Returns:
        SMA
    """
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average

    Args:
        data: Données source
        period: Période

    Returns:
        EMA
    """
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index

    Args:
        data: Prix de clôture
        period: Période (défaut 14)

    Returns:
        RSI (0-100)
    """
    delta = data.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)  # Valeur neutre si pas de données


def calculate_bollinger_bands(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bandes de Bollinger

    Args:
        data: Prix de clôture
        period: Période MA
        std_dev: Nombre d'écarts-types

    Returns:
        Tuple (upper_band, middle_band, lower_band)
    """
    middle = calculate_sma(data, period)
    std = data.rolling(window=period).std()

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    return upper, middle, lower


def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence)

    Args:
        data: Prix de clôture
        fast_period: Période EMA rapide
        slow_period: Période EMA lente
        signal_period: Période signal

    Returns:
        Tuple (macd_line, signal_line, histogram)
    """
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_volatility(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Calcule la volatilité (écart-type des rendements)

    Args:
        data: Prix de clôture
        period: Période

    Returns:
        Volatilité annualisée
    """
    returns = data.pct_change()
    volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualisé

    return volatility


def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calcule le Kelly Criterion pour la taille de position optimale

    Args:
        win_rate: Taux de succès (0-1)
        avg_win: Gain moyen
        avg_loss: Perte moyenne (valeur positive)

    Returns:
        Fraction Kelly (0-1)
    """
    if avg_loss <= 0:
        logger.warning("avg_loss doit être positif pour Kelly")
        return 0.0

    if win_rate <= 0 or win_rate >= 1:
        logger.warning(f"Win rate invalide: {win_rate}")
        return 0.0

    # Kelly = W - (1-W)/R où R = avg_win/avg_loss
    r = safe_divide(avg_win, avg_loss, default=1.0)
    kelly = win_rate - ((1 - win_rate) / r)

    # Limiter entre 0 et 1
    kelly = max(0.0, min(1.0, kelly))

    return kelly


def calculate_position_size_risk_based(
    balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float
) -> float:
    """
    Calcule la taille de position basée sur le risque

    Args:
        balance: Capital disponible
        risk_percent: % du capital à risquer (ex: 1.0 pour 1%)
        entry_price: Prix d'entrée
        stop_loss: Prix du stop loss

    Returns:
        Taille de position en unités
    """
    if balance <= 0:
        logger.error(f"Balance invalide: {balance}")
        return 0.0

    if risk_percent <= 0 or risk_percent > 100:
        logger.error(f"Risk percent invalide: {risk_percent}")
        return 0.0

    risk_amount = balance * (risk_percent / 100)
    risk_per_unit = abs(entry_price - stop_loss)

    if risk_per_unit < 0.00000001:
        logger.error("Distance au stop loss trop faible")
        return 0.0

    size = safe_divide(risk_amount, risk_per_unit, default=0.0)

    return size


def calculate_pnl(
    entry_price: float,
    current_price: float,
    size: float,
    side: str = 'long'
) -> Tuple[float, float]:
    """
    Calcule le P&L d'une position

    Args:
        entry_price: Prix d'entrée
        current_price: Prix actuel
        size: Taille de position
        side: 'long' ou 'short'

    Returns:
        Tuple (pnl_absolute, pnl_percent)
    """
    if side.lower() == 'long':
        pnl = (current_price - entry_price) * size
    else:  # short
        pnl = (entry_price - current_price) * size

    pnl_percent = safe_divide(pnl, (entry_price * size), default=0.0) * 100

    return pnl, pnl_percent


def calculate_drawdown(peak_value: float, current_value: float) -> Tuple[float, float]:
    """
    Calcule le drawdown

    Args:
        peak_value: Valeur maximale atteinte
        current_value: Valeur actuelle

    Returns:
        Tuple (drawdown_absolute, drawdown_percent)
    """
    if peak_value <= 0:
        return 0.0, 0.0

    drawdown = peak_value - current_value
    drawdown_percent = safe_divide(drawdown, peak_value, default=0.0) * 100

    return max(0, drawdown), max(0, drawdown_percent)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calcule le ratio de Sharpe

    Args:
        returns: Série des rendements
        risk_free_rate: Taux sans risque annualisé
        periods_per_year: Périodes par an (252 pour journalier)

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()

    if std_return == 0:
        return 0.0

    sharpe = safe_divide(mean_return, std_return, default=0.0) * np.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calcule le ratio de Sortino (pénalise uniquement les rendements négatifs)

    Args:
        returns: Série des rendements
        risk_free_rate: Taux sans risque
        periods_per_year: Périodes par an

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_return = excess_returns.mean()

    # Downside deviation (seulement les rendements négatifs)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return float('inf')  # Pas de rendements négatifs

    downside_std = negative_returns.std()

    if downside_std == 0:
        return float('inf')

    sortino = safe_divide(mean_return, downside_std, default=0.0) * np.sqrt(periods_per_year)

    return sortino


def calculate_win_rate(wins: int, total: int) -> float:
    """
    Calcule le taux de réussite

    Args:
        wins: Nombre de trades gagnants
        total: Nombre total de trades

    Returns:
        Win rate (0-1)
    """
    return safe_divide(wins, total, default=0.0)


def calculate_profit_factor(gross_profit: float, gross_loss: float) -> float:
    """
    Calcule le profit factor

    Args:
        gross_profit: Somme des gains
        gross_loss: Somme des pertes (valeur positive)

    Returns:
        Profit factor
    """
    if gross_loss <= 0:
        return float('inf') if gross_profit > 0 else 0.0

    return safe_divide(gross_profit, gross_loss, default=0.0)


def round_to_precision(value: float, precision: int = 8) -> float:
    """
    Arrondit une valeur avec une précision donnée

    Args:
        value: Valeur à arrondir
        precision: Nombre de décimales

    Returns:
        Valeur arrondie
    """
    return round(value, precision)


def round_to_tick_size(value: float, tick_size: float) -> float:
    """
    Arrondit une valeur au tick size le plus proche

    Args:
        value: Valeur à arrondir
        tick_size: Taille du tick

    Returns:
        Valeur arrondie au tick
    """
    if tick_size <= 0:
        return value

    return round(value / tick_size) * tick_size


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convertit un timeframe string en minutes.

    Args:
        timeframe: String comme '1m', '5m', '1h', '1d'

    Returns:
        Nombre de minutes

    Examples:
        '1m' -> 1
        '5m' -> 5
        '1h' -> 60
        '4h' -> 240
        '1d' -> 1440
    """
    mapping = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '8h': 480,
        '12h': 720,
        '1d': 1440,
        '3d': 4320,
        '1w': 10080,
        '1M': 43200
    }
    return mapping.get(timeframe, 60)  # Default 1h


__all__ = [
    'calculate_atr',
    'calculate_atr_value',
    'calculate_sma',
    'calculate_ema',
    'calculate_rsi',
    'calculate_bollinger_bands',
    'calculate_macd',
    'calculate_volatility',
    'calculate_kelly_criterion',
    'calculate_position_size_risk_based',
    'calculate_pnl',
    'calculate_drawdown',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_win_rate',
    'calculate_profit_factor',
    'round_to_precision',
    'round_to_tick_size',
    'timeframe_to_minutes'
]
