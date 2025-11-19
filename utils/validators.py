"""
Validators - Quantum Trader Pro
Module centralisé pour toutes les validations communes
"""

from typing import Optional, Tuple
from utils.logger import setup_logger

logger = setup_logger('Validators')


def validate_price(price: float, price_type: str = "price", min_price: float = 0.00000001, max_price: float = 1_000_000) -> bool:
    """
    Valide qu'un prix est dans des limites acceptables

    Args:
        price: Prix à valider
        price_type: Type de prix pour logging (entry, stop_loss, etc.)
        min_price: Prix minimum acceptable
        max_price: Prix maximum acceptable

    Returns:
        True si valide, False sinon
    """
    if price is None:
        logger.error(f"{price_type} est None")
        return False

    if price <= 0:
        logger.error(f"{price_type} invalide (<=0): {price}")
        return False

    if price < min_price:
        logger.error(f"{price_type} trop bas: {price} < {min_price}")
        return False

    if price > max_price:
        logger.error(f"{price_type} trop élevé: {price} > {max_price}")
        return False

    return True


def validate_balance(balance: float, min_balance: float = 0.0) -> bool:
    """
    Valide qu'un solde est suffisant

    Args:
        balance: Solde à valider
        min_balance: Solde minimum requis

    Returns:
        True si valide
    """
    if balance is None:
        logger.error("Balance est None")
        return False

    if balance <= min_balance:
        logger.error(f"Balance insuffisant: {balance} <= {min_balance}")
        return False

    return True


def validate_size(size: float, min_size: float = 0.0, max_size: Optional[float] = None) -> bool:
    """
    Valide la taille d'une position

    Args:
        size: Taille à valider
        min_size: Taille minimum
        max_size: Taille maximum (optionnel)

    Returns:
        True si valide
    """
    if size is None:
        logger.error("Size est None")
        return False

    if size <= min_size:
        logger.error(f"Size invalide: {size} <= {min_size}")
        return False

    if max_size is not None and size > max_size:
        logger.error(f"Size trop grande: {size} > {max_size}")
        return False

    return True


def validate_stop_loss(entry_price: float, stop_loss: float, side: str = 'long') -> Tuple[bool, str]:
    """
    Valide qu'un stop loss est cohérent avec le prix d'entrée

    Args:
        entry_price: Prix d'entrée
        stop_loss: Prix du stop loss
        side: 'long' ou 'short'

    Returns:
        Tuple (is_valid, error_message)
    """
    if not validate_price(entry_price, "entry_price"):
        return False, "Prix d'entrée invalide"

    if not validate_price(stop_loss, "stop_loss"):
        return False, "Stop loss invalide"

    # Vérifier que SL != Entry
    if abs(entry_price - stop_loss) < 0.00000001:
        return False, f"Stop loss égal au prix d'entrée: {entry_price} == {stop_loss}"

    # Pour un long, SL doit être < entry
    if side.lower() == 'long' and stop_loss >= entry_price:
        return False, f"Stop loss ({stop_loss}) doit être < prix d'entrée ({entry_price}) pour un LONG"

    # Pour un short, SL doit être > entry
    if side.lower() == 'short' and stop_loss <= entry_price:
        return False, f"Stop loss ({stop_loss}) doit être > prix d'entrée ({entry_price}) pour un SHORT"

    return True, ""


def validate_take_profit(entry_price: float, take_profit: float, side: str = 'long') -> Tuple[bool, str]:
    """
    Valide qu'un take profit est cohérent avec le prix d'entrée

    Args:
        entry_price: Prix d'entrée
        take_profit: Prix du take profit
        side: 'long' ou 'short'

    Returns:
        Tuple (is_valid, error_message)
    """
    if not validate_price(entry_price, "entry_price"):
        return False, "Prix d'entrée invalide"

    if not validate_price(take_profit, "take_profit"):
        return False, "Take profit invalide"

    # Pour un long, TP doit être > entry
    if side.lower() == 'long' and take_profit <= entry_price:
        return False, f"Take profit ({take_profit}) doit être > prix d'entrée ({entry_price}) pour un LONG"

    # Pour un short, TP doit être < entry
    if side.lower() == 'short' and take_profit >= entry_price:
        return False, f"Take profit ({take_profit}) doit être < prix d'entrée ({entry_price}) pour un SHORT"

    return True, ""


def validate_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    min_rr: float = 1.0,
    side: str = 'long'
) -> Tuple[bool, float, str]:
    """
    Valide et calcule le ratio risque/récompense

    Args:
        entry_price: Prix d'entrée
        stop_loss: Stop loss
        take_profit: Take profit
        min_rr: Ratio minimum requis
        side: 'long' ou 'short'

    Returns:
        Tuple (is_valid, ratio, message)
    """
    # Valider les prix individuellement
    sl_valid, sl_msg = validate_stop_loss(entry_price, stop_loss, side)
    if not sl_valid:
        return False, 0.0, sl_msg

    tp_valid, tp_msg = validate_take_profit(entry_price, take_profit, side)
    if not tp_valid:
        return False, 0.0, tp_msg

    # Calculer le ratio
    if side.lower() == 'long':
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:
        risk = stop_loss - entry_price
        reward = entry_price - take_profit

    if risk <= 0:
        return False, 0.0, f"Risk invalide: {risk}"

    ratio = reward / risk

    if ratio < min_rr:
        return False, ratio, f"Ratio R/R insuffisant: {ratio:.2f} < {min_rr}"

    return True, ratio, f"Ratio R/R: {ratio:.2f}"


def validate_percentage(value: float, name: str = "percentage", min_val: float = 0.0, max_val: float = 100.0) -> bool:
    """
    Valide qu'une valeur est un pourcentage valide

    Args:
        value: Valeur à valider
        name: Nom pour le logging
        min_val: Minimum (défaut 0%)
        max_val: Maximum (défaut 100%)

    Returns:
        True si valide
    """
    if value is None:
        logger.error(f"{name} est None")
        return False

    if value < min_val or value > max_val:
        logger.error(f"{name} hors limites: {value} (doit être entre {min_val} et {max_val})")
        return False

    return True


def validate_timeframe(timeframe: str) -> bool:
    """
    Valide qu'un timeframe est supporté

    Args:
        timeframe: Timeframe à valider ('1m', '5m', '1h', etc.)

    Returns:
        True si valide
    """
    valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

    if timeframe not in valid_timeframes:
        logger.error(f"Timeframe invalide: {timeframe}. Valides: {valid_timeframes}")
        return False

    return True


def validate_symbol(symbol: str) -> bool:
    """
    Valide le format d'un symbole

    Args:
        symbol: Symbole à valider (ex: 'BTC/USDT')

    Returns:
        True si valide
    """
    if not symbol:
        logger.error("Symbol vide")
        return False

    if '/' not in symbol:
        logger.error(f"Symbol invalide (pas de '/'): {symbol}")
        return False

    parts = symbol.split('/')
    if len(parts) != 2:
        logger.error(f"Symbol invalide (format): {symbol}")
        return False

    base, quote = parts
    if not base or not quote:
        logger.error(f"Symbol invalide (base ou quote vide): {symbol}")
        return False

    return True


def validate_order_params(
    side: str,
    order_type: str,
    amount: float,
    price: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Valide les paramètres d'un ordre

    Args:
        side: 'buy' ou 'sell'
        order_type: 'market' ou 'limit'
        amount: Quantité
        price: Prix (requis pour limit)

    Returns:
        Tuple (is_valid, error_message)
    """
    # Valider side
    if side.lower() not in ['buy', 'sell']:
        return False, f"Side invalide: {side}. Doit être 'buy' ou 'sell'"

    # Valider order_type
    if order_type.lower() not in ['market', 'limit']:
        return False, f"Order type invalide: {order_type}. Doit être 'market' ou 'limit'"

    # Valider amount
    if not validate_size(amount):
        return False, f"Amount invalide: {amount}"

    # Valider price pour limit orders
    if order_type.lower() == 'limit':
        if price is None:
            return False, "Prix requis pour un ordre limit"
        if not validate_price(price, "limit_price"):
            return False, f"Prix invalide pour ordre limit: {price}"

    return True, ""


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Division sécurisée évitant la division par zéro

    Args:
        numerator: Numérateur
        denominator: Dénominateur
        default: Valeur par défaut si division impossible

    Returns:
        Résultat ou valeur par défaut
    """
    if denominator == 0 or abs(denominator) < 0.00000001:
        logger.warning(f"Division par zéro évitée: {numerator} / {denominator}")
        return default

    return numerator / denominator


__all__ = [
    'validate_price',
    'validate_balance',
    'validate_size',
    'validate_stop_loss',
    'validate_take_profit',
    'validate_risk_reward_ratio',
    'validate_percentage',
    'validate_timeframe',
    'validate_symbol',
    'validate_order_params',
    'safe_division'
]
