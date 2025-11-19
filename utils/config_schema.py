"""
Configuration Schema Validator - Quantum Trader Pro
Validates configuration structure and required fields
"""

from typing import Dict, Any, List, Tuple

import logging

# Logger simple sans dépendance circulaire
logger = logging.getLogger('ConfigValidator')
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def validate_config_schema(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates the configuration against required schema.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    warnings = []

    # Required top-level keys
    required_sections = ['exchange', 'symbols', 'capital', 'risk', 'timeframes']
    for section in required_sections:
        if section not in config:
            errors.append(f"Section '{section}' manquante dans config")

    if errors:
        return False, errors

    # Validate exchange section
    exchange_errors = _validate_exchange(config.get('exchange', {}))
    errors.extend(exchange_errors)

    # Validate symbols section
    symbols_errors = _validate_symbols(config.get('symbols', {}))
    errors.extend(symbols_errors)

    # Validate capital section
    capital_errors = _validate_capital(config.get('capital', {}))
    errors.extend(capital_errors)

    # Validate risk section
    risk_errors = _validate_risk(config.get('risk', {}))
    errors.extend(risk_errors)

    # Validate timeframes section
    timeframes_errors = _validate_timeframes(config.get('timeframes', {}))
    errors.extend(timeframes_errors)

    is_valid = len(errors) == 0

    if is_valid:
        logger.info("✅ Configuration schema validée avec succès")
    else:
        logger.error(f"❌ {len(errors)} erreurs de validation config")
        for error in errors:
            logger.error(f"   - {error}")

    return is_valid, errors


def _validate_exchange(exchange: Dict) -> List[str]:
    """Validate exchange section"""
    errors = []

    if 'primary' not in exchange:
        errors.append("exchange.primary manquant")
        return errors

    primary = exchange['primary']

    # API keys (peut être vide si .env)
    if 'testnet' not in primary:
        errors.append("exchange.primary.testnet (bool) manquant")
    elif not isinstance(primary['testnet'], bool):
        errors.append("exchange.primary.testnet doit être un booléen")

    return errors


def _validate_symbols(symbols: Dict) -> List[str]:
    """Validate symbols section"""
    errors = []

    if 'primary' not in symbols:
        errors.append("symbols.primary manquant")
    else:
        symbol = symbols['primary']
        if not isinstance(symbol, str):
            errors.append("symbols.primary doit être une string")
        elif '/' not in symbol:
            errors.append(f"symbols.primary '{symbol}' doit contenir '/' (ex: BTC/USDT)")

    return errors


def _validate_capital(capital: Dict) -> List[str]:
    """Validate capital section"""
    errors = []

    if 'initial' not in capital:
        errors.append("capital.initial manquant")
    else:
        initial = capital['initial']
        if not isinstance(initial, (int, float)):
            errors.append("capital.initial doit être un nombre")
        elif initial <= 0:
            errors.append("capital.initial doit être > 0")

    return errors


def _validate_risk(risk: Dict) -> List[str]:
    """Validate risk section"""
    errors = []

    required_risk_fields = [
        ('max_risk_per_trade_percent', (int, float), 0, 100),
        ('max_daily_loss_percent', (int, float), 0, 100),
        ('max_positions_simultaneous', int, 1, 50)
    ]

    for field, field_type, min_val, max_val in required_risk_fields:
        if field not in risk:
            errors.append(f"risk.{field} manquant")
        else:
            value = risk[field]
            if not isinstance(value, field_type):
                errors.append(f"risk.{field} doit être de type {field_type}")
            elif value < min_val or value > max_val:
                errors.append(f"risk.{field} doit être entre {min_val} et {max_val}")

    return errors


def _validate_timeframes(timeframes: Dict) -> List[str]:
    """Validate timeframes section"""
    errors = []

    valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']

    if 'trend' not in timeframes:
        errors.append("timeframes.trend manquant")
    elif timeframes['trend'] not in valid_timeframes:
        errors.append(f"timeframes.trend '{timeframes['trend']}' invalide. Valeurs acceptées: {valid_timeframes}")

    if 'signal' not in timeframes:
        errors.append("timeframes.signal manquant")
    elif timeframes['signal'] not in valid_timeframes:
        errors.append(f"timeframes.signal '{timeframes['signal']}' invalide")

    return errors


def get_default_config() -> Dict[str, Any]:
    """Returns a minimal valid configuration"""
    return {
        'exchange': {
            'primary': {
                'testnet': True,
                'api_key': '',
                'secret_key': ''
            }
        },
        'symbols': {
            'primary': 'BTC/USDT'
        },
        'capital': {
            'initial': 1000.0
        },
        'risk': {
            'max_risk_per_trade_percent': 2.0,
            'max_daily_loss_percent': 5.0,
            'max_positions_simultaneous': 3
        },
        'timeframes': {
            'trend': '1h',
            'signal': '5m'
        }
    }
