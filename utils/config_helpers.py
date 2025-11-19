"""
Safe configuration access helpers for Quantum Trader Pro.
Prevents KeyError and provides default values.
"""
from typing import Any, Dict, Optional, List
from loguru import logger


def get_nested_config(config: Dict[str, Any], *keys, default=None) -> Any:
    """
    Safely access nested configuration values.

    Args:
        config: Configuration dictionary
        *keys: Sequence of keys to traverse
        default: Default value if path not found

    Returns:
        Value at path or default

    Example:
        get_nested_config(config, 'strategies', 'grid_trading', 'enabled', default=False)
    """
    if not isinstance(config, dict):
        logger.debug(f"Config is not a dict: {type(config)}")
        return default

    result = config
    path = []

    for key in keys:
        path.append(str(key))
        if not isinstance(result, dict):
            logger.debug(f"Config path {'.'.join(path[:-1])} is not a dict")
            return default

        if key not in result:
            logger.debug(f"Config key '{key}' not found in path {'.'.join(path[:-1]) or 'root'}")
            return default

        result = result[key]

    return result


def validate_config_section(config: Dict[str, Any], section: str, required_keys: List[str]) -> bool:
    """
    Validate that a config section has all required keys.

    Returns:
        True if valid, False otherwise
    """
    if section not in config:
        logger.error(f"Missing config section: {section}")
        return False

    section_config = config[section]
    if not isinstance(section_config, dict):
        logger.error(f"Config section '{section}' is not a dictionary")
        return False

    missing = [k for k in required_keys if k not in section_config]
    if missing:
        logger.error(f"Missing keys in '{section}': {missing}")
        return False

    return True


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    Override values take precedence.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_strategy_config(config: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific strategy with defaults.

    Args:
        config: Main configuration dictionary
        strategy_name: Name of the strategy

    Returns:
        Strategy configuration with defaults applied
    """
    default_strategy = {
        'enabled': False,
        'weight': 0.0,
        'params': {}
    }

    strategies = config.get('strategies', {})
    if not isinstance(strategies, dict):
        return default_strategy

    strategy_cfg = strategies.get(strategy_name, {})
    if not isinstance(strategy_cfg, dict):
        return default_strategy

    # Merge with defaults
    result = default_strategy.copy()
    result.update(strategy_cfg)

    return result


def get_risk_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get risk configuration with defaults.
    """
    defaults = {
        'max_risk_percent': 2.0,
        'max_daily_loss_percent': 5.0,
        'max_positions': 3,
        'max_position_size_percent': 10.0,
        'min_position_size': 0.001,
        'stop_loss': {
            'method': 'atr',
            'atr_multiplier': 2.0,
            'fixed_percent': 2.0,
            'trailing': {
                'enabled': False,
                'breakeven_atr_trigger': 1.5,
                'trail_distance_atr': 1.0,
                'step_atr': 0.5
            }
        },
        'take_profit': {
            'method': 'single',
            'levels': [{'percent': 100, 'target_percent': 2.0}]
        }
    }

    risk_config = config.get('risk', {})
    if not isinstance(risk_config, dict):
        return defaults

    return merge_configs(defaults, risk_config)


def get_exchange_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get exchange configuration with defaults.
    """
    defaults = {
        'primary': {
            'name': 'binance',
            'testnet': True,
            'timeout_seconds': 30,
            'rate_limit_buffer': 0.2
        }
    }

    exchange_config = config.get('exchange', {})
    if not isinstance(exchange_config, dict):
        return defaults

    return merge_configs(defaults, exchange_config)


def get_symbol(config: Dict[str, Any], symbol_type: str = 'primary') -> str:
    """
    Get trading symbol from config.

    Args:
        config: Configuration dictionary
        symbol_type: 'primary' or index for secondary

    Returns:
        Symbol string (e.g., 'BTC/USDT')
    """
    symbols = config.get('symbols', {})

    if symbol_type == 'primary':
        return symbols.get('primary', 'BTC/USDT')

    secondary = symbols.get('secondary', [])
    if isinstance(symbol_type, int) and 0 <= symbol_type < len(secondary):
        return secondary[symbol_type]

    return 'BTC/USDT'


def validate_symbol_format(symbol: str) -> bool:
    """
    Validate that symbol is in correct format (BASE/QUOTE).
    """
    if not symbol or not isinstance(symbol, str):
        return False

    if '/' not in symbol:
        return False

    parts = symbol.split('/')
    if len(parts) != 2:
        return False

    base, quote = parts
    if not base or not quote:
        return False

    # Check if alphanumeric
    if not base.replace('.', '').isalnum() or not quote.replace('.', '').isalnum():
        return False

    return True


def parse_symbol(symbol: str) -> tuple:
    """
    Parse symbol into base and quote currencies.

    Args:
        symbol: Symbol string (e.g., 'BTC/USDT')

    Returns:
        Tuple of (base, quote) or raises ValueError
    """
    if not validate_symbol_format(symbol):
        raise ValueError(f"Invalid symbol format: {symbol}. Expected BASE/QUOTE")

    parts = symbol.split('/')
    return parts[0], parts[1]


__all__ = [
    'get_nested_config',
    'validate_config_section',
    'merge_configs',
    'get_strategy_config',
    'get_risk_config',
    'get_exchange_config',
    'get_symbol',
    'validate_symbol_format',
    'parse_symbol'
]
