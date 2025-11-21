"""
Logging System - Quantum Trader Pro
Système de logging structuré avec couleurs et rotation
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from colorama import Fore, Back, Style, init

BASE_DIR = Path(__file__).resolve().parent.parent  # ajuste selon la profondeur
sys.path.append(str(BASE_DIR))
# Initialiser colorama
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs pour console"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }
    
    ICONS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️ ',
        'WARNING': '⚠️ ',
        'ERROR': '❌',
        'CRITICAL': '🚨',
    }
    
    def format(self, record):
        # Ajouter couleur et icône
        levelname = record.levelname
        color = self.COLORS.get(levelname, '')
        icon = self.ICONS.get(levelname, '')
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Format module name (court)
        module = record.name.split('.')[-1][:15].ljust(15)
        
        # Message
        message = record.getMessage()
        
        # Format final
        log_fmt = f"{color}{timestamp} | {icon} {module} | {levelname:<8} | {message}{Style.RESET_ALL}"
        
        # Exception
        if record.exc_info:
            log_fmt += "\n" + self.formatException(record.exc_info)
        
        return log_fmt

class StructuredFormatter(logging.Formatter):
    """Formatter JSON pour fichiers"""
    
    def format(self, record):
        import json
        
        log_dict = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.name,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        if record.exc_info:
            log_dict['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_dict)

def setup_logger(
    name: str,
    config: Optional[dict] = None
) -> logging.Logger:
    """
    Configure et retourne un logger
    
    Args:
        name: Nom du logger
        config: Configuration (optionnel, utilise CONFIG global si None)
    
    Returns:
        Logger configuré
    """
    
    # Importer config si non fournie
    if config is None:
        from config import ConfigLoader
        config_loader = ConfigLoader()
        config = config_loader.config['logging']
    
    # Créer logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config['level']))
    
    # Éviter duplication des handlers
    if logger.handlers:
        return logger
    
    # Handler Console
    if config.get('console_output', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        if config.get('colored_output', True):
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
        
        logger.addHandler(console_handler)
    
    # Handler Fichier
    if config.get('save_to_file', True):
        # Créer dossier logs
        log_dir = Path(config.get('log_directory', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Fichier principal
        log_file = log_dir / f"{name.lower()}_{datetime.now():%Y%m%d}.log"
        
        # Rotation
        rotation_type = config.get('rotation', 'daily')
        max_size_mb = config.get('max_file_size_mb', 10)
        backup_count = config.get('backup_count', 10)
        
        if rotation_type == 'daily':
            file_handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=backup_count,
                encoding='utf-8'
            )
        elif rotation_type == 'size-based':
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        file_handler.setLevel(logging.DEBUG)
        
        # Format fichier
        if config.get('format', 'text') == 'json':
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s\n'
                    '  └─ %(pathname)s:%(lineno)d in %(funcName)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
        
        logger.addHandler(file_handler)
    
    return logger

class TradingLogger:
    """Logger spécialisé pour trading avec méthodes utilitaires

    Délègue automatiquement vers le logger standard pour les méthodes de base
    (info, debug, warning, error, critical) tout en ajoutant des méthodes spécialisées.
    """

    def __init__(self, name: str = "TradingLogger"):
        self.logger = setup_logger(name)
        self.name = name

    # Délégation vers logger standard
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)

    def trade_opened(
        self,
        side: str = "",
        symbol: str = "",
        size: float = 0.0,
        price: float = 0.0,
        strategy: str = "",
        order_id: str = ""
    ):
        """Log ouverture de trade

        Args:
            side: 'BUY' ou 'SELL' (ou 'long'/'short')
            symbol: Paire de trading
            size: Taille de la position
            price: Prix d'entrée
            strategy: Nom de la stratégie (optionnel)
            order_id: ID de l'ordre (optionnel)
        """
        # Normaliser side
        side_upper = side.upper()
        side_display = '🟢 LONG' if side_upper in ['BUY', 'LONG'] else '🔴 SHORT'

        msg = f"{side_display} {symbol} | Size: {size:.6f} | Price: ${price:.2f}"
        if strategy:
            msg += f" | Strategy: {strategy}"
        if order_id:
            msg += f" | Order: {order_id}"

        self.logger.info(msg)

    def trade_closed(
        self,
        symbol: str,
        pnl: float,
        pnl_percent: float,
        duration: str = "",
        side: str = "",
        size: float = 0.0,
        entry_price: float = 0.0,
        exit_price: float = 0.0
    ):
        """Log fermeture de trade"""
        emoji = "💰" if pnl > 0 else "💸"
        color = Fore.GREEN if pnl > 0 else Fore.RED

        # Format de base
        msg = f"{emoji} {color}CLOSED {symbol}"

        # Ajouter side si fourni
        if side:
            side_display = "LONG" if side.upper() in ['BUY', 'LONG'] else "SHORT"
            msg += f" {side_display}"

        msg += f" | PnL: ${pnl:+.2f} ({pnl_percent:+.2f}%)"

        # Ajouter détails si fournis
        if entry_price > 0 and exit_price > 0:
            msg += f" | Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}"

        if duration:
            msg += f" | Duration: {duration}"

        msg += Style.RESET_ALL

        self.logger.info(msg)

    def stop_loss_hit(self, symbol: str, loss: float):
        """Log stop loss"""
        self.logger.warning(
            f"🛑 STOP LOSS {symbol} | Loss: ${loss:.2f}"
        )

    def take_profit_hit(self, symbol: str, profit: float, level: int):
        """Log take profit"""
        self.logger.info(
            f"🎯 TAKE PROFIT {level} {symbol} | Profit: ${profit:.2f}"
        )

    def signal_detected(self, signal_type: str, symbol: str, confidence: float):
        """Log signal détecté"""
        self.logger.info(
            f"📊 SIGNAL {signal_type} {symbol} | Confidence: {confidence:.1%}"
        )

    def order_executed(self, order_type: str, side: str, symbol: str, filled: float):
        """Log ordre exécuté"""
        self.logger.debug(
            f"✅ ORDER {order_type} {side} {symbol} | Filled: {filled}"
        )

    def error_critical(self, error_msg: str, exception: Optional[Exception] = None):
        """Log erreur critique"""
        self.logger.critical(
            f"🚨 CRITICAL ERROR: {error_msg}",
            exc_info=exception
        )

    def daily_summary(
        self,
        total_trades: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
        win_rate: float = 0.0,
        total_pnl: float = 0.0,
        balance: float = 0.0,
        # Compatibilité avec ancienne signature
        trades: int = None,
        pnl: float = None
    ):
        """Log résumé journalier

        Args:
            total_trades: Nombre total de trades
            winning_trades: Nombre de trades gagnants
            losing_trades: Nombre de trades perdants
            win_rate: Taux de réussite (0.0 à 1.0)
            total_pnl: PnL total
            balance: Balance actuelle (optionnel)
            trades: Alias pour total_trades (rétrocompatibilité)
            pnl: Alias pour total_pnl (rétrocompatibilité)
        """
        # Rétrocompatibilité
        if trades is not None:
            total_trades = trades
        if pnl is not None:
            total_pnl = pnl

        self.logger.info(
            f"\n{'='*70}\n"
            f"📊 DAILY SUMMARY\n"
            f"   Total Trades: {total_trades}\n"
            f"   Winning: {winning_trades} | Losing: {losing_trades}\n"
            f"   Win Rate: {win_rate:.1%}\n"
            f"   Total PnL: ${total_pnl:+.2f}\n"
            f"   Balance: ${balance:.2f}\n"
            f"{'='*70}"
        )
    
    def performance_metrics(self, metrics: dict):
        """Log métriques de performance"""
        self.logger.info(
            f"\n{'='*70}\n"
            f"📈 PERFORMANCE METRICS\n"
            f"   Sharpe Ratio: {metrics.get('sharpe', 0):.2f}\n"
            f"   Max Drawdown: {metrics.get('max_dd', 0):.2%}\n"
            f"   Profit Factor: {metrics.get('pf', 0):.2f}\n"
            f"   Total Trades: {metrics.get('trades', 0)}\n"
            f"   Win Rate: {metrics.get('win_rate', 0):.1%}\n"
            f"{'='*70}"
        )
    
    def system_status(self, status: str, details: str = ""):
        """Log statut système"""
        icons = {
            'starting': '🚀',
            'running': '✅',
            'paused': '⏸️',
            'stopping': '🛑',
            'error': '❌',
        }
        
        icon = icons.get(status.lower(), 'ℹ️')
        self.logger.info(f"{icon} SYSTEM {status.upper()} {details}")

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def setup_trading_logger(name: str = "TradingLogger") -> TradingLogger:
    """
    Crée et retourne un TradingLogger avec méthodes spécialisées

    Args:
        name: Nom du logger

    Returns:
        TradingLogger avec méthodes info(), trade_opened(), daily_summary(), etc.
    """
    return TradingLogger(name)


def log_banner(title: str, logger: logging.Logger):
    """Affiche une bannière dans les logs"""
    width = 70
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)

def log_dict(data: dict, logger: logging.Logger, title: str = "Data"):
    """Log un dictionnaire de manière formatée"""
    logger.debug(f"\n{title}:")
    for key, value in data.items():
        logger.debug(f"  {key}: {value}")

# Export
__all__ = [
    'setup_logger',
    'setup_trading_logger',
    'TradingLogger',
    'ColoredFormatter',
    'StructuredFormatter',
    'log_banner',
    'log_dict',
]
