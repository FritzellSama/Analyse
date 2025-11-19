"""
Base Client Interface - Quantum Trader Pro
Interface abstraite définissant le contrat commun pour tous les clients exchange
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from utils.validators import validate_price as _validate_price_util


class BaseExchangeClient(ABC):
    """
    Interface abstraite pour les clients exchange.
    Tous les clients (Production, Testnet, Virtual) doivent implémenter ces méthodes
    avec les mêmes signatures et types de retour.
    """

    @abstractmethod
    def get_ticker(self) -> Dict:
        """
        Récupère le ticker actuel pour le symbole configuré

        Returns:
            Dict avec:
                - symbol: str
                - bid: float
                - ask: float
                - last: float
                - spread: float (ask - bid)
                - spread_percent: float
                - volume: float
                - timestamp: datetime ou int (ms)
        """
        pass

    @abstractmethod
    def get_balance(self) -> Dict:
        """
        Récupère le solde du compte

        Returns:
            Dict avec:
                - base: Dict avec 'free', 'used', 'total' (en crypto base, ex: BTC)
                - quote: Dict avec 'free', 'used', 'total' (en devise, ex: USDT)
        """
        pass

    @abstractmethod
    def fetch_ohlcv(
        self,
        timeframe: str = '5m',
        limit: int = 500,
        since: Optional[int] = None
    ) -> List[List]:
        """
        Récupère les données OHLCV

        Args:
            timeframe: Timeframe ('1m', '5m', '1h', etc.)
            limit: Nombre de bougies
            since: Timestamp de début (ms) - optionnel

        Returns:
            Liste de listes: [[timestamp_ms, open, high, low, close, volume], ...]
        """
        pass

    @abstractmethod
    def create_order(
        self,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Crée un ordre

        Args:
            side: 'buy' ou 'sell' (lowercase)
            order_type: 'market' ou 'limit' (lowercase)
            amount: Quantité en base asset
            price: Prix (requis pour limit orders)
            params: Paramètres additionnels optionnels

        Returns:
            Dict avec:
                - id: str (ID de l'ordre)
                - symbol: str
                - side: str
                - type: str
                - amount: float
                - price: float
                - filled: float
                - remaining: float
                - status: str ('open', 'closed', 'canceled')
                - timestamp: int (ms)
                - cost: float (total cost)
                - fee: Dict avec 'cost' et 'currency'
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Annule un ordre

        Args:
            order_id: ID de l'ordre à annuler

        Returns:
            True si annulation réussie, False sinon
        """
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Dict]:
        """
        Récupère les ordres ouverts

        Returns:
            Liste de Dict représentant les ordres ouverts
        """
        pass

    @abstractmethod
    def get_order_book(self, limit: int = 20) -> Dict:
        """
        Récupère l'order book

        Args:
            limit: Profondeur de l'order book

        Returns:
            Dict avec:
                - bids: List[[price, amount], ...]
                - asks: List[[price, amount], ...]
                - timestamp: int (ms) ou datetime
        """
        pass

    @abstractmethod
    def test_connectivity(self) -> bool:
        """
        Teste la connectivité avec l'exchange

        Returns:
            True si connecté, False sinon
        """
        pass

    @abstractmethod
    def close_position(self, position_side: str = 'long') -> bool:
        """
        Ferme une position

        Args:
            position_side: 'long' ou 'short'

        Returns:
            True si succès
        """
        pass

    # Méthodes optionnelles avec implémentation par défaut

    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """
        Récupère les trades récents

        Args:
            limit: Nombre de trades

        Returns:
            Liste de Dict représentant les trades
        """
        return []

    def validate_price(self, price: float, price_type: str = "price") -> bool:
        """
        Valide qu'un prix est acceptable (utilise fonction centralisée)

        Args:
            price: Prix à valider
            price_type: Type pour logging

        Returns:
            True si valide
        """
        return _validate_price_util(price, price_type)

    def fetch_historical(
        self,
        timeframe: str = '5m',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Récupère données historiques complètes avec pagination

        Args:
            timeframe: Timeframe
            start_date: Date de début
            end_date: Date de fin
            limit: Bougies par batch

        Returns:
            DataFrame avec colonnes [open, high, low, close, volume] et index timestamp
        """
        raise NotImplementedError("fetch_historical non implémenté pour ce client")

    def reconnect(self) -> bool:
        """
        Tente de reconnecter à l'exchange

        Returns:
            True si reconnexion réussie
        """
        return self.test_connectivity()


class ClientType:
    """Énumération des types de clients"""
    PRODUCTION = 'production'
    TESTNET = 'testnet'
    VIRTUAL = 'virtual'


def create_client(config: Dict) -> BaseExchangeClient:
    """
    Factory pour créer le bon type de client selon la config

    Args:
        config: Configuration complète

    Returns:
        Instance de client appropriée
    """
    from utils.logger import setup_logger
    logger = setup_logger('ClientFactory')

    # Déterminer le mode
    if config.get('backtest', {}).get('enabled', False):
        # Mode backtest = Virtual client
        logger.info("🔄 Mode BACKTEST - Utilisation VirtualBinanceClient")
        from core.virtual_binance_client import VirtualBinanceClient
        return VirtualBinanceClient(config)

    elif config.get('exchange', {}).get('primary', {}).get('testnet', False):
        # Mode testnet
        logger.info("🔄 Mode TESTNET - Utilisation BinanceTestnetClient")
        from core.binance_testnet_client import BinanceTestnetClient
        return BinanceTestnetClient(config)

    else:
        # Mode production
        logger.warning("⚠️ Mode PRODUCTION - ARGENT RÉEL!")
        from core.binance_client import BinanceClient
        return BinanceClient(config)


__all__ = ['BaseExchangeClient', 'ClientType', 'create_client']
