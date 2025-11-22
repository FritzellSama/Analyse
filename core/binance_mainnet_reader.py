"""
Binance Mainnet Reader - Quantum Trader Pro
Client lecture seule pour récupérer des données historiques depuis Binance mainnet
Pas besoin de clés API pour les données publiques (OHLCV)
"""

import requests
import time
from typing import Dict, Optional, List
from utils.logger import setup_logger


class BinanceMainnetReader:
    """
    Client Binance Mainnet en lecture seule.
    Utilisé uniquement pour récupérer des données historiques (OHLCV).
    Pas besoin d'authentification pour les endpoints publics.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('BinanceMainnetReader')

        # URL mainnet (données réelles)
        self.base_url = 'https://api.binance.com'

        # Symbol configuration
        self.symbol_raw = config.get('symbols', {}).get('primary', 'BTC/USDT')
        self.symbol = self.symbol_raw.replace('/', '')  # BTCUSDT

        self.rateLimit = 50

        self.logger.info(f"✅ Binance Mainnet Reader initialisé (lecture seule)")
        self.logger.info(f"   URL: {self.base_url}")
        self.logger.info(f"   Symbol: {self.symbol_raw}")

    def fetch_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = '5m',
        limit: int = 1000,
        since: Optional[int] = None
    ) -> List[List]:
        """
        Récupère les données OHLCV depuis Binance mainnet

        Args:
            symbol: Symbol (ignoré, utilise self.symbol)
            timeframe: Timeframe ('1m', '5m', '1h', etc.)
            limit: Nombre de bougies (max 1000)
            since: Timestamp de début (ms)

        Returns:
            Liste de [timestamp_ms, open, high, low, close, volume]
        """
        try:
            # Convertir timeframe
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
            }

            params = {
                'symbol': self.symbol,
                'interval': interval_map.get(timeframe, '5m'),
                'limit': min(limit, 1000)  # Max 1000 par requête
            }

            if since is not None:
                params['startTime'] = since

            response = requests.get(
                f"{self.base_url}/api/v3/klines",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            klines = response.json()

            # Format OHLCV standard
            ohlcv = []
            for k in klines:
                ohlcv.append([
                    k[0],           # timestamp
                    float(k[1]),    # open
                    float(k[2]),    # high
                    float(k[3]),    # low
                    float(k[4]),    # close
                    float(k[5])     # volume
                ])

            return ohlcv

        except Exception as e:
            self.logger.error(f"❌ Erreur fetch_ohlcv mainnet: {e}")
            return []

    def get_ticker(self) -> Dict:
        """
        Récupère le ticker (lecture seule)
        """
        try:
            params = {'symbol': self.symbol}
            response = requests.get(
                f"{self.base_url}/api/v3/ticker/24hr",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            ticker = response.json()

            return {
                'symbol': self.symbol_raw,
                'bid': float(ticker.get('bidPrice', 0)),
                'ask': float(ticker.get('askPrice', 0)),
                'last': float(ticker.get('lastPrice', 0)),
                'volume': float(ticker.get('volume', 0)),
                'timestamp': ticker.get('closeTime', int(time.time() * 1000))
            }
        except Exception as e:
            self.logger.error(f"❌ Erreur get_ticker mainnet: {e}")
            return {}

    def test_connectivity(self) -> bool:
        """
        Teste la connexion au mainnet
        """
        try:
            response = requests.get(f"{self.base_url}/api/v3/ping", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    # Méthodes non supportées (lecture seule)
    def get_balance(self, currency: Optional[str] = None) -> Dict:
        """Non supporté - lecture seule"""
        self.logger.warning("⚠️ get_balance non supporté en mode lecture seule")
        return {'base': {'free': 0, 'used': 0, 'total': 0},
                'quote': {'free': 0, 'used': 0, 'total': 0}}

    def create_order(self, *args, **kwargs):
        """Non supporté - lecture seule"""
        raise NotImplementedError("Mode lecture seule - pas de création d'ordres")

    def cancel_order(self, *args, **kwargs):
        """Non supporté - lecture seule"""
        raise NotImplementedError("Mode lecture seule - pas d'annulation d'ordres")


__all__ = ['BinanceMainnetReader']
