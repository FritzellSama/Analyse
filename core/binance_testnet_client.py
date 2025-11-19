"""
Binance Testnet Client avec support RSA - Quantum Trader Pro
Client spécifique pour le testnet Binance qui utilise RSA au lieu de HMAC
"""

import ccxt
import base64
import time
import json
import pandas as pd
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
import requests
from urllib.parse import urlencode
from utils.logger import setup_logger
from core.base_client import BaseExchangeClient
from utils.safe_math import safe_divide


class BinanceTestnetClient(BaseExchangeClient):
    """
    Client Binance Testnet avec authentification RSA
    Implémente BaseExchangeClient pour garantir la compatibilité API
    """

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('BinanceTestnetClient')

        # Configuration - avec fallbacks sécurisés
        exchange_config = config.get('exchange', {}).get('primary', {})
        self.api_key = exchange_config.get('api_key', '')
        self.private_key_path = exchange_config.get('private_key_path', 'test-prv-key.pem')
        self.base_url = 'https://testnet.binance.vision'

        # Symbol configuration - avec parsing sécurisé
        self.symbol_raw = config.get('symbols', {}).get('primary', 'BTC/USDT')
        self.symbol = self.symbol_raw.replace('/', '')  # BTCUSDT
        if '/' in self.symbol_raw:
            self.base, self.quote = self.symbol_raw.split('/')
        else:
            self.base, self.quote = 'BTC', 'USDT'

        # Décalage temps serveur (sera calculé dynamiquement)
        self.time_offset = 0

        # Charger la clé privée RSA
        self._load_private_key()

        # Calculer le décalage temps
        self._sync_time()

        # Pour compatibilité avec l'ancien code
        self.exchange = self  # Permet d'utiliser client.exchange.fetch_xxx

        self.logger.info(f"✅ Binance Testnet Client initialisé (RSA mode)")

        self.rateLimit = 50

    def _load_private_key(self):
        """Charge la clé privée RSA depuis le fichier PEM"""
        try:
            private_key_path = Path(self.private_key_path)

            if not private_key_path.exists():
                # Générer les clés si elles n'existent pas
                self.logger.warning(f"⚠️ Clé privée non trouvée: {private_key_path}")
                self.logger.info("🔑 Génération des clés RSA...")
                self._generate_rsa_keys()

            with open(self.private_key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
            self.logger.info(f"🔑 Clé privée RSA chargée: {self.private_key_path}")

        except Exception as e:
            self.logger.error(f"❌ Erreur chargement clé RSA: {e}")
            raise

    def _generate_rsa_keys(self):
        """Génère une paire de clés RSA pour le testnet"""
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa

            # Générer la paire de clés
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )

            # Sauvegarder la clé privée
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            with open(self.private_key_path, 'wb') as f:
                f.write(private_pem)

            # Générer et sauvegarder la clé publique
            public_key = private_key.public_key()
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            public_key_path = self.private_key_path.replace('-prv-', '-pub-')
            with open(public_key_path, 'wb') as f:
                f.write(public_pem)

            self.logger.info(f"✅ Clés RSA générées:")
            self.logger.info(f"   - Privée: {self.private_key_path}")
            self.logger.info(f"   - Publique: {public_key_path}")
            self.logger.warning(
                f"⚠️  IMPORTANT: Enregistrez la clé publique dans votre compte Binance Testnet!"
            )

            # Sauvegarder dans self pour utilisation immédiate
            self.private_key = private_key

        except Exception as e:
            self.logger.error(f"❌ Erreur génération clés RSA: {e}")
            raise

    def _sync_time(self):
        """Synchronise le timestamp avec le serveur Binance"""
        try:
            # Obtenir le temps serveur
            response = requests.get(f"{self.base_url}/api/v3/time")
            server_time = response.json()['serverTime']
            local_time = int(time.time() * 1000)

            # Calculer le décalage (server - local)
            self.time_offset = server_time - local_time

            if abs(self.time_offset) > 1000:  # Plus de 1 seconde
                self.logger.warning(
                    f"⚠️ Décalage temps détecté: {self.time_offset}ms. "
                    f"Compensation automatique activée."
                )
            else:
                self.logger.info(f"✅ Temps synchronisé (décalage: {self.time_offset}ms)")

        except Exception as e:
            self.logger.warning(f"⚠️ Impossible de synchroniser le temps: {e}")
            self.time_offset = 0

    def _get_timestamp(self) -> int:
        """Retourne le timestamp corrigé avec le décalage serveur"""
        return int(time.time() * 1000) + self.time_offset

    def _sign_request(self, params: str) -> str:
        """
        Signe les paramètres avec RSA-SHA256

        Args:
            params: Paramètres de la requête

        Returns:
            Signature en base64
        """
        # Signer avec RSA-SHA256
        signature = self.private_key.sign(
            params.encode('utf-8'),
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        # Encoder en base64
        return base64.b64encode(signature).decode('utf-8')

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = True
    ) -> Any:
        """
        Fait une requête à l'API Binance Testnet

        Args:
            method: GET, POST, DELETE
            endpoint: Endpoint API (ex: /api/v3/order)
            params: Paramètres de la requête
            signed: Si True, signe la requête

        Returns:
            Réponse JSON
        """
        if params is None:
            params = {}

        # Ajouter timestamp si signé
        if signed:
            params['timestamp'] = self._get_timestamp()

        # Créer query string
        query_string = urlencode(params)

        # Signer si nécessaire
        if signed:
            signature = self._sign_request(query_string)
            query_string += f"&signature={signature}"

        # URL complète
        url = f"{self.base_url}{endpoint}"
        if query_string:
            url += f"?{query_string}"

        # Headers
        headers = {
            'X-MBX-APIKEY': self.api_key
        }

        # Faire la requête
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Méthode non supportée: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Erreur requête API: {e}")
            if hasattr(e.response, 'text'):
                self.logger.error(f"Réponse: {e.response.text}")
            raise

    # ========================================================================
    # MÉTHODES ABSTRAITES DE BaseExchangeClient
    # ========================================================================

    def get_ticker(self) -> Dict:
        """
        Récupère le ticker
        Implémente BaseExchangeClient.get_ticker()

        Returns:
            Dict standardisé avec infos ticker
        """
        try:
            params = {'symbol': self.symbol}
            ticker = self._make_request('GET', '/api/v3/ticker/24hr', params, signed=False)

            last_price = float(ticker.get('lastPrice', 0))
            bid = float(ticker.get('bidPrice', 0))
            ask = float(ticker.get('askPrice', 0))
            spread = ask - bid

            return {
                'symbol': self.symbol_raw,
                'bid': bid,
                'ask': ask,
                'last': last_price,
                'spread': spread,
                'spread_percent': safe_divide(spread, last_price, default=0.0) * 100,
                'volume': float(ticker.get('volume', 0)),
                'timestamp': ticker.get('closeTime', int(time.time() * 1000))
            }
        except Exception as e:
            self.logger.error(f"❌ Erreur get_ticker: {e}")
            return {
                'symbol': self.symbol_raw,
                'bid': 0,
                'ask': 0,
                'last': 0,
                'spread': 0,
                'spread_percent': 0,
                'volume': 0,
                'timestamp': int(time.time() * 1000)
            }

    def get_balance(self, currency: Optional[str] = None) -> Dict:
        """
        Récupère la balance du compte
        Implémente BaseExchangeClient.get_balance()

        Returns:
            Dict standardisé avec 'base' et 'quote' keys
        """
        try:
            account = self._make_request('GET', '/api/v3/account')

            if currency:
                for balance in account.get('balances', []):
                    if balance['asset'] == currency:
                        free = float(balance['free'])
                        locked = float(balance['locked'])
                        return {
                            'free': free,
                            'used': locked,
                            'total': free + locked
                        }

                if hasattr(self, 'testnet') and self.testnet:
                    if quote_balance['free'] == 0.0:
                        quote_balance = {'free': 10000.0, 'used': 0.0, 'total': 10000.0}
                        self.logger.warning("⚠️ Balance testnet = 0, utilisation valeur par défaut: 10000 USDT")

                return {
                    'base': base_balance,
                    'quote': quote_balance
                }

            base_balance = {'free': 0.0, 'used': 0.0, 'total': 0.0}
            quote_balance = {'free': 0.0, 'used': 0.0, 'total': 0.0}

            for balance in account.get('balances', []):
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked

                if asset == self.base:
                    base_balance = {
                        'free': free,
                        'used': locked,
                        'total': total
                    }
                elif asset == self.quote:
                    quote_balance = {
                        'free': free,
                        'used': locked,
                        'total': total
                    }

            return {
                'base': base_balance,
                'quote': quote_balance
            }

        except Exception as e:
            self.logger.error(f"❌ Erreur get_balance: {e}")
            return {
                'base': {'free': 0.0, 'used': 0.0, 'total': 0.0},
                'quote': {'free': 10000.0, 'used': 0.0, 'total': 10000.0}  # Default testnet
            }

    def fetch_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = '5m',
        limit: int = 500,
        since: Optional[int] = None
    ) -> List[List]:
        """
        Récupère les données OHLCV
        Implémente BaseExchangeClient.fetch_ohlcv()

        Args:
            timeframe: Timeframe ('1m', '5m', '1h', etc.)
            limit: Nombre de bougies
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
                'interval': interval_map.get(timeframe, '1h'),
                'limit': limit
            }

            if since is not None:
                params['startTime'] = since

            klines = self._make_request('GET', '/api/v3/klines', params, signed=False)

            # Format OHLCV standard
            ohlcv = []
            for k in klines:
                ohlcv.append([
                    k[0],  # timestamp
                    float(k[1]),  # open
                    float(k[2]),  # high
                    float(k[3]),  # low
                    float(k[4]),  # close
                    float(k[5])   # volume
                ])

            return ohlcv

        except Exception as e:
            self.logger.error(f"❌ Erreur fetch_ohlcv: {e}")
            return []

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
        Implémente BaseExchangeClient.create_order()

        Args:
            side: 'buy' ou 'sell' (lowercase)
            order_type: 'market' ou 'limit'
            amount: Quantité
            price: Prix (pour LIMIT)
            params: Paramètres additionnels

        Returns:
            Dict standardisé représentant l'ordre
        """
        try:
            order_params = {
                'symbol': self.symbol,
                'side': side.upper(),
                'type': order_type.upper(),
                'quantity': amount
            }

            if order_type.upper() == 'LIMIT':
                order_params['timeInForce'] = 'GTC'
                order_params['price'] = round(price, 2)

            # Merge params additionnels
            if params:
                order_params.update(params)

            order = self._make_request('POST', '/api/v3/order', order_params)

            # Calculer le coût et les frais
            fills = order.get('fills', [])
            total_cost = 0.0
            total_fee = 0.0
            fee_currency = self.quote

            for fill in fills:
                total_cost += float(fill['price']) * float(fill['qty'])
                total_fee += float(fill['commission'])
                fee_currency = fill['commissionAsset']

            if not fills:
                # Si pas de fills, estimer
                total_cost = amount * (price or 0)
                total_fee = total_cost * 0.001

            return {
                'id': str(order['orderId']),
                'symbol': self.symbol_raw,
                'side': side.lower(),
                'type': order_type.lower(),
                'amount': amount,
                'price': price or float(order.get('price', 0)),
                'filled': float(order.get('executedQty', 0)),
                'remaining': amount - float(order.get('executedQty', 0)),
                'status': order['status'].lower(),
                'timestamp': order['transactTime'],
                'cost': total_cost,
                'fee': {'cost': total_fee, 'currency': fee_currency}
            }

        except Exception as e:
            self.logger.error(f"❌ Erreur create_order: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """
        Annule un ordre
        Implémente BaseExchangeClient.cancel_order()

        Args:
            order_id: ID de l'ordre

        Returns:
            True si annulé
        """
        try:
            params = {
                'symbol': self.symbol,
                'orderId': order_id
            }

            self._make_request('DELETE', '/api/v3/order', params)
            self.logger.info(f"🗑️ Ordre annulé: {order_id}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Erreur cancel_order: {e}")
            return False

    def get_open_orders(self) -> List[Dict]:
        """
        Récupère les ordres ouverts
        Implémente BaseExchangeClient.get_open_orders()

        Returns:
            Liste de Dict représentant les ordres
        """
        try:
            params = {'symbol': self.symbol}
            orders = self._make_request('GET', '/api/v3/openOrders', params)

            result = []
            for order in orders:
                result.append({
                    'id': str(order['orderId']),
                    'symbol': self.symbol_raw,
                    'side': order['side'].lower(),
                    'type': order['type'].lower(),
                    'amount': float(order['origQty']),
                    'price': float(order['price']),
                    'filled': float(order['executedQty']),
                    'remaining': float(order['origQty']) - float(order['executedQty']),
                    'status': order['status'].lower(),
                    'timestamp': order['time']
                })

            return result

        except Exception as e:
            self.logger.error(f"❌ Erreur get_open_orders: {e}")
            return []

    def get_order_book(self, limit: int = 20) -> Dict:
        """
        Récupère l'orderbook
        Implémente BaseExchangeClient.get_order_book()

        Args:
            limit: Profondeur

        Returns:
            Dict standardisé avec bids et asks
        """
        try:
            params = {
                'symbol': self.symbol,
                'limit': limit
            }

            book = self._make_request('GET', '/api/v3/depth', params, signed=False)

            return {
                'bids': [[float(p), float(q)] for p, q in book.get('bids', [])],
                'asks': [[float(p), float(q)] for p, q in book.get('asks', [])],
                'timestamp': int(time.time() * 1000)
            }
        except Exception as e:
            self.logger.error(f"❌ Erreur get_order_book: {e}")
            return {'bids': [], 'asks': [], 'timestamp': int(time.time() * 1000)}

    def test_connectivity(self) -> bool:
        """
        Teste la connexion
        Implémente BaseExchangeClient.test_connectivity()

        Returns:
            True si connecté
        """
        try:
            self._make_request('GET', '/api/v3/ping', signed=False)
            return True
        except Exception as e:
            self.logger.debug(f"Ping échoué: {e}")
            return False

    def close_position(self, position_side: str = 'long') -> bool:
        """
        Ferme une position
        Implémente BaseExchangeClient.close_position()

        Args:
            position_side: 'long' ou 'short'

        Returns:
            True si succès
        """
        try:
            balance = self.get_balance()

            if position_side == 'long':
                # Vendre tout le base asset
                amount = balance['base']['free']
                if amount > 0:
                    self.create_order('sell', 'market', amount)
                    self.logger.info(f"🔒 Position LONG fermée: {amount} {self.base}")
                    return True
            else:
                # Short non supporté en spot
                self.logger.warning("⚠️ Close short non implémenté pour spot testnet")
                return False

            return False

        except Exception as e:
            self.logger.error(f"❌ Erreur close_position: {e}")
            return False

    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """
        Récupère les trades récents

        Args:
            limit: Nombre de trades

        Returns:
            Liste de trades
        """
        try:
            params = {
                'symbol': self.symbol,
                'limit': limit
            }
            trades = self._make_request('GET', '/api/v3/trades', params, signed=False)
            return trades
        except Exception as e:
            self.logger.error(f"❌ Erreur get_recent_trades: {e}")
            return []

    def fetch_historical(
        self,
        timeframe: str = '5m',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Récupère données historiques complètes

        Args:
            timeframe: Timeframe
            start_date: Date de début
            end_date: Date de fin
            limit: Bougies par batch

        Returns:
            DataFrame avec colonnes [open, high, low, close, volume]
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)

        if end_date is None:
            end_date = datetime.now()

        self.logger.info(
            f"📥 Téléchargement historique {timeframe} "
            f"du {start_date:%Y-%m-%d} au {end_date:%Y-%m-%d}"
        )

        all_data = []
        current_since = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

        batch_num = 0

        while current_since < end_timestamp:
            try:
                ohlcv = self.fetch_ohlcv(self.symbol, timeframe, limit, current_since)

                if not ohlcv:
                    break

                all_data.extend(ohlcv)

                # Mise à jour curseur
                current_since = ohlcv[-1][0] + 1

                batch_num += 1
                if batch_num % 10 == 0:
                    self.logger.info(f"  ↓ {len(all_data)} bougies téléchargées...")

                # Sécurité: limite max 100k bougies
                if len(all_data) > 100000:
                    self.logger.warning("⚠️ Limite 100k bougies atteinte")
                    break

                # Si moins de 'limit' reçu = fin des données
                if len(ohlcv) < limit:
                    break

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"❌ Erreur fetch historique batch {batch_num}: {e}")
                break

        # Conversion DataFrame
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
        df = df.sort_index()

        # Filtrer par date de fin
        df = df[df.index <= end_date]

        self.logger.info(f"✅ {len(df)} bougies chargées")

        return df

    # ========================================================================
    # MÉTHODES DE COMPATIBILITÉ CCXT (pour ancien code)
    # ========================================================================

    def fetch_balance(self):
        """Compatibilité CCXT"""
        account = self._make_request('GET', '/api/v3/account')
        balances = {}
        for b in account.get('balances', []):
            if float(b['free']) > 0 or float(b['locked']) > 0:
                balances[b['asset']] = {
                    'free': float(b['free']),
                    'used': float(b['locked']),
                    'total': float(b['free']) + float(b['locked'])
                }
        return {'info': account, 'free': balances, 'used': {}, 'total': balances}

    def fetch_ticker(self, symbol):
        """Compatibilité CCXT"""
        return self.get_ticker()

    def fetch_time(self):
        """Récupère le temps serveur pour compatibilité CCXT"""
        try:
            response = self._make_request('GET', '/api/v3/time', signed=False)
            return response.get('serverTime', int(time.time() * 1000))
        except Exception as e:
            self.logger.error(f"❌ Erreur fetch_time: {e}")
            return int(time.time() * 1000)

    def fetch_order(self, order_id: str, symbol: str):
        """Récupère les détails d'un ordre"""
        try:
            params = {
                'symbol': symbol.replace('/', ''),
                'orderId': order_id
            }
            return self._make_request('GET', '/api/v3/order', params)
        except Exception as e:
            self.logger.error(f"❌ Erreur fetch_order: {e}")
            return {}

    def fetch_open_orders(self, symbol: Optional[str] = None):
        """Alias pour get_open_orders"""
        return self.get_open_orders()

    def fetch_order_book(self, symbol: str, limit: int = 20):
        """Compatibilité CCXT pour fetch_order_book"""
        return self.get_order_book(limit)

    def load_markets(self):
        """Charge les marchés (compatibilité CCXT)"""
        try:
            exchange_info = self._make_request('GET', '/api/v3/exchangeInfo', signed=False)
            self.markets = {}
            for symbol_info in exchange_info.get('symbols', []):
                if symbol_info['status'] == 'TRADING':
                    base = symbol_info['baseAsset']
                    quote = symbol_info['quoteAsset']
                    symbol = f"{base}/{quote}"
                    self.markets[symbol] = symbol_info
            return self.markets
        except Exception as e:
            self.logger.error(f"❌ Erreur load_markets: {e}")
            return {}

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params=None):
        """Crée un ordre limit (compatible CCXT)"""
        if side.lower() == 'buy':
            return self.create_limit_buy_order(symbol, amount, price)
        else:
            return self.create_limit_sell_order(symbol, amount, price)

    def create_market_order(self, symbol: str, side: str, amount: float, params=None):
        """Crée un ordre market (compatible CCXT)"""
        if side.lower() == 'buy':
            return self.create_market_buy_order(symbol, amount)
        else:
            return self.create_market_sell_order(symbol, amount)

    def close(self):
        """Ferme la connexion (compatibilité CCXT)"""
        pass  # Pas de connexion persistante à fermer

    def create_limit_buy_order(self, symbol, amount, price):
        """Compatibilité CCXT"""
        return self.create_order('buy', 'limit', amount, price)

    def create_limit_sell_order(self, symbol, amount, price):
        """Compatibilité CCXT"""
        return self.create_order('sell', 'limit', amount, price)

    def create_market_buy_order(self, symbol, amount):
        """Compatibilité CCXT"""
        return self.create_order('buy', 'market', amount)

    def create_market_sell_order(self, symbol, amount):
        """Compatibilité CCXT"""
        return self.create_order('sell', 'market', amount)

    # Alias pour compatibilité ancienne API
    def get_orderbook(self, symbol: Optional[str] = None, limit: int = 20) -> Dict:
        """Alias pour get_order_book"""
        return self.get_order_book(limit)


__all__ = ['BinanceTestnetClient']
