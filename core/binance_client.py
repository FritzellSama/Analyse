"""
Binance Client - Quantum Trader Pro
Client CCXT amélioré avec reconnexion automatique et rate limiting intelligent
"""

import ccxt
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from utils.logger import setup_logger
from utils.validators import validate_price as _validate_price_util
import asyncio
from functools import wraps
from core.base_client import BaseExchangeClient

class BinanceConnectionError(Exception):
    """Erreur de connexion Binance"""
    pass

class BinanceClient(BaseExchangeClient):
    """
    Client Binance production-ready avec:
    - Support testnet/production
    - Rate limiting intelligent
    - Reconnexion automatique
    - Gestion d'erreurs robuste
    - Retry logic
    """

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('BinanceClient')

        # Paramètres exchange - avec fallbacks sécurisés
        exchange_config = config.get('exchange', {}).get('primary', {})
        self.api_key = exchange_config.get('api_key', '')
        self.secret_key = exchange_config.get('secret_key', '')
        self.testnet = exchange_config.get('testnet', False)
        self.timeout = exchange_config.get('timeout_seconds', 30)
        self.max_retries = exchange_config.get('retry_attempts', 3)

        # Symbol - avec fallback et parsing sécurisé
        self.symbol = config.get('symbols', {}).get('primary', 'BTC/USDT')
        if '/' in self.symbol:
            self.base, self.quote = self.symbol.split('/')
        else:
            self.base, self.quote = 'BTC', 'USDT'

        # Rate limiting
        self.rate_limit_buffer = exchange_config.get('rate_limit_buffer', 0.1)
        self.last_request_time = {}
        self.request_weights = {}

        # Price validation limits
        self.min_price = 0.00000001
        self.max_price = 1000000.0

        # Connexion tracking
        self.is_connected = False
        self.last_connection_attempt = None
        self.connection_errors = 0

        # Initialiser exchange
        self._initialize_exchange()

        # Vérifier connexion
        self._verify_connection()

    def _initialize_exchange(self):
        """Initialise l'objet exchange CCXT ou Testnet RSA"""

        self.logger.info(f"🔌 Initialisation Binance...")

        # Si testnet, utiliser le client RSA spécial
        if self.testnet:
            self.logger.info("🔐 MODE TESTNET AVEC RSA ACTIVÉ")
            try:
                from core.binance_testnet_client import BinanceTestnetClient

                # Passer le config complet au client testnet
                self.exchange = BinanceTestnetClient(self.config)
                self.connected = True
                self.logger.info("✅ Client Testnet RSA initialisé")
                return  # Sortir ici, pas besoin du reste

            except Exception as e:
                self.logger.error(f"❌ Erreur initialisation Testnet RSA: {e}")
                raise BinanceConnectionError(f"Impossible d'initialiser le client testnet: {e}")

        # Mode Production - utiliser CCXT normal
        self.logger.warning("⚠️  MODE PRODUCTION - ARGENT RÉEL!")

        # Configuration de base
        exchange_params = {
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'enableRateLimit': True,
            'timeout': self.timeout * 1000,  # ms
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 10000,
            }
        }

        try:
            self.exchange = ccxt.binance(exchange_params)
            self.connected = True
            self.logger.info("✅ Exchange CCXT Production initialisé")
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation: {e}")
            raise BinanceConnectionError(f"Impossible d'initialiser l'exchange: {e}")

    def _verify_connection(self):
        """Vérifie la connexion et les clés API"""

        self.logger.info("🔍 Vérification de la connexion...")

        try:
            # Test 1: Récupérer le ticker (public API)
            ticker = self.exchange.fetch_ticker(self.symbol)
            self.logger.info(f"✅ Test ticker OK: {self.symbol} @ ${ticker['last']:.2f}")

            # Test 2: Récupérer le temps serveur
            server_time = self.exchange.fetch_time()
            local_time = int(time.time() * 1000)
            time_diff = abs(server_time - local_time) / 1000

            self.logger.info(f"⏰ Décalage temps: {time_diff:.2f}s")

            if time_diff > 5:
                self.logger.warning(
                    f"⚠️  Décalage temps important ({time_diff:.2f}s)! "
                    "Synchronisez votre horloge système."
                )

            # Test 3: Test API privée (balance ou account)
            # Note: Sur testnet, certaines APIs privées ne marchent pas
            try:
                if self.testnet:
                    # Sur testnet, tester avec fetch_balance qui marche mieux
                    balance = self.exchange.fetch_balance()
                    self.logger.info(f"✅ Test API privée OK (Balance accessible)")
                else:
                    # En production, on peut utiliser fetch_balance
                    balance = self.exchange.fetch_balance()
                    quote_balance = balance.get(self.quote, {}).get('free', 0)
                    self.logger.info(f"💰 Balance {self.quote}: {quote_balance:.2f}")

            except ccxt.AuthenticationError as e:
                self.logger.error(f"❌ ERREUR AUTHENTIFICATION: {e}")
                self.logger.error(
                    "\n🔑 PROBLÈME DE CLÉS API DÉTECTÉ!\n"
                    "\n"
                    "Causes possibles:\n"
                    "1. Vous utilisez des clés PRODUCTION sur TESTNET (ou vice-versa)\n"
                    "2. Les clés sont incorrectes ou invalides\n"
                    "3. Les clés n'ont pas les permissions nécessaires\n"
                    "\n"
                    "Solutions:\n"
                    "- Pour TESTNET: Obtenez des clés sur https://testnet.binance.vision/\n"
                    "- Pour PRODUCTION: Vérifiez vos clés sur binance.com\n"
                    "- Vérifiez que BINANCE_TESTNET dans .env correspond au type de clés\n"
                    "- Assurez-vous que les clés ont 'Enable Reading' et 'Enable Trading'\n"
                )
                raise BinanceConnectionError("Clés API invalides ou incorrectes")

            except Exception as e:
                # Sur testnet, fetch_balance peut ne pas marcher, ce n'est pas grave
                if self.testnet:
                    self.logger.warning(
                        f"⚠️  API privée limitée sur testnet (normal): {e}"
                    )
                else:
                    raise

            # Marquer comme connecté
            self.is_connected = True
            self.connection_errors = 0

            self.logger.info("✅ Connexion vérifiée et fonctionnelle!")

        except ccxt.NetworkError as e:
            self.logger.error(f"❌ Erreur réseau: {e}")
            raise BinanceConnectionError(f"Impossible de se connecter à Binance: {e}")

        except ccxt.ExchangeError as e:
            self.logger.error(f"❌ Erreur exchange: {e}")
            raise BinanceConnectionError(f"Erreur Binance: {e}")

        except Exception as e:
            self.logger.error(f"❌ Erreur inattendue: {e}")
            raise BinanceConnectionError(f"Erreur connexion: {e}")

    def _rate_limit(self, endpoint: str = 'default', weight: int = 1):
        """
        Rate limiting intelligent avec tracking du poids des requêtes

        Args:
            endpoint: Nom de l'endpoint (pour tracking séparé)
            weight: Poids de la requête (selon doc Binance)
        """
        now = time.time()

        # Attendre si nécessaire
        if endpoint in self.last_request_time:
            elapsed = now - self.last_request_time[endpoint]
            min_interval = self.exchange.rateLimit / 1000 * (1 + self.rate_limit_buffer)

            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)

        self.last_request_time[endpoint] = time.time()

        # Tracker le poids
        if endpoint not in self.request_weights:
            self.request_weights[endpoint] = []

        self.request_weights[endpoint].append({
            'timestamp': time.time(),
            'weight': weight
        })

        # Nettoyer vieux poids (> 1 minute)
        self.request_weights[endpoint] = [
            w for w in self.request_weights[endpoint]
            if time.time() - w['timestamp'] < 60
        ]

    def _retry_on_error(max_retries: int = 3):
        """Décorateur pour retry automatique"""
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                last_error = None

                for attempt in range(max_retries):
                    try:
                        return func(self, *args, **kwargs)

                    except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            self.logger.warning(
                                f"⚠️  Erreur réseau, retry {attempt + 1}/{max_retries} "
                                f"dans {wait_time}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            self.logger.error(f"❌ Max retries atteint pour {func.__name__}")

                    except ccxt.RateLimitExceeded as e:
                        last_error = e
                        wait_time = 60  # 1 minute
                        self.logger.warning(f"⚠️  Rate limit atteint, pause {wait_time}s...")
                        time.sleep(wait_time)

                    except Exception as e:
                        # Autres erreurs = pas de retry
                        raise e

                raise last_error

            return wrapper
        return decorator

    @_retry_on_error(max_retries=3)
    def fetch_ohlcv(
        self,
        timeframe: str = '5m',
        limit: int = 500,
        since: Optional[int] = None
    ) -> list:
        """
        Récupère données OHLCV

        Args:
            timeframe: Timeframe ('1m', '5m', '1h', etc.)
            limit: Nombre de bougies (max 1000)
            since: Timestamp de début (ms)

        Returns:
            Liste de listes: [[timestamp, open, high, low, close, volume], ...]

        """
        self._rate_limit('fetch_ohlcv', weight=1)

        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=timeframe,
                limit=min(limit, 1000),
                since=since
            )

            self.logger.debug(f"📥 Récupéré {len(ohlcv)} bougies {timeframe}")

            return ohlcv

        except Exception as e:
            self.logger.error(f"❌ Erreur fetch_ohlcv: {e}")
            raise

    def get_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: str = '5m',
        limit: int = 500,
        since: Optional[int] = None
    ) -> list:
        """
        Alias pour fetch_ohlcv avec support du paramètre symbol

        Args:
            symbol: Paire de trading (ignoré - utilise self.symbol pour compatibilité)
            timeframe: Timeframe ('1m', '5m', '1h', etc.)
            limit: Nombre de bougies (max 1000)
            since: Timestamp de début (ms)

        Returns:
            Liste de listes: [[timestamp, open, high, low, close, volume], ...]
        """
        # Note: Le paramètre symbol est accepté pour compatibilité API mais
        # utilise toujours self.symbol car CCXT est déjà configuré
        if symbol and symbol != self.symbol:
            self.logger.warning(
                f"⚠️  get_ohlcv: Symbole {symbol} différent de {self.symbol}, "
                f"utilisation de {self.symbol}"
            )

        return self.fetch_ohlcv(timeframe=timeframe, limit=limit, since=since)

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
            DataFrame complet
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
            self._rate_limit('fetch_ohlcv', weight=1)

            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit
                )

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
                    self.logger.warning("⚠️  Limite 100k bougies atteinte")
                    break

                # Si moins de 'limit' reçu = fin des données
                if len(ohlcv) < limit:
                    break

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

    @_retry_on_error(max_retries=2)
    def get_ticker(self, symbol: Optional[str] = None) -> Dict:
        """Récupère ticker actuel

        Args:
            symbol: Paire de trading (défaut: self.symbol)

        Returns:
            Dict avec bid, ask, last, spread, volume, timestamp
        """
        self._rate_limit('get_ticker', weight=1)

        # Utiliser le symbole par défaut si non spécifié
        target_symbol = symbol if symbol else self.symbol

        try:
            ticker = self.exchange.fetch_ticker(target_symbol)

            # Gérer volume (quoteVolume ou baseVolume)
            volume = ticker.get('quoteVolume') or ticker.get('baseVolume') or 0

            return {
                'symbol': target_symbol,
                'bid': ticker.get('bid', ticker['last']),
                'ask': ticker.get('ask', ticker['last']),
                'last': ticker['last'],
                'spread': ticker.get('ask', ticker['last']) - ticker.get('bid', ticker['last']),
                'spread_percent': (ticker.get('ask', ticker['last']) - ticker.get('bid', ticker['last'])) / ticker['last'] * 100,
                'volume': volume,
                'timestamp': pd.Timestamp.now()
            }

        except Exception as e:
            self.logger.error(f"❌ Erreur get_ticker: {e}")
            return None

    @_retry_on_error(max_retries=2)
    def get_balance(self, currency: Optional[str] = None) -> Dict:
        """Récupère solde du compte

        Args:
            currency: Devise spécifique (ex: 'USDT', 'BTC')
                     Si None, retourne les balances base et quote

        Returns:
            Si currency spécifié: {'free': float, 'used': float, 'total': float}
            Sinon: {'base': {...}, 'quote': {...}}
        """
        self._rate_limit('get_balance', weight=5)

        try:
            balance = self.exchange.fetch_balance()

            # Si une devise spécifique est demandée
            if currency:
                return {
                    'free': balance.get(currency, {}).get('free', 0),
                    'used': balance.get(currency, {}).get('used', 0),
                    'total': balance.get(currency, {}).get('total', 0),
                }

            # Sinon retourner le format standard base/quote
            return {
                'base': {
                    'free': balance.get(self.base, {}).get('free', 0),
                    'used': balance.get(self.base, {}).get('used', 0),
                    'total': balance.get(self.base, {}).get('total', 0),
                },
                'quote': {
                    'free': balance.get(self.quote, {}).get('free', 0),
                    'used': balance.get(self.quote, {}).get('used', 0),
                    'total': balance.get(self.quote, {}).get('total', 0),
                }
            }

        except Exception as e:
            if self.testnet:
                # Sur testnet, retourner balance fictive
                self.logger.warning(f"⚠️  Balance non disponible sur testnet (normal)")
                if currency:
                    # Retourner balance fictive pour la devise demandée
                    if currency == self.quote:
                        return {'free': 10000, 'used': 0, 'total': 10000}
                    else:
                        return {'free': 0, 'used': 0, 'total': 0}
                return {
                    'base': {'free': 0, 'used': 0, 'total': 0},
                    'quote': {'free': 10000, 'used': 0, 'total': 10000}
                }
            else:
                self.logger.error(f"❌ Erreur get_balance: {e}")
                raise

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
            side: 'buy' ou 'sell'
            order_type: 'limit' ou 'market'
            amount: Quantité
            price: Prix (si limit)
            params: Paramètres additionnels

        Returns:
            Info ordre
        """
        self._rate_limit('create_order', weight=1)

        try:
            order = self.exchange.create_order(
                symbol=self.symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params or {}
            )

            self.logger.info(
                f"✅ Ordre {side.upper()} {order_type.upper()}: "
                f"{amount} {self.base} @ ${price or 'MARKET'}"
            )

            return order

        except Exception as e:
            self.logger.error(f"❌ Erreur create_order: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre"""
        self._rate_limit('cancel_order', weight=1)

        try:
            self.exchange.cancel_order(order_id, self.symbol)
            self.logger.info(f"🗑️  Ordre annulé: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Erreur cancel_order: {e}")
            return False

    def get_open_orders(self) -> List[Dict]:
        """Récupère ordres ouverts"""
        self._rate_limit('get_open_orders', weight=3)

        try:
            orders = self.exchange.fetch_open_orders(self.symbol)
            return orders
        except Exception as e:
            self.logger.error(f"❌ Erreur get_open_orders: {e}")
            return []

    def close_position(self, position_side: str = 'long') -> bool:
        """
        Ferme une position

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
                # Racheter pour fermer short (si futures)
                self.logger.warning("⚠️  Close short non implémenté pour spot")
                return False

            return False

        except Exception as e:
            self.logger.error(f"❌ Erreur close_position: {e}")
            return False

    def get_order_book(self, limit: int = 20) -> Dict:
        """Récupère l'order book"""
        self._rate_limit('get_order_book', weight=1)

        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=limit)

            return {
                'bids': orderbook['bids'],  # [[price, amount], ...]
                'asks': orderbook['asks'],
                'timestamp': pd.Timestamp.now()
            }

        except Exception as e:
            self.logger.error(f"❌ Erreur get_order_book: {e}")
            return None

    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """Récupère trades récents"""
        self._rate_limit('get_recent_trades', weight=1)

        try:
            trades = self.exchange.fetch_trades(self.symbol, limit=limit)
            return trades
        except Exception as e:
            self.logger.error(f"❌ Erreur get_recent_trades: {e}")
            return []

    def validate_price(self, price: float, price_type: str = "price") -> bool:
        """
        Valide qu'un prix est dans des limites acceptables (utilise fonction centralisée)

        Args:
            price: Prix à valider
            price_type: Type de prix pour logging

        Returns:
            True si valide
        """
        return _validate_price_util(price, price_type, self.min_price, self.max_price)

    def test_connectivity(self) -> bool:
        """
        Teste la connectivité avec l'exchange

        Returns:
            True si connecté
        """
        try:
            # Test simple ping
            self.exchange.fetch_time()
            return True
        except Exception as e:
            self.logger.error(f"❌ Test connectivité échoué: {e}")
            return False

    def reconnect(self) -> bool:
        """
        Tente de reconnecter à l'exchange

        Returns:
            True si reconnexion réussie
        """
        try:
            self.logger.info("🔄 Tentative de reconnexion...")

            # Fermer connexion existante
            if hasattr(self, 'exchange'):
                try:
                    self.exchange.close()
                except Exception as e:
                    self.logger.debug(f"Erreur fermeture connexion: {e}")
                    pass

            # Recréer connexion
            self._initialize_exchange()

            # Tester
            if self.test_connectivity():
                self.logger.info("✅ Reconnexion réussie")
                self.is_connected = True
                self.connection_errors = 0
                return True
            else:
                self.is_connected = False
                return False

        except Exception as e:
            self.logger.error(f"❌ Échec reconnexion: {e}")
            self.is_connected = False
            self.connection_errors += 1
            return False

    def __del__(self):
        """Nettoyage à la destruction"""
        if hasattr(self, 'exchange'):
            try:
                self.exchange.close()
            except Exception:
                # Pas de logging dans __del__ (peut causer des problèmes)
                pass

    # =========================================================================
    # ALIASES POUR COHÉRENCE AVEC CCXT
    # =========================================================================

    def fetch_ticker(self, symbol: Optional[str] = None) -> Dict:
        """Alias pour get_ticker (cohérence CCXT)"""
        return self.get_ticker(symbol)

    def fetch_balance(self, currency: Optional[str] = None) -> Dict:
        """Alias pour get_balance (cohérence CCXT)"""
        return self.get_balance(currency)


# Export
__all__ = ['BinanceClient', 'BinanceConnectionError']
