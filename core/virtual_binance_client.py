"""
Virtual Binance Client - Quantum Trader Pro
Client virtuel pour backtesting qui simule l'API Binance avec données historiques
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from decimal import Decimal
from utils.logger import setup_logger
from utils.calculations import timeframe_to_minutes
from core.base_client import BaseExchangeClient
from utils.config_helpers import get_nested_config
from utils.safe_math import safe_divide


class VirtualBinanceClient(BaseExchangeClient):
    """
    Client Binance virtuel pour backtesting
    Simule toutes les méthodes de BinanceClient mais utilise des données historiques
    Implémente BaseExchangeClient pour garantir la compatibilité API
    """

    def __init__(self, config: Dict):
        """
        Initialise le client virtuel

        Args:
            config: Configuration du bot
        """
        self.config = config
        self.logger = setup_logger('VirtualBinanceClient')

        # Symbol configuration - accès sécurisé
        self.symbol = get_nested_config(config, 'symbols', 'primary', default='BTC/USDT')
        if '/' in self.symbol:
            self.base, self.quote = self.symbol.split('/')
        else:
            self.base, self.quote = 'BTC', 'USDT'

        # Données historiques (chargées par replay_backtest)
        self.historical_data = {}  # {timeframe: DataFrame}
        self.current_index = {}  # {timeframe: current_position}

        # État virtuel
        initial_balance = get_nested_config(config, 'backtest', 'simulation', 'initial_balance', default=1000)
        self.virtual_balance = Decimal(str(initial_balance))
        self.virtual_base_balance = Decimal('0')  # Balance en crypto (BTC, etc)
        self.virtual_positions = []
        self.virtual_orders = []
        self.order_id_counter = 1

        # Timestamp courant (simulé)
        self.current_timestamp = None

        # Connectivité simulée
        self.is_connected = True

        self.logger.info(f"✅ Virtual Client initialisé - Balance: ${self.virtual_balance}")

    def load_historical_data(self, data: Dict[str, pd.DataFrame]):
        """
        Charge les données historiques

        Args:
            data: Dict {timeframe: DataFrame avec colonnes [timestamp, open, high, low, close, volume]}
        """
        self.historical_data = data

        # Initialiser les index à 0
        for timeframe in data.keys():
            self.current_index[timeframe] = 0

        # Initialiser le timestamp au premier point de données
        if data:
            first_timeframe = list(data.keys())[0]
            self.current_timestamp = data[first_timeframe].index[0]

        self.logger.info(f"📥 Données historiques chargées: {list(data.keys())}")

    def advance_time(self, timestamp: datetime) -> bool:
        """
        Avance le temps simulé

        Args:
            timestamp: Nouveau timestamp

        Returns:
            True si succès, False si fin des données
        """
        self.current_timestamp = timestamp

        # Vérifier si on a encore des données
        for timeframe, df in self.historical_data.items():
            if self.current_index[timeframe] >= len(df):
                return False

        return True

    def get_ticker(self, symbol: str = None) -> Dict:
        """
        Retourne le ticker au timestamp courant
        Implémente BaseExchangeClient.get_ticker()

        Returns:
            Dict avec les infos ticker standardisées
        """
        if not self.historical_data:
            raise Exception("Données historiques non chargées")

        # Utiliser la plus petite timeframe disponible pour le prix actuel
        smallest_tf = min(self.historical_data.keys(), key=lambda x: timeframe_to_minutes(x))
        df = self.historical_data[smallest_tf]
        idx = self.current_index[smallest_tf]

        if idx >= len(df):
            raise Exception("Fin des données atteinte")

        current_bar = df.iloc[idx]
        last_price = float(current_bar['close'])
        bid = last_price * 0.9999
        ask = last_price * 1.0001
        spread = ask - bid

        return {
            'symbol': self.symbol,
            'last': last_price,
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'spread_percent': safe_divide(spread, last_price, default=0.0) * 100,
            'high': float(current_bar['high']),
            'low': float(current_bar['low']),
            'volume': float(current_bar['volume']),
            'timestamp': self.current_timestamp
        }

    def get_balance(self, currency: Optional[str] = None) -> Dict:
        """
        Retourne le balance virtuel
        Implémente BaseExchangeClient.get_balance()

        Args:
            currency: Devise spécifique (ex: 'USDT', 'BTC')
                    Si None, retourne les balances base et quote

        Returns:
            Si currency spécifié: {'free': float, 'used': float, 'total': float}
            Sinon: {'base': {...}, 'quote': {...}}
        """
        # Calculer la position totale en base asset
        total_base = float(self.virtual_base_balance)
        for pos in self.virtual_positions:
            if pos['size'] > 0:
                total_base += pos['size']

        quote_balance = float(self.virtual_balance)

        # Si une devise spécifique est demandée
        if currency:
            if currency == self.quote:
                return {
                    'free': quote_balance,
                    'used': 0.0,
                    'total': quote_balance
                }
            elif currency == self.base:
                return {
                    'free': total_base,
                    'used': 0.0,
                    'total': total_base
                }
            else:
                return {'free': 0.0, 'used': 0.0, 'total': 0.0}

        # Sinon retourner le format standard base/quote
        return {
            'base': {
                'free': total_base,
                'used': 0.0,
                'total': total_base
            },
            'quote': {
                'free': quote_balance,
                'used': 0.0,
                'total': quote_balance
            }
        }

    def fetch_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = '5m',
        limit: int = 500,
        since: Optional[int] = None
    ) -> List[List]:
        """
        Retourne les données OHLCV historiques jusqu'au timestamp courant
        Implémente BaseExchangeClient.fetch_ohlcv()

        Args:
            timeframe: Timeframe (1m, 5m, 1h, etc)
            limit: Nombre de bougies
            since: Timestamp de début (ms) - ignoré en backtest

        Returns:
            Liste de [timestamp_ms, open, high, low, close, volume]
        """
        if timeframe not in self.historical_data:
            raise Exception(f"Timeframe {timeframe} non disponible")

        df = self.historical_data[timeframe]
        idx = self.current_index[timeframe]

        # Prendre les dernières 'limit' bougies jusqu'à l'index courant
        start_idx = max(0, idx - limit + 1)
        end_idx = idx + 1

        data_slice = df.iloc[start_idx:end_idx]

        # Convertir en format CCXT standard
        ohlcv = []
        for row_idx, row in data_slice.iterrows():
            if isinstance(row_idx, datetime):
                ts = int(row_idx.timestamp() * 1000)
            else:
                ts = row_idx

            ohlcv.append([
                ts,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ])

        return ohlcv

    def create_order(
            self,
            side: str,
            order_type: str,
            amount: float,
            price: Optional[float] = None,
            params: Optional[Dict] = None
        ) -> Dict:
            """
            Crée un ordre virtuel
            Implémente BaseExchangeClient.create_order()
            """
            try:
                # Récupérer prix courant
                ticker = self.get_ticker()
                current_price = ticker['last']

                # Déterminer prix de fill
                if order_type == 'market':
                    fill_price = current_price
                else:
                    fill_price = price if price else current_price

                # Calculer le coût et les frais
                cost = amount * fill_price
                commission = cost * 0.001  # 0.1% de frais

                # TOUTES LES VÉRIFICATIONS D'ABORD (sans modifier le balance)
                if side.lower() == 'buy':
                    total_cost = cost + commission
                    if float(self.virtual_balance) < total_cost:
                        raise Exception(f"Balance insuffisant: {self.virtual_balance} < {total_cost}")
                else:  # sell
                    if float(self.virtual_base_balance) < amount:
                        raise Exception(f"Base asset insuffisant: {self.virtual_base_balance} < {amount}")

                # SI TOUTES LES VÉRIFICATIONS PASSENT, ALORS modifier le balance
                if side.lower() == 'buy':
                    # Déduire du balance quote (coût + frais)
                    self.virtual_balance -= Decimal(str(total_cost))
                    # Ajouter au balance base
                    self.virtual_base_balance += Decimal(str(amount))

                    # Tracker la position
                    self.virtual_positions.append({
                        'symbol': self.symbol,
                        'side': 'long',
                        'size': amount,
                        'entry_price': fill_price,
                        'timestamp': self.current_timestamp
                    })

                else:  # sell
                    # Déduire du base balance
                    self.virtual_base_balance -= Decimal(str(amount))
                    # Ajouter au quote balance (revenus - frais)
                    self.virtual_balance += Decimal(str(cost - commission))

                    # Supprimer de la position
                    for pos in self.virtual_positions:
                        if pos['symbol'] == self.symbol and pos['size'] >= amount:
                            pos['size'] -= amount
                            break

                # Créer l'ordre standardisé
                order = {
                    'id': f"VIRTUAL_{self.order_id_counter}",
                    'symbol': self.symbol,
                    'side': side.lower(),
                    'type': order_type.lower(),
                    'price': fill_price,
                    'amount': amount,
                    'filled': amount,
                    'remaining': 0.0,
                    'status': 'closed',
                    'timestamp': int(self.current_timestamp.timestamp() * 1000) if self.current_timestamp else 0,
                    'cost': cost,
                    'fee': {'cost': commission, 'currency': self.quote},
                    'average': fill_price,
                    'fees': commission  # Ajouter ce champ pour que position_manager puisse l'utiliser
                }

                self.order_id_counter += 1
                self.virtual_orders.append(order)

                self.logger.info(
                    f"📝 Ordre virtuel: {side.upper()} {amount:.8f} {self.symbol} @ ${fill_price:.2f} "
                    f"(Frais: ${commission:.2f}, Balance: ${self.virtual_balance:.2f})"
                )

                return order

            except Exception as e:
                self.logger.error(f"❌ Erreur création ordre virtuel: {e}")
                raise

    def create_market_order(self, symbol: str, side: str, amount: float, params: Optional[Dict] = None) -> Dict:
        """Alias pour create_order en mode market (compatibilité CCXT)"""
        return self.create_order(side=side, order_type='market', amount=amount, price=None, params=params)

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params: Optional[Dict] = None) -> Dict:
        """Alias pour create_order en mode limit (compatibilité CCXT)"""
        return self.create_order(side=side, order_type='limit', amount=amount, price=price, params=params)

    def cancel_order(self, order_id: str) -> bool:
        """
        Annule un ordre virtuel
        Implémente BaseExchangeClient.cancel_order()

        Args:
            order_id: ID de l'ordre

        Returns:
            True si annulé
        """
        # En mode virtuel, les ordres sont exécutés immédiatement
        # Donc pas vraiment d'annulation possible
        self.logger.warning(f"⚠️ Annulation ordre virtuel {order_id} - Ordres déjà exécutés en backtest")
        return False

    def get_open_orders(self) -> List[Dict]:
        """
        Récupère les ordres ouverts
        Implémente BaseExchangeClient.get_open_orders()

        Returns:
            Liste vide (ordres exécutés immédiatement en backtest)
        """
        # En backtest, tous les ordres sont exécutés immédiatement
        return []

    def get_order_book(self, limit: int = 20) -> Dict:
        """
        Simule un orderbook
        Implémente BaseExchangeClient.get_order_book()

        Args:
            limit: Profondeur

        Returns:
            Dict standardisé avec bids et asks
        """
        ticker = self.get_ticker()
        mid_price = ticker['last']

        # Générer un orderbook synthétique
        bids = []
        asks = []

        for i in range(limit):
            # Bids décroissants à partir du mid
            bid_price = mid_price * (1 - 0.0001 * (i + 1))
            bid_size = 1.0 + (i * 0.1)
            bids.append([bid_price, bid_size])

            # Asks croissants à partir du mid
            ask_price = mid_price * (1 + 0.0001 * (i + 1))
            ask_size = 1.0 + (i * 0.1)
            asks.append([ask_price, ask_size])

        return {
            'bids': bids,
            'asks': asks,
            'timestamp': int(self.current_timestamp.timestamp() * 1000) if self.current_timestamp else 0
        }

    def test_connectivity(self) -> bool:
        """
        Teste la connectivité (toujours True en virtual)
        Implémente BaseExchangeClient.test_connectivity()

        Returns:
            True
        """
        return self.is_connected

    def close_position(self, position_side: str = 'long') -> bool:
        """
        Ferme une position virtuelle
        Implémente BaseExchangeClient.close_position()

        Args:
            position_side: 'long' ou 'short'

        Returns:
            True si succès
        """
        try:
            if position_side == 'long':
                # Vendre tout le base asset
                amount = float(self.virtual_base_balance)
                if amount > 0:
                    self.create_order('sell', 'market', amount)
                    self.logger.info(f"🔒 Position LONG virtuelle fermée: {amount} {self.base}")
                    return True
            else:
                # Short non supporté en spot
                self.logger.warning("⚠️ Close short non implémenté pour spot virtuel")
                return False

            return False

        except Exception as e:
            self.logger.error(f"❌ Erreur close_position virtuelle: {e}")
            return False

    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """
        Récupère les trades récents virtuels

        Returns:
            Liste des ordres virtuels
        """
        return self.virtual_orders[-limit:] if len(self.virtual_orders) > limit else self.virtual_orders

    def get_current_price(self) -> float:
        """Raccourci pour obtenir le prix actuel"""
        return self.get_ticker()['last']

    def get_statistics(self) -> Dict:
        """Retourne les statistiques du backtest"""
        return {
            'final_balance': float(self.virtual_balance),
            'final_base_balance': float(self.virtual_base_balance),
            'total_orders': len(self.virtual_orders),
            'open_positions': len([p for p in self.virtual_positions if p['size'] > 0]),
            'current_timestamp': self.current_timestamp
        }

    # Méthodes de compatibilité (alias vers méthodes standard)
    def place_order(self, symbol: str, side: str, order_type: str,
                    amount: float, price: Optional[float] = None, params: Optional[Dict] = None) -> Dict:
        """Alias de compatibilité pour create_order"""
        return self.create_order(side, order_type, amount, price, params)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List:
        """Alias de compatibilité pour fetch_ohlcv"""
        return self.fetch_ohlcv(symbol, timeframe, limit)

    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Alias de compatibilité pour get_order_book"""
        return self.get_order_book(limit)

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Alias pour get_orderbook (compatibilité CCXT)"""
        return self.get_orderbook(symbol, limit)

__all__ = ['VirtualBinanceClient']
