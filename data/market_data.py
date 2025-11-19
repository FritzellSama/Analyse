"""
Market Data - Quantum Trader Pro
Gestion des données de marché en temps réel (orderbook, trades, ticker)
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
from threading import RLock
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
from utils.safe_math import safe_divide
from utils.thread_sync import synchronized
import time

class MarketData:
    """
    Gestionnaire de données de marché en temps réel:
    - Order book (depth)
    - Ticker (prix, volume)
    - Recent trades
    - Spread analysis
    - Liquidity metrics
    """
    
    def __init__(self, client, config: Dict):
        """
        Initialise le gestionnaire de market data
        
        Args:
            client: Instance BinanceClient
            config: Configuration complète du bot
        """
        self.client = client
        self.config = config
        self.logger = setup_logger('MarketData')
        self._lock = RLock()  # Thread safety

        # Configuration - accès sécurisé
        self.symbol = get_nested_config(config, 'symbols', 'primary', default='BTC/USDT')

        # Configuration order flow
        order_flow_config = get_nested_config(config, 'advanced', 'order_flow', default={})
        self.order_flow_enabled = order_flow_config.get('enabled', True)
        self.depth_levels = order_flow_config.get('depth_levels', 20)
        self.imbalance_threshold = order_flow_config.get('imbalance_threshold', 0.3)

        # Cache données
        self.current_ticker = {}
        self.current_orderbook = {'bids': [], 'asks': []}
        self.recent_trades = deque(maxlen=100)
        
        # Métriques
        self.bid_ask_spread = 0.0
        self.orderbook_imbalance = 0.0
        self.liquidity_score = 0.0
        
        # Timestamps
        self.last_ticker_update = 0
        self.last_orderbook_update = 0
        
        self.logger.info("✅ Market Data initialisé")
    
    def update_ticker(self, symbol: Optional[str] = None) -> Dict:
        """
        Met à jour les données ticker
        
        Args:
            symbol: Paire de trading (défaut: config)
        
        Returns:
            Dict avec données ticker
        """
        
        symbol = symbol or self.symbol
        
        try:
            ticker = self.client.get_ticker()
            
            self.current_ticker = {
                'symbol': ticker['symbol'],
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'last': ticker.get('last', 0),
                'high': ticker.get('high', 0),
                'low': ticker.get('low', 0),
                'volume': ticker.get('quoteVolume', 0),
                'timestamp': datetime.now()
            }
            
            # Calculer spread
            if self.current_ticker['bid'] > 0 and self.current_ticker['ask'] > 0:
                self.bid_ask_spread = safe_divide(
                    self.current_ticker['ask'] - self.current_ticker['bid'],
                    self.current_ticker['bid'],
                    default=0.0
                )
            
            self.last_ticker_update = time.time()
            
            return self.current_ticker
            
        except Exception as e:
            self.logger.error(f"❌ Erreur update ticker: {e}")
            return {}
    
    def update_orderbook(
        self,
        symbol: Optional[str] = None,
        limit: int = 20
    ) -> Dict:
        """
        Met à jour l'order book
        
        Args:
            symbol: Paire de trading
            limit: Nombre de niveaux de profondeur
        
        Returns:
            Dict avec bids et asks
        """
        
        symbol = symbol or self.symbol
        limit = min(limit, self.depth_levels)
        
        try:
            orderbook = self.client.exchange.fetch_order_book(symbol, limit)
            self.current_orderbook = {
                'bids': orderbook.get('bids', [])[:limit],
                'asks': orderbook.get('asks', [])[:limit],
                'timestamp': datetime.now()
            }
            
            # Calculer métriques
            self._calculate_orderbook_metrics()
            
            self.last_orderbook_update = time.time()
            
            return self.current_orderbook
            
        except Exception as e:
            self.logger.error(f"❌ Erreur update orderbook: {e}")
            return {'bids': [], 'asks': []}
    
    def get_market_depth(self, symbol: Optional[str] = None) -> Dict:
        """
        Analyse la profondeur du marché
        
        Args:
            symbol: Paire de trading
        
        Returns:
            Dict avec métriques de profondeur
        """
        
        # Update orderbook si nécessaire
        if time.time() - self.last_orderbook_update > 5:
            self.update_orderbook(symbol)
        
        bids = self.current_orderbook.get('bids', [])
        asks = self.current_orderbook.get('asks', [])
        
        if not bids or not asks:
            return {}
        
        # Volume total
        bid_volume = sum(float(bid[1]) for bid in bids)
        ask_volume = sum(float(ask[1]) for ask in asks)
        total_volume = bid_volume + ask_volume
        
        # Prix moyens pondérés
        bid_weighted_price = safe_divide(
            sum(float(bid[0]) * float(bid[1]) for bid in bids),
            bid_volume,
            default=0.0
        )

        ask_weighted_price = safe_divide(
            sum(float(ask[0]) * float(ask[1]) for ask in asks),
            ask_volume,
            default=0.0
        )
        
        # Mid price
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'spread': self.bid_ask_spread,
            'spread_bps': self.bid_ask_spread * 10000,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'total_volume': total_volume,
            'bid_weighted_price': bid_weighted_price,
            'ask_weighted_price': ask_weighted_price,
            'imbalance': self.orderbook_imbalance,
            'liquidity_score': self.liquidity_score
        }
    
    def _calculate_orderbook_metrics(self):
        """Calcule les métriques de l'order book"""
        
        bids = self.current_orderbook.get('bids', [])
        asks = self.current_orderbook.get('asks', [])
        
        if not bids or not asks:
            return
        
        # Imbalance
        bid_volume = sum(float(bid[1]) for bid in bids)
        ask_volume = sum(float(ask[1]) for ask in asks)
        total_volume = bid_volume + ask_volume
        
        self.orderbook_imbalance = safe_divide(bid_volume - ask_volume, total_volume, default=0.0)
        
        # Liquidity score (basé sur volume et spread)
        # Score élevé = bonne liquidité
        if self.bid_ask_spread > 0:
            self.liquidity_score = total_volume / (1 + self.bid_ask_spread * 1000)
        else:
            self.liquidity_score = total_volume
    
    def get_orderbook_imbalance(self, levels: int = 5) -> float:
        """
        Calcule l'imbalance des X premiers niveaux
        
        Args:
            levels: Nombre de niveaux à considérer
        
        Returns:
            Imbalance ratio (-1 à 1)
        """
        
        bids = self.current_orderbook.get('bids', [])[:levels]
        asks = self.current_orderbook.get('asks', [])[:levels]
        
        if not bids or not asks:
            return 0.0
        
        bid_volume = sum(float(bid[1]) for bid in bids)
        ask_volume = sum(float(ask[1]) for ask in asks)
        total = bid_volume + ask_volume
        
        if total == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total
    
    def detect_large_orders(self, threshold_multiplier: float = 3.0) -> List[Dict]:
        """
        Détecte les ordres inhabituellement larges dans le carnet
        
        Args:
            threshold_multiplier: Multiplicateur pour déterminer "large"
        
        Returns:
            Liste des ordres larges détectés
        """
        
        bids = self.current_orderbook.get('bids', [])
        asks = self.current_orderbook.get('asks', [])
        
        all_orders = []
        for bid in bids:
            all_orders.append({'price': float(bid[0]), 'size': float(bid[1]), 'side': 'bid'})
        for ask in asks:
            all_orders.append({'price': float(ask[0]), 'size': float(ask[1]), 'side': 'ask'})
        
        if not all_orders:
            return []
        
        # Calculer taille moyenne
        avg_size = safe_divide(sum(o['size'] for o in all_orders), len(all_orders), default=0.0)
        threshold = avg_size * threshold_multiplier
        
        # Filtrer ordres larges
        large_orders = [
            o for o in all_orders
            if o['size'] > threshold
        ]
        
        if large_orders:
            self.logger.debug(
                f"🐋 {len(large_orders)} ordres larges détectés "
                f"(seuil: {threshold:.4f})"
            )
        
        return large_orders
    
    def get_support_resistance_from_orderbook(self) -> Dict:
        """
        Identifie les niveaux de support/résistance depuis l'orderbook
        
        Returns:
            Dict avec niveaux support et résistance
        """
        
        large_orders = self.detect_large_orders(threshold_multiplier=2.5)
        
        if not large_orders:
            return {'support': [], 'resistance': []}
        
        # Séparer support (bids) et résistance (asks)
        support_levels = [
            {'price': o['price'], 'size': o['size']}
            for o in large_orders
            if o['side'] == 'bid'
        ]
        
        resistance_levels = [
            {'price': o['price'], 'size': o['size']}
            for o in large_orders
            if o['side'] == 'ask'
        ]
        
        # Trier par taille décroissante
        support_levels.sort(key=lambda x: x['size'], reverse=True)
        resistance_levels.sort(key=lambda x: x['size'], reverse=True)
        
        return {
            'support': support_levels[:3],  # Top 3
            'resistance': resistance_levels[:3]
        }
    
    def get_spread_metrics(self) -> Dict:
        """
        Calcule des métriques détaillées sur le spread
        
        Returns:
            Dict avec métriques de spread
        """
        
        if not self.current_ticker:
            self.update_ticker()
        
        bid = self.current_ticker.get('bid', 0)
        ask = self.current_ticker.get('ask', 0)
        
        if bid == 0 or ask == 0:
            return {}
        
        spread_abs = ask - bid
        spread_pct = spread_abs / bid
        spread_bps = spread_pct * 10000
        
        # Classif spread
        if spread_bps < 10:
            classification = 'tight'
        elif spread_bps < 30:
            classification = 'normal'
        elif spread_bps < 50:
            classification = 'wide'
        else:
            classification = 'very_wide'
        
        return {
            'bid': bid,
            'ask': ask,
            'spread_absolute': spread_abs,
            'spread_percent': spread_pct,
            'spread_bps': spread_bps,
            'classification': classification,
            'is_acceptable': spread_bps < get_nested_config(self.config, 'symbols', 'filters', 'max_spread_percent', default=0.2) * 100
        }
    
    def get_volume_profile(
        self,
        symbol: Optional[str] = None,
        num_levels: int = 10
    ) -> Dict:
        """
        Calcule le profil de volume de l'orderbook
        
        Args:
            symbol: Paire de trading
            num_levels: Nombre de niveaux à analyser
        
        Returns:
            Dict avec profil de volume
        """
        
        # Update si nécessaire
        if time.time() - self.last_orderbook_update > 5:
            self.update_orderbook(symbol)
        
        bids = self.current_orderbook.get('bids', [])[:num_levels]
        asks = self.current_orderbook.get('asks', [])[:num_levels]
        
        if not bids or not asks:
            return {}
        
        # Calculer volume par niveau
        bid_profile = [
            {'price': float(bid[0]), 'volume': float(bid[1])}
            for bid in bids
        ]
        
        ask_profile = [
            {'price': float(ask[0]), 'volume': float(ask[1])}
            for ask in asks
        ]
        
        # Point de contrôle (POC) = niveau avec plus de volume
        all_levels = bid_profile + ask_profile
        poc = max(all_levels, key=lambda x: x['volume']) if all_levels else None
        
        return {
            'bid_profile': bid_profile,
            'ask_profile': ask_profile,
            'poc': poc,
            'total_bid_volume': sum(b['volume'] for b in bid_profile),
            'total_ask_volume': sum(a['volume'] for a in ask_profile)
        }
    
    def is_liquid_enough(self, required_volume: float = 0) -> bool:
        """
        Vérifie si le marché est suffisamment liquide
        
        Args:
            required_volume: Volume minimum requis
        
        Returns:
            True si suffisamment liquide
        """
        
        # Update si nécessaire
        if time.time() - self.last_orderbook_update > 5:
            self.update_orderbook()
        
        bids = self.current_orderbook.get('bids', [])
        asks = self.current_orderbook.get('asks', [])
        
        if not bids or not asks:
            return False
        
        # Volume disponible
        bid_volume = sum(float(bid[1]) for bid in bids[:5])  # Top 5 niveaux
        ask_volume = sum(float(ask[1]) for ask in asks[:5])
        min_volume = min(bid_volume, ask_volume)
        
        # Vérifier spread
        spread_ok = self.bid_ask_spread < 0.002  # 0.2%
        
        # Vérifier volume
        volume_ok = min_volume >= required_volume
        
        return spread_ok and volume_ok
    
    def get_market_state(self) -> Dict:
        """
        Retourne l'état complet du marché
        
        Returns:
            Dict avec toutes les métriques de marché
        """
        
        # Update données si nécessaire
        if time.time() - self.last_ticker_update > 5:
            self.update_ticker()
        
        if time.time() - self.last_orderbook_update > 5:
            self.update_orderbook()
        
        depth = self.get_market_depth()
        spread = self.get_spread_metrics()
        levels = self.get_support_resistance_from_orderbook()
        
        return {
            'ticker': self.current_ticker,
            'depth': depth,
            'spread': spread,
            'support_levels': levels.get('support', []),
            'resistance_levels': levels.get('resistance', []),
            'orderbook_imbalance': self.orderbook_imbalance,
            'liquidity_score': self.liquidity_score,
            'is_liquid': self.is_liquid_enough(),
            'timestamp': datetime.now().isoformat()
        }
