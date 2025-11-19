"""
Order Executor - Quantum Trader Pro
Exécution intelligente des ordres avec gestion du slippage et retry logic
"""

import time
from typing import Dict, Optional, List, Tuple
from enum import Enum
from datetime import datetime
from utils.logger import setup_trading_logger
from utils.validators import validate_price as _validate_price_util
from utils.safe_math import safe_divide
from utils.config_helpers import get_nested_config

class OrderType(Enum):
    """Types d'ordres supportés"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Côté de l'ordre"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Statuts d'ordre"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderExecutor:
    """
    Gestionnaire d'exécution d'ordres avec:
    - Gestion du slippage
    - Retry automatique
    - Smart Order Routing (SOR)
    - Iceberg orders
    - Post-trade analysis
    """
    
    def __init__(self, client, config: Dict):
        """
        Initialise l'exécuteur d'ordres
        
        Args:
            client: Instance BinanceClient
            config: Configuration complète du bot
        """
        self.client = client
        self.config = config
        self.logger = setup_trading_logger('OrderExecutor')

        # Configuration d'exécution
        exec_config = config.get('execution', {})
        self.order_type = exec_config.get('order_type', 'limit')
        self.limit_offset = exec_config.get('limit_price_offset_percent', 0.05) / 100
        self.expected_slippage = exec_config.get('expected_slippage_percent', 0.05) / 100
        self.max_slippage = exec_config.get('max_acceptable_slippage_percent', 0.2) / 100
        self.max_retries = exec_config.get('max_retries', 3)
        self.retry_delay = exec_config.get('retry_delay_seconds', 2)
        self.timeout = exec_config.get('timeout_seconds', 30)
        
        # Smart Order Routing
        sor_config = exec_config.get('sor', {})
        self.sor_enabled = sor_config.get('enabled', True)
        self.split_large_orders = sor_config.get('split_large_orders', True)
        self.max_order_size_pct = sor_config.get('max_order_size_percent', 5) / 100
        self.iceberg_enabled = sor_config.get('iceberg_orders', True)
        
        # Post-trade analysis
        post_trade = exec_config.get('post_trade', {})
        self.calc_slippage = post_trade.get('calculate_slippage', True)
        self.calc_impact = post_trade.get('calculate_impact', True)
        self.save_to_db = post_trade.get('save_to_database', True)
        
        # Tracking
        self.orders_history = []
        self.total_slippage = 0
        self.total_orders = 0
        
        self.logger.info("✅ Order Executor initialisé")
    
    def execute_order(
        self,
        side: str,
        size: float,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Exécute un ordre avec gestion intelligente
        
        Args:
            side: 'buy' ou 'sell'
            size: Taille de la position
            symbol: Paire de trading (défaut: config)
            order_type: Type d'ordre (défaut: config)
            limit_price: Prix limite (pour limit orders)
            stop_price: Prix stop (pour stop_limit)
            params: Paramètres additionnels
        
        Returns:
            Dict contenant les détails de l'ordre exécuté
        """
        symbol = symbol or get_nested_config(self.config, 'symbols', 'primary', default='BTC/USDT')
        order_type = order_type or self.order_type
        params = params or {}
        
        # Validation des prix AVANT exécution
        if limit_price and not self._validate_price(limit_price, "limit"):
            raise ValueError(f"Prix limite invalide: {limit_price}")
        
        if stop_price and not self._validate_price(stop_price, "stop"):
            raise ValueError(f"Prix stop invalide: {stop_price}")
        
        # Validation de la taille
        if size <= 0:
            raise ValueError(f"Taille invalide: {size}")
        
        # Log de l'intention
        self.logger.info(
            f"📝 Ordre {side.upper()} {size} {symbol} "
            f"(type: {order_type})"
        )
        
        # Vérifier si split nécessaire
        if self.split_large_orders and self._should_split_order(size, symbol):
            return self._execute_split_order(side, size, symbol, order_type, params)
        
        # Exécution normale avec retry
        for attempt in range(1, self.max_retries + 1):
            try:
                # Calculer prix limite si nécessaire
                if order_type == 'limit' and limit_price is None:
                    limit_price = self._calculate_limit_price(side, symbol)
                
                # Créer l'ordre
                order = self._create_order(
                    symbol, side, order_type, size, 
                    limit_price, stop_price, params
                )
                
                # Post-trade analysis
                if self.calc_slippage or self.calc_impact:
                    order = self._analyze_order(order, side, size)
                
                # Vérifier le slippage après exécution
                if not self._verify_execution_quality(order, limit_price):
                    self.logger.error(f"❌ Qualité d'exécution médiocre pour ordre {order['id']}")
                
                # Sauvegarder historique
                self._save_order_history(order)
                
                # Log succès
                self.logger.trade_opened(
                    symbol=symbol,
                    side=side,
                    size=size,
                    price=order.get('average') or order.get('price', 0),
                    order_id=order['id']
                )
                
                return order
                
            except Exception as e:
                self.logger.error(
                    f"❌ Tentative {attempt}/{self.max_retries} échouée: {e}"
                )
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"❌ Échec définitif après {self.max_retries} tentatives")
                    raise
    
    def _create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        limit_price: Optional[float],
        stop_price: Optional[float],
        params: Dict
    ) -> Dict:
        """Crée l'ordre sur l'exchange"""
        
        if order_type == 'market':
            order = self.client.create_market_order(
                symbol, side, size, params
            )
        elif order_type == 'limit':
            order = self.client.create_limit_order(
                symbol, side, size, limit_price, params
            )
        elif order_type == 'stop_limit':
            order = self.client.create_order(
                symbol, 'limit', side, size, limit_price,
                {'stopPrice': stop_price, **params}
            )
        else:
            raise ValueError(f"Type d'ordre non supporté: {order_type}")
        
        return order
    
    def _calculate_limit_price(self, side: str, symbol: str) -> float:
        """Calcule le prix limite optimal"""
        
        # Récupérer ticker
        ticker = self.client.get_ticker(symbol)
        current_price = ticker['last']
        
        # Offset selon le côté
        if side == 'buy':
            # Acheter légèrement au-dessus pour augmenter chances de fill
            limit_price = current_price * (1 + self.limit_offset)
        else:
            # Vendre légèrement en-dessous
            limit_price = current_price * (1 - self.limit_offset)
        
        self.logger.debug(
            f"Prix limite calculé: {limit_price:.8f} "
            f"(current: {current_price:.8f}, offset: {self.limit_offset:.2%})"
        )
        
        return limit_price
    
    def _should_split_order(self, size: float, symbol: str) -> bool:
        """Détermine si l'ordre doit être splitté"""
        
        if not self.split_large_orders:
            return False
        
        try:
            # Récupérer volume 24h
            ticker = self.client.get_ticker(symbol)
            volume_24h = ticker.get('quoteVolume', 0)
            
            if volume_24h == 0:
                return False
            
            # Calculer taille relative
            current_price = ticker['last']
            order_value = size * current_price
            order_pct = safe_divide(order_value, volume_24h, default=0.0)
            
            should_split = order_pct > self.max_order_size_pct
            
            if should_split:
                self.logger.info(
                    f"📊 Ordre large détecté: {order_pct:.2%} du volume 24h → Split"
                )
            
            return should_split
            
        except Exception as e:
            self.logger.warning(f"⚠️ Impossible de vérifier taille ordre: {e}")
            return False
    
    def _execute_split_order(
        self,
        side: str,
        size: float,
        symbol: str,
        order_type: str,
        params: Dict
    ) -> Dict:
        """Exécute un ordre en plusieurs parties"""
        
        # Déterminer nombre de splits (2-5)
        num_splits = min(5, max(2, int(size / 0.001)))  # Min 0.001 BTC par ordre
        split_size = size / num_splits
        
        self.logger.info(
            f"🔀 Split ordre en {num_splits} parties de {split_size:.8f} chacune"
        )
        
        # Exécuter les splits
        filled_orders = []
        total_filled = 0
        total_cost = 0
        
        for i in range(num_splits):
            try:
                # Délai entre splits pour éviter market impact
                if i > 0:
                    time.sleep(2)
                
                order = self.execute_order(
                    side, split_size, symbol, order_type, params=params
                )
                
                filled_orders.append(order)
                total_filled += order.get('filled', 0)
                total_cost += order.get('cost', 0)
                
            except Exception as e:
                self.logger.error(f"❌ Erreur split {i+1}/{num_splits}: {e}")
        
        # Consolider résultats
        avg_price = safe_divide(total_cost, total_filled, default=0.0)
        
        consolidated = {
            'id': f"SPLIT_{int(time.time())}",
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'amount': size,
            'filled': total_filled,
            'remaining': size - total_filled,
            'average': avg_price,
            'cost': total_cost,
            'status': 'filled' if total_filled >= size * 0.95 else 'partially_filled',
            'splits': filled_orders,
            'timestamp': datetime.now().isoformat()
        }
        
        return consolidated
    
    def _analyze_order(self, order: Dict, side: str, size: float) -> Dict:
        """Analyse post-trade de l'ordre"""
        
        # Calculer slippage si ordre fill
        if order.get('status') == 'filled' and self.calc_slippage:
            slippage = self._calculate_slippage(order, side)
            order['slippage'] = slippage
            self.total_slippage += abs(slippage)
            
            if abs(slippage) > self.max_slippage:
                self.logger.warning(
                    f"⚠️ Slippage élevé: {slippage:.2%} "
                    f"(max: {self.max_slippage:.2%})"
                )
        
        # Calculer market impact
        if self.calc_impact:
            impact = self._calculate_market_impact(order, size)
            order['market_impact'] = impact
        
        self.total_orders += 1
        
        return order
    
    def _calculate_slippage(self, order: Dict, side: str) -> float:
        """Calcule le slippage de l'ordre"""
        
        try:
            # Prix moyen d'exécution
            avg_price = order.get('average', 0)
            
            # Prix attendu (au moment de l'ordre)
            symbol = order['symbol']
            ticker = self.client.get_ticker(symbol)
            expected_price = ticker['last']
            
            # Slippage = (Prix exécuté - Prix attendu) / Prix attendu
            if side == 'buy':
                slippage = safe_divide(avg_price - expected_price, expected_price, default=0.0)
            else:
                slippage = safe_divide(expected_price - avg_price, expected_price, default=0.0)

            return slippage
            
        except Exception as e:
            self.logger.warning(f"⚠️ Impossible de calculer slippage: {e}")
            return 0.0
    
    def _calculate_market_impact(self, order: Dict, size: float) -> float:
        """Estime l'impact sur le marché"""
        
        try:
            symbol = order['symbol']
            ticker = self.client.get_ticker(symbol)
            
            # Volume 24h
            volume_24h = ticker.get('quoteVolume', 0)
            if volume_24h == 0:
                return 0.0

            # Impact = Taille ordre / Volume 24h
            avg_price = order.get('average', ticker['last'])
            order_value = size * avg_price
            impact = safe_divide(order_value, volume_24h, default=0.0)
            
            return impact
            
        except Exception as e:
            self.logger.warning(f"⚠️ Impossible de calculer impact: {e}")
            return 0.0
    
    def _save_order_history(self, order: Dict):
        """Sauvegarde l'ordre dans l'historique"""
        
        order['execution_timestamp'] = datetime.now().isoformat()
        self.orders_history.append(order)
        
        # Limiter taille historique (garder 1000 derniers)
        if len(self.orders_history) > 1000:
            self.orders_history = self.orders_history[-1000:]
    
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """
        Annule un ordre

        Args:
            order_id: ID de l'ordre à annuler
            symbol: Symbole (défaut: config)

        Returns:
            True si annulé avec succès
        """
        symbol = symbol or get_nested_config(self.config, 'symbols', 'primary', default='BTC/USDT')
        
        try:
            self.client.cancel_order(order_id, symbol)
            self.logger.info(f"✅ Ordre {order_id} annulé")
            return True
        except Exception as e:
            self.logger.error(f"❌ Erreur annulation ordre {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        """
        Récupère le statut d'un ordre
        
        Args:
            order_id: ID de l'ordre
            symbol: Symbole (défaut: config)

        Returns:
            Dict avec les détails de l'ordre
        """
        symbol = symbol or get_nested_config(self.config, 'symbols', 'primary', default='BTC/USDT')
        
        try:
            order = self.client.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            self.logger.error(f"❌ Erreur récupération ordre {order_id}: {e}")
            return {}
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère tous les ordres ouverts
        
        Args:
            symbol: Symbole (défaut: tous)
        
        Returns:
            Liste des ordres ouverts
        """
        try:
            orders = self.client.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            self.logger.error(f"❌ Erreur récupération ordres ouverts: {e}")
            return []
    
    def get_execution_stats(self) -> Dict:
        """
        Retourne les statistiques d'exécution
        
        Returns:
            Dict avec les stats
        """
        if self.total_orders == 0:
            avg_slippage = 0
        else:
            avg_slippage = safe_divide(self.total_slippage, self.total_orders, default=0.0)
        
        return {
            'total_orders': self.total_orders,
            'total_slippage': self.total_slippage,
            'avg_slippage': avg_slippage,
            'max_acceptable_slippage': self.max_slippage,
            'orders_history_size': len(self.orders_history)
        }
    
    def _validate_price(self, price: float, price_type: str = "price") -> bool:
        """
        Valide qu'un prix est dans des limites raisonnables (utilise fonction centralisée)

        Args:
            price: Prix à valider
            price_type: Type de prix pour logging

        Returns:
            True si valide
        """
        return _validate_price_util(price, price_type)

    def _verify_execution_quality(self, order: Dict, expected_price: Optional[float]) -> bool:
        """
        Vérifie la qualité d'exécution après ordre
        
        Args:
            order: Ordre exécuté
            expected_price: Prix attendu
            
        Returns:
            True si qualité acceptable
        """
        if not expected_price:
            return True  # Pas de vérification pour market orders
        
        executed_price = order.get('average') or order.get('price', 0)
        if not executed_price:
            return True  # Pas de prix disponible
        
        # Calculer le slippage réel
        actual_slippage = safe_divide(abs(executed_price - expected_price), expected_price, default=0.0)
        
        if actual_slippage > self.max_slippage:
            self.logger.warning(
                f"⚠️ Slippage excessif: {actual_slippage:.2%} > {self.max_slippage:.2%}\n"
                f"   Prix attendu: {expected_price:.8f}\n"
                f"   Prix exécuté: {executed_price:.8f}"
            )
            
            # Si slippage trop important, alerter
            if actual_slippage > self.max_slippage * 2:
                self.logger.error(f"🚨 SLIPPAGE CRITIQUE: {actual_slippage:.2%}")
                # Pourrait déclencher un circuit breaker ici
                return False
        
        return True

__all__ = ['OrderExecutor', 'OrderType', 'OrderSide', 'OrderStatus']
