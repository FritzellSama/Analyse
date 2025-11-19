"""
Market Making Strategy - Quantum Trader Pro
Stratégie de market making avec spread capture
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from strategies.base_strategy import BaseStrategy, Signal
from utils.logger import setup_logger
from utils.safety import ensure_minimum_data
from utils.safe_math import safe_divide
from utils.config_helpers import get_nested_config

class MarketMakingStrategy(BaseStrategy):
    """Stratégie Market Making"""
    
    def __init__(self, config: Dict):
        super().__init__('MarketMaking', config)
        self.logger = setup_logger('MarketMaking')

        # Config Market Making - accès sécurisé
        mm_cfg = get_nested_config(config, 'strategies', 'market_making', default={})
        
        # Spread settings
        self.min_spread_bps = mm_cfg.get('min_spread_bps', 10)  # 10 bps = 0.1%
        self.target_spread_bps = mm_cfg.get('target_spread_bps', 20)  # 20 bps = 0.2%
        self.max_spread_bps = mm_cfg.get('max_spread_bps', 50)  # 50 bps = 0.5%
        
        # Order size settings
        self.base_order_size = mm_cfg.get('base_order_size', 0.01)
        self.max_order_size = mm_cfg.get('max_order_size', 0.1)
        self.size_increment = mm_cfg.get('size_increment', 0.005)
        
        # Inventory management
        self.max_inventory = mm_cfg.get('max_inventory', 0.5)  # Max position size
        self.inventory_skew_factor = mm_cfg.get('inventory_skew_factor', 0.5)
        self.target_inventory = 0.0  # Neutral
        
        # Risk limits
        self.max_order_book_imbalance = mm_cfg.get('max_order_book_imbalance', 0.7)
        self.min_volume_24h = mm_cfg.get('min_volume_24h', 1000000)  # $1M
        
        # Timing
        self.order_refresh_seconds = mm_cfg.get('order_refresh_seconds', 30)
        self.last_order_time = None
        
        # State
        self.current_inventory = 0.0
        self.active_bid_orders = {}  # price → size
        self.active_ask_orders = {}  # price → size
        self.last_mid_price = None
        self.spread_history = []
        self.pnl_from_spread = 0.0
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule indicateurs pour market making"""
        df = df.copy()
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        import talib
        
        # Volatilité
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Volume
        df['volume_ma'] = talib.SMA(volume, timeperiod=20)
        df['volume_ratio'] = volume / (df['volume_ma'] + 1e-8)
        
        # Trend (pour éviter market making contre tendance forte)
        df['ema_fast'] = talib.EMA(close, timeperiod=12)
        df['ema_slow'] = talib.EMA(close, timeperiod=26)
        df['trend_strength'] = safe_divide(abs(df['ema_fast'] - df['ema_slow']), df['close'], default=0.0)
        
        return df
    
    def analyze_order_book(self, orderbook: Dict) -> Dict:
        """
        Analyse l'order book
        
        Args:
            orderbook: {'bids': [[price, size], ...], 'asks': [[price, size], ...]}
        
        Returns:
            Dict avec métriques order book
        """
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return {
                'valid': False,
                'reason': 'Empty order book'
            }
        
        # Best bid/ask
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        
        # Spread
        spread = best_ask - best_bid
        spread_bps = safe_divide(spread, mid_price, default=0.0) * 10000
        
        # Order book depth (top 10 levels)
        bid_depth = sum(size for _, size in bids[:10])
        ask_depth = sum(size for _, size in asks[:10])
        total_depth = bid_depth + ask_depth
        
        # Imbalance
        imbalance = (bid_depth - ask_depth) / (total_depth + 1e-8)
        
        # Liquidity score
        liquidity_score = min(bid_depth, ask_depth)
        
        return {
            'valid': True,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'spread': spread,
            'spread_bps': spread_bps,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'imbalance': imbalance,
            'liquidity_score': liquidity_score
        }
    
    def calculate_optimal_quotes(
        self,
        mid_price: float,
        volatility: float,
        inventory: float,
        ob_imbalance: float
    ) -> Tuple[float, float]:
        """
        Calcule prix bid/ask optimaux
        
        Args:
            mid_price: Prix mid
            volatility: Volatilité
            inventory: Inventaire actuel
            ob_imbalance: Imbalance order book
        
        Returns:
            (bid_price, ask_price)
        """
        # Base spread (fonction de volatilité)
        base_spread = self.target_spread_bps / 10000
        volatility_adj = 1.0 + (volatility * 10)  # Plus volatil = spread plus large
        spread = mid_price * base_spread * volatility_adj
        
        # Clamp spread
        min_spread = mid_price * (self.min_spread_bps / 10000)
        max_spread = mid_price * (self.max_spread_bps / 10000)
        spread = max(min_spread, min(spread, max_spread))
        
        # Inventory skew (déplacer quotes pour réduire inventory)
        inventory_ratio = safe_divide(inventory, self.max_inventory, default=0.0)
        skew = inventory_ratio * self.inventory_skew_factor * spread
        
        # Order book imbalance adjustment
        ob_skew = ob_imbalance * 0.3 * spread
        
        # Total skew
        total_skew = skew + ob_skew
        
        # Calculer bid/ask
        bid_price = mid_price - (spread / 2) - total_skew
        ask_price = mid_price + (spread / 2) - total_skew
        
        return bid_price, ask_price
    
    def calculate_order_sizes(
        self,
        inventory: float,
        volatility: float,
        liquidity_score: float
    ) -> Tuple[float, float]:
        """
        Calcule tailles des ordres bid/ask
        
        Returns:
            (bid_size, ask_size)
        """
        # Base size ajusté par liquidité
        liquidity_factor = min(1.0, liquidity_score / 10.0)
        base_size = self.base_order_size * liquidity_factor
        
        # Ajustement par volatilité (moins de size si volatil)
        volatility_factor = safe_divide(1.0, 1.0 + volatility * 10, default=0.1)
        adjusted_size = base_size * volatility_factor

        # Inventory management: plus d'ordres du côté opposé à l'inventory
        inventory_ratio = safe_divide(inventory, self.max_inventory, default=0.0)
        
        if inventory_ratio > 0:  # Long inventory
            # Réduire bid, augmenter ask
            bid_size = adjusted_size * (1.0 - abs(inventory_ratio))
            ask_size = adjusted_size * (1.0 + abs(inventory_ratio))
        elif inventory_ratio < 0:  # Short inventory
            # Augmenter bid, réduire ask
            bid_size = adjusted_size * (1.0 + abs(inventory_ratio))
            ask_size = adjusted_size * (1.0 - abs(inventory_ratio))
        else:
            # Neutral
            bid_size = adjusted_size
            ask_size = adjusted_size
        
        # Clamp sizes
        bid_size = max(self.base_order_size, min(bid_size, self.max_order_size))
        ask_size = max(self.base_order_size, min(ask_size, self.max_order_size))
        
        return bid_size, ask_size
    
    def should_make_market(
        self,
        df: pd.DataFrame,
        ob_analysis: Dict
    ) -> Tuple[bool, Optional[str]]:
        """
        Vérifie si conditions favorables au market making
        
        Returns:
            (should_make, reason_if_not)
        """
        if not ob_analysis.get('valid'):
            return False, ob_analysis.get('reason', 'Invalid order book')
        
        last = df.iloc[-1]
        
        # Check 1: Spread minimum
        if ob_analysis['spread_bps'] < self.min_spread_bps:
            return False, f"Spread trop petit: {ob_analysis['spread_bps']:.1f} bps"
        
        # Check 2: Spread maximum
        if ob_analysis['spread_bps'] > self.max_spread_bps:
            return False, f"Spread trop large: {ob_analysis['spread_bps']:.1f} bps"
        
        # Check 3: Order book imbalance
        if abs(ob_analysis['imbalance']) > self.max_order_book_imbalance:
            return False, f"OB imbalance: {ob_analysis['imbalance']:.2f}"
        
        # Check 4: Trend trop forte (éviter market making contre tendance)
        if 'trend_strength' in last and last['trend_strength'] > 0.02:  # 2%
            return False, "Trend trop forte"
        
        # Check 5: Volatilité excessive
        if 'volatility' in last and last['volatility'] > 0.05:  # 5%
            return False, "Volatilité excessive"
        
        # Check 6: Inventory limit
        if abs(self.current_inventory) >= self.max_inventory:
            return False, f"Inventory limit: {self.current_inventory:.4f}"
        
        return True, None
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Génère signaux market making"""
        signals = []
        
        # Besoin de données tick/order book (simulé avec M1)
        df = data.get('1m') or data.get(self.timeframes['signal'])
        if df is None or not ensure_minimum_data(df, 50, "market_making_generate_signals"):
            return signals

        # Calculer indicateurs
        df = self.calculate_indicators(df)

        last = df.iloc[-1]
        current_price = last['close']
        volatility = last.get('volatility', 0.02)
        
        # Simuler order book analysis
        # En production, utiliser client.get_order_book()
        simulated_orderbook = {
            'bids': [[current_price * 0.9999, 1.0], [current_price * 0.9998, 2.0]],
            'asks': [[current_price * 1.0001, 1.0], [current_price * 1.0002, 2.0]]
        }
        
        ob_analysis = self.analyze_order_book(simulated_orderbook)
        
        # Vérifier si peut faire market making
        can_make, reason = self.should_make_market(df, ob_analysis)
        
        if not can_make:
            self.logger.debug(f"⏸️  Market making pausé: {reason}")
            return signals
        
        # Check timing (refresh orders)
        if self.last_order_time:
            elapsed = (datetime.now() - self.last_order_time).seconds
            if elapsed < self.order_refresh_seconds:
                return signals
        
        # Calculer quotes optimaux
        mid_price = ob_analysis['mid_price']
        ob_imbalance = ob_analysis['imbalance']
        
        bid_price, ask_price = self.calculate_optimal_quotes(
            mid_price=mid_price,
            volatility=volatility,
            inventory=self.current_inventory,
            ob_imbalance=ob_imbalance
        )
        
        # Calculer sizes
        bid_size, ask_size = self.calculate_order_sizes(
            inventory=self.current_inventory,
            volatility=volatility,
            liquidity_score=ob_analysis['liquidity_score']
        )
        
        # Générer signaux BID et ASK
        # Signal BID (acheter)
        if abs(self.current_inventory) < self.max_inventory:
            bid_signal = Signal(
                timestamp=last.name if hasattr(last, 'name') else datetime.now(),
                action='BUY',
                symbol=self.symbol,
                confidence=0.8,
                entry_price=bid_price,
                size=bid_size,
                stop_loss=bid_price * 0.99,  # 1% stop
                take_profit=[(ask_price, 1.0)],  # Take profit à ask
                reason=['market_making', 'bid_quote'],
                metadata={
                    'order_type': 'limit',
                    'post_only': True,  # Maker only
                    'mid_price': mid_price,
                    'spread_bps': ob_analysis['spread_bps'],
                    'inventory': self.current_inventory,
                    'side': 'bid'
                }
            )
            signals.append(bid_signal)
        
        # Signal ASK (vendre)
        if abs(self.current_inventory) < self.max_inventory:
            ask_signal = Signal(
                timestamp=last.name if hasattr(last, 'name') else datetime.now(),
                action='SELL',
                symbol=self.symbol,
                confidence=0.8,
                entry_price=ask_price,
                size=ask_size,
                stop_loss=ask_price * 1.01,  # 1% stop
                take_profit=[(bid_price, 1.0)],  # Take profit à bid
                reason=['market_making', 'ask_quote'],
                metadata={
                    'order_type': 'limit',
                    'post_only': True,
                    'mid_price': mid_price,
                    'spread_bps': ob_analysis['spread_bps'],
                    'inventory': self.current_inventory,
                    'side': 'ask'
                }
            )
            signals.append(ask_signal)
        
        self.last_order_time = datetime.now()
        
        if signals:
            self.logger.info(
                f"📊 Market Making: Bid={bid_price:.2f} ({bid_size:.4f}) | "
                f"Ask={ask_price:.2f} ({ask_size:.4f}) | "
                f"Spread={ob_analysis['spread_bps']:.1f}bps | "
                f"Inventory={self.current_inventory:.4f}"
            )
        
        return signals
    
    def on_order_filled(
        self,
        side: str,
        price: float,
        size: float,
        is_maker: bool
    ):
        """
        Callback quand ordre rempli
        
        Args:
            side: 'buy' ou 'sell'
            price: Prix de fill
            size: Taille remplie
            is_maker: True si maker order
        """
        # Update inventory
        if side.lower() == 'buy':
            self.current_inventory += size
        else:
            self.current_inventory -= size
        
        # Track spread capture (si maker)
        if is_maker and self.last_mid_price:
            if side.lower() == 'buy':
                spread_capture = self.last_mid_price - price
            else:
                spread_capture = price - self.last_mid_price

            spread_capture_pct = safe_divide(spread_capture, self.last_mid_price, default=0.0) * 100
            self.pnl_from_spread += spread_capture * size
            
            self.logger.info(
                f"💰 Spread captured: ${spread_capture:.2f} "
                f"({spread_capture_pct:.3f}%) on {size:.4f} {self.symbol}"
            )
    
    def get_market_making_stats(self) -> Dict:
        """Stats market making"""
        avg_spread = safe_divide(sum(self.spread_history), len(self.spread_history), default=0.0) if self.spread_history else 0

        return {
            'strategy': 'market_making',
            'current_inventory': self.current_inventory,
            'inventory_pct': safe_divide(self.current_inventory, self.max_inventory, default=0.0) * 100,
            'pnl_from_spread': self.pnl_from_spread,
            'avg_spread_bps': avg_spread,
            'num_quotes': len(self.spread_history),
            'active_bid_orders': len(self.active_bid_orders),
            'active_ask_orders': len(self.active_ask_orders),
        }
    
    def reset_inventory(self):
        """Reset inventory (fermer toutes positions)"""
        self.logger.warning(f"🔄 Reset inventory: {self.current_inventory:.4f}")
        self.current_inventory = 0.0
        self.active_bid_orders = {}
        self.active_ask_orders = {}

__all__ = ['MarketMakingStrategy']
