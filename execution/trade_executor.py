"""
Trade Executor - Quantum Trader Pro
Orchestrateur principal qui coordonne les signaux, l'exécution et la gestion des positions
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from utils.logger import setup_logger
from utils.calculations import calculate_atr
from utils.safe_math import safe_divide
from utils.config_helpers import get_nested_config
from strategies.base_strategy import Signal
from execution.order_executor import OrderExecutor
from execution.position_manager import PositionManager, Position
from risk.position_sizer import PositionSizer
from risk.stop_loss_manager import StopLossManager
from risk.take_profit_manager import TakeProfitManager
from risk.circuit_breaker import CircuitBreaker
from core.thread_safety import thread_manager

class TradeExecutor:
    """
    Orchestrateur principal de trading qui:
    - Reçoit les signaux des stratégies
    - Calcule la taille de position (Kelly)
    - Définit stop-loss et take-profit
    - Exécute les ordres
    - Gère les positions ouvertes
    - Applique circuit breakers
    """

    def __init__(self, client, config: Dict):
        """
        Initialise l'exécuteur de trades

        Args:
            client: Instance BinanceClient
            config: Configuration complète du bot
        """
        self.client = client
        self.config = config
        self.logger = setup_logger('TradeExecutor')

        # Composants
        self.order_executor = OrderExecutor(client, config)
        self.position_manager = PositionManager(config)
        self.position_sizer = PositionSizer(config)
        self.stop_loss_manager = StopLossManager(config)
        self.take_profit_manager = TakeProfitManager(config)
        self.circuit_breaker = CircuitBreaker(config)

        # Configuration - avec fallback sécurisé
        self.symbol = config.get('symbols', {}).get('primary', 'BTC/USDT')
        self.enabled = True

        # Limites de validation
        self.min_price = 0.00000001  # Prix minimum (satoshi)
        self.max_price = 1000000.0   # Prix maximum raisonnable
        self.max_slippage = config.get('execution', {}).get('max_acceptable_slippage_percent', 0.5) / 100

        self.logger.info("✅ Trade Executor initialisé")

    def execute_signal(self, signal: Signal) -> Optional[Position]:
        """
        Exécute un signal de trading complet

        Args:
            signal: Signal de trading de la stratégie

        Returns:
            Position créée ou None si échec
        """
        # Validation du signal AVANT toute exécution
        if not self.validate_signal(signal):
            self.logger.warning(f"❌ Signal invalide rejeté: {signal}")
            return None
        return self._execute_validated_signal(signal)

    def validate_signal(self, signal: Signal) -> bool:
        """
        Valide complètement un signal avant exécution

        Args:
            signal: Signal à valider

        Returns:
            True si valide, False sinon
        """
        # Validation du prix d'entrée
        if not self.validate_price(signal.entry_price, "entry"):
            return False

        # Validation du stop loss
        if signal.stop_loss:
            if not self.validate_price(signal.stop_loss, "stop_loss"):
                return False

            # Cohérence SL vs prix d'entrée
            if signal.action == 'BUY' and signal.stop_loss >= signal.entry_price:
                self.logger.error(f"Stop loss BUY incohérent: SL={signal.stop_loss} >= Entry={signal.entry_price}")
                return False
            elif signal.action == 'SELL' and signal.stop_loss <= signal.entry_price:
                self.logger.error(f"Stop loss SELL incohérent: SL={signal.stop_loss} <= Entry={signal.entry_price}")
                return False

        # Validation take profit
        if signal.take_profit:
            for tp_price, tp_size in signal.take_profit:
                if not self.validate_price(tp_price, "take_profit"):
                    return False

                # Cohérence TP vs prix d'entrée
                if signal.action == 'BUY' and tp_price <= signal.entry_price:
                    self.logger.error(f"Take profit BUY incohérent: TP={tp_price} <= Entry={signal.entry_price}")
                    return False
                elif signal.action == 'SELL' and tp_price >= signal.entry_price:
                    self.logger.error(f"Take profit SELL incohérent: TP={tp_price} >= Entry={signal.entry_price}")
                    return False

        # Validation confidence
        if not 0 <= signal.confidence <= 1:
            self.logger.error(f"Confidence invalide: {signal.confidence}")
            return False

        # Validation taille si spécifiée
        if signal.size and signal.size <= 0:
            self.logger.error(f"Taille invalide: {signal.size}")
            return False

        # Validation action
        if signal.action not in ['BUY', 'SELL', 'CLOSE']:
            self.logger.error(f"Action invalide: {signal.action}")
            return False

        # Vérification liquidité (orderbook)
        if not self.validate_liquidity(signal.symbol, signal.size or 0.001):
            return False

        return True

    def validate_price(self, price: float, price_type: str = "price") -> bool:
        """
        Valide qu'un prix est dans des limites raisonnables

        Args:
            price: Prix à valider
            price_type: Type de prix (pour logging)

        Returns:
            True si prix valide
        """
        if price is None:
            self.logger.error(f"{price_type} is None")
            return False

        if price <= self.min_price:
            self.logger.error(f"{price_type} trop bas: {price} <= {self.min_price}")
            return False

        if price >= self.max_price:
            self.logger.error(f"{price_type} trop élevé: {price} >= {self.max_price}")
            return False

        # Vérifier contre le dernier prix connu si disponible
        if hasattr(self, 'last_known_price') and self.last_known_price:
            deviation = safe_divide(abs(price - self.last_known_price), self.last_known_price, default=0.0)
            if deviation > 0.5:  # Prix dévie de plus de 50%
                self.logger.warning(f"{price_type} déviation suspecte: {deviation:.1%} du dernier prix connu")
                # On continue mais on log

        return True

    def validate_liquidity(self, symbol: str, size: float) -> bool:
        """
        Vérifie la liquidité disponible dans l'orderbook

        Args:
            symbol: Symbole à trader
            size: Taille de l'ordre

        Returns:
            True si liquidité suffisante
        """
        try:
            orderbook = self.client.fetch_order_book(symbol, limit=20)

            if not orderbook or not orderbook['bids'] or not orderbook['asks']:
                self.logger.error("Orderbook vide ou invalide")
                return False

            # Calculer la liquidité disponible
            # Vérifier que l'orderbook n'est pas vide
            if not orderbook.get('bids') or not orderbook.get('asks'):
                self.logger.warning("Orderbook vide")
                return False

            bid_liquidity = sum(bid[1] for bid in orderbook['bids'][:5])
            ask_liquidity = sum(ask[1] for ask in orderbook['asks'][:5])

            min_liquidity = size * 10  # On veut au moins 10x notre taille

            if bid_liquidity < min_liquidity or ask_liquidity < min_liquidity:
                self.logger.warning(f"Liquidité insuffisante: bid={bid_liquidity:.4f}, ask={ask_liquidity:.4f}, needed={min_liquidity:.4f}")
                return False

            # Vérifier le spread
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            spread = safe_divide(best_ask - best_bid, best_bid, default=0.0)

            if spread > 0.01:  # Spread > 1%
                self.logger.warning(f"Spread trop large: {spread:.2%}")
                return False

            # Mettre à jour le dernier prix connu
            self.last_known_price = (best_bid + best_ask) / 2

            return True

        except Exception as e:
            self.logger.error(f"Erreur validation liquidité: {e}")
            return False

    def _execute_validated_signal(self, signal: Signal) -> Optional[Position]:
        """
        Exécute un signal validé

        Args:
            signal: Signal de trading validé

        Returns:
            Position créée ou None si non exécuté
        """

        # Vérifier si trading activé
        if not self.enabled:
            self.logger.warning("⚠️ Trading désactivé")
            return None

        # Vérifier circuit breaker - avec calcul portfolio complet
        symbol = self.config.get('symbols', {}).get('primary', 'BTC/USDT')
        quote_currency = symbol.split('/')[1] if '/' in symbol else 'USDT'
        current_balance = self.client.get_balance(quote_currency).get('free', 0)

        # Calculer la valeur totale du portfolio (cash + positions)
        open_positions = self.position_manager.get_all_open_positions()
        positions_value = 0.0
        if open_positions:
            try:
                ticker = self.client.get_ticker(symbol)
                current_price = ticker['last']
                for pos in open_positions:
                    positions_value += pos.size * current_price
            except:
                pass  # Si erreur, on continue avec positions_value = 0

        total_portfolio_value = current_balance + positions_value

        # Réinitialiser le circuit breaker si nécessaire en début de backtest
        if not hasattr(self.circuit_breaker, 'initialized') or not self.circuit_breaker.initialized:
            self.circuit_breaker.initialize(total_portfolio_value)

        if not self.circuit_breaker.check(total_portfolio_value):
            cb_status = self.circuit_breaker.get_status()
            self.logger.warning(f"🚫 Circuit breaker actif: {cb_status}")
            return None

        # Vérifier si on peut ouvrir position - avec fallback sécurisé
        counts = self.position_manager.get_positions_count()
        max_positions = self.config.get('risk', {}).get('max_positions_simultaneous', 5)
        if counts['total'] >= max_positions:
            self.logger.warning(
                f"⚠️ Limite de positions atteinte: {counts['total']}"
            )
            return None

        try:
            # Thread-safe execution of critical order placement
            with thread_manager.lock('orders'):
                # 1. Récupérer prix courant
                ticker = self.client.get_ticker()
                current_price = ticker['last']

                # 2. Calculer taille de position - avec parsing sécurisé
                symbol = self.config.get('symbols', {}).get('primary', 'BTC/USDT')
                quote_currency = symbol.split('/')[1] if '/' in symbol else 'USDT'
                balance_data = self.client.get_balance(quote_currency)  # Passer la devise en paramètre
                capital = balance_data.get('free', 0)

            # 3. Calculer stop-loss (une seule fois, cohérent)
            atr = self._calculate_atr(signal.symbol)
            position_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal.strategy}"

            if signal.stop_loss:
                stop_loss = signal.stop_loss
                # Enregistrer le SL custom dans le manager
                self.stop_loss_manager.create_stop_loss(
                    position_id=position_id,
                    entry_price=current_price,
                    side=signal.action,
                    atr=atr,
                    custom_sl=signal.stop_loss
                )
            else:
                # Calculer SL basé sur ATR via le manager
                stop_loss = self.stop_loss_manager.create_stop_loss(
                    position_id=position_id,
                    entry_price=current_price,
                    side=signal.action,
                    atr=atr
                )

            # Vérifier que SL != Entry (éviter division par zéro)
            if abs(stop_loss - current_price) < 0.00000001:
                self.logger.error(f"❌ Stop loss trop proche du prix d'entrée: SL={stop_loss}, Entry={current_price}")
                self.stop_loss_manager.remove_stop(position_id)
                return None

            position_size = self.position_sizer.calculate_size(
                account_balance=capital,
                entry_price=current_price,
                stop_loss=stop_loss,
                confidence=signal.confidence
            )

            if position_size == 0:
                self.logger.warning("⚠️ Taille de position = 0, signal ignoré")
                self.stop_loss_manager.remove_stop(position_id)
                return None

            # 4. Calculer take-profit multi-niveaux (utilise le même position_id)
            take_profit_levels = self.take_profit_manager.create_take_profits(
                position_id=position_id,
                entry_price=current_price,
                side=signal.action,
                size=position_size,
                stop_loss=stop_loss
            )

            # 5. Exécuter ordre d'entrée
            order = self.order_executor.execute_order(
                side=signal.action,
                size=position_size,
                symbol=signal.symbol
            )

            if not order or order.get('status') not in ['filled', 'closed']:
                self.logger.error("❌ Ordre non fill")
                return None

            # 6. Créer position
            avg_price = order.get('average') or order.get('price', current_price)

            position = self.position_manager.open_position(
                symbol=signal.symbol,
                side=signal.action,
                entry_price=avg_price,
                size=position_size,
                stop_loss=stop_loss,
                take_profits=[tp.price for tp in take_profit_levels],  # Liste de prix floats
                strategy=signal.strategy,
                order_id=order['id']
            )

            if not position:
                self.logger.error("❌ Impossible de créer position")
                return None

            # 6b. Mettre à jour les managers SL/TP avec le bon ID de position
            # (l'ID pré-généré était temporaire, maintenant on utilise l'ID réel)
            if position_id != position.id:
                # Transférer le stop-loss vers le nouvel ID
                self.stop_loss_manager.transfer_stop(position_id, position.id)
                # Transférer les take-profits vers le nouvel ID
                self.take_profit_manager.transfer_take_profits(position_id, position.id)

            # 7. Notifier circuit breaker SEULEMENT si position créée avec succès
            self.circuit_breaker.record_trade()

            self.logger.info(
                f"✅ Signal exécuté: {signal.symbol} {signal.action.upper()} "
                f"@ {avg_price:.8f} | Size: {position_size:.8f} | "
                f"SL: {stop_loss:.8f} | TP: {len(take_profit_levels)} niveaux"
            )

            return position

        except Exception as e:
            self.logger.error(f"❌ Erreur exécution signal: {e}")
            return None

    def monitor_positions(self):
        """
        Monitore les positions ouvertes en temps réel
        - Met à jour les prix
        - Vérifie stop-loss
        - Vérifie take-profit
        - Met à jour trailing stops
        """

        open_positions = self.position_manager.get_all_open_positions()

        if not open_positions:
            return

        # Récupérer prix courants
        prices = {}
        for position in open_positions:
            try:
                ticker = self.client.get_ticker(position.symbol)
                prices[position.symbol] = ticker['last']
            except Exception as e:
                self.logger.error(f"❌ Erreur récupération prix {position.symbol}: {e}")

        # Mettre à jour positions
        self.position_manager.update_all_positions(prices)

        # Vérifier chaque position
        for position in open_positions:
            current_price = prices.get(position.symbol)
            if not current_price:
                continue

            # 1. Vérifier stop-loss et mettre à jour trailing stop
            sl_result = self.stop_loss_manager.update(
                position.id, current_price
            )

            if sl_result.get('sl_hit', False):
                sl_price = self.stop_loss_manager.get_current_sl(position.id) or current_price
                self.logger.warning(
                    f"🛑 Stop Loss HIT: {position.symbol} {position.side} "
                    f"Entry={position.entry_price:.8f} SL={sl_price:.8f} "
                    f"Loss=${abs(position.unrealized_pnl):.2f}"
                )

                self._close_position_at_stop_loss(position, sl_price)
                continue

            # 2. Vérifier take-profit (peut avoir plusieurs niveaux hit)
            tp_hits = self.take_profit_manager.check_take_profits(
                position.id, current_price
            )

            for tp_data in tp_hits:
                self._execute_take_profit(position, current_price, tp_data)

            # Note: trailing stop est déjà mis à jour dans sl_result ci-dessus

    def _close_position_at_stop_loss(self, position: Position, sl_price: float):
        """Ferme une position au stop-loss"""
        try:
            # Exécuter ordre de sortie
            order = self.order_executor.execute_order(
                side='sell' if position.side == 'long' else 'buy',
                size=position.size,
                symbol=position.symbol
            )

            if not order:
                self.logger.error("❌ Échec exécution stop-loss")
                return

            # Fermer position
            exit_price = order.get('average') or sl_price
            self.position_manager.close_position(
                position.id,
                exit_price,
                reason="stop_loss",
                order_id=order['id']
            )

            # Notifier circuit breaker (ne PAS appeler on_trade_completed après)
            pnl = exit_price - position.entry_price if position.side == 'long' else position.entry_price - exit_price
            pnl_amount = pnl * position.size
            self.circuit_breaker.record_loss(abs(pnl_amount))

            # Nettoyer managers
            self.stop_loss_manager.remove_stop(position.id)
            self.take_profit_manager.remove_take_profits(position.id)

        except Exception as e:
            self.logger.error(f"❌ Erreur fermeture SL: {e}")

    def _execute_take_profit(self, position: Position, current_price: float, tp_data: Dict):
        """Exécute un take-profit (partiel ou total)"""

        try:
            level = tp_data['level']
            size_to_close = tp_data['size']
            tp_price = tp_data['price']

            # Exécuter ordre de sortie
            order = self.order_executor.execute_order(
                side='sell' if position.side == 'long' else 'buy',
                size=size_to_close,
                symbol=position.symbol
            )

            if not order:
                self.logger.error("❌ Échec exécution TP")
                return

            # Fermeture partielle
            exit_price = order.get('average') or tp_price

            self.position_manager.partial_close_position(
                position.id,
                size_to_close,
                exit_price,
                reason=f"take_profit_{level}",
                order_id=order['id']
            )

            # Log
            self.logger.info(
                f"🎯 Take Profit {level} HIT: {position.symbol} {position.side} "
                f"Entry={position.entry_price:.8f} TP={tp_price:.8f} "
                f"Profit=${position.realized_pnl:.2f}"
            )

            # Si position complètement fermée
            if position.status == 'closed':
                # Notifier circuit breaker (ne PAS appeler on_trade_completed après)
                if position.pnl > 0:
                    self.circuit_breaker.record_win(position.pnl)
                else:
                    self.circuit_breaker.record_loss(abs(position.pnl))

                # Nettoyer managers
                self.stop_loss_manager.remove_stop(position.id)
                self.take_profit_manager.remove_take_profits(position.id)

        except Exception as e:
            self.logger.error(f"❌ Erreur exécution TP: {e}")

    def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calcule l'ATR pour un symbole (utilise fonction centralisée)"""
        try:
            # Récupérer OHLCV
            ohlcv = self.client.get_ohlcv(symbol, '1h', limit=period + 10)

            if not ohlcv or len(ohlcv) < period:
                return 0.0

            # Convertir en DataFrame pour utiliser fonction centralisée
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Utiliser fonction centralisée
            atr_series = calculate_atr(df, period)

            if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]):
                return float(atr_series.iloc[-1])

            return 0.0

        except Exception as e:
            self.logger.warning(f"⚠️ Erreur calcul ATR: {e}")
            return 0.0

    def close_position_manual(self, position_id: str, reason: str = "manual"):
        """
        Ferme manuellement une position

        Args:
            position_id: ID de la position
            reason: Raison de fermeture
        """

        position = self.position_manager.get_position(position_id)
        if not position:
            self.logger.warning(f"⚠️ Position {position_id} introuvable")
            return

        try:
            # Récupérer prix courant
            ticker = self.client.get_ticker(position.symbol)
            current_price = ticker['last']

            # Exécuter ordre
            order = self.order_executor.execute_order(
                side='sell' if position.side == 'long' else 'buy',
                size=position.size,
                symbol=position.symbol
            )

            if not order:
                self.logger.error("❌ Échec fermeture manuelle")
                return

            # Fermer position
            exit_price = order.get('average') or current_price
            self.position_manager.close_position(
                position_id,
                exit_price,
                reason=reason,
                order_id=order['id']
            )

            # Nettoyer managers
            self.stop_loss_manager.remove_stop(position_id)
            self.take_profit_manager.remove_take_profits(position_id)

            self.logger.info(
                f"✅ Position fermée manuellement: {position.symbol} | "
                f"PnL: ${position.pnl:.2f} ({position.pnl_percent:.2f}%)"
            )

        except Exception as e:
            self.logger.error(f"❌ Erreur fermeture manuelle: {e}")

    def close_all_positions(self, reason: str = "emergency"):
        """Ferme toutes les positions (urgence)"""

        open_positions = self.position_manager.get_all_open_positions()

        if not open_positions:
            return

        self.logger.warning(
            f"⚠️ Fermeture de {len(open_positions)} positions: {reason}"
        )

        for position in open_positions:
            self.close_position_manual(position.id, reason)

    def enable_trading(self):
        """Active le trading"""
        self.enabled = True
        self.logger.info("✅ Trading activé")

    def disable_trading(self):
        """Désactive le trading"""
        self.enabled = False
        self.logger.warning("⚠️ Trading désactivé")

    def get_status(self) -> Dict:
        """Retourne le statut complet du trade executor"""

        return {
            'enabled': self.enabled,
            'position_manager': self.position_manager.get_statistics(),
            'order_executor': self.order_executor.get_execution_stats(),
            'circuit_breaker': self.circuit_breaker.get_status(),
            'stop_loss_manager': {
                'active_stops': len(self.stop_loss_manager.active_stops)
            },
            'take_profit_manager': {
                'active_tps': len(self.take_profit_manager.active_tps)
            }
        }
