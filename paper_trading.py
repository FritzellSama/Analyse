"""
Paper Trading - Quantum Trader Pro
Mode simulation pour tester en conditions réelles sans risque
"""

import sys
import time
import json
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from config import ConfigLoader
from core.binance_client import BinanceClient
from data.data_loader import DataLoader
from data.market_data import MarketData
from strategies.strategy_manager import StrategyManager
from execution.position_manager import PositionManager
from risk.position_sizer import PositionSizer
from risk.stop_loss_manager import StopLossManager
from risk.take_profit_manager import TakeProfitManager
from risk.circuit_breaker import CircuitBreaker
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
from utils.safe_math import safe_divide

class PaperTradingBot:
    """
    Bot de paper trading qui:
    - Simule le trading en temps réel
    - Utilise des données live
    - N'exécute AUCUN ordre réel
    - Track performance comme en prod
    """

    def __init__(self, config_path: Optional[str] = None, initial_balance: float = 300):
        """
        Initialise le bot de paper trading

        Args:
            config_path: Chemin vers config.yaml
            initial_balance: Balance virtuelle de départ
        """

        self.logger = setup_logger('PaperTradingBot')
        self.logger.info("=" * 70)
        self.logger.info("📝 QUANTUM TRADER PRO - PAPER TRADING")
        self.logger.info("=" * 70)

        # État
        self.running = False
        self.initial_balance = initial_balance
        self.virtual_balance = initial_balance

        # Charger config
        try:
            self.config_loader = ConfigLoader(config_path)
            self.config = self.config_loader.config

            # Forcer testnet - utiliser accès sécurisé
            if 'exchange' not in self.config:
                self.config['exchange'] = {}
            if 'primary' not in self.config['exchange']:
                self.config['exchange']['primary'] = {}
            self.config['exchange']['primary']['testnet'] = True

        except Exception as e:
            self.logger.error(f"❌ Erreur chargement config: {e}")
            sys.exit(1)

        # Initialiser composants
        self._initialize_components()

        # Stats
        self.trades_history = []
        self.start_time = None

    def _initialize_components(self):
        """Initialise tous les composants"""

        try:
            # Client (pour data uniquement)
            self.client = BinanceClient(self.config)

            # Data
            self.data_loader = DataLoader(self.client, self.config)
            self.market_data = MarketData(self.client, self.config)

            # Strategies
            self.strategy_manager = StrategyManager(self.config)

            # Position management (virtuel)
            self.position_manager = PositionManager(self.config)

            # Risk management
            self.position_sizer = PositionSizer(self.config)
            self.stop_loss_manager = StopLossManager(self.config)
            self.take_profit_manager = TakeProfitManager(self.config)
            self.circuit_breaker = CircuitBreaker(self.config)

            self.logger.info("✅ Composants initialisés (MODE PAPER)")

        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation: {e}")
            raise

    def _warmup(self):
        """Précharge les données"""

        self.logger.info("🔥 Warmup...")
        success = self.data_loader.warmup()

        if success:
            self.logger.info("✅ Warmup terminé")

        return success

    def _trading_loop(self):
        """Boucle principale de trading virtuel"""

        check_interval = self.config.get('live', {}).get('check_interval_seconds', 10)

        self.logger.info("🔄 Démarrage boucle paper trading...")
        self.logger.info(f"💰 Balance virtuelle: ${self.virtual_balance:.2f}")

        iteration = 0

        while self.running:
            try:
                iteration += 1

                # 1. Update market data
                self.market_data.update_ticker()
                self.market_data.update_orderbook()

                current_price = self.market_data.current_ticker.get('last', 0)

                # 2. Update positions (virtuel)
                self._update_virtual_positions(current_price)

                # 3. Générer signaux
                if iteration % 6 == 0:  # Toutes les 60s
                    signals = self._generate_signals()

                    # 4. Exécuter signaux (virtuel)
                    if signals:
                        for signal in signals:
                            self._execute_virtual_signal(signal, current_price)

                # 5. Log status
                if iteration % 30 == 0:  # Toutes les 5 min
                    self._log_status()

                # 6. Attendre
                time.sleep(check_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"❌ Erreur: {e}")
                time.sleep(10)

    def _generate_signals(self):
        """Génère les signaux de trading"""

        try:
            symbol = get_nested_config(self.config, 'symbols', 'primary', default='BTC/USDT')

            # Charger données
            data_1h = self.data_loader.load_historical_data(symbol, '1h', 200)
            data_5m = self.data_loader.load_historical_data(symbol, '5m', 200)

            if data_1h.empty or data_5m.empty:
                return []

            market_data = {
                '1h': data_1h,
                '5m': data_5m,
                'ticker': self.market_data.current_ticker,
                'orderbook': self.market_data.current_orderbook
            }

            signals = self.strategy_manager.generate_all_signals(market_data)

            if signals:
                self.logger.info(f"📡 {len(signals)} signaux générés")

            return signals

        except Exception as e:
            self.logger.error(f"❌ Erreur génération signaux: {e}")
            return []

    def _execute_virtual_signal(self, signal, current_price: float):
        """Exécute un signal en mode virtuel"""

        # Vérifier circuit breaker
        if not self.circuit_breaker.can_trade():
            return

        # Vérifier limites positions
        counts = self.position_manager.get_positions_count()
        max_positions = get_nested_config(self.config, 'risk', 'max_positions_simultaneous', default=3)
        if counts['total'] >= max_positions:
            return

        # Calculer stop loss d'abord si pas fourni
        if signal.stop_loss:
            stop_loss_for_sizing = signal.stop_loss
        else:
            atr = self._estimate_atr(current_price)
            default_sl_pct = self.config.get('risk', {}).get('default_sl_percent', 0.02)
            atr_sl_mult = self.config.get('risk', {}).get('stop_loss', {}).get('atr_multiplier', 1.5)
            stop_loss_for_sizing = current_price * (1 - default_sl_pct) if atr == 0 else current_price - (atr * atr_sl_mult)

        # Calculer taille
        size = self.position_sizer.calculate_size(
            account_balance=self.virtual_balance,
            entry_price=current_price,
            stop_loss=stop_loss_for_sizing,
            confidence=signal.confidence
        )

        if size == 0:
            return

        # Calculer SL
        if signal.stop_loss:
            stop_loss = signal.stop_loss
        else:
            atr = self._estimate_atr(current_price)
            position_id = f"paper_{int(time.time())}_{signal.strategy}"
            stop_loss = self.stop_loss_manager.create_stop_loss(
                position_id=position_id,
                entry_price=current_price,
                side=signal.action,
                atr=atr
            )

        # Calculer TP
        position_id = f"paper_{int(time.time())}_{signal.strategy}"
        take_profit_levels = self.take_profit_manager.create_take_profits(
            position_id=position_id,
            entry_price=current_price,
            side=signal.action,
            size=size,
            stop_loss=stop_loss
        )

        # Créer position virtuelle
        position = self.position_manager.open_position(
            symbol=signal.symbol,
            side=signal.action,
            entry_price=current_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=[{'price': tp.price, 'level': i+1, 'filled': False} for i, tp in enumerate(take_profit_levels)],
            strategy=signal.strategy,
            order_id=f"PAPER_{int(time.time())}"
        )

        if not position:
            return

        # Déduire du balance virtuel
        cost = size * current_price
        self.virtual_balance -= cost

        # Notifier circuit breaker
        self.circuit_breaker.record_trade()

        self.logger.info(
            f"✅ Position VIRTUELLE ouverte: {signal.symbol} {signal.action.upper()} "
            f"@ {current_price:.8f} | Size: {size:.8f}"
        )

    def _update_virtual_positions(self, current_price: float):
        """Met à jour les positions virtuelles"""

        open_positions = self.position_manager.get_all_open_positions()

        if not open_positions:
            return

        # Mettre à jour prix
        symbol = get_nested_config(self.config, 'symbols', 'primary', default='BTC/USDT')
        prices = {symbol: current_price}
        self.position_manager.update_all_positions(prices)

        # Vérifier SL/TP
        for position in open_positions:
            # Stop-loss
            sl_hit, sl_price = self.stop_loss_manager.check_stop_loss(
                position.id, current_price
            )

            if sl_hit:
                self._close_virtual_position(position, sl_price, 'stop_loss')
                continue

            # Take-profit
            tp_hit, tp_data = self.take_profit_manager.check_take_profit(
                position.id, current_price
            )

            if tp_hit:
                self._partial_close_virtual_position(
                    position, current_price, tp_data
                )

            # Trailing stop
            self.stop_loss_manager.update_trailing_stop(
                position.id, current_price
            )

    def _close_virtual_position(self, position, exit_price: float, reason: str):
        """Ferme une position virtuelle"""

        # Récupérer capital
        cost = position.size * exit_price
        self.virtual_balance += cost

        # Fermer position
        self.position_manager.close_position(
            position.id,
            exit_price,
            reason=reason,
            order_id=f"PAPER_CLOSE_{int(time.time())}"
        )

        # Notifier circuit breaker
        if position.pnl > 0:
            self.circuit_breaker.record_win(position.pnl)
        else:
            self.circuit_breaker.record_loss(abs(position.pnl))

        # Nettoyer managers
        self.stop_loss_manager.remove_position(position.id)
        self.take_profit_manager.remove_position(position.id)

        # Enregistrer trade
        self.trades_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': position.symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': position.pnl,
            'pnl_percent': position.pnl_percent,
            'duration_minutes': position.duration_minutes,
            'reason': reason
        })

        self.logger.info(
            f"{'✅' if position.pnl > 0 else '❌'} Position VIRTUELLE fermée: "
            f"PnL: ${position.pnl:.2f} ({position.pnl_percent:.2f}%) | "
            f"Raison: {reason}"
        )

    def _partial_close_virtual_position(self, position, current_price: float, tp_data: Dict):
        """Fermeture partielle virtuelle"""

        size_to_close = tp_data['size']
        level = tp_data['level']

        # Récupérer capital partiel
        cost = size_to_close * current_price
        self.virtual_balance += cost

        # Fermeture partielle
        self.position_manager.partial_close_position(
            position.id,
            size_to_close,
            current_price,
            reason=f"take_profit_{level}",
            order_id=f"PAPER_TP_{int(time.time())}"
        )

        self.logger.info(
            f"✅ TP{level} atteint (VIRTUEL): {position.symbol} | "
            f"PnL partiel: ${position.realized_pnl:.2f}"
        )

    def _estimate_atr(self, current_price: float) -> float:
        """Estime l'ATR (utilise config ou valeur par défaut)"""
        atr_fallback_pct = self.config.get('risk', {}).get('atr_fallback_percent', 0.02)
        return current_price * atr_fallback_pct

    def _log_status(self):
        """Log le statut complet"""

        stats = self.position_manager.get_statistics()

        # Calculer equity totale
        unrealized_pnl = stats['unrealized_pnl']
        total_equity = self.virtual_balance + unrealized_pnl

        # ROI
        roi = safe_divide(total_equity - self.initial_balance, self.initial_balance, default=0.0) * 100

        self.logger.daily_summary(
            total_trades=stats['total_trades'],
            winning_trades=stats['winning_trades'],
            losing_trades=stats['losing_trades'],
            win_rate=stats['win_rate'],
            total_pnl=stats['total_pnl']
        )

        self.logger.info(
            f"💰 Balance: ${self.virtual_balance:.2f} | "
            f"Unrealized PnL: ${unrealized_pnl:.2f} | "
            f"Total Equity: ${total_equity:.2f} | "
            f"ROI: {roi:.2f}%"
        )

    def start(self):
        """Démarre le paper trading"""

        if self.running:
            self.logger.warning("⚠️ Déjà en cours")
            return

        # Warmup
        if not self._warmup():
            self.logger.error("❌ Échec warmup")
            return

        self.logger.warning("📝 MODE PAPER TRADING - AUCUN ORDRE RÉEL!")
        self.logger.info(f"💰 Balance virtuelle: ${self.virtual_balance:.2f}")

        self.running = True
        self.start_time = datetime.now()

        self.logger.info("=" * 70)
        self.logger.info("🚀 PAPER TRADING DÉMARRÉ")
        self.logger.info("=" * 70)

        # Boucle principale
        self._trading_loop()

    def stop(self):
        """Arrête le paper trading"""

        if not self.running:
            return

        self.logger.info("🛑 Arrêt paper trading...")
        self.running = False

        # Fermer positions
        current_price = self.market_data.current_ticker.get('last', 0)
        for position in self.position_manager.get_all_open_positions():
            self._close_virtual_position(position, current_price, 'shutdown')

        # Générer rapport final
        self._generate_final_report()

        self.logger.info("✅ Paper trading arrêté")

    def _generate_final_report(self):
        """Génère le rapport final"""

        stats = self.position_manager.get_statistics()

        # Calculer durée
        duration = datetime.now() - self.start_time if self.start_time else None

        # Equity finale
        final_equity = self.virtual_balance + stats['unrealized_pnl']
        total_return = safe_divide(final_equity - self.initial_balance, self.initial_balance, default=0.0) * 100

        self.logger.info("=" * 70)
        self.logger.info("📊 RAPPORT FINAL - PAPER TRADING")
        self.logger.info("=" * 70)

        if duration:
            self.logger.info(f"⏱️  Durée: {duration}")

        self.logger.info(f"💰 Balance initiale: ${self.initial_balance:.2f}")
        self.logger.info(f"💰 Balance finale: ${self.virtual_balance:.2f}")
        self.logger.info(f"💰 Equity totale: ${final_equity:.2f}")
        self.logger.info(f"📈 Return: {total_return:.2f}%")
        self.logger.info("")

        self.logger.info(f"📊 Total trades: {stats['total_trades']}")
        self.logger.info(f"✅ Winning: {stats['winning_trades']}")
        self.logger.info(f"❌ Losing: {stats['losing_trades']}")
        self.logger.info(f"🎯 Win Rate: {stats['win_rate']:.2f}%")
        self.logger.info("")

        self.logger.info(f"💵 Avg Win: ${stats['avg_win']:.2f}")
        self.logger.info(f"💸 Avg Loss: ${stats['avg_loss']:.2f}")
        self.logger.info(f"⚡ Profit Factor: {stats['profit_factor']:.2f}")

        self.logger.info("=" * 70)

        # Sauvegarder trades
        self._save_trades()

    def _save_trades(self):
        """Sauvegarde l'historique des trades"""

        if not self.trades_history:
            return

        filename = f"paper_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = Path('logs') / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(self.trades_history, f, indent=2)

            self.logger.info(f"💾 Trades sauvegardés: {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde trades: {e}")

def main():
    """Point d'entrée pour paper trading"""

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           📝 QUANTUM TRADER PRO - PAPER TRADING 📝               ║
║                                                                   ║
║                    Simulation sans risque                         ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    try:
        bot = PaperTradingBot(initial_balance=300)
        bot.start()

    except KeyboardInterrupt:
        print("\n⚠️ Interruption utilisateur")
        if 'bot' in locals():
            bot.stop()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
