"""
Main Trading Bot - Quantum Trader Pro
Point d'entrée principal pour le trading live
"""

import sys
import time
import signal
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

# Imports locaux
from config import ConfigLoader
from core.binance_client import BinanceClient
from data.data_loader import DataLoader
from data.market_data import MarketData
from strategies.strategy_manager import StrategyManager
from execution.trade_executor import TradeExecutor
from execution.position_manager import PositionManager
from risk.circuit_breaker import CircuitBreaker
from utils.logger import setup_trading_logger
from utils.config_helpers import get_nested_config

class QuantumTraderBot:
    """
    Bot de trading principal qui:
    - Charge la configuration
    - Initialise tous les modules
    - Exécute la boucle de trading
    - Gère les signaux système (Ctrl+C)
    """

    def __init__(self, config_path: Optional[str] = None, custom_client=None):
        """
        Initialise le bot de trading

        Args:
            config_path: Chemin vers config.yaml (optionnel)
            custom_client: Client custom (pour backtest) (optionnel)
        """

        self.logger = setup_trading_logger('QuantumTraderBot')
        self.logger.info("=" * 70)
        self.logger.info("🚀 QUANTUM TRADER PRO - INITIALISATION")
        self.logger.info("=" * 70)

        # État
        self.running = False
        self.initialized = False
        self.last_heartbeat = datetime.now()
        self.heartbeat_interval = 60  # secondes
        self.custom_client = custom_client  # Pour backtest

        # Charger configuration
        try:
            self.config_loader = ConfigLoader(config_path)
            self.config = self.config_loader.config
            self.logger.info("✅ Configuration chargée")
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement configuration: {e}")
            sys.exit(1)

        # Initialiser composants
        self._initialize_components()

        # Gérer Ctrl+C (sauf en mode backtest)
        if not custom_client:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _initialize_components(self) -> None:
        """Initialise tous les composants du bot"""

        try:
            # 1. Client Binance (ou custom pour backtest)
            if self.custom_client:
                self.logger.info("🔌 Utilisation du client custom (mode backtest)...")
                self.client = self.custom_client
            else:
                self.logger.info("🔌 Connexion à Binance...")
                self.client = BinanceClient(self.config)

            # 2. Data Loader
            self.logger.info("📥 Initialisation Data Loader...")
            self.data_loader = DataLoader(self.client, self.config)

            # 3. Market Data
            self.logger.info("📊 Initialisation Market Data...")
            self.market_data = MarketData(self.client, self.config)

            # 4. Strategy Manager
            self.logger.info("🎯 Initialisation Strategy Manager...")
            self.strategy_manager = StrategyManager(self.config)

            # 5. Trade Executor
            self.logger.info("⚡ Initialisation Trade Executor...")
            self.trade_executor = TradeExecutor(self.client, self.config)

            # 6. Position Manager (référence depuis TradeExecutor pour cohérence)
            self.logger.info("📋 Initialisation Position Manager...")
            self.position_manager = self.trade_executor.position_manager

            # 7. Circuit Breaker (référence depuis TradeExecutor pour cohérence)
            self.logger.info("🚦 Initialisation Circuit Breaker...")
            self.circuit_breaker = self.trade_executor.circuit_breaker

            self.initialized = True
            self.logger.info("✅ Tous les composants initialisés")

        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation composants: {e}")
            raise

    def _warmup(self) -> bool:
        """Précharge les données nécessaires"""

        self.logger.info("🔥 Warmup: Chargement données historiques...")

        live_config = self.config.get('live', {})
        warmup_bars = live_config.get('warmup_bars_required', 100)

        # Charger données multi-timeframe
        success = self.data_loader.warmup()

        if not success:
            self.logger.error("❌ Échec warmup")
            return False

        self.logger.info("✅ Warmup terminé")
        return True

    def _trading_loop(self):
        """Boucle principale de trading"""

        live_config = self.config.get('live', {})
        check_interval = live_config.get('check_interval_seconds', 10)

        self.logger.info("🔄 Démarrage boucle de trading...")
        self.logger.info(f"⏱️  Intervalle de vérification: {check_interval}s")

        iteration = 0

        while self.running:
            try:
                iteration += 1

                # 1. Mettre à jour market data
                self.market_data.update_ticker()
                self.market_data.update_orderbook()

                # 2. Mettre à jour données historiques
                # (toutes les N itérations pour économiser API calls)
                if iteration % 6 == 0:  # Toutes les 60 secondes si interval=10s
                    self._update_historical_data()

                # 3. Monitorer positions ouvertes
                self.trade_executor.monitor_positions()

                # 4. Générer signaux
                signals = self._generate_signals()

                # 5. Exécuter signaux
                if signals:
                    for signal in signals:
                        self._execute_signal(signal)

                # 6. Log status périodique
                if iteration % 30 == 0:  # Toutes les 5 minutes
                    self._log_status()

                # 7. Heartbeat check
                self._check_heartbeat()

                # 8. Sauvegarder état périodiquement
                if iteration % 600 == 0:  # Toutes les 100 minutes
                    self._save_state()

                # 9. Attendre
                time.sleep(check_interval)

            except KeyboardInterrupt:
                self.logger.info("⚠️ Interruption détectée")
                break
            except Exception as e:
                self.logger.error(f"❌ Erreur dans boucle trading: {e}")

                # Vérifier si on doit restart
                auto_restart = live_config.get('auto_restart_on_error', True)
                if not auto_restart:
                    break

                # Attendre avant retry
                self.logger.info("⏳ Attente 30s avant retry...")
                time.sleep(30)

    def _update_historical_data(self):
        """Met à jour les données historiques"""

        try:
            symbol = get_nested_config(self.config, 'symbols', 'primary', default='BTC/USDT')
            timeframes = [
                get_nested_config(self.config, 'timeframes', 'trend', default='1h'),
                get_nested_config(self.config, 'timeframes', 'signal', default='5m')
            ]

            for tf in timeframes:
                # Charger nouvelles bougies
                self.data_loader.load_historical_data(
                    symbol=symbol,
                    timeframe=tf,
                    limit=10  # Juste les dernières
                )

        except Exception as e:
            self.logger.error(f"❌ Erreur update données: {e}")

    def _generate_signals(self) -> list:
        """Génère les signaux de trading de toutes les stratégies"""

        try:
            # Charger données
            symbol = get_nested_config(self.config, 'symbols', 'primary', default='BTC/USDT')

            # Données multi-timeframe
            data_1h = self.data_loader.load_historical_data(symbol, '1h', limit=200)
            data_5m = self.data_loader.load_historical_data(symbol, '5m', limit=200)

            if data_1h.empty or data_5m.empty:
                return []

            # Préparer données pour stratégies
            market_data = {
                '1h': data_1h,
                '5m': data_5m,
                'ticker': self.market_data.current_ticker,
                'orderbook': self.market_data.current_orderbook
            }

            # Générer signaux
            all_signals = self.strategy_manager.generate_all_signals(market_data)
            signals = self.strategy_manager.filter_conflicting_signals(all_signals, market_data)

            if signals:
                self.logger.info(f"📡 {len(signals)} signaux générés")

            return signals

        except Exception as e:
            self.logger.error(f"❌ Erreur génération signaux: {e}")
            return []

    def _execute_signal(self, signal):
        """Exécute un signal de trading"""

        try:
            # Log signal
            self.logger.info(
                f"📊 Signal: {signal.symbol} {signal.action.upper()} | "
                f"Stratégie: {signal.strategy} | "
                f"Confiance: {signal.confidence:.2%}"
            )

            # Exécuter
            position = self.trade_executor.execute_signal(signal)

            if position:
                self.logger.info(
                    f"✅ Position ouverte: {position.id} | "
                    f"Taille: {position.size:.8f}"
                )

        except Exception as e:
            import traceback
            self.logger.error(f"❌ Erreur exécution signal: {e}")
            self.logger.error(traceback.format_exc())

    def _log_status(self):
        """Log le statut complet du bot"""

        try:
            # Status trade executor
            status = self.trade_executor.get_status()

            # Positions
            pos_stats = status['position_manager']

            self.logger.daily_summary(
                total_trades=pos_stats['total_trades'],
                winning_trades=pos_stats['winning_trades'],
                losing_trades=pos_stats['losing_trades'],
                win_rate=pos_stats['win_rate'],
                total_pnl=pos_stats['total_pnl']
            )

            # Circuit breaker
            cb_status = status['circuit_breaker']
            if not cb_status['can_trade']:
                self.logger.warning(
                    f"🚫 Circuit Breaker ACTIF: {cb_status['reason']}"
                )

            # Balance
            primary_symbol = get_nested_config(self.config, 'symbols', 'primary', default='BTC/USDT')
            quote_currency = primary_symbol.split('/')[1] if '/' in primary_symbol else 'USDT'
            balance = self.client.get_balance(quote_currency)
            self.logger.info(
                f"💰 Balance: {balance['free']:.2f} {quote_currency} "
                f"(En position: {balance['used']:.2f})"
            )

        except Exception as e:
            self.logger.error(f"❌ Erreur log status: {e}")

    def _signal_handler(self, signum: int, frame) -> None:
        """Gère les signaux système (Ctrl+C)"""
        self.logger.warning("⚠️ Signal d'arrêt reçu")
        self.stop()

    def start(self):
        """Démarre le bot de trading"""

        if not self.initialized:
            self.logger.error("❌ Bot non initialisé")
            return

        if self.running:
            self.logger.warning("⚠️ Bot déjà en cours d'exécution")
            return

        # Warmup
        if not self._warmup():
            self.logger.error("❌ Échec warmup, arrêt")
            return

        # Vérifier mode
        if get_nested_config(self.config, 'exchange', 'primary', 'testnet', default=True):
            self.logger.warning("🧪 MODE TESTNET ACTIVÉ")
        else:
            self.logger.warning("⚠️  MODE PRODUCTION - ARGENT RÉEL!")
            self.logger.warning("⚠️  Appuyez sur Ctrl+C dans les 10s pour annuler")
            time.sleep(10)

        # Démarrer
        self.running = True

        self.logger.info("=" * 70)
        self.logger.info("🚀 BOT DÉMARRÉ - TRADING ACTIF")
        self.logger.info("=" * 70)

        # Boucle principale
        self._trading_loop()

    def stop(self):
        """Arrête le bot proprement"""

        if not self.running:
            return

        self.logger.info("🛑 Arrêt du bot...")
        self.running = False

        # Fermer positions si configuré
        # (normalement on ne ferme PAS automatiquement)
        # self.trade_executor.close_all_positions(reason="shutdown")

        # Sauvegarder état si nécessaire
        self._save_state()

        self.logger.info("✅ Bot arrêté proprement")
        self.logger.info("=" * 70)

    def _save_state(self) -> None:
        """Sauvegarde l'état du bot"""

        try:
            import json
            from pathlib import Path

            # Créer dossier state si nécessaire
            state_dir = Path("state")
            state_dir.mkdir(exist_ok=True)

            # Collecter l'état complet - avec fallback sécurisé
            state = {
                'timestamp': datetime.now().isoformat(),
                'running': self.running,
                'symbol': self.config.get('symbols', {}).get('primary', 'BTC/USDT'),

                # Positions et ordres
                'positions': self.position_manager.get_all_open_positions() if hasattr(self, 'position_manager') else {},
                'open_orders': self.trade_executor.order_executor.get_open_orders() if hasattr(self, 'trade_executor') else [],

                # Circuit breaker status
                'circuit_breaker': self.circuit_breaker.get_status() if hasattr(self, 'circuit_breaker') else {},

                # Performance des stratégies
                'strategies_performance': self.strategy_manager.get_all_performance_stats() if hasattr(self, 'strategy_manager') else {},

                # Stats d'exécution
                'execution_stats': self.trade_executor.order_executor.get_execution_stats() if hasattr(self, 'trade_executor') else {},

                # Balance actuelle
                'current_balance': self.client.get_balance() if hasattr(self, 'client') else 0,

                # Dernières données market
                'last_ticker': self.market_data.current_ticker if hasattr(self, 'market_data') else {},
            }

            # Sauvegarder avec timestamp
            filename = f"state/state_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            # Garder aussi une copie "latest"
            with open("state/state_latest.json", 'w') as f:
                json.dump(state, f, indent=2, default=str)

            self.logger.debug(f"📁 État sauvegardé: {filename}")

            # Nettoyer vieux fichiers (garder 100 derniers)
            state_files = sorted(state_dir.glob("state_*.json"))
            if len(state_files) > 100:
                for old_file in state_files[:-100]:
                    if "latest" not in str(old_file):
                        old_file.unlink()

        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde état: {e}")

    def _check_heartbeat(self):
        """Vérifie que le système est toujours vivant"""

        try:
            current_time = datetime.now()
            time_since_heartbeat = (current_time - self.last_heartbeat).seconds

            # Vérifier la connexion toutes les 60 secondes
            if time_since_heartbeat >= self.heartbeat_interval:
                # Test connexion exchange
                if not self.client.test_connectivity():
                    self.logger.error("❌ Perte de connexion à l'exchange!")
                    # Tenter reconnexion
                    if not self.client.reconnect():
                        self.logger.critical("🚨 Impossible de reconnecter! Arrêt du bot.")
                        self.running = False
                        return False

                # Test connexion base de données si configurée
                # ... autres checks de santé ...

                self.last_heartbeat = current_time
                self.logger.debug(f"💓 Heartbeat OK - {current_time:%H:%M:%S}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Erreur heartbeat: {e}")
            return False

def main() -> None:
    """Point d'entrée principal"""

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║              🚀 QUANTUM TRADER PRO - TRADING BOT 🚀              ║
║                                                                   ║
║                    Production-Ready Trading System                ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    try:
        # Créer et démarrer bot
        bot = QuantumTraderBot()
        bot.start()

    except KeyboardInterrupt:
        print("\n⚠️ Interruption utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
