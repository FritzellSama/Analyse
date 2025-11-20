"""
Replay Backtest - Quantum Trader Pro
Système de backtesting qui utilise le code de production (main.py) avec des données historiques
"""

import sys
import signal
from pathlib import Path

# Ajouter le répertoire parent au path pour imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd

from config import ConfigLoader
from core.virtual_binance_client import VirtualBinanceClient
from data.data_loader import DataLoader
from core.binance_client import BinanceClient
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
from utils.safe_math import safe_divide
from utils.calculations import timeframe_to_minutes

# ============================================================================
# GESTION GLOBALE DE L'INTERRUPTION (CTRL+C)
# ============================================================================
interrupted = False

def signal_handler(sig, frame):
    """Gestionnaire pour Ctrl+C - Arrêt immédiat et propre"""
    global interrupted
    interrupted = True
    print("\n\n🛑 INTERRUPTION DÉTECTÉE - Arrêt immédiat...\n")
    sys.exit(0)

# Installer les gestionnaires de signaux
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class ReplayBacktest:
    """
    Backtester qui:
    1. Charge les données historiques
    2. Crée un VirtualBinanceClient
    3. Lance main.py en mode replay
    4. Avance le temps bougie par bougie
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le replay backtest

        Args:
            config_path: Chemin vers config.yaml
        """
        self.running = True  # Flag d'arrêt propre
        self.logger = setup_logger('ReplayBacktest')
        self.logger.info("=" * 70)
        self.logger.info("🔄 QUANTUM TRADER PRO - REPLAY BACKTEST")
        self.logger.info("=" * 70)

        # Charger config
        try:
            self.config_loader = ConfigLoader(config_path)
            self.config = self.config_loader.config
            self.backtest_config = self.config.get('backtest', {})
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement config: {e}")
            sys.exit(1)

        # Paramètres backtest
        data_config = self.backtest_config.get('data', {})
        self.start_date = data_config.get('start_date', '2023-01-01')
        self.end_date = data_config.get('end_date', '2024-11-08')
        self.warmup_bars = data_config.get('warmup_bars', 100)

        # État
        self.historical_data = {}
        self.virtual_client = None
        self.bot = None

    def load_historical_data(self):
        """Charge les données historiques depuis Binance"""

        self.logger.info(f"📥 Chargement données: {self.start_date} → {self.end_date}")

        # Utiliser un vrai client juste pour charger les données
        temp_client = BinanceClient(self.config)
        data_loader = DataLoader(temp_client, self.config)

        # Charger pour chaque timeframe configuré
        timeframes_config = self.config.get('timeframes', {})
        timeframes = [
            timeframes_config.get('trend', '1h'),
            timeframes_config.get('signal', '5m'),
            timeframes_config.get('micro', '1m')
        ]

        for tf in timeframes:
            # Vérifier interruption
            if interrupted:
                raise KeyboardInterrupt("Interruption pendant le chargement")

            try:
                self.logger.info(f"📥 Chargement {tf}...")

                # Charger les données historiques
                symbol = get_nested_config(self.config, 'symbols', 'primary', default='BTC/USDT')
                df = data_loader.load_historical_data(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    limit=5000  # Plus de données pour le backtest
                )

                if df is not None and len(df) > 0:
                    self.historical_data[tf] = df
                    self.logger.info(f"✅ {len(df)} bougies chargées pour {tf}")
                else:
                    self.logger.warning(f"⚠️ Aucune donnée pour {tf}")

            except Exception as e:
                self.logger.error(f"❌ Erreur chargement {tf}: {e}")
                raise

        if not self.historical_data:
            raise Exception("Aucune donnée historique chargée")

        return self.historical_data

    def prepare_virtual_client(self):
        """Prépare le client virtuel avec les données"""

        self.logger.info("🔧 Préparation du client virtuel...")

        # Créer client virtuel
        self.virtual_client = VirtualBinanceClient(self.config)

        # Charger les données
        self.virtual_client.load_historical_data(self.historical_data)

        self.logger.info("✅ Client virtuel prêt")

        return self.virtual_client

    def run_replay(self):
        """
        Lance le replay en utilisant main.py
        Avance le temps bougie par bougie
        """

        self.logger.info("🔄 Démarrage du replay...")

        # Import ici pour éviter les imports circulaires
        from main import QuantumTraderBot

        # Créer le bot avec le client virtuel
        self.bot = QuantumTraderBot(config_path=None)

        # REMPLACER le client réel par le virtuel
        self.bot.client = self.virtual_client
        self.bot.data_loader.client = self.virtual_client
        self.bot.market_data.client = self.virtual_client
        self.bot.trade_executor.client = self.virtual_client
        self.bot.trade_executor.order_executor.client = self.virtual_client

        # Initialiser le circuit breaker avec la balance de départ
        initial_balance = float(get_nested_config(self.config, 'backtest', 'simulation', 'initial_balance', default=300))
        self.bot.trade_executor.circuit_breaker.initialize(initial_balance)
        self.logger.info(f"🔒 Circuit breaker initialisé: ${initial_balance}")

        # Obtenir la timeframe principale
        main_tf = get_nested_config(self.config, 'timeframes', 'trend', default='1h')

        # Vérifier qu'on a assez de données
        min_required_bars = 300  # Minimum pour backtest significatif
        if len(self.historical_data.get(main_tf, [])) < min_required_bars:
            # Fallback vers une timeframe avec plus de données
            for tf in ['1h', '5m', '1m']:
                if tf in self.historical_data and len(self.historical_data[tf]) >= min_required_bars:
                    main_tf = tf
                    self.logger.warning(f"⚠️ Utilisation de {tf} comme timeframe principale (données insuffisantes en trend)")
                    break

        df_main = self.historical_data[main_tf]

        self.logger.info(f"📊 Timeframe principale: {main_tf}")
        self.logger.info(f"📊 {len(df_main)} bougies à traiter")
        self.logger.info("🔄 Démarrage boucle de trading...")
        self.logger.info("💡 Appuyez sur Ctrl+C pour arrêter proprement")

        # Boucle principale: une bougie à la fois sur la TF principale
        total_bars = len(df_main)

        for i in range(self.warmup_bars, total_bars):
            # Vérifier interruption globale
            if interrupted or not self.running:
                self.logger.warning("🛑 Arrêt demandé")
                break

            if i in [10, 20, 50, 100, 110]:
                self.logger.info(f"🚀 DÉBUT ITÉRATION i={i}")

            try:
                # ========== MISE À JOUR DES INDEX ==========
                for tf in self.historical_data.keys():
                    current_time = df_main.iloc[i].name
                    df_tf = self.historical_data[tf]
                    try:
                        # Trouver l'index le plus proche du timestamp actuel
                        new_idx = df_tf.index.get_indexer([current_time], method='ffill')[0]
                        if new_idx < 0:
                            new_idx = 0
                    except:
                        # Fallback : utiliser le ratio mais limiter à la taille du DataFrame
                        ratio = timeframe_to_minutes(main_tf) / timeframe_to_minutes(tf)
                        new_idx = min(int(i * ratio), len(df_tf) - 1)

                    self.virtual_client.current_index[tf] = new_idx

                    # Debug à i=110
                    if i == 110:
                        ratio = timeframe_to_minutes(main_tf) / timeframe_to_minutes(tf)
                        self.logger.info(f"🔍 INDEX: i={i}, TF={tf}, ratio={ratio:.4f}, new_idx={new_idx}")

                # ========== TIMESTAMP ET PRIX ==========
                current_bar = df_main.iloc[i]
                current_time = current_bar.name
                current_price = current_bar['close']
                self.virtual_client.current_timestamp = current_time

                # ========== LOG PÉRIODIQUE ==========
                if i % 100 == 0:
                    progress = (i - self.warmup_bars) / (total_bars - self.warmup_bars) * 100
                    balance = self.virtual_client.virtual_balance
                    self.logger.info(
                        f"📊 Progress: {progress:.1f}% | "
                        f"Date: {current_time.strftime('%Y-%m-%d %H:%M')} | "
                        f"Prix: ${current_price:.2f} | "
                        f"Balance: ${balance:.2f}"
                    )

                # ========== CONSTRUCTION MARKET_DATA ==========
                market_data = {}
                timeframes_cfg = get_nested_config(self.config, 'timeframes', default={})
                tf_mapping = {
                    'trend': timeframes_cfg.get('trend', '1h'),
                    'signal': timeframes_cfg.get('signal', '5m'),
                    'micro': timeframes_cfg.get('micro', '1m')
                }

                symbol = get_nested_config(self.config, 'symbols', 'primary', default='BTC/USDT')

                for tf_name in ['trend', 'signal', 'micro']:
                    actual_tf = tf_mapping[tf_name]

                    # Debug avant fetch
                    if i == 110:
                        current_idx = self.virtual_client.current_index.get(actual_tf, 0)
                        self.logger.info(f"🔍 AVANT FETCH: tf={actual_tf}, current_idx={current_idx}")

                    ohlcv = self.virtual_client.get_ohlcv(symbol, actual_tf, limit=200)

                    # Debug après fetch
                    if i == 110:
                        self.logger.info(f"🔍 APRÈS FETCH: tf={actual_tf}, got {len(ohlcv)} lignes OHLCV")

                    if not ohlcv or len(ohlcv) == 0:
                        self.logger.warning(f"⚠️ Pas de données pour {actual_tf} à i={i}")
                        continue

                    df_tf = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_tf['timestamp'] = pd.to_datetime(df_tf['timestamp'], unit='ms')
                    df_tf.set_index('timestamp', inplace=True)
                    market_data[actual_tf] = df_tf

                # Debug market_data final
                if i == 110:
                    self.logger.info(f"🔍 MARKET_DATA FINAL: {len(market_data)} timeframes")
                    for tf_key, df in market_data.items():
                        self.logger.info(f"🔍   → {tf_key}: {len(df)} lignes")

                # ========== VÉRIFICATION DONNÉES COMPLÈTES ==========
                if len(market_data) < 3:
                    if i == 110:
                        self.logger.warning(f"⚠️ Market data incomplet: {len(market_data)}/3 - SKIP iteration")
                    continue

                # ========== GÉNÉRATION SIGNAUX ==========
                signals = self.bot.strategy_manager.generate_all_signals(market_data)

                if i == 110 or (signals and i % 10 == 0):
                    self.logger.info(f"🔍 SIGNAUX: i={i}, {len(signals)} générés")

                filtered_signals = self.bot.strategy_manager.filter_conflicting_signals(signals)

                if i == 110 or (filtered_signals and i % 10 == 0):
                    self.logger.info(f"🔍 FILTRÉS: i={i}, {len(filtered_signals)} après filtrage")

                # ========== EXÉCUTION ==========
                for signal in filtered_signals:
                    # Check interruption avant chaque signal
                    if interrupted:
                        raise KeyboardInterrupt("Arrêt pendant exécution signal")

                    try:
                        self.logger.info(f"🎯 EXÉCUTION SIGNAL: {signal.symbol} {signal.action} @ i={i}")
                        position = self.bot.trade_executor.execute_signal(signal)
                        if position:
                            self.logger.info(f"✅ POSITION CRÉÉE: {position}")
                    except KeyboardInterrupt:
                        self.running = False
                        raise  # Propager immédiatement
                    except Exception as e:
                        self.logger.error(f"❌ Erreur exécution signal: {e}")

            except KeyboardInterrupt:
                self.running = False
                self.logger.warning("⚠️ Interruption utilisateur détectée")
                raise  # Propager pour arrêt propre
            except Exception as e:
                current_time_str = current_time.strftime('%Y-%m-%d %H:%M') if 'current_time' in locals() else 'unknown'
                self.logger.error(f"❌ Erreur à la bougie {i} ({current_time_str}): {e}")
                import traceback
                traceback.print_exc()
                continue

        # Fin du backtest
        self.logger.info("🏁 Replay terminé")

        # Fermer toutes les positions ouvertes au dernier prix
        self._close_all_open_positions()

        self._print_results()

    def _close_all_open_positions(self):
        """Ferme toutes les positions ouvertes à la fin du backtest"""
        if not self.bot or not hasattr(self.bot, 'trade_executor'):
            return

        position_manager = self.bot.trade_executor.position_manager
        open_positions = position_manager.get_all_open_positions()

        if not open_positions:
            self.logger.info("📭 Aucune position ouverte à fermer")
            return

        self.logger.info(f"🔒 Fermeture de {len(open_positions)} positions ouvertes...")

        # Obtenir le dernier prix
        try:
            ticker = self.virtual_client.get_ticker()
            last_price = ticker.get('last', 0)
        except Exception:
            # Fallback: utiliser le prix de la dernière position
            last_price = open_positions[0].entry_price if open_positions else 0

        for position in open_positions:
            try:
                position_manager.close_position(
                    position_id=position.id,
                    close_price=last_price,
                    reason="Fin du backtest"
                )
                self.logger.debug(f"✅ Position {position.id} fermée @ ${last_price:.2f}")
            except Exception as e:
                self.logger.error(f"❌ Erreur fermeture position {position.id}: {e}")

        self.logger.info(f"✅ {len(open_positions)} positions fermées")


    def _print_results(self):
        """Affiche les résultats du backtest"""

        self.logger.info("=" * 70)
        self.logger.info("📊 RÉSULTATS DU REPLAY BACKTEST")
        self.logger.info("=" * 70)

        # Stats du client virtuel
        stats = self.virtual_client.get_statistics()

        initial_balance = float(get_nested_config(self.backtest_config, 'simulation', 'initial_balance', default=1000))
        final_balance = stats['final_balance']
        pnl = final_balance - initial_balance
        pnl_pct = safe_divide(pnl, initial_balance, default=0.0) * 100

        self.logger.info(f"💰 Balance initiale: ${initial_balance:.2f}")
        self.logger.info(f"💰 Balance finale: ${final_balance:.2f}")
        self.logger.info(f"📈 PnL Total: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        self.logger.info("")

        # Stats du trade executor
        if self.bot and hasattr(self.bot, 'trade_executor'):
            status = self.bot.trade_executor.get_status()
            pos_stats = status.get('position_manager', {})

            total_trades = pos_stats.get('total_trades', 0)
            winning = pos_stats.get('winning_trades', 0)
            losing = pos_stats.get('losing_trades', 0)
            win_rate = pos_stats.get('win_rate', 0)
            total_pnl = pos_stats.get('total_pnl', 0)

            self.logger.info(f"📊 Total trades: {total_trades}")
            self.logger.info(f"✅ Winning: {winning}")
            self.logger.info(f"❌ Losing: {losing}")
            self.logger.info(f"🎯 Win Rate: {win_rate:.2f}%")
            self.logger.info(f"💵 Total PnL: ${total_pnl:.2f}")

        self.logger.info("=" * 70)


def main():
    """Point d'entrée"""

    print("\n" + "=" * 70)
    print("🔄 QUANTUM TRADER PRO - REPLAY BACKTEST")
    print("=" * 70 + "\n")
    print("💡 Appuyez sur Ctrl+C à tout moment pour arrêter\n")

    try:
        # Créer le backtest
        backtest = ReplayBacktest()

        # Charger données
        backtest.load_historical_data()

        # Préparer client virtuel
        backtest.prepare_virtual_client()

        # Lancer le replay
        backtest.run_replay()

        print("\n✅ Backtest terminé avec succès\n")

    except KeyboardInterrupt:
        print("\n⚠️  Backtest interrompu par l'utilisateur\n")
        print("📊 Résultats partiels affichés ci-dessus\n")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
