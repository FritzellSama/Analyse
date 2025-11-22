#!/usr/bin/env python3
"""
Data Collector - Quantum Trader Pro
Collecte et sauvegarde des données historiques pour entraînement ML
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.binance_testnet_client import BinanceTestnetClient
from data.data_loader import DataLoader
from utils.logger import setup_logger


class DataCollector:
    """
    Collecteur de données avec:
    - Téléchargement historique massif
    - Collection continue en temps réel
    - Sauvegarde CSV automatique
    - Fusion de données existantes
    - Support mainnet (lecture seule) pour plus de données historiques
    """

    def __init__(self, config: dict, use_mainnet: bool = False):
        self.config = config
        self.logger = setup_logger('DataCollector')
        self.use_mainnet = use_mainnet

        # Initialiser client et data loader
        if use_mainnet:
            # Utiliser mainnet pour données historiques (lecture seule, pas de clés API requises)
            self.logger.info("🌐 Mode MAINNET activé (lecture seule)")
            from core.binance_mainnet_reader import BinanceMainnetReader
            self.client = BinanceMainnetReader(config)
        else:
            self.client = BinanceTestnetClient(config)

        self.data_loader = DataLoader(self.client, config)

        # Dossier de sauvegarde
        self.data_dir = Path('data/collected')
        self.data_dir.mkdir(parents=True, exist_ok=True)

        mode_str = "MAINNET" if use_mainnet else "TESTNET"
        self.logger.info(f"✅ Data Collector initialisé ({mode_str})")

    def collect_historical(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '5m',
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Collecte un maximum de données historiques

        Args:
            symbol: Paire de trading
            timeframe: Timeframe (1m, 5m, 15m, 1h, etc.)
            days_back: Nombre de jours à récupérer

        Returns:
            DataFrame avec les données
        """
        self.logger.info("=" * 60)
        self.logger.info(f"📥 COLLECTE HISTORIQUE: {symbol} {timeframe}")
        self.logger.info(f"   Période: {days_back} jours")
        self.logger.info("=" * 60)

        # Calculer nombre de bougies
        tf_minutes = self._timeframe_to_minutes(timeframe)
        candles_per_day = (24 * 60) // tf_minutes
        total_candles = candles_per_day * days_back

        # Calculer la date de début (X jours en arrière)
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        self.logger.info(f"📊 Bougies estimées: {total_candles}")
        self.logger.info(f"📅 Date de début: {start_date}")

        # Charger avec pagination depuis la date de début
        df = self.data_loader.load_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=total_candles,
            start_date=start_date
        )

        if df.empty:
            self.logger.error("❌ Aucune donnée collectée")
            return df

        self.logger.info(f"✅ {len(df)} bougies collectées")

        # Sauvegarder
        filepath = self._save_data(df, symbol, timeframe)
        self.logger.info(f"💾 Données sauvegardées: {filepath}")

        return df

    def collect_continuous(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '5m',
        interval_seconds: int = 60,
        max_hours: int = 24
    ):
        """
        Collecte continue de données

        Args:
            symbol: Paire de trading
            timeframe: Timeframe
            interval_seconds: Intervalle entre les collectes
            max_hours: Durée maximum de collecte (0 = infini)
        """
        self.logger.info("=" * 60)
        self.logger.info(f"🔄 COLLECTE CONTINUE: {symbol} {timeframe}")
        self.logger.info(f"   Intervalle: {interval_seconds}s")
        self.logger.info(f"   Durée max: {max_hours}h" if max_hours > 0 else "   Durée: Infinie")
        self.logger.info("=" * 60)

        start_time = time.time()
        max_duration = max_hours * 3600 if max_hours > 0 else float('inf')

        # Charger données existantes
        existing_df = self._load_existing_data(symbol, timeframe)
        total_collected = len(existing_df) if existing_df is not None else 0

        self.logger.info(f"📊 Données existantes: {total_collected} bougies")

        try:
            while time.time() - start_time < max_duration:
                # Récupérer nouvelles données
                new_df = self.data_loader.load_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=100
                )

                if not new_df.empty:
                    # Fusionner
                    if existing_df is None:
                        existing_df = new_df
                    else:
                        existing_df = pd.concat([existing_df, new_df])
                        existing_df = existing_df[~existing_df.index.duplicated(keep='last')]
                        existing_df = existing_df.sort_index()

                    new_count = len(existing_df) - total_collected
                    if new_count > 0:
                        self.logger.info(f"➕ {new_count} nouvelles bougies | Total: {len(existing_df)}")
                        total_collected = len(existing_df)

                        # Sauvegarder périodiquement
                        self._save_data(existing_df, symbol, timeframe)

                # Attendre
                elapsed = time.time() - start_time
                remaining = max_duration - elapsed if max_hours > 0 else float('inf')

                self.logger.debug(
                    f"⏳ Prochaine collecte dans {interval_seconds}s "
                    f"(écoulé: {elapsed/3600:.1f}h)"
                )

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            self.logger.info("\n⚠️ Collecte interrompue par l'utilisateur")

        # Sauvegarde finale
        if existing_df is not None:
            filepath = self._save_data(existing_df, symbol, timeframe)
            self.logger.info(f"💾 Sauvegarde finale: {filepath}")
            self.logger.info(f"📊 Total collecté: {len(existing_df)} bougies")

        return existing_df

    def _save_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Sauvegarde les données en CSV"""
        symbol_clean = symbol.replace('/', '_')
        filename = f"{symbol_clean}_{timeframe}.csv"
        filepath = self.data_dir / filename

        df.to_csv(filepath)
        return str(filepath)

    def _load_existing_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Charge les données existantes si disponibles"""
        symbol_clean = symbol.replace('/', '_')
        filename = f"{symbol_clean}_{timeframe}.csv"
        filepath = self.data_dir / filename

        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df

        return None

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convertit timeframe en minutes"""
        tf_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return tf_map.get(timeframe, 5)

    def get_stats(self, symbol: str = 'BTC/USDT', timeframe: str = '5m') -> dict:
        """Retourne les statistiques des données collectées"""
        df = self._load_existing_data(symbol, timeframe)

        if df is None or df.empty:
            return {'status': 'no_data'}

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_candles': len(df),
            'start_date': df.index.min().isoformat(),
            'end_date': df.index.max().isoformat(),
            'days_covered': (df.index.max() - df.index.min()).days,
            'file_size_mb': (self.data_dir / f"{symbol.replace('/', '_')}_{timeframe}.csv").stat().st_size / 1024 / 1024
        }


def main():
    parser = argparse.ArgumentParser(description='Collecteur de données pour ML')
    parser.add_argument('--mode', choices=['historical', 'continuous', 'stats'],
                       default='historical', help='Mode de collecte')
    parser.add_argument('--symbol', default='BTC/USDT', help='Paire de trading')
    parser.add_argument('--timeframe', default='5m', help='Timeframe')
    parser.add_argument('--days', type=int, default=30, help='Jours de données historiques')
    parser.add_argument('--hours', type=int, default=24, help='Heures de collecte continue')
    parser.add_argument('--interval', type=int, default=60, help='Intervalle entre collectes (s)')
    parser.add_argument('--mainnet', action='store_true',
                       help='Utiliser Binance mainnet (plus de données historiques)')

    args = parser.parse_args()

    # Charger config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    collector = DataCollector(config, use_mainnet=args.mainnet)

    if args.mode == 'historical':
        print(f"\n🚀 Collecte historique: {args.days} jours de {args.symbol} {args.timeframe}")
        print("   Cela peut prendre quelques minutes...\n")

        df = collector.collect_historical(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days_back=args.days
        )

        if not df.empty:
            print(f"\n✅ Collecte terminée!")
            print(f"   Bougies: {len(df)}")
            print(f"   Période: {df.index.min()} → {df.index.max()}")
            print(f"\n💡 Pour entraîner avec ces données:")
            print(f"   python train_ml.py --data data/collected/{args.symbol.replace('/', '_')}_{args.timeframe}.csv")

    elif args.mode == 'continuous':
        print(f"\n🔄 Collecte continue: {args.symbol} {args.timeframe}")
        print(f"   Intervalle: {args.interval}s | Durée: {args.hours}h")
        print("   Ctrl+C pour arrêter\n")

        collector.collect_continuous(
            symbol=args.symbol,
            timeframe=args.timeframe,
            interval_seconds=args.interval,
            max_hours=args.hours
        )

    elif args.mode == 'stats':
        stats = collector.get_stats(args.symbol, args.timeframe)
        print(f"\n📊 Statistiques des données collectées:")
        for key, value in stats.items():
            print(f"   {key}: {value}")


if __name__ == '__main__':
    main()
