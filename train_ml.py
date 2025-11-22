#!/usr/bin/env python3
"""
ML Training Script - Quantum Trader Pro
Script standalone pour entraîner les modèles ML

Usage:
    python train_ml.py --data data/collected/BTC_USDT_5m.csv  # Depuis fichier CSV
    python train_ml.py --limit 5000                           # Depuis API (5000 bougies)
    python train_ml.py                                        # Depuis API (défaut: 10000)
"""

import sys
import os
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ConfigLoader
from core.binance_client import BinanceClient
from data.data_loader import DataLoader
from ml_models.trainer import MLTrainer
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config


def load_data_from_csv(csv_path: str, logger) -> pd.DataFrame:
    """Charge les données depuis un fichier CSV"""
    path = Path(csv_path)

    if not path.exists():
        logger.error(f"❌ Fichier non trouvé: {csv_path}")
        logger.error(f"   Chemin absolu: {path.absolute()}")
        return pd.DataFrame()

    logger.info(f"📂 Chargement données depuis: {csv_path}")
    logger.info(f"   Chemin absolu: {path.absolute()}")

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        logger.info(f"✅ {len(df)} bougies chargées depuis fichier CSV")
        return df
    except Exception as e:
        logger.error(f"❌ Erreur lecture CSV: {e}")
        return pd.DataFrame()


def load_data_from_api(data_loader, symbol: str, limit: int, logger) -> pd.DataFrame:
    """Charge les données depuis l'API Binance"""
    logger.info(f"📊 Chargement données depuis API pour {symbol}...")
    logger.info(f"   Limite: {limit} bougies")

    df = data_loader.load_historical_data(
        symbol=symbol,
        timeframe='5m',
        limit=limit
    )

    if not df.empty:
        logger.info(f"✅ {len(df)} bougies chargées depuis API")

    return df


def main():
    """Point d'entrée pour training ML"""

    # Parser les arguments
    parser = argparse.ArgumentParser(
        description='Entraînement des modèles ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python train_ml.py --data data/collected/BTC_USDT_5m.csv
  python train_ml.py --limit 5000
  python train_ml.py
        """
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Chemin vers un fichier CSV de données (recommandé)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10000,
        help='Nombre de bougies à charger depuis API (si pas de --data)'
    )

    args = parser.parse_args()

    # Banner
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           🤖 QUANTUM TRADER PRO - ML TRAINING 🤖                 ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Debug: afficher les arguments
    print(f"📋 Arguments reçus:")
    print(f"   --data  = {args.data}")
    print(f"   --limit = {args.limit}")
    print()

    logger = setup_logger('MLTrainingScript')

    try:
        # 1. Charger config
        logger.info("📋 Chargement configuration...")
        config_loader = ConfigLoader()
        config = config_loader.config

        # 2. Charger données (CSV ou API)
        if args.data is not None:
            # === MODE CSV ===
            logger.info("=" * 60)
            logger.info("📂 MODE: Chargement depuis fichier CSV")
            logger.info("=" * 60)

            df = load_data_from_csv(args.data, logger)

            if df.empty:
                logger.error("❌ Impossible de charger les données depuis le CSV")
                sys.exit(1)
        else:
            # === MODE API ===
            logger.info("=" * 60)
            logger.info("🌐 MODE: Chargement depuis API Binance")
            logger.info("=" * 60)

            # Connexion Binance
            logger.info("🔌 Connexion Binance...")
            client = BinanceClient(config)

            # Data loader
            logger.info("📥 Initialisation Data Loader...")
            data_loader = DataLoader(client, config)

            symbol = get_nested_config(config, 'symbols', 'primary', default='BTC/USDT')
            df = load_data_from_api(data_loader, symbol, args.limit, logger)

            if df.empty:
                logger.error("❌ Impossible de charger les données depuis l'API")
                sys.exit(1)

        # Vérifier les données
        if df.empty:
            logger.error("❌ Aucune donnée chargée")
            sys.exit(1)

        logger.info(f"\n✅ {len(df)} bougies prêtes pour entraînement")
        logger.info(f"   Période: {df.index.min()} → {df.index.max()}")
        logger.info(f"   Colonnes: {list(df.columns)}")

        # 3. Initialiser trainer (besoin de client pour certaines fonctions)
        logger.info("\n🤖 Initialisation ML Trainer...")

        # Créer un client minimal si on utilise CSV
        if args.data is not None:
            client = BinanceClient(config)

        trainer = MLTrainer(client, config)

        # 4. Entraîner modèles
        logger.info("\n🚀 Début entraînement ML...")
        results = trainer.train_all_models(df)

        if not results:
            logger.error("❌ Échec entraînement")
            sys.exit(1)

        # 5. Afficher résumé final
        logger.info("\n" + "=" * 70)
        logger.info("✅ TRAINING ML TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 70)

        logger.info(f"\n📊 Résultats:")
        logger.info(f"   - Samples entraînés: {results.get('samples_trained', 0)}")
        logger.info(f"   - Samples testés: {results.get('samples_tested', 0)}")
        logger.info(f"   - Durée: {results.get('training_time_seconds', 0):.1f}s")

        # XGBoost
        if 'xgboost' in results and 'error' not in results['xgboost']:
            logger.info(f"\n🌳 XGBoost:")
            logger.info(f"   - Accuracy (val): {results['xgboost'].get('val_accuracy', 0):.4f}")
            logger.info(f"   - F1 Score (val): {results['xgboost'].get('val_f1', 0):.4f}")
            logger.info(f"   - Modèle: {results['xgboost'].get('model_path', 'N/A')}")

        # LSTM
        if 'lstm' in results and 'error' not in results['lstm']:
            logger.info(f"\n🧠 LSTM:")
            logger.info(f"   - Accuracy (val): {results['lstm'].get('val_accuracy', 0):.4f}")
            logger.info(f"   - Loss (val): {results['lstm'].get('val_loss', 0):.4f}")
            logger.info(f"   - Modèle: {results['lstm'].get('model_path', 'N/A')}")

        # Ensemble
        if 'ensemble' in results and 'error' not in results['ensemble']:
            logger.info(f"\n🎯 Ensemble:")
            logger.info(f"   - Accuracy: {results['ensemble'].get('accuracy', 0):.4f}")
            logger.info(f"   - F1 Score: {results['ensemble'].get('f1', 0):.4f}")

        logger.info("\n💡 Prochaines étapes:")
        logger.info("   1. Les modèles sont sauvegardés dans ml_models/saved_models/")
        logger.info("   2. Lancez 'python main.py' pour utiliser les modèles ML en live")
        logger.info("   3. Ou 'python paper_trading.py' pour tester en simulation")

    except KeyboardInterrupt:
        print("\n⚠️ Interruption utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
