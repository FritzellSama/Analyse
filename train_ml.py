"""
ML Training Script - Quantum Trader Pro
Script standalone pour entraîner les modèles ML
"""

import sys
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from config import ConfigLoader
from core.binance_client import BinanceClient
from data.data_loader import DataLoader
from ml_models.trainer import MLTrainer
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config

def main():
    """Point d'entrée pour training ML"""

    parser = argparse.ArgumentParser(description='Entraînement des modèles ML')
    parser.add_argument('--data', type=str, help='Chemin vers un fichier CSV de données')
    parser.add_argument('--limit', type=int, default=10000, help='Nombre de bougies (si pas de --data)')
    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           🤖 QUANTUM TRADER PRO - ML TRAINING 🤖                 ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    logger = setup_logger('MLTrainingScript')

    try:
        # 1. Charger config
        logger.info("📋 Chargement configuration...")
        config_loader = ConfigLoader()
        config = config_loader.config

        # 2. Connexion Binance
        logger.info("🔌 Connexion Binance...")
        client = BinanceClient(config)

        # 3. Data loader
        logger.info("📥 Initialisation Data Loader...")
        data_loader = DataLoader(client, config)

        # 4. ML Trainer
        logger.info("🤖 Initialisation ML Trainer...")
        trainer = MLTrainer(client, config)

        # 5. Charger données
        if args.data:
            # Charger depuis fichier CSV
            csv_path = Path(args.data)
            if not csv_path.exists():
                logger.error(f"❌ Fichier non trouvé: {args.data}")
                sys.exit(1)

            logger.info(f"📂 Chargement données depuis: {args.data}")
            df = pd.read_csv(args.data, index_col=0, parse_dates=True)
            logger.info(f"✅ {len(df)} bougies chargées depuis fichier")
        else:
            # Charger depuis exchange
            symbol = get_nested_config(config, 'symbols', 'primary', default='BTC/USDT')
            logger.info(f"📊 Chargement données historiques pour {symbol}...")

            df = data_loader.load_historical_data(
                symbol=symbol,
                timeframe='5m',
                limit=args.limit
            )

        if df.empty:
            logger.error("❌ Aucune donnée chargée")
            sys.exit(1)

        logger.info(f"✅ {len(df)} bougies prêtes pour entraînement")
        logger.info(f"   Période: {df.index.min()} → {df.index.max()}")

        # 6. Entraîner modèles
        logger.info("\n🚀 Début entraînement ML...")
        results = trainer.train_all_models(df)
        
        if not results:
            logger.error("❌ Échec entraînement")
            sys.exit(1)
        
        # 7. Afficher résumé final
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
