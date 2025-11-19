"""
ML Training Script - Quantum Trader Pro
Script standalone pour entraîner les modèles ML
"""

import sys
from datetime import datetime
from config import ConfigLoader
from core.binance_client import BinanceClient
from data.data_loader import DataLoader
from ml_models.trainer import MLTrainer
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config

def main():
    """Point d'entrée pour training ML"""
    
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
        
        # 5. Charger données historiques
        symbol = get_nested_config(config, 'symbols', 'primary', default='BTC/USDT')
        logger.info(f"📊 Chargement données historiques pour {symbol}...")
        
        df = data_loader.load_historical_data(
            symbol=symbol,
            timeframe='5m',
            limit=5000
        )
        
        if df.empty:
            logger.error("❌ Aucune donnée chargée")
            sys.exit(1)
        
        logger.info(f"✅ {len(df)} bougies chargées")
        
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
