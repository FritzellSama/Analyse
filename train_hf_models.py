#!/usr/bin/env python3
"""
HF Models Training Script - Quantum Trader Pro
Script pour entraîner XGBoost + LightGBM pour le système 90% Win Rate

CHANGEMENTS vs train_ml.py :
1. Target: threshold=0.003 (0.3%) au lieu de 0.001 (0.1%)
2. Entraîne LightGBM (pas LSTM)
3. SMOTE pour équilibrer les classes
4. Optimisé pour le ML Signal Filter

Usage:
    python train_hf_models.py --data data/collected/BTC_USDT_5m.csv
    python train_hf_models.py --limit 5000
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ConfigLoader
from core.binance_client import BinanceClient
from data.data_loader import DataLoader
from ml_models.xgboost_model import XGBoostModel
from ml_models.lightgbm_model import LightGBMModel
from ml_models.feature_engineering import FeatureEngineer
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config

# SMOTE pour équilibrage des classes
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️ imblearn non installé. Installer avec: pip install imbalanced-learn")


class HFModelTrainer:
    """
    Trainer optimisé pour le système High Frequency

    Différences avec MLTrainer original :
    - threshold = 0.003 (0.3%) au lieu de 0.001 (0.1%)
    - Pas de LSTM (remplacé par LightGBM)
    - Focus sur la confirmation de signaux
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger('HFModelTrainer')

        # Configuration
        ml_config = config.get('ml', {})
        training_config = ml_config.get('training', {})

        self.min_samples = training_config.get('min_samples', 500)
        self.validation_split = training_config.get('validation_split', 0.2)
        self.test_split = training_config.get('test_split', 0.1)

        # IMPORTANT: Nouveaux paramètres de target
        self.horizon_bars = 5           # 5 bougies (25 min sur 5m)
        self.target_threshold = 0.003   # 0.3% = mouvement significatif (pas 0.1%!)

        # Composants
        self.feature_engineer = FeatureEngineer(config)
        self.xgboost = XGBoostModel(config)
        self.lightgbm = LightGBMModel(config)

        self.logger.info("✅ HF Model Trainer initialisé")
        self.logger.info(f"   Target: {self.target_threshold*100:.1f}% sur {self.horizon_bars} bougies")

    def create_better_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée un target MEILLEUR que l'original

        CHANGEMENTS :
        - threshold = 0.3% au lieu de 0.1%
        - Ajoute des features de confirmation
        """

        df = df.copy()

        # Future return sur horizon
        df['future_return'] = df['close'].shift(-self.horizon_bars) / df['close'] - 1

        # Target binaire avec seuil plus élevé
        df['target'] = (df['future_return'] > self.target_threshold).astype(int)

        # Analyser la distribution
        target_counts = df['target'].value_counts()
        total = len(df.dropna())

        if total > 0:
            pct_positive = target_counts.get(1, 0) / total * 100
            pct_negative = target_counts.get(0, 0) / total * 100

            self.logger.info(f"📊 Distribution target:")
            self.logger.info(f"   - UP (>+{self.target_threshold*100:.1f}%): {pct_positive:.1f}%")
            self.logger.info(f"   - DOWN/NEUTRAL: {pct_negative:.1f}%")

            # Avertissement si déséquilibre extrême
            if pct_positive < 20 or pct_positive > 80:
                self.logger.warning(f"⚠️ Déséquilibre de classes détecté!")

        return df

    def train_all(self, df: pd.DataFrame) -> dict:
        """Entraîne XGBoost + LightGBM"""

        self.logger.info("=" * 70)
        self.logger.info("🚀 DÉBUT ENTRAÎNEMENT HF MODELS")
        self.logger.info("=" * 70)

        start_time = datetime.now()

        # 1. Feature engineering
        self.logger.info("\n🔨 Feature engineering...")
        df_features = self.feature_engineer.generate_features(df)

        if df_features.empty:
            self.logger.error("❌ Feature engineering a échoué")
            return {}

        # 2. Créer target AMÉLIORÉ
        self.logger.info(f"\n🎯 Création target (threshold={self.target_threshold*100:.1f}%)...")
        df_features = self.create_better_target(df_features)

        # 3. Nettoyer NaN
        df_features = df_features.dropna()

        if len(df_features) < self.min_samples:
            self.logger.error(f"❌ Pas assez de données: {len(df_features)} < {self.min_samples}")
            return {}

        self.logger.info(f"✅ {len(df_features)} samples prêts")

        # 4. Séparer features et target
        feature_names = self.feature_engineer.get_feature_names(df_features)
        X = df_features[feature_names]
        y = df_features['target']

        # 5. Split train/test (garder ordre temporel)
        test_size = int(len(X) * self.test_split)
        X_train = X.iloc[:-test_size]
        y_train = y.iloc[:-test_size]
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]

        self.logger.info(f"📊 Split: Train={len(X_train)} | Test={len(X_test)}")

        # 6. SMOTE pour équilibrer les classes d'entraînement
        if SMOTE_AVAILABLE:
            self.logger.info("\n⚖️ Application de SMOTE pour équilibrer les classes...")
            try:
                # Compter avant SMOTE
                n_pos_before = y_train.sum()
                n_neg_before = len(y_train) - n_pos_before
                self.logger.info(f"   Avant SMOTE: {n_neg_before} négatifs / {n_pos_before} positifs")

                # Appliquer SMOTE
                smote = SMOTE(random_state=42, sampling_strategy=0.5)  # 50% = 1 positif pour 2 négatifs
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

                # Compter après SMOTE
                n_pos_after = y_train_balanced.sum()
                n_neg_after = len(y_train_balanced) - n_pos_after
                self.logger.info(f"   Après SMOTE: {n_neg_after} négatifs / {n_pos_after} positifs")
                self.logger.info(f"   ✅ Classes équilibrées (ratio 2:1)")

                # Utiliser les données équilibrées pour l'entraînement
                X_train = pd.DataFrame(X_train_balanced, columns=X_train.columns)
                y_train = pd.Series(y_train_balanced)

            except Exception as e:
                self.logger.warning(f"⚠️ SMOTE a échoué: {e} - utilisation données originales")
        else:
            self.logger.warning("⚠️ SMOTE non disponible - classes non équilibrées")

        results = {}

        # 7. Entraîner XGBoost
        self.logger.info("\n" + "=" * 50)
        self.logger.info("🌳 ENTRAÎNEMENT XGBOOST")
        self.logger.info("=" * 50)

        try:
            xgb_metrics = self.xgboost.train(
                X_train, y_train,
                validation_split=self.validation_split,
                verbose=False
            )

            # Évaluer sur test set
            y_test_pred = self.xgboost.predict(X_test)
            from sklearn.metrics import accuracy_score, f1_score
            xgb_metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            xgb_metrics['test_f1'] = f1_score(y_test, y_test_pred)

            # Sauvegarder
            xgb_path = self.xgboost.save()
            xgb_metrics['model_path'] = xgb_path

            results['xgboost'] = xgb_metrics

            self.logger.info(f"✅ XGBoost entraîné:")
            self.logger.info(f"   - Val Accuracy: {xgb_metrics.get('val_accuracy', 0):.4f}")
            self.logger.info(f"   - Test Accuracy: {xgb_metrics.get('test_accuracy', 0):.4f}")
            self.logger.info(f"   - Test F1: {xgb_metrics.get('test_f1', 0):.4f}")

        except Exception as e:
            self.logger.error(f"❌ Erreur XGBoost: {e}")
            results['xgboost'] = {'error': str(e)}

        # 7. Entraîner LightGBM
        self.logger.info("\n" + "=" * 50)
        self.logger.info("💡 ENTRAÎNEMENT LIGHTGBM")
        self.logger.info("=" * 50)

        try:
            lgb_metrics = self.lightgbm.train(
                X_train, y_train,
                validation_split=self.validation_split,
                verbose=False
            )

            # Évaluer sur test set
            y_test_pred = self.lightgbm.predict(X_test)
            lgb_metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            lgb_metrics['test_f1'] = f1_score(y_test, y_test_pred)

            # Sauvegarder
            lgb_path = self.lightgbm.save()
            lgb_metrics['model_path'] = lgb_path

            results['lightgbm'] = lgb_metrics

            self.logger.info(f"✅ LightGBM entraîné:")
            self.logger.info(f"   - Val Accuracy: {lgb_metrics.get('val_accuracy', 0):.4f}")
            self.logger.info(f"   - Test Accuracy: {lgb_metrics.get('test_accuracy', 0):.4f}")
            self.logger.info(f"   - Test F1: {lgb_metrics.get('test_f1', 0):.4f}")

        except Exception as e:
            self.logger.error(f"❌ Erreur LightGBM: {e}")
            results['lightgbm'] = {'error': str(e)}

        # 8. Résumé
        elapsed = (datetime.now() - start_time).total_seconds()
        results['training_time_seconds'] = elapsed
        results['samples_trained'] = len(X_train)
        results['samples_tested'] = len(X_test)
        results['target_threshold'] = self.target_threshold
        results['horizon_bars'] = self.horizon_bars

        self._print_summary(results)

        return results

    def _print_summary(self, results: dict):
        """Affiche le résumé final"""

        self.logger.info("\n" + "=" * 70)
        self.logger.info("✅ ENTRAÎNEMENT HF MODELS TERMINÉ")
        self.logger.info("=" * 70)

        self.logger.info(f"\n📊 Configuration:")
        self.logger.info(f"   - Target threshold: {results.get('target_threshold', 0)*100:.1f}%")
        self.logger.info(f"   - Horizon: {results.get('horizon_bars', 0)} bougies")
        self.logger.info(f"   - Samples entraînés: {results.get('samples_trained', 0)}")
        self.logger.info(f"   - Durée: {results.get('training_time_seconds', 0):.1f}s")

        self.logger.info(f"\n📈 PERFORMANCES:")

        # XGBoost
        if 'xgboost' in results and 'error' not in results['xgboost']:
            xgb = results['xgboost']
            self.logger.info(f"\n   🌳 XGBoost:")
            self.logger.info(f"      Val Accuracy:  {xgb.get('val_accuracy', 0):.2%}")
            self.logger.info(f"      Test Accuracy: {xgb.get('test_accuracy', 0):.2%}")
            self.logger.info(f"      Test F1:       {xgb.get('test_f1', 0):.2%}")

        # LightGBM
        if 'lightgbm' in results and 'error' not in results['lightgbm']:
            lgb = results['lightgbm']
            self.logger.info(f"\n   💡 LightGBM:")
            self.logger.info(f"      Val Accuracy:  {lgb.get('val_accuracy', 0):.2%}")
            self.logger.info(f"      Test Accuracy: {lgb.get('test_accuracy', 0):.2%}")
            self.logger.info(f"      Test F1:       {lgb.get('test_f1', 0):.2%}")

        # Comparer les deux
        xgb_acc = results.get('xgboost', {}).get('test_accuracy', 0)
        lgb_acc = results.get('lightgbm', {}).get('test_accuracy', 0)

        if xgb_acc > 0 and lgb_acc > 0:
            combined = (xgb_acc * 0.6 + lgb_acc * 0.4)  # Weighted average
            self.logger.info(f"\n   🎯 Combined (60/40): {combined:.2%}")

            if combined >= 0.70:
                self.logger.info(f"   ✅ BON! Les modèles peuvent améliorer le win rate")
            elif combined >= 0.60:
                self.logger.info(f"   ⚠️ ACCEPTABLE mais peut être amélioré")
            else:
                self.logger.info(f"   ❌ FAIBLE - Plus de données nécessaires")


def load_data_from_csv(csv_path: str, logger) -> pd.DataFrame:
    """Charge les données depuis un fichier CSV"""

    path = Path(csv_path)

    if not path.exists():
        logger.error(f"❌ Fichier non trouvé: {csv_path}")
        return pd.DataFrame()

    logger.info(f"📂 Chargement: {csv_path}")

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        # Vérifier colonnes requises
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]

        if missing:
            logger.error(f"❌ Colonnes manquantes: {missing}")
            return pd.DataFrame()

        logger.info(f"✅ {len(df)} bougies chargées")
        logger.info(f"   Période: {df.index.min()} → {df.index.max()}")

        return df

    except Exception as e:
        logger.error(f"❌ Erreur lecture: {e}")
        return pd.DataFrame()


def load_data_from_api(config: dict, limit: int, logger) -> pd.DataFrame:
    """Charge les données depuis l'API Binance"""

    logger.info(f"🌐 Connexion API Binance...")

    client = BinanceClient(config)
    data_loader = DataLoader(client, config)

    symbol = get_nested_config(config, 'symbols', 'primary', default='BTC/USDT')

    logger.info(f"📊 Chargement {symbol} 5m ({limit} bougies)...")

    df = data_loader.load_historical_data(
        symbol=symbol,
        timeframe='5m',
        limit=limit
    )

    if not df.empty:
        logger.info(f"✅ {len(df)} bougies chargées")
        logger.info(f"   Période: {df.index.min()} → {df.index.max()}")

    return df


def main():
    """Point d'entrée"""

    parser = argparse.ArgumentParser(
        description='Entraînement HF Models (XGBoost + LightGBM)'
    )
    parser.add_argument('--data', type=str, default=None,
                       help='Chemin vers fichier CSV')
    parser.add_argument('--limit', type=int, default=10000,
                       help='Nombre de bougies depuis API')
    parser.add_argument('--threshold', type=float, default=0.003,
                       help='Seuil de target (défaut: 0.3%%)')

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║      🚀 HF MODELS TRAINING - 90%% WIN RATE SYSTEM 🚀              ║
║                                                                   ║
║      XGBoost + LightGBM (pas de LSTM!)                           ║
║      Target: {:.1f}%% (pas 0.1%%)                                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """.format(args.threshold * 100))

    logger = setup_logger('HFTraining')

    try:
        # Charger config
        logger.info("📋 Chargement configuration...")
        config_loader = ConfigLoader()
        config = config_loader.config

        # Charger données
        if args.data:
            df = load_data_from_csv(args.data, logger)
        else:
            df = load_data_from_api(config, args.limit, logger)

        if df.empty:
            logger.error("❌ Pas de données!")
            sys.exit(1)

        # Entraîner
        trainer = HFModelTrainer(config)
        trainer.target_threshold = args.threshold  # Override si spécifié

        results = trainer.train_all(df)

        if not results:
            logger.error("❌ Échec entraînement")
            sys.exit(1)

        # Instructions finales
        print("\n" + "=" * 70)
        print("💡 PROCHAINES ÉTAPES:")
        print("=" * 70)
        print("""
1. Les modèles sont sauvegardés dans ml_models/saved_models/

2. Pour utiliser la nouvelle config:
   cp config/config_hf_90wr.yaml config/config.yaml

3. Pour lancer le backtest:
   python paper_trading.py

4. Les modèles seront automatiquement chargés par MLSignalFilter
        """)

    except KeyboardInterrupt:
        print("\n⚠️ Interruption")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
