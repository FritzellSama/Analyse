#!/usr/bin/env python3
"""
Script d'entraînement AMÉLIORÉ pour modèles XGBoost et LightGBM
Optimisé pour gérer le déséquilibre de classes extrême (10% positifs)

AMÉLIORATIONS PRINCIPALES:
1. Utilise les modèles améliorés avec SMOTE, ADASYN
2. Optimisation automatique des seuils de décision
3. Ensemble de modèles pour LightGBM
4. Cross-validation stratifiée
5. Métriques métier (F2-score favorisant le recall)
6. Analyse détaillée des performances

Usage:
    python train_hf_models.py --data data/BTC_USDT_5m.csv
    python train_hf_models.py --limit 20000 --threshold 0.003
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ConfigLoader
from core.binance_client import BinanceClient
from data.data_loader import DataLoader
from ml_models.xgboost_model import XGBoostModel
from ml_models.lightgbm_model import LightGBMModel
from ml_models.feature_engineering import FeatureEngineer
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class HFModelTrainer:
    """
    Trainer optimisé pour déséquilibre de classes extrême
    Utilise les techniques avancées des modèles améliorés
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger('HFModelTrainer')

        # Configuration
        ml_config = config.get('ml', {})
        training_config = ml_config.get('training', {})

        self.min_samples = training_config.get('min_samples', 1000)
        self.validation_split = training_config.get('validation_split', 0.2)
        self.test_split = training_config.get('test_split', 0.1)

        # Paramètres de target
        self.horizon_bars = 5
        self.target_threshold = 0.003

        # Initialiser composants
        self.feature_engineer = FeatureEngineer(config)
        self.xgboost = XGBoostModel(config)
        self.lightgbm = LightGBMModel(config)

        self.logger.info("✅ HF Model Trainer initialisé")
        self.logger.info(f"   Target: {self.target_threshold*100:.1f}% sur {self.horizon_bars} bougies")
        self.logger.info(f"   Techniques: SMOTE, ADASYN, Ensemble, Seuil optimisé")

    def create_target_with_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée le target et analyse sa distribution"""

        df = df.copy()

        # Future return sur horizon
        df['future_return'] = df['close'].shift(-self.horizon_bars) / df['close'] - 1

        # Target binaire
        df['target'] = (df['future_return'] > self.target_threshold).astype(int)

        # Analyse détaillée
        df_clean = df.dropna()
        target_counts = df_clean['target'].value_counts()
        total = len(df_clean)

        if total > 0:
            pct_positive = target_counts.get(1, 0) / total * 100
            pct_negative = target_counts.get(0, 0) / total * 100

            self.logger.info(f"\n📊 ANALYSE DE LA DISTRIBUTION TARGET:")
            self.logger.info(f"   - Échantillons totaux: {total}")
            self.logger.info(f"   - UP (>{self.target_threshold*100:.1f}%): {target_counts.get(1, 0)} ({pct_positive:.1f}%)")
            self.logger.info(f"   - DOWN/NEUTRAL: {target_counts.get(0, 0)} ({pct_negative:.1f}%)")

            if pct_positive < 5:
                self.logger.warning(f"⚠️ TRÈS PEU de positifs! Considérer:")
                self.logger.warning(f"   - Réduire le seuil (actuellement {self.target_threshold*100:.1f}%)")
                self.logger.warning(f"   - Augmenter l'horizon (actuellement {self.horizon_bars} bougies)")
            elif pct_positive < 10:
                self.logger.warning(f"⚠️ Déséquilibre sévère - Techniques avancées activées:")
                self.logger.warning(f"   - SMOTE/ADASYN pour rééquilibrage")
                self.logger.warning(f"   - Optimisation du seuil de décision")
            elif pct_positive < 20:
                self.logger.info(f"✅ Déséquilibre modéré - gérable avec techniques avancées")
            else:
                self.logger.info(f"✅ Distribution équilibrée")

        return df

    def analyze_model_performance(self, results: dict):
        """Analyse détaillée des performances et recommandations"""

        self.logger.info("\n" + "="*70)
        self.logger.info("📊 ANALYSE DES PERFORMANCES")
        self.logger.info("="*70)

        # Analyser XGBoost
        if 'xgboost' in results and 'error' not in results['xgboost']:
            xgb = results['xgboost']
            self.logger.info("\n🌳 XGBoost:")

            val_f1 = xgb.get('val_f1', 0)
            val_precision = xgb.get('val_precision', 0)
            val_recall = xgb.get('val_recall', 0)

            self.logger.info(f"   F1: {val_f1:.3f} | Precision: {val_precision:.3f} | Recall: {val_recall:.3f}")

            if val_f1 < 0.2:
                self.logger.warning("   ⚠️ F1 très faible - Le modèle peine à détecter les positifs")
            elif val_f1 < 0.4:
                self.logger.info("   📈 F1 acceptable pour déséquilibre sévère")
            else:
                self.logger.info("   ✅ Excellente performance!")

        # Analyser LightGBM
        if 'lightgbm' in results and 'error' not in results['lightgbm']:
            lgb = results['lightgbm']
            self.logger.info("\n💡 LightGBM Ensemble:")

            val_f1 = lgb.get('val_f1', 0)
            val_precision = lgb.get('val_precision', 0)
            val_recall = lgb.get('val_recall', 0)

            self.logger.info(f"   F1: {val_f1:.3f} | Precision: {val_precision:.3f} | Recall: {val_recall:.3f}")

            if 'cv_scores' in lgb:
                cv = lgb['cv_scores']
                self.logger.info(f"   Cross-validation F1: {cv['mean_f1']:.3f} +/- {cv['std_f1']:.3f}")

        # Recommandations
        self.logger.info("\n🎯 RECOMMANDATIONS:")

        xgb_f1 = results.get('xgboost', {}).get('val_f1', 0)
        lgb_f1 = results.get('lightgbm', {}).get('val_f1', 0)

        if xgb_f1 > lgb_f1 * 1.2:
            self.logger.info("   → XGBoost meilleur - Utiliser comme modèle principal")
        elif lgb_f1 > xgb_f1 * 1.2:
            self.logger.info("   → LightGBM Ensemble meilleur - Utiliser comme modèle principal")
        else:
            self.logger.info("   → Combiner les deux modèles (moyenne pondérée)")

        avg_f1 = (xgb_f1 + lgb_f1) / 2
        if avg_f1 >= 0.4:
            self.logger.info("\n✅ EXCELLENTES PERFORMANCES! Prêt pour le trading.")
        elif avg_f1 >= 0.25:
            self.logger.info("\n📈 PERFORMANCES CORRECTES. Utilisable avec prudence.")
        else:
            self.logger.warning("\n⚠️ PERFORMANCES FAIBLES - Améliorations nécessaires:")
            self.logger.warning("   - Collecter plus de données")
            self.logger.warning("   - Ajuster le seuil de target")
            self.logger.warning("   - Ajouter des features techniques")

    def train_all(self, df: pd.DataFrame) -> dict:
        """Entraîne tous les modèles avec techniques avancées"""

        self.logger.info("="*70)
        self.logger.info("🚀 DÉBUT ENTRAÎNEMENT MODÈLES AMÉLIORÉS")
        self.logger.info("="*70)

        start_time = datetime.now()

        # 1. Feature engineering
        self.logger.info("\n🔨 Génération des features...")
        df_features = self.feature_engineer.generate_features(df)

        if df_features.empty:
            self.logger.error("❌ Échec génération features")
            return {}

        # 2. Créer target avec analyse
        self.logger.info(f"\n🎯 Création target (threshold={self.target_threshold*100:.1f}%)...")
        df_features = self.create_target_with_analysis(df_features)

        # 3. Nettoyer NaN
        df_features = df_features.dropna()

        if len(df_features) < self.min_samples:
            self.logger.error(f"❌ Pas assez de données: {len(df_features)} < {self.min_samples}")
            return {}

        self.logger.info(f"\n✅ {len(df_features)} samples prêts pour entraînement")

        # 4. Identifier features
        feature_names = self.feature_engineer.get_feature_names(df_features)
        X = df_features[feature_names]
        y = df_features['target']

        # 5. Split train/test
        test_size = int(len(X) * self.test_split)
        X_train = X.iloc[:-test_size]
        y_train = y.iloc[:-test_size]
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]

        self.logger.info(f"📊 Split: Train={len(X_train)} | Test={len(X_test)}")

        # Afficher distribution
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_ratio = n_pos / len(y_train)
        self.logger.info(f"⚖️ Classes: {n_neg} négatifs / {n_pos} positifs ({pos_ratio*100:.1f}% positifs)")

        results = {}

        # 6. Entraîner XGBoost amélioré
        self.logger.info("\n" + "="*50)
        self.logger.info("🌳 ENTRAÎNEMENT XGBOOST AMÉLIORÉ")
        self.logger.info("="*50)

        try:
            # Configurer stratégie selon déséquilibre
            if pos_ratio < 0.1:
                self.xgboost.resampling_strategy = 'smote'
            else:
                self.xgboost.resampling_strategy = 'none'

            xgb_metrics = self.xgboost.train(
                X_train, y_train,
                validation_split=self.validation_split,
                verbose=False
            )

            # Test
            y_test_pred = self.xgboost.predict(X_test)

            xgb_metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            xgb_metrics['test_f1'] = f1_score(y_test, y_test_pred, zero_division=0)
            xgb_metrics['test_precision'] = precision_score(y_test, y_test_pred, zero_division=0)
            xgb_metrics['test_recall'] = recall_score(y_test, y_test_pred, zero_division=0)

            # Sauvegarder
            xgb_path = self.xgboost.save()
            xgb_metrics['model_path'] = xgb_path

            results['xgboost'] = xgb_metrics

            self.logger.info(f"\n✅ XGBoost entraîné (seuil: {self.xgboost.optimal_threshold:.3f}):")
            self.logger.info(f"   Test: F1={xgb_metrics['test_f1']:.3f}, P={xgb_metrics['test_precision']:.3f}, R={xgb_metrics['test_recall']:.3f}")

        except Exception as e:
            self.logger.error(f"❌ Erreur XGBoost: {e}")
            import traceback
            traceback.print_exc()
            results['xgboost'] = {'error': str(e)}

        # 7. Entraîner LightGBM ensemble amélioré
        self.logger.info("\n" + "="*50)
        self.logger.info("💡 ENTRAÎNEMENT LIGHTGBM ENSEMBLE AMÉLIORÉ")
        self.logger.info("="*50)

        try:
            # Configurer selon déséquilibre
            if pos_ratio < 0.1:
                self.lightgbm.resampling_strategy = 'adasyn'
            else:
                self.lightgbm.resampling_strategy = 'borderline'

            lgb_metrics = self.lightgbm.train(
                X_train, y_train,
                validation_split=self.validation_split,
                verbose=False
            )

            # Test
            y_test_pred = self.lightgbm.predict(X_test)

            lgb_metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            lgb_metrics['test_f1'] = f1_score(y_test, y_test_pred, zero_division=0)
            lgb_metrics['test_precision'] = precision_score(y_test, y_test_pred, zero_division=0)
            lgb_metrics['test_recall'] = recall_score(y_test, y_test_pred, zero_division=0)

            # Sauvegarder
            lgb_path = self.lightgbm.save()
            lgb_metrics['model_path'] = lgb_path

            results['lightgbm'] = lgb_metrics

            self.logger.info(f"\n✅ LightGBM entraîné (ensemble {self.lightgbm.n_models} modèles, seuil: {self.lightgbm.ensemble_threshold:.3f}):")
            self.logger.info(f"   Test: F1={lgb_metrics['test_f1']:.3f}, P={lgb_metrics['test_precision']:.3f}, R={lgb_metrics['test_recall']:.3f}")

        except Exception as e:
            self.logger.error(f"❌ Erreur LightGBM: {e}")
            import traceback
            traceback.print_exc()
            results['lightgbm'] = {'error': str(e)}

        # 8. Méta-données
        elapsed = (datetime.now() - start_time).total_seconds()
        results['training_time_seconds'] = elapsed
        results['samples_trained'] = len(X_train)
        results['samples_tested'] = len(X_test)
        results['target_threshold'] = self.target_threshold
        results['horizon_bars'] = self.horizon_bars
        results['positive_ratio'] = float(pos_ratio)

        # 9. Analyse des performances
        self.analyze_model_performance(results)

        # 10. Résumé final
        self._print_summary(results)

        return results

    def _print_summary(self, results: dict):
        """Affiche le résumé final"""

        self.logger.info("\n" + "="*70)
        self.logger.info("✅ ENTRAÎNEMENT TERMINÉ")
        self.logger.info("="*70)

        self.logger.info(f"\n📊 Configuration:")
        self.logger.info(f"   - Target threshold: {results.get('target_threshold', 0)*100:.1f}%")
        self.logger.info(f"   - Horizon: {results.get('horizon_bars', 0)} bougies")
        self.logger.info(f"   - Samples entraînés: {results.get('samples_trained', 0)}")
        self.logger.info(f"   - Durée: {results.get('training_time_seconds', 0):.1f}s")

        self.logger.info(f"\n📈 PERFORMANCES FINALES:")

        xgb_f1 = results.get('xgboost', {}).get('test_f1', 0)
        lgb_f1 = results.get('lightgbm', {}).get('test_f1', 0)

        if 'xgboost' in results and 'error' not in results['xgboost']:
            xgb = results['xgboost']
            self.logger.info(f"\n   🌳 XGBoost:")
            self.logger.info(f"      Test F1:       {xgb.get('test_f1', 0):.2%}")
            self.logger.info(f"      Test Precision: {xgb.get('test_precision', 0):.2%}")
            self.logger.info(f"      Test Recall:    {xgb.get('test_recall', 0):.2%}")

        if 'lightgbm' in results and 'error' not in results['lightgbm']:
            lgb = results['lightgbm']
            self.logger.info(f"\n   💡 LightGBM Ensemble:")
            self.logger.info(f"      Test F1:       {lgb.get('test_f1', 0):.2%}")
            self.logger.info(f"      Test Precision: {lgb.get('test_precision', 0):.2%}")
            self.logger.info(f"      Test Recall:    {lgb.get('test_recall', 0):.2%}")

        if xgb_f1 > 0 and lgb_f1 > 0:
            combined = (xgb_f1 * 0.6 + lgb_f1 * 0.4)
            self.logger.info(f"\n   🎯 Combined (60/40): {combined:.2%}")


def load_data_from_csv(csv_path: str, logger) -> pd.DataFrame:
    """Charge les données depuis un fichier CSV"""

    path = Path(csv_path)

    if not path.exists():
        logger.error(f"❌ Fichier non trouvé: {csv_path}")
        return pd.DataFrame()

    logger.info(f"📂 Chargement: {csv_path}")

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

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
    """Point d'entrée principal"""

    parser = argparse.ArgumentParser(
        description='Entraînement de modèles ML améliorés pour trading'
    )
    parser.add_argument('--data', type=str, default=None,
                       help='Chemin vers fichier CSV')
    parser.add_argument('--limit', type=int, default=10000,
                       help='Nombre de bougies depuis API')
    parser.add_argument('--threshold', type=float, default=0.003,
                       help='Seuil de target en %% (défaut: 0.3%%)')
    parser.add_argument('--horizon', type=int, default=5,
                       help='Horizon de prédiction en bougies')

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║            🚀 ENTRAÎNEMENT MODÈLES ML AMÉLIORÉS 🚀                       ║
║                                                                           ║
║   Techniques avancées pour déséquilibre de classes:                      ║
║   - SMOTE / ADASYN pour rééchantillonnage                               ║
║   - Optimisation automatique des seuils                                  ║
║   - Ensemble de modèles LightGBM                                         ║
║   - Cross-validation stratifiée                                          ║
║                                                                           ║
║   Target: {:.1f}% sur {} bougies                                              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """.format(args.threshold * 100, args.horizon))

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
        trainer.target_threshold = args.threshold
        trainer.horizon_bars = args.horizon

        results = trainer.train_all(df)

        if not results:
            logger.error("❌ Échec entraînement")
            sys.exit(1)

        # Instructions finales
        print("\n" + "="*70)
        print("💡 PROCHAINES ÉTAPES:")
        print("="*70)

        xgb_f1 = results.get('xgboost', {}).get('test_f1', 0)
        lgb_f1 = results.get('lightgbm', {}).get('test_f1', 0)

        print(f"""
MODÈLES SAUVEGARDÉS:
• XGBoost: {results.get('xgboost', {}).get('model_path', 'N/A')}
• LightGBM: {results.get('lightgbm', {}).get('model_path', 'N/A')}

PERFORMANCES:
• XGBoost F1: {xgb_f1:.3f}
• LightGBM F1: {lgb_f1:.3f}

UTILISATION:
1. Les modèles sont automatiquement chargés par MLSignalFilter
2. Lancer le backtest: python paper_trading.py
3. Pour re-entraîner avec plus de données: python train_hf_models.py --limit 50000
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
