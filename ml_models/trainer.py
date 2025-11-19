"""
ML Trainer - Quantum Trader Pro
Pipeline d'entraînement automatique pour les modèles ML
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import time
from utils.logger import setup_logger

from ml_models.feature_engineering import FeatureEngineer
from ml_models.xgboost_model import XGBoostModel
from ml_models.lstm_model import LSTMModel
from ml_models.ensemble import EnsembleModel
from utils.config_helpers import get_nested_config

class MLTrainer:
    """
    Gestionnaire d'entraînement ML qui:
    - Charge et prépare les données
    - Entraîne tous les modèles
    - Évalue les performances
    - Sauvegarde les modèles
    - Auto-retraining périodique
    """
    
    def __init__(self, client, config: Dict):
        """
        Initialise le trainer ML
        
        Args:
            client: Instance BinanceClient
            config: Configuration complète du bot
        """
        self.client = client
        self.config = config
        self.logger = setup_logger('MLTrainer')
        
        # Configuration
        ml_config = config.get('ml', {})
        training_config = ml_config.get('training', {})
        
        self.retrain_interval_hours = training_config.get('retrain_interval_hours', 24)
        self.min_samples = training_config.get('min_samples', 1000)
        self.validation_split = training_config.get('validation_split', 0.2)
        self.test_split = training_config.get('test_split', 0.1)
        
        # Prediction config
        pred_config = ml_config.get('prediction', {})
        self.horizon_bars = pred_config.get('horizon_bars', 5)
        self.confidence_threshold = pred_config.get('confidence_threshold', 0.7)
        
        # Components
        self.feature_engineer = FeatureEngineer(config)
        self.xgboost_model = XGBoostModel(config)
        self.lstm_model = LSTMModel(config)
        self.ensemble_model = EnsembleModel(config)
        
        # State
        self.last_training_time = None
        self.is_trained = False
        
        self.logger.info("✅ ML Trainer initialisé")
    
    def train_all_models(
        self,
        df: pd.DataFrame,
        orderbook_data: Optional[Dict] = None
    ) -> Dict:
        """
        Entraîne tous les modèles ML
        
        Args:
            df: DataFrame OHLCV
            orderbook_data: Données orderbook (optionnel)
        
        Returns:
            Dict avec résultats d'entraînement
        """
        
        self.logger.info("=" * 70)
        self.logger.info("🤖 DÉBUT ENTRAÎNEMENT ML")
        self.logger.info("=" * 70)
        
        start_time = time.time()
        
        # 1. Feature engineering
        self.logger.info("🔨 Feature engineering...")
        df_features = self.feature_engineer.generate_features(df, orderbook_data)
        
        # 2. Créer target
        df_features = self.feature_engineer.create_target(
            df_features,
            horizon=self.horizon_bars
        )
        
        # 3. Supprimer NaN
        df_features = df_features.dropna()
        
        if len(df_features) < self.min_samples:
            self.logger.error(
                f"❌ Pas assez de données: {len(df_features)} < {self.min_samples}"
            )
            return {}
        
        self.logger.info(f"✅ {len(df_features)} samples prêts pour entraînement")
        
        # 4. Séparer features et target
        feature_names = self.feature_engineer.get_feature_names(df_features)
        X = df_features[feature_names]
        y = df_features['target']
        
        # 5. Split train/test
        test_size = int(len(X) * self.test_split)
        X_train = X.iloc[:-test_size]
        y_train = y.iloc[:-test_size]
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]
        
        self.logger.info(
            f"📊 Split: Train={len(X_train)} | Test={len(X_test)}"
        )
        
        # 6. Scale features
        X_train_scaled, scaler = self.feature_engineer.scale_features(X_train)
        X_test_scaled, _ = self.feature_engineer.scale_features(X_test, scaler)
        
        results = {}
        
        # 7. Entraîner XGBoost
        self.logger.info("\n📊 Entraînement XGBoost...")
        try:
            xgb_metrics = self.xgboost_model.train(
                X_train, y_train,
                validation_split=self.validation_split,
                verbose=False
            )
            results['xgboost'] = xgb_metrics
            
            # Sauvegarder
            xgb_path = self.xgboost_model.save()
            results['xgboost']['model_path'] = xgb_path
            
        except Exception as e:
            self.logger.error(f"❌ Erreur XGBoost: {e}")
            results['xgboost'] = {'error': str(e)}
        
        # 8. Entraîner LSTM
        self.logger.info("\n🧠 Entraînement LSTM...")
        try:
            lstm_metrics = self.lstm_model.train(
                X_train_scaled, y_train,
                validation_split=self.validation_split,
                verbose=0
            )
            results['lstm'] = lstm_metrics
            
            # Sauvegarder
            lstm_path = self.lstm_model.save()
            results['lstm']['model_path'] = lstm_path
            
        except Exception as e:
            self.logger.error(f"❌ Erreur LSTM: {e}")
            results['lstm'] = {'error': str(e)}
        
        # 9. Créer ensemble
        self.logger.info("\n🎯 Création Ensemble...")
        try:
            self.ensemble_model.add_model('xgboost', self.xgboost_model)
            self.ensemble_model.add_model('lstm', self.lstm_model)
            
            # Évaluer ensemble
            ensemble_metrics = self.ensemble_model.evaluate(X_test, y_test)
            results['ensemble'] = ensemble_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Erreur Ensemble: {e}")
            results['ensemble'] = {'error': str(e)}
        
        # 10. Résumé
        elapsed = time.time() - start_time
        results['training_time_seconds'] = elapsed
        results['timestamp'] = datetime.now().isoformat()
        results['samples_trained'] = len(X_train)
        results['samples_tested'] = len(X_test)
        
        self.last_training_time = datetime.now()
        self.is_trained = True
        
        self.logger.info("=" * 70)
        self.logger.info("✅ ENTRAÎNEMENT TERMINÉ")
        self.logger.info(f"⏱️  Durée: {elapsed:.1f}s")
        self.logger.info("=" * 70)
        
        # Afficher résumé
        self._print_training_summary(results)
        
        return results
    
    def _print_training_summary(self, results: Dict):
        """Affiche un résumé des résultats"""
        
        self.logger.info("\n📊 RÉSUMÉ DES PERFORMANCES:")
        
        # XGBoost
        if 'xgboost' in results and 'error' not in results['xgboost']:
            xgb = results['xgboost']
            self.logger.info(f"\n🌳 XGBoost:")
            self.logger.info(f"   - Accuracy (val): {xgb.get('val_accuracy', 0):.4f}")
            self.logger.info(f"   - F1 Score (val): {xgb.get('val_f1', 0):.4f}")
            self.logger.info(f"   - ROC AUC (val): {xgb.get('val_roc_auc', 0):.4f}")
        
        # LSTM
        if 'lstm' in results and 'error' not in results['lstm']:
            lstm = results['lstm']
            self.logger.info(f"\n🧠 LSTM:")
            self.logger.info(f"   - Accuracy (val): {lstm.get('val_accuracy', 0):.4f}")
            self.logger.info(f"   - Loss (val): {lstm.get('val_loss', 0):.4f}")
        
        # Ensemble
        if 'ensemble' in results and 'error' not in results['ensemble']:
            ens = results['ensemble']
            self.logger.info(f"\n🎯 Ensemble:")
            self.logger.info(f"   - Accuracy: {ens.get('accuracy', 0):.4f}")
            self.logger.info(f"   - F1 Score: {ens.get('f1', 0):.4f}")
            
            # Modèles individuels
            if 'individual' in ens:
                self.logger.info(f"\n   Performances individuelles:")
                for name, metrics in ens['individual'].items():
                    self.logger.info(
                        f"   - {name}: Acc={metrics.get('accuracy', 0):.4f}, "
                        f"F1={metrics.get('f1', 0):.4f}"
                    )
    
    def should_retrain(self) -> bool:
        """
        Détermine si un retraining est nécessaire
        
        Returns:
            True si retraining nécessaire
        """
        
        if not self.is_trained:
            return True
        
        if self.last_training_time is None:
            return True
        
        # Check interval
        time_since_training = datetime.now() - self.last_training_time
        hours_since = time_since_training.total_seconds() / 3600
        
        should_retrain = hours_since >= self.retrain_interval_hours
        
        if should_retrain:
            self.logger.info(
                f"🔄 Retraining nécessaire: {hours_since:.1f}h depuis dernier training"
            )
        
        return should_retrain
    
    def auto_retrain_loop(self, data_loader):
        """
        Boucle de retraining automatique
        
        Args:
            data_loader: Instance DataLoader
        """
        
        self.logger.info("🔄 Démarrage boucle auto-retraining...")
        
        while True:
            try:
                if self.should_retrain():
                    # Charger données
                    symbol = get_nested_config(self.config, 'symbols', 'primary', default='BTC/USDT')
                    df = data_loader.load_historical_data(
                        symbol=symbol,
                        timeframe='5m',
                        limit=5000
                    )
                    
                    if df.empty:
                        self.logger.error("❌ Pas de données pour retraining")
                    else:
                        # Entraîner
                        self.train_all_models(df)
                
                # Attendre intervalle
                sleep_hours = self.retrain_interval_hours / 2  # Check à mi-chemin
                self.logger.info(f"💤 Prochaine vérification dans {sleep_hours}h")
                time.sleep(sleep_hours * 3600)
                
            except KeyboardInterrupt:
                self.logger.info("⚠️ Arrêt auto-retraining")
                break
            except Exception as e:
                self.logger.error(f"❌ Erreur auto-retraining: {e}")
                time.sleep(3600)  # Attendre 1h avant retry
    
    def load_latest_models(self):
        """Charge les derniers modèles sauvegardés"""
        
        self.logger.info("📂 Chargement derniers modèles...")
        
        model_dir = Path('ml_models/saved_models')
        
        if not model_dir.exists():
            self.logger.warning("⚠️ Aucun modèle sauvegardé trouvé")
            return False
        
        # Trouver derniers modèles
        xgb_files = sorted(model_dir.glob('xgboost_*.pkl'))
        lstm_files = sorted(model_dir.glob('lstm_*.h5'))
        
        loaded = False
        
        # Charger XGBoost
        if xgb_files:
            latest_xgb = str(xgb_files[-1])
            try:
                self.xgboost_model.load(latest_xgb)
                self.ensemble_model.add_model('xgboost', self.xgboost_model)
                loaded = True
            except Exception as e:
                self.logger.error(f"❌ Erreur chargement XGBoost: {e}")
        
        # Charger LSTM
        if lstm_files:
            latest_lstm = str(lstm_files[-1])
            try:
                self.lstm_model.load(latest_lstm)
                self.ensemble_model.add_model('lstm', self.lstm_model)
                loaded = True
            except Exception as e:
                self.logger.error(f"❌ Erreur chargement LSTM: {e}")
        
        if loaded:
            self.is_trained = True
            self.last_training_time = datetime.now()
            self.logger.info("✅ Modèles chargés avec succès")
        else:
            self.logger.warning("⚠️ Aucun modèle chargé")
        
        return loaded
    
    def get_prediction(
        self,
        df: pd.DataFrame,
        orderbook_data: Optional[Dict] = None
    ) -> Dict:
        """
        Génère une prédiction avec l'ensemble
        
        Args:
            df: DataFrame OHLCV
            orderbook_data: Données orderbook (optionnel)
        
        Returns:
            Dict avec prédiction et détails
        """
        
        if not self.is_trained:
            self.logger.warning("⚠️ Modèles non entraînés")
            return {
                'signal': 0,
                'confidence': 0.0,
                'should_trade': False,
                'error': 'Models not trained'
            }
        
        try:
            # Feature engineering
            df_features = self.feature_engineer.generate_features(df, orderbook_data)
            
            # Get features
            feature_names = self.feature_engineer.get_feature_names(df_features)
            X = df_features[feature_names]
            
            # Scale
            X_scaled, _ = self.feature_engineer.scale_features(X)
            
            # Prédiction ensemble
            should_trade, signal, confidence = self.ensemble_model.should_trade(
                X_scaled,
                min_confidence=self.confidence_threshold
            )
            
            # Agreement score
            agreement = self.ensemble_model.get_agreement_score(X_scaled)
            
            # Contributions
            contributions = self.ensemble_model.get_model_contributions(X_scaled)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'agreement': agreement,
                'should_trade': should_trade,
                'contributions': contributions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erreur prédiction: {e}")
            return {
                'signal': 0,
                'confidence': 0.0,
                'should_trade': False,
                'error': str(e)
            }
    
    def get_status(self) -> Dict:
        """Retourne le statut du trainer"""
        
        return {
            'is_trained': self.is_trained,
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'retrain_interval_hours': self.retrain_interval_hours,
            'ensemble_summary': self.ensemble_model.get_summary() if self.is_trained else {}
        }
