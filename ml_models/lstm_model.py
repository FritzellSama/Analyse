"""
LSTM Model - Quantum Trader Pro
Modèle LSTM (Long Short-Term Memory) pour prédiction de séquences temporelles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from utils.logger import setup_logger

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from tensorflow.keras.models import Sequential, load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class LSTMModel:
    """
    Modèle LSTM pour prédiction directionnelle:
    - Traite séquences temporelles
    - 2-3 couches LSTM
    - Dropout pour régularisation
    - Early stopping
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le modèle LSTM
        
        Args:
            config: Configuration complète du bot
        """
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow non installé. Installez avec: pip install tensorflow"
            )
        
        self.config = config
        self.logger = setup_logger('LSTMModel')
        
        # Configuration LSTM
        lstm_config = config.get('ml', {}).get('models', {}).get('lstm', {})
        
        self.sequence_length = lstm_config.get('sequence_length', 50)
        self.hidden_layers = lstm_config.get('hidden_layers', [128, 64])
        self.dropout = lstm_config.get('dropout', 0.2)
        self.epochs = lstm_config.get('epochs', 100)
        
        # Model
        self.model = None
        self.feature_count = 0
        self.training_history = {}
        
        # Paths
        self.model_dir = Path('ml_models/saved_models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ LSTM Model initialisé")
    
    def build_model(self, input_shape: Tuple[int, int]):
        """
        Construit l'architecture du modèle LSTM
        
        Args:
            input_shape: (sequence_length, n_features)
        """
        
        self.logger.info(f"🏗️ Construction modèle LSTM: input_shape={input_shape}")
        
        self.model = Sequential()
        
        # Première couche LSTM
        self.model.add(layers.LSTM(
            self.hidden_layers[0],
            return_sequences=len(self.hidden_layers) > 1,
            input_shape=input_shape
        ))
        self.model.add(layers.Dropout(self.dropout))
        
        # Couches intermédiaires
        for i, units in enumerate(self.hidden_layers[1:], 1):
            return_sequences = i < len(self.hidden_layers) - 1
            self.model.add(layers.LSTM(units, return_sequences=return_sequences))
            self.model.add(layers.Dropout(self.dropout))
        
        # Couche de sortie
        self.model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compilation
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        self.logger.info(f"✅ Modèle construit:")
        self.logger.info(f"   - Layers: {len(self.hidden_layers)} LSTM")
        self.logger.info(f"   - Hidden units: {self.hidden_layers}")
        self.logger.info(f"   - Dropout: {self.dropout}")
        
        # Afficher résumé
        self.model.summary(print_fn=self.logger.info)
    
    def create_sequences(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des séquences pour l'entraînement LSTM
        
        Args:
            X: Features DataFrame
            y: Target Series
        
        Returns:
            (X_sequences, y_sequences)
        """
        
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - self.sequence_length):
            # Séquence de features
            X_seq = X.iloc[i:i + self.sequence_length].values
            X_sequences.append(X_seq)
            
            # Target correspondant (dernier point de la séquence)
            y_seq = y.iloc[i + self.sequence_length]
            y_sequences.append(y_seq)
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        self.logger.info(
            f"📊 Séquences créées: {len(X_sequences)} × "
            f"({self.sequence_length}, {X.shape[1]})"
        )
        
        return X_sequences, y_sequences
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> Dict:
        """
        Entraîne le modèle LSTM
        
        Args:
            X: Features DataFrame
            y: Target Series
            validation_split: Proportion pour validation
            verbose: Niveau de verbosité (0, 1, 2)
        
        Returns:
            Dict avec historique d'entraînement
        """
        
        self.logger.info(f"🚀 Début entraînement LSTM ({len(X)} samples)")
        
        # Créer séquences
        X_seq, y_seq = self.create_sequences(X, y)
        
        # Sauvegarder feature count
        self.feature_count = X.shape[1]
        
        # Build model si pas encore fait
        if self.model is None:
            input_shape = (self.sequence_length, self.feature_count)
            self.build_model(input_shape)
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Entraînement
        history = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        # Sauvegarder historique
        self.training_history = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Log résultats finaux
        final_metrics = {
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'train_accuracy': history.history['accuracy'][-1],
            'val_accuracy': history.history['val_accuracy'][-1]
        }
        
        self.logger.info(f"✅ Entraînement terminé:")
        self.logger.info(f"   - Loss (val): {final_metrics['val_loss']:.4f}")
        self.logger.info(f"   - Accuracy (val): {final_metrics['val_accuracy']:.4f}")
        
        # Check overfitting
        accuracy_diff = final_metrics['train_accuracy'] - final_metrics['val_accuracy']
        if accuracy_diff > 0.1:
            self.logger.warning(
                f"⚠️ Possible overfitting: Train acc - Val acc = {accuracy_diff:.4f}"
            )
        
        return final_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédiction binaire (0 ou 1)
        
        Args:
            X: Features DataFrame
        
        Returns:
            Array de prédictions (0=DOWN, 1=UP)
        """
        
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        
        # Créer séquences (sans target)
        X_sequences = []
        
        for i in range(len(X) - self.sequence_length + 1):
            X_seq = X.iloc[i:i + self.sequence_length].values
            X_sequences.append(X_seq)
        
        if len(X_sequences) == 0:
            # Pas assez de données
            return np.array([])
        
        X_sequences = np.array(X_sequences)
        
        # Prédiction
        probabilities = self.model.predict(X_sequences, verbose=0)
        predictions = (probabilities > 0.5).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédiction de probabilités
        
        Args:
            X: Features DataFrame
        
        Returns:
            Array de probabilités [P(DOWN), P(UP)]
        """
        
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        
        # Créer séquences
        X_sequences = []
        
        for i in range(len(X) - self.sequence_length + 1):
            X_seq = X.iloc[i:i + self.sequence_length].values
            X_sequences.append(X_seq)
        
        if len(X_sequences) == 0:
            return np.array([])
        
        X_sequences = np.array(X_sequences)
        
        # Prédiction
        prob_up = self.model.predict(X_sequences, verbose=0).flatten()
        prob_down = 1 - prob_up
        
        # Format [P(DOWN), P(UP)]
        probabilities = np.column_stack([prob_down, prob_up])
        
        return probabilities
    
    def get_signal_with_confidence(self, X: pd.DataFrame) -> Tuple[int, float]:
        """
        Retourne signal et confidence pour la dernière prédiction
        
        Args:
            X: Features DataFrame (doit avoir au moins sequence_length lignes)
        
        Returns:
            (signal, confidence) où signal = 1 (UP) ou 0 (DOWN)
        """
        
        if len(X) < self.sequence_length:
            self.logger.warning(
                f"⚠️ Pas assez de données: {len(X)} < {self.sequence_length}"
            )
            return 0, 0.0
        
        # Prendre les dernières sequence_length lignes
        X_last = X.iloc[-self.sequence_length:]
        
        # Prédiction
        proba = self.predict_proba(X_last)
        
        if len(proba) == 0:
            return 0, 0.0
        
        # Dernière prédiction
        proba_last = proba[-1]
        signal = 1 if proba_last[1] > 0.5 else 0
        confidence = proba_last[signal]
        
        return int(signal), float(confidence)
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Sauvegarde le modèle
        
        Args:
            filename: Nom du fichier (optionnel)
        
        Returns:
            Chemin du fichier sauvegardé
        """
        
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'lstm_{timestamp}.h5'
        
        filepath = self.model_dir / filename
        
        # Sauvegarder modèle Keras
        self.model.save(filepath)
        
        # Sauvegarder config séparément
        import json
        config_file = filepath.with_suffix('.json')
        
        config_data = {
            'sequence_length': self.sequence_length,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'feature_count': self.feature_count,
            'training_history': self.training_history
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"💾 Modèle sauvegardé: {filepath}")
        
        return str(filepath)
    
    def load(self, filepath: str):
        """
        Charge un modèle sauvegardé
        
        Args:
            filepath: Chemin vers le fichier .h5
        """
        
        self.logger.info(f"📂 Chargement modèle: {filepath}")
        
        # Charger modèle Keras
        self.model = load_model(filepath)
        
        # Charger config
        import json
        config_file = Path(filepath).with_suffix('.json')
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            self.sequence_length = config_data.get('sequence_length', self.sequence_length)
            self.hidden_layers = config_data.get('hidden_layers', self.hidden_layers)
            self.dropout = config_data.get('dropout', self.dropout)
            self.feature_count = config_data.get('feature_count', 0)
            self.training_history = config_data.get('training_history', {})
        
        self.logger.info("✅ Modèle chargé avec succès")
        
        if self.training_history:
            val_acc = self.training_history.get('val_accuracy', [0])[-1]
            self.logger.info(f"   - Accuracy (val): {val_acc:.4f}")
    
    def get_history(self) -> Dict:
        """Retourne l'historique d'entraînement"""
        return self.training_history.copy()
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Génère des graphiques de l'historique d'entraînement
        
        Args:
            save_path: Chemin pour sauvegarder (optionnel)
        """
        
        if not self.training_history:
            self.logger.warning("⚠️ Pas d'historique à afficher")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.error("❌ Matplotlib non installé")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.training_history['loss'], label='Train Loss')
        ax1.plot(self.training_history['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.training_history['accuracy'], label='Train Accuracy')
        ax2.plot(self.training_history['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"📊 Graphique sauvegardé: {save_path}")
        else:
            plt.show()
        
        plt.close()
