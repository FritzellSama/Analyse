# Guide d'Entraînement ML - Quantum Trader Pro

## Vue d'ensemble

Le bot utilise 3 modèles ML:
- **XGBoost**: Gradient boosting pour prédiction rapide
- **LSTM**: Réseau de neurones pour patterns séquentiels
- **Ensemble**: Combine XGBoost + LSTM pour de meilleures prédictions

---

## 1. Collecte de Données

### Collecter des données historiques

```bash
# 30 jours de données (recommandé minimum)
python collect_data.py --mode historical --days 30

# 60 jours pour plus de précision
python collect_data.py --mode historical --days 60

# Changer le timeframe (défaut: 5m)
python collect_data.py --mode historical --days 30 --timeframe 15m

# Changer la paire (défaut: BTC/USDT)
python collect_data.py --mode historical --days 30 --symbol ETH/USDT
```

### Collecte continue (temps réel)

```bash
# Collecter pendant 24 heures
python collect_data.py --mode continuous --hours 24

# Collecter indéfiniment (Ctrl+C pour arrêter)
python collect_data.py --mode continuous --hours 0

# Changer l'intervalle de vérification (défaut: 60s)
python collect_data.py --mode continuous --hours 24 --interval 30
```

### Voir les statistiques des données

```bash
python collect_data.py --mode stats
python collect_data.py --mode stats --symbol ETH/USDT --timeframe 15m
```

### Fichiers générés

Les données sont sauvegardées dans `data/collected/`:
```
data/collected/
├── BTC_USDT_5m.csv
├── BTC_USDT_15m.csv
├── ETH_USDT_5m.csv
└── ...
```

---

## 2. Entraînement des Modèles

### Entraînement direct depuis Binance

```bash
# Entraînement par défaut (10000 bougies)
python train_ml.py

# Spécifier le nombre de bougies
python train_ml.py --limit 5000
python train_ml.py --limit 20000
```

### Entraînement avec fichier CSV

```bash
# Utiliser les données collectées
python train_ml.py --data data/collected/BTC_USDT_5m.csv

# Utiliser un autre fichier
python train_ml.py --data /chemin/vers/mon_fichier.csv
```

### Modèles générés

Les modèles sont sauvegardés dans `ml_models/saved_models/`:
```
ml_models/saved_models/
├── xgboost_20251122_010850.pkl    # Modèle XGBoost
├── lstm_20251122_010902.h5        # Poids LSTM
└── lstm_20251122_010902.json      # Architecture LSTM
```

---

## 3. Workflow Recommandé

### Pour le Testnet (données limitées)

```bash
# 1. Entraîner directement
python train_ml.py --limit 10000

# 2. Lancer le bot
python main.py
```

### Pour la Production (meilleure précision)

```bash
# 1. Collecter beaucoup de données
python collect_data.py --mode historical --days 60

# 2. Entraîner avec les données collectées
python train_ml.py --data data/collected/BTC_USDT_5m.csv

# 3. Lancer le bot
python main.py
```

### Collecte continue + Ré-entraînement

```bash
# Terminal 1: Collecte continue
python collect_data.py --mode continuous --hours 0

# Terminal 2: Ré-entraîner périodiquement (ex: chaque jour)
python train_ml.py --data data/collected/BTC_USDT_5m.csv
```

---

## 4. Paramètres de Configuration

Les paramètres ML sont dans `config/config.yaml`:

```yaml
ml:
  models:
    xgboost:
      n_estimators: 200      # Nombre d'arbres
      max_depth: 6           # Profondeur max
      learning_rate: 0.1     # Taux d'apprentissage

    lstm:
      sequence_length: 50    # Longueur des séquences
      hidden_layers: [128, 64]
      dropout: 0.2
      epochs: 100

    ensemble:
      method: weighted
      min_confidence: 0.60
      min_agreement: 0.50

  training:
    min_samples: 700         # Minimum de samples requis
    validation_split: 0.2    # 20% pour validation
```

---

## 5. Métriques Attendues

| Données | Accuracy attendue | F1 Score |
|---------|-------------------|----------|
| < 1000 samples | 30-50% | < 0.3 |
| 1000-5000 samples | 50-60% | 0.3-0.5 |
| 5000-10000 samples | 55-65% | 0.4-0.6 |
| > 10000 samples | 60-70% | 0.5-0.7 |

> **Note**: En trading, une accuracy de 55-60% avec un bon ratio risk/reward peut être très profitable.

---

## 6. Dépannage

### "Pas assez de données"
```bash
# Augmenter le nombre de bougies
python train_ml.py --limit 10000

# Ou réduire min_samples dans config.yaml
```

### Overfitting (Train acc >> Val acc)
- Normal avec peu de données
- Se résout avec plus de données en production

### Modèles non trouvés
```bash
# Vérifier que les modèles existent
ls ml_models/saved_models/

# Ré-entraîner si nécessaire
python train_ml.py
```

---

## 7. Commandes Rapides

```bash
# Collecter + Entraîner (one-liner)
python collect_data.py --mode historical --days 30 && python train_ml.py --data data/collected/BTC_USDT_5m.csv

# Voir l'aide
python collect_data.py --help
python train_ml.py --help
```
