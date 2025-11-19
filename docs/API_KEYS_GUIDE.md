# 🔑 Guide de Configuration des Clés API Binance

## ❌ Problème: "Invalid Api-Key ID"

Cette erreur signifie que Binance ne reconnaît pas vos clés API. Voici pourquoi et comment résoudre.

---

## 🎯 Cause Principale

**Vous utilisez des clés de PRODUCTION sur le TESTNET (ou vice-versa).**

Les clés API Binance sont **différentes** selon l'environnement:
- 🧪 **Testnet**: Clés obtenues sur https://testnet.binance.vision/
- ⚠️  **Production**: Clés obtenues sur https://www.binance.com/

**Les clés ne sont PAS interchangeables!**

---

## ✅ Solution 1: Utiliser le Testnet (RECOMMANDÉ)

### Pourquoi le testnet?
- ✅ Argent fictif (pas de risque)
- ✅ Teste toutes les fonctionnalités
- ✅ Gratuit et illimité
- ✅ Parfait pour développement

### Comment obtenir des clés testnet:

#### Étape 1: Créer un compte testnet
```
1. Allez sur: https://testnet.binance.vision/
2. Cliquez sur "Register" (en haut à droite)
3. Créez un compte avec email + mot de passe
   (Peut être différent de votre compte Binance principal)
4. Confirmez votre email
```

#### Étape 2: Générer vos clés API testnet
```
1. Connectez-vous sur https://testnet.binance.vision/
2. Cliquez sur votre profil (en haut à droite)
3. Allez dans "API Management"
4. Cliquez "Create API"
5. Notez:
   - API Key (64 caractères environ)
   - Secret Key (64 caractères environ)
```

#### Étape 3: Configurer dans .env
```env
BINANCE_API_KEY=votre_cle_testnet_ici
BINANCE_SECRET_KEY=votre_secret_testnet_ici
BINANCE_TESTNET=true  # ← IMPORTANT!
```

#### Étape 4: Obtenir des fonds fictifs
```
1. Sur testnet.binance.vision, allez dans "Wallet"
2. Cliquez sur "Test Faucet" ou "Add Test Funds"
3. Ajoutez 10,000 USDT et 1 BTC (fictifs)
```

---

## ⚠️  Solution 2: Utiliser la Production (DANGER!)

### ⚠️  ATTENTION
- Argent RÉEL à risque
- Toujours tester en testnet d'abord
- Commencer avec capital minimum
- **NE JAMAIS** activer "Enable Withdrawals"

### Comment obtenir des clés production:

#### Étape 1: Créer clés API
```
1. Connectez-vous sur https://www.binance.com/
2. Profil → API Management
3. Create API
4. Suivez la vérification 2FA
5. Notez API Key et Secret Key
```

#### Étape 2: Configurer les permissions
```
✅ Enable Reading
✅ Enable Spot & Margin Trading
❌ Enable Withdrawals (JAMAIS!)
❌ Enable Futures (optionnel, si vous tradez futures)

⚠️  Configurez IP Whitelist si possible
```

#### Étape 3: Configurer dans .env
```env
BINANCE_API_KEY=votre_vraie_cle_production
BINANCE_SECRET_KEY=votre_vrai_secret_production
BINANCE_TESTNET=false  # ← IMPORTANT!
```

---

## 🔍 Comment Diagnostiquer Votre Problème

### Vérification 1: Type de clés
```
Question: Où avez-vous créé vos clés?
- Si sur testnet.binance.vision → BINANCE_TESTNET=true
- Si sur binance.com → BINANCE_TESTNET=false
```

### Vérification 2: Format des clés
```
Les clés Binance ressemblent à:
- API Key: environ 64 caractères alphanumériques
- Secret Key: environ 64 caractères alphanumériques

Exemple:
API_KEY=oZwxoNQd8Bs3bfOn2o7cyrJvqeHXOuag2mU2TGgwMTAJgtDSDY2FJyG42yjSErYb
```

### Vérification 3: Fichier .env
```bash
# Vérifiez que .env existe
ls -la .env

# Vérifiez qu'il n'y a pas d'espaces ou de quotes
cat .env

# Bon format:
BINANCE_API_KEY=votre_cle
BINANCE_SECRET_KEY=votre_secret
BINANCE_TESTNET=true

# Mauvais format:
BINANCE_API_KEY = votre_cle  # ← Espaces = erreur
BINANCE_API_KEY="votre_cle"  # ← Quotes = erreur
```

---

## 🧪 Tester Votre Configuration

### Test 1: Validateur automatique
```bash
python -m core.config_validator
```

### Test 2: Test de connexion
```bash
python test_connection.py
```

### Test 3: Test manuel
```python
from config import CONFIG
from core import BinanceClient

client = BinanceClient(CONFIG)
ticker = client.get_ticker()
print(f"Prix BTC: ${ticker['last']}")
```

---

## 🔧 Autres Problèmes Possibles

### Problème 1: "Timestamp for this request is outside of the recvWindow"
**Cause**: Horloge système désynchronisée
**Solution**: 
```bash
# Windows: Synchroniser l'heure
# Panneau de configuration → Date et heure → Synchroniser maintenant

# Linux:
sudo ntpdate pool.ntp.org
```

### Problème 2: "API-key format invalid"
**Cause**: Clés mal copiées (espaces, retours à la ligne)
**Solution**: 
- Copier/coller directement depuis Binance
- Vérifier qu'il n'y a pas d'espaces avant/après
- Pas de retours à la ligne

### Problème 3: "Invalid API-key, IP, or permissions"
**Cause**: Restrictions IP ou permissions insuffisantes
**Solution**:
- Vérifier que votre IP n'est pas bloquée
- Activer "Enable Reading" et "Enable Trading"
- Désactiver IP whitelist temporairement pour tester

---

## 📋 Checklist Finale

Avant de lancer le bot, vérifiez:

- [ ] `.env` existe et est configuré
- [ ] `BINANCE_API_KEY` est remplie (pas de placeholder)
- [ ] `BINANCE_SECRET_KEY` est remplie (pas de placeholder)
- [ ] `BINANCE_TESTNET` correspond au type de clés
- [ ] Clés copiées sans espaces ni quotes
- [ ] `python -m core.config_validator` passe
- [ ] `python test_connection.py` passe
- [ ] Horloge système synchronisée
- [ ] (Production) Permissions API correctes
- [ ] (Production) "Enable Withdrawals" DÉSACTIVÉ

---

## 🆘 Toujours Bloqué?

### Étape 1: Supprimer et recréer les clés
```
1. Sur Binance (testnet ou prod), supprimez vos clés actuelles
2. Créez de nouvelles clés API
3. Recopiez-les dans .env
4. Relancez les tests
```

### Étape 2: Essayer avec des clés testnet fraîches
```
1. Créez un nouveau compte sur testnet.binance.vision
2. Générez de nouvelles clés
3. Testez uniquement en testnet d'abord
```

### Étape 3: Vérifier les logs
```
Les logs sont dans: logs/
Cherchez des messages d'erreur détaillés
```

---

## 📞 Support

Si le problème persiste:

1. Lancez: `python -m core.config_validator`
2. Copiez le message d'erreur complet
3. Vérifiez les logs dans `logs/`
4. Consultez la FAQ dans README.md

---

## 🎉 Ça Marche!

Une fois la connexion établie:

1. ✅ Lancez un backtest pour tester les stratégies
2. ✅ Paper trading pour tester en temps réel (sans risque)
3. ✅ Optimisez vos paramètres
4. ✅ Si satisfait, passez en live avec capital minimum

**Bon trading! 🚀📈**
