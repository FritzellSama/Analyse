"""
Configuration Validator - Quantum Trader Pro
Script pour valider la configuration avant de lancer le bot
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

def validate_environment():
    """Valide l'environnement et la configuration"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║          QUANTUM TRADER PRO - Configuration Validator             ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    errors = []
    warnings = []
    
    # 1. Vérifier fichier .env
    print("📋 Étape 1/5: Vérification fichier .env...")
    env_path = Path(__file__).parent.parent / '.env'
    
    if not env_path.exists():
        errors.append("❌ Fichier .env manquant")
        print("   ❌ ERREUR: Fichier .env introuvable")
        print("   → Copiez .env.example vers .env et remplissez vos clés")
    else:
        print("   ✅ Fichier .env trouvé")
    
    # 2. Vérifier clés API
    print("\n🔑 Étape 2/5: Vérification clés API...")
    
    from dotenv import load_dotenv
    load_dotenv(env_path)
    
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    
    if not api_key or api_key == 'your_testnet_api_key_here':
        errors.append("❌ BINANCE_API_KEY non configurée")
        print("   ❌ ERREUR: BINANCE_API_KEY manquante ou non modifiée")
    elif len(api_key) < 20:
        warnings.append("⚠️  BINANCE_API_KEY semble trop courte")
        print("   ⚠️  WARNING: Clé API semble invalide (trop courte)")
    else:
        print(f"   ✅ API Key configurée ({api_key[:10]}...{api_key[-4:]})")
    
    if not secret_key or secret_key == 'your_testnet_secret_key_here':
        if testnet:
            print("   ✅ Ok")
        else:
            errors.append("❌ BINANCE_SECRET_KEY non configurée")
            print("   ❌ ERREUR: BINANCE_SECRET_KEY manquante ou non modifiée")
    elif len(secret_key) < 20:
        warnings.append("⚠️  BINANCE_SECRET_KEY semble trop courte")
        print("   ⚠️  WARNING: Secret Key semble invalide (trop courte)")
    else:
        print(f"   ✅ Secret Key configurée ({secret_key[:10]}...{secret_key[-4:]})")
    
    print(f"   ℹ️  Mode: {'🧪 TESTNET' if testnet else '⚠️  PRODUCTION (argent réel!)'}")
    
    if testnet:
        print("\n   💡 RAPPEL IMPORTANT:")
        print("   → Les clés TESTNET sont différentes des clés de production")
        print("   → Obtenez vos clés testnet sur: https://testnet.binance.vision/")
        print("   → Les clés de Binance.com ne marchent PAS sur le testnet")
    
    # 3. Vérifier config.yaml
    print("\n⚙️  Étape 3/5: Vérification config.yaml...")
    
    try:
        from config import CONFIG
        print("   ✅ config.yaml chargé avec succès")
        
        # Vérifier valeurs critiques
        capital = CONFIG.get('capital', {}).get('initial', 0)
        if capital < 50:
            warnings.append(f"⚠️  Capital initial faible: ${capital}")
            print(f"   ⚠️  WARNING: Capital initial faible (${capital})")
        else:
            print(f"   ✅ Capital initial: ${capital}")
        
        max_risk = CONFIG.get('risk', {}).get('max_risk_per_trade_percent', 0)
        if max_risk > 5:
            warnings.append(f"⚠️  Risk per trade élevé: {max_risk}%")
            print(f"   ⚠️  WARNING: Risk per trade élevé ({max_risk}%)")
        else:
            print(f"   ✅ Risk per trade: {max_risk}%")
        
    except Exception as e:
        errors.append(f"❌ Erreur chargement config: {e}")
        print(f"   ❌ ERREUR: {e}")
    
    # 4. Tester connexion Binance
    print("\n🔌 Étape 4/5: Test connexion Binance...")
    
    if not errors:  # Seulement si pas d'erreurs critiques avant
        try:
            from core import BinanceClient
            
            print("   → Tentative de connexion...")
            client = BinanceClient(CONFIG)
            print("   ✅ Connexion Binance réussie!")
            
            # Tester récupération ticker
            ticker = client.get_ticker()
            if ticker:
                print(f"   ✅ Ticker {CONFIG['symbols']['primary']}: ${ticker['last']:.2f}")
            
            # Tester balance (peut échouer sur testnet)
            try:
                balance = client.get_balance()
                quote = CONFIG['symbols']['primary'].split('/')[1]
                print(f"   ✅ Balance {quote}: {balance['quote']['free']:.2f}")
            except Exception as e:
                print(f"   ⚠️  Balance non accessible (normal sur testnet): {type(e).__name__}")
        
        except Exception as e:
            errors.append(f"❌ Erreur connexion Binance: {e}")
            print(f"   ❌ ERREUR: {e}")
            
            if "Invalid Api-Key ID" in str(e) or "Authentication" in str(e):
                print("\n   💡 DIAGNOSTIC:")
                print("   → Vous utilisez probablement des clés de production sur testnet (ou vice-versa)")
                print("   → Vérifiez que BINANCE_TESTNET dans .env correspond au type de clés")
                print("   → Clés TESTNET: https://testnet.binance.vision/")
                print("   → Clés PRODUCTION: https://www.binance.com/")
    else:
        print("   ⏭️  Test ignoré (erreurs précédentes)")
    
    # 5. Vérifier dépendances Python
    print("\n📦 Étape 5/5: Vérification dépendances...")
    
    required_packages = [
        'ccxt', 'pandas', 'numpy', 'yaml', 'dotenv', 
        'talib', 'colorama', 'sklearn', 'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'dotenv':
                __import__('dotenv')
            elif package == 'yaml':
                __import__('yaml')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} manquant")
    
    if missing_packages:
        warnings.append(f"⚠️  Packages manquants: {', '.join(missing_packages)}")
        print(f"\n   💡 Installez avec: pip install {' '.join(missing_packages)}")
    
    # Résumé final
    print("\n" + "="*70)
    print("📊 RÉSUMÉ VALIDATION")
    print("="*70)
    
    if errors:
        print("\n❌ ERREURS CRITIQUES:")
        for error in errors:
            print(f"   {error}")
        print("\n⛔ IMPOSSIBLE DE DÉMARRER LE BOT")
        print("   Corrigez les erreurs ci-dessus puis relancez la validation.")
        return False
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
        print("\n⚠️  Le bot peut démarrer mais soyez prudent!")
        
        response = input("\n   Continuer malgré les warnings? (y/n): ")
        if response.lower() != 'y':
            print("   Validation annulée.")
            return False
    
    print("\n✅ VALIDATION RÉUSSIE!")
    print("\n🚀 Vous pouvez maintenant lancer le bot:")
    print("   - Backtest: python main_backtest.py")
    print("   - Paper Trading: python main_paper.py")
    print("   - Live Trading: python main_live.py (DANGER!)")
    
    print("\n" + "="*70)
    
    return True

if __name__ == '__main__':
    try:
        success = validate_environment()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
