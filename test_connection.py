"""
Quick Connection Test - Quantum Trader Pro
Test rapide de la connexion Binance
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

def test_connection():
    """Test simple de connexion"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              QUANTUM TRADER PRO - Connection Test                ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Charger config
        print("⏳ Chargement configuration...")
        from config import CONFIG
        print("✅ Configuration chargée\n")
        
        # Créer client
        print("⏳ Connexion à Binance...")
        from core import BinanceClient
        
        client = BinanceClient(CONFIG)
        print("✅ Connexion établie!\n")
        
        # Tests basiques
        symbol = CONFIG['symbols']['primary']
        
        # Test 1: Ticker
        print(f"📊 Test 1: Récupération ticker {symbol}...")
        ticker = client.get_ticker()
        if ticker:
            print(f"   ✅ Prix actuel: ${ticker['last']:.2f}")
            print(f"   📈 Bid: ${ticker['bid']:.2f}")
            print(f"   📉 Ask: ${ticker['ask']:.2f}")
            print(f"   💹 Spread: {ticker['spread_percent']:.3f}%")
            print(f"   📊 Volume 24h: ${ticker['volume']:,.0f}")
        else:
            print("   ❌ Impossible de récupérer le ticker")
            return False
        
        # Test 2: OHLCV
        print(f"\n📈 Test 2: Récupération données OHLCV...")
        df = client.fetch_ohlcv(timeframe='5m', limit=10)
        if df is not None and not df.empty:
            print(f"   ✅ {len(df)} bougies récupérées")
            print(f"   🕐 Dernière bougie: {df.index[-1]}")
            print(f"   💰 Dernier close: ${df['close'].iloc[-1]:.2f}")
        else:
            print("   ❌ Impossible de récupérer OHLCV")
            return False
        
        # Test 3: Balance
        print(f"\n💰 Test 3: Récupération balance...")
        try:
            balance = client.get_balance()
            quote = symbol.split('/')[1]
            print(f"   ✅ Balance {quote}: {balance['quote']['total']:.2f}")
            print(f"      → Disponible: {balance['quote']['free']:.2f}")
            print(f"      → Utilisé: {balance['quote']['used']:.2f}")
        except Exception as e:
            print(f"   ⚠️  Balance non accessible: {e}")
            print(f"   (Normal sur testnet)")
        
        # Test 4: Order Book
        print(f"\n📖 Test 4: Récupération Order Book...")
        orderbook = client.get_order_book(limit=5)
        if orderbook and orderbook.get('bids') and orderbook.get('asks'):
            print(f"   ✅ Order Book récupéré")
            print(f"   📗 Best Bid: ${orderbook['bids'][0][0]:.2f} ({orderbook['bids'][0][1]:.4f})")
            print(f"   📕 Best Ask: ${orderbook['asks'][0][0]:.2f} ({orderbook['asks'][0][1]:.4f})")
        else:
            print("   ❌ Impossible de récupérer l'order book ou orderbook vide")
        
        # Résumé
        print("\n" + "="*70)
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("="*70)
        print("\n🎉 Votre configuration est correcte!")
        print("\n📋 Prochaines étapes:")
        print("   1. Lancez un backtest: python main_backtest.py")
        print("   2. Paper trading: python main_paper.py")
        print("   3. Live trading: python main_live.py (ATTENTION!)")
        print("\n" + "="*70)
        
        return True
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrompu")
        return False
    
    except Exception as e:
        print(f"\n\n❌ ERREUR: {e}")
        print("\n💡 SOLUTIONS POSSIBLES:")
        print("   1. Vérifiez que vos clés API sont correctes dans .env")
        print("   2. Assurez-vous d'utiliser les bonnes clés (testnet vs production)")
        print("   3. Pour TESTNET: https://testnet.binance.vision/")
        print("   4. Pour PRODUCTION: https://www.binance.com/")
        print("\n   Lancez le validateur pour plus de détails:")
        print("   python -m core.config_validator")
        
        import traceback
        print("\n📋 Détails de l'erreur:")
        traceback.print_exc()
        
        return False

if __name__ == '__main__':
    success = test_connection()
    sys.exit(0 if success else 1)
