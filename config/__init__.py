"""
Configuration Module - Quantum Trader Pro
Chargement et validation de la configuration avec gestion des clés API
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import sys

# Import sécurité (avec fallback si module non disponible)
try:
    from utils.security import (
        APIKeyValidator,
        SecretsMasker,
        InputSanitizer,
        validate_env_security
    )
    SECURITY_MODULE_AVAILABLE = True
except ImportError:
    SECURITY_MODULE_AVAILABLE = False

# Import validateur de schéma
try:
    from utils.config_schema import validate_config_schema
    SCHEMA_VALIDATOR_AVAILABLE = True
except ImportError:
    SCHEMA_VALIDATOR_AVAILABLE = False

class ConfigError(Exception):
    """Erreur de configuration"""
    pass

class ConfigLoader:
    """Chargeur de configuration avec validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or Path(__file__).parent / 'config.yaml'
        self.env_path = Path(__file__).parent.parent / '.env'
        
        # Charger .env
        self._load_env()
        
        # Charger YAML
        self.config = self._load_yaml()
        
        # Injecter variables d'environnement
        self._inject_env_vars()
        
        # Valider configuration
        self._validate()
    
    def _load_env(self):
        """Charge les variables d'environnement depuis .env"""
        if not self.env_path.exists():
            print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    ⚠️  FICHIER .env MANQUANT                     ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Le fichier .env est OBLIGATOIRE pour configurer les clés API.   ║
║                                                                   ║
║  📋 INSTRUCTIONS:                                                ║
║  1. Copiez .env.example vers .env                                ║
║  2. Éditez .env avec vos clés API                                ║
║  3. Relancez le programme                                        ║
║                                                                   ║
║  🔑 CLÉS TESTNET BINANCE:                                        ║
║  → https://testnet.binance.vision/                               ║
║  → Créez un compte et générez vos clés API                       ║
║                                                                   ║
║  ⚠️  NE PAS utiliser vos clés de production!                     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
            """)
            sys.exit(1)
        
        load_dotenv(self.env_path)
        print("✅ Fichier .env chargé")
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Charge configuration depuis YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration chargée depuis {self.config_path}")
            return config
        except FileNotFoundError:
            raise ConfigError(f"❌ Fichier config.yaml introuvable: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigError(f"❌ Erreur parsing YAML: {e}")
    
    def _inject_env_vars(self):
        """Injecte les variables d'environnement dans la config"""
        
        # BINANCE API
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        private_key_path = os.getenv('BINANCE_PRIVATE_KEY_PATH', 'test-prv-key.pem')
        testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        
        if not api_key:
            print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                 ⚠️  CLÉS API MANQUANTES                            ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Les clés API Binance ne sont pas configurées dans .env           ║
║                                                                   ║
║  📋 POUR TESTNET (RSA):                                           ║
║  1. Générez les clés RSA:                                         ║
║     openssl genrsa -out test-prv-key.pem 2048                     ║
║     openssl rsa -in test-prv-key.pem -pubout -out test-pub-key.pem║
║                                                                   ║
║  2. Enregistrez test-pub-key.pem sur:                             ║
║     https://testnet.binance.vision/                               ║
║                                                                   ║
║  3. Mettez l'API Key dans .env:                                   ║
║     BINANCE_API_KEY=votre_api_key_testnet                         ║
║     BINANCE_PRIVATE_KEY_PATH=test-prv-key.pem                     ║
║     BINANCE_TESTNET=true                                          ║
║                                                                   ║
║  📋 POUR PRODUCTION (HMAC):                                       ║
║     BINANCE_API_KEY=votre_api_key                                 ║
║     BINANCE_SECRET_KEY=votre_secret_key                         b ║
║     BINANCE_TESTNET=false                                       b ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
            """)
            sys.exit(1)
        
        # Pour testnet, pas besoin de secret_key mais de private_key_path
        if testnet and not os.path.exists(private_key_path):
            if not secret_key:  # Si pas de fichier PEM, on peut essayer avec secret (compatibilité)
                print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║              ⚠️  CONFIGURATION TESTNET RSA                       ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Le testnet Binance nécessite maintenant RSA au lieu de HMAC!    ║
║                                                                   ║
║  Fichier clé privée non trouvé: {private_key_path:<33} ║
║                                                                   ║
║  Générez les clés avec:                                          ║
║  openssl genrsa -out test-prv-key.pem 2048                       ║
║  openssl rsa -in test-prv-key.pem -pubout -out test-pub-key.pem  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
                """)
        
        # Pour production, on a besoin du secret_key
        if not testnet and not secret_key:
            print("❌ Secret Key requis pour le mode production!")
            sys.exit(1)
        
        # Injecter dans config
        self.config['exchange']['primary']['api_key'] = api_key
        self.config['exchange']['primary']['secret_key'] = secret_key
        self.config['exchange']['primary']['private_key_path'] = private_key_path
        self.config['exchange']['primary']['testnet'] = testnet
        
        # Symbol
        if symbol := os.getenv('SYMBOL'):
            self.config['symbols']['primary'] = symbol
        
        # Capital
        if initial_capital := os.getenv('INITIAL_CAPITAL'):
            self.config['capital']['initial'] = float(initial_capital)
        
        # Risk
        if max_risk := os.getenv('MAX_RISK_PER_TRADE'):
            self.config['risk']['max_risk_per_trade_percent'] = float(max_risk)
        
        if max_daily_loss := os.getenv('MAX_DAILY_LOSS'):
            self.config['risk']['max_daily_loss_percent'] = float(max_daily_loss)
        
        if max_positions := os.getenv('MAX_POSITIONS'):
            self.config['risk']['max_positions_simultaneous'] = int(max_positions)
        
        # Telegram
        if telegram_enabled := os.getenv('TELEGRAM_ENABLED'):
            self.config['monitoring']['telegram']['enabled'] = telegram_enabled.lower() == 'true'
        
        if telegram_token := os.getenv('TELEGRAM_BOT_TOKEN'):
            self.config['monitoring']['telegram']['bot_token'] = telegram_token
        
        if telegram_chat_id := os.getenv('TELEGRAM_CHAT_ID'):
            self.config['monitoring']['telegram']['chat_id'] = telegram_chat_id
        
        # ML
        if ml_enabled := os.getenv('ML_ENABLED'):
            self.config['ml']['enabled'] = ml_enabled.lower() == 'true'
        
        # Database
        if db_url := os.getenv('DATABASE_URL'):
            self.config['database']['url'] = db_url
        
        # Logging
        if log_level := os.getenv('LOG_LEVEL'):
            self.config['logging']['level'] = log_level
        
        print("✅ Variables d'environnement injectées")
    
    def _validate(self):
        """Valide la configuration avec vérifications de sécurité approfondies"""
        errors = []
        warnings = []

        # =========================================================================
        # 0. VALIDATION SCHÉMA STRUCTUREL
        # =========================================================================
        if SCHEMA_VALIDATOR_AVAILABLE:
            print("🔍 Validation schéma configuration...")
            schema_valid, schema_errors = validate_config_schema(self.config)
            if not schema_valid:
                for error in schema_errors:
                    errors.append(f"❌ Schema: {error}")
        else:
            warnings.append("⚠️ Validateur de schéma non disponible")

        # =========================================================================
        # 1. VALIDATION SÉCURITÉ ENVIRONNEMENT
        # =========================================================================
        if SECURITY_MODULE_AVAILABLE:
            print("🔒 Vérification sécurité environnement...")
            env_ok, env_issues = validate_env_security()
            for issue in env_issues:
                if '❌' in issue:
                    errors.append(issue)
                elif '⚠️' in issue:
                    warnings.append(issue)

        # =========================================================================
        # 2. VALIDATION API KEYS
        # =========================================================================
        exchange = self.config.get('exchange', {}).get('primary', {})
        testnet = exchange.get('testnet', False)

        # API Key
        api_key = exchange.get('api_key', '')
        if not api_key:
            errors.append("❌ API Key manquante")
        elif SECURITY_MODULE_AVAILABLE:
            valid, error_msg = APIKeyValidator.validate_binance_api_key(api_key)
            if not valid:
                errors.append(f"❌ API Key invalide: {error_msg}")
        elif len(api_key) < 20:
            errors.append("⚠️  API Key semble invalide (trop courte)")

        # Secret Key (requis pour production HMAC)
        secret_key = exchange.get('secret_key', '')
        if not testnet and not secret_key:
            errors.append("❌ Secret Key requise pour le mode production")
        elif secret_key and SECURITY_MODULE_AVAILABLE:
            valid, error_msg = APIKeyValidator.validate_secret_key(secret_key)
            if not valid:
                warnings.append(f"⚠️  Secret Key: {error_msg}")

        # Private Key Path (pour testnet RSA)
        if testnet:
            private_key_path = exchange.get('private_key_path', '')
            if private_key_path and SECURITY_MODULE_AVAILABLE:
                valid, error_msg = APIKeyValidator.validate_private_key_path(private_key_path)
                if not valid:
                    warnings.append(f"⚠️  Private Key: {error_msg}")
            elif not private_key_path and not secret_key:
                warnings.append("⚠️  Testnet: ni private_key_path ni secret_key configuré")

        # =========================================================================
        # 3. VALIDATION CAPITAL ET RISQUE
        # =========================================================================
        capital = self.config.get('capital', {})
        initial_capital = capital.get('initial', 0)

        if initial_capital <= 0:
            errors.append("❌ Capital initial doit être > 0")
        elif initial_capital < 100:
            warnings.append(f"⚠️  Capital initial faible: ${initial_capital} (recommandé: >= $100)")
        elif initial_capital > 100000:
            warnings.append(f"⚠️  Capital élevé: ${initial_capital}. Vérifiez que c'est intentionnel.")

        risk = self.config.get('risk', {})
        max_risk = risk.get('max_risk_per_trade_percent', 0)
        if max_risk <= 0:
            errors.append("❌ max_risk_per_trade_percent doit être > 0")
        elif max_risk > 5:
            warnings.append(f"⚠️  Risque par trade élevé: {max_risk}% (recommandé: <= 2%)")

        max_daily_loss = risk.get('max_daily_loss_percent', 0)
        if max_daily_loss <= 0:
            errors.append("❌ max_daily_loss_percent doit être > 0")
        elif max_daily_loss > 20:
            warnings.append(f"⚠️  Perte journalière max élevée: {max_daily_loss}% (recommandé: <= 10%)")

        max_positions = risk.get('max_positions_simultaneous', 0)
        if max_positions <= 0:
            errors.append("❌ max_positions_simultaneous doit être > 0")
        elif max_positions > 10:
            warnings.append(f"⚠️  Beaucoup de positions simultanées: {max_positions}")

        # =========================================================================
        # 4. VALIDATION SYMBOL
        # =========================================================================
        symbol = self.config.get('symbols', {}).get('primary', '')
        if not symbol:
            errors.append("❌ Symbol principal manquant")
        elif SECURITY_MODULE_AVAILABLE:
            try:
                sanitized = InputSanitizer.sanitize_symbol(symbol)
                self.config['symbols']['primary'] = sanitized  # Utiliser version nettoyée
            except ValueError as e:
                errors.append(f"❌ Symbol invalide: {e}")
        elif '/' not in symbol:
            errors.append(f"❌ Symbol invalide: {symbol} (format: BASE/QUOTE)")

        # =========================================================================
        # 5. AFFICHAGE DES RÉSULTATS
        # =========================================================================
        if errors or warnings:
            print("\n" + "="*70)
            if errors:
                print("❌ ERREURS DE CONFIGURATION")
            else:
                print("⚠️  AVERTISSEMENTS DE CONFIGURATION")
            print("="*70)

            for error in errors:
                print(f"  {error}")
            for warning in warnings:
                print(f"  {warning}")

            print("="*70)

            if errors:
                print("\n⛔ Erreurs critiques détectées. Arrêt du programme.")
                sys.exit(1)
            else:
                print("\n⚠️  Warnings détectés. Continuez à vos risques et périls.")
                # En mode non-interactif, on continue après affichage des warnings
                if sys.stdin.isatty():
                    input("Appuyez sur Entrée pour continuer...")
        else:
            print("✅ Configuration validée avec succès")
    
    def get(self) -> Dict[str, Any]:
        """Retourne la configuration complète"""
        return self.config
    
    def display_summary(self):
        """Affiche un résumé de la configuration"""
        exchange = self.config.get('exchange', {}).get('primary', {})
        capital = self.config.get('capital', {})
        risk = self.config.get('risk', {})
        symbol = self.config.get('symbols', {}).get('primary', 'BTC/USDT')

        testnet_indicator = "🧪 TESTNET" if exchange.get('testnet', True) else "⚠️  PRODUCTION"
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                QUANTUM TRADER PRO - CONFIGURATION                 ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  📊 MODE: {testnet_indicator:<52} ║
║  💰 Capital Initial: ${capital['initial']:<38.2f} ║
║  🎯 Symbol: {symbol:<49} ║
║  📈 Risk per Trade: {risk['max_risk_per_trade_percent']:<43.1f}% ║
║  🛡️  Max Daily Loss: {risk['max_daily_loss_percent']:<42.1f}% ║
║  📦 Max Positions: {risk['max_positions_simultaneous']:<45} ║
║                                                                   ║
║  🔑 API Key: {exchange['api_key'][:20]}...{exchange['api_key'][-4:]:<21} ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
        """)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def load_config(config_path: Optional[str] = None, display_summary: bool = True) -> Dict[str, Any]:
    """
    Charge et retourne la configuration

    Args:
        config_path: Chemin vers config.yaml (optionnel)
        display_summary: Afficher le résumé de configuration

    Returns:
        Dictionnaire de configuration
    """
    loader = ConfigLoader(config_path)
    if display_summary:
        loader.display_summary()
    return loader.get()


# Cache pour singleton paresseux
_config_cache: Optional[Dict[str, Any]] = None
_config_loader_cache: Optional[ConfigLoader] = None


def get_config() -> Dict[str, Any]:
    """
    Récupère la configuration (chargement paresseux)

    Charge la configuration une seule fois et la met en cache.
    Équivalent de l'ancien CONFIG global mais en lazy loading.

    Returns:
        Dictionnaire de configuration
    """
    global _config_cache, _config_loader_cache

    if _config_cache is None:
        _config_loader_cache = ConfigLoader()
        _config_loader_cache.display_summary()
        _config_cache = _config_loader_cache.get()

    return _config_cache


def reset_config_cache() -> None:
    """
    Reset le cache de configuration

    Utile pour les tests ou rechargement dynamique.
    """
    global _config_cache, _config_loader_cache
    _config_cache = None
    _config_loader_cache = None


# NOTE: Plus de chargement automatique à l'import!
# Utilisez ConfigLoader() directement ou appelez get_config() pour lazy loading.

# Export
__all__ = ['load_config', 'get_config', 'reset_config_cache', 'ConfigLoader', 'ConfigError']
