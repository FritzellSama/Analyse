"""
Security Module - Quantum Trader Pro
Gestion sécurisée des secrets, validation des clés API et protection des données sensibles
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
from utils.logger import setup_logger

logger = setup_logger('Security')


class SecurityError(Exception):
    """Erreur de sécurité"""
    pass


class APIKeyValidator:
    """Validateur de clés API avec vérifications de sécurité"""

    # Patterns pour différents exchanges
    PATTERNS = {
        'binance': {
            'api_key': r'^[A-Za-z0-9]{64}$',
            'secret_key': r'^[A-Za-z0-9]{64}$',
            'min_length': 64,
            'max_length': 64
        },
        'binance_testnet': {
            'api_key': r'^[A-Za-z0-9]{64}$',
            'min_length': 64,
            'max_length': 64
        }
    }

    @staticmethod
    def validate_binance_api_key(api_key: str) -> Tuple[bool, str]:
        """
        Valide une clé API Binance

        Args:
            api_key: Clé API à valider

        Returns:
            Tuple (is_valid, error_message)
        """
        if not api_key:
            return False, "API key is empty"

        # Vérifier longueur
        if len(api_key) < 20:
            return False, f"API key too short ({len(api_key)} chars, expected ~64)"

        if len(api_key) > 128:
            return False, f"API key too long ({len(api_key)} chars)"

        # Vérifier caractères valides
        if not re.match(r'^[A-Za-z0-9]+$', api_key):
            return False, "API key contains invalid characters"

        # Vérifier que ce n'est pas une valeur par défaut
        forbidden_values = [
            'your_api_key_here',
            'apikey',
            'test',
            'demo',
            'YOUR_API_KEY',
            'BINANCE_API_KEY'
        ]

        if api_key.lower() in [v.lower() for v in forbidden_values]:
            return False, "API key appears to be a placeholder value"

        logger.debug(f"✅ API key validated (length: {len(api_key)})")
        return True, ""

    @staticmethod
    def validate_secret_key(secret_key: str) -> Tuple[bool, str]:
        """
        Valide une clé secrète

        Args:
            secret_key: Clé secrète à valider

        Returns:
            Tuple (is_valid, error_message)
        """
        if not secret_key:
            return False, "Secret key is empty"

        if len(secret_key) < 20:
            return False, f"Secret key too short ({len(secret_key)} chars)"

        if len(secret_key) > 128:
            return False, f"Secret key too long ({len(secret_key)} chars)"

        # Vérifier caractères valides
        if not re.match(r'^[A-Za-z0-9]+$', secret_key):
            return False, "Secret key contains invalid characters"

        # Vérifier que ce n'est pas une valeur par défaut
        forbidden_values = [
            'your_secret_key_here',
            'secretkey',
            'secret',
            'YOUR_SECRET_KEY',
            'BINANCE_SECRET_KEY'
        ]

        if secret_key.lower() in [v.lower() for v in forbidden_values]:
            return False, "Secret key appears to be a placeholder value"

        logger.debug(f"✅ Secret key validated (length: {len(secret_key)})")
        return True, ""

    @staticmethod
    def validate_private_key_path(path: str) -> Tuple[bool, str]:
        """
        Valide le chemin vers une clé privée RSA

        Args:
            path: Chemin vers le fichier de clé privée

        Returns:
            Tuple (is_valid, error_message)
        """
        if not path:
            return False, "Private key path is empty"

        key_path = Path(path)

        if not key_path.exists():
            return False, f"Private key file not found: {path}"

        if not key_path.is_file():
            return False, f"Private key path is not a file: {path}"

        # Vérifier extension
        if key_path.suffix not in ['.pem', '.key']:
            logger.warning(f"⚠️  Unusual private key extension: {key_path.suffix}")

        # Vérifier taille (RSA keys are typically 1-4KB)
        file_size = key_path.stat().st_size
        if file_size < 100:
            return False, f"Private key file too small ({file_size} bytes)"

        if file_size > 10000:
            return False, f"Private key file too large ({file_size} bytes)"

        # Vérifier permissions (should not be world-readable)
        if os.name != 'nt':  # Unix-like systems
            mode = key_path.stat().st_mode
            if mode & 0o004:  # World readable
                logger.warning(
                    f"⚠️  SECURITY WARNING: Private key file {path} is world-readable! "
                    f"Run: chmod 600 {path}"
                )

        # Vérifier contenu (doit contenir header PEM)
        try:
            with open(key_path, 'r') as f:
                content = f.read(100)  # Lire début seulement
                if '-----BEGIN' not in content:
                    return False, "File does not appear to be a valid PEM key"
        except Exception as e:
            return False, f"Cannot read private key file: {e}"

        logger.debug(f"✅ Private key path validated: {path}")
        return True, ""


class SecretsMasker:
    """Masque les secrets dans les logs et erreurs"""

    SENSITIVE_PATTERNS = [
        (r'api_key["\']?\s*[:=]\s*["\']?([A-Za-z0-9]{20,})', 'api_key=***MASKED***'),
        (r'secret_key["\']?\s*[:=]\s*["\']?([A-Za-z0-9]{20,})', 'secret_key=***MASKED***'),
        (r'password["\']?\s*[:=]\s*["\']?([^\s"\']+)', 'password=***MASKED***'),
        (r'token["\']?\s*[:=]\s*["\']?([A-Za-z0-9_-]{20,})', 'token=***MASKED***'),
        (r'Bearer\s+([A-Za-z0-9_-]{20,})', 'Bearer ***MASKED***'),
    ]

    @classmethod
    def mask_string(cls, text: str) -> str:
        """
        Masque les secrets dans une chaîne

        Args:
            text: Texte potentiellement sensible

        Returns:
            Texte avec secrets masqués
        """
        result = text
        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    @classmethod
    def mask_dict(cls, data: Dict, sensitive_keys: Optional[list] = None) -> Dict:
        """
        Masque les valeurs sensibles dans un dictionnaire

        Args:
            data: Dictionnaire avec potentiellement des secrets
            sensitive_keys: Liste de clés à masquer (défaut: api_key, secret_key, etc.)

        Returns:
            Dictionnaire avec valeurs masquées
        """
        if sensitive_keys is None:
            sensitive_keys = [
                'api_key', 'secret_key', 'password', 'token',
                'private_key', 'secret', 'apikey', 'secretkey'
            ]

        result = {}
        for key, value in data.items():
            if any(s in key.lower() for s in sensitive_keys):
                if isinstance(value, str) and len(value) > 4:
                    result[key] = value[:4] + '***MASKED***'
                else:
                    result[key] = '***MASKED***'
            elif isinstance(value, dict):
                result[key] = cls.mask_dict(value, sensitive_keys)
            else:
                result[key] = value

        return result


class InputSanitizer:
    """Nettoie et valide les entrées utilisateur"""

    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """
        Nettoie et valide un symbole de trading

        Args:
            symbol: Symbole brut (ex: "BTC/USDT")

        Returns:
            Symbole nettoyé

        Raises:
            ValueError: Si symbole invalide
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")

        # Supprimer espaces
        symbol = symbol.strip().upper()

        # Vérifier format
        if '/' not in symbol:
            raise ValueError(f"Invalid symbol format: {symbol} (expected: BASE/QUOTE)")

        parts = symbol.split('/')
        if len(parts) != 2:
            raise ValueError(f"Invalid symbol format: {symbol}")

        base, quote = parts

        # Vérifier caractères
        if not re.match(r'^[A-Z0-9]{2,10}$', base):
            raise ValueError(f"Invalid base currency: {base}")

        if not re.match(r'^[A-Z0-9]{2,10}$', quote):
            raise ValueError(f"Invalid quote currency: {quote}")

        return f"{base}/{quote}"

    @staticmethod
    def sanitize_timeframe(timeframe: str) -> str:
        """
        Valide un timeframe

        Args:
            timeframe: Timeframe brut (ex: "5m", "1h")

        Returns:
            Timeframe validé

        Raises:
            ValueError: Si timeframe invalide
        """
        valid_timeframes = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]

        timeframe = timeframe.strip()

        if timeframe not in valid_timeframes:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid: {', '.join(valid_timeframes)}"
            )

        return timeframe

    @staticmethod
    def sanitize_float(value: any, name: str, min_val: float = None, max_val: float = None) -> float:
        """
        Valide et convertit en float

        Args:
            value: Valeur à convertir
            name: Nom du paramètre (pour messages d'erreur)
            min_val: Valeur minimale autorisée
            max_val: Valeur maximale autorisée

        Returns:
            Float validé

        Raises:
            ValueError: Si valeur invalide
        """
        try:
            result = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number, got: {type(value).__name__}")

        if min_val is not None and result < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got: {result}")

        if max_val is not None and result > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got: {result}")

        return result

    @staticmethod
    def sanitize_int(value: any, name: str, min_val: int = None, max_val: int = None) -> int:
        """
        Valide et convertit en int

        Args:
            value: Valeur à convertir
            name: Nom du paramètre
            min_val: Valeur minimale autorisée
            max_val: Valeur maximale autorisée

        Returns:
            Int validé

        Raises:
            ValueError: Si valeur invalide
        """
        try:
            result = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an integer, got: {type(value).__name__}")

        if min_val is not None and result < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got: {result}")

        if max_val is not None and result > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got: {result}")

        return result


def validate_env_security() -> Tuple[bool, list]:
    """
    Valide la sécurité de l'environnement

    Returns:
        Tuple (all_ok, list of warnings/errors)
    """
    issues = []

    # 1. Vérifier .env existe
    env_path = Path('.env')
    if not env_path.exists():
        issues.append("❌ CRITICAL: .env file not found")
        return False, issues

    # 2. Vérifier permissions .env
    if os.name != 'nt':
        mode = env_path.stat().st_mode
        if mode & 0o004:  # World readable
            issues.append(
                "⚠️  WARNING: .env file is world-readable. "
                "Run: chmod 600 .env"
            )

    # 3. Vérifier que .env est dans .gitignore
    gitignore_path = Path('.gitignore')
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
            if '.env' not in gitignore_content:
                issues.append(
                    "⚠️  WARNING: .env is NOT in .gitignore! "
                    "Your secrets could be committed to git!"
                )
    else:
        issues.append("⚠️  WARNING: No .gitignore file found")

    # 4. Vérifier variables d'environnement critiques
    api_key = os.getenv('BINANCE_API_KEY')
    if api_key:
        valid, error = APIKeyValidator.validate_binance_api_key(api_key)
        if not valid:
            issues.append(f"❌ API Key validation failed: {error}")

    # 5. Vérifier mode testnet vs production
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    if not testnet:
        issues.append(
            "⚠️  CAUTION: Running in PRODUCTION mode with REAL MONEY!"
        )

    # 6. Vérifier capital initial
    capital = os.getenv('INITIAL_CAPITAL', '0')
    try:
        capital_float = float(capital)
        if capital_float < 100:
            issues.append(f"⚠️  WARNING: Low initial capital: ${capital_float}")
        if capital_float > 100000:
            issues.append(
                f"⚠️  WARNING: High initial capital: ${capital_float}. "
                "Double-check this is intentional."
            )
    except ValueError:
        issues.append(f"❌ Invalid INITIAL_CAPITAL: {capital}")

    # 7. Vérifier paramètres de risque
    max_risk = os.getenv('MAX_RISK_PER_TRADE', '1.0')
    try:
        max_risk_float = float(max_risk)
        if max_risk_float > 5.0:
            issues.append(
                f"⚠️  WARNING: High risk per trade: {max_risk_float}%. "
                "Recommended: <= 2%"
            )
    except ValueError:
        pass

    all_ok = not any('❌' in issue for issue in issues)

    return all_ok, issues


def hash_sensitive_data(data: str) -> str:
    """
    Hash des données sensibles pour logging sécurisé

    Args:
        data: Données à hasher

    Returns:
        Hash SHA256 tronqué
    """
    return hashlib.sha256(data.encode()).hexdigest()[:16]


__all__ = [
    'SecurityError',
    'APIKeyValidator',
    'SecretsMasker',
    'InputSanitizer',
    'validate_env_security',
    'hash_sensitive_data'
]
