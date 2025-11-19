"""
Safety Module - Quantum Trader Pro
Module centralisé pour la sécurité des opérations critiques
Gestion des race conditions, DataFrames vides, ordres partiels
"""

import threading
from typing import Dict, Optional, Any, List
from contextlib import contextmanager
from functools import wraps
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger('Safety')


# ============================================================================
# THREAD-SAFE COUNTERS
# ============================================================================

class ThreadSafeCounter:
    """Compteur thread-safe pour statistiques et métriques"""

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, value: int = 1) -> int:
        """
        Incrémente un compteur de manière thread-safe

        Args:
            name: Nom du compteur
            value: Valeur à ajouter

        Returns:
            Nouvelle valeur du compteur
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = 0
            self._counters[name] += value
            return self._counters[name]

    def get(self, name: str) -> int:
        """Récupère la valeur d'un compteur"""
        with self._lock:
            return self._counters.get(name, 0)

    def reset(self, name: str) -> None:
        """Reset un compteur"""
        with self._lock:
            self._counters[name] = 0

    def get_all(self) -> Dict[str, int]:
        """Récupère tous les compteurs"""
        with self._lock:
            return self._counters.copy()


# Instance globale pour compteurs
global_counters = ThreadSafeCounter()


# ============================================================================
# ORDER QUEUE (THREAD-SAFE)
# ============================================================================

class OrderQueue:
    """
    File d'attente thread-safe pour les ordres
    """

    def __init__(self, max_size: int = 100):
        self.queue: List[Dict] = []
        self.max_size = max_size
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def put(self, order: Dict, timeout: Optional[float] = None) -> bool:
        """
        Ajoute un ordre à la queue

        Args:
            order: Ordre à ajouter
            timeout: Timeout en secondes

        Returns:
            True si ajouté avec succès
        """
        with self.not_full:
            while len(self.queue) >= self.max_size:
                if not self.not_full.wait(timeout):
                    return False

            self.queue.append(order)
            self.not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Récupère un ordre de la queue

        Args:
            timeout: Timeout en secondes

        Returns:
            Ordre ou None si timeout
        """
        with self.not_empty:
            while not self.queue:
                if not self.not_empty.wait(timeout):
                    return None

            order = self.queue.pop(0)
            self.not_full.notify()
            return order

    def size(self) -> int:
        """Retourne la taille de la queue"""
        with self.lock:
            return len(self.queue)

    def clear(self) -> None:
        """Vide la queue"""
        with self.lock:
            self.queue.clear()
            self.not_full.notify_all()


# ============================================================================
# THREAD SAFETY - Locks centralisés
# ============================================================================

class GlobalLockManager:
    """
    Gestionnaire de locks global pour éviter les race conditions.
    Singleton pour garantir l'unicité des locks.
    """

    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._locks = {
            'orders': threading.RLock(),
            'positions': threading.RLock(),
            'balance': threading.RLock(),
            'market_data': threading.RLock(),
            'risk': threading.RLock(),
        }
        self._initialized = True
        logger.info("✅ GlobalLockManager initialisé")

    @contextmanager
    def lock(self, resource: str):
        """
        Context manager pour verrouiller une ressource

        Args:
            resource: Nom de la ressource ('orders', 'positions', etc.)
        """
        if resource not in self._locks:
            logger.warning(f"Lock '{resource}' non existant, création...")
            self._locks[resource] = threading.RLock()

        lock = self._locks[resource]
        acquired = False

        try:
            # Timeout de 30 secondes pour éviter deadlocks
            acquired = lock.acquire(timeout=30)
            if not acquired:
                logger.error(f"❌ Timeout acquiring lock '{resource}'")
                raise TimeoutError(f"Impossible d'acquérir le lock '{resource}'")
            yield
        finally:
            if acquired:
                lock.release()

    @contextmanager
    def multi_lock(self, *resources: str):
        """
        Verrouille plusieurs ressources dans un ordre déterministe
        pour éviter les deadlocks

        Args:
            *resources: Noms des ressources à verrouiller
        """
        # Trier pour ordre déterministe
        sorted_resources = sorted(resources)
        locks_acquired = []

        try:
            for resource in sorted_resources:
                if resource not in self._locks:
                    self._locks[resource] = threading.RLock()

                if self._locks[resource].acquire(timeout=30):
                    locks_acquired.append(resource)
                else:
                    raise TimeoutError(f"Timeout acquiring lock '{resource}'")

            yield

        finally:
            # Libérer dans l'ordre inverse
            for resource in reversed(locks_acquired):
                self._locks[resource].release()


# Instance globale
global_locks = GlobalLockManager()


def synchronized(resource: str):
    """
    Décorateur pour synchroniser une méthode

    Args:
        resource: Nom de la ressource à verrouiller
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with global_locks.lock(resource):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# SAFE DATAFRAME OPERATIONS
# ============================================================================

def safe_dataframe_access(df: pd.DataFrame, operation: str = "check") -> bool:
    """
    Vérifie si un DataFrame est valide pour des opérations

    Args:
        df: DataFrame à vérifier
        operation: Type d'opération ('check', 'iloc', 'compute')

    Returns:
        True si le DataFrame est valide
    """
    if df is None:
        logger.warning("DataFrame est None")
        return False

    if not isinstance(df, pd.DataFrame):
        logger.warning(f"Objet n'est pas un DataFrame: {type(df)}")
        return False

    if df.empty:
        logger.warning("DataFrame est vide")
        return False

    if len(df) == 0:
        logger.warning("DataFrame a 0 lignes")
        return False

    return True


def safe_iloc(df: pd.DataFrame, index: int, default: Any = None) -> Any:
    """
    Accès sécurisé à iloc avec gestion des index invalides

    Args:
        df: DataFrame
        index: Index à accéder (peut être négatif)
        default: Valeur par défaut si accès impossible

    Returns:
        Ligne ou valeur par défaut
    """
    if not safe_dataframe_access(df):
        return default

    try:
        # Gérer les indices négatifs
        if index < 0:
            actual_index = len(df) + index
        else:
            actual_index = index

        if actual_index < 0 or actual_index >= len(df):
            logger.warning(f"Index {index} hors limites (taille: {len(df)})")
            return default

        return df.iloc[actual_index]

    except Exception as e:
        logger.error(f"Erreur safe_iloc: {e}")
        return default


def safe_column_access(df: pd.DataFrame, column: str, default: Any = None) -> Any:
    """
    Accès sécurisé à une colonne

    Args:
        df: DataFrame
        column: Nom de la colonne
        default: Valeur par défaut

    Returns:
        Colonne ou valeur par défaut
    """
    if not safe_dataframe_access(df):
        return default

    if column not in df.columns:
        logger.warning(f"Colonne '{column}' non trouvée. Disponibles: {list(df.columns)}")
        return default

    return df[column]


def safe_last_value(series_or_df: Any, column: Optional[str] = None, default: float = 0.0) -> float:
    """
    Récupère la dernière valeur d'une Series ou DataFrame de manière sécurisée

    Args:
        series_or_df: Series ou DataFrame
        column: Nom de colonne si DataFrame
        default: Valeur par défaut

    Returns:
        Dernière valeur ou défaut
    """
    try:
        if isinstance(series_or_df, pd.DataFrame):
            if not safe_dataframe_access(series_or_df):
                return default

            if column and column in series_or_df.columns:
                value = series_or_df[column].iloc[-1]
            else:
                logger.warning(f"Colonne '{column}' non spécifiée ou absente")
                return default

        elif isinstance(series_or_df, pd.Series):
            if len(series_or_df) == 0:
                return default
            value = series_or_df.iloc[-1]

        else:
            return default

        # Vérifier NaN
        if pd.isna(value):
            logger.warning("Dernière valeur est NaN")
            return default

        return float(value)

    except Exception as e:
        logger.error(f"Erreur safe_last_value: {e}")
        return default


def ensure_minimum_data(df: pd.DataFrame, min_rows: int, operation_name: str = "operation") -> bool:
    """
    Vérifie qu'il y a assez de données pour une opération

    Args:
        df: DataFrame à vérifier
        min_rows: Nombre minimum de lignes requises
        operation_name: Nom de l'opération pour logging

    Returns:
        True si assez de données
    """
    if not safe_dataframe_access(df):
        logger.error(f"Données insuffisantes pour {operation_name}: DataFrame invalide")
        return False

    if len(df) < min_rows:
        logger.warning(
            f"Données insuffisantes pour {operation_name}: "
            f"{len(df)} lignes < {min_rows} requises"
        )
        return False

    return True


# ============================================================================
# ORDER VALIDATION & PARTIAL FILLS
# ============================================================================

def validate_order_result(order_result: Dict, expected_amount: float) -> Dict[str, Any]:
    """
    Valide le résultat d'un ordre et gère les remplissages partiels

    Args:
        order_result: Résultat de l'ordre de l'exchange
        expected_amount: Montant attendu

    Returns:
        Dict avec statut et informations
    """
    if not order_result:
        return {
            'valid': False,
            'error': "Ordre vide ou None",
            'filled_amount': 0,
            'fill_percentage': 0,
            'is_partial': False
        }

    # Extraire les informations
    filled = float(order_result.get('filled', 0))
    remaining = float(order_result.get('remaining', expected_amount))
    status = order_result.get('status', 'unknown')

    # Calculer le pourcentage de remplissage
    fill_percentage = (filled / expected_amount * 100) if expected_amount > 0 else 0

    # Déterminer si c'est un remplissage partiel
    is_partial = 0 < fill_percentage < 95  # Moins de 95% = partiel
    is_complete = fill_percentage >= 95

    result = {
        'valid': filled > 0,
        'filled_amount': filled,
        'remaining_amount': remaining,
        'fill_percentage': fill_percentage,
        'is_partial': is_partial,
        'is_complete': is_complete,
        'status': status,
        'order_id': order_result.get('id', 'unknown'),
        'error': None
    }

    # Logging approprié
    if is_complete:
        logger.info(f"✅ Ordre rempli complètement: {filled:.6f} ({fill_percentage:.1f}%)")
    elif is_partial:
        logger.warning(
            f"⚠️ Ordre partiellement rempli: {filled:.6f}/{expected_amount:.6f} "
            f"({fill_percentage:.1f}%)"
        )
        result['error'] = f"Remplissage partiel: {fill_percentage:.1f}%"
    else:
        logger.error(f"❌ Ordre non rempli: {status}")
        result['error'] = f"Ordre non rempli: {status}"
        result['valid'] = False

    return result


def handle_partial_fill(
    order_result: Dict,
    position_id: str,
    on_complete: callable = None,
    on_partial: callable = None,
    retry_remaining: bool = False
) -> Dict[str, Any]:
    """
    Gère intelligemment les ordres partiellement remplis

    Args:
        order_result: Résultat de l'ordre
        position_id: ID de la position associée
        on_complete: Callback si ordre complet
        on_partial: Callback si ordre partiel
        retry_remaining: Si True, suggère de réessayer le montant restant

    Returns:
        Dict avec actions à prendre
    """
    validation = validate_order_result(
        order_result,
        order_result.get('amount', 0)
    )

    actions = {
        'proceed': validation['valid'],
        'retry_needed': False,
        'retry_amount': 0,
        'cancel_remaining': False,
        'validation': validation
    }

    if validation['is_complete']:
        if on_complete:
            on_complete(position_id, validation)
        actions['proceed'] = True

    elif validation['is_partial']:
        if on_partial:
            on_partial(position_id, validation)

        if retry_remaining and validation['remaining_amount'] > 0:
            actions['retry_needed'] = True
            actions['retry_amount'] = validation['remaining_amount']
            logger.info(f"🔄 Retry recommandé pour {validation['remaining_amount']:.6f}")
        else:
            actions['cancel_remaining'] = True
            logger.warning(f"⚠️ Acceptation du remplissage partiel")

    else:
        actions['proceed'] = False
        logger.error(f"❌ Ordre échoué pour position {position_id}")

    return actions


# ============================================================================
# SAFE PRICE MONITORING
# ============================================================================

def safe_get_current_price(
    client: Any,
    fallback_price: Optional[float] = None,
    max_retries: int = 3
) -> Optional[float]:
    """
    Récupère le prix actuel de manière sécurisée avec retry

    Args:
        client: Client exchange
        fallback_price: Prix de secours
        max_retries: Nombre de tentatives

    Returns:
        Prix actuel ou None si échec
    """
    for attempt in range(max_retries):
        try:
            ticker = client.get_ticker()

            if not ticker:
                logger.warning(f"Ticker vide (tentative {attempt + 1}/{max_retries})")
                continue

            price = ticker.get('last')

            if price is None or price <= 0:
                logger.warning(f"Prix invalide: {price}")
                continue

            return float(price)

        except Exception as e:
            logger.error(f"Erreur récupération prix (tentative {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                import time
                time.sleep(1)  # Attendre avant retry

    # Utiliser fallback si disponible
    if fallback_price and fallback_price > 0:
        logger.warning(f"⚠️ Utilisation du prix de secours: {fallback_price}")
        return fallback_price

    logger.error("❌ Impossible de récupérer le prix actuel")
    return None


def validate_price_movement(
    current_price: float,
    reference_price: float,
    max_deviation_percent: float = 10.0
) -> bool:
    """
    Vérifie que le prix n'a pas bougé de manière anormale

    Args:
        current_price: Prix actuel
        reference_price: Prix de référence
        max_deviation_percent: Déviation maximale acceptable

    Returns:
        True si le prix est dans les limites normales
    """
    if reference_price <= 0:
        return False

    deviation = abs(current_price - reference_price) / reference_price * 100

    if deviation > max_deviation_percent:
        logger.warning(
            f"⚠️ Mouvement de prix anormal: {deviation:.2f}% "
            f"(max: {max_deviation_percent}%)"
        )
        return False

    return True


# ============================================================================
# ATOMIC OPERATIONS
# ============================================================================

@contextmanager
def atomic_position_update(position_id: str):
    """
    Garantit une mise à jour atomique d'une position

    Args:
        position_id: ID de la position
    """
    with global_locks.multi_lock('positions', 'orders'):
        logger.debug(f"🔒 Mise à jour atomique position {position_id}")
        yield
        logger.debug(f"🔓 Fin mise à jour atomique position {position_id}")


@contextmanager
def atomic_order_execution():
    """
    Garantit une exécution atomique d'un ordre
    """
    with global_locks.multi_lock('orders', 'balance', 'positions'):
        logger.debug("🔒 Exécution atomique ordre")
        yield
        logger.debug("🔓 Fin exécution atomique ordre")


def safe_execute_with_retry(
    func: callable,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    *args,
    **kwargs
) -> Any:
    """
    Exécute une fonction avec retry automatique

    Args:
        func: Fonction à exécuter
        max_retries: Nombre de tentatives
        retry_delay: Délai entre tentatives (secondes)
        *args, **kwargs: Arguments pour la fonction

    Returns:
        Résultat de la fonction ou None si échec
    """
    import time

    last_error = None

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            logger.warning(
                f"Tentative {attempt + 1}/{max_retries} échouée: {e}"
            )

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Backoff exponentiel

    logger.error(f"❌ Toutes les tentatives ont échoué: {last_error}")
    return None


class ThreadSafetyManager:
    """
    Gestionnaire de sécurité des threads - Wrapper de compatibilité

    Cette classe fournit une interface compatible avec l'ancien ThreadSafetyManager
    tout en utilisant GlobalLockManager en interne.
    """

    def __init__(self):
        self.logger = logger
        self._lock_manager = GlobalLockManager()
        self._counters = ThreadSafeCounter()
        self.logger.info("✅ Thread Safety Manager initialisé")

    @contextmanager
    def lock(self, resource: str):
        """
        Context manager pour lock une ressource

        Usage:
            with thread_manager.lock('orders'):
                # Code critique pour ordres
                pass
        """
        with self._lock_manager.lock(resource):
            yield

    def synchronized_method(self, resource: str):
        """
        Décorateur pour synchroniser une méthode entière

        Usage:
            @thread_manager.synchronized_method('orders')
            def place_order(self, ...):
                pass
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.lock(resource):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def increment_counter(self, name: str, value: int = 1) -> int:
        """Incrémente un compteur de manière thread-safe"""
        return self._counters.increment(name, value)

    def get_counter(self, name: str) -> int:
        """Récupère la valeur d'un compteur"""
        return self._counters.get(name)

    def reset_counter(self, name: str) -> None:
        """Reset un compteur"""
        self._counters.reset(name)


# Instance globale pour compatibilité
thread_manager = ThreadSafetyManager()


__all__ = [
    # Thread Safety
    'GlobalLockManager',
    'global_locks',
    'synchronized',
    'ThreadSafetyManager',
    'thread_manager',

    # Thread-Safe Counters
    'ThreadSafeCounter',
    'global_counters',

    # Order Queue
    'OrderQueue',

    # Safe DataFrame
    'safe_dataframe_access',
    'safe_iloc',
    'safe_column_access',
    'safe_last_value',
    'ensure_minimum_data',

    # Order Validation
    'validate_order_result',
    'handle_partial_fill',

    # Price Monitoring
    'safe_get_current_price',
    'validate_price_movement',

    # Atomic Operations
    'atomic_position_update',
    'atomic_order_execution',
    'safe_execute_with_retry'
]
