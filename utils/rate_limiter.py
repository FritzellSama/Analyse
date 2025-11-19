"""
Rate Limiter - Quantum Trader Pro
Protection contre le dépassement des limites API avec tracking intelligent
"""

import time
import threading
from typing import Dict, Optional
from collections import deque
from utils.logger import setup_logger

logger = setup_logger('RateLimiter')


class RateLimitExceeded(Exception):
    """Exception levée quand la limite de rate est dépassée"""
    pass


class BinanceRateLimiter:
    """
    Rate limiter spécialisé pour Binance API avec:
    - Tracking des poids de requêtes
    - Sliding window (fenêtre glissante)
    - Protection contre le ban
    - Support multi-endpoint
    """

    # Limites Binance (par minute)
    LIMITS = {
        'request_weight': 1200,  # Poids total par minute
        'orders': 10,  # Ordres par seconde
        'orders_per_day': 200000,  # Ordres par jour
    }

    # Poids par endpoint
    WEIGHTS = {
        'fetch_ticker': 1,
        'fetch_balance': 5,
        'fetch_ohlcv': 1,
        'create_order': 1,
        'cancel_order': 1,
        'fetch_order': 1,
        'fetch_orders': 5,
        'fetch_my_trades': 5,
        'fetch_time': 1,
        'default': 1
    }

    def __init__(self, safety_margin: float = 0.2):
        """
        Initialise le rate limiter

        Args:
            safety_margin: Marge de sécurité (0.2 = utiliser max 80% de la limite)
        """
        self.safety_margin = safety_margin
        self.lock = threading.RLock()

        # Tracking par fenêtre glissante
        self._request_history: deque = deque()  # (timestamp, weight)
        self._order_history: deque = deque()  # timestamps des ordres
        self._daily_orders: int = 0
        self._daily_reset_time: float = time.time()

        # Stats
        self.total_requests = 0
        self.total_weight = 0
        self.rate_limit_hits = 0

        logger.info(
            f"✅ Rate Limiter initialisé (marge sécurité: {safety_margin*100:.0f}%)"
        )

    def _clean_old_requests(self) -> None:
        """Nettoie les requêtes de plus d'une minute"""
        current_time = time.time()
        cutoff = current_time - 60  # Fenêtre de 1 minute

        # Nettoyer request_history
        while self._request_history and self._request_history[0][0] < cutoff:
            self._request_history.popleft()

        # Nettoyer order_history (fenêtre de 1 seconde pour les ordres)
        order_cutoff = current_time - 1
        while self._order_history and self._order_history[0] < order_cutoff:
            self._order_history.popleft()

        # Reset compteur journalier si nouveau jour
        if current_time - self._daily_reset_time > 86400:  # 24h
            self._daily_orders = 0
            self._daily_reset_time = current_time
            logger.info("📅 Compteur journalier d'ordres réinitialisé")

    def get_current_weight(self) -> int:
        """Retourne le poids total utilisé dans la dernière minute"""
        self._clean_old_requests()
        return sum(weight for _, weight in self._request_history)

    def get_remaining_weight(self) -> int:
        """Retourne le poids restant disponible"""
        current = self.get_current_weight()
        max_safe = int(self.LIMITS['request_weight'] * (1 - self.safety_margin))
        return max(0, max_safe - current)

    def can_request(self, endpoint: str = 'default') -> bool:
        """
        Vérifie si une requête peut être effectuée

        Args:
            endpoint: Nom de l'endpoint

        Returns:
            True si la requête est permise
        """
        with self.lock:
            self._clean_old_requests()

            weight = self.WEIGHTS.get(endpoint, self.WEIGHTS['default'])
            current_weight = self.get_current_weight()
            max_safe = int(self.LIMITS['request_weight'] * (1 - self.safety_margin))

            # Vérifier poids
            if current_weight + weight > max_safe:
                return False

            # Vérifier ordres par seconde
            if endpoint in ['create_order', 'cancel_order']:
                if len(self._order_history) >= self.LIMITS['orders']:
                    return False

                # Vérifier limite journalière
                if self._daily_orders >= self.LIMITS['orders_per_day']:
                    return False

            return True

    def wait_if_needed(self, endpoint: str = 'default', timeout: float = 60) -> bool:
        """
        Attend si nécessaire avant d'effectuer une requête

        Args:
            endpoint: Nom de l'endpoint
            timeout: Timeout max en secondes

        Returns:
            True si ok pour continuer, False si timeout

        Raises:
            RateLimitExceeded: Si timeout dépassé
        """
        start_time = time.time()

        while not self.can_request(endpoint):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"❌ Rate limit timeout après {timeout}s pour {endpoint}")
                raise RateLimitExceeded(f"Timeout waiting for rate limit on {endpoint}")

            # Calculer temps d'attente optimal
            wait_time = self._calculate_wait_time(endpoint)
            logger.debug(f"⏳ Rate limit: attente {wait_time:.2f}s pour {endpoint}")
            time.sleep(min(wait_time, timeout - elapsed))

        return True

    def _calculate_wait_time(self, endpoint: str) -> float:
        """Calcule le temps d'attente optimal"""
        with self.lock:
            self._clean_old_requests()

            # Si pas d'historique, attendre 1 seconde minimum
            if not self._request_history:
                return 1.0

            # Temps jusqu'à expiration de la plus ancienne requête
            oldest_time = self._request_history[0][0]
            wait_until = oldest_time + 60  # +1 minute
            wait_time = max(0.1, wait_until - time.time())

            return min(wait_time, 5.0)  # Max 5 secondes

    def record_request(self, endpoint: str = 'default') -> None:
        """
        Enregistre une requête effectuée

        Args:
            endpoint: Nom de l'endpoint
        """
        with self.lock:
            current_time = time.time()
            weight = self.WEIGHTS.get(endpoint, self.WEIGHTS['default'])

            # Ajouter à l'historique
            self._request_history.append((current_time, weight))

            # Si c'est un ordre
            if endpoint in ['create_order', 'cancel_order']:
                self._order_history.append(current_time)
                self._daily_orders += 1

            # Stats
            self.total_requests += 1
            self.total_weight += weight

            logger.debug(
                f"📊 Request recorded: {endpoint} (weight: {weight}, "
                f"total minute: {self.get_current_weight()}/{self.LIMITS['request_weight']})"
            )

    def get_status(self) -> Dict:
        """Retourne le statut du rate limiter"""
        with self.lock:
            self._clean_old_requests()

            current_weight = self.get_current_weight()
            max_weight = self.LIMITS['request_weight']
            usage_percent = (current_weight / max_weight) * 100

            return {
                'current_weight': current_weight,
                'max_weight': max_weight,
                'remaining_weight': self.get_remaining_weight(),
                'usage_percent': usage_percent,
                'orders_this_second': len(self._order_history),
                'orders_today': self._daily_orders,
                'total_requests': self.total_requests,
                'rate_limit_hits': self.rate_limit_hits,
                'safety_margin': self.safety_margin,
                'status': 'OK' if usage_percent < 80 else 'WARNING' if usage_percent < 95 else 'CRITICAL'
            }

    def __str__(self) -> str:
        status = self.get_status()
        return (
            f"RateLimiter: {status['current_weight']}/{status['max_weight']} "
            f"({status['usage_percent']:.1f}% used)"
        )


class AdaptiveRateLimiter(BinanceRateLimiter):
    """
    Rate limiter adaptatif qui ajuste automatiquement en fonction
    des réponses de l'API (headers 429, etc.)
    """

    def __init__(self, safety_margin: float = 0.2):
        super().__init__(safety_margin)
        self._backoff_multiplier = 1.0
        self._consecutive_errors = 0

    def handle_rate_limit_error(self) -> None:
        """Appelé quand on reçoit une erreur 429"""
        with self.lock:
            self.rate_limit_hits += 1
            self._consecutive_errors += 1

            # Augmenter le backoff exponentiellement
            self._backoff_multiplier = min(10.0, self._backoff_multiplier * 2)

            logger.warning(
                f"⚠️  Rate limit error #{self.rate_limit_hits}! "
                f"Backoff multiplier: {self._backoff_multiplier}x"
            )

    def handle_success(self) -> None:
        """Appelé quand une requête réussit"""
        with self.lock:
            if self._consecutive_errors > 0:
                self._consecutive_errors = 0
                # Réduire progressivement le backoff
                self._backoff_multiplier = max(1.0, self._backoff_multiplier * 0.9)

    def _calculate_wait_time(self, endpoint: str) -> float:
        """Calcule le temps d'attente avec backoff adaptatif"""
        base_wait = super()._calculate_wait_time(endpoint)
        return base_wait * self._backoff_multiplier


__all__ = [
    'RateLimitExceeded',
    'BinanceRateLimiter',
    'AdaptiveRateLimiter'
]
