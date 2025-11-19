"""
Circuit Breaker - Quantum Trader Pro
Protection contre drawdown excessif et conditions dangereuses
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from threading import RLock
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
from utils.safe_math import safe_divide

@dataclass
class CircuitBreakerState:
    """État du circuit breaker"""
    is_active: bool = False
    trigger_reason: Optional[str] = None
    triggered_at: Optional[datetime] = None
    resume_at: Optional[datetime] = None
    consecutive_losses: int = 0
    daily_loss: float = 0.0
    max_drawdown_reached: float = 0.0

class CircuitBreaker:
    """Circuit breaker pour protection trading"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('CircuitBreaker')
        self._lock = RLock()  # Thread safety

        # Config circuit breakers - accès sécurisé
        cb_cfg = get_nested_config(config, 'risk', 'circuit_breakers', default={})
        self.enabled = cb_cfg.get('enabled', True)
        self.max_drawdown_pct = cb_cfg.get('max_drawdown_percent', 8.0)
        self.consecutive_losses_threshold = cb_cfg.get('consecutive_losses_threshold', 5)
        self.pause_duration_minutes = cb_cfg.get('pause_duration_minutes', 60)
        self.volatility_threshold = cb_cfg.get('volatility_threshold', 3.0)

        # State
        self.state = CircuitBreakerState()

        # Tracking
        self.peak_balance = 0.0
        self.daily_start_balance = 0.0
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()

        # Volatility tracking
        self.normal_volatility = 0.02  # 2% normal
        self.current_volatility = 0.02

    def initialize(self, initial_balance: float):
        """Initialise avec balance"""
        self.peak_balance = initial_balance
        self.daily_start_balance = initial_balance
        self.initialized = True
        self.logger.info(f"🔒 Circuit Breaker initialisé: balance=${initial_balance:.2f}")

    def check(
        self,
        current_balance: float,
        last_trade_won: Optional[bool] = None,
        current_volatility: Optional[float] = None
    ) -> bool:
        """
        Vérifie si circuit breaker doit s'activer

        Args:
            current_balance: Balance actuelle
            last_trade_won: True/False si dernier trade gagné/perdu
            current_volatility: Volatilité actuelle du marché

        Returns:
            True si trading autorisé, False si breaker actif
        """
        if not self.enabled:
            return True

        # Auto-initialisation si pas fait
        if self.peak_balance == 0 and current_balance > 0:
            self.initialize(current_balance)

        # Check si déjà actif
        if self.state.is_active:
            return self._check_resume()

        # Reset quotidien
        self._check_daily_reset(current_balance)

        # Update peak
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # Update volatilité
        if current_volatility is not None:
            self.current_volatility = current_volatility

        # Check 1: Max Drawdown
        if self._check_max_drawdown(current_balance):
            return False

        # Check 2: Daily Loss
        if self._check_daily_loss(current_balance):
            return False

        # Check 3: Consecutive Losses
        if last_trade_won is not None:
            if self._check_consecutive_losses(last_trade_won):
                return False

        # Check 4: Volatility Spike
        if self._check_volatility_spike():
            return False

        return True

    def _check_max_drawdown(self, current_balance: float) -> bool:
        """Vérifie drawdown maximum"""
        if self.peak_balance == 0:
            self.peak_balance = current_balance

        drawdown = safe_divide(self.peak_balance - current_balance, self.peak_balance, default=0.0) * 100

        if drawdown >= self.max_drawdown_pct:
            self._activate(
                f"Max Drawdown {drawdown:.2f}% >= {self.max_drawdown_pct}%",
                pause_minutes=self.pause_duration_minutes * 2  # Double pause
            )
            self.state.max_drawdown_reached = drawdown
            return True

        return False

    def _check_daily_loss(self, current_balance: float) -> bool:
        """Vérifie perte journalière"""
        if self.daily_start_balance == 0:
            self.daily_start_balance = current_balance

        daily_loss = safe_divide(self.daily_start_balance - current_balance, self.daily_start_balance, default=0.0) * 100
        self.state.daily_loss = daily_loss

        max_daily_loss = get_nested_config(self.config, 'risk', 'max_daily_loss_percent', default=5.0)

        if daily_loss >= max_daily_loss:
            self._activate(
                f"Daily Loss {daily_loss:.2f}% >= {max_daily_loss}%",
                pause_minutes=60 * 24  # Pause jusqu'au lendemain
            )
            return True

        return False

    def _check_consecutive_losses(self, last_won: bool) -> bool:
        """Vérifie pertes consécutives"""
        if last_won:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1

        if self.state.consecutive_losses >= self.consecutive_losses_threshold:
            self._activate(
                f"Consecutive Losses: {self.state.consecutive_losses}",
                pause_minutes=self.pause_duration_minutes
            )
            return True

        return False

    def _check_volatility_spike(self) -> bool:
        """Vérifie spike de volatilité"""
        if self.current_volatility > self.normal_volatility * self.volatility_threshold:
            self._activate(
                f"Volatility Spike: {self.current_volatility:.2%} "
                f"(normal: {self.normal_volatility:.2%})",
                pause_minutes=30
            )
            return True

        return False

    def activate(self, reason: str, pause_minutes: Optional[int] = None) -> None:
        """
        Active le circuit breaker (méthode publique)

        Args:
            reason: Raison de l'activation
            pause_minutes: Durée de pause en minutes
        """
        if pause_minutes is None:
            pause_minutes = self.pause_duration_minutes

        self._activate(reason, pause_minutes)

    def _activate(self, reason: str, pause_minutes: int) -> None:
        """Active le circuit breaker"""
        self.state.is_active = True
        self.state.trigger_reason = reason
        self.state.triggered_at = datetime.now()
        self.state.resume_at = datetime.now() + timedelta(minutes=pause_minutes)

        self.logger.warning(
            f"🚨 CIRCUIT BREAKER ACTIVÉ!\n"
            f"   Raison: {reason}\n"
            f"   Pause jusqu'à: {self.state.resume_at:%H:%M:%S}\n"
            f"   Duration: {pause_minutes} minutes"
        )

    def _check_resume(self) -> bool:
        """Vérifie si peut reprendre trading"""
        if not self.state.is_active:
            return True

        if datetime.now() >= self.state.resume_at:
            self.logger.info(
                f"✅ Circuit Breaker désactivé - Reprise du trading\n"
                f"   Pause duration: {(datetime.now() - self.state.triggered_at).seconds // 60} min"
            )
            self._reset()
            return True

        remaining = (self.state.resume_at - datetime.now()).seconds // 60
        if remaining % 10 == 0:  # Log toutes les 10 minutes
            self.logger.info(f"⏸️  Trading pausé - Reprise dans {remaining} min")

        return False

    def _check_daily_reset(self, current_balance: float):
        """Reset quotidien"""
        today = datetime.now().date()

        if today > self.last_reset_date:
            self.logger.info(f"📅 Reset quotidien du circuit breaker")
            self.daily_start_balance = current_balance
            self.daily_trade_count = 0
            self.state.daily_loss = 0.0
            self.state.consecutive_losses = 0
            self.last_reset_date = today

    def _reset(self):
        """Reset l'état du breaker"""
        self.state = CircuitBreakerState()

    def force_activate(self, reason: str, pause_minutes: int = 60):
        """Active manuellement le breaker"""
        self._activate(reason, pause_minutes)
        self.logger.warning(f"⚠️  Circuit Breaker activé manuellement: {reason}")

    def force_deactivate(self):
        """Désactive manuellement"""
        if self.state.is_active:
            self.logger.warning("⚠️  Circuit Breaker désactivé manuellement!")
            self._reset()

    def get_status(self) -> Dict:
        """Retourne statut actuel"""
        return {
            'enabled': self.enabled,
            'is_active': self.state.is_active,
            'trigger_reason': self.state.trigger_reason,
            'triggered_at': self.state.triggered_at,
            'resume_at': self.state.resume_at,
            'consecutive_losses': self.state.consecutive_losses,
            'daily_loss_pct': self.state.daily_loss,
            'max_drawdown_reached': self.state.max_drawdown_reached,
            'current_volatility': self.current_volatility,
            'peak_balance': self.peak_balance,
        }


    def record_trade(self):
        """Enregistre un trade (utilisé pour tracking)"""
        self.daily_trade_count += 1
        self.logger.debug(f"Trade enregistré - Total journalier: {self.daily_trade_count}")

    def record_loss(self, loss_amount: float):
        """
        Enregistre une perte

        Args:
            loss_amount: Montant de la perte (valeur positive)
        """
        self.state.consecutive_losses += 1
        self.logger.warning(
            f"❌ Perte enregistrée: ${loss_amount:.2f} "
            f"(Pertes consécutives: {self.state.consecutive_losses})"
        )

    def record_win(self, win_amount: float):
        """
        Enregistre un gain

        Args:
            win_amount: Montant du gain
        """
        self.state.consecutive_losses = 0
        self.logger.info(f"✅ Gain enregistré: ${win_amount:.2f}")

__all__ = ['CircuitBreaker', 'CircuitBreakerState']
