"""
Position Sizer - Quantum Trader Pro
Kelly Criterion et autres méthodes de sizing
"""

import numpy as np
from typing import Dict, Optional
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
from utils.safe_math import safe_divide

class PositionSizer:
    """Calcul de la taille des positions"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('PositionSizer')

        # Config - accès sécurisé
        sizing_cfg = get_nested_config(config, 'capital', 'position_sizing', default={})
        self.method = sizing_cfg.get('method', 'fixed_percent')
        self.kelly_fraction = sizing_cfg.get('kelly_fraction', 0.25)
        self.min_size = sizing_cfg.get('min_position_size', 0.001)
        self.max_size = sizing_cfg.get('max_position_size', 1.0)
        self.decimals = sizing_cfg.get('decimals', 4)

        # Risk config - accès sécurisé
        risk_cfg = get_nested_config(config, 'risk', default={})
        self.max_risk_pct = risk_cfg.get('max_risk_per_trade_percent', 2.0)

        # Performance tracking pour Kelly
        self.win_rate = 0.5  # Default
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.num_trades = 0

    def calculate_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        confidence: float = 0.7,
        atr: Optional[float] = None
    ) -> float:
        """
        Calcule taille de position

        Args:
            account_balance: Balance du compte
            entry_price: Prix d'entrée
            stop_loss: Prix stop loss
            confidence: Confiance du signal (0-1)
            atr: ATR pour volatility-based

        Returns:
            Taille de position en unités de base asset
        """
        # Validation des prix
        if not self._validate_inputs(account_balance, entry_price, stop_loss, confidence):
            self.logger.error("Inputs invalides pour position sizing")
            return self.min_size  # Retourner taille minimale par sécurité

        if self.method == 'kelly_criterion':
            size = self._kelly_criterion(
                account_balance, entry_price, stop_loss, confidence
            )

        elif self.method == 'fixed_percent':
            size = self._fixed_percent(account_balance, entry_price, stop_loss)

        elif self.method == 'volatility_based' and atr:
            size = self._volatility_based(
                account_balance, entry_price, atr, confidence
            )

        else:
            # Fallback to fixed percent
            size = self._fixed_percent(account_balance, entry_price, stop_loss)

        # Appliquer limites
        size = max(self.min_size, min(size, self.max_size))

        # Arrondir
        size = round(size, self.decimals)

        return size

    def _kelly_criterion(
        self,
        balance: float,
        entry: float,
        stop_loss: float,
        confidence: float
    ) -> float:
        """
        Kelly Criterion: f = (bp - q) / b
        où:
        - b = ratio win/loss
        - p = win rate
        - q = loss rate (1-p)
        """
        # Utiliser confidence comme proxy pour win rate
        # Si peu de trades, utiliser confidence
        if self.num_trades < 10:
            win_rate = confidence
        else:
            # Sinon utiliser win rate historique ajusté par confidence
            win_rate = (self.win_rate * 0.7) + (confidence * 0.3)

        loss_rate = 1 - win_rate

        # Ratio win/loss (avg_win / avg_loss)
        if self.avg_loss == 0:
            # Default conservateur
            win_loss_ratio = 1.5
        else:
            win_loss_ratio = abs(safe_divide(self.avg_win, self.avg_loss, default=1.5))

        # Kelly formula
        kelly_pct = safe_divide(win_rate * win_loss_ratio - loss_rate, win_loss_ratio, default=0.0)

        # Fraction de Kelly (conservative)
        kelly_pct = max(0, kelly_pct) * self.kelly_fraction

        # Risk amount
        risk_distance = safe_divide(abs(entry - stop_loss), entry, default=0.0)

        # Protection division par zéro
        if risk_distance < 0.0001:  # Moins de 0.01% de distance
            self.logger.warning(f"Risk distance trop faible: {risk_distance:.6f}")
            return self.min_size

        # Capital à risquer
        capital_at_risk = balance * (self.max_risk_pct / 100) * kelly_pct

        # Taille position
        size = safe_divide(capital_at_risk, entry * risk_distance, default=self.min_size)

        self.logger.debug(
            f"Kelly: wr={win_rate:.2f} wl_ratio={win_loss_ratio:.2f} "
            f"kelly%={kelly_pct:.2%} size={size:.4f}"
        )

        return size

    def _fixed_percent(
        self,
        balance: float,
        entry: float,
        stop_loss: float
    ) -> float:
        """Fixed percent risk per trade"""
        # Risk distance
        risk_distance = safe_divide(abs(entry - stop_loss), entry, default=0.0)

        # Protection division par zéro
        if risk_distance < 0.0001:  # Moins de 0.01% de distance
            self.logger.warning(f"Risk distance trop faible: {risk_distance:.6f}")
            return self.min_size

        # Capital à risquer
        capital_at_risk = balance * (self.max_risk_pct / 100)

        # Taille
        size = safe_divide(capital_at_risk, entry * risk_distance, default=self.min_size)

        return size

    def _volatility_based(
        self,
        balance: float,
        entry: float,
        atr: float,
        confidence: float
    ) -> float:
        """Volatility-based sizing avec ATR"""
        # Protection si ATR invalide
        if atr <= 0 or entry <= 0:
            self.logger.warning(f"ATR ou Entry invalide: ATR={atr}, Entry={entry}")
            return self.min_size

        # Plus la volatilité est haute, plus la position est petite
        volatility_factor = safe_divide(atr, entry, default=0.0)

        # Ajuster par volatilité (inverse)
        volatility_adj = safe_divide(1.0, 1.0 + volatility_factor * 10, default=0.1)

        # Base size
        base_capital = balance * (self.max_risk_pct / 100) * confidence

        # Ajuster par volatilité
        adjusted_capital = base_capital * volatility_adj

        size = safe_divide(adjusted_capital, entry, default=self.min_size)

        return size

    def update_performance(
        self,
        win: bool,
        pnl: float,
        entry: float,
        exit_price: float
    ) -> None:
        """
        Met à jour les stats pour Kelly

        Args:
            win: True si gagnant
            pnl: Profit/Loss en $
            entry: Prix d'entrée
            exit_price: Prix de sortie
        """
        self.num_trades += 1

        # Calcul return
        trade_return = safe_divide(exit_price - entry, entry, default=0.0)

        if win:
            # Moyenne mobile des wins
            if self.avg_win == 0:
                self.avg_win = abs(trade_return)
            else:
                self.avg_win = (self.avg_win * 0.9) + (abs(trade_return) * 0.1)
        else:
            # Moyenne mobile des losses
            if self.avg_loss == 0:
                self.avg_loss = abs(trade_return)
            else:
                self.avg_loss = (self.avg_loss * 0.9) + (abs(trade_return) * 0.1)

        # Update win rate (moyenne mobile)
        if self.win_rate == 0.5 and self.num_trades == 1:
            self.win_rate = 1.0 if win else 0.0
        else:
            self.win_rate = (self.win_rate * 0.95) + ((1.0 if win else 0.0) * 0.05)

        self.logger.debug(
            f"Performance updated: wr={self.win_rate:.2%} "
            f"avg_win={self.avg_win:.2%} avg_loss={self.avg_loss:.2%}"
        )

    def _validate_inputs(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        confidence: float
    ) -> bool:
        """
        Valide les inputs avant calcul de position

        Returns:
            True si inputs valides
        """
        # Validation balance
        if balance <= 0:
            self.logger.error(f"Balance invalide: {balance}")
            return False

        # Validation prix
        if entry_price <= 0 or entry_price > 1_000_000:
            self.logger.error(f"Prix d'entrée invalide: {entry_price}")
            return False

        if stop_loss <= 0 or stop_loss > 1_000_000:
            self.logger.error(f"Stop loss invalide: {stop_loss}")
            return False

        # Validation confidence
        if not 0 <= confidence <= 1:
            self.logger.error(f"Confidence invalide: {confidence}")
            return False

        # Vérifier que SL != Entry (évite division par zéro)
        if abs(entry_price - stop_loss) < 0.00000001:
            self.logger.error(f"Stop loss égal au prix d'entrée: {entry_price} == {stop_loss}")
            return False

        return True

    def get_stats(self) -> Dict:
        """Retourne stats de performance"""
        return {
            'method': self.method,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'win_loss_ratio': safe_divide(self.avg_win, self.avg_loss, default=0.0)
        }

__all__ = ['PositionSizer']
