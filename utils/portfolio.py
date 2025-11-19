"""
Portfolio Performance Tracking - Quantum Trader Pro
Suivi et analyse des performances du portefeuille
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from utils.logger import setup_logger
from utils.safe_math import safe_divide

logger = setup_logger('Portfolio')


@dataclass
class PortfolioSnapshot:
    """Snapshot du portefeuille à un moment donné"""
    timestamp: datetime
    total_value: float
    available_balance: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    daily_pnl_percent: float
    positions_count: int


@dataclass
class PerformanceMetrics:
    """Métriques de performance calculées"""
    # Rendements
    total_return: float = 0.0
    total_return_percent: float = 0.0
    daily_return_avg: float = 0.0
    weekly_return: float = 0.0
    monthly_return: float = 0.0

    # Risque
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    current_drawdown: float = 0.0

    # Trading
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0  # en heures
    trades_per_day: float = 0.0
    total_trades: int = 0

    # Capital
    initial_capital: float = 0.0
    current_capital: float = 0.0
    peak_capital: float = 0.0
    lowest_capital: float = 0.0


class PortfolioTracker:
    """
    Gestionnaire de suivi de performance du portefeuille.

    Features:
    - Snapshots périodiques de l'état du portefeuille
    - Calcul de métriques avancées (Sharpe, Sortino, Drawdown)
    - Historique des performances
    - Export des données
    """

    def __init__(
        self,
        initial_capital: float,
        data_dir: str = "data/portfolio",
        snapshot_interval: int = 3600  # 1 heure par défaut
    ):
        """
        Initialise le tracker de portefeuille

        Args:
            initial_capital: Capital initial
            data_dir: Répertoire pour les données
            snapshot_interval: Intervalle entre snapshots (secondes)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.lowest_capital = initial_capital

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_interval = snapshot_interval
        self.snapshots: List[PortfolioSnapshot] = []
        self.trade_history: List[Dict[str, Any]] = []

        self._last_snapshot_time = datetime.now()
        self._daily_start_value = initial_capital

        # Charger historique existant
        self._load_history()

        logger.info(f"PortfolioTracker initialisé avec {initial_capital} USDT")

    def record_snapshot(
        self,
        available_balance: float,
        positions: List[Dict[str, Any]],
        current_prices: Dict[str, float]
    ) -> PortfolioSnapshot:
        """
        Enregistre un snapshot de l'état du portefeuille

        Args:
            available_balance: Solde disponible
            positions: Liste des positions ouvertes
            current_prices: Prix actuels des assets

        Returns:
            Snapshot créé
        """
        # Calculer valeur des positions
        positions_value = 0.0
        unrealized_pnl = 0.0

        for pos in positions:
            symbol = pos.get('symbol', '')
            entry_price = pos.get('entry_price', 0)
            quantity = pos.get('quantity', 0)
            side = pos.get('side', 'long')

            current_price = current_prices.get(symbol, entry_price)

            position_value = current_price * quantity
            positions_value += position_value

            if side == 'long':
                unrealized_pnl += (current_price - entry_price) * quantity
            else:
                unrealized_pnl += (entry_price - current_price) * quantity

        total_value = available_balance + positions_value
        realized_pnl = total_value - self.initial_capital - unrealized_pnl

        # Calculer P&L journalier
        daily_pnl = total_value - self._daily_start_value
        daily_pnl_percent = safe_divide(daily_pnl, self._daily_start_value, default=0.0) * 100

        # Mettre à jour pics
        self.current_capital = total_value
        if total_value > self.peak_capital:
            self.peak_capital = total_value
        if total_value < self.lowest_capital:
            self.lowest_capital = total_value

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=total_value,
            available_balance=available_balance,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            daily_pnl=daily_pnl,
            daily_pnl_percent=daily_pnl_percent,
            positions_count=len(positions)
        )

        self.snapshots.append(snapshot)
        self._last_snapshot_time = datetime.now()

        # Sauvegarder périodiquement
        if len(self.snapshots) % 10 == 0:
            self._save_history()

        logger.debug(f"Snapshot: {total_value:.2f} USDT, P&L: {unrealized_pnl:.2f}")

        return snapshot

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        Enregistre un trade complété

        Args:
            trade: Détails du trade
        """
        trade['recorded_at'] = datetime.now().isoformat()
        self.trade_history.append(trade)

        logger.debug(f"Trade enregistré: {trade.get('symbol')} {trade.get('side')}")

    def should_snapshot(self) -> bool:
        """Vérifie si un snapshot est nécessaire"""
        elapsed = (datetime.now() - self._last_snapshot_time).total_seconds()
        return elapsed >= self.snapshot_interval

    def reset_daily_tracking(self) -> None:
        """Réinitialise le suivi journalier (à appeler à minuit)"""
        self._daily_start_value = self.current_capital
        logger.info(f"Tracking journalier reset à {self.current_capital:.2f} USDT")

    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calcule les métriques de performance

        Returns:
            Métriques calculées
        """
        metrics = PerformanceMetrics()

        if not self.snapshots:
            metrics.initial_capital = self.initial_capital
            metrics.current_capital = self.current_capital
            return metrics

        # Capital
        metrics.initial_capital = self.initial_capital
        metrics.current_capital = self.current_capital
        metrics.peak_capital = self.peak_capital
        metrics.lowest_capital = self.lowest_capital

        # Rendements
        metrics.total_return = self.current_capital - self.initial_capital
        metrics.total_return_percent = safe_divide(metrics.total_return, self.initial_capital, default=0.0) * 100

        # Calculer returns journaliers
        values = [s.total_value for s in self.snapshots]
        if len(values) > 1:
            # Protection contre division par zéro dans np.diff
            values_arr = np.array(values[:-1])
            values_arr = np.where(values_arr == 0, 1e-10, values_arr)  # Éviter division par zéro
            returns = np.diff(values) / values_arr

            metrics.daily_return_avg = float(np.mean(returns) * 100)
            metrics.volatility = float(np.std(returns) * np.sqrt(365) * 100)  # Annualisé

            # Sharpe Ratio (assume risk-free rate = 0)
            std_returns = np.std(returns)
            if std_returns > 1e-10:
                metrics.sharpe_ratio = float(safe_divide(np.mean(returns), std_returns, default=0.0) * np.sqrt(365))

            # Sortino Ratio
            downside_returns = returns[returns < 0]
            std_downside = np.std(downside_returns) if len(downside_returns) > 0 else 0
            if len(downside_returns) > 0 and std_downside > 1e-10:
                metrics.sortino_ratio = float(safe_divide(np.mean(returns), std_downside, default=0.0) * np.sqrt(365))

        # Drawdown
        peak = self.initial_capital
        max_dd = 0
        current_dd = 0

        for snapshot in self.snapshots:
            if snapshot.total_value > peak:
                peak = snapshot.total_value
            dd = peak - snapshot.total_value
            if dd > max_dd:
                max_dd = dd
            current_dd = dd

        metrics.max_drawdown = max_dd
        metrics.max_drawdown_percent = safe_divide(max_dd, self.peak_capital, default=0.0) * 100
        metrics.current_drawdown = current_dd

        # Trading stats
        if self.trade_history:
            metrics.total_trades = len(self.trade_history)

            # Win rate
            winning = [t for t in self.trade_history if t.get('pnl', 0) > 0]
            metrics.win_rate = safe_divide(len(winning), len(self.trade_history), default=0.0) * 100

            # Profit factor
            total_wins = sum(t.get('pnl', 0) for t in winning)
            losing = [t for t in self.trade_history if t.get('pnl', 0) <= 0]
            total_losses = abs(sum(t.get('pnl', 0) for t in losing))

            if total_losses > 0:
                metrics.profit_factor = safe_divide(total_wins, total_losses, default=0.0)
            else:
                metrics.profit_factor = float('inf') if total_wins > 0 else 0

            # Durée moyenne des trades
            durations = []
            for trade in self.trade_history:
                if 'entry_time' in trade and 'exit_time' in trade:
                    try:
                        entry = datetime.fromisoformat(trade['entry_time'])
                        exit_time = datetime.fromisoformat(trade['exit_time'])
                        duration = (exit_time - entry).total_seconds() / 3600  # heures
                        durations.append(duration)
                    except Exception as e:
                        logger.debug(f"Erreur parsing dates trade: {e}")
                        pass

            if durations:
                metrics.avg_trade_duration = safe_divide(sum(durations), len(durations), default=0.0)

            # Trades par jour
            if self.snapshots:
                first_snapshot = self.snapshots[0].timestamp
                days = max(1, (datetime.now() - first_snapshot).days)
                metrics.trades_per_day = safe_divide(len(self.trade_history), days, default=0.0)

        # Rendements périodiques
        if len(self.snapshots) >= 7:
            weekly_start = self.snapshots[-7].total_value
            metrics.weekly_return = safe_divide(self.current_capital - weekly_start, weekly_start, default=0.0) * 100

        if len(self.snapshots) >= 30:
            monthly_start = self.snapshots[-30].total_value
            metrics.monthly_return = safe_divide(self.current_capital - monthly_start, monthly_start, default=0.0) * 100

        return metrics

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """
        Retourne la courbe d'equity

        Returns:
            Liste de points (timestamp, value)
        """
        return [
            {
                'timestamp': s.timestamp.isoformat(),
                'value': s.total_value,
                'pnl': s.unrealized_pnl + s.realized_pnl
            }
            for s in self.snapshots
        ]

    def get_daily_summary(self) -> Dict[str, Any]:
        """
        Résumé de la journée actuelle

        Returns:
            Statistiques du jour
        """
        if not self.snapshots:
            return {
                'start_value': self._daily_start_value,
                'current_value': self.current_capital,
                'pnl': 0,
                'pnl_percent': 0,
                'trades': 0
            }

        latest = self.snapshots[-1]

        # Compter trades du jour
        today = datetime.now().date()
        trades_today = sum(
            1 for t in self.trade_history
            if datetime.fromisoformat(t.get('recorded_at', '')).date() == today
        ) if self.trade_history else 0

        return {
            'start_value': self._daily_start_value,
            'current_value': latest.total_value,
            'pnl': latest.daily_pnl,
            'pnl_percent': latest.daily_pnl_percent,
            'trades': trades_today,
            'positions': latest.positions_count
        }

    def generate_report(self) -> str:
        """
        Génère un rapport de performance texte

        Returns:
            Rapport formaté
        """
        metrics = self.calculate_metrics()
        daily = self.get_daily_summary()

        report = []
        report.append("=" * 60)
        report.append("RAPPORT DE PERFORMANCE - Quantum Trader Pro")
        report.append("=" * 60)

        report.append(f"\nCAPITAL:")
        report.append(f"  Initial: {metrics.initial_capital:.2f} USDT")
        report.append(f"  Actuel: {metrics.current_capital:.2f} USDT")
        report.append(f"  Pic: {metrics.peak_capital:.2f} USDT")
        report.append(f"  Retour total: {metrics.total_return:.2f} USDT ({metrics.total_return_percent:.2f}%)")

        report.append(f"\nPERFORMANCE:")
        report.append(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        report.append(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
        report.append(f"  Volatilité (ann.): {metrics.volatility:.2f}%")
        report.append(f"  Max Drawdown: {metrics.max_drawdown:.2f} USDT ({metrics.max_drawdown_percent:.2f}%)")
        report.append(f"  Drawdown actuel: {metrics.current_drawdown:.2f} USDT")

        report.append(f"\nTRADING:")
        report.append(f"  Total trades: {metrics.total_trades}")
        report.append(f"  Win Rate: {metrics.win_rate:.1f}%")
        report.append(f"  Profit Factor: {metrics.profit_factor:.2f}")
        report.append(f"  Trades/jour: {metrics.trades_per_day:.1f}")
        report.append(f"  Durée moyenne: {metrics.avg_trade_duration:.1f}h")

        report.append(f"\nRENDEMENTS:")
        report.append(f"  Aujourd'hui: {daily['pnl']:.2f} USDT ({daily['pnl_percent']:.2f}%)")
        report.append(f"  Semaine: {metrics.weekly_return:.2f}%")
        report.append(f"  Mois: {metrics.monthly_return:.2f}%")

        report.append("=" * 60)

        return "\n".join(report)

    def _save_history(self) -> None:
        """Sauvegarde l'historique sur disque"""
        try:
            # Sauvegarder snapshots
            snapshots_file = self.data_dir / "snapshots.json"
            snapshots_data = [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'total_value': s.total_value,
                    'available_balance': s.available_balance,
                    'positions_value': s.positions_value,
                    'unrealized_pnl': s.unrealized_pnl,
                    'realized_pnl': s.realized_pnl,
                    'daily_pnl': s.daily_pnl,
                    'daily_pnl_percent': s.daily_pnl_percent,
                    'positions_count': s.positions_count
                }
                for s in self.snapshots[-1000:]  # Garder 1000 derniers
            ]

            with open(snapshots_file, 'w') as f:
                json.dump(snapshots_data, f, indent=2)

            # Sauvegarder trades
            trades_file = self.data_dir / "trades.json"
            with open(trades_file, 'w') as f:
                json.dump(self.trade_history[-10000:], f, indent=2, default=str)

            # Sauvegarder métriques
            metrics_file = self.data_dir / "metrics.json"
            metrics = self.calculate_metrics()
            with open(metrics_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2)

            logger.debug("Historique portfolio sauvegardé")

        except Exception as e:
            logger.error(f"Erreur sauvegarde historique: {e}")

    def _load_history(self) -> None:
        """Charge l'historique depuis le disque"""
        try:
            # Charger snapshots
            snapshots_file = self.data_dir / "snapshots.json"
            if snapshots_file.exists():
                with open(snapshots_file, 'r') as f:
                    snapshots_data = json.load(f)

                for data in snapshots_data:
                    snapshot = PortfolioSnapshot(
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        total_value=data['total_value'],
                        available_balance=data['available_balance'],
                        positions_value=data['positions_value'],
                        unrealized_pnl=data['unrealized_pnl'],
                        realized_pnl=data['realized_pnl'],
                        daily_pnl=data['daily_pnl'],
                        daily_pnl_percent=data['daily_pnl_percent'],
                        positions_count=data['positions_count']
                    )
                    self.snapshots.append(snapshot)

                if self.snapshots:
                    self.current_capital = self.snapshots[-1].total_value
                    self.peak_capital = max(s.total_value for s in self.snapshots)
                    self.lowest_capital = min(s.total_value for s in self.snapshots)

                logger.info(f"Chargé {len(self.snapshots)} snapshots")

            # Charger trades
            trades_file = self.data_dir / "trades.json"
            if trades_file.exists():
                with open(trades_file, 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"Chargé {len(self.trade_history)} trades")

        except Exception as e:
            logger.warning(f"Erreur chargement historique: {e}")

    def export_to_csv(self, filename: str = "portfolio_history.csv") -> None:
        """
        Exporte l'historique en CSV

        Args:
            filename: Nom du fichier
        """
        try:
            import csv

            filepath = self.data_dir / filename

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    'timestamp', 'total_value', 'available_balance',
                    'positions_value', 'unrealized_pnl', 'realized_pnl',
                    'daily_pnl', 'daily_pnl_percent', 'positions_count'
                ])

                # Data
                for s in self.snapshots:
                    writer.writerow([
                        s.timestamp.isoformat(),
                        s.total_value,
                        s.available_balance,
                        s.positions_value,
                        s.unrealized_pnl,
                        s.realized_pnl,
                        s.daily_pnl,
                        s.daily_pnl_percent,
                        s.positions_count
                    ])

            logger.info(f"Exporté vers {filepath}")

        except Exception as e:
            logger.error(f"Erreur export CSV: {e}")


__all__ = [
    'PortfolioTracker',
    'PortfolioSnapshot',
    'PerformanceMetrics'
]
