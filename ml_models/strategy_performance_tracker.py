"""
Strategy Performance Tracker - Quantum Trader Pro
Suit la performance de chaque stratégie en temps réel pour informer le Meta-Model
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from utils.logger import setup_logger
from utils.safe_math import safe_divide

class StrategyPerformanceTracker:
    """
    Tracker qui:
    - Enregistre chaque signal et son résultat (PnL)
    - Calcule métriques par stratégie (win rate, avg profit, etc.)
    - Suit la performance récente (fenêtre glissante)
    - Identifie quelles stratégies performent dans quel contexte
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le tracker
        
        Args:
            config: Configuration du bot
        """
        self.config = config
        self.logger = setup_logger('StrategyPerformanceTracker')
        
        # Historique des trades par stratégie
        self.trade_history = {}  # {strategy_name: [trades]}
        
        # Métriques en temps réel
        self.metrics = {}  # {strategy_name: {win_rate, avg_pnl, sharpe, etc.}}
        
        # Fenêtre temporelle pour performance récente
        self.window_days = config.get('ml', {}).get('meta_model', {}).get('performance_window_days', 7)
        
        # Performance par contexte de marché
        self.context_performance = {}  # {strategy: {context: metrics}}
        
        self.logger.info(f"✅ Performance Tracker initialisé (fenêtre: {self.window_days}j)")
    
    def record_trade(
        self,
        strategy_name: str,
        signal_time: datetime,
        entry_price: float,
        exit_price: float,
        exit_time: datetime,
        pnl: float,
        action: str,
        market_context: Optional[Dict] = None
    ):
        """
        Enregistre un trade complété
        
        Args:
            strategy_name: Nom de la stratégie
            signal_time: Timestamp du signal
            entry_price: Prix d'entrée
            exit_price: Prix de sortie
            exit_time: Timestamp de sortie
            pnl: Profit/Loss
            action: BUY ou SELL
            market_context: Contexte marché au moment du signal
        """
        
        # Créer entrée si première fois
        if strategy_name not in self.trade_history:
            self.trade_history[strategy_name] = []
        
        # Enregistrer le trade
        trade = {
            'signal_time': signal_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'pnl': pnl,
            'pnl_pct': safe_divide(exit_price - entry_price, entry_price, default=0.0) * 100 if action == 'BUY' else safe_divide(entry_price - exit_price, entry_price, default=0.0) * 100,
            'action': action,
            'duration_hours': (exit_time - signal_time).total_seconds() / 3600,
            'market_context': market_context or {}
        }
        
        self.trade_history[strategy_name].append(trade)
        
        # Mettre à jour métriques
        self._update_metrics(strategy_name)
        
        # Mettre à jour performance par contexte
        if market_context:
            self._update_context_performance(strategy_name, market_context, pnl > 0)
    
    def _update_metrics(self, strategy_name: str):
        """Recalcule les métriques pour une stratégie"""
        
        trades = self.trade_history.get(strategy_name, [])
        
        if not trades:
            return
        
        # Filtre sur fenêtre récente
        cutoff_time = datetime.now() - timedelta(days=self.window_days)
        recent_trades = [t for t in trades if t['exit_time'] >= cutoff_time]
        
        if not recent_trades:
            recent_trades = trades[-10:]  # Au moins les 10 derniers
        
        # Calculer métriques
        pnls = [t['pnl'] for t in recent_trades]
        winning_trades = [t for t in recent_trades if t['pnl'] > 0]
        losing_trades = [t for t in recent_trades if t['pnl'] <= 0]
        
        total = len(recent_trades)
        wins = len(winning_trades)
        losses = len(losing_trades)

        win_rate = safe_divide(wins, total, default=0.0)

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

        profit_factor = abs(safe_divide(avg_win * wins, avg_loss * losses, default=0.0)) if losses > 0 else 0

        # Sharpe ratio simplifié
        returns = [t['pnl_pct'] for t in recent_trades]
        std_returns = np.std(returns) if len(returns) > 1 else 0
        sharpe = safe_divide(np.mean(returns), std_returns, default=0.0) if std_returns > 1e-10 else 0
        
        # Streak actuel
        current_streak = self._calculate_streak(recent_trades)
        
        self.metrics[strategy_name] = {
            'total_trades': total,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': win_rate,
            'avg_pnl': np.mean(pnls),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'current_streak': current_streak,
            'total_pnl': sum(pnls),
            'last_updated': datetime.now()
        }
    
    def _calculate_streak(self, trades: List[Dict]) -> int:
        """Calcule le streak actuel (positif = winning, négatif = losing)"""
        
        if not trades:
            return 0
        
        # Trier par temps
        sorted_trades = sorted(trades, key=lambda x: x['exit_time'])
        
        # Partir de la fin
        streak = 0
        last_result = None
        
        for trade in reversed(sorted_trades):
            is_win = trade['pnl'] > 0
            
            if last_result is None:
                last_result = is_win
                streak = 1 if is_win else -1
            elif is_win == last_result:
                streak += 1 if is_win else -1
            else:
                break
        
        return streak
    
    def _update_context_performance(self, strategy_name: str, context: Dict, is_win: bool):
        """Met à jour performance par contexte marché"""
        
        if strategy_name not in self.context_performance:
            self.context_performance[strategy_name] = {}
        
        # Clé de contexte (simplifié)
        trend = context.get('trend', 'neutral')
        volatility = 'high' if context.get('volatility', 0) > 0.03 else 'low'
        context_key = f"{trend}_{volatility}"
        
        if context_key not in self.context_performance[strategy_name]:
            self.context_performance[strategy_name][context_key] = {'wins': 0, 'total': 0}
        
        self.context_performance[strategy_name][context_key]['total'] += 1
        if is_win:
            self.context_performance[strategy_name][context_key]['wins'] += 1
    
    def get_strategy_metrics(self, strategy_name: str) -> Optional[Dict]:
        """
        Récupère les métriques d'une stratégie
        
        Args:
            strategy_name: Nom de la stratégie
            
        Returns:
            Dict avec métriques ou None
        """
        return self.metrics.get(strategy_name)
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """Récupère métriques de toutes les stratégies"""
        return self.metrics.copy()
    
    def get_best_strategy_for_context(self, market_context: Dict) -> Optional[str]:
        """
        Trouve la meilleure stratégie pour un contexte donné
        
        Args:
            market_context: Contexte marché actuel
            
        Returns:
            Nom de la meilleure stratégie ou None
        """
        
        trend = market_context.get('trend', 'neutral')
        volatility = 'high' if market_context.get('volatility', 0) > 0.03 else 'low'
        context_key = f"{trend}_{volatility}"
        
        best_strategy = None
        best_win_rate = 0
        
        for strategy_name, contexts in self.context_performance.items():
            if context_key in contexts:
                stats = contexts[context_key]
                if stats['total'] >= 5:  # Minimum de trades
                    win_rate = safe_divide(stats['wins'], stats['total'], default=0.0)
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_strategy = strategy_name
        
        return best_strategy
    
    def get_strategy_ranking(self) -> List[Tuple[str, float]]:
        """
        Classe les stratégies par performance récente
        
        Returns:
            Liste de (strategy_name, score) triée par score décroissant
        """
        
        rankings = []
        
        for strategy_name, metrics in self.metrics.items():
            # Score composite
            score = (
                metrics['win_rate'] * 0.4 +
                (metrics['profit_factor'] / 10) * 0.3 +  # Normaliser profit_factor
                (metrics['sharpe_ratio'] / 5) * 0.3  # Normaliser sharpe
            )
            
            rankings.append((strategy_name, score))
        
        # Trier par score décroissant
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def should_disable_strategy(self, strategy_name: str) -> bool:
        """
        Détermine si une stratégie devrait être désactivée temporairement
        
        Args:
            strategy_name: Nom de la stratégie
            
        Returns:
            True si devrait être désactivée
        """
        
        metrics = self.metrics.get(strategy_name)
        
        if not metrics:
            return False
        
        # Critères de désactivation
        disable_threshold = {
            'min_trades': 10,  # Minimum de trades pour décider
            'max_loss_streak': -5,  # 5 pertes consécutives
            'min_win_rate': 0.3,  # Win rate minimum 30%
            'min_profit_factor': 0.5  # Profit factor minimum
        }
        
        if metrics['total_trades'] < disable_threshold['min_trades']:
            return False  # Pas assez de données
        
        # Vérifier critères
        if metrics['current_streak'] <= disable_threshold['max_loss_streak']:
            self.logger.warning(
                f"⚠️ {strategy_name}: Losing streak de {abs(metrics['current_streak'])}"
            )
            return True
        
        if metrics['win_rate'] < disable_threshold['min_win_rate']:
            self.logger.warning(
                f"⚠️ {strategy_name}: Win rate trop faible ({metrics['win_rate']:.1%})"
            )
            return True
        
        if metrics['profit_factor'] < disable_threshold['min_profit_factor']:
            self.logger.warning(
                f"⚠️ {strategy_name}: Profit factor trop faible ({metrics['profit_factor']:.2f})"
            )
            return True
        
        return False
    
    def get_summary(self) -> Dict:
        """Retourne un résumé du tracker"""
        
        return {
            'tracked_strategies': list(self.trade_history.keys()),
            'total_trades_recorded': sum(len(trades) for trades in self.trade_history.values()),
            'metrics_available': list(self.metrics.keys()),
            'window_days': self.window_days
        }

__all__ = ['StrategyPerformanceTracker']
