"""
ML Meta-Model - Quantum Trader Pro
Sélectionne dynamiquement les meilleures stratégies selon le contexte marché
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from utils.logger import setup_logger
from utils.calculations import timeframe_to_minutes
from utils.safety import safe_last_value, ensure_minimum_data
from utils.safe_math import safe_divide
from ml_models.strategy_performance_tracker import StrategyPerformanceTracker

class MLMetaModel:
    """
    Meta-Model qui:
    1. Analyse le contexte marché actuel
    2. Évalue les performances récentes de chaque stratégie
    3. Sélectionne dynamiquement quelle(s) stratégie(s) utiliser
    4. Ajuste les poids/confidences des signaux

    Approches supportées:
    - winner_takes_all: Une seule stratégie (la meilleure)
    - weighted_ensemble: Combiner avec poids dynamiques
    - context_adaptive: Choisir selon contexte marché
    """

    # Mapping stratégie → conditions optimales
    STRATEGY_CONTEXT_MAP = {
        'grid': {
            'optimal_trend': 'neutral',  # Range trading
            'optimal_volatility': (0.005, 0.025),  # Volatilité moyenne
            'description': 'Range/Sideways market'
        },
        'ichimoku': {
            'optimal_trend': 'trending',  # Bull ou Bear
            'optimal_volatility': (0.01, 0.05),  # Volatilité moyenne-haute
            'description': 'Trending market'
        },
        'dca': {
            'optimal_trend': 'any',  # Fonctionne partout
            'optimal_volatility': (0.0, 1.0),  # Toute volatilité
            'description': 'All market conditions'
        },
        'ml': {
            'optimal_trend': 'any',  # Adaptatif
            'optimal_volatility': (0.01, 0.04),  # Volatilité modérée
            'description': 'Data-driven adaptation'
        }
    }

    def __init__(self, config: Dict, performance_tracker: StrategyPerformanceTracker):
        """
        Initialise le meta-model
        
        Args:
            config: Configuration du bot
            performance_tracker: Tracker de performance
        """
        self.config = config
        self.performance_tracker = performance_tracker
        self.logger = setup_logger('MLMetaModel')
        
        # Configuration
        meta_config = config.get('ml', {}).get('meta_model', {})
        
        self.selection_mode = meta_config.get('selection_mode', 'weighted_ensemble')
        # winner_takes_all, weighted_ensemble, context_adaptive
        
        self.min_confidence_threshold = meta_config.get('min_confidence', 0.6)
        self.context_weight = meta_config.get('context_weight', 0.5)  # Poids du contexte vs performance
        
        # Historique de décisions
        self.decision_history = []
        
        self.logger.info(f"✅ Meta-Model initialisé (mode: {self.selection_mode})")
    
    def analyze_market_context(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyse le contexte marché actuel
        
        Args:
            data: Dict timeframe → DataFrame
            
        Returns:
            Dict avec features du contexte marché
        """
        
        # Utiliser la timeframe principale (plus petite)
        main_tf = min(data.keys(), key=lambda x: timeframe_to_minutes(x))
        df = data[main_tf]
        
        if len(df) < 50:
            return {'trend': 'neutral', 'volatility': 0.01, 'volume': 'normal'}

        # Calculer features - besoin d'au moins 20 rows pour les calculs
        recent = df.tail(20)
        if not ensure_minimum_data(recent, 20, "meta_model_context"):
            return {'trend': 'neutral', 'volatility': 0.01, 'volume': 'normal'}

        # Trend - avec vérification NaN
        sma_short_series = recent['close'].rolling(5).mean()
        sma_long_series = recent['close'].rolling(20).mean()

        sma_short = safe_last_value(sma_short_series, default=0.0)
        sma_long = safe_last_value(sma_long_series, default=0.0)

        if sma_long == 0.0 or pd.isna(sma_short) or pd.isna(sma_long):
            trend = 'neutral'
        elif sma_short > sma_long * 1.01:
            trend = 'bullish'
        elif sma_short < sma_long * 0.99:
            trend = 'bearish'
        else:
            trend = 'neutral'

        # Volatility (ATR normalized)
        high_low = recent['high'] - recent['low']
        atr = high_low.mean()
        current_price = safe_last_value(recent['close'], default=0.0)
        if current_price > 0:
            volatility = atr / current_price
        else:
            volatility = 0.01

        # Volume - avec protection
        avg_volume_series = df['volume'].rolling(20).mean()
        avg_volume = safe_last_value(avg_volume_series, default=1.0)
        current_volume = safe_last_value(recent['volume'], default=1.0)

        if avg_volume > 0 and current_volume > avg_volume * 1.5:
            volume_state = 'high'
        elif avg_volume > 0 and current_volume < avg_volume * 0.5:
            volume_state = 'low'
        else:
            volume_state = 'normal'

        # Price momentum - avec vérification des indices
        if len(recent) >= 12:
            close_current = safe_last_value(recent['close'], default=0.0)
            close_12_ago = recent['close'].iloc[-12] if len(recent) >= 12 else close_current
            close_start = recent['close'].iloc[0]

            if close_12_ago > 0:
                price_change_1h = (close_current - close_12_ago) / close_12_ago
            else:
                price_change_1h = 0.0

            if close_start > 0:
                price_change_4h = (close_current - close_start) / close_start
            else:
                price_change_4h = 0.0
        else:
            price_change_1h = 0.0
            price_change_4h = 0.0

        context = {
            'trend': trend,
            'volatility': float(volatility),
            'volume': volume_state,
            'price_momentum_1h': float(price_change_1h),
            'price_momentum_4h': float(price_change_4h),
            'current_price': float(current_price),
            'timestamp': datetime.now()
        }

        return context

    def _calculate_context_score(self, strategy_name: str, context: Dict) -> float:
        """
        Calcule le score de compatibilité stratégie/contexte marché

        Args:
            strategy_name: Nom de la stratégie
            context: Contexte marché actuel

        Returns:
            Score entre 0.0 et 1.0
        """
        # Identifier le type de stratégie
        strategy_type = None
        for key in self.STRATEGY_CONTEXT_MAP.keys():
            if key in strategy_name.lower():
                strategy_type = key
                break

        if not strategy_type:
            # Stratégie inconnue: score neutre
            return 0.5

        strategy_config = self.STRATEGY_CONTEXT_MAP[strategy_type]
        score = 0.0

        # 1. Score de tendance (50% du score)
        optimal_trend = strategy_config['optimal_trend']
        market_trend = context.get('trend', 'neutral')

        if optimal_trend == 'any':
            trend_score = 0.8  # Bon partout
        elif optimal_trend == 'neutral' and market_trend == 'neutral':
            trend_score = 1.0  # Parfait pour range
        elif optimal_trend == 'trending' and market_trend in ('bullish', 'bearish'):
            trend_score = 1.0  # Parfait pour trend
        elif optimal_trend == 'neutral' and market_trend in ('bullish', 'bearish'):
            trend_score = 0.3  # Grid en trend = mauvais
        elif optimal_trend == 'trending' and market_trend == 'neutral':
            trend_score = 0.4  # Ichimoku en range = sous-optimal
        else:
            trend_score = 0.5

        score += trend_score * 0.5

        # 2. Score de volatilité (30% du score)
        vol_min, vol_max = strategy_config['optimal_volatility']
        current_vol = context.get('volatility', 0.01)

        if vol_min <= current_vol <= vol_max:
            vol_score = 1.0  # Dans la zone optimale
        elif current_vol < vol_min:
            # Trop calme
            vol_score = max(0.3, safe_divide(current_vol, vol_min, default=0.3))
        else:
            # Trop volatile
            vol_score = max(0.3, safe_divide(vol_max, current_vol, default=0.3))

        score += vol_score * 0.3

        # 3. Score de momentum (20% du score)
        momentum = abs(context.get('price_momentum_1h', 0))

        if strategy_type == 'grid':
            # Grid préfère faible momentum
            momentum_score = max(0.2, 1.0 - momentum * 10)
        elif strategy_type == 'ichimoku':
            # Ichimoku préfère fort momentum
            momentum_score = min(1.0, 0.5 + momentum * 5)
        else:
            # Autres: score neutre
            momentum_score = 0.7

        score += momentum_score * 0.2

        return min(1.0, max(0.0, score))

    def select_strategies(
        self,
        all_signals: Dict[str, List],
        market_context: Dict
    ) -> Dict[str, float]:
        """
        Sélectionne les stratégies et leurs poids
        
        Args:
            all_signals: Dict strategy_name → signals
            market_context: Contexte marché actuel
            
        Returns:
            Dict strategy_name → weight (0-1)
        """
        
        if not all_signals:
            return {}
        
        # Obtenir métriques de performance
        all_metrics = self.performance_tracker.get_all_metrics()
        
        # Stratégies disponibles
        strategies = list(all_signals.keys())
        
        if self.selection_mode == 'winner_takes_all':
            return self._winner_takes_all(strategies, all_metrics, market_context)
        
        elif self.selection_mode == 'weighted_ensemble':
            return self._weighted_ensemble(strategies, all_metrics, market_context)
        
        elif self.selection_mode == 'context_adaptive':
            return self._context_adaptive(strategies, all_metrics, market_context)
        
        else:
            # Default: equal weights
            return {s: 1.0 / len(strategies) for s in strategies}
    
    def _winner_takes_all(
        self,
        strategies: List[str],
        metrics: Dict[str, Dict],
        context: Dict
    ) -> Dict[str, float]:
        """Sélectionne UNE seule stratégie (la meilleure)"""
        
        best_strategy = None
        best_score = -999
        
        for strategy_name in strategies:
            score = self._calculate_strategy_score(strategy_name, metrics, context)
            
            if score > best_score:
                best_score = score
                best_strategy = strategy_name
        
        if best_strategy:
            self.logger.info(f"🏆 Stratégie sélectionnée: {best_strategy} (score: {best_score:.2f})")
            return {best_strategy: 1.0}
        
        return {}
    
    def _weighted_ensemble(
        self,
        strategies: List[str],
        metrics: Dict[str, Dict],
        context: Dict
    ) -> Dict[str, float]:
        """Combine toutes les stratégies avec poids dynamiques"""
        
        scores = {}
        
        for strategy_name in strategies:
            # Calculer score
            score = self._calculate_strategy_score(strategy_name, metrics, context)
            
            # Appliquer softmax pour normaliser
            scores[strategy_name] = max(0, score)  # Pas de scores négatifs
        
        # Normaliser les poids
        total_score = sum(scores.values())

        if total_score > 0:
            weights = {s: safe_divide(score, total_score, default=0.0) for s, score in scores.items()}
        else:
            # Equal weights si pas de données
            weights = {s: safe_divide(1.0, len(strategies), default=0.0) for s in strategies}
        
        self.logger.info("📊 Poids des stratégies:")
        for strategy, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"   - {strategy}: {weight:.2%}")
        
        return weights
    
    def _context_adaptive(
        self,
        strategies: List[str],
        metrics: Dict[str, Dict],
        context: Dict
    ) -> Dict[str, float]:
        """Sélectionne selon le contexte marché"""
        
        # Vérifier si une stratégie est spécialisée pour ce contexte
        best_for_context = self.performance_tracker.get_best_strategy_for_context(context)
        
        if best_for_context and best_for_context in strategies:
            # Donner plus de poids à la stratégie adaptée
            weights = {s: 0.1 for s in strategies}
            weights[best_for_context] = 0.7

            # Normaliser
            total = sum(weights.values())
            weights = {s: safe_divide(w, total, default=0.0) for s, w in weights.items()}
            
            self.logger.info(f"🎯 Stratégie adaptée au contexte: {best_for_context}")
            return weights
        
        # Sinon, fallback sur weighted_ensemble
        return self._weighted_ensemble(strategies, metrics, context)
    
    def _calculate_strategy_score(
        self,
        strategy_name: str,
        all_metrics: Dict[str, Dict],
        context: Dict
    ) -> float:
        """
        Calcule un score pour une stratégie
        
        Score basé sur:
        - Performance récente (win rate, profit factor, sharpe)
        - Adaptation au contexte marché
        - Streak actuel
        """
        
        # Métriques de la stratégie
        metrics = all_metrics.get(strategy_name)
        
        if not metrics or metrics['total_trades'] < 5:
            # Pas assez de données: score neutre
            return 0.5
        
        # 1. Score de performance (0-1)
        perf_score = (
            metrics['win_rate'] * 0.4 +
            min(metrics['profit_factor'] / 5, 1.0) * 0.3 +  # Normaliser profit_factor
            min((metrics['sharpe_ratio'] + 2) / 4, 1.0) * 0.3  # Sharpe normalisé
        )
        
        # 2. Pénalité pour losing streak
        if metrics['current_streak'] < -3:
            perf_score *= 0.5  # Réduire de moitié si mauvaise passe
        elif metrics['current_streak'] > 3:
            perf_score *= 1.2  # Bonus si bonne passe
        
        # 3. Score de contexte (basé sur compatibilité stratégie/marché)
        context_score = self._calculate_context_score(strategy_name, context)

        self.logger.debug(
            f"📊 {strategy_name}: perf={perf_score:.2f}, context={context_score:.2f}"
        )

        # 4. Score final
        final_score = (
            perf_score * (1 - self.context_weight) +
            context_score * self.context_weight
        )
        
        # 5. Vérifier si stratégie devrait être désactivée
        if self.performance_tracker.should_disable_strategy(strategy_name):
            final_score = 0.0
            self.logger.warning(f"⚠️ {strategy_name}: Temporairement désactivée")
        
        return final_score
    
    def adjust_signal_confidence(
        self,
        signal,
        strategy_name: str,
        strategy_weight: float
    ):
        """
        Ajuste la confidence d'un signal selon le poids de la stratégie
        
        Args:
            signal: Signal à ajuster
            strategy_name: Nom de la stratégie
            strategy_weight: Poids assigné (0-1)
        """
        
        # Ajuster confidence
        original_confidence = signal.confidence
        adjusted_confidence = original_confidence * strategy_weight
        
        # Limiter entre 0 et 1
        signal.confidence = max(0.0, min(1.0, adjusted_confidence))
        
        # Ajouter metadata
        if not hasattr(signal, 'metadata'):
            signal.metadata = {}
        
        signal.metadata['original_confidence'] = original_confidence
        signal.metadata['strategy_weight'] = strategy_weight
        signal.metadata['meta_model_adjusted'] = True
        
        self.logger.debug(
            f"🔧 {strategy_name}: confidence {original_confidence:.2f} → {signal.confidence:.2f} "
            f"(weight: {strategy_weight:.2f})"
        )
    
    def record_decision(
        self,
        market_context: Dict,
        selected_strategies: Dict[str, float],
        final_signals: List
    ):
        """Enregistre une décision pour analyse future"""
        
        decision = {
            'timestamp': datetime.now(),
            'market_context': market_context,
            'selected_strategies': selected_strategies,
            'num_final_signals': len(final_signals),
            'selection_mode': self.selection_mode
        }
        
        self.decision_history.append(decision)
        
        # Garder seulement les 1000 dernières décisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def get_decision_stats(self) -> Dict:
        """Statistiques sur les décisions du meta-model"""
        
        if not self.decision_history:
            return {}
        
        # Analyser les décisions récentes
        recent = self.decision_history[-100:]
        
        strategy_selections = {}
        for decision in recent:
            for strategy, weight in decision['selected_strategies'].items():
                if strategy not in strategy_selections:
                    strategy_selections[strategy] = []
                strategy_selections[strategy].append(weight)
        
        # Moyennes
        avg_weights = {s: np.mean(weights) for s, weights in strategy_selections.items()}
        
        return {
            'total_decisions': len(self.decision_history),
            'recent_decisions': len(recent),
            'average_weights': avg_weights,
            'selection_mode': self.selection_mode
        }


__all__ = ['MLMetaModel']
