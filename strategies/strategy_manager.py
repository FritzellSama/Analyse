"""
Strategy Manager - Quantum Trader Pro
Gère les stratégies multiples et allocation avec ML Meta-Model
"""

from typing import Dict, List
from strategies.base_strategy import BaseStrategy, Signal
from strategies.ichimoku_scalping import IchimokuScalpingStrategy
from strategies.grid_trading import GridTradingStrategy
from strategies.dca_bot import DCABotStrategy
from strategies.ml_strategy import MLStrategy
from ml_models.meta_model import MLMetaModel
from ml_models.strategy_performance_tracker import StrategyPerformanceTracker
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
from utils.safe_math import safe_divide
import pandas as pd

class StrategyManager:
    """Gestionnaire de stratégies multiples"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('StrategyManager')
        
        self.strategies = {}
        self.allocations = {}
        
        # ML Meta-Model components
        self.use_meta_model = config.get('ml', {}).get('meta_model', {}).get('enabled', False)
        
        if self.use_meta_model:
            self.performance_tracker = StrategyPerformanceTracker(config)
            self.meta_model = MLMetaModel(config, self.performance_tracker)
            self.logger.info("🧠 ML Meta-Model activé")
        else:
            self.performance_tracker = None
            self.meta_model = None
        
        # Initialiser stratégies activées
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialise les stratégies configurées"""
        strat_config = get_nested_config(self.config, 'strategies', default={})

        # Ichimoku Scalping
        ichimoku_cfg = get_nested_config(strat_config, 'ichimoku_scalping', default={})
        if ichimoku_cfg.get('enabled', False):
            self.strategies['ichimoku_scalping'] = IchimokuScalpingStrategy(self.config)
            self.allocations['ichimoku_scalping'] = ichimoku_cfg.get('weight', 0.3)
            self.logger.info("✅ Ichimoku Scalping activée")

        # Grid Trading
        grid_cfg = get_nested_config(strat_config, 'grid_trading', default={})
        if grid_cfg.get('enabled', False):
            self.strategies['grid_trading'] = GridTradingStrategy(self.config)
            self.allocations['grid_trading'] = grid_cfg.get('weight', 0.3)
            self.logger.info("✅ Grid Trading activée")

        # DCA Bot
        dca_cfg = get_nested_config(strat_config, 'dca_bot', default={})
        if dca_cfg.get('enabled', False):
            self.strategies['dca_bot'] = DCABotStrategy(self.config)
            self.allocations['dca_bot'] = dca_cfg.get('weight', 0.3)
            self.logger.info("✅ DCA Bot activé")

        # ML Strategy
        ml_cfg = get_nested_config(strat_config, 'ml_strategy', default={})
        if ml_cfg.get('enabled', False):
            ml_strat = MLStrategy(self.config)
            # Charger modèles si path fourni
            models_path = ml_cfg.get('models_path')
            if models_path:
                ml_strat.load_models(models_path)
            self.strategies['ml_strategy'] = ml_strat
            self.allocations['ml_strategy'] = ml_cfg.get('weight', 0.3)
            self.logger.info("✅ ML Strategy activée")

        # Normaliser allocations
        total_weight = sum(self.allocations.values())
        if total_weight > 0:
            self.allocations = {k: safe_divide(v, total_weight, default=0.0) for k, v in self.allocations.items()}
        
        self.logger.info(f"📊 Stratégies: {list(self.strategies.keys())}")
        self.logger.info(f"💰 Allocations: {self.allocations}")
    
    def generate_all_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Signal]]:
        """
        Génère signaux de toutes les stratégies
        
        Args:
            data: Dict avec timeframe → DataFrame
        
        Returns:
            Dict avec strategy_name → List[Signal]
        """
        all_signals = {}
        
        for name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data)
                
                if signals:
                    all_signals[name] = signals
                    self.logger.info(f"📊 {name}: {len(signals)} signaux")
            
            except Exception as e:
                self.logger.error(f"❌ Erreur {name}: {e}")
                continue
        
        return all_signals
    
    def filter_conflicting_signals(self, all_signals: Dict[str, List[Signal]], data: Dict[str, pd.DataFrame] = None) -> List[Signal]:
        """
        Filtre les signaux conflictuels avec ML Meta-Model
        
        Args:
            all_signals: Dict strategy → signals
            data: Dict timeframe → DataFrame (pour contexte marché)
        
        Returns:
            Liste de signaux validés
        """
        if not all_signals:
            return []
        
        # Si Meta-Model activé, l'utiliser pour sélection intelligente
        if self.use_meta_model and self.meta_model and data:
            return self._meta_model_selection(all_signals, data)
        
        # Sinon, méthode classique par score
        return self._classic_selection(all_signals)
    
    def _meta_model_selection(self, all_signals: Dict[str, List[Signal]], data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Sélection des signaux via Meta-Model"""
        
        # 1. Analyser contexte marché
        market_context = self.meta_model.analyze_market_context(data)
        
        self.logger.info(
            f"📊 Contexte marché: {market_context['trend']} | "
            f"Vol: {market_context['volatility']:.3f} | "
            f"Volume: {market_context['volume']}"
        )
        
        # 2. Sélectionner stratégies et poids
        strategy_weights = self.meta_model.select_strategies(all_signals, market_context)
        
        if not strategy_weights:
            self.logger.warning("⚠️ Aucune stratégie sélectionnée par Meta-Model")
            return []
        
        # 3. Ajuster confidence des signaux selon poids
        weighted_signals = []
        
        for strategy_name, signals in all_signals.items():
            weight = strategy_weights.get(strategy_name, 0)
            
            if weight == 0:
                continue  # Stratégie filtrée par Meta-Model
            
            for signal in signals:
                # Ajouter nom stratégie
                signal.strategy = strategy_name
                
                # Ajuster confidence
                self.meta_model.adjust_signal_confidence(signal, strategy_name, weight)
                
                # Score final
                score = signal.confidence * weight
                
                weighted_signals.append({
                    'signal': signal,
                    'strategy': strategy_name,
                    'weight': weight,
                    'score': score
                })
        
        # 4. Trier par score
        weighted_signals.sort(key=lambda x: x['score'], reverse=True)
        
        # 5. Filtrer conflits
        final_signals = []
        used_symbols = set()
        
        for item in weighted_signals:
            signal = item['signal']
            
            if signal.symbol in used_symbols:
                self.logger.warning(
                    f"⚠️ Conflit: {item['strategy']} {signal.action} {signal.symbol} ignoré"
                )
                continue
            
            final_signals.append(signal)
            used_symbols.add(signal.symbol)
            
            self.logger.info(
                f"✅ Signal retenu: {item['strategy']} {signal.action} "
                f"conf={signal.confidence:.2f} weight={item['weight']:.2f} score={item['score']:.2f}"
            )
        
        # 6. Enregistrer décision
        self.meta_model.record_decision(market_context, strategy_weights, final_signals)
        
        return final_signals
    
    def _classic_selection(self, all_signals: Dict[str, List[Signal]]) -> List[Signal]:
        """Sélection classique par score (sans Meta-Model)"""
        
        # Collecter tous les signaux avec priorité
        weighted_signals = []
        
        for strategy_name, signals in all_signals.items():
            weight = self.allocations.get(strategy_name, 0)
            
            for signal in signals:
                # Ajouter le nom de la stratégie au signal
                signal.strategy = strategy_name
                # Score = confidence * weight
                score = signal.confidence * weight
                weighted_signals.append({
                    'signal': signal,
                    'strategy': strategy_name,
                    'weight': weight,
                    'score': score
                })
        
        # Trier par score décroissant
        weighted_signals.sort(key=lambda x: x['score'], reverse=True)
        
        # Filtrer conflits (même symbol, actions opposées)
        final_signals = []
        used_symbols = set()
        
        for item in weighted_signals:
            signal = item['signal']
            
            # Si symbol déjà utilisé avec action différente, skip
            if signal.symbol in used_symbols:
                self.logger.warning(
                    f"⚠️  Conflit: {item['strategy']} {signal.action} {signal.symbol} ignoré"
                )
                continue
            
            final_signals.append(signal)
            used_symbols.add(signal.symbol)
            
            self.logger.info(
                f"✅ Signal retenu: {item['strategy']} {signal.action} "
                f"conf={signal.confidence:.2f} score={item['score']:.2f}"
            )
        
        return final_signals
    
    def get_strategy_allocation(self, strategy_name: str, total_capital: float) -> float:
        """
        Calcule capital alloué à une stratégie
        
        Args:
            strategy_name: Nom stratégie
            total_capital: Capital total
        
        Returns:
            Capital alloué
        """
        weight = self.allocations.get(strategy_name, 0)
        return total_capital * weight
    
    def get_all_performance_stats(self) -> Dict:
        """Récupère stats de toutes les stratégies"""
        stats = {}
        
        for name, strategy in self.strategies.items():
            stats[name] = strategy.get_performance_stats()
        
        # Stats globales
        total_signals = sum(s['total_signals'] for s in stats.values())
        total_pnl = sum(s['total_pnl'] for s in stats.values())
        
        stats['global'] = {
            'total_signals': total_signals,
            'total_pnl': total_pnl,
            'avg_pnl': safe_divide(total_pnl, total_signals, default=0.0),
            'strategies_active': len(self.strategies)
        }
        
        return stats
    
    def reset_all_strategies(self):
        """Reset toutes les stratégies"""
        for strategy in self.strategies.values():
            strategy.reset_performance()
        
        self.logger.info("🔄 Toutes les stratégies reset")
    
    def record_trade_result(
        self,
        strategy_name: str,
        signal_time,
        entry_price: float,
        exit_price: float,
        exit_time,
        pnl: float,
        action: str,
        market_context: Dict = None
    ):
        """
        Enregistre le résultat d'un trade pour le Performance Tracker
        
        Args:
            strategy_name: Nom de la stratégie
            signal_time: Timestamp du signal
            entry_price: Prix d'entrée
            exit_price: Prix de sortie
            exit_time: Timestamp de sortie
            pnl: Profit/Loss
            action: BUY ou SELL
            market_context: Contexte marché
        """
        if self.performance_tracker:
            self.performance_tracker.record_trade(
                strategy_name=strategy_name,
                signal_time=signal_time,
                entry_price=entry_price,
                exit_price=exit_price,
                exit_time=exit_time,
                pnl=pnl,
                action=action,
                market_context=market_context
            )
    
    def get_meta_model_stats(self) -> Dict:
        """Retourne stats du Meta-Model"""
        if not self.meta_model:
            return {}
        
        return {
            'enabled': self.use_meta_model,
            'selection_mode': self.meta_model.selection_mode,
            'decision_stats': self.meta_model.get_decision_stats(),
            'performance_tracker': self.performance_tracker.get_summary() if self.performance_tracker else {}
        }

__all__ = ['StrategyManager']
