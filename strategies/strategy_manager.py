"""
Strategy Manager - Quantum Trader Pro
Gère les stratégies HF avec ML Signal Filter
"""

from typing import Dict, List, Optional
from strategies.base_strategy import BaseStrategy, Signal
from strategies.ichimoku_scalping import IchimokuScalpingStrategy
from strategies.ml_strategy import MLStrategy
from utils.logger import setup_logger
from utils.config_helpers import get_nested_config
from utils.safe_math import safe_divide
import pandas as pd

# HF Strategies
try:
    from strategies.high_frequency.hf_manager import HFManager
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ML Signal Filter
try:
    from ml_models.signal_filter import MLSignalFilter
    ML_FILTER_AVAILABLE = True
except ImportError:
    ML_FILTER_AVAILABLE = False


class StrategyManager:
    """Gestionnaire de stratégies multiples avec HF Manager"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('StrategyManager')

        self.strategies = {}
        self.allocations = {}

        # HF Manager (nouveau système principal)
        self.use_hf_manager = config.get('strategies', {}).get('hf_manager', {}).get('enabled', False)
        self.hf_manager = None

        # ML Signal Filter (remplace Meta-Model)
        self.use_ml_filter = config.get('strategies', {}).get('use_ml_filter', False)
        self.ml_filter = None

        # Initialiser composants
        self._initialize_ml_filter()
        self._initialize_strategies()

        self.logger.info(f"HF Manager: {'actif' if self.hf_manager else 'inactif'}")
        self.logger.info(f"ML Filter: {'actif' if self.ml_filter else 'inactif'}")

    def _initialize_ml_filter(self):
        """Initialise le ML Signal Filter"""
        if self.use_ml_filter and ML_FILTER_AVAILABLE:
            try:
                self.ml_filter = MLSignalFilter(self.config)
                self.logger.info("ML Signal Filter actif")
            except Exception as e:
                self.logger.warning(f"ML Filter init error: {e}")
                self.ml_filter = None

    def _initialize_strategies(self):
        """Initialise les stratégies configurées"""
        strat_config = get_nested_config(self.config, 'strategies', default={})

        # HF Manager (prioritaire si activé)
        if self.use_hf_manager and HF_AVAILABLE:
            try:
                self.hf_manager = HFManager(self.config)
                self.strategies['hf_manager'] = self.hf_manager
                self.allocations['hf_manager'] = 1.0  # 100% allocation au HF Manager
                self.logger.info("HF Manager activé (5 stratégies HF)")
            except Exception as e:
                self.logger.error(f"HF Manager init error: {e}")
                self.hf_manager = None

        # Ichimoku Scalping (fallback ou complément)
        ichimoku_cfg = get_nested_config(strat_config, 'ichimoku_scalping', default={})
        if ichimoku_cfg.get('enabled', False):
            self.strategies['ichimoku_scalping'] = IchimokuScalpingStrategy(self.config)
            self.allocations['ichimoku_scalping'] = ichimoku_cfg.get('weight', 0.3)
            self.logger.info("Ichimoku Scalping activée")

        # ML Strategy (si pas de HF Manager)
        ml_cfg = get_nested_config(strat_config, 'ml_strategy', default={})
        if ml_cfg.get('enabled', False) and not self.use_hf_manager:
            ml_strat = MLStrategy(self.config)
            models_path = ml_cfg.get('models_path')
            if models_path:
                ml_strat.load_models(models_path)
            self.strategies['ml_strategy'] = ml_strat
            self.allocations['ml_strategy'] = ml_cfg.get('weight', 0.3)
            self.logger.info("ML Strategy activée")

        # Normaliser allocations
        total_weight = sum(self.allocations.values())
        if total_weight > 0:
            self.allocations = {k: safe_divide(v, total_weight, default=0.0) for k, v in self.allocations.items()}

        self.logger.info(f"Stratégies: {list(self.strategies.keys())}")
        self.logger.info(f"Allocations: {self.allocations}")

    def generate_all_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Signal]]:
        """
        Génère signaux de toutes les stratégies

        Args:
            data: Dict avec timeframe -> DataFrame

        Returns:
            Dict avec strategy_name -> List[Signal]
        """
        all_signals = {}

        # Mode HF Manager (prioritaire)
        if self.hf_manager:
            try:
                hf_signals = self.hf_manager.generate_signals(data)
                if hf_signals:
                    all_signals['hf_manager'] = hf_signals
                    self.logger.info(f"HF Manager: {len(hf_signals)} signaux")
            except Exception as e:
                self.logger.error(f"HF Manager error: {e}")

        # Autres stratégies
        for name, strategy in self.strategies.items():
            if name == 'hf_manager':
                continue  # Déjà traité

            try:
                signals = strategy.generate_signals(data)
                if signals:
                    all_signals[name] = signals
                    self.logger.info(f"{name}: {len(signals)} signaux")
            except Exception as e:
                self.logger.error(f"Erreur {name}: {e}")

        return all_signals

    def filter_conflicting_signals(self, all_signals: Dict[str, List[Signal]], data: Dict[str, pd.DataFrame] = None) -> List[Signal]:
        """
        Filtre les signaux avec ML Signal Filter

        Args:
            all_signals: Dict strategy -> signals
            data: Dict timeframe -> DataFrame (pour ML filter)

        Returns:
            Liste de signaux validés
        """
        if not all_signals:
            return []

        # Collecter tous les signaux
        all_signal_list = []
        for strategy_name, signals in all_signals.items():
            for signal in signals:
                signal.strategy = strategy_name
                all_signal_list.append({
                    'signal': signal,
                    'strategy': strategy_name,
                    'weight': self.allocations.get(strategy_name, 0.5),
                    'ml_validated': True,
                    'ml_confidence': 0.0
                })

        # Appliquer ML Filter si disponible
        if self.ml_filter and self.ml_filter.is_ready and data:
            for item in all_signal_list:
                signal = item['signal']
                try:
                    # Préparer signal pour validation ML
                    signal_dict = {
                        'side': signal.action,
                        'strategy': item['strategy'],
                        'confidence': signal.confidence
                    }
                    ml_result = self.ml_filter.validate_signal(signal_dict, data)

                    item['ml_validated'] = ml_result['validated']
                    item['ml_confidence'] = ml_result['confidence']

                    # Ajuster confidence du signal
                    if ml_result['validated']:
                        # Boost si ML confirme
                        signal.confidence = min(signal.confidence * 1.1, 0.99)
                    else:
                        # Réduire si ML rejette
                        signal.confidence *= 0.7

                    self.logger.debug(
                        f"ML Filter: {item['strategy']} {signal.action} | "
                        f"ML conf: {ml_result['confidence']:.2%} | "
                        f"{ml_result['recommendation']}"
                    )
                except Exception as e:
                    self.logger.debug(f"ML Filter error: {e}")

        # Filtrer signaux rejetés par ML (si ML actif)
        if self.ml_filter and self.ml_filter.is_ready:
            min_ml_confidence = self.config.get('strategies', {}).get('ml_min_confidence', 0.65)
            all_signal_list = [
                item for item in all_signal_list
                if item['ml_validated'] or item['ml_confidence'] >= min_ml_confidence
            ]

        # Calculer score final
        for item in all_signal_list:
            signal = item['signal']
            item['score'] = signal.confidence * item['weight']
            if item['ml_confidence'] > 0:
                item['score'] *= (1 + item['ml_confidence'] * 0.2)  # Boost ML

        # Trier par score
        all_signal_list.sort(key=lambda x: x['score'], reverse=True)

        # Filtrer conflits (un signal par symbol)
        final_signals = []
        used_symbols = set()

        for item in all_signal_list:
            signal = item['signal']

            if signal.symbol in used_symbols:
                self.logger.debug(f"Conflit: {item['strategy']} {signal.action} {signal.symbol} ignoré")
                continue

            final_signals.append(signal)
            used_symbols.add(signal.symbol)

            self.logger.info(
                f"Signal retenu: {item['strategy']} {signal.action} "
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
            if hasattr(strategy, 'get_performance_stats'):
                stats[name] = strategy.get_performance_stats()
            elif hasattr(strategy, 'get_stats'):
                stats[name] = strategy.get_stats()

        # Stats globales
        total_signals = sum(s.get('total_signals', 0) for s in stats.values())
        total_pnl = sum(s.get('total_pnl', 0) for s in stats.values())

        stats['global'] = {
            'total_signals': total_signals,
            'total_pnl': total_pnl,
            'avg_pnl': safe_divide(total_pnl, total_signals, default=0.0),
            'strategies_active': len(self.strategies),
            'hf_manager_active': self.hf_manager is not None,
            'ml_filter_active': self.ml_filter is not None and self.ml_filter.is_ready
        }

        return stats

    def reset_all_strategies(self):
        """Reset toutes les stratégies"""
        for strategy in self.strategies.values():
            if hasattr(strategy, 'reset_performance'):
                strategy.reset_performance()
            elif hasattr(strategy, 'reset'):
                strategy.reset()

        self.logger.info("Toutes les stratégies reset")

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
        """Enregistre le résultat d'un trade"""
        # HF Manager tracking
        if self.hf_manager and hasattr(self.hf_manager, 'record_trade'):
            self.hf_manager.record_trade(
                strategy_name=strategy_name,
                pnl=pnl,
                is_win=pnl > 0
            )

    def get_ml_filter_status(self) -> Dict:
        """Retourne statut du ML Filter"""
        if not self.ml_filter:
            return {'enabled': False}

        return self.ml_filter.get_status()

    def get_hf_manager_stats(self) -> Dict:
        """Retourne stats du HF Manager"""
        if not self.hf_manager:
            return {'enabled': False}

        return self.hf_manager.get_stats()


__all__ = ['StrategyManager']
