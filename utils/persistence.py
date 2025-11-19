"""
State Persistence - Quantum Trader Pro
Sauvegarde et restauration de l'état pour reprise après crash
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from utils.logger import setup_logger

logger = setup_logger('Persistence')


class StatePersistence:
    """
    Gestionnaire de persistance d'état pour reprise après crash.

    Sauvegarde périodiquement:
    - Positions ouvertes
    - Ordres en attente
    - État du circuit breaker
    - Statistiques de performance
    - Configuration runtime
    """

    def __init__(self, state_dir: str = "state", auto_save_interval: int = 300):
        """
        Initialise le gestionnaire de persistance

        Args:
            state_dir: Répertoire pour les fichiers d'état
            auto_save_interval: Intervalle de sauvegarde auto (secondes)
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save_interval = auto_save_interval

        self.current_state_file = self.state_dir / "current_state.json"
        self.positions_file = self.state_dir / "positions.json"
        self.orders_file = self.state_dir / "pending_orders.json"
        self.performance_file = self.state_dir / "performance.json"

        self._last_save_time = datetime.now()

        logger.info(f"✅ State Persistence initialisé dans {state_dir}/")

    def save_state(self, state: Dict[str, Any]) -> bool:
        """
        Sauvegarde l'état complet du bot

        Args:
            state: Dictionnaire contenant l'état complet

        Returns:
            True si succès
        """
        try:
            # Ajouter métadonnées
            state['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'checksum': self._calculate_checksum(state)
            }

            # Sauvegarder état principal
            with open(self.current_state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            # Sauvegarder backup horodaté
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.state_dir / f"state_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            # Nettoyer vieux backups (garder 50 derniers)
            self._cleanup_old_backups(keep=50)

            self._last_save_time = datetime.now()
            logger.debug(f"💾 État sauvegardé: {self.current_state_file}")

            return True

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde état: {e}")
            return False

    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Charge le dernier état sauvegardé

        Returns:
            État ou None si non trouvé/invalide
        """
        try:
            if not self.current_state_file.exists():
                logger.info("📂 Aucun état précédent trouvé")
                return None

            with open(self.current_state_file, 'r') as f:
                state = json.load(f)

            # Vérifier intégrité
            if not self._verify_state(state):
                logger.warning("⚠️  État corrompu, tentative avec backup")
                return self._load_latest_backup()

            saved_at = state.get('_metadata', {}).get('saved_at', 'unknown')
            logger.info(f"✅ État chargé (sauvegardé: {saved_at})")

            return state

        except Exception as e:
            logger.error(f"❌ Erreur chargement état: {e}")
            return self._load_latest_backup()

    def save_positions(self, positions: List[Dict[str, Any]]) -> bool:
        """
        Sauvegarde les positions ouvertes

        Args:
            positions: Liste des positions

        Returns:
            True si succès
        """
        try:
            data = {
                'positions': positions,
                'count': len(positions),
                'saved_at': datetime.now().isoformat()
            }

            with open(self.positions_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"💾 {len(positions)} positions sauvegardées")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde positions: {e}")
            return False

    def load_positions(self) -> List[Dict[str, Any]]:
        """
        Charge les positions sauvegardées

        Returns:
            Liste des positions
        """
        try:
            if not self.positions_file.exists():
                return []

            with open(self.positions_file, 'r') as f:
                data = json.load(f)

            positions = data.get('positions', [])
            logger.info(f"✅ {len(positions)} positions chargées")

            return positions

        except Exception as e:
            logger.error(f"❌ Erreur chargement positions: {e}")
            return []

    def save_performance(self, metrics: Dict[str, Any]) -> bool:
        """
        Sauvegarde les métriques de performance

        Args:
            metrics: Métriques de performance

        Returns:
            True si succès
        """
        try:
            # Charger historique existant
            history = []
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                    history = data.get('history', [])

            # Ajouter nouvelles métriques
            metrics['timestamp'] = datetime.now().isoformat()
            history.append(metrics)

            # Garder 1000 derniers points
            if len(history) > 1000:
                history = history[-1000:]

            data = {
                'current': metrics,
                'history': history,
                'updated_at': datetime.now().isoformat()
            }

            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            return True

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde performance: {e}")
            return False

    def load_performance(self) -> Dict[str, Any]:
        """
        Charge l'historique de performance

        Returns:
            Métriques et historique
        """
        try:
            if not self.performance_file.exists():
                return {'current': {}, 'history': []}

            with open(self.performance_file, 'r') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"❌ Erreur chargement performance: {e}")
            return {'current': {}, 'history': []}

    def should_auto_save(self) -> bool:
        """Vérifie si une sauvegarde auto est nécessaire"""
        elapsed = (datetime.now() - self._last_save_time).total_seconds()
        return elapsed >= self.auto_save_interval

    def _calculate_checksum(self, data: Dict) -> str:
        """Calcule un checksum simple pour vérification d'intégrité"""
        import hashlib
        # Exclure métadonnées du calcul
        data_copy = {k: v for k, v in data.items() if not k.startswith('_')}
        data_str = json.dumps(data_copy, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def _verify_state(self, state: Dict) -> bool:
        """Vérifie l'intégrité d'un état"""
        if '_metadata' not in state:
            return False

        saved_checksum = state['_metadata'].get('checksum', '')
        calculated = self._calculate_checksum(state)

        return saved_checksum == calculated

    def _load_latest_backup(self) -> Optional[Dict[str, Any]]:
        """Charge le backup le plus récent"""
        try:
            backups = sorted(self.state_dir.glob("state_*.json"), reverse=True)

            for backup in backups[:5]:  # Essayer les 5 plus récents
                try:
                    with open(backup, 'r') as f:
                        state = json.load(f)

                    if self._verify_state(state):
                        logger.info(f"✅ Backup chargé: {backup.name}")
                        return state
                except Exception as e:
                    logger.debug(f"Backup {backup.name} invalide: {e}")
                    continue

            logger.error("❌ Aucun backup valide trouvé")
            return None

        except Exception as e:
            logger.error(f"❌ Erreur chargement backup: {e}")
            return None

    def _cleanup_old_backups(self, keep: int = 50) -> None:
        """Supprime les vieux backups"""
        try:
            backups = sorted(self.state_dir.glob("state_*.json"))

            if len(backups) > keep:
                for old_backup in backups[:-keep]:
                    old_backup.unlink()
                logger.debug(f"🗑️  {len(backups) - keep} vieux backups supprimés")

        except Exception as e:
            logger.warning(f"⚠️  Erreur nettoyage backups: {e}")

    def clear_state(self) -> None:
        """Efface tout l'état (attention !)"""
        try:
            for file in [self.current_state_file, self.positions_file, self.orders_file]:
                if file.exists():
                    file.unlink()
            logger.warning("⚠️  État effacé")
        except Exception as e:
            logger.error(f"❌ Erreur effacement état: {e}")


class CrashRecovery:
    """
    Gestionnaire de récupération après crash.

    Détecte les crashs et restaure l'état du bot.
    """

    def __init__(self, persistence: StatePersistence):
        """
        Initialise le gestionnaire de récupération

        Args:
            persistence: Instance de StatePersistence
        """
        self.persistence = persistence
        self.lock_file = persistence.state_dir / ".running.lock"
        self.crash_detected = False

        logger.info("✅ Crash Recovery initialisé")

    def check_for_crash(self) -> bool:
        """
        Vérifie si le bot a crashé précédemment

        Returns:
            True si crash détecté
        """
        if self.lock_file.exists():
            # Lock file présent = crash probable
            logger.warning("⚠️  Crash détecté (lock file présent)")
            self.crash_detected = True
            return True

        return False

    def acquire_lock(self) -> bool:
        """
        Acquiert le lock de démarrage

        Returns:
            True si lock acquis
        """
        try:
            with open(self.lock_file, 'w') as f:
                f.write(f"PID: {os.getpid()}\nStarted: {datetime.now().isoformat()}\n")
            logger.debug("🔒 Lock acquis")
            return True
        except Exception as e:
            logger.error(f"❌ Impossible d'acquérir le lock: {e}")
            return False

    def release_lock(self) -> None:
        """Libère le lock (arrêt propre)"""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
            logger.debug("🔓 Lock libéré")
        except Exception as e:
            logger.warning(f"⚠️  Erreur libération lock: {e}")

    def recover_positions(self) -> List[Dict[str, Any]]:
        """
        Récupère les positions après crash

        Returns:
            Liste des positions à restaurer
        """
        if not self.crash_detected:
            return []

        positions = self.persistence.load_positions()

        if positions:
            logger.warning(
                f"⚠️  {len(positions)} positions à restaurer après crash"
            )

        return positions

    def recover_state(self) -> Optional[Dict[str, Any]]:
        """
        Récupère l'état complet après crash

        Returns:
            État à restaurer ou None
        """
        if not self.crash_detected:
            return None

        state = self.persistence.load_state()

        if state:
            logger.warning("⚠️  État récupéré après crash")

        return state


# Import os pour getpid
import os

__all__ = ['StatePersistence', 'CrashRecovery']
