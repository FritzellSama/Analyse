"""
Thread Safety Manager - Quantum Trader Pro
Gestion de la concurrence et synchronisation des threads

NOTE: Ce module réexporte depuis utils.safety pour unification.
      Conservé pour rétrocompatibilité des imports existants.
"""

# Réexporter depuis utils.safety pour unification du code
from utils.safety import (
    ThreadSafetyManager,
    OrderQueue,
    thread_manager,
    ThreadSafeCounter,
    global_counters,
    GlobalLockManager,
    global_locks,
    synchronized
)

__all__ = [
    'ThreadSafetyManager',
    'OrderQueue',
    'thread_manager',
    'ThreadSafeCounter',
    'global_counters',
    'GlobalLockManager',
    'global_locks',
    'synchronized'
]
