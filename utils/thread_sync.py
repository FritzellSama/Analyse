"""
Thread synchronization utilities for Quantum Trader Pro.
Provides thread-safe data structures and synchronization primitives.
"""
from threading import RLock, Lock, Event
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Iterator
from loguru import logger
import time


class ThreadSafeDict:
    """Thread-safe dictionary wrapper with RLock protection."""

    def __init__(self, initial_data: Optional[Dict] = None):
        self._dict = initial_data.copy() if initial_data else {}
        self._lock = RLock()

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def __len__(self):
        with self._lock:
            return len(self._dict)

    def __iter__(self):
        with self._lock:
            return iter(self._dict.copy())

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def pop(self, key, *args):
        with self._lock:
            return self._dict.pop(key, *args)

    def keys(self):
        with self._lock:
            return list(self._dict.keys())

    def values(self):
        with self._lock:
            return list(self._dict.values())

    def items(self):
        with self._lock:
            return list(self._dict.items())

    def copy(self):
        with self._lock:
            return self._dict.copy()

    def clear(self):
        with self._lock:
            self._dict.clear()

    def update(self, other: Dict):
        with self._lock:
            self._dict.update(other)

    def setdefault(self, key, default=None):
        with self._lock:
            return self._dict.setdefault(key, default)

    @contextmanager
    def locked(self):
        """Context manager for bulk operations."""
        with self._lock:
            yield self._dict


class ThreadSafeList:
    """Thread-safe list wrapper with RLock protection."""

    def __init__(self, initial_data: Optional[List] = None):
        self._list = list(initial_data) if initial_data else []
        self._lock = RLock()

    def append(self, item):
        with self._lock:
            self._list.append(item)

    def extend(self, items):
        with self._lock:
            self._list.extend(items)

    def remove(self, item):
        with self._lock:
            self._list.remove(item)

    def pop(self, index=-1):
        with self._lock:
            return self._list.pop(index)

    def insert(self, index, item):
        with self._lock:
            self._list.insert(index, item)

    def __getitem__(self, index):
        with self._lock:
            return self._list[index]

    def __setitem__(self, index, value):
        with self._lock:
            self._list[index] = value

    def __len__(self):
        with self._lock:
            return len(self._list)

    def __iter__(self):
        with self._lock:
            return iter(self._list.copy())

    def __contains__(self, item):
        with self._lock:
            return item in self._list

    def copy(self):
        with self._lock:
            return self._list.copy()

    def clear(self):
        with self._lock:
            self._list.clear()

    def index(self, item):
        with self._lock:
            return self._list.index(item)

    @contextmanager
    def locked(self):
        """Context manager for bulk operations."""
        with self._lock:
            yield self._list


class ThreadSafeCounter:
    """Thread-safe counter with atomic operations."""

    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = RLock()

    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        with self._lock:
            self._value -= amount
            return self._value

    def get(self) -> int:
        with self._lock:
            return self._value

    def set(self, value: int) -> None:
        with self._lock:
            self._value = value

    def reset(self) -> None:
        with self._lock:
            self._value = 0


def synchronized(lock_attr: str = '_lock'):
    """
    Decorator to synchronize method execution using instance lock.

    Usage:
        class MyClass:
            def __init__(self):
                self._lock = RLock()

            @synchronized()
            def critical_method(self):
                # Thread-safe code here
                pass

            @synchronized('_custom_lock')
            def other_method(self):
                pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_attr, None)
            if lock is None:
                raise AttributeError(f"Object has no attribute '{lock_attr}'. Add self.{lock_attr} = RLock() in __init__")
            with lock:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def atomic_operation(lock: RLock, operation_name: str = "operation", log_debug: bool = False):
    """
    Context manager for atomic operations with optional logging.

    Args:
        lock: RLock instance to use
        operation_name: Name for logging
        log_debug: Whether to log lock acquisition/release
    """
    if log_debug:
        logger.debug(f"Acquiring lock for {operation_name}")

    try:
        lock.acquire()
        if log_debug:
            logger.debug(f"Lock acquired for {operation_name}")
        yield
    finally:
        lock.release()
        if log_debug:
            logger.debug(f"Lock released for {operation_name}")


@contextmanager
def multi_lock(*locks: RLock) -> Iterator[None]:
    """
    Acquire multiple locks in order to prevent deadlock.

    Usage:
        with multi_lock(lock1, lock2, lock3):
            # Critical section with all locks held
            pass
    """
    # Sort locks by id to ensure consistent ordering and prevent deadlock
    sorted_locks = sorted(locks, key=id)
    acquired = []

    try:
        for lock in sorted_locks:
            lock.acquire()
            acquired.append(lock)
        yield
    finally:
        # Release in reverse order
        for lock in reversed(acquired):
            lock.release()


class ReadWriteLock:
    """
    Read-write lock allowing multiple readers or single writer.

    Useful for data that is read frequently but written rarely.
    """

    def __init__(self):
        self._read_ready = Lock()
        self._readers = 0
        self._readers_lock = Lock()

    @contextmanager
    def read_lock(self):
        """Acquire read lock (multiple readers allowed)."""
        with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                self._read_ready.acquire()

        try:
            yield
        finally:
            with self._readers_lock:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.release()

    @contextmanager
    def write_lock(self):
        """Acquire write lock (exclusive access)."""
        self._read_ready.acquire()
        try:
            yield
        finally:
            self._read_ready.release()


class RateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, max_calls: int, period_seconds: float = 1.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls = []
        self._lock = RLock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Try to acquire permission to make a call.

        Args:
            timeout: Maximum time to wait (None = no wait)

        Returns:
            True if permission granted, False otherwise
        """
        start_time = time.time()

        while True:
            with self._lock:
                now = time.time()

                # Remove old calls outside the window
                self._calls = [t for t in self._calls if now - t < self.period]

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return True

            if timeout is None:
                return False

            if time.time() - start_time >= timeout:
                return False

            time.sleep(0.01)  # Small sleep before retry

    def reset(self):
        """Reset the rate limiter."""
        with self._lock:
            self._calls.clear()


__all__ = [
    'ThreadSafeDict',
    'ThreadSafeList',
    'ThreadSafeCounter',
    'synchronized',
    'atomic_operation',
    'multi_lock',
    'ReadWriteLock',
    'RateLimiter'
]
