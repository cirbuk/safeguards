"""Cache management and optimization for the Agent Safety Framework."""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")
KT = TypeVar("KT")
VT = TypeVar("VT")


class LRUCache(Generic[KT, VT]):
    """Least Recently Used (LRU) cache implementation."""

    def __init__(self, capacity: int):
        """Initialize LRU cache.

        Args:
            capacity: Maximum number of items to store
        """
        self._cache = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def get(self, key: KT) -> VT | None:
        """Get value from cache and update access order."""
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: KT, value: VT) -> None:
        """Add or update cache entry."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._capacity:
                self._cache.popitem(last=False)


class TimedCache(Generic[KT, VT]):
    """Cache with time-based expiration."""

    def __init__(self, ttl_seconds: int):
        """Initialize timed cache.

        Args:
            ttl_seconds: Time to live in seconds for cache entries
        """
        self._cache: dict[KT, tuple[VT, float]] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self, key: KT) -> VT | None:
        """Get value from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                return None
            return value

    def put(self, key: KT, value: VT) -> None:
        """Add or update cache entry with current timestamp."""
        with self._lock:
            self._cache[key] = (value, time.time())

    def cleanup(self) -> None:
        """Remove expired entries."""
        with self._lock:
            current_time = time.time()
            expired = [k for k, (_, t) in self._cache.items() if current_time - t > self._ttl]
            for key in expired:
                del self._cache[key]


class CacheManager:
    """Manages different caching strategies and configurations."""

    def __init__(self):
        """Initialize cache manager."""
        self._lru_caches: dict[str, LRUCache] = {}
        self._timed_caches: dict[str, TimedCache] = {}
        self._function_cache: dict[str, Any] = {}
        self._stats: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def create_lru_cache(self, name: str, capacity: int) -> None:
        """Create a new LRU cache.

        Args:
            name: Unique name for the cache
            capacity: Maximum number of items to store
        """
        with self._lock:
            self._lru_caches[name] = LRUCache(capacity)
            self._stats[name] = {"hits": 0, "misses": 0}

    def create_timed_cache(self, name: str, ttl_seconds: int) -> None:
        """Create a new timed cache.

        Args:
            name: Unique name for the cache
            ttl_seconds: Time to live in seconds for cache entries
        """
        with self._lock:
            self._timed_caches[name] = TimedCache(ttl_seconds)
            self._stats[name] = {"hits": 0, "misses": 0}

    def get_from_cache(self, cache_type: str, name: str, key: Any) -> Any | None:
        """Get value from specified cache.

        Args:
            cache_type: Type of cache ('lru' or 'timed')
            name: Cache name
            key: Cache key

        Returns:
            Cached value if found, None otherwise
        """
        cache = (self._lru_caches if cache_type == "lru" else self._timed_caches).get(
            name,
        )
        if not cache:
            return None

        value = cache.get(key)
        with self._lock:
            if value is not None:
                self._stats[name]["hits"] += 1
            else:
                self._stats[name]["misses"] += 1
        return value

    def put_in_cache(self, cache_type: str, name: str, key: Any, value: Any) -> None:
        """Store value in specified cache.

        Args:
            cache_type: Type of cache ('lru' or 'timed')
            name: Cache name
            key: Cache key
            value: Value to cache
        """
        cache = (self._lru_caches if cache_type == "lru" else self._timed_caches).get(
            name,
        )
        if cache:
            cache.put(key, value)

    def memoize(self, ttl_seconds: int | None = None) -> Callable:
        """Decorator for function memoization.

        Args:
            ttl_seconds: Optional time to live for cached results

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            cache_key = f"{func.__module__}.{func.__name__}"

            def wrapper(*args, **kwargs) -> Any:
                key = str((args, frozenset(kwargs.items())))

                if ttl_seconds is not None:
                    # Use timed cache
                    if cache_key not in self._timed_caches:
                        self.create_timed_cache(cache_key, ttl_seconds)
                    return self.get_from_cache(
                        "timed",
                        cache_key,
                        key,
                    ) or self._cache_and_return(
                        "timed",
                        cache_key,
                        key,
                        func,
                        *args,
                        **kwargs,
                    )
                # Use LRU cache
                if cache_key not in self._lru_caches:
                    self.create_lru_cache(cache_key, 100)  # Default capacity
                return self.get_from_cache(
                    "lru",
                    cache_key,
                    key,
                ) or self._cache_and_return(
                    "lru",
                    cache_key,
                    key,
                    func,
                    *args,
                    **kwargs,
                )

            return wrapper

        return decorator

    def _cache_and_return(
        self,
        cache_type: str,
        name: str,
        key: Any,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Helper to compute, cache and return function results."""
        result = func(*args, **kwargs)
        self.put_in_cache(cache_type, name, key, result)
        return result

    def get_stats(self, name: str) -> dict[str, int]:
        """Get cache statistics.

        Args:
            name: Cache name

        Returns:
            Dict with hit/miss statistics
        """
        return dict(self._stats.get(name, {"hits": 0, "misses": 0}))

    def clear_cache(self, cache_type: str, name: str) -> None:
        """Clear specified cache.

        Args:
            cache_type: Type of cache ('lru' or 'timed')
            name: Cache name
        """
        with self._lock:
            if cache_type == "lru" and name in self._lru_caches:
                self._lru_caches[name] = LRUCache(self._lru_caches[name]._capacity)
            elif cache_type == "timed" and name in self._timed_caches:
                self._timed_caches[name] = TimedCache(self._timed_caches[name]._ttl)
            if name in self._stats:
                self._stats[name] = {"hits": 0, "misses": 0}

    def cleanup_expired(self) -> None:
        """Clean up expired entries in all timed caches."""
        for cache in self._timed_caches.values():
            cache.cleanup()
