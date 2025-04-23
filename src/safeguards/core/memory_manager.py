"""Memory management and optimization for the Agent Safety Framework."""

import gc
import threading
import weakref
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ObjectPool(Generic[T]):
    """Generic object pool for reusing objects."""

    def __init__(self, factory, max_size: int = 100):
        """Initialize the object pool.

        Args:
            factory: Callable that creates new objects
            max_size: Maximum number of objects to keep in pool
        """
        self._factory = factory
        self._max_size = max_size
        self._pool = []
        self._lock = threading.Lock()

    def acquire(self) -> T:
        """Get an object from the pool or create a new one."""
        with self._lock:
            if self._pool:
                return self._pool.pop()
            return self._factory()

    def release(self, obj: T) -> None:
        """Return an object to the pool."""
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(obj)


class MemoryManager:
    """Manages memory optimization and resource cleanup."""

    def __init__(self, gc_threshold: tuple[int, int, int] | None = None):
        """Initialize the memory manager.

        Args:
            gc_threshold: Optional tuple of (threshold0, threshold1, threshold2) for
                         garbage collection thresholds
        """
        self._pools: dict[str, ObjectPool] = {}
        self._resource_refs = weakref.WeakSet()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._cache: dict[str, dict[Any, Any]] = defaultdict(dict)
        self._cache_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "misses": 0},
        )

        if gc_threshold:
            gc.set_threshold(*gc_threshold)

    def create_pool(self, name: str, factory: Any, max_size: int = 100) -> ObjectPool:
        """Create a new object pool.

        Args:
            name: Unique name for the pool
            factory: Callable that creates new objects
            max_size: Maximum number of objects to keep in pool

        Returns:
            New ObjectPool instance
        """
        pool = ObjectPool(factory, max_size)
        self._pools[name] = pool
        return pool

    def get_pool(self, name: str) -> ObjectPool | None:
        """Get an existing object pool by name."""
        return self._pools.get(name)

    def track_resource(self, resource: Any) -> None:
        """Track a resource for cleanup."""
        self._resource_refs.add(resource)

    def cleanup_resources(self) -> None:
        """Clean up tracked resources that are no longer referenced."""
        self._executor.submit(self._async_cleanup)

    def _async_cleanup(self) -> None:
        """Asynchronously clean up resources and run garbage collection."""
        # Clear any expired weak references
        self._resource_refs.clear()

        # Run garbage collection
        gc.collect()

    def cache_get(self, namespace: str, key: Any) -> Any | None:
        """Get a value from the cache.

        Args:
            namespace: Cache namespace
            key: Cache key

        Returns:
            Cached value if found, None otherwise
        """
        if value := self._cache[namespace].get(key):
            self._cache_stats[namespace]["hits"] += 1
            return value
        self._cache_stats[namespace]["misses"] += 1
        return None

    def cache_set(self, namespace: str, key: Any, value: Any) -> None:
        """Set a value in the cache.

        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
        """
        self._cache[namespace][key] = value

    def get_cache_stats(self, namespace: str) -> dict[str, int]:
        """Get cache hit/miss statistics for a namespace."""
        return dict(self._cache_stats[namespace])

    def clear_cache(self, namespace: str | None = None) -> None:
        """Clear the cache for a namespace or all caches.

        Args:
            namespace: Optional namespace to clear. If None, clears all caches.
        """
        if namespace:
            self._cache[namespace].clear()
            self._cache_stats[namespace] = {"hits": 0, "misses": 0}
        else:
            self._cache.clear()
            self._cache_stats.clear()
