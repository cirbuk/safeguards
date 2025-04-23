"""Example usage of memory and cache management features."""

import time
from dataclasses import dataclass

from safeguards.core.cache_manager import CacheManager
from safeguards.core.memory_manager import MemoryManager


@dataclass
class Request:
    """Example request object for pooling."""

    id: int | None = None
    data: str | None = None

    def reset(self):
        """Reset request data."""
        self.id = None
        self.data = None


def example_memory_management():
    """Demonstrate memory management features."""
    # Initialize memory manager
    memory_manager = MemoryManager(gc_threshold=(700, 10, 10))

    # Create request pool
    request_pool = memory_manager.create_pool(
        name="request_pool",
        factory=lambda: Request(),
        max_size=100,
    )

    # Example of using pooled objects
    for i in range(5):
        # Get request from pool
        request = request_pool.acquire()
        try:
            # Use request
            request.id = i
            request.data = f"Request {i}"
            print(f"Processing: {request}")
        finally:
            # Reset and return to pool
            request.reset()
            request_pool.release(request)

    # Example of resource tracking
    class Resource:
        def __init__(self, name):
            self.name = name

        def __del__(self):
            print(f"Cleaning up resource: {self.name}")

    # Create and track resources
    for i in range(3):
        resource = Resource(f"Resource {i}")
        memory_manager.track_resource(resource)

    # Trigger cleanup
    print("\nTriggering resource cleanup...")
    memory_manager.cleanup_resources()


def example_cache_management():
    """Demonstrate cache management features."""
    cache_manager = CacheManager()

    # LRU Cache example
    print("\nLRU Cache Example:")
    cache_manager.create_lru_cache("results", capacity=5)

    # Add items
    for i in range(6):
        cache_manager.put_in_cache("lru", "results", f"key{i}", f"value{i}")
        time.sleep(0.1)  # Simulate work

    # Check cache contents
    for i in range(6):
        result = cache_manager.get_from_cache("lru", "results", f"key{i}")
        print(f"key{i}: {result}")

    # Timed Cache example
    print("\nTimed Cache Example:")
    cache_manager.create_timed_cache("api_results", ttl_seconds=2)

    # Add items
    cache_manager.put_in_cache("timed", "api_results", "temp", "data")
    print(
        "Initial value:",
        cache_manager.get_from_cache("timed", "api_results", "temp"),
    )

    # Wait for expiration
    time.sleep(3)
    print(
        "After expiration:",
        cache_manager.get_from_cache("timed", "api_results", "temp"),
    )

    # Function memoization example
    print("\nMemoization Example:")

    @cache_manager.memoize(ttl_seconds=5)
    def expensive_operation(x: int, y: int) -> int:
        print(f"Computing {x} + {y}...")
        time.sleep(1)  # Simulate expensive operation
        return x + y

    # First call - will compute
    print("First call:", expensive_operation(2, 3))
    # Second call - will use cache
    print("Second call:", expensive_operation(2, 3))

    # Different arguments - will compute
    print("Different args:", expensive_operation(3, 4))

    # Check cache statistics
    print("\nCache Statistics:")
    print("LRU Cache:", cache_manager.get_stats("results"))
    print("Timed Cache:", cache_manager.get_stats("api_results"))


if __name__ == "__main__":
    print("Memory Management Example:")
    example_memory_management()

    print("\nCache Management Example:")
    example_cache_management()
