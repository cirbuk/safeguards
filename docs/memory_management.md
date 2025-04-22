# Memory Management Guide

## Overview

The Safeguards includes comprehensive memory management and caching features to optimize resource usage and improve performance. This guide covers the key components and their usage.

## Memory Manager

The `MemoryManager` class provides centralized memory optimization and resource tracking:

```python
from safeguards.core.memory_manager import MemoryManager

# Initialize memory manager
memory_manager = MemoryManager(gc_threshold=(700, 10, 10))

# Create object pool
pool = memory_manager.create_pool(
    name="request_pool",
    factory=lambda: Request(),
    max_size=100
)

# Use pooled objects
obj = pool.acquire()
try:
    # Use object
    process_request(obj)
finally:
    pool.release(obj)

# Track resources
resource = create_resource()
memory_manager.track_resource(resource)

# Cleanup
memory_manager.cleanup_resources()
```

### Features

1. **Object Pooling**
   - Reuse objects to reduce allocation overhead
   - Thread-safe pool operations
   - Configurable pool sizes

2. **Resource Tracking**
   - Automatic cleanup of unused resources
   - Weak reference tracking
   - Asynchronous cleanup

3. **Cache Management**
   - Namespace-based caching
   - Cache statistics tracking
   - Selective cache clearing

## Cache Manager

The `CacheManager` provides advanced caching strategies:

```python
from safeguards.core.cache_manager import CacheManager

cache_manager = CacheManager()

# LRU Cache
cache_manager.create_lru_cache("results", capacity=1000)
cache_manager.put_in_cache("lru", "results", key, value)
result = cache_manager.get_from_cache("lru", "results", key)

# Timed Cache
cache_manager.create_timed_cache("api_results", ttl_seconds=300)
cache_manager.put_in_cache("timed", "api_results", key, value)
result = cache_manager.get_from_cache("timed", "api_results", key)

# Function Memoization
@cache_manager.memoize(ttl_seconds=60)
def expensive_operation(x, y):
    return x + y
```

### Caching Strategies

1. **LRU (Least Recently Used)**
   - Maintains most recently used items
   - Fixed capacity
   - Thread-safe operations

2. **Timed Cache**
   - Time-based expiration
   - Automatic cleanup of expired entries
   - Configurable TTL

3. **Function Memoization**
   - Automatic caching of function results
   - Support for both LRU and timed caching
   - Handles complex argument types

## Best Practices

1. **Memory Management**
   - Use object pools for frequently allocated objects
   - Track resources that need cleanup
   - Configure GC thresholds based on application needs

2. **Caching**
   - Choose appropriate cache type (LRU vs Timed)
   - Monitor cache statistics
   - Set reasonable capacities and TTLs
   - Use memoization for expensive computations

3. **Resource Cleanup**
   - Always release pooled objects
   - Regularly call cleanup methods
   - Monitor memory usage

## Configuration

### Memory Manager

```python
MemoryManager(
    gc_threshold=(700, 10, 10)  # Optional GC thresholds
)
```

### Cache Types

1. **LRU Cache**
   ```python
   create_lru_cache(
       name="cache_name",
       capacity=1000  # Maximum items
   )
   ```

2. **Timed Cache**
   ```python
   create_timed_cache(
       name="cache_name",
       ttl_seconds=300  # Time to live
   )
   ```

## Monitoring

### Cache Statistics

```python
# Get cache stats
stats = cache_manager.get_stats("cache_name")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
```

### Memory Usage

- Monitor object pool utilization
- Track cache sizes
- Watch for memory leaks
- Use system monitoring tools

## Error Handling

- Handle pool exhaustion
- Manage cache misses
- Implement retry mechanisms
- Log memory issues

## Examples

See the [examples](examples/) directory for:
- Object pool usage patterns
- Caching strategies
- Resource cleanup
- Performance optimization
