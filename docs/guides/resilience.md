# Resilience Patterns Guide

This guide explains how to use the built-in resilience patterns in Safeguards to create robust, fault-tolerant applications that can gracefully handle transient failures.

## Overview

The Safeguards framework provides several resilience patterns to help your application handle failures:

1. **Retry Mechanisms** - Automatically retry operations that fail due to transient issues
2. **Circuit Breakers** - Prevent cascading failures by temporarily disabling failing services
3. **Timeouts** - Set maximum time limits for operations to prevent resource exhaustion
4. **Fallbacks** - Provide alternative pathways when primary operations fail

This guide focuses on the retry mechanism with exponential backoff, which is a critical strategy for handling transient failures.

## Retry Handler with Exponential Backoff

The `RetryHandler` class provides a simple way to add retry behavior to any operation that might fail due to transient issues.

### Basic Usage

The simplest way to use the `RetryHandler` is as a decorator:

```python
from safeguards.core.resilience import RetryHandler

@RetryHandler(max_attempts=3)
def fetch_data_from_api(url):
    # This function will be retried up to 3 times if it raises
    # a RetryableException, ConnectionError, or TimeoutError
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
```

### Configuring Retry Behavior

The `RetryHandler` is highly configurable:

```python
from safeguards.core.resilience import RetryHandler, RetryStrategy

@RetryHandler(
    max_attempts=5,               # Maximum number of attempts
    strategy=RetryStrategy.EXPONENTIAL,  # Use exponential backoff
    base_delay=1.0,               # Start with 1 second delay
    max_delay=30.0,               # Cap delay at 30 seconds
    jitter=0.1,                  # Add 10% random jitter to delay
    retryable_exceptions=[ConnectionError, TimeoutError, ValueError]
)
def fetch_data_from_api(url):
    # This will be retried with exponential backoff
    # if it raises any of the specified exceptions
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
```

### Retry Strategies

The `RetryHandler` supports three retry strategies:

1. **Fixed** - Uses the same delay between each retry attempt
2. **Linear** - Increases delay linearly between retry attempts
3. **Exponential** - Increases delay exponentially between retry attempts (recommended for most cases)

```python
from safeguards.core.resilience import RetryHandler, RetryStrategy

# Fixed delay of 2 seconds between each retry
@RetryHandler(
    max_attempts=3,
    strategy=RetryStrategy.FIXED,
    base_delay=2.0
)
def example_fixed():
    # Function implementation...

# Linear delay: 1s, 2s, 3s, 4s, ...
@RetryHandler(
    max_attempts=5,
    strategy=RetryStrategy.LINEAR,
    base_delay=1.0
)
def example_linear():
    # Function implementation...

# Exponential delay: 1s, 2s, 4s, 8s, 16s, ...
@RetryHandler(
    max_attempts=5,
    strategy=RetryStrategy.EXPONENTIAL,
    base_delay=1.0
)
def example_exponential():
    # Function implementation...
```

### Using as a Context Manager

For more control over retries, you can use `RetryHandler` as a context manager:

```python
from safeguards.core.resilience import RetryHandler

def process_data(data_url):
    retry_handler = RetryHandler(
        max_attempts=3,
        base_delay=1.0
    )

    try:
        with retry_handler:
            # Operations in this block will be retried
            response = requests.get(data_url)
            response.raise_for_status()
            data = response.json()
            return process_json_data(data)
    except Exception as e:
        # Handle non-retryable exceptions or max retries exceeded
        logger.error(f"Failed to process data after {retry_handler.attempt} attempts: {str(e)}")
        return None
```

### Custom Retryable Exceptions

You can define custom exception types that should trigger retries:

```python
from safeguards.core.resilience import RetryHandler, RetryableException

# Define a custom exception that should be retried
class APIRateLimitError(RetryableException):
    pass

# All RetryableException subclasses will be retried by default
@RetryHandler(max_attempts=5)
def call_rate_limited_api():
    if is_rate_limited():
        raise APIRateLimitError("Rate limit exceeded, try again later")
    return get_api_data()
```

Alternatively, you can specify exactly which exceptions should be retried:

```python
@RetryHandler(
    max_attempts=3,
    retryable_exceptions=[ConnectionError, APIRateLimitError]
)
def call_api():
    # Only ConnectionError and APIRateLimitError will trigger retries
    # Other exceptions will propagate immediately
    return get_api_data()
```

## Handling Retry Failures

When all retry attempts fail, the `RetryHandler` will raise a `MaxRetriesExceeded` exception:

```python
from safeguards.core.resilience import RetryHandler, MaxRetriesExceeded

@RetryHandler(max_attempts=3)
def fetch_data():
    # Implementation that might fail

try:
    data = fetch_data()
    process_data(data)
except MaxRetriesExceeded as e:
    # All retry attempts failed
    logger.error(f"Failed after {e.max_attempts} attempts: {str(e.last_exception)}")
    # Implement fallback behavior here
```

## Combining with Circuit Breakers

For maximum resilience, you can combine retry mechanisms with circuit breakers:

```python
from safeguards.core.resilience import RetryHandler
from safeguards.types.guardrail import CircuitBreakerGuardrail

# Create a circuit breaker
circuit_breaker = CircuitBreakerGuardrail(
    failure_threshold=3,
    reset_timeout_seconds=300  # 5 minutes
)

# Create a retry handler
@RetryHandler(max_attempts=3)
def call_service_with_retry():
    return circuit_breaker.run(actual_service_call)

def actual_service_call():
    # Actual implementation that might fail
    return requests.get("https://api.example.com/data")
```

## Best Practices

1. **Use exponential backoff with jitter** - This prevents "thundering herd" problems where all retries happen simultaneously
2. **Set reasonable maximum delays** - Balance retry attempts with overall operation latency
3. **Identify truly retryable operations** - Only retry idempotent operations or operations that can be safely repeated
4. **Add logging for retry attempts** - This helps diagnose issues during development and production
5. **Combine with circuit breakers** - For operations calling external services that might be down
6. **Set meaningful timeout values** - Ensure operations do not hang indefinitely
7. **Use fallbacks for critical functionality** - Have backup options when all retries fail

## Common Retry Scenarios

### HTTP/API Requests

```python
@RetryHandler(
    max_attempts=3,
    retryable_exceptions=[
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError  # For 5xx status codes
    ]
)
def call_api(url, params=None):
    response = requests.get(url, params=params, timeout=10)

    # Only retry 5xx server errors, not 4xx client errors
    if response.status_code >= 500:
        response.raise_for_status()  # Will raise HTTPError
    elif response.status_code >= 400:
        # Client errors should not be retried
        raise ValueError(f"Client error: {response.status_code}")

    return response.json()
```

### Database Operations

```python
@RetryHandler(
    max_attempts=5,
    strategy=RetryStrategy.EXPONENTIAL,
    base_delay=0.5,
    max_delay=10.0,
    retryable_exceptions=[
        sqlalchemy.exc.OperationalError,  # Connection/timeout issues
        sqlalchemy.exc.IntegrityError     # Deadlocks, etc.
    ]
)
def execute_database_query(session, query):
    try:
        result = session.execute(query)
        session.commit()
        return result
    except Exception as e:
        session.rollback()
        raise  # Re-raise for retry handler
```

### File Operations

```python
@RetryHandler(
    max_attempts=3,
    strategy=RetryStrategy.LINEAR,
    base_delay=1.0,
    retryable_exceptions=[
        IOError,
        OSError
    ]
)
def read_file_with_retry(file_path):
    with open(file_path, 'r') as f:
        return f.read()
```

## Conclusion

The `RetryHandler` provides a powerful yet simple way to add resilience to your application. By automatically retrying operations that fail due to transient issues, you can create more robust applications that can handle real-world failure scenarios gracefully.

For more advanced resilience patterns, see the other guides in this series:

- Circuit Breaker Pattern Guide
- Fallback Patterns Guide
- Timeout and Bulkhead Patterns Guide
