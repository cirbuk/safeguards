"""Resilience patterns for handling transient failures.

This module provides implementation of common resilience patterns:
1. RetryHandler - For retrying operations with exponential backoff
2. CircuitBreaker - For preventing cascading failures (imported if used)

These patterns can be used together or separately to improve the reliability
of operations that might fail due to transient issues.
"""

import logging
import random
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any, TypeVar, cast

T = TypeVar("T")  # Return type of the wrapped function

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Strategies for retrying operations."""

    FIXED = "fixed"  # Fixed delay between retries
    LINEAR = "linear"  # Linearly increasing delay
    EXPONENTIAL = "exponential"  # Exponentially increasing delay


class RetryableException(Exception):
    """Base class for exceptions that should trigger a retry."""

    pass


class MaxRetriesExceeded(Exception):
    """Exception raised when maximum retry attempts are exceeded."""

    def __init__(self, max_attempts: int, last_exception: Exception | None = None):
        """Initialize with max attempts and the last exception encountered.

        Args:
            max_attempts: Maximum number of retry attempts
            last_exception: The last exception that triggered a retry
        """
        self.max_attempts = max_attempts
        self.last_exception = last_exception
        message = f"Maximum retry attempts ({max_attempts}) exceeded"
        if last_exception:
            message += f": {last_exception!s}"
        super().__init__(message)


class RetryHandler:
    """Handles retrying operations with configurable backoff.

    This class provides both a context manager and a decorator for retrying
    operations that might fail due to transient issues.

    Example usage as a decorator:
        ```python
        @RetryHandler(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL)
        def call_external_api(url):
            return requests.get(url)
        ```

    Example usage as a context manager:
        ```python
        with RetryHandler(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL):
            response = requests.get(url)
        ```
    """

    def __init__(
        self,
        max_attempts: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: float = 0.1,
        retryable_exceptions: list[type[Exception]] | None = None,
    ):
        """Initialize retry handler with configurable parameters.

        Args:
            max_attempts: Maximum number of retry attempts
            strategy: Retry timing strategy
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Random jitter factor (0-1) to add to delay
            retryable_exceptions: List of exception types that should trigger retries
        """
        self.max_attempts = max_attempts
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            RetryableException,
            ConnectionError,
            TimeoutError,
        ]
        self.attempt = 0
        self.last_exception: Exception | None = None

    def _calculate_delay(self) -> float:
        """Calculate retry delay based on strategy and current attempt.

        Returns:
            Delay in seconds
        """
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (self.attempt)
        else:  # EXPONENTIAL
            delay = self.base_delay * (2 ** (self.attempt - 1))

        # Apply jitter
        jitter_amount = delay * self.jitter
        delay += random.uniform(-jitter_amount, jitter_amount)

        # Ensure delay is not negative and does not exceed max_delay
        return max(0.1, min(delay, self.max_delay))

    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry.

        Args:
            exception: The exception to check

        Returns:
            True if the exception should trigger a retry, False otherwise
        """
        # Handle cases where retryable_exceptions might include strings
        # (for when exceptions are defined as inner classes)
        if isinstance(self.retryable_exceptions, list):
            for exc_type in self.retryable_exceptions:
                if isinstance(exc_type, str):
                    # Try to evaluate exception class from string (e.g., 'MyClass.InnerException')
                    continue  # Skip string entries for now
                if isinstance(exception, exc_type):
                    return True
        # Default list check
        return any(
            isinstance(exception, exc_type)
            for exc_type in self.retryable_exceptions
            if not isinstance(exc_type, str)
        )

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorate a function to automatically retry on failures.

        Args:
            func: The function to decorate

        Returns:
            Decorated function with retry logic
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            self.attempt = 0
            while True:
                self.attempt += 1
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.last_exception = e
                    if self.attempt >= self.max_attempts or not self._should_retry(e):
                        if self.attempt >= self.max_attempts:
                            raise MaxRetriesExceeded(
                                self.max_attempts,
                                self.last_exception,
                            ) from e
                        raise e

                    delay = self._calculate_delay()
                    logger.warning(
                        f"Retry {self.attempt}/{self.max_attempts} after {delay:.2f}s "
                        f"due to: {e!s}",
                    )
                    time.sleep(delay)

        return wrapper

    def __enter__(self) -> "RetryHandler":
        """Enter context manager.

        Returns:
            Self
        """
        # Don't reset attempt counter here, as this might be re-entered in a loop
        # Only increment attempt on re-entry if we've had an exception
        if not hasattr(self, "_entered_before") or not self._entered_before:
            self.attempt = 0
            self._entered_before = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context manager.

        Args:
            exc_type: Type of exception raised
            exc_val: Exception instance raised
            exc_tb: Traceback

        Returns:
            True if exception was handled, False otherwise
        """
        if exc_type is None:
            # No exception, this was a successful attempt
            self._entered_before = False  # Reset for future use
            return False

        exc = cast(Exception, exc_val)
        self.last_exception = exc

        # Check if we should retry this exception
        if not self._should_retry(exc):
            return False

        # Increment attempt counter for the next try
        self.attempt += 1

        # Check if we've exceeded max attempts
        if self.attempt >= self.max_attempts:
            # Reset for future use
            self._entered_before = False
            # Replace with MaxRetriesExceeded
            raise MaxRetriesExceeded(self.max_attempts, self.last_exception) from exc

        # Calculate delay and wait before next retry
        delay = self._calculate_delay()
        logger.warning(
            f"Retry {self.attempt}/{self.max_attempts} after {delay:.2f}s due to: {exc!s}",
        )
        time.sleep(delay)

        # Tell Python to suppress the exception, allowing another context entry
        return True
