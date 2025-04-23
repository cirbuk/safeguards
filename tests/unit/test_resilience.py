"""Unit tests for resilience patterns."""

import unittest
from unittest import mock

from safeguards.core.resilience import (
    MaxRetriesExceeded,
    RetryableException,
    RetryHandler,
    RetryStrategy,
)


class TestRetryHandler(unittest.TestCase):
    """Tests for RetryHandler class."""

    def test_decorator_successful_first_try(self):
        """Test decorator with successful first try."""
        mock_fn = mock.Mock(return_value="success")
        decorated_fn = RetryHandler()(mock_fn)

        result = decorated_fn()

        self.assertEqual(result, "success")
        mock_fn.assert_called_once()

    def test_decorator_retry_and_succeed(self):
        """Test decorator with retry and eventual success."""
        # Mock function that fails twice then succeeds
        mock_fn = mock.Mock(
            side_effect=[
                RetryableException("First failure"),
                RetryableException("Second failure"),
                "success",
            ],
        )

        decorated_fn = RetryHandler(
            max_attempts=3,
            strategy=RetryStrategy.FIXED,
            base_delay=0.01,  # Fast for testing
        )(mock_fn)

        result = decorated_fn()

        self.assertEqual(result, "success")
        self.assertEqual(mock_fn.call_count, 3)

    def test_decorator_max_retries_exceeded(self):
        """Test decorator with max retries exceeded."""
        # Mock function that always fails
        mock_fn = mock.Mock(side_effect=RetryableException("Always fails"))

        decorated_fn = RetryHandler(
            max_attempts=2,
            strategy=RetryStrategy.FIXED,
            base_delay=0.01,  # Fast for testing
        )(mock_fn)

        with self.assertRaises(MaxRetriesExceeded) as context:
            decorated_fn()

        self.assertEqual(mock_fn.call_count, 2)
        self.assertEqual(context.exception.max_attempts, 2)
        self.assertIsInstance(context.exception.last_exception, RetryableException)

    def test_strategy_fixed(self):
        """Test fixed retry strategy."""
        handler = RetryHandler(
            strategy=RetryStrategy.FIXED,
            base_delay=1.0,
            jitter=0,  # Disable jitter for deterministic testing
        )

        handler.attempt = 1
        self.assertEqual(handler._calculate_delay(), 1.0)

        handler.attempt = 2
        self.assertEqual(handler._calculate_delay(), 1.0)

    def test_strategy_linear(self):
        """Test linear retry strategy."""
        handler = RetryHandler(
            strategy=RetryStrategy.LINEAR,
            base_delay=1.0,
            jitter=0,  # Disable jitter for deterministic testing
        )

        handler.attempt = 1
        self.assertEqual(handler._calculate_delay(), 1.0)

        handler.attempt = 2
        self.assertEqual(handler._calculate_delay(), 2.0)

        handler.attempt = 3
        self.assertEqual(handler._calculate_delay(), 3.0)

    def test_strategy_exponential(self):
        """Test exponential retry strategy."""
        handler = RetryHandler(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            jitter=0,  # Disable jitter for deterministic testing
        )

        handler.attempt = 1
        self.assertEqual(handler._calculate_delay(), 1.0)

        handler.attempt = 2
        self.assertEqual(handler._calculate_delay(), 2.0)

        handler.attempt = 3
        self.assertEqual(handler._calculate_delay(), 4.0)

        handler.attempt = 4
        self.assertEqual(handler._calculate_delay(), 8.0)

    def test_max_delay(self):
        """Test max delay cap."""
        handler = RetryHandler(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=2.0,
            max_delay=5.0,
            jitter=0,  # Disable jitter for deterministic testing
        )

        handler.attempt = 1
        self.assertEqual(handler._calculate_delay(), 2.0)

        handler.attempt = 2
        self.assertEqual(handler._calculate_delay(), 4.0)

        handler.attempt = 3
        self.assertEqual(handler._calculate_delay(), 5.0)  # Should cap at max_delay

    def test_should_retry_with_retryable_exception(self):
        """Test should_retry with RetryableException."""
        handler = RetryHandler()

        # Should retry for RetryableException
        self.assertTrue(handler._should_retry(RetryableException()))

        # Should retry for ConnectionError and TimeoutError
        self.assertTrue(handler._should_retry(ConnectionError()))
        self.assertTrue(handler._should_retry(TimeoutError()))

        # Should not retry for other exceptions
        self.assertFalse(handler._should_retry(ValueError()))

    def test_custom_retryable_exceptions(self):
        """Test custom retryable exceptions."""
        handler = RetryHandler(retryable_exceptions=[ValueError, KeyError])

        # Should retry for specified exceptions
        self.assertTrue(handler._should_retry(ValueError()))
        self.assertTrue(handler._should_retry(KeyError()))

        # Should not retry for default or other exceptions
        self.assertFalse(handler._should_retry(RetryableException()))
        self.assertFalse(handler._should_retry(TypeError()))

    def test_context_manager_successful(self):
        """Test context manager with successful operation."""
        with RetryHandler():
            result = 1 + 1

        self.assertEqual(result, 2)

    def test_context_manager_retry_and_succeed(self):
        """Test context manager with retry and eventual success."""
        mock_fn = mock.Mock(
            side_effect=[
                RetryableException("First failure"),
                RetryableException("Second failure"),
                "success",
            ],
        )

        # Create a retry handler function specifically for this test
        def run_with_retries():
            handler = RetryHandler(
                max_attempts=5,
                strategy=RetryStrategy.FIXED,
                base_delay=0.01,  # Fast for testing
            )

            attempt = 0
            while attempt < 5:  # Limit to 5 iterations max
                try:
                    with handler:
                        return mock_fn()
                except RetryableException:
                    # This will be caught by the context manager
                    attempt += 1
                    continue
                except MaxRetriesExceeded:
                    # Context manager should raise this when max retries exceeded
                    return "max retries exceeded"
                except Exception as e:
                    # Any other exception
                    return f"unexpected error: {e!s}"

        # Run the test function
        result = run_with_retries()

        # Should get 'success' after 2 retries (3rd call)
        self.assertEqual(result, "success")
        self.assertEqual(mock_fn.call_count, 3)

    def test_context_manager_max_retries_exceeded(self):
        """Test context manager with max retries exceeded."""

        # Create a function that always raises a RetryableException
        def always_fail():
            raise RetryableException("Always fails")

        # Create a retry handler with max 3 attempts
        handler = RetryHandler(
            max_attempts=3,
            strategy=RetryStrategy.FIXED,
            base_delay=0.01,  # Fast for testing
        )

        retry_count = 0

        # Try to run with retries, should raise MaxRetriesExceeded after 3 attempts
        try:
            while True:
                try:
                    with handler:
                        retry_count += 1
                        always_fail()
                        return "success"  # Should never get here
                except MaxRetriesExceeded:
                    raise  # Re-raise to be caught by outer try
                except RetryableException:
                    # Normal flow - will retry
                    continue
        except MaxRetriesExceeded as e:
            # Should get here after max retries
            self.assertEqual(retry_count, 3)
            self.assertEqual(e.max_attempts, 3)
            self.assertIsInstance(e.last_exception, RetryableException)


if __name__ == "__main__":
    unittest.main()
