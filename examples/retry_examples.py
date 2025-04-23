"""Examples demonstrating the use of RetryHandler for resilient operations.

This module provides real-world examples of how to use the RetryHandler
to make your code more resilient against transient failures.
"""

import logging
import random
import time
from typing import Any

from safeguards.core.resilience import RetryableException, RetryHandler, RetryStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Example 1: Simple API call with retry
@RetryHandler(
    max_attempts=3,
    strategy=RetryStrategy.EXPONENTIAL,
    base_delay=1.0,
    jitter=0.2,
)
def fetch_user_data(user_id: str) -> dict[str, Any]:
    """Fetch user data from API with retry behavior.

    Args:
        user_id: User ID to fetch data for

    Returns:
        User data dictionary

    Raises:
        MaxRetriesExceeded: If maximum retry attempts are exceeded
    """
    url = f"https://api.example.com/users/{user_id}"

    # In a real application, this would be an actual API call
    # For demonstration, we'll simulate random failures
    if random.random() < 0.5:  # 50% chance of failure
        logger.warning(f"Simulating network error for {url}")
        msg = "Connection refused"
        raise ConnectionError(msg)

    # Simulate successful response
    logger.info(f"Successfully fetched data from {url}")
    return {"id": user_id, "name": "Test User", "email": "user@example.com"}


# Example 2: Database operation with custom retry behavior
class DatabaseClient:
    """Example database client with retry behavior."""

    def __init__(self, connection_string: str):
        """Initialize database client.

        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        self.connected = False

    def connect(self) -> bool:
        """Connect to database with retry behavior.

        Returns:
            True if connection successful
        """
        # Custom retry handler for connection attempts
        retry_handler = RetryHandler(
            max_attempts=5,
            strategy=RetryStrategy.LINEAR,  # Linear backoff for DB connections
            base_delay=2.0,
            retryable_exceptions=[ConnectionError, TimeoutError],
        )

        try:
            # Use as context manager
            with retry_handler:
                logger.info(f"Connecting to database: {self.connection_string}")

                # Simulate connection attempt that might fail
                if random.random() < 0.3:  # 30% chance of failure
                    logger.warning("Simulating database connection error")
                    msg = "Database connection failed"
                    raise ConnectionError(msg)

                self.connected = True
                logger.info("Database connection established")
                return True
        except Exception as e:
            logger.error(
                f"Failed to connect to database after {retry_handler.attempt} attempts: {e!s}",
            )
            self.connected = False
            return False

    @RetryHandler(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL)
    def query(self, sql: str) -> list:
        """Execute SQL query with retry behavior.

        Args:
            sql: SQL query to execute

        Returns:
            Query results

        Raises:
            ValueError: If not connected to database
            MaxRetriesExceeded: If maximum retry attempts are exceeded
        """
        if not self.connected:
            msg = "Not connected to database"
            raise ValueError(msg)

        logger.info(f"Executing query: {sql}")

        # Simulate random database timeout
        if random.random() < 0.4:  # 40% chance of timeout
            logger.warning("Simulating database timeout")
            msg = "Query timeout"
            raise TimeoutError(msg)

        # Simulate successful query
        logger.info("Query executed successfully")
        return [{"id": 1, "name": "Test"}]


# Example 3: Custom retry for specific exceptions
class PaymentProcessor:
    """Example payment processor with specialized retry behavior."""

    class PaymentGatewayError(RetryableException):
        """Payment gateway error that should be retried."""

        pass

    class PaymentDeclinedError(Exception):
        """Payment declined error that should not be retried."""

        pass

    @RetryHandler(
        max_attempts=4,
        strategy=RetryStrategy.EXPONENTIAL,
        base_delay=2.0,
        jitter=0.1,
        retryable_exceptions=["PaymentProcessor.PaymentGatewayError", ConnectionError],
    )
    def process_payment(self, amount: float, card_token: str) -> dict[str, Any]:
        """Process payment with specialized retry for gateway errors.

        Args:
            amount: Payment amount
            card_token: Card token for payment

        Returns:
            Payment receipt

        Raises:
            PaymentDeclinedError: If payment is declined (won't retry)
            MaxRetriesExceeded: If gateway errors persist beyond retries
        """
        logger.info(f"Processing payment of ${amount} with token {card_token}")

        # Simulate various payment scenarios
        rand_val = random.random()

        if rand_val < 0.3:  # 30% gateway error (will retry)
            logger.warning("Simulating payment gateway error")
            msg = "Gateway unavailable"
            raise self.PaymentGatewayError(msg)

        if rand_val < 0.5:  # 20% declined (won't retry)
            logger.warning("Simulating payment declined")
            msg = "Insufficient funds"
            raise self.PaymentDeclinedError(msg)

        # Successful payment
        logger.info("Payment processed successfully")
        return {
            "transaction_id": f"txn_{random.randint(10000, 99999)}",
            "amount": amount,
            "status": "completed",
            "timestamp": time.time(),
        }


def main():
    """Run the examples."""
    # Example 1: API call
    try:
        user_data = fetch_user_data("12345")
        print(f"User data: {user_data}")
    except Exception as e:
        print(f"Failed to fetch user data: {e!s}")

    # Example 2: Database operations
    db = DatabaseClient("postgres://example:password@localhost:5432/mydb")
    if db.connect():
        try:
            results = db.query("SELECT * FROM users LIMIT 10")
            print(f"Query results: {results}")
        except Exception as e:
            print(f"Query failed: {e!s}")

    # Example 3: Payment processing
    payment_processor = PaymentProcessor()
    try:
        receipt = payment_processor.process_payment(99.99, "card_token_12345")
        print(f"Payment receipt: {receipt}")
    except PaymentProcessor.PaymentDeclinedError as e:
        print(f"Payment declined: {e!s}")
    except Exception as e:
        print(f"Payment processing error: {e!s}")


if __name__ == "__main__":
    main()
