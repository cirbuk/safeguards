"""Token usage tracking and cost calculation for LLM operations.

This module provides functionality for:
- Tracking input and output tokens for LLM calls
- Calculating costs based on model-specific pricing
- Monitoring embedding token usage
- Aggregating token statistics
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional

from founderx.core.cost_management import CostType
from founderx.core.cost_tracker import CostTracker


@dataclass
class ModelRates:
    """Pricing rates for different LLM operations."""

    input_price: Decimal  # Price per 1K input tokens
    output_price: Decimal  # Price per 1K output tokens
    embedding_price: Decimal  # Price per 1K tokens for embeddings


class TokenUsageTracker:
    """Tracks token usage and calculates costs for LLM operations."""

    # Default pricing for different models (per 1K tokens)
    DEFAULT_RATES = {
        "gpt-4": ModelRates(
            input_price=Decimal("0.03"),
            output_price=Decimal("0.06"),
            embedding_price=Decimal("0.0"),
        ),
        "gpt-3.5-turbo": ModelRates(
            input_price=Decimal("0.0015"),
            output_price=Decimal("0.002"),
            embedding_price=Decimal("0.0"),
        ),
        "text-embedding-ada-002": ModelRates(
            input_price=Decimal("0.0"),
            output_price=Decimal("0.0"),
            embedding_price=Decimal("0.0001"),
        ),
    }

    def __init__(
        self,
        cost_tracker: CostTracker,
        model_rates: Optional[Dict[str, ModelRates]] = None,
    ):
        """Initialize token tracker.

        Args:
            cost_tracker: Cost tracker instance for recording costs
            model_rates: Optional custom pricing rates for models
        """
        self.cost_tracker = cost_tracker
        self.model_rates = model_rates or self.DEFAULT_RATES
        self._reset_counters()

    def _reset_counters(self) -> None:
        """Reset token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_embedding_tokens = 0
        self.model_usage: Dict[str, Dict[str, int]] = {}

    def track_completion(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        record_cost: bool = True,
    ) -> Decimal:
        """Track token usage for a completion call.

        Args:
            model: Name of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            record_cost: Whether to record the cost

        Returns:
            Total cost for this operation

        Raises:
            ValueError: If model is not recognized
        """
        if model not in self.model_rates:
            raise ValueError(f"Unknown model: {model}")

        # Update counters
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Update per-model usage
        if model not in self.model_usage:
            self.model_usage[model] = {"input": 0, "output": 0, "embedding": 0}
        self.model_usage[model]["input"] += input_tokens
        self.model_usage[model]["output"] += output_tokens

        # Calculate costs
        rates = self.model_rates[model]
        input_cost = (Decimal(input_tokens) / 1000) * rates.input_price
        output_cost = (Decimal(output_tokens) / 1000) * rates.output_price
        total_cost = input_cost + output_cost

        if record_cost:
            # Record input and output costs separately
            self.cost_tracker.add_cost(CostType.LLM_INPUT, input_cost)
            self.cost_tracker.add_cost(CostType.LLM_OUTPUT, output_cost)

        return total_cost

    def track_embedding(
        self, model: str, token_count: int, record_cost: bool = True
    ) -> Decimal:
        """Track token usage for embedding generation.

        Args:
            model: Name of the model used
            token_count: Number of tokens embedded
            record_cost: Whether to record the cost

        Returns:
            Total cost for this operation

        Raises:
            ValueError: If model is not recognized
        """
        if model not in self.model_rates:
            raise ValueError(f"Unknown model: {model}")

        # Update counters
        self.total_embedding_tokens += token_count

        # Update per-model usage
        if model not in self.model_usage:
            self.model_usage[model] = {"input": 0, "output": 0, "embedding": 0}
        self.model_usage[model]["embedding"] += token_count

        # Calculate cost
        rates = self.model_rates[model]
        cost = (Decimal(token_count) / 1000) * rates.embedding_price

        if record_cost:
            self.cost_tracker.add_cost(CostType.EMBEDDING, cost)

        return cost

    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get token usage statistics.

        Returns:
            Dictionary containing token usage statistics
        """
        return {
            "total": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "embedding_tokens": self.total_embedding_tokens,
            },
            "per_model": self.model_usage,
        }

    def estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int = 0
    ) -> Decimal:
        """Estimate cost for a potential LLM operation.

        Args:
            model: Name of the model to use
            input_tokens: Number of input tokens
            output_tokens: Number of expected output tokens

        Returns:
            Estimated cost for the operation

        Raises:
            ValueError: If model is not recognized
        """
        if model not in self.model_rates:
            raise ValueError(f"Unknown model: {model}")

        rates = self.model_rates[model]
        input_cost = (Decimal(input_tokens) / 1000) * rates.input_price
        output_cost = (Decimal(output_tokens) / 1000) * rates.output_price
        return input_cost + output_cost
