"""Cost estimation module for predicting agent operation costs."""

from dataclasses import dataclass
from decimal import Decimal

from ..base.budget import BudgetPeriod


@dataclass
class CostEstimate:
    """Represents a cost estimate with breakdown."""

    total_cost: Decimal
    breakdown: dict[str, Decimal]
    confidence_score: float  # 0-1 score indicating estimation confidence
    period: BudgetPeriod
    margin_of_error: float  # Percentage of potential variance


class CostEstimator:
    """Estimates costs for various agent operations."""

    def __init__(
        self,
        llm_costs: dict[str, Decimal],  # Model ID to cost per 1K tokens mapping
        api_costs: dict[str, Decimal],  # API endpoint to cost per call mapping
        storage_cost_per_gb: Decimal,
        compute_cost_per_hour: Decimal,
    ):
        """Initialize cost estimator.

        Args:
            llm_costs: Mapping of model IDs to their costs per 1K tokens
            api_costs: Mapping of API endpoints to their costs per call
            storage_cost_per_gb: Cost per GB of storage
            compute_cost_per_hour: Compute cost per hour
        """
        self.llm_costs = llm_costs
        self.api_costs = api_costs
        self.storage_cost_per_gb = storage_cost_per_gb
        self.compute_cost_per_hour = compute_cost_per_hour

    def estimate_llm_cost(
        self,
        model_id: str,
        estimated_tokens: int,
        is_output: bool = False,
    ) -> Decimal:
        """Estimate cost for LLM API calls.

        Args:
            model_id: ID of the LLM model
            estimated_tokens: Estimated number of tokens
            is_output: Whether these are output tokens (typically more expensive)

        Returns:
            Estimated cost in decimal
        """
        base_cost = self.llm_costs.get(model_id, Decimal("0"))
        # Output tokens typically cost more, apply multiplier if needed
        multiplier = Decimal("1.3") if is_output else Decimal("1.0")
        # Convert to per-token cost (costs are per 1K tokens)
        per_token_cost = base_cost / Decimal("1000")
        return per_token_cost * Decimal(str(estimated_tokens)) * multiplier

    def estimate_api_cost(
        self,
        endpoint: str,
        num_calls: int,
        data_size_mb: float | None = None,
    ) -> Decimal:
        """Estimate cost for API calls.

        Args:
            endpoint: API endpoint identifier
            num_calls: Estimated number of API calls
            data_size_mb: Optional data transfer size in MB

        Returns:
            Estimated cost in decimal
        """
        base_cost = self.api_costs.get(endpoint, Decimal("0"))
        cost = base_cost * Decimal(str(num_calls))

        # Add data transfer costs if applicable
        if data_size_mb:
            data_cost = Decimal("0.01") * Decimal(
                str(data_size_mb / 1000),
            )  # Cost per GB
            cost += data_cost

        return cost

    def estimate_storage_cost(
        self,
        size_gb: float,
        duration_hours: float,
    ) -> Decimal:
        """Estimate storage costs.

        Args:
            size_gb: Size in gigabytes
            duration_hours: Duration in hours

        Returns:
            Estimated cost in decimal
        """
        return (
            self.storage_cost_per_gb
            * Decimal(str(size_gb))
            * Decimal(str(duration_hours / 24))  # Convert to days
        )

    def estimate_compute_cost(
        self,
        cpu_hours: float,
        gpu_hours: float = 0.0,
    ) -> Decimal:
        """Estimate compute costs.

        Args:
            cpu_hours: CPU hours required
            gpu_hours: GPU hours required (typically more expensive)

        Returns:
            Estimated cost in decimal
        """
        cpu_cost = self.compute_cost_per_hour * Decimal(str(cpu_hours))
        gpu_cost = (
            self.compute_cost_per_hour * Decimal("10") * Decimal(str(gpu_hours))
        )  # GPU typically 10x more expensive
        return cpu_cost + gpu_cost

    def create_total_estimate(
        self,
        llm_usage: dict[str, int],  # model_id -> estimated tokens
        api_calls: dict[str, int],  # endpoint -> number of calls
        storage_gb: float,
        cpu_hours: float,
        gpu_hours: float = 0.0,
        period: BudgetPeriod = BudgetPeriod.DAILY,
    ) -> CostEstimate:
        """Create a comprehensive cost estimate.

        Args:
            llm_usage: Mapping of model IDs to estimated token usage
            api_calls: Mapping of API endpoints to estimated number of calls
            storage_gb: Storage requirements in GB
            cpu_hours: Estimated CPU hours
            gpu_hours: Estimated GPU hours
            period: Budget period for the estimate

        Returns:
            Complete cost estimate with breakdown
        """
        breakdown = {}

        # Calculate LLM costs
        llm_cost = Decimal("0")
        for model_id, tokens in llm_usage.items():
            input_cost = self.estimate_llm_cost(model_id, tokens, is_output=False)
            output_cost = self.estimate_llm_cost(
                model_id,
                tokens // 4,
                is_output=True,
            )  # Assume 4:1 input:output ratio
            llm_cost += input_cost + output_cost
            breakdown[f"llm_{model_id}"] = input_cost + output_cost

        # Calculate API costs
        api_cost = Decimal("0")
        for endpoint, calls in api_calls.items():
            endpoint_cost = self.estimate_api_cost(endpoint, calls)
            api_cost += endpoint_cost
            breakdown[f"api_{endpoint}"] = endpoint_cost

        # Calculate storage and compute costs
        storage_cost = self.estimate_storage_cost(storage_gb, cpu_hours)
        compute_cost = self.estimate_compute_cost(cpu_hours, gpu_hours)

        breakdown["storage"] = storage_cost
        breakdown["compute"] = compute_cost

        total_cost = llm_cost + api_cost + storage_cost + compute_cost

        # Calculate confidence score based on estimation factors
        confidence_score = self._calculate_confidence_score(
            llm_usage,
            api_calls,
            storage_gb,
            cpu_hours,
            gpu_hours,
        )

        # Calculate margin of error based on confidence
        margin_of_error = (1 - confidence_score) * 100

        return CostEstimate(
            total_cost=total_cost,
            breakdown=breakdown,
            confidence_score=confidence_score,
            period=period,
            margin_of_error=margin_of_error,
        )

    def _calculate_confidence_score(
        self,
        llm_usage: dict[str, int],
        api_calls: dict[str, int],
        storage_gb: float,
        cpu_hours: float,
        gpu_hours: float,
    ) -> float:
        """Calculate confidence score for the estimate.

        Args:
            llm_usage: LLM usage estimates
            api_calls: API call estimates
            storage_gb: Storage estimate
            cpu_hours: CPU hours estimate
            gpu_hours: GPU hours estimate

        Returns:
            Confidence score between 0 and 1
        """
        # Start with base confidence
        confidence = 0.8

        # Adjust based on estimation factors
        if sum(llm_usage.values()) > 1000000:  # Large token counts are less predictable
            confidence *= 0.9
        if sum(api_calls.values()) > 1000:  # Large API usage is less predictable
            confidence *= 0.9
        if storage_gb > 100:  # Large storage requirements are less predictable
            confidence *= 0.95
        if cpu_hours > 24 or gpu_hours > 24:  # Long compute times are less predictable
            confidence *= 0.9

        return max(0.5, confidence)  # Never go below 50% confidence
