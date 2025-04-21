"""Cost estimation module for FounderX."""

from decimal import Decimal
from typing import Dict, Optional, Union

from founderx.core.cost_management import CostType


class ModelCosts:
    """Constants for model costs."""

    # LLM Costs (per 1K tokens)
    GPT4_INPUT = Decimal("0.03")
    GPT4_OUTPUT = Decimal("0.06")
    GPT35_INPUT = Decimal("0.0015")
    GPT35_OUTPUT = Decimal("0.002")
    EMBEDDING = Decimal("0.0001")

    # Infrastructure Costs
    CHROMADB_STORAGE = Decimal("0.00001")  # per KB
    CHROMADB_QUERY = Decimal("0.0001")  # per query
    COMPUTE = Decimal("0.0001")  # per CPU second
    STORAGE = Decimal("0.00001")  # per KB

    # Social Media API Costs (per call)
    LINKEDIN_API = Decimal("0.001")
    TWITTER_API = Decimal("0.001")
    INSTAGRAM_API = Decimal("0.001")
    FACEBOOK_API = Decimal("0.001")

    # Content Generation Costs
    IMAGE_GEN = Decimal("0.02")  # per image
    VIDEO_GEN = Decimal("0.1")  # per minute

    # Analytics Costs
    ANALYTICS_API = Decimal("0.0005")  # per metric


class CostEstimator:
    """Cost estimation for various operations."""

    def estimate_llm_cost(
        self, input_tokens: int, model: str, output_tokens: Optional[int] = None
    ) -> Decimal:
        """Estimate cost for LLM usage.

        Args:
            input_tokens: Number of input tokens
            model: Model name (gpt-4 or gpt-3.5-turbo)
            output_tokens: Number of output tokens (optional, estimated if not provided)

        Returns:
            Estimated cost in USD
        """
        if output_tokens is None:
            # Estimate output tokens as half of input if not provided
            output_tokens = input_tokens // 2

        if model == "gpt-4":
            input_cost = Decimal(str(input_tokens)) / 1000 * ModelCosts.GPT4_INPUT
            output_cost = Decimal(str(output_tokens)) / 1000 * ModelCosts.GPT4_OUTPUT
        else:  # gpt-3.5-turbo
            input_cost = Decimal(str(input_tokens)) / 1000 * ModelCosts.GPT35_INPUT
            output_cost = Decimal(str(output_tokens)) / 1000 * ModelCosts.GPT35_OUTPUT

        return input_cost + output_cost

    def estimate_embedding_cost(self, num_tokens: int) -> Decimal:
        """Estimate cost for embedding generation.

        Args:
            num_tokens: Number of tokens to embed

        Returns:
            Estimated cost in USD
        """
        return Decimal(str(num_tokens)) / 1000 * ModelCosts.EMBEDDING

    def estimate_api_cost(
        self, api_type: CostType, num_calls: int, data_size: Optional[int] = None
    ) -> Decimal:
        """Estimate cost for API usage.

        Args:
            api_type: Type of API being used
            num_calls: Number of API calls
            data_size: Size of data in KB (optional)

        Returns:
            Estimated cost in USD
        """
        base_cost = Decimal("0.0001") * num_calls

        if data_size:
            storage_cost = Decimal(str(data_size)) * Decimal("0.00001")
            return base_cost + storage_cost

        return base_cost

    def estimate_infrastructure_cost(self, cost_type: CostType, usage: int) -> Decimal:
        """Estimate infrastructure costs.

        Args:
            cost_type: Type of infrastructure cost
            usage: Usage metric (depends on cost type)

        Returns:
            Estimated cost in USD
        """
        if cost_type == CostType.CHROMADB:
            return Decimal(str(usage)) * Decimal("0.0001")
        elif cost_type == CostType.COMPUTE:
            return Decimal(str(usage)) * Decimal("0.0001")
        return Decimal("0")

    def estimate_chromadb_cost(
        self, storage_kb: int = 0, num_queries: int = 0
    ) -> Decimal:
        """Estimate ChromaDB costs.

        Args:
            storage_kb: Storage size in KB
            num_queries: Number of vector queries

        Returns:
            Total estimated cost
        """
        storage_cost = Decimal(str(storage_kb)) * ModelCosts.CHROMADB_STORAGE
        query_cost = Decimal(str(num_queries)) * ModelCosts.CHROMADB_QUERY
        return storage_cost + query_cost

    def estimate_agent_operation(
        self, agent_type: CostType, params: Dict[str, Union[int, str, list, dict]]
    ) -> Decimal:
        """Estimate cost for agent operations.

        Args:
            agent_type: Type of agent operation
            params: Operation parameters

        Returns:
            Estimated cost in USD
        """
        total_cost = Decimal("0")

        if agent_type == CostType.SOCIAL_LISTENING:
            # Social API costs
            api_calls = params.get("api_calls", {})
            for platform, calls in api_calls.items():
                cost_per_call = getattr(ModelCosts, f"{platform}_API")
                total_cost += Decimal(str(calls)) * cost_per_call

            # Embedding and storage costs
            if "storage_kb" in params:
                total_cost += self.estimate_chromadb_cost(
                    storage_kb=params["storage_kb"],
                    num_queries=params.get("vector_queries", 0),
                )

        elif agent_type == CostType.TOPIC_SYNTHESIS:
            # LLM costs for research and writing
            total_cost += self.estimate_llm_cost(
                input_tokens=params.get("input_tokens", 1000),
                output_tokens=params.get("output_tokens", 2000),
                model=params.get("model", "gpt-4"),
            )

            # Vector store query costs
            if "vector_queries" in params:
                total_cost += self.estimate_chromadb_cost(
                    num_queries=params["vector_queries"]
                )

        elif agent_type == CostType.FORMATTING:
            # Content generation costs
            if "images" in params:
                total_cost += Decimal(str(params["images"])) * ModelCosts.IMAGE_GEN
            if "video_minutes" in params:
                total_cost += (
                    Decimal(str(params["video_minutes"])) * ModelCosts.VIDEO_GEN
                )

            # LLM costs for formatting
            total_cost += self.estimate_llm_cost(
                input_tokens=params.get("input_tokens", 500),
                output_tokens=params.get("output_tokens", 1000),
                model=params.get("model", "gpt-3.5-turbo"),
            )

        elif agent_type == CostType.PUBLISHING:
            # API costs for publishing
            platforms = params.get("platforms", [])
            for platform in platforms:
                cost_per_call = getattr(ModelCosts, f"{platform}_API")
                total_cost += cost_per_call

        elif agent_type == CostType.ANALYSIS:
            # Analytics API costs
            metrics = params.get("metrics", 0)
            total_cost += Decimal(str(metrics)) * ModelCosts.ANALYTICS_API

            # Storage costs for analytics data
            if "storage_kb" in params:
                total_cost += Decimal(str(params["storage_kb"])) * ModelCosts.STORAGE

        return total_cost

    def estimate_operation_cost(
        self, operation_type: str, params: Dict[str, Union[int, str, list, dict]]
    ) -> Decimal:
        """Estimate cost for complex operations."""
        total_cost = Decimal("0")

        if operation_type == "content_pipeline":
            # Social Listening
            total_cost += self.estimate_agent_operation(
                CostType.SOCIAL_LISTENING, params.get("social_listening", {})
            )

            # Topic Synthesis
            total_cost += self.estimate_agent_operation(
                CostType.TOPIC_SYNTHESIS, params.get("topic_synthesis", {})
            )

            # Formatting
            total_cost += self.estimate_agent_operation(
                CostType.FORMATTING, params.get("formatting", {})
            )

            # Publishing
            total_cost += self.estimate_agent_operation(
                CostType.PUBLISHING, params.get("publishing", {})
            )

            # Analysis
            total_cost += self.estimate_agent_operation(
                CostType.ANALYSIS, params.get("analysis", {})
            )

        return total_cost
