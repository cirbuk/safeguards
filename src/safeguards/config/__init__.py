"""Configuration module for Agent Safety Framework."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ..types import BudgetConfig, ResourceThresholds


@dataclass
class Config:
    """Global configuration for the Agent Safety Framework."""

    budget: BudgetConfig
    resources: ResourceThresholds
    log_level: str = "INFO"
    enable_monitoring: bool = True
    enable_notifications: bool = True
    notification_endpoints: list[dict[str, Any]] = field(default_factory=list)
    api_keys: dict[str, str] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration instance."""
        return cls(
            budget=BudgetConfig(
                total_budget=Decimal("1000"),
                hourly_limit=Decimal("100"),
                daily_limit=Decimal("500"),
                warning_threshold=75.0,
            ),
            resources=ResourceThresholds(
                cpu_percent=80.0,
                memory_percent=80.0,
                disk_percent=90.0,
                network_mbps=100.0,
            ),
        )
