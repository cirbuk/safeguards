"""Cost management module for FounderX."""

from enum import Enum, auto


class CostType(Enum):
    """Types of costs that can be estimated."""

    # Infrastructure Costs
    CHROMADB = auto()  # Vector store costs using ChromaDB
    COMPUTE = auto()  # General compute costs
    STORAGE = auto()  # Storage costs for media, embeddings, etc.

    # LLM API Costs
    LLM_INPUT = auto()  # Input token costs for LLM calls
    LLM_OUTPUT = auto()  # Output token costs for LLM calls
    EMBEDDING = auto()  # Embedding generation costs

    # Social Media API Costs
    LINKEDIN_API = auto()  # LinkedIn API usage
    TWITTER_API = auto()  # Twitter API usage
    INSTAGRAM_API = auto()  # Instagram API usage
    FACEBOOK_API = auto()  # Facebook API usage

    # Content Generation Costs
    IMAGE_GENERATION = auto()  # Image generation API costs
    VIDEO_GENERATION = auto()  # Video generation API costs

    # Analytics and Monitoring
    ANALYTICS_API = auto()  # Analytics API costs
    MONITORING = auto()  # System monitoring costs

    # Agent-Specific Costs
    SOCIAL_LISTENING = auto()  # Social listening agent costs
    TOPIC_SYNTHESIS = auto()  # Topic synthesis agent costs
    FORMATTING = auto()  # Formatting agent costs
    PUBLISHING = auto()  # Publishing agent costs
    ANALYSIS = auto()  # Analysis agent costs


class BudgetPeriod(Enum):
    """Time periods for budget tracking and alerts."""

    HOURLY = auto()  # Track costs on an hourly basis
    DAILY = auto()  # Track costs on a daily basis
    WEEKLY = auto()  # Track costs on a weekly basis
    MONTHLY = auto()  # Track costs on a monthly basis
    QUARTERLY = auto()  # Track costs on a quarterly basis
    YEARLY = auto()  # Track costs on a yearly basis
    CUSTOM = auto()  # Custom time period for budget tracking
