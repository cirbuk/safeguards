#!/usr/bin/env python
"""
Advanced integration of the Agent Safety Framework with the OpenAI Agents SDK.

This example demonstrates:
1. Multi-agent orchestration with priority-based resource allocation
2. Content filtering and policy enforcement
3. Dynamic budget reallocation based on agent performance
4. Comprehensive metrics collection and reporting
5. Advanced violation handling with custom response strategies
"""

import os
import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Import OpenAI Agents SDK components
from agents_sdk import Agent, RunSpec, AgentResponse
from agents_sdk.models import CompletionRequest, Message

# Import Agent Safety Framework components
from agent_safety.notifications import NotificationManager, NotificationLevel
from agent_safety.violations import ViolationReporter, ViolationType, Violation
from agent_safety.core import BudgetCoordinator, BudgetPool
from agent_safety.metrics import MetricsCollector
from agent_safety.types import Agent as SafetyAgent
from agent_safety.guardrails import ContentFilter, TokenUsageGuardrail
from agent_safety.policies import PolicySet, Policy, Rule, Action

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gpt-4o"
TOKEN_COST_PER_1K = {
    "gpt-4o": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

# Content categories for filtering
SENSITIVE_TOPICS = [
    "personal data",
    "financial information",
    "health information",
    "political views",
    "religious beliefs",
    "illegal activities",
]


class ContentGuardrail:
    """
    Guardrail for filtering sensitive content in agent inputs and outputs.
    """

    def __init__(
        self, violation_reporter: ViolationReporter, sensitive_topics: List[str]
    ):
        self.violation_reporter = violation_reporter
        self.sensitive_topics = sensitive_topics

    def check_content(self, agent_id: str, content: str) -> Tuple[bool, Optional[str]]:
        """
        Check if content contains sensitive topics.

        Returns:
            Tuple of (is_safe, filtered_content)
        """
        lower_content = content.lower()

        # Check for sensitive topics
        detected_topics = []
        for topic in self.sensitive_topics:
            if topic.lower() in lower_content:
                detected_topics.append(topic)

        if detected_topics:
            # Create a violation report
            violation = Violation(
                type=ViolationType.CONTENT,
                description=f"Detected sensitive topics: {', '.join(detected_topics)}",
                agent_id=agent_id,
                severity=NotificationLevel.WARNING,
                context={"original_content": content, "topics": detected_topics},
            )
            self.violation_reporter.report(violation)

            # Return filtered content
            filtered_content = content
            for topic in detected_topics:
                # Replace the topic with [REDACTED]
                filtered_content = filtered_content.replace(topic, "[REDACTED]")

            return False, filtered_content

        return True, content


class EnhancedOpenAIAgentWrapper(SafetyAgent):
    """
    Enhanced wrapper for OpenAI agent with advanced safety features.
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: str = DEFAULT_MODEL,
        budget_coordinator: Optional[BudgetCoordinator] = None,
        violation_reporter: Optional[ViolationReporter] = None,
        content_guardrail: Optional[ContentGuardrail] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        token_limit: int = 4000,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.budget_coordinator = budget_coordinator
        self.violation_reporter = violation_reporter
        self.content_guardrail = content_guardrail
        self.metrics_collector = metrics_collector
        self.token_limit = token_limit

        # Create the OpenAI agent
        self.agent = Agent(name=name)

        # Track token usage and performance
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent with the provided query, with safety guardrails.
        """
        self.request_count += 1

        # Apply content filtering to the input
        if self.content_guardrail:
            is_safe, filtered_query = self.content_guardrail.check_content(
                self.id, query
            )
            if not is_safe:
                logger.warning(
                    f"Input query for agent {self.name} filtered for sensitive content"
                )
                query = filtered_query

        # Check if we're under budget before running
        if self.budget_coordinator:
            metrics = self.budget_coordinator.get_agent_metrics(self.id)
            if metrics.remaining_budget <= 0:
                # Report budget violation
                violation = Violation(
                    type=ViolationType.BUDGET,
                    description=f"Agent {self.name} has no remaining budget",
                    agent_id=self.id,
                    severity=NotificationLevel.ERROR,
                )
                if self.violation_reporter:
                    self.violation_reporter.report(violation)

                self.error_count += 1
                return {
                    "error": "No budget remaining",
                    "response": "Unable to process request due to budget limitations.",
                    "tokens": {"input": 0, "output": 0},
                    "model": self.model,
                }

        try:
            # Prepare the run specification
            run_spec = RunSpec(
                messages=[{"role": "user", "content": query}], model=self.model
            )

            # Run the agent
            response: AgentResponse = await self.agent.run(run_spec)
            response_text = response.message.content if response.message else ""

            # Apply content filtering to the output
            if self.content_guardrail and response_text:
                is_safe, filtered_response = self.content_guardrail.check_content(
                    self.id, response_text
                )
                if not is_safe:
                    logger.warning(
                        f"Output from agent {self.name} filtered for sensitive content"
                    )
                    response_text = filtered_response

            # Extract token usage
            usage = response.usage
            if usage:
                self.total_input_tokens += usage.prompt_tokens
                self.total_output_tokens += usage.completion_tokens

                # Calculate cost
                cost = self._calculate_cost(
                    usage.prompt_tokens, usage.completion_tokens
                )
                self.total_cost += cost

                # Update budget if available
                if self.budget_coordinator:
                    self.budget_coordinator.update_usage(self.id, cost)

                    # Log budget information
                    metrics = self.budget_coordinator.get_agent_metrics(self.id)
                    logger.info(
                        f"Agent {self.name} used ${cost:.6f}, "
                        f"remaining budget: ${metrics.remaining_budget:.6f}"
                    )

                # Check for token limit violations
                if usage.prompt_tokens + usage.completion_tokens > self.token_limit:
                    # Report token limit violation
                    violation = Violation(
                        type=ViolationType.TOKEN_LIMIT,
                        description=f"Agent {self.name} exceeded token limit of {self.token_limit}",
                        agent_id=self.id,
                        severity=NotificationLevel.WARNING,
                        context={
                            "token_limit": self.token_limit,
                            "tokens_used": usage.prompt_tokens
                            + usage.completion_tokens,
                        },
                    )
                    if self.violation_reporter:
                        self.violation_reporter.report(violation)

            # Update metrics
            if self.metrics_collector:
                self.metrics_collector.track_agent_usage(
                    agent_id=self.id,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    cost=cost if usage else 0,
                    model=self.model,
                )

            self.success_count += 1

            # Return the result
            return {
                "response": response_text,
                "tokens": {
                    "input": usage.prompt_tokens if usage else 0,
                    "output": usage.completion_tokens if usage else 0,
                },
                "model": self.model,
                "cost": cost if usage else 0,
            }

        except Exception as e:
            logger.error(f"Error running agent {self.name}: {str(e)}")
            self.error_count += 1

            # Report the error
            if self.violation_reporter:
                violation = Violation(
                    type=ViolationType.OPERATIONAL,
                    description=f"Error running agent {self.name}: {str(e)}",
                    agent_id=self.id,
                    severity=NotificationLevel.ERROR,
                    context={"error": str(e)},
                )
                self.violation_reporter.report(violation)

            return {
                "error": str(e),
                "response": "An error occurred while processing your request.",
                "tokens": {"input": 0, "output": 0},
                "model": self.model,
            }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost based on token usage.
        """
        if self.model not in TOKEN_COST_PER_1K:
            # Default to gpt-4o pricing if model not found
            model_costs = TOKEN_COST_PER_1K[DEFAULT_MODEL]
        else:
            model_costs = TOKEN_COST_PER_1K[self.model]

        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]

        return input_cost + output_cost

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this agent.
        """
        return {
            "name": self.name,
            "id": self.id,
            "request_count": self.request_count,
            "success_rate": (self.success_count / self.request_count)
            if self.request_count > 0
            else 0,
            "total_tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_input_tokens + self.total_output_tokens,
            },
            "total_cost": self.total_cost,
            "avg_cost_per_request": (self.total_cost / self.request_count)
            if self.request_count > 0
            else 0,
        }


class AgentOrchestrator:
    """
    Orchestrates multiple agents with dynamic budget allocation.
    """

    def __init__(
        self,
        budget_coordinator: BudgetCoordinator,
        violation_reporter: ViolationReporter,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self.budget_coordinator = budget_coordinator
        self.violation_reporter = violation_reporter
        self.metrics_collector = metrics_collector
        self.agents: Dict[str, EnhancedOpenAIAgentWrapper] = {}

    def register_agent(
        self,
        agent: EnhancedOpenAIAgentWrapper,
        pool_id: str,
        initial_budget: float,
        priority: int,
    ) -> str:
        """
        Register an agent with the orchestrator.
        """
        self.agents[agent.id] = agent

        # Register with budget coordinator
        self.budget_coordinator.register_agent(
            agent_id=agent.id,
            pool_id=pool_id,
            initial_budget=initial_budget,
            priority=priority,
        )

        return agent.id

    async def run_agent(self, agent_id: str, query: str) -> Dict[str, Any]:
        """
        Run a specific agent.
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent with ID {agent_id} not found")

        return await self.agents[agent_id].run(query)

    async def run_parallel(self, queries: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple agents in parallel.
        """
        tasks = {}
        for agent_id, query in queries.items():
            if agent_id in self.agents:
                tasks[agent_id] = asyncio.create_task(self.agents[agent_id].run(query))

        # Wait for all tasks to complete
        results = {}
        for agent_id, task in tasks.items():
            try:
                results[agent_id] = await task
            except Exception as e:
                logger.error(f"Error running agent {agent_id}: {str(e)}")
                results[agent_id] = {"error": str(e)}

        return results

    def rebalance_budgets(self) -> None:
        """
        Rebalance budgets based on agent performance.
        """
        # Get performance metrics for all agents
        metrics = {}
        for agent_id, agent in self.agents.items():
            metrics[agent_id] = agent.get_performance_metrics()

        # Calculate success rates
        success_rates = {
            agent_id: data["success_rate"] for agent_id, data in metrics.items()
        }

        # Calculate cost efficiency (lower is better)
        cost_efficiency = {
            agent_id: data["avg_cost_per_request"]
            if data["success_rate"] > 0
            else float("inf")
            for agent_id, data in metrics.items()
        }

        # Rank agents by success rate and cost efficiency
        success_rank = {
            agent_id: rank
            for rank, (agent_id, _) in enumerate(
                sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
            )
        }

        efficiency_rank = {
            agent_id: rank
            for rank, (agent_id, _) in enumerate(
                sorted(cost_efficiency.items(), key=lambda x: x[1])
            )
        }

        # Combined rank (lower is better)
        combined_rank = {
            agent_id: success_rank[agent_id] + efficiency_rank[agent_id]
            for agent_id in self.agents.keys()
        }

        # Sort agents by combined rank
        sorted_agents = sorted(combined_rank.items(), key=lambda x: x[1])

        # Adjust priorities based on rank
        for i, (agent_id, rank) in enumerate(sorted_agents):
            # Scale priority from 1-10 based on position
            new_priority = 10 - min(9, int((i / len(sorted_agents)) * 10))

            # Update agent priority
            current_priority = self.budget_coordinator.get_agent_priority(agent_id)
            if current_priority != new_priority:
                logger.info(
                    f"Adjusting priority for agent {self.agents[agent_id].name} from {current_priority} to {new_priority}"
                )
                self.budget_coordinator.update_agent_priority(agent_id, new_priority)

    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics for all agents.
        """
        agent_metrics = {}
        total_cost = 0.0
        total_tokens = 0
        total_requests = 0

        for agent_id, agent in self.agents.items():
            performance = agent.get_performance_metrics()
            budget = self.budget_coordinator.get_agent_metrics(agent_id)

            agent_metrics[agent_id] = {
                "name": agent.name,
                "performance": performance,
                "budget": {
                    "initial": budget.initial_budget,
                    "used": budget.used_budget,
                    "remaining": budget.remaining_budget,
                    "percentage_used": (budget.used_budget / budget.initial_budget)
                    * 100
                    if budget.initial_budget > 0
                    else 0,
                },
                "priority": self.budget_coordinator.get_agent_priority(agent_id),
            }

            total_cost += performance["total_cost"]
            total_tokens += performance["total_tokens"]["total"]
            total_requests += performance["request_count"]

        return {
            "agents": agent_metrics,
            "summary": {
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "total_requests": total_requests,
                "avg_cost_per_request": total_cost / total_requests
                if total_requests > 0
                else 0,
            },
        }


def setup_enhanced_safety_framework():
    """
    Set up an enhanced version of the Agent Safety Framework.
    """
    # Create notification manager with multiple channels
    notification_manager = NotificationManager()

    # Configure console notifications (always enabled)
    notification_manager.configure_console()

    # Configure Slack notifications if webhook URL is available
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        notification_manager.configure_slack(webhook_url=slack_webhook_url)

    # Configure email notifications if credentials are available
    email_user = os.environ.get("EMAIL_USER")
    email_password = os.environ.get("EMAIL_PASSWORD")
    email_recipient = os.environ.get("EMAIL_RECIPIENT")
    if email_user and email_password and email_recipient:
        notification_manager.configure_email(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username=email_user,
            password=email_password,
            sender=email_user,
            recipients=[email_recipient],
        )

    # Create violation reporter with custom handlers
    violation_reporter = ViolationReporter(notification_manager=notification_manager)

    # Create budget coordinator
    budget_coordinator = BudgetCoordinator(violation_reporter=violation_reporter)

    # Create metrics collector
    metrics_collector = MetricsCollector()

    # Create content guardrail
    content_guardrail = ContentGuardrail(
        violation_reporter=violation_reporter, sensitive_topics=SENSITIVE_TOPICS
    )

    return {
        "notification_manager": notification_manager,
        "violation_reporter": violation_reporter,
        "budget_coordinator": budget_coordinator,
        "metrics_collector": metrics_collector,
        "content_guardrail": content_guardrail,
    }


async def run_agent_pipeline(
    orchestrator, research_agent_id, analysis_agent_id, summary_agent_id, topic
):
    """
    Run a pipeline of agents to process a topic.
    """
    logger.info(f"Starting agent pipeline for topic: {topic}")

    # Step 1: Research agent gathers information
    logger.info("Step 1: Running research agent...")
    research_result = await orchestrator.run_agent(
        research_agent_id,
        f"Gather key information about {topic}. Focus on facts, statistics, and recent developments.",
    )

    if "error" in research_result:
        logger.error(f"Research agent failed: {research_result['error']}")
        return {"error": f"Research agent failed: {research_result['error']}"}

    research_data = research_result["response"]
    logger.info(
        f"Research completed. Used {research_result['tokens']['total'] if 'total' in research_result['tokens'] else sum(research_result['tokens'].values())} tokens."
    )

    # Step 2: Analysis agent processes the information
    logger.info("Step 2: Running analysis agent...")
    analysis_result = await orchestrator.run_agent(
        analysis_agent_id,
        f"Analyze the following information about {topic} and identify patterns, insights, and implications:\n\n{research_data}",
    )

    if "error" in analysis_result:
        logger.error(f"Analysis agent failed: {analysis_result['error']}")
        return {"error": f"Analysis agent failed: {analysis_result['error']}"}

    analysis_data = analysis_result["response"]
    logger.info(
        f"Analysis completed. Used {analysis_result['tokens']['total'] if 'total' in analysis_result['tokens'] else sum(analysis_result['tokens'].values())} tokens."
    )

    # Step 3: Summary agent creates a final report
    logger.info("Step 3: Running summary agent...")
    summary_result = await orchestrator.run_agent(
        summary_agent_id,
        f"Create a concise summary report on {topic} based on the following research and analysis:\n\nRESEARCH:\n{research_data}\n\nANALYSIS:\n{analysis_data}",
    )

    if "error" in summary_result:
        logger.error(f"Summary agent failed: {summary_result['error']}")
        return {"error": f"Summary agent failed: {summary_result['error']}"}

    summary_data = summary_result["response"]
    logger.info(
        f"Summary completed. Used {summary_result['tokens']['total'] if 'total' in summary_result['tokens'] else sum(summary_result['tokens'].values())} tokens."
    )

    # Return the complete pipeline results
    return {
        "topic": topic,
        "research": {
            "content": research_data,
            "tokens": research_result["tokens"],
            "cost": research_result.get("cost", 0),
        },
        "analysis": {
            "content": analysis_data,
            "tokens": analysis_result["tokens"],
            "cost": analysis_result.get("cost", 0),
        },
        "summary": {
            "content": summary_data,
            "tokens": summary_result["tokens"],
            "cost": summary_result.get("cost", 0),
        },
    }


async def main_async():
    """
    Main async function to set up and run the enhanced example.
    """
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Set up enhanced safety framework
    framework = setup_enhanced_safety_framework()
    notification_manager = framework["notification_manager"]
    violation_reporter = framework["violation_reporter"]
    budget_coordinator = framework["budget_coordinator"]
    metrics_collector = framework["metrics_collector"]
    content_guardrail = framework["content_guardrail"]

    # Create budget pools for different purposes
    research_pool = BudgetPool(
        name="research_pool",
        initial_budget=2.0,  # $2.00 initial budget
        low_budget_threshold=0.3,  # 30% threshold for low budget warning
        description="Budget pool for research-focused agents",
    )

    analysis_pool = BudgetPool(
        name="analysis_pool",
        initial_budget=1.5,  # $1.50 initial budget
        low_budget_threshold=0.25,  # 25% threshold for low budget warning
        description="Budget pool for analysis-focused agents",
    )

    summary_pool = BudgetPool(
        name="summary_pool",
        initial_budget=1.0,  # $1.00 initial budget
        low_budget_threshold=0.2,  # 20% threshold for low budget warning
        description="Budget pool for summary-focused agents",
    )

    # Create budget pools
    budget_coordinator.create_budget_pool(research_pool)
    budget_coordinator.create_budget_pool(analysis_pool)
    budget_coordinator.create_budget_pool(summary_pool)

    # Create agent orchestrator
    orchestrator = AgentOrchestrator(
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
        metrics_collector=metrics_collector,
    )

    # Create specialized agents
    research_agent = EnhancedOpenAIAgentWrapper(
        name="Research Agent",
        description="Specialized in gathering comprehensive information on topics",
        model="gpt-4o",
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
        content_guardrail=content_guardrail,
        metrics_collector=metrics_collector,
        token_limit=8000,  # Higher token limit for research
    )

    analysis_agent = EnhancedOpenAIAgentWrapper(
        name="Analysis Agent",
        description="Specialized in analyzing information and identifying patterns",
        model="gpt-4o",
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
        content_guardrail=content_guardrail,
        metrics_collector=metrics_collector,
    )

    summary_agent = EnhancedOpenAIAgentWrapper(
        name="Summary Agent",
        description="Specialized in creating concise summaries from complex information",
        model="gpt-3.5-turbo",  # Using a less expensive model for summaries
        budget_coordinator=budget_coordinator,
        violation_reporter=violation_reporter,
        content_guardrail=content_guardrail,
        metrics_collector=metrics_collector,
    )

    # Register agents with the orchestrator
    research_agent_id = orchestrator.register_agent(
        agent=research_agent,
        pool_id=research_pool.id,
        initial_budget=1.0,  # $1.00 initial budget
        priority=8,  # High priority
    )

    analysis_agent_id = orchestrator.register_agent(
        agent=analysis_agent,
        pool_id=analysis_pool.id,
        initial_budget=0.8,  # $0.80 initial budget
        priority=6,  # Medium-high priority
    )

    summary_agent_id = orchestrator.register_agent(
        agent=summary_agent,
        pool_id=summary_pool.id,
        initial_budget=0.5,  # $0.50 initial budget
        priority=4,  # Medium priority
    )

    # Run a multi-agent pipeline
    topics = [
        "renewable energy developments in 2023",
        "advancements in quantum computing",
    ]

    results = []
    for topic in topics:
        result = await run_agent_pipeline(
            orchestrator, research_agent_id, analysis_agent_id, summary_agent_id, topic
        )
        results.append(result)

        # Rebalance budgets after each pipeline run
        orchestrator.rebalance_budgets()

        # Display updated metrics
        logger.info("\nUpdated budget metrics after pipeline run:")
        for agent_id, agent in orchestrator.agents.items():
            metrics = budget_coordinator.get_agent_metrics(agent_id)
            logger.info(
                f"Agent {agent.name}: Initial=${metrics.initial_budget:.2f}, "
                f"Used=${metrics.used_budget:.2f}, Remaining=${metrics.remaining_budget:.2f}"
            )

    # Get final metrics and save to file
    final_metrics = orchestrator.get_orchestrator_metrics()
    logger.info("\nFinal orchestrator metrics:")
    logger.info(f"Total cost: ${final_metrics['summary']['total_cost']:.4f}")
    logger.info(f"Total tokens: {final_metrics['summary']['total_tokens']}")
    logger.info(
        f"Average cost per request: ${final_metrics['summary']['avg_cost_per_request']:.4f}"
    )

    # Save results and metrics to files
    with open("pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("orchestrator_metrics.json", "w") as f:
        # Convert to a JSON-serializable format
        metrics_json = {}
        for key, value in final_metrics.items():
            if key == "agents":
                metrics_json[key] = {}
                for agent_id, agent_data in value.items():
                    metrics_json[key][agent_id] = {
                        k: (v.__dict__ if hasattr(v, "__dict__") else v)
                        for k, v in agent_data.items()
                    }
            else:
                metrics_json[key] = value

        json.dump(metrics_json, f, indent=2)

    logger.info(
        "Results and metrics saved to pipeline_results.json and orchestrator_metrics.json"
    )


def main():
    """
    Main entry point.
    """
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
