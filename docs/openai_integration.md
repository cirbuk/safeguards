# Integrating with OpenAI Agent SDK

This guide demonstrates how to integrate the Safeguards with the OpenAI Agent SDK to implement budget controls and monitoring.

## Installation

```bash
pip install safeguards openai
```

## Basic Usage

```python
from openai import OpenAI
from safeguards import AgentSafety, ConfigManager
from safeguards.monitoring import MetricsAnalyzer

# Initialize configuration
config_manager = ConfigManager()
config = config_manager.load_config("config.yaml")

# Initialize Agent Safety
safeguards = AgentSafety(config)

# Initialize OpenAI client with safety wrapper
client = OpenAI()
safe_client = safeguards.wrap_openai_client(client)

# Create an agent with budget controls
agent = safe_client.beta.agents.create(
    name="research_assistant",
    description="A research assistant with budget controls",
    model="gpt-4",
    tools=[{"type": "code_interpreter"}]
)

# The agent will now be monitored and budget-controlled
thread = safe_client.beta.threads.create()

# Add a message to the thread
message = safe_client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Research the latest developments in AI safety"
)

# Run the agent with safety controls
run = safe_client.beta.threads.runs.create(
    thread_id=thread.id,
    agent_id=agent.id
)

# Monitor the run
metrics = safeguards.get_metrics(agent.id)
print(f"Current budget usage: {metrics.budget_usage}%")
print(f"CPU usage: {metrics.cpu_percent}%")
```

## Advanced Configuration

### Environment Variables

```bash
export SAFEGUARDS_API_API_KEY=your_api_key
export SAFEGUARDS_BUDGET_DEFAULT_POOL_SIZE=2000.0
export SAFEGUARDS_MONITORING_ALERT_THRESHOLD_CPU=90.0
```

### YAML Configuration

```yaml
agent:
  name: research_assistant
  model: gpt-4
  max_budget: 1000.0
  priority: 1

monitoring:
  metrics_retention_days: 30
  alert_threshold_cpu: 80.0
  metrics_interval: 60
```

## Budget Control Features

### Pool Management

```python
# Create a new budget pool
pool = safeguards.create_budget_pool(
    name="research_pool",
    initial_size=1000.0,
    priority=1
)

# Assign agent to pool
safeguards.assign_agent_to_pool(agent.id, pool.id)

# Monitor pool usage
pool_metrics = safeguards.get_pool_metrics(pool.id)
print(f"Pool utilization: {pool_metrics.utilization}%")
```

### Dynamic Budget Allocation

```python
# Enable auto-scaling for the pool
safeguards.enable_pool_auto_scaling(
    pool_id=pool.id,
    min_size=500.0,
    max_size=2000.0,
    scale_threshold=80.0
)

# Set up budget alerts
safeguards.set_budget_alert(
    agent_id=agent.id,
    threshold=90.0,
    callback=lambda: print("Budget alert triggered!")
)
```

## Monitoring and Analytics

### Real-time Monitoring

```python
# Get real-time metrics
metrics = safeguards.get_real_time_metrics(agent.id)

# Set up monitoring dashboard
dashboard_url = safeguards.get_dashboard_url()
print(f"Monitor your agents at: {dashboard_url}")
```

### Trend Analysis

```python
# Analyze resource usage trends
analyzer = MetricsAnalyzer()
trends = analyzer.analyze_resource_trends(
    metrics_history=safeguards.get_metrics_history(agent.id),
    metric_name="cpu_percent"
)

print(f"Trend direction: {trends.trend_direction}")
print(f"Forecast next hour: {trends.forecast_next_hour}%")
```

### Usage Patterns

```python
# Analyze usage patterns
patterns = analyzer.analyze_usage_patterns(
    metrics_history=safeguards.get_metrics_history(agent.id)
)

print("Peak usage hours:", patterns.peak_hours)
print("Weekly pattern:", patterns.weekly_pattern)
```

## Best Practices

1. **Always set budget limits**: Define maximum budgets for agents to prevent runaway costs.
2. **Monitor resource usage**: Regularly check resource metrics to optimize performance.
3. **Use auto-scaling**: Enable dynamic budget allocation for efficient resource use.
4. **Set up alerts**: Configure alerts for budget and resource thresholds.
5. **Analyze patterns**: Use the metrics analyzer to understand usage patterns.

## Error Handling

```python
from safeguards.exceptions import BudgetExceededError, ResourceLimitError

try:
    # Run agent with safety controls
    run = safe_client.beta.threads.runs.create(
        thread_id=thread.id,
        agent_id=agent.id
    )
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
    # Handle budget exceeded case
except ResourceLimitError as e:
    print(f"Resource limit reached: {e}")
    # Handle resource limit case
```

## Dashboard Integration

The Safeguards provides a web dashboard for monitoring and controlling your agents. Access it at `http://localhost:8000` after starting the dashboard server:

```python
from safeguards.dashboard import start_dashboard

# Start the dashboard server
start_dashboard(host="localhost", port=8000)
```

The dashboard provides:
- Real-time metrics visualization
- Budget pool management
- Agent monitoring and control
- Usage pattern analysis
- Alert configuration

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.
