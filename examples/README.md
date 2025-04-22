# Safeguards Examples

This directory contains example implementations demonstrating how to use the Safeguards with different agent systems.

## OpenAI Agents SDK Integration

The examples in this directory show how to integrate the Safeguards with the OpenAI Agents SDK.

### Prerequisites

- Python 3.8 or higher
- Safeguards installed
- OpenAI Agents SDK installed
- OpenAI API key

### Available Examples

1. **Basic Integration** (`openai_agents_integration.py`)
   - Simple integration of a single OpenAI agent with budget tracking
   - Demonstrates basic budget management and resource monitoring
   - Good starting point for understanding the integration process
   - ✅ Production-ready

2. **Advanced Integration** (`openai_agents_advanced.py`)
   - Multi-agent system with research and writing agents
   - Demonstrates content filtering, budget guardrails, and resource monitoring
   - Includes violation reporting and notification systems
   - Shows how to collect and display metrics
   - ✅ Production-ready

3. **Single Agent Example** (`single_agent.py`)
   - Basic example of setting up safety controls for a single agent
   - Demonstrates budget tracking and violation reporting
   - ✅ Production-ready

4. **Multi-Agent Example** (`multi_agent.py`)
   - Demonstrates coordinating multiple agents with shared budget controls
   - Shows how to use BudgetCoordinator for agent coordination
   - ✅ Production-ready

5. **Budget Monitoring Example** (`budget_monitoring.py`)
   - Shows how to implement budget monitoring and alerts
   - Demonstrates real-time tracking and visualization
   - ✅ Production-ready

6. **Budget Control Example** (`budget_control_example.py`)
   - Comprehensive example of budget control integration
   - Demonstrates budget pools, agent registration, and metrics
   - ✅ Production-ready

7. **Guardrails Example** (`guardrails_example.py`)
   - Demonstrates implementing and using safety guardrails
   - Shows how to combine multiple guardrails for comprehensive protection
   - ✅ Production-ready

8. **Notification Setup Example** (`notification_setup.py`)
   - Shows how to configure Slack and Email notifications
   - Demonstrates notification channels and formatting
   - ✅ Production-ready

### Experimental Components

The following components are still in development and should be used with caution:

- **SwarmController** (`safeguards.swarm.SwarmController`): This component is still experimental and not ready for production use. For multi-agent coordination, please use the BudgetCoordinator directly as shown in `examples/multi_agent.py`.

For details on component status, see the [Component Status](../COMPONENT_STATUS.md) document.

### Setup and Usage

1. **Install dependencies**:
   ```bash
   pip install -e .  # Install Safeguards in dev mode
   pip install agents-sdk  # Install OpenAI Agents SDK
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   # Optional for Slack notifications
   export SLACK_WEBHOOK_URL=your_slack_webhook_url
   ```

3. **Run an example**:
   ```bash
   python examples/openai_agents_integration.py
   # or
   python examples/openai_agents_advanced.py
   ```

### Key Concepts Demonstrated

- **Agent Wrapping**: Creating a wrapper class to make external agents compatible with the Safeguards
- **Budget Management**: Setting up and tracking budget usage for agents and pools
- **Guardrails**: Implementing budget, resource, and content guardrails
- **Violation Reporting**: Reporting and handling safety violations
- **Notifications**: Setting up notification channels for alerts
- **Metrics Collection**: Gathering and displaying usage metrics

### Customizing the Examples

To adapt these examples for your own use case:

1. Modify the agent instructions and tools based on your requirements
2. Adjust budget amounts and thresholds to fit your usage patterns
3. Add or modify the guardrails based on your safety requirements
4. Customize the content filtering rules in the `ContentGuardrail` class
5. Add additional notification channels as needed

### Troubleshooting

If you encounter issues when running the examples:

- Ensure your OpenAI API key is set correctly
- Check that all dependencies are installed
- Verify that the Safeguards is installed properly
- Look for error messages that might indicate version incompatibilities

For more assistance, please refer to the main documentation or create an issue on the GitHub repository.

## Examples Overview

1. `single_agent.py` - Basic example of setting up safety controls for a single agent
2. `multi_agent.py` - Demonstrates coordinating multiple agents with shared budget pools
3. `budget_monitoring.py` - Shows how to implement budget monitoring and alerts
4. `budget_control_example.py` - Comprehensive example of budget control integration
5. `guardrails_example.py` - Demonstrates implementing and using safety guardrails
6. `notification_setup.py` - Shows how to configure Slack and Email notifications
7. `config.yaml` - Example configuration file for the framework

## Running the Examples

### Prerequisites
- Python 3.8 or higher
- Safeguards installed
- Required dependencies (see requirements.txt in root directory)

### Setup
1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Configure the example settings (optional):
- Edit `config.yaml` to adjust budget limits, monitoring thresholds, etc.
- Modify agent parameters in the example files as needed

### Running Individual Examples

1. Single Agent Example:
```bash
python single_agent.py
```

2. Multi-Agent Example:
```bash
python multi_agent.py
```

3. Budget Monitoring Example:
```bash
python budget_monitoring.py
```

4. Budget Control Example:
```bash
python budget_control_example.py
```

5. Guardrails Example:
```bash
python guardrails_example.py
```

6. Notification Setup Example:
```bash
python notification_setup.py
```

## Example Features

### Budget Control Example
The `budget_control_example.py` demonstrates:
- Setting up budget pools with different priorities
- Creating and monitoring multiple AI agents
- Real-time budget tracking and updates
- Health monitoring and alerts
- Handling budget violations
- Generating recommendations based on usage patterns

### Multi-Agent Example
The `multi_agent.py` shows:
- Coordinating multiple agents
- Shared budget pool management
- Resource allocation strategies
- Inter-agent communication

### Budget Monitoring Example
The `budget_monitoring.py` covers:
- Real-time budget tracking
- Alert configuration
- Violation reporting
- Usage analytics

### Guardrails Example
The `guardrails_example.py` demonstrates:
- Creating custom safety guardrails
- Implementing content safety checks
- Combining multiple guardrails (budget, resource, content)
- Handling guardrail violations
- Using the composite guardrail pattern

### Notification Setup Example
The `notification_setup.py` demonstrates:
- Configuring different notification channels (Slack, Email, Webhook)
- Securely managing notification credentials
- Sending different types of alerts
- Custom templates and formatting
- Integration with Budget Coordinator
- Best practices for notification security

## Best Practices
1. Always set up proper error handling
2. Monitor budget usage in real-time
3. Configure appropriate alert thresholds
4. Implement proper cleanup in your agents
5. Use the health monitoring system for proactive management
6. Apply multiple guardrails for comprehensive protection

## Customization
The examples can be customized by:
1. Modifying the configuration in `config.yaml`
2. Adjusting budget thresholds and limits
3. Implementing custom agent logic
4. Adding additional monitoring metrics
5. Customizing alert and violation handling
6. Creating your own custom guardrails

## Support
For questions or issues:
- Check the main documentation
- Review the inline comments in example files
- Submit issues through the project's issue tracker
