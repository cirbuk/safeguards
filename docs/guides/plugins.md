# Plugin Framework

The Safeguards library includes a flexible plugin framework that allows you to create custom safeguards for your specific use cases, including industry-specific checks and rules.

## Plugin Architecture

The plugin system is based on the `SafeguardPlugin` abstract base class, which all plugins must extend. Plugins are managed through the `PluginManager`, which handles registration, configuration, and lifecycle management.

### Plugin Manager

The `PluginManager` provides a central registry for all plugins:

```python
from safeguards.plugins import PluginManager

# Create a plugin manager
plugin_manager = PluginManager()

# Register plugins
plugin_manager.register_plugin(my_plugin, config={"key": "value"})

# Get a plugin by name
my_plugin = plugin_manager.get_plugin("my_plugin_name")

# List all registered plugins
plugin_names = plugin_manager.list_plugins()

# Unregister a plugin
plugin_manager.unregister_plugin("my_plugin_name")

# Shutdown all plugins
plugin_manager.shutdown_all()
```

## Creating Custom Plugins

To create a custom plugin, extend the `SafeguardPlugin` abstract base class:

```python
from safeguards.plugins import SafeguardPlugin
from typing import Dict, Any

class MyCustomPlugin(SafeguardPlugin):
    """Example custom plugin."""

    def __init__(self):
        """Initialize the plugin."""
        self._config = {}

    @property
    def name(self) -> str:
        """Return the name of the plugin."""
        return "my_custom_plugin"

    @property
    def version(self) -> str:
        """Return the version of the plugin."""
        return "1.0.0"

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Plugin-specific configuration
        """
        self._config = config
        # Perform any initialization based on config

    def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        # Clean up any resources used by the plugin
```

## Industry-Specific Safeguards

The Safeguards library includes pre-built industry-specific safeguards for common use cases. These extend the `IndustrySafeguard` base class, which itself extends `SafeguardPlugin`.

### Available Industry Safeguards

Currently, the library provides:

- `FinancialServicesSafeguard`: For financial services operations
- `HealthcareSafeguard`: For healthcare operations

### Using Industry Safeguards

Here's how to use a financial services safeguard:

```python
from decimal import Decimal
from safeguards.plugins.industry import FinancialServicesSafeguard
from safeguards.types.agent import Agent

# Create and configure the safeguard
financial_safeguard = FinancialServicesSafeguard()
financial_safeguard.initialize({
    "restricted_actions": ["high_risk_investment", "unauthorized_withdrawal"],
    "compliance_rules": {
        "kyc_required": True,
        "aml_check": True
    },
    "transaction_limits": {
        "agent_1": Decimal("1000.00")
    }
})

# Start monitoring an agent
financial_safeguard.monitor_agent("agent_1")

# Validate an action
agent = get_agent("agent_1")  # Get your agent instance
action_context = {
    "action_type": "transaction",
    "amount": Decimal("1500.00"),
    "description": "Investment purchase"
}

# Check for violations
alerts = financial_safeguard.validate_agent_action(agent, action_context)
if alerts:
    print(f"Action violates safeguards: {alerts}")
else:
    # Proceed with action
    agent.run(**action_context)

# Stop monitoring when done
financial_safeguard.stop_monitoring_agent("agent_1")
```

### Healthcare Safeguard Example

Similarly, for healthcare operations:

```python
from safeguards.plugins.industry import HealthcareSafeguard

# Create and configure the safeguard
healthcare_safeguard = HealthcareSafeguard()
healthcare_safeguard.initialize({
    "phi_patterns": ["SSN", "DOB", "MRN"],
    "restricted_operations": ["mass_record_access", "export_all_records"],
    "required_approvals": {
        "prescription": ["doctor", "pharmacist"]
    }
})

# Monitor and validate as with the financial safeguard
```

## Creating Industry-Specific Safeguards

To create a custom industry safeguard, extend the `IndustrySafeguard` class:

```python
from typing import Dict, Any, List
from safeguards.plugins.industry import IndustrySafeguard
from safeguards.types.agent import Agent
from safeguards.types import SafetyAlert, AlertSeverity

class RetailSafeguard(IndustrySafeguard):
    """Safeguards specific to the retail industry."""

    def __init__(self):
        """Initialize the retail safeguard."""
        super().__init__("retail")
        self._pricing_limits = {}
        self._discount_limits = {}

    @property
    def version(self) -> str:
        """Return the version of the plugin."""
        return "1.0.0"

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        self._config = config
        self._pricing_limits = config.get("pricing_limits", {})
        self._discount_limits = config.get("discount_limits", {})
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        self._initialized = False
        self._monitored_agents.clear()

    def validate_agent_action(self, agent: Agent, action_context: Dict[str, Any]) -> List[SafetyAlert]:
        """Validate a retail agent action."""
        alerts = []

        # Check pricing within limits
        if action_context.get("action_type") == "price_change":
            product = action_context.get("product", "")
            price = action_context.get("price", 0)
            min_price = self._pricing_limits.get(product, {}).get("min", 0)
            max_price = self._pricing_limits.get(product, {}).get("max", float("inf"))

            if price < min_price:
                alerts.append(SafetyAlert(
                    title="Price Below Minimum",
                    description=f"Price {price} is below minimum {min_price} for {product}",
                    severity=AlertSeverity.WARNING
                ))

            if price > max_price:
                alerts.append(SafetyAlert(
                    title="Price Above Maximum",
                    description=f"Price {price} is above maximum {max_price} for {product}",
                    severity=AlertSeverity.WARNING
                ))

        return alerts
```

## Integrating with the Safety Controller

You can integrate your custom safeguards with the `SafetyController` for centralized management:

```python
from safeguards.core.safety_controller import SafetyController
from safeguards.types import SafetyConfig
from safeguards.plugins.industry import FinancialServicesSafeguard

# Create controller
config = SafetyConfig(...)
controller = SafetyController(config)

# Create industry safeguard
financial_safeguard = FinancialServicesSafeguard()
financial_safeguard.initialize({...})

# Register agent with controller and safeguard
agent = MyAgent("agent_1")
controller.register_agent(agent, budget=Decimal("1000.00"))
financial_safeguard.monitor_agent(agent.id)

# When validating actions, combine checks
def execute_safe_action(agent, action_context):
    # First check controller rules
    validation = controller.validate_action(agent.id, action_context)
    if not validation.valid:
        return {"success": False, "violations": validation.violations}

    # Then check industry safeguards
    alerts = financial_safeguard.validate_agent_action(agent, action_context)
    if alerts:
        return {"success": False, "alerts": alerts}

    # If all checks pass, execute the action
    return agent.run(**action_context)
```
