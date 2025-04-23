"""Plugin system for extending Safeguards functionality."""

from abc import ABC, abstractmethod
from typing import Any


class SafeguardPlugin(ABC):
    """Base class for all Safeguard plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the plugin."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version of the plugin."""
        pass

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Plugin-specific configuration
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        pass


class PluginManager:
    """Manages the loading, configuration and lifecycle of plugins."""

    def __init__(self):
        self._plugins: dict[str, SafeguardPlugin] = {}
        self._plugin_configs: dict[str, dict[str, Any]] = {}

    def register_plugin(
        self,
        plugin: SafeguardPlugin,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Register a plugin with the manager.

        Args:
            plugin: The plugin instance to register
            config: Optional configuration for the plugin
        """
        if plugin.name in self._plugins:
            msg = f"Plugin with name '{plugin.name}' already registered"
            raise ValueError(msg)

        self._plugins[plugin.name] = plugin
        self._plugin_configs[plugin.name] = config or {}

        # Initialize the plugin with its configuration
        plugin.initialize(self._plugin_configs[plugin.name])

    def get_plugin(self, name: str) -> SafeguardPlugin | None:
        """Get a plugin by name.

        Args:
            name: Name of the plugin to retrieve

        Returns:
            The plugin instance if found, None otherwise
        """
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        """List all registered plugins.

        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())

    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin by name.

        Args:
            name: Name of the plugin to unregister
        """
        if name not in self._plugins:
            msg = f"Plugin '{name}' not registered"
            raise ValueError(msg)

        # Allow plugin to clean up
        self._plugins[name].shutdown()

        # Remove from registries
        del self._plugins[name]
        del self._plugin_configs[name]

    def shutdown_all(self) -> None:
        """Shutdown all plugins."""
        for name, plugin in list(self._plugins.items()):
            plugin.shutdown()
            del self._plugins[name]
            del self._plugin_configs[name]
