from __future__ import annotations
import threading
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field

from toolkit.base import Registry
from platform.schema.agent_spec import AgentSpec
from .loader import LoadedPlugin


class ResolvedPlugins(BaseModel):
    tools: Registry = Field(default_factory=Registry, description="Registry compatible tools")
    prompts: Dict[str, str] = Field(default_factory=dict, description="Prompts")
    guardrails: List[Any] = Field(default_factory=list, description="Guardrails")
    connectors: List[Any] = Field(default_factory=list, description="Connectors")
    
    model_config = {"arbitrary_types_allowed": True}


class PluginRegistry:
    """Central registry for loaded plugins."""

    def __init__(self):
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._lock = threading.Lock()

    def register(self, plugin: LoadedPlugin) -> None:
        """Register a loaded plugin."""
        with self._lock:
            name = plugin.spec.metadata.name
            self._plugins[name] = plugin

    def unregister(self, name: str) -> None:
        """Unregister a plugin by name."""
        with self._lock:
            if name in self._plugins:
                del self._plugins[name]

    def get(self, name: str) -> Optional[LoadedPlugin]:
        """Get a plugin by name."""
        with self._lock:
            return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """List names of all registered plugins."""
        with self._lock:
            return list(self._plugins.keys())

    def compose_tools(self, plugin_names: List[str]) -> Registry:
        """Compose a toolkit Registry from selected plugins' tools."""
        registry = Registry()
        with self._lock:
            for name in plugin_names:
                plugin = self._plugins.get(name)
                if plugin:
                    # In a real implementation, we would convert ToolPluginEntry to actual Tool/FunctionTool
                    # and register them into `registry`.
                    pass
        return registry

    def compose_prompts(self, plugin_names: List[str]) -> Dict[str, str]:
        """Compose a prompt dict from selected plugins."""
        prompts = {}
        with self._lock:
            for name in plugin_names:
                plugin = self._plugins.get(name)
                if plugin:
                    prompts.update(plugin.resolved_prompts)
        return prompts

    def resolve_for_agent(self, agent_spec: AgentSpec) -> ResolvedPlugins:
        """Resolve all plugins referenced by an agent spec."""
        # For simplicity, assuming agent_spec might reference plugins by name or imply standard plugins
        # Currently agent_spec doesn't have an explicit 'plugins' field, so we might need to deduce or
        # if project_spec is used instead, we'd pass that.
        # This is a stub implementation.
        plugin_names = self.list_plugins() # Or a subset defined by agent tags/metadata
        
        tools = self.compose_tools(plugin_names)
        prompts = self.compose_prompts(plugin_names)
        guardrails = []
        connectors = []
        
        with self._lock:
            for name in plugin_names:
                plugin = self._plugins.get(name)
                if plugin:
                    guardrails.extend(plugin.spec.spec.guardrails)
                    connectors.extend(plugin.spec.spec.connectors)

        return ResolvedPlugins(
            tools=tools,
            prompts=prompts,
            guardrails=guardrails,
            connectors=connectors
        )
