from __future__ import annotations
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal

from pydantic import BaseModel, Field


class ToolPluginEntry(BaseModel):
    """Configuration for a tool plugin entry."""
    name: str = Field(..., description="Name of the tool")
    type: Literal["function", "database", "azure-storage", "api"] = Field("function", description="Type of the tool")
    source_file: Optional[str] = Field(None, description="Path to the source file for function tools")
    function_name: Optional[str] = Field(None, description="Name of the function in the source file")
    connection_string: Optional[str] = Field(None, description="Connection string for database or storage tools")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration dict")

class PromptPluginEntry(BaseModel):
    """Configuration for a prompt plugin entry."""
    name: str = Field(..., description="Name of the prompt")
    file: Optional[str] = Field(None, description="Path to the prompt file")
    content: Optional[str] = Field(None, description="Inline prompt content")
    variables: List[str] = Field(default_factory=list, description="List of variables used in the prompt")

class PluginMetadata(BaseModel):
    name: str
    version: str
    description: Optional[str] = None
    author: Optional[str] = None

class PluginSpecModel(BaseModel):
    tools: List[ToolPluginEntry] = Field(default_factory=list)
    prompts: List[PromptPluginEntry] = Field(default_factory=list)
    guardrails: List[Dict[str, Any]] = Field(default_factory=list)
    connectors: List[Dict[str, Any]] = Field(default_factory=list)

class PluginSpec(BaseModel):
    """Specification model for a Plugin."""
    apiVersion: Literal["agentcomponents/v1"] = Field("agentcomponents/v1", description="API Version")
    kind: Literal["Plugin"] = Field("Plugin", description="Resource kind")
    metadata: PluginMetadata = Field(..., description="Plugin metadata")
    spec: PluginSpecModel = Field(default_factory=PluginSpecModel, description="Plugin specification details")

class LoadedPlugin(BaseModel):
    """Represents a loaded and resolved plugin."""
    spec: PluginSpec
    resolved_tools: List[Any] = Field(default_factory=list)
    resolved_prompts: Dict[str, str] = Field(default_factory=dict)
    source_path: Path

class PluginLoader:
    """Discovers and loads plugins from files or directories."""

    def load(self, path: str | Path) -> LoadedPlugin:
        path = Path(path).resolve()
        if path.suffix in (".yaml", ".yml"):
            return self.load_yaml(path)
        elif path.suffix == ".json":
            return self.load_json(path)
        elif path.suffix in (".txt", ".md"):
            return self.load_text(path)
        else:
            raise ValueError(f"Unsupported plugin file format: {path.suffix}")

    def load_yaml(self, path: Path) -> LoadedPlugin:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        return self._resolve_and_create(data, path)

    def load_json(self, path: Path) -> LoadedPlugin:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        return self._resolve_and_create(data, path)

    def load_text(self, path: Path) -> LoadedPlugin:
        content = path.read_text(encoding="utf-8")
        # Treat text files as simple inline prompt plugins
        name = path.stem
        spec_data = {
            "apiVersion": "agentcomponents/v1",
            "kind": "Plugin",
            "metadata": {
                "name": f"{name}-prompt",
                "version": "1.0.0"
            },
            "spec": {
                "prompts": [
                    {
                        "name": name,
                        "content": content
                    }
                ]
            }
        }
        return self._resolve_and_create(spec_data, path)

    def discover(self, directory: str | Path) -> List[LoadedPlugin]:
        directory = Path(directory)
        plugins = []
        if not directory.exists() or not directory.is_dir():
            return plugins

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in (".yaml", ".yml", ".json", ".txt", ".md"):
                try:
                    plugins.append(self.load(file_path))
                except Exception as e:
                    # Log or handle invalid plugins silently during discovery
                    pass
        return plugins

    def _resolve_and_create(self, data: Dict[str, Any], path: Path) -> LoadedPlugin:
        spec = PluginSpec.model_validate(data)
        
        resolved_prompts = {}
        # Resolve paths relative to plugin source path
        for prompt in spec.spec.prompts:
            if prompt.file:
                prompt_file = path.parent / prompt.file
                if prompt_file.exists():
                    resolved_prompts[prompt.name] = prompt_file.read_text(encoding="utf-8")
                else:
                    resolved_prompts[prompt.name] = f"Error: Prompt file {prompt.file} not found."
            elif prompt.content:
                resolved_prompts[prompt.name] = prompt.content

        resolved_tools = []
        for tool in spec.spec.tools:
            resolved_tools.append(tool) # Storing entries, could be resolved to actual functions if needed

        return LoadedPlugin(
            spec=spec,
            resolved_tools=resolved_tools,
            resolved_prompts=resolved_prompts,
            source_path=path
        )
