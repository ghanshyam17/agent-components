from __future__ import annotations
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Union
import yaml

from pydantic import BaseModel

from platform.schema.agent_spec import AgentSpec
from platform.schema.project_spec import ProjectSpec
from platform.schema.infrastructure_spec import InfrastructureSpec

BaseSpec = Union[AgentSpec, ProjectSpec, InfrastructureSpec]

def resolve_variables(data: Any, env: Dict[str, str]) -> Any:
    """Recursively resolves ${VAR_NAME} placeholders from environment."""
    if isinstance(data, dict):
        return {k: resolve_variables(v, env) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_variables(v, env) for v in data]
    elif isinstance(data, str):
        pattern = re.compile(r'\$\{([^}]+)\}')
        def replace_match(match: re.Match) -> str:
            var_name = match.group(1)
            return env.get(var_name, match.group(0))
        return pattern.sub(replace_match, data)
    return data

def resolve_includes(data: Any, base_dir: Path) -> Any:
    """Recursively resolves relative file path references.
    (Simple walk, extend based on specific include directive)."""
    if isinstance(data, dict):
        return {k: resolve_includes(v, base_dir) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_includes(v, base_dir) for v in data]
    return data

def validate_spec(data: Dict[str, Any]) -> BaseSpec:
    """Dispatches to correct Pydantic model based on the `kind` field."""
    kind = data.get("kind")
    if kind == "Agent":
        return AgentSpec.model_validate(data)
    elif kind == "Project":
        return ProjectSpec.model_validate(data)
    elif kind == "Infrastructure":
        return InfrastructureSpec.model_validate(data)
    else:
        raise ValueError(f"Unknown or missing kind field: {kind}")

def load_spec(path: Union[str, Path]) -> BaseSpec:
    """Auto-detects YAML/JSON/text, validates, and returns typed spec."""
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    
    data = None
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(content)
    elif path.suffix == ".json":
        data = json.loads(content)
    else:
        # Fallback
        try:
            data = yaml.safe_load(content)
        except Exception:
            try:
                data = json.loads(content)
            except Exception as e:
                raise ValueError(f"Could not parse file {path} as YAML or JSON") from e
                
    if not isinstance(data, dict):
        raise ValueError(f"File content in {path} must be a dictionary at root")
    
    # Resolve env variables
    data = resolve_variables(data, os.environ.copy())
    # Resolve includes
    data = resolve_includes(data, path.parent)
    
    return validate_spec(data)

def load_project(dir_path: Union[str, Path]) -> ProjectSpec:
    """Loads a project spec from a directory, parsing its project.yaml or json."""
    dir_path = Path(dir_path)
    project_file = dir_path / "project.yaml"
    if not project_file.exists():
        project_file = dir_path / "project.json"
        
    if not project_file.exists():
        raise FileNotFoundError(f"No project.yaml or project.json found in {dir_path}")
        
    spec = load_spec(project_file)
    if not isinstance(spec, ProjectSpec):
        raise ValueError("Loaded spec is not a ProjectSpec")
        
    return spec
