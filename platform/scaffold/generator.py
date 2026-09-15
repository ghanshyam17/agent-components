from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field


class ScaffoldConfig(BaseModel):
    pattern: str = Field(..., description="Project pattern (react, supervisor, network, data-analyst, automation)")
    name: str = Field(..., description="Project name")
    output_dir: Path = Field(..., description="Output directory base path")
    monorepo: bool = Field(False, description="Whether scaffolding into a monorepo")
    template_dir: Optional[Path] = Field(None, description="Custom template directory path")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variables for Jinja rendering")


def render_template_file(src: Path, dest: Path, variables: Dict[str, Any]):
    """Renders a single template file with jinja2-like variable replacement."""
    if not dest.parent.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
    
    content = src.read_text(encoding="utf-8")
    for key, val in variables.items():
        content = content.replace(f"{{{{ {key} }}}}", str(val))
        content = content.replace(f"{{{{{key}}}}}", str(val))
    dest.write_text(content, encoding="utf-8")

def scaffold_project(config: ScaffoldConfig) -> Path:
    """Scaffold a new project from templates."""
    target_dir = config.output_dir / config.name
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    # Determine template directory
    current_dir = Path(__file__).parent
    template_base = config.template_dir or (current_dir / "templates" / config.pattern)
    
    if not template_base.exists():
        raise FileNotFoundError(f"Template directory {template_base} does not exist.")

    variables = {
        "project_name": config.name,
        **config.variables
    }

    # Ensure base directories exist
    for dir_name in ["agents", "tools", "prompts", "tests", "infra"]:
        (target_dir / dir_name).mkdir(parents=True, exist_ok=True)

    for src_file in template_base.rglob("*"):
        if src_file.is_file():
            rel_path = src_file.relative_to(template_base)
            dest_file = target_dir / rel_path
            
            # Simple text files can be rendered
            if src_file.suffix in [".yaml", ".yml", ".md", ".txt", ".py", ".json"]:
                render_template_file(src_file, dest_file, variables)
            else:
                if not dest_file.parent.exists():
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)

    if not config.monorepo:
        # Generate standalone pyproject.toml
        pyproject_content = f"""[project]
name = "{config.name}"
version = "0.1.0"
description = "Agent Components project: {config.name}"
authors = [{{ name = "Agent Dev" }}]
dependencies = [
    "agent-components",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
        (target_dir / "pyproject.toml").write_text(pyproject_content, encoding="utf-8")

    return target_dir
