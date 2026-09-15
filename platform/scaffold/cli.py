from __future__ import annotations
import click
from pathlib import Path
from typing import Optional

from .generator import scaffold_project, ScaffoldConfig
from platform.plugins.loader import PluginLoader
from platform.schema.loader import load_spec

@click.group()
def agcomps():
    """Main CLI group for agent-components."""
    pass

@agcomps.command()
@click.option('--pattern', type=click.Choice(['react', 'supervisor', 'network', 'data-analyst', 'automation', 'autogen', 'langgraph', 'framework-router']), required=True, help="Scaffold pattern")
@click.option('--name', required=True, help="Project name")
@click.option('--monorepo', is_flag=True, help="Scaffold into the monorepo vs standalone")
@click.option('--template', type=click.Path(exists=True, file_okay=False), help="Path to custom template")
@click.option('--output', type=click.Path(), default=".", help="Output directory")
def new(pattern: str, name: str, monorepo: bool, template: Optional[str], output: str):
    """Scaffold a new project."""
    config = ScaffoldConfig(
        pattern=pattern,
        name=name,
        output_dir=Path(output),
        monorepo=monorepo,
        template_dir=Path(template) if template else None
    )
    out_path = scaffold_project(config)
    click.echo(f"Successfully scaffolded project {name} at {out_path}")

@agcomps.command()
@click.option('--project', type=click.Path(exists=True, dir_okay=False), required=True, help="Path to project.yaml")
@click.option('--env', type=click.Choice(['dev', 'staging', 'prod']), required=True, help="Deployment environment")
@click.option('--dry-run', is_flag=True, help="Plan without deploying")
def deploy(project: str, env: str, dry_run: bool):
    """Deploy a project."""
    click.echo(f"Deploying project {project} to {env} (dry-run: {dry_run})")
    # Calls DeployEngine.deploy() in real implementation

@agcomps.command()
@click.option('--agent', type=click.Path(exists=True, dir_okay=False), required=True, help="Path to agent.yaml")
@click.option('--port', type=int, default=8000, help="Local server port")
def dev(agent: str, port: int):
    """Run agent locally."""
    click.echo(f"Starting local dev server for {agent} on port {port}")
    # Loads spec, builds agent, starts local server in real implementation

@agcomps.command()
@click.option('--path', type=click.Path(exists=True), required=True, help="File or directory to validate")
def validate(path: str):
    """Validate YAML specs."""
    try:
        path_obj = Path(path)
        if path_obj.is_file():
            load_spec(path_obj)
            click.echo(f"{path} is valid.")
        else:
            for p in path_obj.rglob("*.yaml"):
                load_spec(p)
            click.echo(f"All YAML files in {path} are valid.")
    except Exception as e:
        click.echo(f"Validation failed: {e}", err=True)

@agcomps.command(name='list-patterns')
def list_patterns():
    """List available patterns with descriptions."""
    patterns = {
        'react': 'Single ReAct agent with basic tools',
        'supervisor': 'Supervisor pattern with multiple workers',
        'network': 'Network of equal peer agents',
        'data-analyst': 'Data analyst agent with code interpreter',
        'automation': 'Automation agent with custom tools',
        'autogen': 'AutoGen multi-agent conversational group chat & peer collaboration',
        'langgraph': 'LangGraph stateful cyclical graph & workflow orchestration',
        'framework-router': 'First-route classifier routing tasks to AutoGen, LangGraph, or ReAct',
    }
    for p, desc in patterns.items():
        click.echo(f"{p}: {desc}")



@agcomps.group()
def plugins():
    """Plugin management subgroup."""
    pass

@plugins.command(name='list')
def plugins_list():
    """List loaded plugins."""
    click.echo("Listing loaded plugins...")
    # Requires a registry instance populated from config/directories

@plugins.command(name='validate')
@click.option('--file', type=click.Path(exists=True, dir_okay=False), required=True, help="Plugin file to validate")
def plugins_validate(file: str):
    """Validate a plugin file."""
    try:
        loader = PluginLoader()
        plugin = loader.load(file)
        click.echo(f"Plugin {plugin.spec.metadata.name} is valid.")
    except Exception as e:
        click.echo(f"Plugin validation failed: {e}", err=True)

if __name__ == '__main__':
    agcomps()
