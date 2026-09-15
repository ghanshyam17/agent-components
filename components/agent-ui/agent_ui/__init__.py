"""agent-ui: Universal metadata-driven Agent UI and Server Bridge.

Exports the FastAPI application factory and mounting utilities to serve
an enterprise conversational and graph interface for any agent or component graph.
"""
from __future__ import annotations

from agent_ui.bridge import UIBridge
from agent_ui.server import create_ui_app, mount_agent_ui

__all__ = ["UIBridge", "create_ui_app", "mount_agent_ui"]
