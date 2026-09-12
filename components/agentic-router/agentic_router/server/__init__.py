# Re-export under names that don't shadow the `app` submodule.
from agentic_router.server.app import create_app as create_app
from agentic_router.server.app import app as application

__all__ = ["create_app", "application"]