"""FrenchCase → agent-components chassis bridges.

Three adapters that let the chassis components drive FrenchCase's real engines
instead of duplicating them:

| Chassis component | Bridge | FrenchCase engine |
|---|---|---|
| memory-store (SessionStore) | `memory_bridge.FrenchCaseSessionStore` | `app/core/session_store.py` |
| retriever (VectorMemory)    | `retriever_bridge.FrenchCaseVectorMemory` | `app/core/vector.py::retrieve` |
| model-gateway (Gateway)     | `llm_bridge.FrenchCaseGateway` | `app/core/llm_router.py::LLMRouter` |

All three degrade to no-op mode (with a warning) when FrenchCase is not mounted
on `sys.path`, so this package imports cleanly inside the chassis repo alone.
"""
from __future__ import annotations

from projects.frenchcase.bridges.llm_bridge import (
    FrenchCaseGateway,
    FrenchCaseGatewayClient,
    build_gateway,
)
from projects.frenchcase.bridges.memory_bridge import (
    FrenchCaseSessionStore,
    build_session_store,
)
from projects.frenchcase.bridges.retriever_bridge import (
    FrenchCaseVectorMemory,
    build_vector_memory,
)

__all__ = [
    "FrenchCaseGateway",
    "FrenchCaseGatewayClient",
    "build_gateway",
    "FrenchCaseSessionStore",
    "build_session_store",
    "FrenchCaseVectorMemory",
    "build_vector_memory",
]