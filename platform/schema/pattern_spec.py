from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel, Field

class WorkerRef(BaseModel):
    """Reference to a worker agent."""
    agent_ref: str = Field(..., description="Path or reference to the agent specification file")
    role: str = Field(..., description="Role description for the worker")
    capabilities: List[str] = Field(default_factory=list, description="List of capabilities of the worker")

class SupervisorConfig(BaseModel):
    """Configuration for the Supervisor pattern."""
    strategy: Literal["plan_and_delegate", "round_robin", "adaptive"] = Field(..., description="Supervisor strategy")
    max_rounds: int = Field(10, description="Maximum number of conversation rounds")
    workers: List[WorkerRef] = Field(..., description="List of worker agents")

class NetworkConfig(BaseModel):
    """Configuration for the Network pattern."""
    topology: Literal["mesh", "star", "ring"] = Field(..., description="Network topology")
    message_bus: Literal["in-memory", "redis", "service-bus"] = Field("in-memory", description="Message bus type")

class SequentialConfig(BaseModel):
    """Configuration for the Sequential pattern."""
    stages: List[str] = Field(..., description="List of agent references for each stage")

class MapReduceConfig(BaseModel):
    """Configuration for the MapReduce pattern."""
    mapper: str = Field(..., description="Mapper agent reference")
    reducer: str = Field(..., description="Reducer agent reference")
    max_parallel: int = Field(5, description="Maximum number of parallel mapper executions")


# --- AutoGen Pattern Configuration ---

class AutoGenAgentConfig(BaseModel):
    """Individual agent configuration in an AutoGen multi-agent setup."""
    name: str = Field(..., description="Unique name of the agent")
    role: str = Field(..., description="Role description of the agent")
    system_message: str = Field(..., description="System prompt / instructions for the agent")
    is_user_proxy: bool = Field(False, description="Whether this agent acts as UserProxyAgent")
    human_input_mode: Literal["NEVER", "ALWAYS", "TERMINATE"] = Field("NEVER", description="Human input mode")
    code_execution: bool = Field(False, description="Enable code execution for this agent")
    model_override: Optional[str] = Field(None, description="Optional model deployment override")


class AutoGenConfig(BaseModel):
    """Configuration for AutoGen multi-agent conversational patterns."""
    mode: Literal["group_chat", "two_agent", "round_robin", "magentic_one"] = Field(
        "group_chat", description="Conversation mode"
    )
    max_rounds: int = Field(12, description="Maximum conversation rounds")
    admin_name: Optional[str] = Field(None, description="Name of the admin / coordinator agent")
    speaker_selection_method: Literal["auto", "round_robin", "random", "manual"] = Field(
        "auto", description="Speaker selection method for group chat"
    )
    agents: List[AutoGenAgentConfig] = Field(..., description="List of agents participating in the conversation")
    termination_keyword: str = Field("TERMINATE", description="Keyword indicating conversation completion")


# --- LangGraph Pattern Configuration ---

class LangGraphNodeConfig(BaseModel):
    """Configuration of a node in a LangGraph workflow."""
    name: str = Field(..., description="Node name")
    description: Optional[str] = Field(None, description="Node purpose description")
    agent_ref: Optional[str] = Field(None, description="Agent spec reference or tool name executed by node")
    function_name: Optional[str] = Field(None, description="Python callable/node handler function name")
    source_file: Optional[str] = Field(None, description="Path to file defining the node function")


class LangGraphEdgeConfig(BaseModel):
    """Direct edge between two nodes in a LangGraph."""
    from_node: str = Field(..., description="Source node name")
    to_node: str = Field(..., description="Target node name")


class LangGraphConditionalEdgeConfig(BaseModel):
    """Conditional branching edge in a LangGraph."""
    source_node: str = Field(..., description="Source node name where branching occurs")
    condition_function: str = Field(..., description="Function name that evaluates state and returns route key")
    route_map: Dict[str, str] = Field(..., description="Mapping of condition return values to target node names")


class LangGraphConfig(BaseModel):
    """Configuration for LangGraph stateful cyclical workflows."""
    graph_type: Literal["state_graph", "message_graph"] = Field("state_graph", description="Type of LangGraph")
    entry_point: str = Field(..., description="Starting node name")
    finish_point: Optional[str] = Field(None, description="Termination node name (defaults to END)")
    nodes: List[LangGraphNodeConfig] = Field(..., description="List of graph nodes")
    edges: List[LangGraphEdgeConfig] = Field(default_factory=list, description="Direct edges")
    conditional_edges: List[LangGraphConditionalEdgeConfig] = Field(
        default_factory=list, description="Conditional routing edges"
    )
    checkpointer: Literal["memory", "redis", "cosmos", "sqlite"] = Field(
        "memory", description="State persistence checkpointer"
    )


# --- Framework First-Route Configuration ---

class FrameworkRouteRule(BaseModel):
    """A rule mapping tasks/intents to target framework routes."""
    intent: str = Field(..., description="Intent keyword or category (e.g., 'workflow', 'debate', 'code_review')")
    target_framework: Literal["autogen", "langgraph", "react"] = Field(
        ..., description="Target agentic framework route"
    )
    target_agent_ref: Optional[str] = Field(None, description="Target agent spec file for this route")


class FrameworkRouterConfig(BaseModel):
    """Configuration for routing incoming tasks to AutoGen, LangGraph, or ReAct first."""
    strategy: Literal["intent_classifier", "heuristic", "llm_as_judge"] = Field(
        "intent_classifier", description="Routing strategy"
    )
    default_route: Literal["autogen", "langgraph", "react"] = Field(
        "react", description="Default route when no specific rule matches"
    )
    routes: List[FrameworkRouteRule] = Field(default_factory=list, description="List of framework routing rules")
    autogen_agent_ref: Optional[str] = Field(None, description="Reference to AutoGen agent spec")
    langgraph_agent_ref: Optional[str] = Field(None, description="Reference to LangGraph agent spec")
    react_agent_ref: Optional[str] = Field(None, description="Reference to ReAct agent spec")

