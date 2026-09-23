# UML Mapping — agent-components

> Class-level map of the agentic chassis. The spine: `AgentPattern` is the abstract
> base every pattern implements, `PatternFactory` registers them, and the schema
> layer validates the declarative YAML that wires a pattern to a model, tools,
> memory, and infra.

## 1. The pattern base + factory

```mermaid
classDiagram
    class AgentPattern {
        <<abstract>>
        +name : str
        +run(context) : PatternResult
        +stream(context) : AsyncIterator
        +to_foundry_config() : dict
    }
    class PatternContext {
        +task : str
        +session_id : str
        +config : dict
        +tools : Registry
        +memory : SessionState
        +model_clients : dict
    }
    class PatternResult {
        +final_answer : str
        +metadata : dict
        +events_log : list
        +token_usage : dict
        +duration_ms : int
    }
    class PatternFactory {
        +_registry : dict
        +register(name, cls)
        +create(name, kwargs) : AgentPattern
        +available() : list
    }
    AgentPattern --> PatternContext : run(stream)
    AgentPattern --> PatternResult : returns
    PatternFactory --> AgentPattern : registers / creates
```

## 2. The eight pattern implementations

```mermaid
classDiagram
    class AgentPattern { <<abstract>> }
    class ReactPattern { +name = "react" }
    class SequentialPattern { +stages : list[AgentPattern] }
    class SupervisorPattern { +workers }
    class NetworkPattern { +peers }
    class MapReducePattern { +map +reduce }
    class AutoGenPattern { +agents }
    class LangGraphPattern { +graph }
    class FrameworkRouterPattern {
        +default_route
        +autogen_pattern
        +langgraph_pattern
        +react_pattern
        +classify_first_route(task)
    }
    AgentPattern <|-- ReactPattern
    AgentPattern <|-- SequentialPattern
    AgentPattern <|-- SupervisorPattern
    AgentPattern <|-- NetworkPattern
    AgentPattern <|-- MapReducePattern
    AgentPattern <|-- AutoGenPattern
    AgentPattern <|-- LangGraphPattern
    AgentPattern <|-- FrameworkRouterPattern
    FrameworkRouterPattern --> AutoGenPattern : delegates
    FrameworkRouterPattern --> LangGraphPattern : delegates
    FrameworkRouterPattern --> ReactPattern : delegates
```

## 3. The schema layer

```mermaid
classDiagram
    class ProjectSpec {
        +apiVersion = "agentcomponents/v1"
        +kind = "Project"
        +metadata : ProjectMetadata
        +spec : ProjectSpecModel
    }
    class AgentSpec {
        +apiVersion
        +kind = "Agent"
        +metadata : Metadata
        +spec : AgentSpecModel
    }
    class AgentSpecModel {
        +pattern : Literal[react..framework_router]
        +model : ModelConfig
        +tools : ToolsConfig
        +memory : MemoryConfig
        +guardrails : GuardrailsConfig
        +infrastructure : InfrastructureConfig
        +supervisor / network / sequential / map_reduce / autogen / langgraph / framework_router
    }
    class ModelConfig {
        +provider : Literal[azure-foundry, openai, vllm, ollama]
        +deployment_name
        +tier_strategy
    }
    class InfrastructureSpec { +compute +data +ml planes }
    class ComponentGraph { +DE +AML +tools +agents }
    AgentSpec *-- AgentSpecModel
    AgentSpecModel *-- ModelConfig
    AgentSpecModel ..> InfrastructureConfig
```

## 4. Plugins + engineering

```mermaid
classDiagram
    class PluginLoader {
        +load(path) : LoadedPlugin
    }
    class PluginRegistry {
        +register(plugin)
        +get(name)
        +compose_tools(names)
        +compose_prompts(names)
        +resolve_for_agent(agent_spec) : ResolvedPlugins
    }
    class ResolvedPlugins {
        +tools : Registry
        +prompts : dict
        +guardrails : list
        +connectors : list
    }
    class EvalHarness {
        +spec : EvalSpec
        +run() : EvalResult
        +generate_report(result)
        +compare(results)
    }
    class EvalSpec {
        +agent_ref
        +dataset_ref
        +metrics
        +thresholds
    }
    PluginLoader --> PluginRegistry : loads into
    PluginRegistry --> ResolvedPlugins : resolve_for_agent()
```

## 5. The deployer

```mermaid
classDiagram
    class FoundryDeployer {
        +plan(spec)
        +deploy(spec) : DeployResult
    }
    class DeployResult {
        +success : bool
        +agent_name
        +version
        +endpoints
        +errors
    }
    class BicepGenerator {
        +generate()
    }
    class ContainerAppsDeployer
    class FunctionsDeployer
    class AppServiceDeployer
    FoundryDeployer --> DeployResult : returns
```

## 6. Component graph — the end-to-end wiring

```mermaid
flowchart LR
    DE["Data Engineering"] --> AML["AML / Model"]
    AML --> TOOLS["Tools"]
    TOOLS --> AGENTS["Agents"]
    AGENTS --> FOUNDRY["Azure AI Foundry"]
    DE --> FOUNDRY
    AML --> FOUNDRY
```
