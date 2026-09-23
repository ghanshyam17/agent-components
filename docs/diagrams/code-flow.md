# Code Flow — agent-components

> The three flows that define the chassis: **spec → pattern → runtime**, **the
> framework-router decision**, and **spec → deploy**.

## 1. Spec loading → pattern instantiation

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant L as loader.load_spec()
    participant S as schema (pydantic)
    participant F as PatternFactory

    U->>L: load_spec(path)
    L->>L: read file, parse YAML/JSON
    L->>L: resolve_variables (${ENV})
    L->>L: resolve_includes (relative paths)
    L->>S: validate_spec(data) by kind
    alt kind = Project
        S-->>L: ProjectSpec
    else kind = Agent
        S-->>L: AgentSpec
    else kind = Infrastructure
        S-->>L: InfrastructureSpec
    else kind = ComponentGraph
        S-->>L: ComponentGraph
    else unknown
        S-->>L: ValueError (never guess)
    end
    L->>F: create(pattern, **kwargs)
    F-->>U: AgentPattern instance
```

## 2. Framework router — first-route classification

```mermaid
sequenceDiagram
    autonumber
    participant C as PatternContext (task)
    participant R as FrameworkRouterPattern
    participant AG as AutoGenPattern
    participant LG as LangGraphPattern
    participant RE as ReactPattern

    C->>R: stream(context)
    R->>R: classify_first_route(task)
    Note over R: custom routes first, then AUTOGEN_SIGNALS (debate/collaborate),<br/>then LANGGRAPH_SIGNALS (workflow/state machine), else default (ReAct)
    alt autogen signal
        R-->>C: route event {autogen}
        R->>AG: stream(context) — delegate
    else langgraph signal
        R-->>C: route event {langgraph}
        R->>LG: stream(context) — delegate
    else default
        R-->>C: route event {react}
        R->>RE: stream(context) — delegate
    end
```

## 3. Sequential pattern — context threading

```mermaid
sequenceDiagram
    autonumber
    participant S as SequentialPattern
    participant A as Stage 0
    participant B as Stage 1

    S->>S: current_task = context.task
    loop each stage
        S->>A: subtask_start
        S->>A: stream(sub_context)
        A-->>S: events, final answer
        S->>S: current_task = stage_answer
        Note over S: stage N's output becomes stage N+1's input
    end
    S-->>S: final event {content: current_task}
```

## 4. Plugin resolution

```mermaid
sequenceDiagram
    autonumber
    participant A as AgentSpec
    participant R as PluginRegistry
    participant L as PluginLoader

    L->>R: register(LoadedPlugin) per plugin
    A->>R: resolve_for_agent(agent_spec)
    R->>R: compose_tools(names)
    R->>R: compose_prompts(names)
    R->>R: collect guardrails + connectors
    R-->>A: ResolvedPlugins {tools, prompts, guardrails, connectors}
```

## 5. Deploy flow (`agcomps deploy`)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as agcomps (click)
    participant L as load_spec
    participant D as FoundryDeployer

    CLI->>L: load_spec(project)
    L-->>CLI: ProjectSpec | AgentSpec
    CLI->>D: FoundryDeployer(config)
    alt --dry-run
        CLI->>D: plan(spec)
        D-->>CLI: JSON plan (nothing deployed)
    else deploy
        CLI->>D: deploy(spec)
        D-->>CLI: DeployResult {success, agent_name, version, endpoints}
    end
    CLI-->>CLI: echo result / ClickException on failure
```

## 6. Scaffold flow (`agcomps new`)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as agcomps (click)
    participant G as generator.scaffold_project
    participant T as templates/

    CLI->>G: ScaffoldConfig(pattern, name, output, monorepo, template_dir)
    G->>T: select template by pattern (langgraph / automation / ...)
    G-->>CLI: scaffolded project path
```
