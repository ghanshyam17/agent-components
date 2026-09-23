# Diagrams — agent-components

Architecture, UML, and code-flow for the agentic design-pattern chassis.

| File | View | Format |
|---|---|---|
| [`architecture.html`](architecture.html) | Layered architecture (schema → patterns → engineering/plugins → deploy/scaffold/sandbox → projects) | Standalone dark-theme SVG/HTML |
| [`uml.md`](uml.md) | Class diagrams (AgentPattern hierarchy, schema layer, plugins, deployer, component graph) | Mermaid — renders on GitHub |
| [`code-flow.md`](code-flow.md) | Spec→pattern, framework-router, sequential, plugin, deploy, scaffold sequence diagrams | Mermaid — renders on GitHub |

## How they relate

- **architecture.html** — the *what*: five layers, one declarative spec flowing
  into a pattern library, engineering runtime, plugin registry, and Azure deploy
  targets.
- **uml.md** — the *types*: `AgentPattern` as the abstract base with eight concrete
  patterns, the pydantic schema layer, the plugin registry, and the deployer.
- **code-flow.md** — the *execution*: how a YAML spec becomes a running pattern,
  how the framework-router classifies intent, and how `agcomps` deploys to Foundry.

## Accuracy note

This is a **reference chassis** — 21 test files, patterns fully implemented,
schema validated by pydantic. Some layers carry explicit stubs (e.g.
`PluginRegistry.compose_tools` and `EvalHarness.run` are documented placeholders);
the diagrams mark real implementations vs. mapped seams faithfully.
