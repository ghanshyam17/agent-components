# context-manager

**Context window management** for agent pipelines. Decides what goes into the
window, what gets rewritten, and what gets left out — and records why.

Sibling to `memory-store`, which solves a different problem: `memory-store`
*stores* conversation and vector memory; `context-manager` decides which of it
fits in the model's window right now. Storing everything and sending it all is
how a working prototype becomes a production incident.

## The idea

A context window is a **budget, not a buffer**. Everything here follows from
taking that seriously:

| Principle | Why |
|---|---|
| **Reserve the output before spending the input** | A prompt that fills the window leaves no room to answer; the request fails at the API boundary. The most common way a context bug reaches production. |
| **Protected content is admitted unconditionally** | A system prompt does not compete with a log dump. If it alone overflows, that is reported as `over_budget` rather than silently dropping the instructions. |
| **Evict by priority, never by arrival** | Truncating the oldest messages is the naive default and it discards the task statement first. |
| **Deduplicate by content** | The same retrieved chunk twice costs budget twice and *looks* like corroboration. |
| **Over-count tokens, never under-count** | Under-counting means a rejected request; over-counting means wasted space. The failure modes are not symmetric. |
| **Record every decision** | "Why didn't the model know X?" is answerable only if assembly was logged at the time. |

## Install

```bash
uv sync --all-packages      # from the monorepo root
```

## Usage

```python
from context_manager import ContextBudget, ContextAssembler, ContextSegment, SegmentPriority

budget = ContextBudget(total_tokens=8192, reserve_output=1024)
assembler = ContextAssembler(budget)

package = await assembler.assemble([
    ContextSegment(role="system", content=SYSTEM_PROMPT,
                   priority=SegmentPriority.CRITICAL, pinned=True, compactable=False),
    *[ContextSegment(role="user", content=m, source="history", sequence=i)
      for i, m in enumerate(history)],
    *[ContextSegment(role="retrieval", content=d, priority=SegmentPriority.LOW,
                     source="retrieval")
      for d in retrieved_docs],
])

await llm(package.to_openai_messages())   # what actually goes to the model
print(package.explain())                  # what was dropped, and why
```

The one-liner form, for the common `system + history` shape:

```python
package = await SingleShotAssembler().build(SYSTEM_PROMPT, history, query=user_question)
```

## Decision order

The order is the design; each step exists because doing it later produces a
specific failure.

```
1. price          → no decision is possible before the budget is known
2. deduplicate    → a duplicate costs budget twice and looks like evidence
3. split          → protected content is not competing for space
4. admit protected → unconditionally; report over_budget if it alone overflows
5. source caps    → compact before dropping; no source monopolises the window
6. segment ceiling → one huge tool result cannot eat the window
7. greedy fill    → by priority, most recent first; compact on a near miss
8. emit           → sequence order, framing hoisted to the front
```

## Compaction strategies

Compaction is lossy and irreversible, so it always follows dropping something
genuinely worthless — and each strategy documents its own failure mode:

| Strategy | Keeps | Honest failure mode |
|---|---|---|
| `truncate` | Head and tail, elides the middle | Loses connections between distant parts. Head+tail beats head-only because the middle is usually the redundant part. |
| `extractive` | Sentences matching the query, in original order | Preserves statements without their relationships — reasoning degrades into assertive fragments. |
| `summarize` | Reasoning, via a model | **Can invent.** A summary is a new claim about the text, so it is marked as compacted rather than trusted silently. |
| `hybrid` | Extractive first, model only if still over target | The default: deterministic wherever determinism suffices. |

Order preservation in `extractive` is deliberate — reordering a passage changes
its argument even when every sentence survives.

## What it does not do

- **No storage.** Memory persistence is `memory-store`'s job; this component
  takes segments and returns messages.
- **No retrieval.** It decides how retrieved text is *budgeted*, not what to
  retrieve — that is `retriever`.
- **No summarisation of long-term memory.** Compaction shrinks one segment
  under pressure; it is not a memory hierarchy.
