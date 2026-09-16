"""Synthetic content generation from a frontier teacher model.

The generator drives a teacher through ``model-gateway`` (so the teacher can be
a local vLLM replica or a cloud endpoint without changing this code) and
captures two things per example:

* the **completion** — the answer to distil,
* the **teacher_reasoning** — the chain-of-thought behind it.

Reasoning traces are the point. A small student trained only on (prompt, answer)
pairs learns to imitate surface form; trained on the teacher's reasoning it
learns the *procedure*, which is what generalises. The trace is captured
separately rather than concatenated into the completion so the curator can score
and filter the two independently.

Generating multiple distinct completions per prompt also mines preference pairs
for DPO: given several candidates, the best and worst become ``chosen`` and
``rejected``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Any, Awaitable, Callable, Sequence

from components_core import Message

from distillation.models import (
    ContentSample,
    DatasetFormat,
    DistillationConfig,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SyntheticContentGenerator",
    "TeacherClient",
    "DEFAULT_SEED_PROMPTS",
    "STYLE_BLUEPRINTS",
]

#: A teacher call: (messages, **kw) -> (content, tool_calls). Matches the
#: `.chat()` shape of both `model_gateway.GatewayClient` and the router's
#: `ModelClient`, so either can be injected.
TeacherClient = Callable[..., Awaitable[tuple[str, list[Any]]]]


#: Style/format blueprints. These are the *prompt-registry* templates' default
#: content: each turns a bare topic into a distinct generation task, which is
#: what produces format diversity in the dataset rather than near-identical
#: samples. Kept inline as defaults so the component works with an empty
#: registry; `register_blueprints()` pushes them into a PromptRegistry.
STYLE_BLUEPRINTS: dict[str, dict[str, str]] = {
    "explanation": {
        "system": "You are an expert subject-matter teacher.",
        "template": (
            "Explain the concept of {topic} to a competent beginner.\n\n"
            "Work through your reasoning step by step inside  thinking...</think> "
            "tags, then give a clear final explanation."
        ),
    },
    "worked_example": {
        "system": "You are a meticulous instructor who teaches by example.",
        "template": (
            "Construct a fully worked example that demonstrates {topic}.\n\n"
            "Think through the solution approach inside  thinking...</think> tags, "
            "then present the worked solution with each step shown."
        ),
    },
    "socratic": {
        "system": "You are a tutor who leads learners to insight with questions.",
        "template": (
            "Help a learner understand {topic} by asking a short sequence of "
            "guiding questions and answering each one.\n\n"
            "Plan your question sequence inside  thinking...</think> tags first."
        ),
    },
    "debugging": {
        "system": "You are a senior engineer reviewing a flawed approach.",
        "template": (
            "Describe a common mistake people make regarding {topic} and how to "
            "correct it.\n\n"
            "Reason about why the mistake is tempting inside  thinking...</think> "
            "tags, then state the correction."
        ),
    },
    "summarisation": {
        "system": "You are a concise technical summariser.",
        "template": (
            "Summarise the essential facts about {topic} as a structured list.\n\n"
            "Decide what is essential inside  thinking...</think> tags, then give "
            "the summary."
        ),
    },
}


DEFAULT_SEED_PROMPTS: list[str] = [
    "the Medallion lakehouse architecture (Bronze, Silver, Gold)",
    "idempotent data pipeline design",
    "the difference between precision and recall",
    "how gradient descent reaches a local minimum",
    "when to use a message queue instead of a synchronous call",
    "feature stores and training/serving skew",
    "retrieval-augmented generation and why chunking matters",
    "the CAP theorem and its practical trade-offs",
    "prompt injection and how to defend against it",
    "model distillation from a large teacher to a small student",
]

_THINK_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Separate a `` thinking...</think>`` trace from the answer that follows.

    Returns ``(reasoning, completion)``. Teachers that ignore the instruction to
    emit a trace are handled gracefully: the whole response becomes the
    completion and ``reasoning`` is None — a sample without a trace is still
    usable, just less valuable.
    """
    if not text:
        return None, ""
    match = _THINK_RE.search(text)
    if match is None:
        return None, text.strip()
    reasoning = match.group(1).strip()
    completion = (text[: match.start()] + text[match.end():]).strip()
    return (reasoning or None), completion


class SyntheticContentGenerator:
    """Generate synthetic examples (and DPO pairs) from a teacher model.

    Parameters
    ----------
    client:
        Teacher callable. Any object with an OpenAI-shaped ``chat(messages, ...)``
        coroutine works — a ``model_gateway`` ``GatewayClient`` is the intended
        one. When omitted, the generator is *inert*: ``generate`` returns an
        empty list rather than raising, so a pipeline can be constructed before
        its teacher is wired.
    registry:
        Optional ``prompt_registry.PromptRegistry``. When supplied, blueprints
        are rendered through it (giving versioning and A/B for free); otherwise
        the inline templates are formatted directly.
    rng:
        Seeded ``random.Random`` for reproducible topic/blueprint shuffling.
    """

    def __init__(
        self,
        client: TeacherClient | None = None,
        *,
        registry: Any | None = None,
        rng: random.Random | None = None,
        blueprints: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.rng = rng or random.Random(0)
        self._seed = getattr(self.rng, "_seed_used", 0)
        self.blueprints = blueprints or STYLE_BLUEPRINTS

    # ---- prompt construction --------------------------------------------- #
    def register_blueprints(self, registry: Any) -> int:
        """Push the blueprints into a PromptRegistry; returns how many were added.

        Registered as version ``1.0.0`` under names like ``distill_explanation``
        so they coexist with an application's own prompts.
        """
        from prompt_registry import PromptTemplate

        added = 0
        for name, bp in self.blueprints.items():
            try:
                # The inline templates use str.format braces ({topic}) because
                # that is how build_messages() interpolates them without a
                # registry. PromptTemplate renders with Jinja, where {topic} is
                # literal text and would ship an unsubstituted placeholder to the
                # teacher — so convert the braces on registration.
                registry.register(
                    PromptTemplate(
                        name=f"distill_{name}",
                        version="1.0.0",
                        template=bp["template"].replace("{topic}", "{{ topic }}"),
                        description=bp["system"],
                        variables=["topic"],
                    )
                )
                added += 1
            except Exception as e:  # noqa: BLE001 - a bad blueprint must not stop the run
                logger.warning("could not register blueprint %s: %s", name, e)
        return added

    def build_messages(self, topic: str, blueprint: str) -> list[Message]:
        """Render the teacher prompt for one (topic, blueprint) pair."""
        bp = self.blueprints.get(blueprint) or next(iter(self.blueprints.values()))
        if self.registry is not None:
            try:
                rendered = self.registry.render(f"distill_{blueprint}", topic=topic)
                return [
                    Message(role="system", content=bp["system"]),
                    Message(role="user", content=rendered),
                ]
            except Exception as e:  # noqa: BLE001 - fall back to the inline template
                logger.debug("registry render failed for %s: %s", blueprint, e)
        return [
            Message(role="system", content=bp["system"]),
            Message(role="user", content=bp["template"].format(topic=topic)),
        ]

    def plan(self, config: DistillationConfig) -> list[tuple[str, str]]:
        """Decide which (topic, blueprint) pairs to generate.

        A fixed plan makes a run reproducible and lets ``generate`` report the
        intended shape before any model call happens.
        """
        topics = list(config.seed_prompts) or list(DEFAULT_SEED_PROMPTS)
        names = list(self.blueprints)
        # Build the full topic × blueprint product, then truncate and shuffle.
        # Advancing the blueprint only once every `len(topics)` samples (the
        # obvious-looking `i // len(topics)`) collapses to a single blueprint
        # whenever `num_samples <= len(topics)` — every example then shares one
        # style, which defeats the format diversity the dataset is for.
        combos: list[tuple[str, str]] = [(t, b) for t in topics for b in names]
        while len(combos) < config.num_samples:
            combos = combos * 2
        out = combos[: config.num_samples]
        # Shuffle via a *seeded throwaway* RNG rather than self.rng. A stateful
        # shared RNG advances on every call, so plan() would return a different
        # order each time — making the same config produce different datasets
        # and breaking reproducibility, which is the only reason the seed exists.
        random.Random(getattr(self, "_seed", 0)).shuffle(out)
        return out

    # ---- generation ------------------------------------------------------- #
    async def generate_one(
        self,
        topic: str,
        blueprint: str,
        *,
        config: DistillationConfig,
        temperature: float = 0.7,
    ) -> ContentSample | None:
        """Generate a single sample; None when the teacher is unavailable."""
        if self.client is None:
            return None
        messages = self.build_messages(topic, blueprint)
        content, _calls = await self.client(
            messages, temperature=temperature, max_tokens=None
        )
        reasoning, completion = split_reasoning(content or "")
        if not completion:
            logger.debug("teacher returned nothing for %r/%s", topic, blueprint)
            return None
        return ContentSample(
            prompt=messages[-1].content,
            completion=completion,
            teacher_reasoning=reasoning,
            format_type=config.format,
            teacher_model=config.teacher_model,
            metadata={
                "topic": topic,
                "blueprint": blueprint,
                "has_reasoning": reasoning is not None,
            },
        )

    async def generate(
        self,
        config: DistillationConfig,
        *,
        concurrency: int = 4,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ContentSample]:
        """Generate ``config.num_samples`` samples, concurrently.

        Failures are swallowed per-sample (logged, not raised): one bad teacher
        response should not discard a run of fifty. The caller sees whatever
        succeeded and compares it against ``len(plan)``.
        """
        if self.client is None:
            logger.warning("no teacher client bound; generate() produced nothing")
            return []

        plan = self.plan(config)
        sem = asyncio.Semaphore(max(1, concurrency))
        done = 0

        async def _one(topic: str, bp: str) -> ContentSample | None:
            nonlocal done
            async with sem:
                try:
                    sample = await self.generate_one(topic, bp, config=config)
                except Exception as e:  # noqa: BLE001
                    logger.warning("generation failed for %r: %s", topic, e)
                    sample = None
                done += 1
                if on_progress is not None:
                    try:
                        on_progress(done, len(plan))
                    except Exception:  # noqa: BLE001 - a reporter must not break the run
                        pass
                return sample

        results = await asyncio.gather(*(_one(t, b) for t, b in plan))
        return [s for s in results if s is not None]

    # ---- preference pairs ------------------------------------------------- #
    async def generate_preference_pairs(
        self,
        config: DistillationConfig,
        *,
        candidates: int = 2,
    ) -> list[tuple[str, list[str]]]:
        """Generate several completions per prompt, for DPO pair mining.

        Returns ``(prompt, [completions])``. The caller scores them and builds
        ``DPOPair``s. Sampling temperature is raised per extra candidate so the
        completions actually differ — identical candidates yield a zero-signal
        preference pair.
        """
        if self.client is None:
            return []
        plan = self.plan(config)[: max(1, config.num_samples // max(1, candidates))]
        out: list[tuple[str, list[str]]] = []
        for topic, bp in plan:
            messages = self.build_messages(topic, bp)
            texts: list[str] = []
            for k in range(max(2, candidates)):
                try:
                    content, _ = await self.client(
                        messages, temperature=min(0.5 + 0.2 * k, 1.2), max_tokens=None
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug("preference candidate failed: %s", e)
                    continue
                _reasoning, completion = split_reasoning(content or "")
                if completion:
                    texts.append(completion)
            if len(texts) >= 2:
                out.append((messages[-1].content, texts))
        return out

    # ---- convenience ------------------------------------------------------ #
    @staticmethod
    def stats(samples: Sequence[ContentSample]) -> dict[str, Any]:
        """Quick shape report on a generated batch."""
        n = len(samples)
        with_reasoning = sum(1 for s in samples if s.teacher_reasoning)
        topics = {str(s.metadata.get("topic", "")) for s in samples}
        return {
            "samples": n,
            "with_reasoning": with_reasoning,
            "reasoning_rate": round(with_reasoning / n, 3) if n else 0.0,
            "distinct_topics": len({t for t in topics if t}),
            "avg_completion_chars": (
                round(sum(len(s.completion) for s in samples) / n, 1) if n else 0.0
            ),
        }
