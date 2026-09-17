"""Token counting.

Every context-management decision reduces to an integer: how many tokens will
this text cost? Get it wrong and the failure is asymmetric:

* **Undercount** → the request exceeds the real window and the API rejects it.
  This is the dangerous direction.
* **Overcount** → we waste budget on empty space. Annoying, not fatal.

So the default counter is deliberately *conservative*: it rounds up and never
claims fewer tokens than the text can plausibly need.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Protocol, runtime_checkable

# Words are more than one token: punctuation, spacing and subword splits all
# add tokens. 1.35 is a deliberate over-estimate for English prose.
_TOKENS_PER_WORD = 1.35
_CHARS_PER_TOKEN = 4.0
_WHITESPACE = re.compile(r"\s+")


@runtime_checkable
class TokenCounter(Protocol):
    """Anything that can price a string in tokens."""

    name: str

    def count(self, text: str) -> int: ...

    def count_message(self, role: str, text: str) -> int: ...

    def describe(self) -> dict[str, object]: ...


class HeuristicCounter:
    """Dependency-free, deterministic, conservative token estimate.

    Takes the *maximum* of a word-based and a character-based estimate. Either
    heuristic alone under-counts some real input — CJK text has few "words" but
    many tokens; a long identifier is one word but many subword tokens — so the
    max is the safer estimator.
    """

    name = "heuristic"

    def __init__(self, per_message_overhead: int = 4) -> None:
        self.per_message_overhead = per_message_overhead

    def count(self, text: str) -> int:
        if not text:
            return 0
        words = len(_WHITESPACE.split(text.strip()))
        by_words = math.ceil(words * _TOKENS_PER_WORD)
        by_chars = math.ceil(len(text) / _CHARS_PER_TOKEN)
        return max(by_words, by_chars)

    def count_message(self, role: str, text: str) -> int:
        """A message costs its content plus per-message framing tokens."""
        return self.count(text) + self.count(role) + self.per_message_overhead

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tokens_per_word": _TOKENS_PER_WORD,
            "chars_per_token": _CHARS_PER_TOKEN,
            "message_overhead": self.per_message_overhead,
        }


class CallableCounter:
    """Wrap a real tokenizer (tiktoken, a HF tokenizer, a tokenizer endpoint).

    The callable must return an exact count. Because an exact count is
    trustworthy, it is used as given rather than padded.
    """

    name = "callable"

    def __init__(self, fn: Callable[[str], int], name: str = "callable") -> None:
        self._fn = fn
        self.name = name

    def count(self, text: str) -> int:
        if not text:
            return 0
        n = int(self._fn(text))
        if n < 0:
            raise ValueError("tokenizer returned a negative count")
        return n

    def count_message(self, role: str, text: str) -> int:
        return self.count(text) + self.count(role) + 4

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "exact": True}


class CachingCounter:
    """Memoise a counter. Assembling the same history repeatedly is common.

    A no-op unless the underlying count is deterministic, which is why this
    wraps rather than being built into the base.
    """

    def __init__(self, inner: TokenCounter, max_entries: int = 4096) -> None:
        self._inner = inner
        self._cache: dict[str, int] = {}
        self._max = max_entries
        self.name = f"cached:{inner.name}"

    def count(self, text: str) -> int:
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        if len(self._cache) >= self._max:
            # Bound memory; a context manager must not become the leak.
            self._cache.clear()
        val = self._inner.count(text)
        self._cache[text] = val
        return val

    def count_message(self, role: str, text: str) -> int:
        return self.count(text) + self.count(role) + 4

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "inner": self._inner.describe(), "cached": len(self._cache)}


def build_counter(kind: str = "heuristic", **kwargs) -> TokenCounter:
    """Factory. ``tiktoken`` is optional — absent means heuristic, not failure."""
    if kind == "heuristic":
        return HeuristicCounter(**kwargs)
    if kind == "tiktoken":
        try:
            import tiktoken  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "tiktoken is not installed; use kind='heuristic' or install tiktoken"
            ) from exc
        enc = tiktoken.get_encoding(kwargs.pop("encoding", "cl100k_base"))
        return CallableCounter(lambda s: len(enc.encode(s)), name="tiktoken")
    if kind == "callable":
        return CallableCounter(kwargs["fn"], name=kwargs.get("name", "callable"))
    raise ValueError(f"unknown counter kind: {kind!r}")
