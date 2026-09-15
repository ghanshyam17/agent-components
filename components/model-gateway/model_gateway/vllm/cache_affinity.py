"""KV-cache prefix affinity: route by shared prompt prefix, not by luck.

vLLM caches computed prompt prefixes per instance (PagedAttention automatic
prefix caching). A request whose opening tokens a replica has already
processed pays only for the *new* suffix, so sending requests that share a
system prompt to the replica that most recently served that prefix raises
cache hit rate and cuts TTFT.

Two structures cooperate here:

* :class:`RadixTree` — a token-keyed prefix tree recording which replicas have
  *observed* a given prefix. It answers "which replicas already hold a cache
  for this opening?".
* :class:`ConsistentHashRing` — maps a prefix key to a preferred replica
  deterministically. It answers "which replica *should* hold this prefix?" and
  keeps that answer stable as replicas come and go (only ~1/N of keys move
  when a replica is added or removed, unlike ``hash % n``).

:class:`PrefixCacheAffinity` uses the ring as the tie-break/default and the
tree as the learned signal, so affinity is correct on a cold start (ring) and
gets better as traffic arrives (tree).

No third-party dependencies: hashing is ``hashlib.blake2b`` and ring lookup is
``bisect``.
"""
from __future__ import annotations

import bisect
import hashlib
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "ConsistentHashRing",
    "RadixTree",
    "PrefixCacheAffinity",
    "tokens_for_messages",
    "message_role",
    "message_text",
]

# Separator used when hashing a token sequence, so ("ab","c") != ("a","bc").
_SEP = "\x1f"


def _hash32(text: str, salt: str = "") -> int:
    """Stable 64-bit hash of `text` (process-independent, unlike hash())."""
    digest = hashlib.blake2b(
        (salt + _SEP + text).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def _tokenize(text: str) -> list[str]:
    """Split text into prefix tokens.

    Whitespace tokens are deliberate: they are stable across tokenizer
    versions, and prefix caching only cares that *identical* text produces
    *identical* tokens. A shared system prompt necessarily yields an identical
    leading token sequence.
    """
    return text.split()


# --------------------------------------------------------------------------- #
# Message helpers — accept components_core.Message, dicts, or anything with
# .role/.content, so the affinity layer stays usable from tests and from the
# OpenAI-shaped path alike.
# --------------------------------------------------------------------------- #
def message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def message_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # OpenAI content-parts form: [{"type": "text", "text": "..."}, ...]
    if isinstance(content, Sequence):
        parts = [
            str(p.get("text", "")) for p in content
            if isinstance(p, dict) and p.get("type") in (None, "text")
        ]
        return " ".join(parts)
    return str(content)


def tokens_for_messages(messages: Sequence[Any], *, limit: int = 64) -> list[str]:
    """Extract the cacheable prefix of a conversation as tokens.

    The **system prompt comes first** because it is the stable, widely shared
    part of a request; the earliest conversation turns follow, so two requests
    that begin identically collapse to the same prefix. `limit` caps how much
    prefix is considered: past a point the extra tokens are more likely to
    *diverge* than to be shared, and a long key dilutes the affinity signal.
    """
    system_text = " ".join(
        message_text(m) for m in messages if message_role(m) == "system"
    )
    conversation = " ".join(
        message_text(m) for m in messages if message_role(m) != "system"
    )
    tokens = _tokenize(system_text) + _tokenize(conversation)
    return tokens[:limit]


# --------------------------------------------------------------------------- #
# Radix tree over token sequences
# --------------------------------------------------------------------------- #
@dataclass
class _Node:
    """One token on an edge of the prefix tree."""

    children: dict[str, "_Node"] = field(default_factory=dict)
    # replicas observed to have served a request ending at (or through) here
    replicas: Counter[str] = field(default_factory=Counter)


class RadixTree:
    """Token-keyed prefix tree recording which replicas saw which prefix.

    A trie is a radix tree with radix 1 (one token per edge); storing a token
    per edge keeps insertion and lookup O(len(tokens)) and avoids the edge
    splitting a compressed variant would need for no gain at this depth.

    Lookup is by *longest match*: :meth:`lookup` walks as far as the stored
    tokens allow and returns the replica counts accumulated at that deepest
    node. A request may therefore match a prefix shorter than its own and
    still get a useful affinity signal — which is exactly what partial prefix
    caching gives you.
    """

    def __init__(self) -> None:
        self._root = _Node()
        self._lock = threading.Lock()
        self._observations = 0

    def observe(self, tokens: Sequence[str], replica: str) -> None:
        """Record that `replica` served a request whose prefix was `tokens`."""
        if not tokens:
            return
        with self._lock:
            node = self._root
            for token in tokens:
                node = node.children.setdefault(token, _Node())
            node.replicas[replica] += 1
            self._observations += 1

    def lookup(self, tokens: Sequence[str]) -> Counter[str]:
        """Replica counts at the deepest node matching a prefix of `tokens`."""
        node = self._root
        deepest = node
        for token in tokens:
            nxt = node.children.get(token)
            if nxt is None:
                break
            node = nxt
            if node.replicas:
                # Remember the deepest node that actually carries an
                # observation; intermediate nodes on the path may be empty.
                deepest = node
        return Counter(deepest.replicas)

    def clear(self) -> None:
        with self._lock:
            self._root = _Node()
            self._observations = 0

    @property
    def observations(self) -> int:
        return self._observations

    def __len__(self) -> int:
        """Number of distinct prefixes with at least one observation."""
        total = 0
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node.replicas:
                total += 1
            stack.extend(node.children.values())
        return total


# --------------------------------------------------------------------------- #
# Consistent hash ring
# --------------------------------------------------------------------------- #
class ConsistentHashRing:
    """Consistent-hash ring with weighted virtual nodes.

    Each replica owns ``virtual_nodes * weight`` points on a 64-bit ring.
    Adding or removing a replica moves only the keys in the arcs it claims
    (~1/N of keys), so prefix affinity survives replica churn — the property
    plain ``hash(key) % len(replicas)`` lacks, and the reason a request that
    was cache-warm on replica B is not suddenly sent to C.
    """

    def __init__(self, replicas: Iterable[str] | None = None, *, virtual_nodes: int = 160) -> None:
        if virtual_nodes < 1:
            raise ValueError("virtual_nodes must be >= 1")
        self.virtual_nodes = virtual_nodes
        self._lock = threading.Lock()
        self._points: list[tuple[int, str]] = []
        self._weights: dict[str, int] = {}
        for replica in replicas or ():
            self.add(replica)

    def add(self, replica: str, weight: int = 1) -> None:
        """Add (or re-weight) a replica."""
        if weight < 1:
            raise ValueError("weight must be >= 1")
        with self._lock:
            if replica in self._weights:
                self.remove(replica)
            self._weights[replica] = weight
            for i in range(self.virtual_nodes * weight):
                self._points.append((_hash32(f"{replica}#{i}", salt="ring"), replica))
            self._points.sort()

    def remove(self, replica: str) -> bool:
        """Remove a replica; returns False if it was not present."""
        with self._lock:
            if replica not in self._weights:
                return False
            del self._weights[replica]
            self._points = [p for p in self._points if p[1] != replica]
            return True

    def get(self, key: str) -> str | None:
        """The replica owning `key`, or None when the ring is empty."""
        if not self._points:
            return None
        h = _hash32(key, salt="key")
        idx = bisect.bisect(self._points, (h, ""))
        if idx == len(self._points):
            idx = 0  # wrap around the ring
        return self._points[idx][1]

    @property
    def replicas(self) -> list[str]:
        return sorted(self._weights)

    def __len__(self) -> int:
        return len(self._points)

    def distribution(self, keys: Iterable[str]) -> Counter[str]:
        """Ownership histogram over `keys` — useful for ring-balance checks."""
        return Counter(r for k in keys if (r := self.get(k)) is not None)


# --------------------------------------------------------------------------- #
# The affinity policy
# --------------------------------------------------------------------------- #
class PrefixCacheAffinity:
    """Decide which replicas are most likely to have a request's prefix cached.

    :meth:`order` is the entry point: given a conversation and the replicas
    currently eligible, it returns them best-affinity-first. The ranking is:

    1. replicas the radix tree has *observed* serving this prefix, most
       observations first (strongest evidence of a warm cache);
    2. the ring's designated owner for the prefix key (correct on a cold start
       and stable under churn);
    3. the remaining candidates in their original order.

    :meth:`observe` should be called after a successful response so the tree
    learns the real mapping as traffic flows.
    """

    def __init__(
        self,
        replicas: Iterable[str] | None = None,
        *,
        prefix_tokens: int = 64,
        virtual_nodes: int = 160,
        weights: dict[str, int] | None = None,
    ) -> None:
        self.prefix_tokens = prefix_tokens
        self.tree = RadixTree()
        self.ring = ConsistentHashRing(virtual_nodes=virtual_nodes)
        for replica in replicas or ():
            self.ring.add(replica, weight=(weights or {}).get(replica, 1))

    # -- membership --------------------------------------------------------- #
    def add_replica(self, replica: str, weight: int = 1) -> None:
        self.ring.add(replica, weight=weight)

    def remove_replica(self, replica: str) -> bool:
        return self.ring.remove(replica)

    @property
    def replicas(self) -> list[str]:
        return self.ring.replicas

    # -- prefix extraction -------------------------------------------------- #
    def tokens_for(self, messages: Sequence[Any]) -> list[str]:
        return tokens_for_messages(messages, limit=self.prefix_tokens)

    def key_for(self, messages: Sequence[Any]) -> str:
        """The stable hash key for a conversation's cacheable prefix."""
        return self.key_for_tokens(self.tokens_for(messages))

    @staticmethod
    def key_for_tokens(tokens: Sequence[str]) -> str:
        return _SEP.join(tokens)

    # -- the decision ------------------------------------------------------- #
    def owner(self, key: str) -> str | None:
        """The ring's preferred replica for `key` (None if the ring is empty)."""
        return self.ring.get(key)

    def order(self, messages_or_tokens: Sequence[Any], candidates: Sequence[str]) -> list[str]:
        """Order `candidates` by KV-cache affinity, best first.

        Accepts either a message list or an already-tokenized prefix, so
        callers that cached the tokens can avoid re-tokenizing.
        """
        if not candidates:
            return []
        tokens = self._as_tokens(messages_or_tokens)
        observed = self.tree.lookup(tokens)
        owner = self.owner(self.key_for_tokens(tokens))
        positions = {name: i for i, name in enumerate(candidates)}

        def score(name: str) -> tuple[int, int, int]:
            # Descending sort: observations, then is-owner, then input order.
            return (observed.get(name, 0), 1 if name == owner else 0, -positions[name])

        return sorted(candidates, key=score, reverse=True)

    @staticmethod
    def _as_tokens(value: Sequence[Any]) -> list[str]:
        """Distinguish a token list from a message list cheaply."""
        if not value:
            return []
        first = value[0]
        if isinstance(first, str):
            return list(value)
        return tokens_for_messages(value)

    def observe(self, messages_or_tokens: Sequence[Any], replica: str) -> None:
        """Record a successful response, teaching the tree the mapping."""
        self.tree.observe(self._as_tokens(messages_or_tokens), replica)

    def stats(self) -> dict[str, Any]:
        return {
            "replicas": self.replicas,
            "ring_points": len(self.ring),
            "known_prefixes": len(self.tree),
            "observations": self.tree.observations,
            "prefix_tokens": self.prefix_tokens,
        }
