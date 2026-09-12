"""Evaluation datasets.

An `Example` is one evaluation case — an input string, an optional expected
value (a scalar for equality-style metrics, a dict for structured targets such
as expected tool calls) and free-form metadata. A `Dataset` is just a list of
examples with loaders for JSON / JSONL / CSV / YAML.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator, Self

from pydantic import BaseModel, Field


class Example(BaseModel):
    """A single evaluation case."""

    id: str
    input: str
    expected: str | dict[str, Any] | list[Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """A list of `Example`s with file loaders."""

    examples: list[Example] = Field(default_factory=list)

    # -- container protocol ------------------------------------------------ #
    def __iter__(self) -> Iterator[Example]:  # type: ignore[override]
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

    # -- loaders ------------------------------------------------------------ #
    @classmethod
    def load_json(cls, path: str | Path) -> Self:
        """Load from a JSON file: either a bare list of examples or an object
        with an ``examples`` key."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("examples", [])
        return cls(examples=[Example.model_validate(e) for e in data])

    @classmethod
    def load_jsonl(cls, path: str | Path) -> Self:
        """Load from a JSONL file (one JSON object per line)."""
        examples: list[Example] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            examples.append(Example.model_validate(json.loads(line)))
        return cls(examples=examples)

    @classmethod
    def load_csv(cls, path: str | Path) -> Self:
        """Load from a CSV file with columns: id,input,expected,metadata.

        ``expected`` may be a plain string or a JSON-encoded value (object/list).
        ``metadata`` is an optional JSON-encoded object (empty → {}).
        """
        examples: list[Example] = []
        with Path(path).open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw_expected = (row.get("expected") or "").strip()
                expected: Any = None
                if raw_expected:
                    if raw_expected[0] in "{[":
                        try:
                            expected = json.loads(raw_expected)
                        except json.JSONDecodeError:
                            expected = raw_expected
                    else:
                        expected = raw_expected
                raw_meta = (row.get("metadata") or "").strip()
                metadata: dict[str, Any] = {}
                if raw_meta:
                    try:
                        parsed = json.loads(raw_meta)
                        if isinstance(parsed, dict):
                            metadata = parsed
                    except json.JSONDecodeError:
                        metadata = {"_raw": raw_meta}
                examples.append(
                    Example(
                        id=row["id"],
                        input=row.get("input", ""),
                        expected=expected,
                        metadata=metadata,
                    )
                )
        return cls(examples=examples)

    @classmethod
    def load_yaml(cls, path: str | Path) -> Self:
        """Load from a YAML file (requires the ``yaml`` extra: pyyaml)."""
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "YAML datasets require the 'yaml' extra: pip install eval-harness[yaml]"
            ) from e
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("examples", [])
        return cls(examples=[Example.model_validate(e) for e in data or []])

    # -- writers ------------------------------------------------------------ #
    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"examples": [e.model_dump() for e in self.examples]}, indent=2),
            encoding="utf-8",
        )