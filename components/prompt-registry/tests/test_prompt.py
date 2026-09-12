"""prompt-registry tests — offline: register+render, missing vars, latest,
A/B determinism, load_dir, json/yaml structured load."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompt_registry import (
    PromptNotFound,
    PromptRegistry,
    PromptRenderError,
    PromptTemplate,
    build_registry,
)


# ---------------- register + render ----------------
def test_register_and_render():
    reg = PromptRegistry()
    reg.register(PromptTemplate(
        name="greet", version="1.0.0",
        template="Hello {{ name }}!", variables=["name"],
    ))
    assert reg.render("greet", name="Ada") == "Hello Ada!"
    # explicit version works too
    assert reg.render("greet", version="1.0.0", name="Bo") == "Hello Bo!"


def test_render_missing_var_raises():
    reg = PromptRegistry()
    reg.register(PromptTemplate(
        name="greet", version="1.0.0",
        template="Hello {{ name }}, {{ role }}", variables=["name", "role"],
    ))
    with pytest.raises(PromptRenderError):
        reg.render("greet", name="Ada")  # role missing


def test_template_rejects_undeclared_vars_on_construction():
    with pytest.raises(PromptRenderError):
        PromptTemplate(
            name="bad", version="1.0.0",
            template="{{ a }} {{ b }}", variables=["a"],  # b missing
        )


def test_template_rejects_bad_version():
    from prompt_registry import PromptVersionError
    with pytest.raises(PromptVersionError):
        PromptTemplate(name="x", version="1.x", template="ok", variables=[])


# ---------------- latest() picks highest semver ----------------
def test_latest_picks_highest_semver():
    reg = PromptRegistry()
    for v in ["1.0.0", "1.2.0", "2.0.0", "1.10.0"]:
        reg.register(PromptTemplate(
            name="p", version=v, template=v, variables=[],
        ))
    assert reg.latest("p").version == "2.0.0"
    # 1.10.0 sorts above 1.2.0
    assert reg.versions("p") == ["1.0.0", "1.2.0", "1.10.0", "2.0.0"]
    # render with version=None -> latest
    assert reg.render("p") == "2.0.0"


def test_get_missing_raises():
    reg = PromptRegistry()
    with pytest.raises(PromptNotFound):
        reg.get("nope", "1.0.0")
    with pytest.raises(PromptNotFound):
        reg.latest("nope")


def test_list_filters_by_name():
    reg = PromptRegistry()
    reg.register(PromptTemplate(name="a", version="1.0.0", template="a", variables=[]))
    reg.register(PromptTemplate(name="b", version="1.0.0", template="b", variables=[]))
    assert {t.name for t in reg.list()} == {"a", "b"}
    assert {t.version for t in reg.list("a")} == {"1.0.0"}


# ---------------- A/B determinism ----------------
def test_ab_uniform_deterministic_with_seed():
    reg = PromptRegistry(seed=7)
    reg.register(PromptTemplate(name="ad", version="1.0.0", template="A {{ x }}", variables=["x"]))
    reg.register(PromptTemplate(name="ad", version="2.0.0", template="B {{ x }}", variables=["x"]))

    # Same seed -> same sequence of picks.
    reg2 = PromptRegistry(seed=7)
    reg2.register(PromptTemplate(name="ad", version="1.0.0", template="A", variables=[]))
    reg2.register(PromptTemplate(name="ad", version="2.0.0", template="B", variables=[]))

    seq1 = [reg.ab("ad").version for _ in range(50)]
    seq2 = [reg2.ab("ad").version for _ in range(50)]
    assert seq1 == seq2
    # Both versions should appear in a 50-draw uniform sample (vanishingly rare otherwise).
    assert set(seq1) == {"1.0.0", "2.0.0"}


def test_ab_weighted_deterministic_with_seed():
    reg = PromptRegistry(seed=1)
    reg.register(PromptTemplate(name="ad", version="1.0.0", template="A", variables=[]))
    reg.register(PromptTemplate(name="ad", version="2.0.0", template="B", variables=[]))
    seq = [reg.ab("ad", weights={"1.0.0": 0.9, "2.0.0": 0.1}).version for _ in range(200)]
    # Heavy weight on 1.0.0 -> it dominates.
    assert seq.count("1.0.0") > seq.count("2.0.0")
    # Deterministic: a fresh registry with same seed reproduces it.
    reg2 = PromptRegistry(seed=1)
    reg2.register(PromptTemplate(name="ad", version="1.0.0", template="A", variables=[]))
    reg2.register(PromptTemplate(name="ad", version="2.0.0", template="B", variables=[]))
    seq2 = [reg2.ab("ad", weights={"1.0.0": 0.9, "2.0.0": 0.1}).version for _ in range(200)]
    assert seq == seq2


def test_ab_missing_weight_raises():
    reg = PromptRegistry(seed=1)
    reg.register(PromptTemplate(name="ad", version="1.0.0", template="A", variables=[]))
    with pytest.raises(Exception):
        reg.ab("ad", weights={"2.0.0": 1.0})


# ---------------- load_dir ----------------
def test_load_dir_text_files(tmp_path: Path):
    (tmp_path / "greet.j2").write_text(
        "# name: greet\n# version: 1.2.0\nHello {{ name }}!\n", encoding="utf-8"
    )
    (tmp_path / "summary.txt").write_text(
        "Summarize: {{ text }}", encoding="utf-8"
    )
    reg = PromptRegistry()
    reg.load_dir(tmp_path)
    assert reg.render("greet", name="Ada") == "Hello Ada!"
    assert reg.latest("greet").version == "1.2.0"
    assert reg.render("summary", text="x") == "Summarize: x"
    # default version for unnamed file is 1.0.0
    assert reg.versions("summary") == ["1.0.0"]


def test_load_dir_json_structured(tmp_path: Path):
    (tmp_path / "prompts.json").write_text(json.dumps([
        {"name": "greet", "version": "1.0.0", "template": "Hi {{ name }}",
         "variables": ["name"], "tags": ["v1"]},
        {"name": "greet", "version": "2.0.0", "template": "Hey {{ name }}",
         "variables": ["name"], "tags": ["v2"]},
    ]), encoding="utf-8")
    reg = PromptRegistry()
    reg.load_dir(tmp_path)
    assert reg.latest("greet").version == "2.0.0"
    assert reg.render("greet", name="Ada") == "Hey Ada"


def test_load_dir_yaml_structured(tmp_path: Path):
    yaml = pytest.importorskip("yaml")
    (tmp_path / "prompts.yaml").write_text(
        "- name: greet\n"
        "  version: '1.0.0'\n"
        "  template: 'Hello {{ name }}'\n"
        "  variables: ['name']\n"
        "- name: greet\n"
        "  version: '3.0.0'\n"
        "  template: 'Yo {{ name }}'\n"
        "  variables: ['name']\n",
        encoding="utf-8",
    )
    reg = PromptRegistry()
    reg.load_dir(tmp_path)
    assert reg.latest("greet").version == "3.0.0"
    assert reg.render("greet", name="Ada") == "Yo Ada"


def test_reload_picks_up_changes(tmp_path: Path):
    (tmp_path / "greet.j2").write_text(
        "# version: 1.0.0\nHello {{ name }}\n", encoding="utf-8"
    )
    reg = PromptRegistry()
    reg.load_dir(tmp_path)
    assert reg.render("greet", name="Ada") == "Hello Ada"
    # change the file
    (tmp_path / "greet.j2").write_text(
        "# version: 2.0.0\nHi {{ name }}\n", encoding="utf-8"
    )
    reg.reload()
    assert reg.latest("greet").version == "2.0.0"
    assert reg.render("greet", name="Ada") == "Hi Ada"


# ---------------- build_registry ----------------
def test_build_registry_with_dir(tmp_path: Path):
    (tmp_path / "x.j2").write_text(
        "# name: x\n# version: 1.0.0\nX {{ n }}", encoding="utf-8"
    )
    from prompt_registry.config import PromptSettings
    reg = build_registry(PromptSettings(dir=str(tmp_path)))
    assert reg.render("x", n="1") == "X 1"