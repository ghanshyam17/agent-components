"""Server smoke tests (no real vLLM — only clear-cut heuristic routes)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import agentic_router.server.app as mod


@pytest.fixture()
def client():
    with TestClient(mod.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_route_heuristic_lower(client):
    r = client.post("/route", json={"task": "hi there"})
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "lower"
    assert body["method"] == "heuristic"


def test_route_heuristic_higher(client):
    LOADED = (
        "Debug this stack trace, design a step-by-step plan to refactor the "
        "authentication module, then implement an algorithm and run tests. "
        "```\ndef f(): pass\n```\n Please read the file, search the web, "
        "and finally optimize the regex and compare results."
    )
    r = client.post("/route", json={"task": LOADED})
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "higher"
    assert body["method"] == "heuristic"


def test_sessions_initially_empty(client):
    assert client.get("/sessions").json() == {"session_ids": []}


def test_missing_session_404(client):
    assert client.get("/sessions/nope").status_code == 404