"""Tests for agent-ui FastAPI server bridge, metadata introspection, and SSE streaming."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_ui.bridge import UIBridge
from agent_ui.server import create_ui_app, mount_agent_ui


def test_ui_bridge_defaults():
    bridge = UIBridge()
    config = bridge.get_config()
    assert "name" in config
    assert "Enterprise Agent" in config["name"]
    assert len(config.get("starter_prompts", [])) > 0

    graph = bridge.get_graph()
    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["nodes"]) > 0


def test_agent_ui_endpoints():
    app = create_ui_app()
    client = TestClient(app)

    # Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # Config endpoint
    res = client.get("/api/config")
    assert res.status_code == 200
    cfg = res.json()
    assert "name" in cfg
    assert "starter_prompts" in cfg

    # Graph endpoint
    res = client.get("/api/graph")
    assert res.status_code == 200
    graph = res.json()
    assert "nodes" in graph
    assert "edges" in graph

    # HTML root serving
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Component Graph" in res.text or "Enterprise Agent" in res.text

    # SSE Chat endpoint
    chat_res = client.post("/api/chat", json={"message": "Run sales pipeline", "session_id": "test_sess"})
    assert chat_res.status_code == 200
    assert "text/event-stream" in chat_res.headers["content-type"]
    body = chat_res.text
    assert "data: " in body
    assert "[DONE]" in body


def test_mount_agent_ui():
    parent_app = FastAPI()
    mount_agent_ui(parent_app, prefix="/agent")
    client = TestClient(parent_app)

    res = client.get("/agent/api/health")
    assert res.status_code == 200
    assert res.json()["service"] == "agent-ui"

    res_cfg = client.get("/agent/api/config")
    assert res_cfg.status_code == 200
    assert "name" in res_cfg.json()
