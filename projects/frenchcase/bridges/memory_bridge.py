"""Bridge: FrenchCase SessionStore → chassis memory-store SessionStore.

FrenchCase's `app/core/session_store.py` is a synchronous, JSON-persisted store
keyed by session_id holding plain dicts. The chassis `memory_store.base.SessionStore`
is an async ABC over `components_core.SessionState` (pydantic).

This adapter wraps the FrenchCase store so the router and patterns can use it
unchanged, and so prod can swap to the chassis Redis backend without touching
FrenchCase code.

Usage:
    from projects.frenchcase.bridges.memory_bridge import FrenchCaseSessionStore
    store = FrenchCaseSessionStore()          # wraps app.core.session_store.SessionStore
    state = await store.get_or_create("abc")  # → components_core.SessionState
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from components_core import Message, Plan, SessionState
from memory_store.base import SessionStore

logger = logging.getLogger(__name__)

# ── FrenchCase import (mounted at runtime) ───────────────────────────────
# `mount_frenchcase()` puts the repo root first (so `app.*` resolves to the
# package) and app/ last as a fallback (so bare `from config import ...` works).
from projects.frenchcase.mount import mount_frenchcase as _mount

_mount()

try:
    from app.core.session_store import SessionStore as _FCStore  # type: ignore
    FRENCHCASE_AVAILABLE = True
except ImportError:
    try:
        from core.session_store import SessionStore as _FCStore  # type: ignore
        FRENCHCASE_AVAILABLE = True
    except ImportError:
        _FCStore = None  # type: ignore
        FRENCHCASE_AVAILABLE = False


def _to_session_state(sid: str, raw: dict[str, Any]) -> SessionState:
    """Convert FrenchCase's plain-dict session into a chassis SessionState."""
    messages: list[Message] = []
    for m in raw.get("messages", []) or []:
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        messages.append(
            Message(
                role=role,
                content=m.get("content", "") or "",
                name=m.get("name"),
            )
        )
    created = 0.0
    raw_created = raw.get("created_at")
    if isinstance(raw_created, str):
        # FrenchCase stores ISO-8601 strings (datetime.now().isoformat()).
        try:
            from datetime import datetime as _dt
            created = _dt.fromisoformat(raw_created).timestamp()
        except Exception:
            created = 0.0
    elif isinstance(raw_created, (int, float)):
        created = float(raw_created)
    return SessionState(
        session_id=sid,
        messages=messages,
        plan=None,
        created_at=created,
    )


class FrenchCaseSessionStore(SessionStore):
    """Adapter exposing FrenchCase's JSON store as the chassis async SessionStore.

    Runs the synchronous FrenchCase store in a thread so we never block the
    event loop on file I/O. Every mutation also writes through to the
    FrenchCase store so its own API routes (which read the store directly)
    stay coherent.
    """

    def __init__(self, fc_store: Any | None = None):
        if fc_store is not None:
            self._fc = fc_store
        elif FRENCHCASE_AVAILABLE and _FCStore is not None:
            self._fc = _FCStore()
        else:
            self._fc = None
            logger.warning("FrenchCase SessionStore unavailable — memory bridge in no-op mode")

    # ── internal helpers ────────────────────────────────────────────────
    def _raw(self) -> dict[str, Any]:
        if self._fc is None:
            return {}
        return getattr(self._fc, "_sessions", {}) or {}

    def _create_sync(self, session_id: str | None) -> str:
        """Ensure a session exists and return its id.

        FrenchCase's `create()` mints its own 8-char uuid and ignores any
        requested id, so when a caller supplies `session_id` we register it
        directly in the backing dict (same shape as `create()` builds) rather
        than letting the id drift.
        """
        if self._fc is None:
            return session_id or "ephemeral"

        sessions = self._raw()

        if session_id:
            if session_id not in sessions:
                from datetime import datetime as _dt
                now = _dt.now().isoformat()
                sessions[session_id] = {
                    "id": session_id,
                    "title": "Nouvelle conversation",
                    "messages": [],
                    "cefr": "B2",
                    "section": "ALL",
                    "model": "tef-tutor-3b",
                    "interaction_type": "tutor",
                    "created_at": now,
                    "updated_at": now,
                }
                save = getattr(self._fc, "_save", None)
                if callable(save):
                    save()
            return session_id

        # No id requested — defer to FrenchCase's own generator.
        create = getattr(self._fc, "create", None)
        if callable(create):
            try:
                created = create()
                if isinstance(created, dict):
                    return created.get("id") or created.get("session_id") or ""
                if isinstance(created, str):
                    return created
            except Exception as exc:  # pragma: no cover
                logger.debug("FC create() failed: %s", exc)
        return ""

    def _append_sync(self, session_id: str, message: Message) -> None:
        if self._fc is None:
            return
        # FrenchCase's real signature: append_message(sid, role, content, ...).
        fn = getattr(self._fc, "append_message", None)
        if callable(fn):
            try:
                fn(session_id, message.role, message.content)
                return
            except Exception as exc:
                logger.debug("FC append_message failed: %s", exc)
        # Secondary shapes some versions expose.
        for meth in ("add_message",):
            alt = getattr(self._fc, meth, None)
            if not callable(alt):
                continue
            try:
                alt(session_id, {"role": message.role, "content": message.content})
                return
            except Exception as exc:
                logger.debug("FC %s failed: %s", meth, exc)
        # Last resort: mutate the backing dict directly.
        sessions = self._raw()
        if session_id in sessions:
            sessions[session_id].setdefault("messages", []).append(
                {"role": message.role, "content": message.content}
            )
            save = getattr(self._fc, "_save", None)
            if callable(save):
                save()

    # ── SessionStore ABC ────────────────────────────────────────────────
    async def create(self) -> SessionState:
        sid = await asyncio.to_thread(self._create_sync, None)
        return await self.get_or_create(sid or None)

    async def get(self, session_id: str) -> Optional[SessionState]:
        raw = await asyncio.to_thread(lambda: self._raw().get(session_id))
        if raw is None:
            return None
        return _to_session_state(session_id, raw)

    async def get_or_create(self, session_id: Optional[str]) -> SessionState:
        sid = await asyncio.to_thread(self._create_sync, session_id)
        if not sid:
            sid = "ephemeral"
        state = await self.get(sid)
        if state is not None:
            return state
        return SessionState(session_id=sid, messages=[])

    async def add_message(self, session_id: str, message: Message) -> None:
        await asyncio.to_thread(self._append_sync, session_id, message)

    async def set_plan(self, session_id: str, plan: Plan) -> None:
        if self._fc is None:
            return

        def _set() -> None:
            sessions = self._raw()
            entry = sessions.setdefault(session_id, {"messages": []})
            entry["plan"] = plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)
            save = getattr(self._fc, "_save", None)
            if callable(save):
                save()

        await asyncio.to_thread(_set)

    async def delete(self, session_id: str) -> bool:
        if self._fc is None:
            return False

        def _del() -> bool:
            sessions = self._raw()
            existed = sessions.pop(session_id, None) is not None
            if existed:
                save = getattr(self._fc, "_save", None)
                if callable(save):
                    save()
            return existed

        return await asyncio.to_thread(_del)

    async def list_ids(self) -> list[str]:
        return await asyncio.to_thread(lambda: list(self._raw().keys()))


def build_session_store(fc_store: Any | None = None) -> SessionStore:
    """Factory matching the chassis `memory_store.build_session_store()` shape."""
    return FrenchCaseSessionStore(fc_store)


if __name__ == "__main__":
    import asyncio as _aio

    async def _main() -> None:
        store = build_session_store()
        print(f"FrenchCase mounted: {FRENCHCASE_AVAILABLE}")
        state = await store.get_or_create("smoke-test")
        print(f"session: {state.session_id}")
        await store.add_message("smoke-test", Message(role="user", content="explique le subjonctif"))
        again = await store.get("smoke-test")
        print(f"messages: {len(again.messages) if again else 0}")
        print(f"all ids: {await store.list_ids()}")

    _aio.run(_main())