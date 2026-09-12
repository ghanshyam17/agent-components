"""Path sandboxing and command allow/deny lists for the built-in tools.

These are defence-in-depth guardrails, not a security boundary — they prevent
the most common accidental foot-guns (escaping a workspace, clobbering files
outside it, running rm -rf). They are configurable per-tool.
"""
from __future__ import annotations

import os
from pathlib import Path


class SandboxError(Exception):
    """Raised when a requested path/command violates the sandbox."""


def resolve_within(path: str, root: str | os.PathLike[str] | None) -> Path:
    """Resolve `path` against `root` and ensure it stays inside `root`.

    If `path` is absolute and `root` is None, the absolute path is returned
    (no sandbox). Otherwise the resolved real path is checked to be within the
    real root, and `SandboxError` is raised on escape attempts (including
    symlinks that point outside after resolution).

    Note: the parent directory must exist for `resolve()` to expand symlinks;
    when it does not, we fall back to comparing lexical normalised paths.
    """
    p = Path(path)
    if root is None:
        return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()

    root_path = Path(root).resolve()
    base = p if p.is_absolute() else (root_path / p)
    try:
        real = base.resolve(strict=True)
    except FileNotFoundError:
        real = base.resolve(strict=False)
        # Lexical check for escapes when the target doesn't exist yet.
        if root_path not in real.parents and real != root_path:
            raise SandboxError(f"path escapes sandbox: {path} -> {real}")
        return real
    if root_path not in real.parents and real != root_path:
        raise SandboxError(f"path escapes sandbox: {path} -> {real}")
    return real


def is_command_allowed(command: str, *, allow: list[str] | None = None,
                       deny: list[str] | None = None) -> bool:
    """Return True if `command` is permitted by the allow/deny lists.

    Matching is on the leading program token. An empty allow list means "all
    allowed"; a non-empty allow list restricts to those programs. The deny
    list always wins and is matched as a substring of the command (so
    `deny=["rm -rf"]` blocks `rm -rf /`).
    """
    if deny:
        for d in deny:
            if d in command:
                return False
    head = command.strip().split(None, 1)[0] if command.strip() else ""
    # strip a leading env assignment like `FOO=bar cmd`
    while head and "=" in head and not head.startswith("="):
        head = command.strip().split(None, 1)
        if len(head) > 1:
            head = head[1].split(None, 1)[0]
        else:
            break
    if allow:
        return head in allow
    return True