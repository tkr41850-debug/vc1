from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def probe_dirs(name: str, root: str = "/tmp"):
    base = Path(root) / name
    home = base / "home"
    workspace = base / "workspace"
    home.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    yield home, workspace
