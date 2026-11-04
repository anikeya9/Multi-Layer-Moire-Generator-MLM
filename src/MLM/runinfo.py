"""``run_info.yaml``: how, when and with which software a run directory was produced.

Written by :func:`MLM.search.build_stack` at the start of a run, after every step and at
the end, so an interrupted run shows ``status: running`` and how far it got. Example::

    status: finished
    mlm_version: 0.2.0
    git_commit: 46f615cd…   # of the MLM source tree, if it is a git checkout
    git_dirty: false        # uncommitted changes in that tree?
    python: 3.13.1
    numpy: 2.3.1
    started_at: '2026-11-04T10:12:03+00:00'
    finished_at: '2026-11-04T10:12:11+00:00'
    wall_seconds: 8.2
    steps:
      - {round: 1, step: L1 + L2 -> L1-2, searches: 1, found: 1, passed: 1, seconds: 0.2}
"""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

import MLM

RUN_INFO_FILENAME = "run_info.yaml"


def _git(*args: str) -> Optional[str]:
    """Run git in the MLM source tree; ``None`` if that is not a git checkout."""
    source = Path(MLM.__file__).resolve().parent
    try:
        output = subprocess.run(["git", *args], cwd=source, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return output.stdout.strip() if output.returncode == 0 else None


def environment_info() -> dict[str, Any]:
    """Versions of MLM and its main dependencies, and the git commit of the MLM source.

    >>> sorted(environment_info())[:3]
    ['ase', 'git_commit', 'git_dirty']
    """
    import ase
    import numpy

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "mlm_version": MLM.__version__,
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "ase": ase.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunInfo:
    """Collects the run metadata and keeps ``run_info.yaml`` up to date."""

    run_dir: Path
    data: dict[str, Any] = field(default_factory=dict)
    _started: float = 0.0

    def start(self, extra: Optional[dict[str, Any]] = None) -> None:
        """Begin a run: environment, start time, ``status: running``."""
        self._started = time.perf_counter()
        self.data = {"status": "running", **environment_info(), "started_at": _now(),
                     "finished_at": None, "wall_seconds": None, "steps": [], **(extra or {})}
        self.write()

    def add_step(self, row: dict[str, Any]) -> None:
        """Record one finished step of the plan."""
        self.data["steps"].append(row)
        self.write()

    def finish(self, status: str = "finished", **extra: Any) -> None:
        """End the run with ``status`` ("finished" or "failed")."""
        self.data.update(status=status, finished_at=_now(),
                         wall_seconds=round(time.perf_counter() - self._started, 3), **extra)
        self.write()

    def write(self) -> Path:
        path = Path(self.run_dir) / RUN_INFO_FILENAME
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.data, handle, sort_keys=False)
        return path


def load_run_info(run_dir: Path | str) -> dict[str, Any]:
    """Read ``run_info.yaml`` of a run directory (empty dict if there is none)."""
    path = Path(run_dir) / RUN_INFO_FILENAME
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
