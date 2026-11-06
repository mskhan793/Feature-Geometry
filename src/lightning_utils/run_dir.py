"""Run folder layout and the record of every resolved setting of a run."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


def make_run_dir(root: str | Path, experiment: str, timestamp: str | None = None) -> Path:
    """Create ``root/experiment/timestamp``.

    Pass one shared ``timestamp`` to every process of a multi-process run so that they all use the same folder.
    """
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / experiment / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_resolved_config(run_dir: str | Path, config: dict) -> Path | None:
    """Write ``config_resolved.json`` from the global-rank-0 process only; return its path, or None on other ranks."""
    if os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")) != "0":
        return None
    path = Path(run_dir) / "config_resolved.json"
    path.write_text(json.dumps(config, indent=1, sort_keys=True, default=str))
    return path
