"""Run facts — the one place a recorded run becomes a manifest-shaped dict.

The analysis runner (building a spec), the drafting assistant (building a
prompt) and the publishing renderer (describing a run on a notebook page) all
describe a run in the same words, taken from here. Neither the analysis stage
nor the notebook owns this: it sits beside both so neither imports the other.
"""

from __future__ import annotations

from typing import Any


def manifest_from_run(run: Any) -> dict[str, Any]:
    """Return manifest-shaped facts built from one recorded run.

    Duck-typed on the record rather than importing it, so any module in the
    session layer can call it.

    Args:
        run: A ``RunRecord``-shaped object.

    Returns:
        ``run_id``, ``procedure``, ``kind``, ``params``, ``data_file``,
        ``started_utc``, ``finished_utc``, ``status``, ``reason`` and
        ``params_digest``.
    """
    return {
        "run_id": getattr(run, "run_id", ""),
        "procedure": getattr(run, "procedure", ""),
        "kind": getattr(run, "kind", ""),
        "params": dict(getattr(run, "params", {}) or {}),
        "data_file": getattr(run, "data_file", ""),
        "started_utc": getattr(run, "started_utc", ""),
        "finished_utc": getattr(run, "finished_utc", ""),
        "status": getattr(run, "status", ""),
        "reason": getattr(run, "reason", ""),
        "params_digest": getattr(run, "params_digest", ""),
    }
