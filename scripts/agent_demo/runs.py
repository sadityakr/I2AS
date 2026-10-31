"""Print the open experiment's runs compactly: id, prefix, status (exit 0 if <run_id> given and done)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent import send  # noqa: E402

answer = send("tools/call", {"name": "list_runs", "arguments": {}})
runs = json.loads(answer["result"]["content"][0]["text"])["result"]["runs"]
for run in runs:
    print(run["run_id"], run.get("file_prefix") or (run.get("params") or {}).get("file_prefix", ""), run["status"])
if len(sys.argv) > 1:
    raise SystemExit(0 if any(r["run_id"] == sys.argv[1] and r["status"] == "done" for r in runs) else 1)
