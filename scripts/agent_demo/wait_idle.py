"""Block until the engine is IDLE with nothing queued (polls read_status every 3 s)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent import send  # noqa: E402

idle_seen = 0
while True:
    answer = send("tools/call", {"name": "read_status", "arguments": {}})
    status = json.loads(answer["result"]["content"][0]["text"])["result"]
    queued = status.get("queue") or status.get("queue_entries") or []
    if status["state"] == "IDLE" and not status.get("run") and not queued:
        idle_seen += 1
        if idle_seen >= 2:
            break
    else:
        idle_seen = 0
    time.sleep(3)
print("idle")
