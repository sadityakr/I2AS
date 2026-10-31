"""Submit an analysis script file for a run through the bridge, wait, print its summary.

Usage: python scripts/agent_demo/submit_script.py <run_id> <script.py> <name>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent import send  # noqa: E402


def _tool(name: str, args: dict) -> dict:
    answer = send("tools/call", {"name": name, "arguments": args})
    return json.loads(answer["result"]["content"][0]["text"])


def main() -> int:
    run_id, path, name = sys.argv[1], sys.argv[2], sys.argv[3]
    started = _tool("run_analysis_script", {"run_id": run_id, "source": Path(path).read_text(), "name": name})
    if not started.get("ok"):
        print(json.dumps(started, indent=1))
        return 1
    script_id = started["result"]["script_id"]
    for _ in range(90):
        time.sleep(2)
        result = _tool("read_analysis_script_result", {"run_id": run_id, "script_id": script_id})
        status = (result.get("result") or {}).get("status")
        if status and status not in ("running", "pending", "started", "none"):
            body = result["result"]
            print(script_id, status, body.get("error", ""))
            for line in body.get("summary", []):
                print(" ", line)
            for value in body.get("results", []):
                unc = f" ± {value['uncertainty']:.3g}" if value.get("uncertainty") else ""
                print(f"  {value['name']} = {value['value']:.4g}{unc} {value['unit']}")
            return 0
    print("timed out")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
