#!/usr/bin/env bash
# Connect the agent: start mcp_bridge.py against the app started by
# start_station.sh in <workdir>. Connect AFTER the operator has opened the
# experiment — the gateway binds a connection's agent feed to the experiment
# open when it connects.
#
# Usage: scripts/agent_demo/connect_agent.sh <workdir> [actor-id]
set -euo pipefail
WORK=$(cd "${1:?usage: connect_agent.sh <workdir> [actor-id]}" && pwd)
ACTOR=${2:-claude-agent}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export XDG_CONFIG_HOME="$WORK/config" XDG_STATE_HOME="$WORK/state" I2AS_MEASUREMENT_ROOT="$WORK/root"
cd "$REPO"
python scripts/agent_demo/mcp_bridge.py --transcript "$WORK/transcript.jsonl" --actor-id "$ACTOR" >"$WORK/bridge.log" 2>&1 &
for _ in $(seq 40); do grep -q "bridge ready" "$WORK/bridge.log" 2>/dev/null && break; sleep 0.5; done
cat "$WORK/bridge.log"
