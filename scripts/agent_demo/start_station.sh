#!/usr/bin/env bash
# Bring up the simulated spintronics station headless: a virtual display with
# a window manager and the I2AS app on the sim_spintronics rack. Everything the
# app writes (settings, logs, the gateway descriptor, experiments, run files)
# goes under <workdir>, so a demo starts clean and leaves the user's own
# configuration alone.
#
# Usage: scripts/agent_demo/start_station.sh <workdir> [WIDTHxHEIGHT]
# Then, once the operator has opened an experiment, connect the agent with
#   scripts/agent_demo/connect_agent.sh <workdir>
set -euo pipefail
WORK=$(mkdir -p "${1:?usage: start_station.sh <workdir> [WxH]}" && cd "$1" && pwd)
SIZE=${2:-2560x1440}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export DISPLAY=${DISPLAY:-:100}
export XDG_CONFIG_HOME="$WORK/config" XDG_STATE_HOME="$WORK/state" I2AS_MEASUREMENT_ROOT="$WORK/root"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_STATE_HOME" "$I2AS_MEASUREMENT_ROOT"
if ! xdpyinfo >/dev/null 2>&1; then
  Xvfb "$DISPLAY" -screen 0 "${SIZE}x24" -nolisten tcp >"$WORK/xvfb.log" 2>&1 &
  for _ in $(seq 20); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.25; done
fi
pgrep -x openbox >/dev/null || { openbox >"$WORK/openbox.log" 2>&1 & sleep 1; }
cd "$REPO"
python -m i2as.main --config sim_spintronics >"$WORK/app.log" 2>&1 &
for _ in $(seq 60); do grep -q "Gateway server listening" "$WORK/app.log" 2>/dev/null && break; sleep 0.5; done
echo "station up: display $DISPLAY ($SIZE), workdir $WORK"
