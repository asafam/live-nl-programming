#!/bin/bash
# Entrypoint for the LNL OpenClaw worker container.
# Starts the OpenClaw gateway and the LNL mock server as sibling processes.
# Exits (triggering container restart) if either process dies.
set -euo pipefail

# ── Copy plugin into the bind-mount dir (hidden at image build time) ─────────
# The bind-mount over /home/node/.openclaw hides any files baked into the image
# at that path.  Restore the plugin from its staging location on every start.
mkdir -p "${HOME}/.openclaw/extensions/lnl-mock-external"
cp -f "${HOME}/openclaw-extensions/lnl-mock-external/index.js" \
      "${HOME}/.openclaw/extensions/lnl-mock-external/index.js"
cp -f "${HOME}/openclaw-extensions/lnl-mock-external/openclaw.plugin.json" \
      "${HOME}/.openclaw/extensions/lnl-mock-external/openclaw.plugin.json"

# ── Write gateway config ──────────────────────────────────────────────────────
CONFIG_DIR="${HOME}/.openclaw"
mkdir -p "${CONFIG_DIR}"
# Always write from the baked-in template so the gateway gets a clean config.
# Template lives outside .openclaw so a pool bind-mount doesn't hide it.
envsubst '${OPENCLAW_GATEWAY_TOKEN}' \
    < "${HOME}/openclaw.json.tpl" \
    > "${CONFIG_DIR}/openclaw.json"

# ── Start the LNL mock server ─────────────────────────────────────────────────
echo "[entrypoint] Starting mock server on port 18888..."
cd /app && python3 -m src.data.mock_server \
    --port 18888 \
    --openclaw-url "http://localhost:18789" &
MOCK_PID=$!

# Wait for the mock server to be ready before starting the gateway
# (gateway startup can take a moment; mock server is quick)
for i in $(seq 1 30); do
    if curl -sf http://localhost:18888/health > /dev/null 2>&1; then
        echo "[entrypoint] Mock server ready."
        break
    fi
    sleep 0.5
done

# ── Start the OpenClaw gateway ────────────────────────────────────────────────
# Pass auth token via CLI flags so gateway auth works regardless of config file
# state.  OPENCLAW_GATEWAY_TOKEN is injected by docker-compose from the host env.
echo "[entrypoint] Starting OpenClaw gateway..."
openclaw gateway run --auth token --token "${OPENCLAW_GATEWAY_TOKEN}" &
OC_PID=$!

# ── Monitor both processes — exit if either dies ──────────────────────────────
wait -n ${MOCK_PID} ${OC_PID}
EXIT_CODE=$?
echo "[entrypoint] A child process exited (code ${EXIT_CODE}). Shutting down."
kill ${MOCK_PID} ${OC_PID} 2>/dev/null || true
exit ${EXIT_CODE}
