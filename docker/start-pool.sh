#!/bin/bash
# Start the LNL OpenClaw worker pool.
#
# Reads the operator token from ~/.openclaw/identity/device-auth.json and
# exports it as OPENCLAW_GATEWAY_TOKEN_1 (and _2/_3/_4 if you have multiple
# workers) before bringing up docker-compose.
#
# Usage:
#   ./docker/start-pool.sh          # start (or restart) the pool
#   ./docker/start-pool.sh down     # stop and remove containers
#   ./docker/start-pool.sh logs     # tail container logs
#
# The pool uses host ports 19789/19888 (worker-1) so it won't collide with a
# locally-running OpenClaw instance on the default 18789/18888 ports.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

# ── Read operator token from local OpenClaw identity ─────────────────────────
DEVICE_AUTH="${HOME}/.openclaw/identity/device-auth.json"
if [ ! -f "${DEVICE_AUTH}" ]; then
    echo "ERROR: ${DEVICE_AUTH} not found." >&2
    echo "Make sure OpenClaw is installed and you have logged in (openclaw auth login)." >&2
    exit 1
fi

OPERATOR_TOKEN=$(python3 -c "
import json, sys
try:
    d = json.load(open('${DEVICE_AUTH}'))
    print(d['tokens']['operator']['token'])
except (KeyError, json.JSONDecodeError) as e:
    print(f'ERROR: could not read operator token: {e}', file=sys.stderr)
    sys.exit(1)
")

export OPENCLAW_GATEWAY_TOKEN_1="${OPERATOR_TOKEN}"
export OPENCLAW_GATEWAY_TOKEN_2="${OPERATOR_TOKEN}"
export OPENCLAW_GATEWAY_TOKEN_3="${OPERATOR_TOKEN}"
export OPENCLAW_GATEWAY_TOKEN_4="${OPERATOR_TOKEN}"

# ── Dispatch subcommand ───────────────────────────────────────────────────────
CMD="${1:-up}"

case "${CMD}" in
    up)
        echo "Starting LNL OpenClaw worker pool..."
        docker compose -f "${COMPOSE_FILE}" up -d --remove-orphans
        echo ""
        echo "Workers ready:"
        echo "  worker-1  gateway=ws://localhost:19789  mock=http://localhost:19888"
        echo ""
        echo "Run evaluation:"
        echo "  python -m src.data.evaluate_baseline -i <test_cases.jsonl> --pool docker/worker-pool.yaml"
        ;;
    down)
        echo "Stopping LNL OpenClaw worker pool..."
        docker compose -f "${COMPOSE_FILE}" down
        ;;
    restart)
        echo "Restarting LNL OpenClaw worker pool..."
        docker compose -f "${COMPOSE_FILE}" down
        docker compose -f "${COMPOSE_FILE}" up -d --remove-orphans
        ;;
    logs)
        docker compose -f "${COMPOSE_FILE}" logs -f "${@:2}"
        ;;
    *)
        echo "Usage: $0 [up|down|restart|logs]" >&2
        exit 1
        ;;
esac
