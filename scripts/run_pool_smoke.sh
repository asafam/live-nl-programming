#!/bin/bash
set -euo pipefail

cd /home/nlp/achimoa/workspace/live-nl-programming
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export LNL_POOL_DATA_DIR="${SLURM_TMPDIR:-/tmp}/lnl-pool-${SLURM_JOB_ID:-local}"
mkdir -p "$LNL_POOL_DATA_DIR"

echo "=== SMOKE: node $(hostname) | job ${SLURM_JOB_ID:-local} | pool data $LNL_POOL_DATA_DIR ==="

echo "=== ensuring docker image lnl-openclaw-worker is built (cached if unchanged) ==="
docker build -f docker/Dockerfile -t lnl-openclaw-worker .

dump_diagnostics() {
    echo "=== diagnostics: docker ps ==="
    docker ps -a --filter "name=lnl-oc-multi-worker" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>&1 || true
    for n in 1 2 3 4; do
        echo "=== diagnostics: worker-$n last 80 log lines ==="
        docker logs --tail 80 "lnl-oc-multi-worker-$n" 2>&1 || true
    done
    echo "=== diagnostics: health probes ==="
    for n in 1 2 3 4; do
        port=$((20887 + n))
        echo "--- worker-$n mock http://localhost:$port/health ---"
        curl -sS --max-time 3 "http://localhost:$port/health" 2>&1 || echo "(no response)"
    done
    echo "=== diagnostics: openclaw effective config (worker-1, via docker exec) ==="
    docker exec lnl-oc-multi-worker-1 cat /home/node/.openclaw/openclaw.json 2>&1 | head -60 || \
        cat "$LNL_POOL_DATA_DIR/multi-worker-1/openclaw.json" 2>&1 | head -60 || true
    echo "=== diagnostics: stability bundles (worker-1) ==="
    find "$LNL_POOL_DATA_DIR/multi-worker-1/logs/stability" -type f -name "*.json" 2>/dev/null | while read -r f; do
        echo "--- $f ---"
        cat "$f" 2>&1 | head -100
    done
}

cleanup() {
    rc=$?
    echo "=== eval/pool exited with code $rc — dumping diagnostics BEFORE teardown ==="
    dump_diagnostics
    echo "=== cleanup: tearing down pool ==="
    ./docker/start-pool.sh --type multi --workers 4 down || true
}
trap cleanup EXIT

echo "=== pre-creating worker dirs with chmod 777 (UID mismatch workaround) ==="
umask 000
for n in 1 2 3 4; do
    mkdir -p "$LNL_POOL_DATA_DIR/multi-worker-$n/identity"
    mkdir -p "$LNL_POOL_DATA_DIR/multi-worker-$n/devices"
    chmod 777 "$LNL_POOL_DATA_DIR/multi-worker-$n" \
              "$LNL_POOL_DATA_DIR/multi-worker-$n/identity" \
              "$LNL_POOL_DATA_DIR/multi-worker-$n/devices"
done
# keep umask 000 through start-pool.sh so seeded files (paired.json, device-auth.json)
# come out 666 — container's `node` user (UID 1000) needs write access since
# the openclaw gateway updates paired.json during pairing.

echo "=== umask before start-pool.sh ==="
umask

echo "=== starting pool: 4 multi workers ==="
./docker/start-pool.sh --type multi --workers 4 restart

echo "=== forcing perms on seeded files (in case umask was overridden) ==="
for n in 1 2 3 4; do
    chmod 666 "$LNL_POOL_DATA_DIR/multi-worker-$n/identity/device-auth.json" 2>&1 || true
    chmod 666 "$LNL_POOL_DATA_DIR/multi-worker-$n/devices/paired.json" 2>&1 || true
done

echo "=== file modes after chmod ==="
ls -la "$LNL_POOL_DATA_DIR/multi-worker-1/identity/" "$LNL_POOL_DATA_DIR/multi-worker-1/devices/" 2>&1

echo "=== docker restart all workers so they re-init with proper perms ==="
docker restart lnl-oc-multi-worker-1 lnl-oc-multi-worker-2 lnl-oc-multi-worker-3 lnl-oc-multi-worker-4 2>&1 | head -10

echo "=== sleeping 20s for workers to settle, then probing mock + gateway WS ==="
sleep 20
for n in 1 2 3 4; do
    mock_port=$((20887 + n))
    echo "--- worker-$n mock http://localhost:$mock_port/health ---"
    curl -sS --max-time 5 "http://localhost:$mock_port/health" || echo "(no response yet)"
    echo ""
done

echo "=== waiting for gateway WS ports to accept connections (max 180s per port) ==="
python3 - <<'PYEOF'
import asyncio, sys
from websockets.asyncio.client import connect

PORTS = [20789, 20790, 20791, 20792]
PER_PORT_TIMEOUT = 180

async def wait_one(port):
    deadline = asyncio.get_event_loop().time() + PER_PORT_TIMEOUT
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with connect(f"ws://localhost:{port}", open_timeout=2):
                print(f"  ws://localhost:{port} ready", flush=True)
                return True
        except Exception:
            await asyncio.sleep(1)
    print(f"  ws://localhost:{port} STILL DOWN after {PER_PORT_TIMEOUT}s", flush=True)
    return False

async def main():
    results = await asyncio.gather(*[wait_one(p) for p in PORTS])
    sys.exit(0 if all(results) else 1)

asyncio.run(main())
PYEOF

echo "=== generating job-specific pool YAML (data_dir must match LNL_POOL_DATA_DIR) ==="
JOB_POOL_YAML="$LNL_POOL_DATA_DIR/worker-pool-multi-4.yaml"
cat > "$JOB_POOL_YAML" <<YAML_EOF
workers:
  - name: multi-worker-1
    container_name: lnl-oc-multi-worker-1
    gateway_url: ws://localhost:20789
    mock_server_url: http://localhost:20888
    data_dir: $LNL_POOL_DATA_DIR/multi-worker-1
  - name: multi-worker-2
    container_name: lnl-oc-multi-worker-2
    gateway_url: ws://localhost:20790
    mock_server_url: http://localhost:20889
    data_dir: $LNL_POOL_DATA_DIR/multi-worker-2
  - name: multi-worker-3
    container_name: lnl-oc-multi-worker-3
    gateway_url: ws://localhost:20791
    mock_server_url: http://localhost:20890
    data_dir: $LNL_POOL_DATA_DIR/multi-worker-3
  - name: multi-worker-4
    container_name: lnl-oc-multi-worker-4
    gateway_url: ws://localhost:20792
    mock_server_url: http://localhost:20891
    data_dir: $LNL_POOL_DATA_DIR/multi-worker-4
YAML_EOF
echo "  Pool YAML: $JOB_POOL_YAML"
echo "  data_dir base: $LNL_POOL_DATA_DIR"

echo "=== pre-eval config snapshot (worker-1, shows initial state before per-TC writes) ==="
docker exec lnl-oc-multi-worker-1 cat /home/node/.openclaw/openclaw.json 2>&1 | python3 -c "
import sys, json
try:
    c = json.load(sys.stdin)
    agents = [a.get('id') for a in c.get('agents', {}).get('list', [])]
    allow = c.get('tools', {}).get('agentToAgent', {}).get('allow', 'MISSING')
    print(f'  agents.list: {agents}')
    print(f'  agentToAgent.allow: {allow}')
except Exception as e:
    print(f'  (parse error: {e})')
" || echo "  (docker exec failed)"

echo "=== running smoke eval (4 reference TCs, 1 run each, no -o) ==="
stdbuf -oL -eL ./scripts/run-eval-baseline.sh \
    -i data/zapier/workflows-mods-smoke4.jsonl \
    --model gpt-5.4-mini --provider azure \
    --judge-model gpt-5.4 --judge-provider azure \
    --pool "$JOB_POOL_YAML" \
    --timeout 900 --runs 1 \
    --peer-message-timeout 150

echo "=== smoke eval finished ==="
