#!/bin/bash
# Hypothesis B smoke: pre-registration via direct file write + per-TC peer pre-warm.
#
# Root cause of all prior crashes (smoke-B1–B7):
#   config_mgr.patch (SDK) causes the gateway to write back its full config to disk,
#   including channels/gateway.tailscale.  inotify fires, gateway diffs the new file
#   against old, detects channels/tailscale changed → "full process restart (spawned
#   pid NNN)".  Original gateway PID exits; the container entrypoint monitors that PID
#   and interprets its exit as "exited without restart" → shuts down container.
#
# Fix (this script):
#   Write openclaw.json DIRECTLY to each worker's data dir BEFORE starting containers.
#   The gateway reads the pre-written config at startup without writing back.
#   Direct file writes only trigger in-process hot-reloads (no PID change, no crash).
#
# Strategy:
#   1. Pre-write openclaw.json with all 19 agents + skipBootstrap=True + agentToAgent
#      allow list to every worker's data dir, BEFORE containers start.
#   2. Export all TC workspace files (AGENTS.md/SOUL.md/state.md) to all workers.
#   3. Start pool — gateway reads pre-written config, no cascade.
#   4. Per-TC pre-warm (in _execute_tc_async): send a brief "Initialization check"
#      to every TC agent in parallel so they're paired before sessions_send.
#
# Goal: confirm that direct-write pre-registration + per-TC pre-warm eliminates
#       "pairing required" on sessions_send without crashing containers.
set -euo pipefail

cd /home/nlp/achimoa/workspace/live-nl-programming
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export LNL_POOL_DATA_DIR="${SLURM_TMPDIR:-/tmp}/lnl-pool-B-${SLURM_JOB_ID:-local}"
mkdir -p "$LNL_POOL_DATA_DIR"

echo "=== SMOKE-B: node $(hostname) | job ${SLURM_JOB_ID:-local} | pool data $LNL_POOL_DATA_DIR ==="

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
        mock_port=$((20887 + n))
        echo "--- worker-$n mock http://localhost:$mock_port/health ---"
        curl -sS --max-time 3 "http://localhost:$mock_port/health" 2>&1 || echo "(no response)"
        echo ""
    done
    echo "=== diagnostics: openclaw effective config (worker-1, via docker exec) ==="
    docker exec lnl-oc-multi-worker-1 cat /home/node/.openclaw/openclaw.json 2>&1 | head -80 || \
        cat "$LNL_POOL_DATA_DIR/multi-worker-1/openclaw.json" 2>&1 | head -80 || true
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

echo "=== pre-registering: exporting workspaces + writing openclaw.json (no SDK/gateway needed) ==="
# Write the full agent config BEFORE starting containers so the gateway reads it at
# startup without triggering a write-back or cascade.  The operator token is read from
# the same source as start-pool.sh so the file matches the --token CLI arg the gateway
# is started with.
LNL_POOL_DATA_DIR="$LNL_POOL_DATA_DIR" python3 - <<'PYEOF'
import json, os, sys
from pathlib import Path

sys.path.insert(0, "/home/nlp/achimoa/workspace/live-nl-programming")
from src.data.schema import ObjectDef
from src.lnl.openclaw_export import export_workflow_from_objects

# Read operator token — same source as start-pool.sh uses.
_dev_auth_path = Path.home() / ".openclaw/identity/device-auth.json"
OPERATOR_TOKEN = json.loads(_dev_auth_path.read_text())["tokens"]["operator"]["token"]

INPUT = "data/zapier/workflows-mods-smoke4.jsonl"
POOL_DATA_DIR = Path(os.environ["LNL_POOL_DATA_DIR"])
CONTAINER_HOME = Path("/home/node/.openclaw")

# Load all TCs and collect unique objects across all TCs.
all_tcs = []
objects_by_id: dict = {}
with open(INPUT) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        tc_data = json.loads(line)
        all_tcs.append(tc_data)
        for obj_data in tc_data.get("objects", []):
            oid = obj_data["object_id"]
            if oid not in objects_by_id:
                objects_by_id[oid] = ObjectDef(**obj_data)

all_objects = list(objects_by_id.values())
all_ids_sorted = sorted(o.object_id for o in all_objects)
print(f"Unique agents ({len(all_objects)}): {all_ids_sorted}", flush=True)
print(f"TCs to export: {len(all_tcs)}", flush=True)

DATA_DIRS = [
    POOL_DATA_DIR / "multi-worker-1",
    POOL_DATA_DIR / "multi-worker-2",
    POOL_DATA_DIR / "multi-worker-3",
    POOL_DATA_DIR / "multi-worker-4",
]

for data_dir in DATA_DIRS:
    # Export all TC workspace files (AGENTS.md, SOUL.md, state.md).
    # write_config=False: skip openclaw.json — we write it ourselves below.
    print(f"  [{data_dir.name}] Exporting {len(all_tcs)} TC workspaces ...", flush=True)
    for tc_data in all_tcs:
        tc_objects = [ObjectDef(**o) for o in tc_data.get("objects", [])]
        export_workflow_from_objects(tc_objects, str(data_dir), force=True, write_config=False)
        # Remove BOOTSTRAP.md so agents connect silently (no LLM call on first pair).
        for obj in tc_objects:
            bs = data_dir / f"workspace-{obj.object_id}" / "BOOTSTRAP.md"
            bs.unlink(missing_ok=True)

    # Write openclaw.json directly — NO config_mgr.patch, NO SDK call, NO running gateway.
    # Direct writes trigger only an in-process hot-reload when the gateway is running,
    # NOT a "full process restart (spawned pid NNN)" that crashes the container.
    # The format matches exactly what _write_worker_config() produces so per-TC writes
    # detect "content unchanged" (when all agents are pre-registered) and skip the write.
    config = {
        "gateway": {
            "auth": {"mode": "token", "token": OPERATOR_TOKEN},
            "mode": "local",
            "trustedProxies": ["127.0.0.1", "::1", "172.16.0.0/12"],
            "controlUi": {"dangerouslyAllowHostHeaderOriginFallback": True},
        },
        "plugins": {"allow": ["lnl-mock-external"]},
        "tools": {
            "sessions": {"visibility": "all"},
            "agentToAgent": {"enabled": True, "allow": all_ids_sorted},
        },
        "commands": {"native": "auto", "nativeSkills": "auto", "restart": True},
        "agents": {
            "defaults": {"model": "azure/gpt-5.4-mini", "skipBootstrap": True},
            "list": [
                {
                    "id": obj.object_id,
                    "name": obj.object_id.replace("-", " ").title(),
                    "workspace": str(CONTAINER_HOME / f"workspace-{obj.object_id}"),
                    "agentDir": str(CONTAINER_HOME / "agents" / obj.object_id / "agent"),
                    "model": {"primary": "azure/gpt-5.4-mini"},
                }
                for obj in all_objects
            ],
        },
    }
    config_path = data_dir / "openclaw.json"
    config_path.unlink(missing_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"  [{data_dir.name}] Wrote openclaw.json ({len(all_objects)} agents, mode {oct(config_path.stat().st_mode)[-3:]})", flush=True)

print("Pre-write complete.", flush=True)
PYEOF

echo "=== starting pool: 4 multi workers (reads pre-written openclaw.json at startup) ==="
./docker/start-pool.sh --type multi --workers 4 restart

echo "=== forcing perms on seeded files ==="
for n in 1 2 3 4; do
    chmod 666 "$LNL_POOL_DATA_DIR/multi-worker-$n/identity/device-auth.json" 2>&1 || true
    chmod 666 "$LNL_POOL_DATA_DIR/multi-worker-$n/devices/paired.json" 2>&1 || true
done

echo "=== docker restart all workers so they re-init with proper perms ==="
docker restart lnl-oc-multi-worker-1 lnl-oc-multi-worker-2 lnl-oc-multi-worker-3 lnl-oc-multi-worker-4 2>&1 | head -10

echo "=== sleeping 20s for workers to settle ==="
sleep 20

echo "=== waiting for gateway HTTP health endpoints (max 180s) ==="
python3 - <<'PYEOF'
import sys, time, httpx
PORTS = [20789, 20790, 20791, 20792]
for port in PORTS:
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            if r.status_code == 200:
                print(f"  http://localhost:{port}/health ready", flush=True)
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print(f"  http://localhost:{port} STILL DOWN after 180s", flush=True)
        sys.exit(1)
PYEOF

echo "=== generating job-specific pool YAML ==="
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

echo "=== post-start config snapshot (all workers, via host file read) ==="
for n in 1 2 3 4; do
    echo "  worker-$n:"
    python3 -c "
import json
from pathlib import Path
import os
p = Path(os.environ.get('LNL_POOL_DATA_DIR', '/tmp')) / 'multi-worker-$n' / 'openclaw.json'
try:
    c = json.loads(p.read_text())
    agents = [a.get('id') for a in c.get('agents', {}).get('list', [])]
    allow = c.get('tools', {}).get('agentToAgent', {}).get('allow', [])
    skip = c.get('agents', {}).get('defaults', {}).get('skipBootstrap', 'MISSING')
    n_agents = len(agents)
    mode = oct(p.stat().st_mode)[-3:]
    print(f'    mode={mode}  skipBootstrap={skip}  agents={n_agents}  allow_count={len(allow)}')
    if n_agents > 0:
        print(f'    first 3 agents: {agents[:3]}')
except Exception as e:
    print(f'    (error: {e})')
" 2>/dev/null
done

echo "=== running smoke-B eval (4 TCs, 1 run each) ==="
stdbuf -oL -eL ./scripts/run-eval-baseline.sh \
    -i data/zapier/workflows-mods-smoke4.jsonl \
    --model gpt-5.4-mini --provider azure \
    --judge-model gpt-5.4 --judge-provider azure \
    --pool "$JOB_POOL_YAML" \
    --timeout 900 --runs 1 \
    --peer-message-timeout 150

echo "=== smoke-B eval finished ==="
