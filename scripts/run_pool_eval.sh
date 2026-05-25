#!/bin/bash
set -euo pipefail

cd /home/nlp/achimoa/workspace/live-nl-programming
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export LNL_POOL_DATA_DIR="${SLURM_TMPDIR:-/tmp}/lnl-pool-${SLURM_JOB_ID:-local}"
NUM_WORKERS=24

mkdir -p "$LNL_POOL_DATA_DIR"
mkdir -p outputs/data/zapier/runs/experiments/modifications_v3

echo "=== node $(hostname) | job ${SLURM_JOB_ID:-local} | pool data $LNL_POOL_DATA_DIR ==="

echo "=== ensuring docker image lnl-openclaw-worker is built (cached if unchanged) ==="
docker build -f docker/Dockerfile -t lnl-openclaw-worker .

echo "=== pre-creating worker dirs with chmod 777 (UID mismatch workaround) ==="
umask 000
for n in $(seq 1 $NUM_WORKERS); do
    mkdir -p "$LNL_POOL_DATA_DIR/multi-worker-$n/identity"
    mkdir -p "$LNL_POOL_DATA_DIR/multi-worker-$n/devices"
    chmod 777 "$LNL_POOL_DATA_DIR/multi-worker-$n" \
              "$LNL_POOL_DATA_DIR/multi-worker-$n/identity" \
              "$LNL_POOL_DATA_DIR/multi-worker-$n/devices"
done

cleanup() {
    echo "=== cleanup: tearing down pool ==="
    ./docker/start-pool.sh --type multi --workers $NUM_WORKERS down || true
}
trap cleanup EXIT

echo "=== starting pool: $NUM_WORKERS multi workers ==="
./docker/start-pool.sh --type multi --workers $NUM_WORKERS restart

echo "=== forcing perms on seeded files (paired.json, device-auth.json) ==="
for n in $(seq 1 $NUM_WORKERS); do
    chmod 666 "$LNL_POOL_DATA_DIR/multi-worker-$n/identity/device-auth.json" 2>&1 || true
    chmod 666 "$LNL_POOL_DATA_DIR/multi-worker-$n/devices/paired.json" 2>&1 || true
done

echo "=== docker restart all workers so they re-init under fixed perms ==="
restart_args=""
for n in $(seq 1 $NUM_WORKERS); do
    restart_args="$restart_args lnl-oc-multi-worker-$n"
done
docker restart $restart_args 2>&1 | tail -5

echo "=== sleeping 25s for workers to settle ==="
sleep 25

echo "=== generating job-specific pool YAML (data_dir must match LNL_POOL_DATA_DIR) ==="
JOB_POOL_YAML="$LNL_POOL_DATA_DIR/worker-pool-multi-$NUM_WORKERS.yaml"
{
    echo "workers:"
    GW_BASE=20788  # gateway port = GW_BASE + n
    MK_BASE=20887  # mock port   = MK_BASE + n
    for n in $(seq 1 $NUM_WORKERS); do
        gw_port=$((GW_BASE + n))
        mk_port=$((MK_BASE + n))
        echo "  - name: multi-worker-$n"
        echo "    container_name: lnl-oc-multi-worker-$n"
        echo "    gateway_url: ws://localhost:$gw_port"
        echo "    mock_server_url: http://localhost:$mk_port"
        echo "    data_dir: $LNL_POOL_DATA_DIR/multi-worker-$n"
    done
} > "$JOB_POOL_YAML"
echo "  Pool YAML: $JOB_POOL_YAML"

echo "=== running eval ==="
stdbuf -oL -eL ./scripts/run-eval-baseline.sh \
    -i data/zapier/workflows-mods.jsonl \
    -o outputs/data/zapier/runs/experiments/modifications_v3/eval_mods_baseline_multi.jsonl \
    --model gpt-5.4-mini --provider azure \
    --judge-model gpt-5.4 --judge-provider azure \
    --pool "$JOB_POOL_YAML" \
    --timeout 900 --runs 3 \
    --peer-message-timeout 150

echo "=== eval finished ==="
