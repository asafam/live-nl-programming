# Multi-agent pool smoke test — debug log

## Goal

Get the 4-TC smoke test (`workflows-mods-smoke4.jsonl`) to run end-to-end on
SLURM using the Docker multi-agent pool, so we can confirm agents make tool
calls and A2A (`sessions_send`) communications.

---

## Working baseline (jobs 14461489 / 14461491)

These jobs completed successfully:
- ~90k tokens consumed across 4 TCs
- Mean pass rate ~15.6%
- **Agents were NOT making tool calls or A2A sessions_send** — judge consistently
  said "no tool call, bus message, or recorded state entry"
- This is the actual research problem: the infrastructure works, but agents
  don't use their tools

---

## What broke (jobs 14461492 – 14461503)

A series of PermissionError crashes appeared when trying to write to files
owned by the Docker container's `node` user (UID 1000), plus a cascade of
gateway hot-reload issues. Each fix exposed the next bug.

### Bug chain (in order)

1. **PermissionError: `openclaw.json` write in `_write_worker_gateway_config`**
   - Cause: gateway rewrites openclaw.json owned by `node` (mode 600).
     Subsequent writes by our host code fail.
   - Fix: `unlink()` before `write_text()` — we own the parent dir so unlink
     succeeds without file ownership.

2. **PermissionError: `agents/…/sessions` in `_clear_agent_sessions`**
   - Cause: Python 3.10's `Path.exists()` raises PermissionError (unlike 3.12
     which suppresses it) when the parent is mode 0o000.
   - Fix: wrap `sessions_dir.exists()` and entire function body in
     `try/except PermissionError`.

3. **PermissionError: `agents/…/agent` mkdir in `_write_worker_config`**
   - Cause: `agents/<oid>/` is owned by `node`; can't create subdirs.
   - Fix: wrap the `mkdir` calls in `try/except PermissionError`.

4. **PermissionError: `agents/…/agent` mkdir in `openclaw_export.py`**
   - Same cause in `export_workflow_from_objects` and
     `export_single_agent_workspace`.
   - Fix: same `try/except PermissionError` guards.

5. **PermissionError: per-TC `openclaw.json` write in `_write_worker_config`**
   - Cause: the gateway's hot-reload cascade rewrites openclaw.json (mode 600).
     The per-TC write then fails.
   - Fix: `unlink()` before `write_text()` (same pattern as bug 1).

6. **"OpenClaw gateway did not become ready within 120s"**
   - Cause 1: `_probe_ws_connection` always returns False because the gateway
     requires token auth for WS upgrades — unauthenticated probes are rejected
     at the HTTP level.
   - Cause 2: After a per-TC `openclaw.json` write, the gateway does a
     hot-reload cascade: 3–4 self-rewrites (~2s each via inotify) before
     settling. `_wait_for_gateway_restart` polled WS (always fails) for 120s.
   - Fix: replaced `_wait_for_gateway_restart` with `asyncio.sleep(12.0)` after
     per-TC config writes. HTTP `/health` stays 200 throughout the cascade so
     no WS probe is needed.

7. **"No gateway auth token found" — SDK can't connect**
   - Cause: after the cascade, the gateway's `openclaw.json` no longer contains
     `gateway.auth.token` (the gateway manages auth from its `--token` CLI arg,
     not from the file). `_load_openclaw_token()` returns None → SDK connects
     without token → gateway rejects.
   - Fix: `_openclaw_connect_kwargs` now falls back to
     `_load_device_operator_token()` (reads `~/.openclaw/identity/device-auth.json`
     on the host — always readable, always has the operator token).
   - Also added the same fallback to `_ensure_config_auth` so future per-TC
     writes re-seed the auth section.

8. **"Gateway failed to start: non-loopback Control UI requires
   `gateway.controlUi.allowedOrigins`"**
   - Cause: when `_write_worker_config` falls back to `config = {}` (because
     the cascade-written openclaw.json is unreadable), the `controlUi` section
     is lost. The gateway binary requires `controlUi.allowedOrigins` when bound
     to a non-loopback interface (`--bind lan`).
   - Fix 1: added `"controlUi": {"dangerouslyAllowHostHeaderOriginFallback": true}`
     to `docker/openclaw-config.json` (the entrypoint template).
   - Fix 2: `_write_worker_config` now always sets
     `config["gateway"]["controlUi"]["dangerouslyAllowHostHeaderOriginFallback"] = True`
     after reading (or defaulting) the config.

---

## Files changed in this session

| File | Change |
|---|---|
| `src/data/evaluate_baseline.py` | Bugs 1–8 fixes (see above) |
| `src/lnl/openclaw_export.py` | Bug 4 fix — PermissionError guards in `mkdir` calls |
| `docker/openclaw-config.json` | Bug 8 fix — added `controlUi.dangerouslyAllowHostHeaderOriginFallback` |

---

## Remaining problem: agents don't use tools or A2A

Even in the working baseline (jobs 14461489/14461491), agents consumed ~90k
tokens but made **zero tool calls and zero sessions_send** calls. The judge
consistently reported "no tool call, bus message, or recorded state entry."

### Likely causes to investigate

1. **AGENTS.md `sessions_send` instructions**: the doctor check confirms
   sessions_send appears in AGENTS.md (`sessions_send ×3`, `sessions_send ×5`,
   etc.). The format and session target names may need verification.

2. **A2A allow list**: doctor confirms `agentToAgent.allow` is set correctly
   per-TC (e.g., `allow=['content-submissions', 'idea-expansion',
   'submissions-store']`). The gateway should route sessions_send between these.

3. **Mock tool responses**: tool calls require the mock server to return
   plausible responses. If the mock returns empty/error responses, the agent may
   decide the tool didn't help and stop calling.

4. **Agent prompt / system prompt**: the agent may be answering questions via
   its context window rather than using tools. Check `config/prompts/baseline/agent.yaml`.

5. **Model capability**: `gpt-5.4-mini` may be declining tool use — try
   `gpt-5.4` as the agent model.

---

## How to run

```bash
# Quick smoke (4 TCs, 1 run each):
~/workspace/misc/run_with_slurm.sh \
  --partition cpu192G-48h --gpus 0 --cpus 4 --mem 16G --time 1:00:00 \
  --job-name lnl-smoke --script scripts/run_pool_smoke.sh

# Full eval (24 workers):
~/workspace/misc/run_with_slurm.sh \
  --partition cpu192G-48h --gpus 0 --cpus 48 --mem 180G --time 24:00:00 \
  --job-name lnl-eval-multi24 --script scripts/run_pool_eval.sh

# Watch live:
tail -F .slurm/logs/lnl-smoke_<jobid>.out
```
