# Per-Trace Plans, Typed Step Results, and Concurrent Cascades

Design doc for moving the LLM-object runtime from single-plan-per-object to
trace-isolated plans with durable typed step results.

Anchor: `git tag checkpoint-pre-per-trace-plans` (commit `21e0961`).

---

## 1. Motivation

Today the runtime has three structural issues that compound when an object
participates in more than one cascade at a time:

1. **One plan per object.** `LLMObject._active_plan: Optional[Plan]` is
   singular. A second `DOMAIN` message arriving while a prior plan is still
   open does not get its own plan — the planner gate skips, and any new
   outgoings either extend the existing plan or get fuzzy-matched onto its
   steps. Two unrelated cascades through the same object interfere.

2. **Step results are not durable.** A reply marks its step `done` but the
   reply payload only lands in `_history` (bounded, 6 messages) and in
   `_state` *iff* the LLM remembers to extract it. If the LLM forgets, the
   data is lost before later steps can use it. Tool returns have the same
   problem — they live in `_history` as `[Tool result for ...]` strings,
   never attached to a plan step.

3. **State semantics are unclear.** `_state` is used as both durable world
   knowledge and per-request scratch. This is what forces the LLM to copy
   reply payloads into state — there's nowhere else to put them.

## 2. Core decisions

| Decision | Choice | Rationale |
|---|---|---|
| Plan isolation | Per `trace_id`, dict-keyed | Replies already carry trace_id; the runtime just needs to use it |
| Within-object concurrency | Single slot, FIFO mailbox (no change) | True parallel processing in one object requires state locking and a much bigger rewrite. Per-trace plan isolation is the correctness win; serial execution per-object is fine. |
| State scope | Shared across traces, serialized by mailbox | State is for durable world facts. Two cascades both writing the same fact is acceptable (deterministic last-write-wins by mailbox order). |
| Working memory | Per-trace, on the plan's steps | Reply payloads and tool returns live on `PlanStep.result`, not state. |
| Step result format | Native shape preserved | NL string for peer replies; structured dict/list for tool returns; short note for reason steps. No coercion. |
| Step identifier | Position index (`plan_step_index: int`) | Unchanged. LLM reasons about steps by position, never authors IDs. |
| Step kinds | `ask` \| `tell` \| `tool` \| `reason` | Rename `effect` → `reason`; promote `tool` to first-class. |
| Executor model | Parallel-dispatch per ReAct turn (no change) | One turn may dispatch multiple `planned` steps when they have no data dependency. Preserves `ask B / ask C` parallelism. |
| Evaluator scope | Per-turn (no change) | One LLM call grades the whole turn. Per-step would multiply LLM cost. |
| State schema enforcement | Deferred | Out of scope for this doc. |

## 3. Data model changes

### `Plan` (src/lnl/types.py)

```python
@dataclass
class Plan:
    trace_id: str                            # NEW — keys _active_plans
    goal: str
    steps: list[PlanStep]
    status: str = "active"                   # active|complete|cancelled|abandoned|failed
    created_at: datetime                     # NEW
    last_progress_at: datetime               # NEW — bumped on every step transition
```

### `PlanStep`

```python
@dataclass
class PlanStep:
    kind: str                                # "ask" | "tell" | "tool" | "reason"
    description: str
    target: Optional[str]                    # peer_id for ask/tell; tool name for tool; None for reason
    status: str = "planned"                  # planned | dispatched | done | failed
    result: Optional[Union[str, dict, list]] = None   # NEW — native shape
    result_kind: Optional[str] = None        # NEW — "nl" | "tool" | "reason"
    completed_at: Optional[datetime] = None  # NEW
    result_summary: Optional[str] = None     # existing — LLM-emitted note
```

### `LLMObject` (src/lnl/object.py)

```python
self._active_plans: dict[str, Plan] = {}                       # was Optional[Plan]
self._completed_plans: deque[Plan] = deque(maxlen=64)          # bounded archive
# REMOVE: self._planned_traces  (subsumed by _active_plans key check)
```

### `SystemConfig` (src/lnl/runtime.py)

```python
stale_plan_seconds: float = 180.0           # NEW — idle timeout for plan abandonment
max_active_plans_per_object: int = 32       # NEW — cardinality cap
```

## 4. Stages

### Stage 1 — Per-trace plan storage

Plumb `trace_id` through every plan op. Each method that touches `_active_plan`
takes a `trace_id` arg (read from the current `Message`):

| Method | New signature |
|---|---|
| `active_plan` property | `plan_for(trace_id) -> Optional[Plan]` |
| `_apply_plan_update` | `(update, trace_id)` |
| `_auto_mark_step_on_reply` | `(idx, trace_id, reply_content)` |
| `_correlate_outgoing` | `(outgoing, trace_id)` |
| `_auto_create_plan_from_outgoing` | `(outgoing, message)` — uses `message.trace_id` |
| `mark_step_dispatched` | `(idx, trace_id)` |
| `_auto_close_plan_if_complete` | `(trace_id)` |
| `_mark_reason_steps_done` | `(trace_id)` (renamed from `_mark_effect_steps_done`) |

**Planner gate** changes from `active_plan is None` to
`trace_id not in self._active_plans`.

**Prompt rendering** (`build_system_prompt`): accept `trace_id`, render
*only* `_active_plans.get(trace_id)`. Never expose other traces' plans —
they are noise and a workflow-isolation issue.

**Runtime side** (`runtime.py::_on_result`): when calling
`mark_step_dispatched`, pass `out.trace_id` (already on the outgoing's
chained `Message`).

### Stage 2 — Typed durable step results

Every reply, tool return, or reasoning closure auto-populates `step.result`
with its native shape:

| Step kind | Trigger | `result` shape | `result_kind` |
|---|---|---|---|
| `ask` | Peer reply arrives | `str` (reply content) | `"nl"` |
| `tell` | Bus dispatch | `None` | — |
| `tool` | Tool returns | structured (`dict`/`list`/scalar) | `"tool"` |
| `reason` | Evaluator PASS or explicit `plan_update.step_updates[i]` | `str` (LLM's note) | `"reason"` |

**Reply path** (`_auto_mark_step_on_reply`):

```python
step = plan.steps[step_index]
if step.status not in STEP_TERMINAL_STATUSES:
    step.status = "done"
    step.result = reply_content       # message.content, verbatim
    step.result_kind = "nl"
    step.completed_at = utcnow()
    plan.last_progress_at = utcnow()
```

**Tool path** (`_run_react_cycle`):
- The LLM's `tool_call` emission may include `plan_step_index: int`.
- After `ToolResult` is returned, if `plan_step_index` is set, store the
  tool's structured output on that step:

```python
if tc.plan_step_index is not None:
    plan = self._active_plans.get(message.trace_id)
    if plan and 0 <= tc.plan_step_index < len(plan.steps):
        step = plan.steps[tc.plan_step_index]
        step.status = "done"
        step.result = result.output   # raw, structured
        step.result_kind = "tool"
        step.completed_at = utcnow()
```

This requires plumbing `plan_step_index` on `ToolCall` (`src/lnl/types.py`)
and updating the parser in `brain.py` to round-trip it.

**Reason path** (`_mark_reason_steps_done`):
- On evaluator PASS or explicit `plan_update.step_updates[i].status="done"`,
  if `result_summary` was provided by the LLM, copy it to `step.result` with
  `result_kind="reason"`.

**Prompt rendering** of each step:

```
[0] ask objB: get customer email
    status: done
    result (nl): "john@snow.com"
[1] tool python: compute discount tier
    status: done
    result (tool): {"tier": "gold", "discount_pct": 15}
[2] reason: decide whether to attach receipt PDF based on order total
    status: done
    result (reason): "Order total > $500, attach PDF"
[3] tell objD: dispatch the email with email, discount, and attachment flag
    status: planned
```

Downstream steps' ReAct turns see prior results in the prompt directly —
no state-write gymnastics required.

### Stage 3 — Retirement policy

A plan is retired (moved from `_active_plans` to `_completed_plans`) when
any of:

| Trigger | Status set |
|---|---|
| All steps terminal | `complete` |
| `now - last_progress_at > stale_plan_seconds` | `abandoned` |
| Evaluator FAIL cycle cap reached AND plan still has non-terminal steps | `failed` |
| `len(_active_plans) > max_active_plans_per_object` | force-retire the oldest by `last_progress_at` as `abandoned` |

**Sweep placement:** `_sweep_stale_plans()` runs at the top of
`process_message`, before the planner gate. Cheap — single iteration over
`_active_plans`.

**`_completed_plans`** becomes a bounded `deque(maxlen=64)` so retired plans
don't grow without bound.

### Stage 4 — State contract clarification

No code change to `_state` itself. Update prompts and docs to make the
contract explicit:

`config/prompts/lnl/object.yaml` adds a section:

> **State vs step results**
> - `state` is for durable facts about the world that future requests will
>   care about (inventory, preferences, accumulated knowledge).
> - Step results from the current plan are already visible to you in the
>   plan checklist — DO NOT copy them into state unless they are durable
>   world facts.
> - When composing a downstream step, reference prior step results from the
>   plan directly; only persist to state what should outlive this request.

`LLMObject._state` docstring adds:

> State writes from different traces are serialized by the mailbox FIFO —
> there is no concurrent mutation. "Last write wins" means "the last message
> processed wins," which is deterministic by arrival order, not a race.

## 5. Explicit non-goals

Write these into the code as comments at the relevant call sites so they
don't get refactored away by accident:

1. **Executor stays parallel-dispatch-per-turn.** One `process_message` turn
   may dispatch multiple `planned` steps in a single ReAct finish when there
   are no data dependencies. The runtime does **not** walk steps one-at-a-
   time. This is what gives parallelism on `ask B / ask C`.

2. **Step identifier stays `plan_step_index: int`** (position-based). Do
   not introduce step IDs. The LLM reasons about steps by position.

3. **Evaluator stays per-turn.** One LLM call grades the whole turn against
   the active plan for the current trace. Retries capped per-trace via
   `evaluator_max_cycles_per_trace`.

4. **Evaluator returns `verdict + criteria + feedback`**, never an edited
   plan. The executor mutates the plan on the next ReAct cycle in response
   to feedback.

5. **State stays shared across traces.** Do not partition `_state` by
   `trace_id`. Working memory belongs on plans.

6. **Single-slot execution per object.** Do not break the `_active` /
   mailbox-FIFO model. Two messages for the same object never run
   concurrently.

## 6. Migration & risk

**Backward compatibility**
- `.md` object files: no change.
- `LLMObject.active_plan` property: keep as a thin accessor that returns
  `None` if `_active_plans` is empty, or the most-recently-updated plan if
  there is exactly one. Callers that need a specific trace should call
  `plan_for(trace_id)`. Mark deprecated.
- `snapshot()` returns `active_plans: dict[trace_id, plan_dict]` instead of
  a single `active_plan` field. Tests + debug tooling that read snapshots
  need an update.

**Tests to add** (alongside existing `test_object.py`, `test_runtime.py`):
- Two concurrent traces through the same object → two separate plans, no
  cross-talk in step indices or results.
- Reply to step N auto-populates `step.result` and `step.result_kind`.
- Tool call tagged with `plan_step_index` lands as structured `result`.
- Stale plan past `stale_plan_seconds` transitions to `abandoned`.
- Cardinality cap evicts oldest active plan.
- Step result is rendered in the system prompt for downstream steps.
- Reason step closure on evaluator PASS populates `result_summary` →
  `result`.

**Risk surface**
- The `mark_step_dispatched` call site in `runtime.py:702` must pass the
  outgoing's `trace_id`. Missing this leaves Ask steps stuck in `planned`.
- Tool-call plumbing of `plan_step_index` is the largest new concept; if
  the LLM doesn't tag tool calls with a step index, behavior degrades
  gracefully (result not stored, step not auto-closed) — same as today.
- Plan retirement timing: aggressive sweep + a slow downstream peer can
  abandon a still-valid plan. Initial `stale_plan_seconds=180` is
  conservative; tune based on telemetry.

## 7. Implementation order

Recommended sequence (each stage shippable independently):

1. **Stage 1** (per-trace plans): big mechanical change, no behavior change
   when traces don't overlap. Largest correctness win.
2. **Stage 2 reply path** (`step.result` for `ask` replies): unlocks the
   prompt-rendering improvement; no LLM API contract change.
3. **Stage 2 reason path** (`reason` steps populate `result` on closure):
   small follow-up.
4. **Stage 2 tool path** (`plan_step_index` on `ToolCall`): requires LLM
   prompt update + parser change. Ship last; gracefully degrades if LLM
   omits the field.
5. **Stage 3** (retirement policy): can land any time after Stage 1.
6. **Stage 4** (prompt + docstring): independent, can land first or last.

## 8. Open follow-ups (not in scope)

- **EVENT vs DOMAIN unification.** Should externally-injected `EVENT`
  messages also trigger the planner? Today only `DOMAIN` does. Recommend
  yes, but defer the decision to a separate small PR.
- **State JSON schema.** Per-object state schema for the LLM to validate
  against at commit time. Defer.
- **Within-object concurrency.** Allow two `process_message` calls in
  parallel on the same object (one per trace), with explicit `_state`
  locking. Big change; only justified if telemetry shows per-object
  serialization becoming a bottleneck.
