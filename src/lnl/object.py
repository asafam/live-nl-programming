"""LLMObject — the single runtime entity in the LNL system."""
from __future__ import annotations

import datetime
import json
import logging
import threading
from collections import deque
from dataclasses import asdict
from typing import Callable, Optional

from .brain import (
    LLMBrain,
    _build_chat_messages,
    build_system_prompt,
)
from .tools import ToolRegistry
from .types import (
    PLAN_TERMINAL_STATUSES,
    STEP_TERMINAL_STATUSES,
    InferenceMetrics,
    Message,
    MessageType,
    ObjectDefinition,
    PeerDeclaration,
    Plan,
    PlanStep,
    PlanUpdate,
    ProcessingResult,
    ReactFinish,
    StateDelta,
    ToolResult,
)

logger = logging.getLogger(__name__)


class LLMObject:
    """An LLM-object: definition + brain + mutable NL state."""

    def __init__(
        self,
        definition: ObjectDefinition,
        brain: LLMBrain,
        tool_registry: ToolRegistry | None = None,
        tool_context_factory: object = None,
        max_tool_rounds: int = 5,
        max_history: int = 6,
        react_cross_objects: bool = True,
        pending_timeout_seconds: float = 90.0,
        heartbeat_interval_seconds: float = 30.0,
    ) -> None:
        self._definition = definition
        self._brain = brain
        self._state = ""  # mutable runtime state (str from LLM; dict from mock scripts)
        self._history: list[Message] = []
        self._mailbox: deque[Message] = deque()
        self._tool_registry = tool_registry
        self._tool_context_factory = tool_context_factory
        self._lock = threading.Lock()   # guards _mailbox and _active
        self._active = False            # True while scheduled or running on pool
        self._max_tool_rounds = max_tool_rounds
        self._max_history = max_history
        self._react_cross_objects = react_cross_objects
        self._pending_timeout_seconds = pending_timeout_seconds
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        # Plan registry — holds both active and terminated plans. Only active
        # plans are rendered into the prompt; terminated ones are kept for
        # debugging/introspection.
        self._plans: dict[str, Plan] = {}
        self._plan_counter: int = 0
        self._plans_lock = threading.Lock()
        # Pending inbound Asks this object has not yet replied to. Keyed by
        # sender; each value is (plan_id, step_id, message_id) — the
        # correlation tags the asker attached to the Ask, which we'll stamp
        # onto the eventual reply so the asker's plan auto-correlates.
        # Only replied-to once: cleared when we route a reply back.
        self._pending_inbound_asks: dict[str, tuple[Optional[str], Optional[str], str]] = {}
        self._pending_inbound_lock = threading.Lock()

    # --- Properties ---

    @property
    def object_id(self) -> str:
        return self._definition.object_id

    @property
    def state(self):
        """Return state: a dict if parseable, otherwise the raw string (or {} if empty)."""
        return _coerce_state(self._state)

    @property
    def definition(self) -> ObjectDefinition:
        return self._definition

    @property
    def peer_ids(self) -> list[str]:
        return [p.object_id for p in self._definition.peers]

    @property
    def subscriptions(self) -> list[str]:
        return list(self._definition.subscriptions)

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    # --- Plans ---

    @property
    def plans(self) -> dict[str, Plan]:
        """All plans (active + terminated), keyed by id. Read-only snapshot."""
        with self._plans_lock:
            return dict(self._plans)

    def active_plans(self) -> list[Plan]:
        """Return non-terminal plans in creation order. Used to render the prompt."""
        with self._plans_lock:
            return [p for p in self._plans.values() if p.status not in PLAN_TERMINAL_STATUSES]

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Look up a plan by id (active or terminated)."""
        with self._plans_lock:
            return self._plans.get(plan_id)

    def _next_plan_id(self) -> str:
        """Mint a deterministic plan id: plan-{object_id}-{n}."""
        with self._plans_lock:
            n = self._plan_counter
            self._plan_counter += 1
        return f"plan-{self.object_id}-{n}"

    def mark_step_dispatched(self, plan_id: str, step_id: str, message_id: str) -> None:
        """Called by the runtime after a plan-tagged outgoing message is put on the bus.

        Records the bus message id and `dispatched_at` on the step. For Ask
        steps (still 'planned'), advances status to 'dispatched' so replies
        correlate and heartbeat timeouts can be computed. For Tell steps
        (already 'done' after auto-correlation) or otherwise-terminal steps,
        leaves status intact.
        """
        with self._plans_lock:
            plan = self._plans.get(plan_id)
            if not plan:
                return
            now = datetime.datetime.now(datetime.timezone.utc)
            for step in plan.steps:
                if step.id == step_id:
                    step.message_id = message_id
                    step.dispatched_at = now
                    if step.status not in STEP_TERMINAL_STATUSES:
                        step.status = "dispatched"
                    plan.updated_at = now
                    return

    # --- Mailbox ---

    @property
    def has_pending(self) -> bool:
        """True if the mailbox has messages waiting to be processed."""
        return bool(self._mailbox)

    @property
    def mailbox(self) -> deque[Message]:
        return self._mailbox

    def deliver(self, message: Message, schedule_callback: Optional[Callable] = None) -> None:
        """Put a message in this object's mailbox.

        If a schedule_callback is provided and the object is not already active,
        marks the object active and calls the callback to schedule it on the pool.
        """
        with self._lock:
            self._mailbox.append(message)
            if not self._active:
                self._active = True
                if schedule_callback:
                    schedule_callback(self)

    def read(self, on_result: Callable[[ProcessingResult], None]) -> None:
        """Execute pending messages until the mailbox is empty, then yield.

        Designed to run on a thread pool. The object owns its execution:
        it dequeues messages one at a time and calls on_result after each,
        releasing its active flag only when the mailbox is confirmed empty.
        """
        while True:
            with self._lock:
                if not self._mailbox:
                    self._active = False
                    return
                message = self._mailbox.popleft()
            result = self.process_message(message)  # LLM call outside lock
            on_result(result)

    def process_next(self) -> ProcessingResult | None:
        """Process the next message from the mailbox (batch/test helper)."""
        if not self._mailbox:
            return None
        message = self._mailbox.popleft()
        return self.process_message(message)

    # --- Core Processing (ReAct loop) ---

    def process_message(self, message: Message) -> ProcessingResult:
        """Process an incoming message via a ReAct loop: think → act → observe → repeat."""
        state_before = self._state  # snapshot — state only committed after successful loop

        # Reply-driven auto-mark: if this incoming message is a correlated reply
        # to one of our planned steps, mark that step done BEFORE the LLM runs,
        # so the rendered Current Plans snapshot reflects reality.
        if message.plan_id and message.step_id:
            self._auto_mark_step_on_reply(message.plan_id, message.step_id)

        # Record pending inbound Asks so that a later reply to the asker
        # — possibly on a different turn — can auto-correlate back with the
        # asker's original plan tags. Only peer Asks count; heartbeats and
        # system messages are excluded.
        if (
            message.expects_reply
            and message.type == MessageType.DOMAIN
            and message.sender not in ("__user__", "__system__", "__external__", "__code__")
        ):
            with self._pending_inbound_lock:
                self._pending_inbound_asks[message.sender] = (
                    message.plan_id, message.step_id, message.id,
                )

        tools_desc = self._tool_registry.describe() if self._tool_registry else ""
        sys_prompt = build_system_prompt(
            self._definition, self._state,
            tools=tools_desc,
            react_cross_objects=self._react_cross_objects,
            pending_timeout_seconds=self._pending_timeout_seconds,
            heartbeat_interval_seconds=self._heartbeat_interval_seconds,
            current_plans=self.active_plans(),
        )
        messages = _build_chat_messages(sys_prompt, self._history, message)

        total_metrics = InferenceMetrics(model="")
        finish: ReactFinish | None = None
        tool_rounds = 0
        pending_deltas: list[StateDelta] = []
        pending_plan_updates: list[PlanUpdate] = []

        while True:
            step, metrics = self._brain.react_call(messages, object_id=self.object_id)
            total_metrics = _accumulate_metrics(total_metrics, metrics)

            if step.state_update:
                pending_deltas.append(step.state_update)

            if step.plan_update:
                pending_plan_updates.append(step.plan_update)

            if step.action == "finish":
                finish = step.finish
                break

            # action == "tool_call"
            if tool_rounds >= self._max_tool_rounds:
                # Hard stop — manufacture an empty finish to avoid infinite loops.
                finish = ReactFinish(reply="", updated_state=self._state)
                break

            tc = step.tool_call
            if not self._tool_registry or tc is None:
                # No registry — tell the LLM tools are unavailable and let it finish.
                messages.append({"role": "assistant", "content": json.dumps({
                    "thought": step.thought,
                    "action": "tool_call",
                    "tool_call": {"id": tc.id if tc else "", "tool": tc.tool if tc else "", "arguments": {}},
                })})
                messages.append({"role": "user", "content": "[Tool execution unavailable — no tool registry is configured. Please provide your final answer.]"})
                continue

            tool_rounds += 1
            ctx = self._tool_context_factory(self) if self._tool_context_factory else {}
            try:
                result = self._tool_registry.execute(tc, ctx)
            except Exception as exc:
                result = ToolResult(id=tc.id, output="", error=f"Tool execution raised an exception: {exc}")

            messages.append({"role": "assistant", "content": json.dumps({
                "thought": step.thought,
                "action": "tool_call",
                "tool_call": {"id": tc.id, "tool": tc.tool, "arguments": tc.arguments},
            })})
            messages.append({"role": "user", "content": f"[Tool result for {tc.id}]: {result.output}" + (f"\nError: {result.error}" if result.error else "")})

        if finish is None:
            finish = ReactFinish(reply="")

        if pending_deltas:
            current = _coerce_state(self._state)
            if not isinstance(current, dict):
                current = {}
            for delta in pending_deltas:
                current = _apply_delta(current, delta)
            self._state = json.dumps(current)
        elif finish.updated_state:
            # Backward compat: MockBrain / test scripts that set updated_state directly
            self._state = finish.updated_state
        # else: no deltas, no updated_state → state unchanged

        for update in pending_plan_updates:
            self._apply_plan_update(update, incoming_message=message)

        outgoing = self._correlate_outgoing_to_plans(finish.outgoing_messages)

        if finish.updated_definition:
            self._apply_definition_update(finish.updated_definition)
        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return ProcessingResult(
            object_id=self.object_id,
            reply=finish.reply,
            outgoing_messages=outgoing,
            state_before=_coerce_state(state_before),
            state_after=_coerce_state(self._state),
            metrics=total_metrics,
            in_reply_to=message.sender,
            source_message_type=message.type,
            depth_remaining=message.depth_remaining,
            source_message_id=message.id,
            source_plan_id=message.plan_id,
            source_step_id=message.step_id,
        )

    # --- Plan application ---

    def _resolve_plan_ref(self, plan_ref: Optional[str]) -> Optional[Plan]:
        """Resolve an LLM-provided `plan` goal-string to one active plan.

        Matching rules (case-insensitive):
        1. Exact goal match → that plan.
        2. Substring match to exactly one active plan → that plan.
        3. Zero or multiple matches → None (caller logs a warning).
        """
        if not plan_ref:
            return None
        ref = plan_ref.strip().lower()
        if not ref:
            return None
        with self._plans_lock:
            active = [p for p in self._plans.values() if p.status not in PLAN_TERMINAL_STATUSES]
        if not active:
            return None
        exact = [p for p in active if p.goal.strip().lower() == ref]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            # Multiple plans share this goal exactly — pick the most recent.
            return max(exact, key=lambda p: p.created_at)
        partial = [p for p in active if ref in p.goal.strip().lower()]
        if len(partial) == 1:
            return partial[0]
        return None  # ambiguous or no match

    def _apply_plan_update(self, update: PlanUpdate, incoming_message: Optional[Message] = None) -> None:
        """Apply one plan update.

        Semantic references only — plan ids never cross the LLM boundary:
        - `create`: runtime mints plan id + step ids.
        - `add_step` / `complete` / `cancel`: plan identified by `update.plan` (goal match).
        - `update_step`: plan+step identified either by incoming reply's plan tags
          (message.plan_id/step_id) or, for off-message updates, by
          `update.plan` + `update.step_index`.
        """
        op = update.op
        now = datetime.datetime.now(datetime.timezone.utc)
        trigger_message_id = incoming_message.id if incoming_message else ""

        if op == "create":
            plan_id = self._next_plan_id()
            steps = []
            for i, raw in enumerate(update.steps or []):
                if not isinstance(raw, dict):
                    continue
                kind = raw.get("kind")
                if kind not in ("ask", "tell"):
                    logger.warning(
                        "Plan create for %s: dropping step with unsupported kind=%r (only 'ask'/'tell' are plan steps)",
                        self.object_id, kind,
                    )
                    continue
                steps.append(PlanStep(
                    id=f"{plan_id}-s{i}",
                    kind=kind,
                    description=raw.get("description", ""),
                    target=raw.get("target"),
                ))
            plan = Plan(
                id=plan_id,
                goal=update.goal or "",
                trigger_message_id=trigger_message_id,
                steps=steps,
                created_at=now,
                updated_at=now,
            )
            with self._plans_lock:
                self._plans[plan_id] = plan
            return

        # All other ops require a plan reference.
        plan: Optional[Plan] = None

        if op == "update_step":
            # Prefer the incoming reply's plan context when present.
            if incoming_message and incoming_message.plan_id and incoming_message.step_id:
                with self._plans_lock:
                    plan = self._plans.get(incoming_message.plan_id)
                if plan:
                    self._apply_step_update(plan, incoming_message.step_id, update, now)
                    return
                # Tagged message but no such plan — fall through to explicit ref.
            # Off-message update_step: require plan + step_index.
            plan = self._resolve_plan_ref(update.plan)
            if not plan:
                logger.warning(
                    "Plan update_step for %s: could not resolve plan ref=%r — dropped",
                    self.object_id, update.plan,
                )
                return
            if update.step_index is None or update.step_index < 0:
                logger.warning(
                    "Plan update_step for %s plan=%r: missing or invalid step_index — dropped",
                    self.object_id, plan.goal,
                )
                return
            with self._plans_lock:
                if update.step_index >= len(plan.steps):
                    logger.warning(
                        "Plan update_step for %s plan=%r: step_index=%d out of range (plan has %d steps) — dropped",
                        self.object_id, plan.goal, update.step_index, len(plan.steps),
                    )
                    return
                step = plan.steps[update.step_index]
            self._apply_step_update(plan, step.id, update, now)
            return

        # add_step / complete / cancel — resolve plan by goal ref.
        plan = self._resolve_plan_ref(update.plan)
        if not plan:
            logger.warning(
                "Plan update op=%s for %s: could not resolve plan ref=%r — dropped",
                op, self.object_id, update.plan,
            )
            return

        if op == "add_step":
            with self._plans_lock:
                base_n = len(plan.steps)
                for i, raw in enumerate(update.steps or []):
                    if not isinstance(raw, dict):
                        continue
                    kind = raw.get("kind")
                    if kind not in ("ask", "tell"):
                        continue
                    plan.steps.append(PlanStep(
                        id=f"{plan.id}-s{base_n + i}",
                        kind=kind,
                        description=raw.get("description", ""),
                        target=raw.get("target"),
                    ))
                plan.updated_at = now
            return

        if op in ("complete", "cancel"):
            with self._plans_lock:
                plan.status = "complete" if op == "complete" else "cancelled"
                plan.updated_at = now
            return

        logger.warning("Plan update unsupported op=%r for %s — dropped", op, self.object_id)

    def _apply_step_update(self, plan: Plan, step_id: str, update: PlanUpdate, now: datetime.datetime) -> None:
        """Internal: apply a status/result_summary change to a specific step within a plan."""
        with self._plans_lock:
            for step in plan.steps:
                if step.id == step_id:
                    if update.status in STEP_TERMINAL_STATUSES:
                        step.status = update.status
                    elif update.status:
                        logger.warning(
                            "Plan update_step for %s/%s: status=%r not allowed (terminal only: %s) — ignored",
                            self.object_id, plan.id, update.status, STEP_TERMINAL_STATUSES,
                        )
                    if update.result_summary is not None:
                        step.result_summary = update.result_summary
                    plan.updated_at = now
                    return
            logger.warning(
                "Plan update_step for %s/%s: unknown step_id=%s — dropped",
                self.object_id, plan.id, step_id,
            )

    def _auto_mark_step_on_reply(self, plan_id: str, step_id: str) -> None:
        """When a correlated reply arrives, mark the target step 'done' automatically.

        The LLM can still override to 'failed' via an update_step op in the same
        turn — that override runs after the LLM responds, so the override wins.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        with self._plans_lock:
            plan = self._plans.get(plan_id)
            if not plan:
                return
            for step in plan.steps:
                if step.id == step_id:
                    # Only auto-promote if not already terminal.
                    if step.status not in STEP_TERMINAL_STATUSES:
                        step.status = "done"
                        plan.updated_at = now
                    return

    def _correlate_outgoing_to_plans(self, outgoing):
        """Auto-stamp correlation on outgoing messages.

        Two paths, checked in order:
        1. **Plan dispatch**: the outgoing matches one of our own active plan
           steps (`target == recipient` AND `kind` matches `expects_reply`).
           On match, stamp `plan_id`/`step_id` with our plan's correlation and
           transition the step (Tell → done, Ask → dispatched on bus send).
        2. **Pending inbound reply**: the recipient had sent us an Ask we
           haven't replied to yet. Stamp the outgoing with the ASKER's plan
           correlation (from our `_pending_inbound_asks`), mark `is_reply=True`
           so the runtime delivers it as MessageType.REPLY, and clear the
           pending entry. This is how a nested chain (A→B→C→B→A) lets B's
           eventual answer to A auto-correlate back to A's original plan.

        Messages that match neither pass through as regular uncorrelated sends.
        """
        if not outgoing:
            return outgoing
        with self._plans_lock:
            active = [p for p in self._plans.values() if p.status not in PLAN_TERMINAL_STATUSES]

        now = datetime.datetime.now(datetime.timezone.utc)
        for out in outgoing:
            # The LLM no longer authors correlation — always start clean.
            out.plan_id = None
            out.step_id = None
            out.in_reply_to = None
            out.is_reply = False

            # Path 1: match to one of our own plan steps.
            wanted_kind = "ask" if out.expects_reply else "tell"
            matched_step = None
            matched_plan = None
            for plan in active:
                for step in plan.steps:
                    if (
                        step.status == "planned"
                        and step.kind == wanted_kind
                        and step.target == out.recipient
                    ):
                        matched_step = step
                        matched_plan = plan
                        break
                if matched_step:
                    break
            if matched_step and matched_plan:
                out.plan_id = matched_plan.id
                out.step_id = matched_step.id
                with self._plans_lock:
                    if matched_step.kind == "tell":
                        matched_step.status = "done"
                    matched_plan.updated_at = now
                continue

            # Path 2: reply to a pending inbound Ask.
            with self._pending_inbound_lock:
                pending = self._pending_inbound_asks.pop(out.recipient, None)
            if pending is not None:
                pid, sid, mid = pending
                if pid:
                    out.plan_id = pid
                if sid:
                    out.step_id = sid
                if mid:
                    out.in_reply_to = mid
                out.is_reply = True

        return outgoing

    # --- Live Modification ---

    def modify_definition(self, **updates: object) -> None:
        """Change definition fields WITHOUT resetting state."""
        for key, value in updates.items():
            if not hasattr(self._definition, key):
                raise AttributeError(f"ObjectDefinition has no field '{key}'")
            setattr(self._definition, key, value)

    _PATCHABLE_DEFINITION_FIELDS = {"role", "behavior"}

    def _apply_definition_update(self, patch: dict) -> None:
        """Apply a definition patch from the LLM (admin-driven self-modification)."""
        updates = {k: v for k, v in patch.items() if k in self._PATCHABLE_DEFINITION_FIELDS}
        if "peers" in patch and isinstance(patch["peers"], list):
            updates["peers"] = [
                PeerDeclaration(object_id=p["object_id"], relationship=p["relationship"])
                for p in patch["peers"]
                if isinstance(p, dict)
            ]
        if updates:
            self.modify_definition(**updates)

    # --- Testing / Debugging ---

    def set_state(self, state: str | dict) -> None:
        """Set state directly (for testing). Accepts str or dict (dict is JSON-encoded)."""
        if isinstance(state, dict):
            import json as _json
            self._state = _json.dumps(state)
        else:
            self._state = state

    def snapshot(self) -> dict:
        """Return a debug snapshot of the object."""
        with self._plans_lock:
            plans_snap = {pid: asdict(p) for pid, p in self._plans.items()}
        return {
            "object_id": self.object_id,
            "state": _coerce_state(self._state),
            "definition": asdict(self._definition),
            "history_length": len(self._history),
            "plans": plans_snap,
        }


def _coerce_state(s):
    """Return state as dict if possible, otherwise the raw string (or {} if empty)."""
    if isinstance(s, dict):
        return s
    if not s:
        return {}
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return s


def _apply_delta(state: dict, delta: StateDelta) -> dict:
    """Apply a single state delta to a state dict in-place and return it."""
    if delta.op == "set":
        state[delta.key] = delta.value
    elif delta.op == "delete":
        state.pop(delta.key, None)
    elif delta.op == "append":
        lst = state.get(delta.key, [])
        if not isinstance(lst, list):
            lst = [lst]
        lst.append(delta.value)
        state[delta.key] = lst
    return state


def _accumulate_metrics(base: InferenceMetrics, add: InferenceMetrics) -> InferenceMetrics:
    """Combine metrics from multiple LLM calls."""
    return InferenceMetrics(
        input_tokens=base.input_tokens + add.input_tokens,
        output_tokens=base.output_tokens + add.output_tokens,
        latency_ms=base.latency_ms + add.latency_ms,
        model=base.model or add.model,
    )
