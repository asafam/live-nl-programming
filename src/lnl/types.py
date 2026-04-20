"""Core data types for the LNL runtime."""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MessageType(Enum):
    """Type of message exchanged between LLM-objects."""
    DOMAIN = "domain"
    ADMIN = "admin"
    EVENT = "event"
    REPLY = "reply"
    HEARTBEAT = "heartbeat"


@dataclass
class PeerDeclaration:
    """Declares a peer relationship for an LLM-object."""
    object_id: str
    relationship: str


@dataclass
class ObjectDefinition:
    """Complete definition of an LLM-object parsed from markdown."""
    object_id: str
    role: str
    behavior: str = ""
    peers: list[PeerDeclaration] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    subscriptions: list[str] = field(default_factory=list)
    event_sources: list[str] = field(default_factory=list)
    initial_state: str = ""  # optional ## State section from markdown


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


@dataclass
class Message:
    """A message passed between LLM-objects or from external senders."""
    sender: str
    recipient: str
    type: MessageType
    content: str
    topic: Optional[str] = None
    depth_remaining: int = 10  # hops remaining before chain is cut
    timestamp: datetime.datetime = field(default_factory=_utcnow)
    id: str = ""                         # runtime-assigned deterministic ID
    in_reply_to: Optional[str] = None    # ID of the message being replied to
    reference: Optional[str] = None      # LLM-assigned correlation tag (copied from OutgoingMessage)
    expects_reply: bool = False          # True = Ask (sender wants a reply); False = Tell (propagated from OutgoingMessage)
    plan_id: Optional[str] = None        # runtime-stamped: plan this message belongs to (if any)
    step_id: Optional[str] = None        # runtime-stamped: plan step this message fulfills (if any)


@dataclass
class OutgoingMessage:
    """An outgoing message produced by the LLM.

    `plan_id` / `step_id` / `in_reply_to` / `is_reply` are runtime-populated
    via auto-correlation — the LLM never authors them.
    """
    recipient: str
    content: str
    expects_reply: bool = False        # True = Ask (sender wants a reply); False = Tell (fire-and-forget)
    plan_id: Optional[str] = None      # runtime-stamped: plan this message dispatches a step of, OR the originating plan when this is a reply to a pending inbound Ask
    step_id: Optional[str] = None      # runtime-stamped: plan step this message fulfills (or the originating plan's step)
    in_reply_to: Optional[str] = None  # runtime-stamped: message id of the originating Ask (when this is a reply)
    is_reply: bool = False             # runtime-stamped: true when this outgoing fulfills a pending inbound Ask (routed as MessageType.REPLY)


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""
    id: str
    tool: str
    arguments: dict


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    id: str
    output: str
    error: str = ""


@dataclass
class ExternalAction:
    """A structured action directed at an external system (Slack, Email, Jira, etc.)."""
    system: str    # e.g. "slack", "email", "jira"
    action: str    # e.g. "send_message", "send", "create_issue"
    content: str   # NL content: message body, email text, ticket description, etc.
    params: dict = field(default_factory=dict)  # structured params: channel, to, subject, project, etc.


@dataclass
class LLMResponse:
    """Structured response returned by an LLM brain."""
    updated_state: str
    reply: str
    outgoing_messages: list[OutgoingMessage] = field(default_factory=list)
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class InferenceMetrics:
    """Metrics from a single LLM inference call."""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""


@dataclass
class ProcessingResult:
    """Result of processing a message by an LLM-object."""
    object_id: str
    reply: str
    outgoing_messages: list[OutgoingMessage] = field(default_factory=list)
    state_before: Any = None  # dict if JSON-parseable state, else str; None = {}
    state_after: Any = None   # dict if JSON-parseable state, else str; None = {}
    metrics: Optional[InferenceMetrics] = None
    in_reply_to: Optional[str] = None  # sender of the message that was processed
    source_message_type: Optional[MessageType] = None  # type of the message that was processed
    depth_remaining: int = 10  # propagated from the processed message
    sequence: int = 0          # assigned by Runtime for ordering concurrent results
    source_message_id: str = ""  # ID of the message that was processed
    source_plan_id: Optional[str] = None   # plan_id on the processed message (propagated to replies)
    source_step_id: Optional[str] = None   # step_id on the processed message (propagated to replies)


@dataclass
class StateDelta:
    """A single state change operation emitted by the LLM at any ReAct step."""
    op: str    # "set" | "delete" | "append"
    key: str
    value: Any = None  # required for set/append; ignored for delete


# Steps that legitimately need async correlation (cross-drain-cycle replies).
# Tool calls resolve synchronously within a single ReAct loop and don't belong here.
PLAN_STEP_KINDS = ("ask", "tell")

# Terminal plan/step statuses — the runtime prunes terminated plans from the
# snapshot rendered into the prompt but keeps them in the archive for debugging.
PLAN_TERMINAL_STATUSES = ("complete", "cancelled", "failed")
STEP_TERMINAL_STATUSES = ("done", "failed", "skipped")


@dataclass
class PlanStep:
    """A single step in a plan — typically one peer message with its correlation."""
    id: str                              # runtime-minted: "{plan_id}-s{n}"
    kind: str                            # "ask" | "tell"
    description: str                     # NL description of what this step accomplishes
    status: str = "planned"              # "planned" | "dispatched" | "done" | "failed" | "skipped"
    target: Optional[str] = None         # peer id
    message_id: Optional[str] = None     # bus message id once dispatched
    dispatched_at: Optional[datetime.datetime] = None
    result_summary: Optional[str] = None  # short NL note once the step resolves


@dataclass
class Plan:
    """A multi-step execution plan owned by an LLM-object.

    Plans encapsulate in-flight coordination with peers — what the object is
    mid-way through doing. Domain state (object.state) is kept clean of
    transient bookkeeping.
    """
    id: str                              # runtime-minted: "plan-{object_id}-{n}"
    goal: str                            # NL description of the overall goal
    trigger_message_id: str              # id of the incoming message that spawned this plan
    steps: list[PlanStep] = field(default_factory=list)
    status: str = "active"               # "active" | "complete" | "cancelled" | "failed"
    created_at: datetime.datetime = field(default_factory=_utcnow)
    updated_at: datetime.datetime = field(default_factory=_utcnow)


@dataclass
class PlanUpdate:
    """A single plan change emitted by the LLM at any ReAct step.

    Analogous to StateDelta: one op per step, accumulated and applied by the
    runtime after the ReAct loop finishes. The LLM never authors plan or step
    ids — references to existing plans use `plan` (goal-string match against
    active plans) and, for off-message `update_step`, `step_index`.
    """
    op: str                              # "create" | "add_step" | "update_step" | "complete" | "cancel"
    plan: Optional[str] = None           # goal-string ref to an active plan (for add_step/update_step/complete/cancel)
    step_index: Optional[int] = None     # 0-based step index for off-message update_step
    goal: Optional[str] = None           # for "create"
    steps: Optional[list[dict]] = None   # for "create" / "add_step": [{kind, description, target}]
    status: Optional[str] = None         # for "update_step": "done" | "failed" | "skipped"
    result_summary: Optional[str] = None  # for "update_step": short NL note


@dataclass
class ReactFinish:
    """The finish action in a ReAct step — commits reply and outgoing messages."""
    reply: str
    updated_state: str = ""  # legacy compat for MockBrain/tests; not in LLM schema
    outgoing_messages: list[OutgoingMessage] = field(default_factory=list)
    updated_definition: Optional[dict] = None  # set when an ADMIN message triggers a definition change


@dataclass
class ReactStep:
    """One step in a ReAct loop: an explicit thought and a single action."""
    thought: str
    action: str  # "tool_call" | "finish"
    state_update: Optional[StateDelta] = None  # optional at any step; accumulated by runtime
    plan_update: Optional[PlanUpdate] = None   # optional at any step; accumulated by runtime
    tool_call: Optional[ToolCall] = None
    finish: Optional[ReactFinish] = None


@dataclass
class MessageLog:
    """Log entry for a message delivered through the bus."""
    message: Message
    delivered: bool = True
    error: Optional[str] = None
    metrics: Optional[InferenceMetrics] = None
