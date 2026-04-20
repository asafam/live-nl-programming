"""Tests for the single-active-plan mechanism.

Contract:
- One active plan per object at a time.
- LLM never authors plan or step ids; it references steps by position (index).
- Runtime handles all correlation (outgoing stamping, reply tagging,
  auto-mark on reply, auto-done for Tell dispatches).
- `plan_update` emits exactly one of three shapes per turn: create/replace
  (goal+steps), incremental (step_updates/add_steps), or close (status).
"""
import pytest

from src.lnl import (
    LLMObject,
    LLMResponse,
    Message,
    MessageType,
    MockBrain,
    ObjectDefinition,
    OutgoingMessage,
    PeerDeclaration,
)
from src.lnl.runtime import Runtime
from src.lnl.types import PlanUpdate, ReactFinish, ReactStep


def _defn(object_id="obj", **overrides):
    return ObjectDefinition(object_id=object_id, role="A test object.", **overrides)


def _user_msg(content, recipient="obj"):
    return Message(
        sender="__user__",
        recipient=recipient,
        type=MessageType.DOMAIN,
        content=content,
    )


class TestCreateAndClose:
    def test_create_sets_active_plan(self):
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Need multi-step.",
            action="finish",
            plan_update=PlanUpdate(
                goal="notify the team",
                steps=[
                    {"kind": "ask", "description": "look up manager", "target": "hr"},
                    {"kind": "tell", "description": "notify manager", "target": "notifier"},
                ],
            ),
            finish=ReactFinish(reply="Working on it"),
        ))
        obj = LLMObject(_defn(), brain)
        obj.process_message(_user_msg("go"))

        plan = obj.active_plan
        assert plan is not None
        assert plan.goal == "notify the team"
        assert plan.status == "active"
        assert len(plan.steps) == 2
        assert plan.steps[0].kind == "ask"
        assert plan.steps[0].target == "hr"
        assert plan.steps[0].status == "planned"

    def test_close_completes_active_plan(self):
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Create one-step plan.",
            action="finish",
            plan_update=PlanUpdate(
                goal="log action",
                steps=[{"kind": "tell", "description": "log it", "target": "log"}],
            ),
            finish=ReactFinish(reply=""),
        ))
        brain.script_react(ReactStep(
            thought="Done; close.",
            action="finish",
            plan_update=PlanUpdate(status="complete"),
            finish=ReactFinish(reply=""),
        ))
        obj = LLMObject(_defn(), brain)
        obj.process_message(_user_msg("go"))
        assert obj.active_plan is not None
        obj.process_message(_user_msg("close"))
        assert obj.active_plan is None
        assert len(obj.completed_plans) == 1
        assert obj.completed_plans[0].status == "complete"


class TestIncrementalUpdate:
    def test_step_update_by_index(self):
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Create.",
            action="finish",
            plan_update=PlanUpdate(
                goal="g",
                steps=[
                    {"kind": "ask", "description": "ask peer", "target": "peer"},
                ],
            ),
            finish=ReactFinish(reply=""),
        ))
        brain.script_react(ReactStep(
            thought="Mark step 0 failed.",
            action="finish",
            plan_update=PlanUpdate(
                step_updates=[{"index": 0, "status": "failed", "result_summary": "peer refused"}],
            ),
            finish=ReactFinish(reply=""),
        ))
        obj = LLMObject(_defn(), brain)
        obj.process_message(_user_msg("go"))
        obj.process_message(_user_msg("fail it"))

        plan = obj.active_plan
        assert plan is not None
        assert plan.steps[0].status == "failed"
        assert plan.steps[0].result_summary == "peer refused"

    def test_add_steps_extends_plan(self):
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Create.",
            action="finish",
            plan_update=PlanUpdate(
                goal="g",
                steps=[{"kind": "ask", "description": "first", "target": "p"}],
            ),
            finish=ReactFinish(reply=""),
        ))
        brain.script_react(ReactStep(
            thought="Add step.",
            action="finish",
            plan_update=PlanUpdate(
                add_steps=[{"kind": "tell", "description": "second", "target": "p"}],
            ),
            finish=ReactFinish(reply=""),
        ))
        obj = LLMObject(_defn(), brain)
        obj.process_message(_user_msg("go"))
        obj.process_message(_user_msg("extend"))

        plan = obj.active_plan
        assert len(plan.steps) == 2
        assert plan.steps[1].kind == "tell"


class TestOutgoingAutoCorrelation:
    def test_tell_auto_marks_done_on_dispatch(self):
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Create and dispatch.",
            action="finish",
            plan_update=PlanUpdate(
                goal="notify",
                steps=[{"kind": "tell", "description": "notify peer", "target": "peer-b"}],
            ),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(recipient="peer-b", content="fyi", expects_reply=False)],
            ),
        ))
        brain.set_default(LLMResponse(updated_state={}, reply=""))

        rt = Runtime(brain)
        a = rt.create_object(_defn("obj-a", peers=[PeerDeclaration("peer-b", "obs")]))
        rt.create_object(_defn("peer-b"))

        rt.send("obj-a", "trigger")

        plan = a.active_plan or (a.completed_plans[0] if a.completed_plans else None)
        assert plan is not None
        assert plan.steps[0].status == "done"

    def test_ask_flips_to_dispatched(self):
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Ask peer.",
            action="finish",
            plan_update=PlanUpdate(
                goal="query",
                steps=[{"kind": "ask", "description": "q", "target": "peer-b"}],
            ),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(recipient="peer-b", content="?", expects_reply=True)],
            ),
        ))
        brain.set_default(LLMResponse(updated_state={}, reply=""))

        rt = Runtime(brain)
        a = rt.create_object(_defn("obj-a", peers=[PeerDeclaration("peer-b", "resp")]))
        rt.create_object(_defn("peer-b"))

        rt.send("obj-a", "trigger")

        plan = a.active_plan
        assert plan is not None
        assert plan.steps[0].status == "dispatched"

    def test_outgoing_without_matching_step_passes_uncorrelated(self):
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Send without plan.",
            action="finish",
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(recipient="peer-b", content="hi")],
            ),
        ))
        brain.set_default(LLMResponse(updated_state={}, reply=""))

        rt = Runtime(brain)
        rt.create_object(_defn("obj-a", peers=[PeerDeclaration("peer-b", "x")]))
        rt.create_object(_defn("peer-b"))

        rt.send("obj-a", "trigger")

        delivered = [log.message for log in rt.message_log if log.message.recipient == "peer-b"]
        assert any(m.sender == "obj-a" for m in delivered)
        for m in delivered:
            assert m.plan_step_index is None


class TestReplyAutoMark:
    def test_reply_auto_marks_step_done(self):
        brain = MockBrain()
        # Turn 1: A creates plan + asks B.
        brain.script_react(ReactStep(
            thought="Ask B.",
            action="finish",
            plan_update=PlanUpdate(
                goal="get X",
                steps=[{"kind": "ask", "description": "ask B", "target": "b"}],
            ),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(recipient="b", content="What is X?", expects_reply=True)],
            ),
        ))
        # Turn 2: B answers.
        brain.script_react(ReactStep(
            thought="Answer.",
            action="finish",
            finish=ReactFinish(reply="42"),
        ))
        # Turn 3: A receives reply — runtime auto-marks step done before A runs.
        brain.script_react(ReactStep(
            thought="Got it. Close.",
            action="finish",
            plan_update=PlanUpdate(status="complete"),
            finish=ReactFinish(reply=""),
        ))

        rt = Runtime(brain)
        a = rt.create_object(_defn("obj-a", peers=[PeerDeclaration("b", "resp")]))
        rt.create_object(_defn("b"))
        rt.send("obj-a", "go")

        # Plan closed, step was auto-marked done.
        assert a.active_plan is None
        assert len(a.completed_plans) == 1
        plan = a.completed_plans[0]
        assert plan.steps[0].status == "done"


class TestNestedPlans:
    """A→B→C→B→A. B has its own plan mid-flow. Mid-plan B→A steps use
    B's ids internally; the final non-step reply from B to A uses A's
    correlation (runtime propagates from B's pending-inbound Ask)."""

    def test_nested_chain(self):
        brain = MockBrain()
        # Turn 1: A asks B.
        brain.script_react(ReactStep(
            thought="Ask B.",
            action="finish",
            plan_update=PlanUpdate(
                goal="get data",
                steps=[{"kind": "ask", "description": "ask b", "target": "b"}],
            ),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(recipient="b", content="data?", expects_reply=True)],
            ),
        ))
        # Turn 2: B receives A's ask — needs to ask C.
        brain.script_react(ReactStep(
            thought="Need C.",
            action="finish",
            plan_update=PlanUpdate(
                goal="fulfill A ask",
                steps=[{"kind": "ask", "description": "ask c", "target": "c"}],
            ),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(recipient="c", content="raw?", expects_reply=True)],
            ),
        ))
        # Turn 3: C answers B.
        brain.script_react(ReactStep(
            thought="Here.",
            action="finish",
            finish=ReactFinish(reply="XYZ"),
        ))
        # Turn 4: B receives C's reply; step 0 auto-done; closes its plan
        # and replies to A via an outgoing Tell (runtime treats as reply).
        brain.script_react(ReactStep(
            thought="Done.",
            action="finish",
            plan_update=PlanUpdate(status="complete"),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(recipient="obj-a", content="XYZ", expects_reply=False)],
            ),
        ))
        # Turn 5: A receives B's reply; closes plan.
        brain.script_react(ReactStep(
            thought="Got it.",
            action="finish",
            plan_update=PlanUpdate(status="complete"),
            finish=ReactFinish(reply=""),
        ))

        rt = Runtime(brain)
        a = rt.create_object(_defn("obj-a", peers=[PeerDeclaration("b", "r")]))
        b = rt.create_object(_defn("b", peers=[
            PeerDeclaration("c", "r"),
            PeerDeclaration("obj-a", "asker"),
        ]))
        rt.create_object(_defn("c"))

        rt.send("obj-a", "start")

        # A's plan closed cleanly.
        assert a.active_plan is None
        assert len(a.completed_plans) == 1
        assert a.completed_plans[0].steps[0].status == "done"

        # B's plan closed cleanly and is independent of A's.
        assert b.active_plan is None
        assert len(b.completed_plans) == 1
        b_plan = b.completed_plans[0]
        assert b_plan.goal == "fulfill A ask"
        assert b_plan.steps[0].status == "done"

        # The final B→A reply was routed as MessageType.REPLY (not DOMAIN).
        replies_to_a = [
            log.message for log in rt.message_log
            if log.message.type == MessageType.REPLY
            and log.message.sender == "b"
            and log.message.recipient == "obj-a"
        ]
        assert len(replies_to_a) == 1


class TestPromptRendering:
    def test_active_plan_rendered_without_ids(self):
        from src.lnl.brain import build_system_prompt

        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Create.",
            action="finish",
            plan_update=PlanUpdate(
                goal="notify team",
                steps=[{"kind": "ask", "description": "find email", "target": "hr"}],
            ),
            finish=ReactFinish(reply=""),
        ))
        obj = LLMObject(_defn(), brain)
        obj.process_message(_user_msg("go"))

        sys_prompt = build_system_prompt(
            obj.definition, obj.state, active_plan=obj.active_plan,
        )
        # The plan's semantic content is visible.
        assert "notify team" in sys_prompt
        assert "find email" in sys_prompt
        # Steps rendered by index, no runtime ids.
        assert "[0]" in sys_prompt

    def test_no_active_plan_renders_none(self):
        from src.lnl.brain import build_system_prompt
        sys_prompt = build_system_prompt(_defn(), current_state={}, active_plan=None)
        assert "(none)" in sys_prompt
        assert "Active Plan" in sys_prompt
