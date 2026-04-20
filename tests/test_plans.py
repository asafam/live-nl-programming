"""Tests for plan-based execution tracking on LLM-objects.

Contract under test: the LLM reasons about plans *semantically* (by goal and
step description) and never authors plan or step ids. The runtime owns all
correlation — minting ids, stamping outgoing messages, propagating tags onto
replies, auto-marking steps done on correlated reply arrival.
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
from src.lnl.types import (
    PlanUpdate,
    ReactFinish,
    ReactStep,
)


def _defn(object_id="obj", **overrides):
    return ObjectDefinition(object_id=object_id, role="A test object.", **overrides)


def _user_msg(content, recipient="obj"):
    return Message(
        sender="__user__",
        recipient=recipient,
        type=MessageType.DOMAIN,
        content=content,
    )


class TestPlanCreate:
    def test_create_mints_runtime_plan_id(self):
        """LLM emits only semantic fields; runtime mints deterministic ids internally."""
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Need to coordinate with peer.",
            action="finish",
            plan_update=PlanUpdate(
                op="create",
                goal="Look up manager and notify",
                steps=[
                    {"kind": "ask", "description": "Ask HR for manager.", "target": "hr"},
                    {"kind": "tell", "description": "Notify the manager.", "target": "notifier"},
                ],
            ),
            finish=ReactFinish(reply="Plan created."),
        ))
        obj = LLMObject(_defn(), brain)

        obj.process_message(_user_msg("set up notification"))

        active = obj.active_plans()
        assert len(active) == 1
        plan = active[0]
        assert plan.id == "plan-obj-0"
        assert plan.goal == "Look up manager and notify"
        assert [s.id for s in plan.steps] == ["plan-obj-0-s0", "plan-obj-0-s1"]
        assert plan.steps[0].kind == "ask"
        assert plan.steps[0].target == "hr"
        assert plan.steps[0].status == "planned"

    def test_second_create_mints_unique_id(self):
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="First plan.",
            action="finish",
            plan_update=PlanUpdate(op="create", goal="A", steps=[
                {"kind": "ask", "description": "x", "target": "peer"},
            ]),
            finish=ReactFinish(reply="ok"),
        ))
        brain.script_react(ReactStep(
            thought="Second plan.",
            action="finish",
            plan_update=PlanUpdate(op="create", goal="B", steps=[
                {"kind": "tell", "description": "y", "target": "peer"},
            ]),
            finish=ReactFinish(reply="ok"),
        ))
        obj = LLMObject(_defn(), brain)

        obj.process_message(_user_msg("first"))
        obj.process_message(_user_msg("second"))

        assert [p.id for p in obj.active_plans()] == ["plan-obj-0", "plan-obj-1"]

    def test_create_drops_unsupported_step_kind(self):
        """Tool calls must NOT be plan steps — they stay in the ReAct loop."""
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Try to include a tool step (illegal).",
            action="finish",
            plan_update=PlanUpdate(op="create", goal="mixed", steps=[
                {"kind": "ask", "description": "valid", "target": "peer"},
                {"kind": "tool", "description": "illegal", "target": "calc"},
            ]),
            finish=ReactFinish(reply="ok"),
        ))
        obj = LLMObject(_defn(), brain)

        obj.process_message(_user_msg("go"))

        plan = obj.active_plans()[0]
        assert len(plan.steps) == 1
        assert plan.steps[0].kind == "ask"


class TestPlanStepDispatch:
    def test_outgoing_auto_correlates_to_planned_step(self):
        """LLM emits an outgoing with only recipient/content/expects_reply;
        runtime matches it to the matching planned step and stamps correlation."""
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Create plan and dispatch its only step.",
            action="finish",
            plan_update=PlanUpdate(op="create", goal="ask peer", steps=[
                {"kind": "ask", "description": "query peer", "target": "peer-b"},
            ]),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(
                    recipient="peer-b",
                    content="What is X?",
                    expects_reply=True,
                )],
            ),
        ))
        brain.set_default(LLMResponse(updated_state={}, reply=""))

        rt = Runtime(brain)
        a = rt.create_object(_defn("obj-a", peers=[PeerDeclaration("peer-b", "responder")]))
        rt.create_object(_defn("peer-b"))

        rt.send("obj-a", "trigger")

        plan = a.get_plan("plan-obj-a-0")
        assert plan is not None
        step = plan.steps[0]
        assert step.status == "dispatched"
        assert step.message_id  # runtime stamped the bus message id
        assert step.dispatched_at is not None

    def test_tell_step_auto_marks_done_on_dispatch(self):
        """Tell steps are fire-and-forget: runtime marks them `done` immediately."""
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Create plan with a tell and dispatch it.",
            action="finish",
            plan_update=PlanUpdate(op="create", goal="notify peer", steps=[
                {"kind": "tell", "description": "notify peer", "target": "peer-b"},
            ]),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(
                    recipient="peer-b",
                    content="FYI: X happened",
                    expects_reply=False,
                )],
            ),
        ))
        brain.set_default(LLMResponse(updated_state={}, reply=""))

        rt = Runtime(brain)
        a = rt.create_object(_defn("obj-a", peers=[PeerDeclaration("peer-b", "observer")]))
        rt.create_object(_defn("peer-b"))

        rt.send("obj-a", "trigger")

        plan = a.get_plan("plan-obj-a-0")
        assert plan.steps[0].status == "done"

    def test_outgoing_without_matching_plan_step_is_uncorrelated(self):
        """An outgoing that doesn't match any planned step goes out without plan tags."""
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Send an unrelated message (no plan).",
            action="finish",
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(
                    recipient="peer-b",
                    content="hi",
                )],
            ),
        ))
        brain.set_default(LLMResponse(updated_state={}, reply=""))

        rt = Runtime(brain)
        rt.create_object(_defn("obj-a", peers=[PeerDeclaration("peer-b", "responder")]))
        rt.create_object(_defn("peer-b"))

        rt.send("obj-a", "trigger")

        delivered = [log.message for log in rt.message_log if log.message.recipient == "peer-b"]
        assert any(m.sender == "obj-a" for m in delivered)
        for m in delivered:
            assert m.plan_id is None
            assert m.step_id is None


class TestReplyCorrelation:
    def test_reply_auto_marks_step_done_without_llm_intervention(self):
        """When a correlated reply arrives, the runtime auto-marks the step done.
        The LLM doesn't need to emit update_step for normal replies."""
        brain = MockBrain()

        # Step 1 — obj-a: create plan + dispatch the Ask.
        brain.script_react(ReactStep(
            thought="Create plan and ask peer-b.",
            action="finish",
            plan_update=PlanUpdate(op="create", goal="get X", steps=[
                {"kind": "ask", "description": "ask peer-b", "target": "peer-b"},
            ]),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(
                    recipient="peer-b",
                    content="What is X?",
                    expects_reply=True,
                )],
            ),
        ))
        # Step 2 — peer-b: answer the Ask.
        brain.script_react(ReactStep(
            thought="I know X.",
            action="finish",
            finish=ReactFinish(reply="X is 42"),
        ))
        # Step 3 — obj-a: receive the reply. LLM emits nothing plan-related;
        # runtime has already auto-marked the step done.
        brain.script_react(ReactStep(
            thought="Got the reply. Runtime has auto-marked the step done.",
            action="finish",
            finish=ReactFinish(reply=""),
        ))

        rt = Runtime(brain)
        a = rt.create_object(_defn("obj-a", peers=[PeerDeclaration("peer-b", "responder")]))
        rt.create_object(_defn("peer-b"))

        rt.send("obj-a", "trigger")

        # The reply back to obj-a carries the plan correlation tags the runtime
        # stamped at dispatch time — the LLM never touched them.
        reply_msgs = [
            log.message for log in rt.message_log
            if log.message.type == MessageType.REPLY and log.message.recipient == "obj-a"
        ]
        assert len(reply_msgs) == 1
        reply = reply_msgs[0]
        assert reply.plan_id == "plan-obj-a-0"
        assert reply.step_id == "plan-obj-a-0-s0"

        # The step is done without the LLM having to emit update_step.
        plan = a.get_plan("plan-obj-a-0")
        assert plan.steps[0].status == "done"

    def test_llm_can_override_auto_done_to_failed(self):
        """If a reply is actually a failure, the LLM's explicit update_step wins."""
        brain = MockBrain()

        brain.script_react(ReactStep(
            thought="Ask peer-b.",
            action="finish",
            plan_update=PlanUpdate(op="create", goal="get X", steps=[
                {"kind": "ask", "description": "ask peer-b", "target": "peer-b"},
            ]),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(
                    recipient="peer-b",
                    content="What is X?",
                    expects_reply=True,
                )],
            ),
        ))
        brain.script_react(ReactStep(
            thought="Reject the request.",
            action="finish",
            finish=ReactFinish(reply="I don't know"),
        ))
        # obj-a receives "I don't know" — treat as a failure. Emit update_step
        # with no refs; runtime infers the step from the incoming reply.
        brain.script_react(ReactStep(
            thought="Peer couldn't answer. Mark the step failed.",
            action="finish",
            plan_update=PlanUpdate(
                op="update_step",
                status="failed",
                result_summary="peer couldn't answer",
            ),
            finish=ReactFinish(reply=""),
        ))

        rt = Runtime(brain)
        a = rt.create_object(_defn("obj-a", peers=[PeerDeclaration("peer-b", "responder")]))
        rt.create_object(_defn("peer-b"))

        rt.send("obj-a", "trigger")

        plan = a.get_plan("plan-obj-a-0")
        assert plan.steps[0].status == "failed"
        assert plan.steps[0].result_summary == "peer couldn't answer"


class TestPlanLifecycle:
    def test_update_step_unresolvable_ref_is_no_op(self):
        """Off-message update_step with an unknown plan goal-ref is dropped."""
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Create plan.",
            action="finish",
            plan_update=PlanUpdate(op="create", goal="g", steps=[
                {"kind": "ask", "description": "s", "target": "x"},
            ]),
            finish=ReactFinish(reply=""),
        ))
        brain.script_react(ReactStep(
            thought="Try to update a step on a plan that doesn't exist.",
            action="finish",
            plan_update=PlanUpdate(
                op="update_step",
                plan="no such goal",
                step_index=0,
                status="done",
            ),
            finish=ReactFinish(reply=""),
        ))
        obj = LLMObject(_defn(), brain)
        obj.process_message(_user_msg("go"))
        obj.process_message(_user_msg("again"))

        plan = obj.get_plan("plan-obj-0")
        assert plan.steps[0].status == "planned"

    def test_complete_by_goal_ref_removes_plan_from_active(self):
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Create plan.",
            action="finish",
            plan_update=PlanUpdate(op="create", goal="g", steps=[
                {"kind": "tell", "description": "s", "target": "x"},
            ]),
            finish=ReactFinish(reply=""),
        ))
        brain.script_react(ReactStep(
            thought="Complete plan by goal ref.",
            action="finish",
            plan_update=PlanUpdate(op="complete", plan="g"),
            finish=ReactFinish(reply=""),
        ))
        obj = LLMObject(_defn(), brain)

        obj.process_message(_user_msg("go"))
        assert len(obj.active_plans()) == 1

        obj.process_message(_user_msg("close"))
        assert obj.active_plans() == []
        # Terminated plan kept in the archive for inspection
        assert obj.get_plan("plan-obj-0").status == "complete"

    def test_add_step_by_goal_ref_extends_existing_plan(self):
        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Create plan.",
            action="finish",
            plan_update=PlanUpdate(op="create", goal="g", steps=[
                {"kind": "ask", "description": "first", "target": "peer"},
            ]),
            finish=ReactFinish(reply=""),
        ))
        brain.script_react(ReactStep(
            thought="Add a second step by goal ref.",
            action="finish",
            plan_update=PlanUpdate(op="add_step", plan="g", steps=[
                {"kind": "tell", "description": "second", "target": "peer"},
            ]),
            finish=ReactFinish(reply=""),
        ))
        obj = LLMObject(_defn(), brain)
        obj.process_message(_user_msg("go"))
        obj.process_message(_user_msg("more"))

        plan = obj.get_plan("plan-obj-0")
        assert [s.id for s in plan.steps] == ["plan-obj-0-s0", "plan-obj-0-s1"]
        assert plan.steps[1].kind == "tell"


class TestPromptRendering:
    def test_active_plans_rendered_into_prompt_without_ids(self):
        """The LLM sees plans by goal + step descriptions — never raw ids."""
        from src.lnl.brain import build_system_prompt

        brain = MockBrain()
        brain.script_react(ReactStep(
            thought="Create plan.",
            action="finish",
            plan_update=PlanUpdate(op="create", goal="notify team", steps=[
                {"kind": "ask", "description": "find email", "target": "hr"},
            ]),
            finish=ReactFinish(reply=""),
        ))
        obj = LLMObject(_defn(), brain)

        obj.process_message(_user_msg("start"))

        sys_prompt = build_system_prompt(
            obj.definition,
            obj.state,
            current_plans=obj.active_plans(),
        )
        # Goal and step description ARE visible to the LLM
        assert "notify team" in sys_prompt
        assert "find email" in sys_prompt
        # Raw ids are NOT visible — the LLM references by goal.
        assert "plan-obj-0" not in sys_prompt
        assert "plan-obj-0-s0" not in sys_prompt

    def test_no_pending_block_in_prompt(self):
        """Regression: the prompt no longer mentions _pending as a state field."""
        from src.lnl.brain import build_system_prompt

        sys_prompt = build_system_prompt(_defn(), current_state={"status": "ok"})
        assert "_pending:" not in sys_prompt
        assert "Plans" in sys_prompt


class TestEndToEndSinglePlan:
    """End-to-end: object A owns a plan, asks peer B, receives B's reply,
    the runtime auto-marks the step done, and A closes the plan — all without
    the LLM touching any plan id."""

    def test_single_plan_full_lifecycle(self):
        brain = MockBrain()

        # Turn 1 (obj-a receives trigger): create plan + dispatch Ask to B.
        brain.script_react(ReactStep(
            thought="Need X from B. Create a plan and ask.",
            action="finish",
            plan_update=PlanUpdate(
                op="create",
                goal="get X from B",
                steps=[{"kind": "ask", "description": "ask B for X", "target": "b"}],
            ),
            finish=ReactFinish(
                reply="Working on it",
                outgoing_messages=[OutgoingMessage(
                    recipient="b", content="What is X?", expects_reply=True,
                )],
            ),
        ))
        # Turn 2 (obj-b receives the Ask): reply with X.
        brain.script_react(ReactStep(
            thought="Answer with X=42.",
            action="finish",
            finish=ReactFinish(reply="X is 42"),
        ))
        # Turn 3 (obj-a receives the reply): runtime has already marked the
        # step done. LLM closes the plan by goal reference.
        brain.script_react(ReactStep(
            thought="Got X. Close the plan.",
            action="finish",
            plan_update=PlanUpdate(op="complete", plan="get X from B"),
            finish=ReactFinish(reply="X is 42"),
        ))

        rt = Runtime(brain)
        a = rt.create_object(_defn("obj-a", peers=[PeerDeclaration("b", "responder")]))
        rt.create_object(_defn("b"))

        rt.send("obj-a", "Please find X")

        # --- Plan state: step auto-marked done, plan complete.
        plan = a.get_plan("plan-obj-a-0")
        assert plan is not None
        assert plan.status == "complete"
        assert plan.steps[0].status == "done"
        assert a.active_plans() == []

        # --- Message tags: runtime stamped the correlation on the Ask and
        # propagated it onto B's reply. The LLM never authored any id.
        ask_to_b = [
            log.message for log in rt.message_log
            if log.message.recipient == "b" and log.message.sender == "obj-a"
        ]
        assert len(ask_to_b) == 1
        assert ask_to_b[0].plan_id == "plan-obj-a-0"
        assert ask_to_b[0].step_id == "plan-obj-a-0-s0"

        reply_to_a = [
            log.message for log in rt.message_log
            if log.message.type == MessageType.REPLY and log.message.recipient == "obj-a"
        ]
        assert len(reply_to_a) == 1
        assert reply_to_a[0].plan_id == "plan-obj-a-0"
        assert reply_to_a[0].step_id == "plan-obj-a-0-s0"


class TestEndToEndNestedPlan:
    """End-to-end nested: A→B→C.

    A creates plan P_A to get data. A asks B.
    B receives A's Ask, creates its OWN plan P_B to fulfill it. B asks C.
    C replies to B. B's step auto-marks done under P_B. B replies to A.
    A receives B's reply — correlated to P_A via the runtime's tag propagation.
    A's P_A step auto-marks done. A closes P_A.

    Invariants verified:
    - Each object owns its own plan; ids never cross object boundaries.
    - A's plan correlation survives the chain (runtime handles propagation).
    - B's plan is independent of A's plan (different ids, different goals).
    - Neither LLM emits any plan or step id in its JSON.
    """

    def test_nested_plan_flow(self):
        brain = MockBrain()

        # Turn 1 — A: create plan P_A, ask B.
        brain.script_react(ReactStep(
            thought="Need data from B. Plan and ask.",
            action="finish",
            plan_update=PlanUpdate(
                op="create",
                goal="get data",
                steps=[{"kind": "ask", "description": "ask B for data", "target": "b"}],
            ),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(
                    recipient="b", content="Please fetch data", expects_reply=True,
                )],
            ),
        ))
        # Turn 2 — B: receives A's Ask. Need to ask C to fulfill. Create B's own plan.
        brain.script_react(ReactStep(
            thought="A wants data. I need to ask C. Create my own plan.",
            action="finish",
            plan_update=PlanUpdate(
                op="create",
                goal="fulfill data request",
                steps=[{"kind": "ask", "description": "ask C", "target": "c"}],
            ),
            finish=ReactFinish(
                reply="",  # B does NOT reply to A yet — it's still gathering.
                outgoing_messages=[OutgoingMessage(
                    recipient="c", content="Raw data please", expects_reply=True,
                )],
            ),
        ))
        # Turn 3 — C: replies with the raw data.
        brain.script_react(ReactStep(
            thought="Here's the data.",
            action="finish",
            finish=ReactFinish(reply="data=XYZ"),
        ))
        # Turn 4 — B: receives C's reply. B's step auto-marks done. Close B's
        # plan and answer A. B emits an outgoing_message to A; the runtime
        # auto-tags it with Plan A's ids (from B's pending-inbound record)
        # because A is in B's pending_inbound_asks AND no Plan B step targets A.
        brain.script_react(ReactStep(
            thought="Got C's answer. Close my plan and answer A.",
            action="finish",
            plan_update=PlanUpdate(op="complete", plan="fulfill data request"),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(
                    recipient="obj-a",
                    content="data=XYZ",
                    expects_reply=False,
                )],
            ),
        ))
        # Turn 5 — A: receives B's reply. Step auto-marked done. Close A's plan.
        brain.script_react(ReactStep(
            thought="Got the answer. Close my plan.",
            action="finish",
            plan_update=PlanUpdate(op="complete", plan="get data"),
            finish=ReactFinish(reply=""),
        ))

        rt = Runtime(brain)
        a = rt.create_object(_defn("obj-a", peers=[PeerDeclaration("b", "responder")]))
        b = rt.create_object(_defn("b", peers=[
            PeerDeclaration("c", "responder"),
            PeerDeclaration("obj-a", "requester"),  # B can message A (as reply)
        ]))
        rt.create_object(_defn("c"))

        rt.send("obj-a", "Please gather data")

        # --- A's plan: completed cleanly with step done.
        plan_a = a.get_plan("plan-obj-a-0")
        assert plan_a is not None
        assert plan_a.goal == "get data"
        assert plan_a.status == "complete"
        assert plan_a.steps[0].status == "done"

        # --- B's plan: independent of A's; completed cleanly.
        plan_b = b.get_plan("plan-b-0")
        assert plan_b is not None
        assert plan_b.goal == "fulfill data request"
        assert plan_b.status == "complete"
        assert plan_b.steps[0].status == "done"

        # --- Plan id isolation: A's id is never referenced by B's plan, and
        # vice-versa.
        assert plan_a.id != plan_b.id
        assert plan_a.id not in [s.id for s in plan_b.steps]
        assert plan_b.id not in [s.id for s in plan_a.steps]

        # --- Correlation chain on the wire:
        # (1) A→B Ask carries P_A's ids.
        ask_a_to_b = [
            log.message for log in rt.message_log
            if log.message.recipient == "b" and log.message.sender == "obj-a"
        ]
        assert len(ask_a_to_b) == 1
        assert ask_a_to_b[0].plan_id == "plan-obj-a-0"
        assert ask_a_to_b[0].step_id == "plan-obj-a-0-s0"

        # (2) B→C Ask carries P_B's ids — NOT A's.
        ask_b_to_c = [
            log.message for log in rt.message_log
            if log.message.recipient == "c" and log.message.sender == "b"
        ]
        assert len(ask_b_to_c) == 1
        assert ask_b_to_c[0].plan_id == "plan-b-0"
        assert ask_b_to_c[0].step_id == "plan-b-0-s0"

        # (3) C→B reply carries P_B's ids.
        reply_c_to_b = [
            log.message for log in rt.message_log
            if log.message.type == MessageType.REPLY
            and log.message.sender == "c" and log.message.recipient == "b"
        ]
        assert len(reply_c_to_b) == 1
        assert reply_c_to_b[0].plan_id == "plan-b-0"
        assert reply_c_to_b[0].step_id == "plan-b-0-s0"

        # (4) B→A reply carries P_A's ids — the original correlation survives
        # the hop through B. This is the core promise of the runtime-owned
        # correlation model: each object's plan is independent, but a reply
        # chain back to the asker stays tied to the asker's plan.
        reply_b_to_a = [
            log.message for log in rt.message_log
            if log.message.type == MessageType.REPLY
            and log.message.sender == "b" and log.message.recipient == "obj-a"
        ]
        assert len(reply_b_to_a) == 1
        assert reply_b_to_a[0].plan_id == "plan-obj-a-0"
        assert reply_b_to_a[0].step_id == "plan-obj-a-0-s0"

    def test_midplan_step_back_to_asker_uses_own_plan_ids(self):
        """When B's own Plan B has a step asking A for more info mid-plan,
        that Ask from B to A must carry Plan B's ids — NOT Plan A's.
        Only the final non-step reply to A uses Plan A's correlation."""
        brain = MockBrain()

        # Turn 1 — A: Plan A has one step: ask B.
        brain.script_react(ReactStep(
            thought="Ask B for X.",
            action="finish",
            plan_update=PlanUpdate(
                op="create",
                goal="need X",
                steps=[{"kind": "ask", "description": "ask B", "target": "b"}],
            ),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(
                    recipient="b", content="What is X?", expects_reply=True,
                )],
            ),
        ))
        # Turn 2 — B: needs clarification from A. Plan B has a step
        # explicitly declaring an Ask back to A. This step MUST use Plan B's
        # ids, not Plan A's (even though A is in B's pending_inbound_asks).
        brain.script_react(ReactStep(
            thought="Ambiguous request. Clarify with A.",
            action="finish",
            plan_update=PlanUpdate(
                op="create",
                goal="fulfill X request",
                steps=[{"kind": "ask", "description": "clarify with A", "target": "obj-a"}],
            ),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(
                    recipient="obj-a",
                    content="Which X do you mean?",
                    expects_reply=True,
                )],
            ),
        ))
        # Turn 3 — A: receives B's mid-plan Ask. Answers it. This is a fresh
        # Ask from B's perspective; A just replies. (A's own Plan A has a step
        # awaiting B's reply to A's original Ask — that's unchanged here.)
        brain.script_react(ReactStep(
            thought="Answer B's clarification.",
            action="finish",
            finish=ReactFinish(reply="X means the first one"),
        ))
        # Turn 4 — B: receives A's clarification. Plan B's step auto-marked
        # done. Close Plan B and reply to A's ORIGINAL Ask (not a plan step).
        brain.script_react(ReactStep(
            thought="Got clarification. Close my plan and answer A's original Ask.",
            action="finish",
            plan_update=PlanUpdate(op="complete", plan="fulfill X request"),
            finish=ReactFinish(
                reply="",
                outgoing_messages=[OutgoingMessage(
                    recipient="obj-a",
                    content="X is the first one",
                    expects_reply=False,
                )],
            ),
        ))
        # Turn 5 — A: receives B's final reply to the ORIGINAL Ask.
        brain.script_react(ReactStep(
            thought="Got final answer. Close Plan A.",
            action="finish",
            plan_update=PlanUpdate(op="complete", plan="need X"),
            finish=ReactFinish(reply=""),
        ))

        rt = Runtime(brain)
        rt.create_object(_defn("obj-a", peers=[PeerDeclaration("b", "responder")]))
        rt.create_object(_defn("b", peers=[PeerDeclaration("obj-a", "requester")]))

        rt.send("obj-a", "Please get X")

        # --- The mid-plan B→A Ask carries PLAN B's ids (NOT Plan A's).
        mid_plan_ask = [
            log.message for log in rt.message_log
            if log.message.sender == "b"
            and log.message.recipient == "obj-a"
            and log.message.expects_reply  # the clarification Ask, not the final Tell
        ]
        assert len(mid_plan_ask) == 1
        assert mid_plan_ask[0].plan_id == "plan-b-0", \
            "Mid-plan Ask back to A must use B's plan id, not A's"
        assert mid_plan_ask[0].step_id == "plan-b-0-s0"

        # --- The FINAL reply from B to A (answering A's original Ask) carries
        # PLAN A's ids.
        final_reply = [
            log.message for log in rt.message_log
            if log.message.type == MessageType.REPLY
            and log.message.sender == "b"
            and log.message.recipient == "obj-a"
        ]
        assert len(final_reply) == 1
        assert final_reply[0].plan_id == "plan-obj-a-0", \
            "Final reply to A must use A's plan id (the original Ask's correlation)"
        assert final_reply[0].step_id == "plan-obj-a-0-s0"
