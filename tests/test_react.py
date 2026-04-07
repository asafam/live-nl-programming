"""Tests for the ReAct loop's behavioral mechanics.

Specifically tests loop continuation (thought propagation, tool result injection,
and cross-object chaining) rather than input/output at the object boundary.

Difficulty levels:
  Level 1 (Easy)   — TestReActInnerThought: reasoning/thought propagation
  Level 2 (Medium) — TestReActToolResultPropagation: tool result injection into messages
  Level 3 (Hard)   — TestReActTwoObjectChain: cross-object chaining via Runtime
  Level 4 (Combined) — TestReActFullChain: think → tool → peer in one scenario
  Level 5 (Failure modes) — TestReActFailureModes: documents real experiment failure patterns
"""
import json

from src.lnl import (
    LLMObject,
    LLMResponse,
    Message,
    MessageType,
    MockBrain,
    ObjectDefinition,
    OutgoingMessage,
)
from src.lnl.brain import _parse_react_step
from src.lnl.runtime import Runtime
from src.lnl.tools import MockToolExecutor, ToolRegistry
from src.lnl.types import ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_defn(object_id: str = "agent", role: str = "Test agent.", **kwargs) -> ObjectDefinition:
    return ObjectDefinition(object_id=object_id, role=role, **kwargs)


def _domain_msg(content: str, recipient: str = "agent", sender: str = "__user__") -> Message:
    return Message(sender=sender, recipient=recipient, type=MessageType.DOMAIN, content=content)


def _tool_registry_with_mock(tool_name: str, output: str) -> tuple[ToolRegistry, MockToolExecutor]:
    executor = MockToolExecutor()
    executor.script(output)
    reg = ToolRegistry()
    reg.register(tool_name, executor)
    return reg, executor


# ---------------------------------------------------------------------------
# Level 1 (Easy) — Inner Thought
#
# The `reasoning` field of an LLMResponse becomes the `thought` field of the
# ReactStep. When the loop has a tool-call round, it serialises that thought
# as JSON into the assistant turn *before* calling the LLM again. This class
# tests that the thought is visible (or correctly absent) in brain.call_log.
# ---------------------------------------------------------------------------

class TestReActInnerThought:
    def test_thought_propagates_into_next_brain_call(self):
        """Reasoning from step 1 appears as 'thought' in the assistant JSON injected into step 2's messages."""
        brain = MockBrain()
        brain.script("agent", LLMResponse(
            updated_state={}, reply="",
            reasoning="I need to look up the account balance before answering",
            tool_calls=[ToolCall(id="t1", tool="lookup", arguments={"key": "balance"})],
        ))
        brain.script("agent", LLMResponse(
            updated_state={"checked": True}, reply="The balance is available.",
        ))

        reg, _ = _tool_registry_with_mock("lookup", "balance: 500")
        obj = LLMObject(_make_defn(), brain, tool_registry=reg)

        obj.process_message(_domain_msg("What is my balance?"))

        assert len(brain.call_log) == 2, "Expected exactly two brain calls (tool round + finish)"
        assistant_turns = [m for m in brain.call_log[1].messages if m["role"] == "assistant"]
        assert len(assistant_turns) >= 1, "No assistant turn found in second brain call"
        parsed = json.loads(assistant_turns[-1]["content"])
        assert parsed["thought"] == "I need to look up the account balance before answering"

    def test_no_tool_thought_not_in_call_messages(self):
        """When there is no tool call, the thought/reasoning is never injected into the messages list.

        A single-step finish breaks the loop immediately — no continuation call is made,
        so the reasoning string never gets appended as an assistant turn. This is the
        control case: thought as output-metadata vs. thought-in-context.
        """
        brain = MockBrain()
        brain.script("agent", LLMResponse(
            updated_state={"answered": True},
            reply="That is a simple question.",
            reasoning="Trivial question, answer directly without tools",
        ))
        obj = LLMObject(_make_defn(), brain)

        result = obj.process_message(_domain_msg("What is 2 + 2?"))

        assert len(brain.call_log) == 1, "No continuation call expected for a direct finish"
        all_contents = " ".join(m["content"] for m in brain.call_log[0].messages)
        assert "Trivial question" not in all_contents, (
            "Thought should not appear in messages when there is no continuation call"
        )
        assert result.reply == "That is a simple question."


# ---------------------------------------------------------------------------
# Level 2 (Medium) — Tool Result Propagation
#
# After a tool executes, its output is injected into the running messages list
# as a user-role turn prefixed with "[Tool result for {id}]:". The next brain
# call receives this augmented list, proving the loop passes *observations*
# forward — not a reset context.
# ---------------------------------------------------------------------------

class TestReActToolResultPropagation:
    def test_tool_result_injected_into_continuation_call(self):
        """The tool executor's output appears verbatim in the second brain call's messages."""
        brain = MockBrain()
        brain.script("agent", LLMResponse(
            updated_state={}, reply="",
            tool_calls=[ToolCall(id="inv-001", tool="inventory_check", arguments={"sku": "ABC"})],
        ))
        brain.script("agent", LLMResponse(
            updated_state={"sku": "ABC", "qty": 42}, reply="There are 42 units in stock.",
        ))

        reg, _ = _tool_registry_with_mock("inventory_check", "qty: 42 units")
        obj = LLMObject(_make_defn(), brain, tool_registry=reg)

        result = obj.process_message(_domain_msg("How many ABC do we have?"))

        second_msgs = brain.call_log[1].messages
        user_turns = [m for m in second_msgs if m["role"] == "user"]
        tool_result_turns = [m for m in user_turns if "[Tool result for inv-001]" in m["content"]]
        assert len(tool_result_turns) == 1, "Expected exactly one tool result turn"
        assert "qty: 42 units" in tool_result_turns[0]["content"]
        assert result.state_after == {"sku": "ABC", "qty": 42}

    def test_assistant_turn_contains_tool_call_arguments(self):
        """The assistant JSON appended before the tool result carries the full tool call spec."""
        brain = MockBrain()
        brain.script("agent", LLMResponse(
            updated_state={}, reply="",
            tool_calls=[ToolCall(id="t1", tool="search", arguments={"query": "climate", "limit": 5})],
        ))
        brain.script("agent", LLMResponse(updated_state={"found": True}, reply="Found results."))

        reg, _ = _tool_registry_with_mock("search", "3 results found")
        obj = LLMObject(_make_defn(), brain, tool_registry=reg)
        obj.process_message(_domain_msg("Search for climate articles."))

        assistant_turns = [m for m in brain.call_log[1].messages if m["role"] == "assistant"]
        assert len(assistant_turns) >= 1
        parsed = json.loads(assistant_turns[-1]["content"])
        assert parsed["action"] == "tool_call"
        assert parsed["tool_call"]["tool"] == "search"
        assert parsed["tool_call"]["id"] == "t1"
        assert parsed["tool_call"]["arguments"]["query"] == "climate"
        assert parsed["tool_call"]["arguments"]["limit"] == 5

    def test_message_count_grows_by_two_per_tool_round(self):
        """Each tool round appends exactly 2 messages: one assistant turn + one tool result.

        MockBrain stores a reference to the messages list (not a copy), so both
        call_log entries point to the same list after mutation. We therefore count
        roles directly: after 1 tool round the list must contain exactly one
        assistant turn and at least one tool-result user turn, on top of the initial
        system+user pair — 4 messages total.
        """
        brain = MockBrain()
        brain.script("agent", LLMResponse(
            updated_state={}, reply="",
            tool_calls=[ToolCall(id="t1", tool="lookup", arguments={})],
        ))
        brain.script("agent", LLMResponse(updated_state={}, reply="done"))

        reg, _ = _tool_registry_with_mock("lookup", "some output")
        obj = LLMObject(_make_defn(), brain, tool_registry=reg)
        obj.process_message(_domain_msg("go"))

        # Both call_log entries reference the same (mutated) list.
        # After 1 tool round: system + user + assistant_tool_turn + tool_result = 4.
        final_msgs = brain.call_log[1].messages
        roles = [m["role"] for m in final_msgs]
        assert roles.count("assistant") == 1, "Expected exactly one assistant turn (the tool-call step)"
        tool_result_turns = [
            m for m in final_msgs
            if m["role"] == "user" and "[Tool result for t1]" in m["content"]
        ]
        assert len(tool_result_turns) == 1, "Expected exactly one tool-result user turn"
        # system + original user + assistant tool turn + tool result = 4
        assert len(final_msgs) == 4, (
            f"Expected 4 messages after 1 tool round, got {len(final_msgs)}: {roles}"
        )


# ---------------------------------------------------------------------------
# Level 3 (Hard) — Two-Object Chain via Runtime
#
# Object A finishes with outgoing_messages targeting Object B. The Runtime
# routes those messages as DOMAIN messages to B, whose brain then processes
# them. This tests the full ReAct-to-cross-object pathway end-to-end.
# ---------------------------------------------------------------------------

class TestReActTwoObjectChain:
    def _setup_runtime(self) -> tuple[MockBrain, Runtime]:
        brain = MockBrain()
        brain.script("agent-a", LLMResponse(
            updated_state={"dispatched": True},
            reply="Sent to B.",
            outgoing_messages=[OutgoingMessage(recipient="agent-b", content="Please summarize Q4.")],
        ))
        brain.script("agent-b", LLMResponse(
            updated_state={"summary": "done"},
            reply="Summary complete.",
        ))
        rt = Runtime(brain)
        rt.create_object(ObjectDefinition(object_id="agent-a", role="Dispatcher"))
        rt.create_object(ObjectDefinition(object_id="agent-b", role="Summarizer"))
        return brain, rt

    def test_object_b_processes_message_from_a(self):
        """Both objects process their respective messages and update state correctly."""
        brain, rt = self._setup_runtime()
        results = rt.send("agent-a", "Start the workflow.")

        ids = {r.object_id for r in results}
        assert "agent-a" in ids, "agent-a result missing"
        assert "agent-b" in ids, "agent-b was not triggered by agent-a's outgoing message"

        a = next(r for r in results if r.object_id == "agent-a")
        b = next(r for r in results if r.object_id == "agent-b")
        assert a.state_after == {"dispatched": True}
        assert b.state_after == {"summary": "done"}

    def test_b_brain_receives_a_outgoing_content(self):
        """B's brain call receives A's outgoing message content verbatim as a user turn."""
        brain, rt = self._setup_runtime()
        rt.send("agent-a", "Start the workflow.")

        b_calls = [rec for rec in brain.call_log if rec.object_id == "agent-b"]
        assert len(b_calls) >= 1, "agent-b's brain was never called"

        user_turns = [m for m in b_calls[0].messages if m["role"] == "user"]
        assert any("Please summarize Q4." in m["content"] for m in user_turns), (
            "agent-a's outgoing content not found in agent-b's brain call messages"
        )

    def test_b_source_message_is_from_a(self):
        """B's ProcessingResult records that the triggering message came from agent-a via DOMAIN type."""
        brain, rt = self._setup_runtime()
        results = rt.send("agent-a", "Start the workflow.")

        # agent-b may appear multiple times if A's reply triggers further processing;
        # we want the result caused by the DOMAIN message A sent, not a REPLY.
        b_results = [r for r in results if r.object_id == "agent-b" and r.source_message_type == MessageType.DOMAIN]
        assert len(b_results) >= 1, "No DOMAIN-sourced result found for agent-b"

        b = b_results[0]
        assert b.in_reply_to == "agent-a", (
            f"Expected in_reply_to='agent-a', got '{b.in_reply_to}'"
        )


# ---------------------------------------------------------------------------
# Level 4 (Combined) — Full ReAct Chain: Think → Tool → Peer Object
#
# A single scenario that exercises all three levels in sequence:
#   1. Object A receives a message and has an inner thought (reasoning)
#   2. A calls a tool and observes the result
#   3. A finishes by sending a message to Object B with content informed by the tool
#   4. Object B processes A's message and updates its own state
# ---------------------------------------------------------------------------

class TestReActFullChain:
    def test_think_then_tool_then_peer(self):
        """Full ReAct chain: thought → tool call → peer object, all in one scenario.

        Object A:
          step 1 — reasons ("I should check stock levels") and calls a tool
          step 2 — observes the tool result and forwards findings to Object B
        Object B:
          step 1 — receives A's message and updates its state accordingly
        """
        brain = MockBrain()

        # A: step 1 — inner thought + tool call
        brain.script("agent-a", LLMResponse(
            updated_state={}, reply="",
            reasoning="I should check stock levels before notifying the buyer",
            tool_calls=[ToolCall(id="stock-01", tool="check_stock", arguments={"sku": "WIDGET-42"})],
        ))
        # A: step 2 — after observing tool result, finish and notify B
        brain.script("agent-a", LLMResponse(
            updated_state={"checked_sku": "WIDGET-42", "stock": 7},
            reply="Stock confirmed. Notifying buyer agent.",
            outgoing_messages=[OutgoingMessage(
                recipient="agent-b",
                content="Stock update: WIDGET-42 has 7 units remaining.",
            )],
        ))
        # B: processes A's notification
        brain.script("agent-b", LLMResponse(
            updated_state={"last_update": "WIDGET-42: 7 units"},
            reply="Acknowledged.",
        ))

        reg, executor = _tool_registry_with_mock("check_stock", "units_available: 7")
        # A may receive B's reply as a follow-up DOMAIN message; handle it with a no-op.
        brain.set_default(LLMResponse(updated_state={}, reply=""))

        rt = Runtime(brain, tool_registry=reg)
        rt.create_object(ObjectDefinition(object_id="agent-a", role="Inventory checker"))
        rt.create_object(ObjectDefinition(object_id="agent-b", role="Buyer agent"))

        results = rt.send("agent-a", "Check WIDGET-42 and notify the buyer.")

        # --- Level 1 check: A's inner thought appeared in the second brain call ---
        a_calls = [rec for rec in brain.call_log if rec.object_id == "agent-a"]
        # A makes 2 calls for its own processing (tool round + finish).
        # It may also receive B's reply, adding a 3rd call — that's fine.
        assert len(a_calls) >= 2, "agent-a should have made at least 2 brain calls (tool round + finish)"
        assistant_turns = [m for m in a_calls[1].messages if m["role"] == "assistant"]
        parsed_thought = json.loads(assistant_turns[-1]["content"])
        assert parsed_thought["thought"] == "I should check stock levels before notifying the buyer"

        # --- Level 2 check: tool result was injected into the second brain call ---
        user_turns = [m for m in a_calls[1].messages if m["role"] == "user"]
        tool_result_turns = [m for m in user_turns if "[Tool result for stock-01]" in m["content"]]
        assert len(tool_result_turns) == 1
        assert "units_available: 7" in tool_result_turns[0]["content"]
        assert len(executor.call_log) == 1
        assert executor.call_log[0].arguments == {"sku": "WIDGET-42"}

        # --- Level 3 check: B processed A's outgoing message ---
        ids = {r.object_id for r in results}
        assert "agent-b" in ids, "agent-b was not triggered by agent-a"

        a_result = next(r for r in results if r.object_id == "agent-a")
        b_result = next(
            r for r in results
            if r.object_id == "agent-b" and r.source_message_type == MessageType.DOMAIN
        )
        assert a_result.state_after == {"checked_sku": "WIDGET-42", "stock": 7}
        assert b_result.state_after == {"last_update": "WIDGET-42: 7 units"}
        assert b_result.in_reply_to == "agent-a"

        # B's brain received A's outgoing content verbatim
        b_calls = [rec for rec in brain.call_log if rec.object_id == "agent-b"]
        b_user_turns = [m for m in b_calls[0].messages if m["role"] == "user"]
        assert any("WIDGET-42 has 7 units remaining" in m["content"] for m in b_user_turns)


# ---------------------------------------------------------------------------
# Level 5 — Failure Modes
#
# These tests document the real failure patterns observed in experiments.
# They do NOT test happy-path mechanics — they test what the runtime does
# when things go wrong, and verify the exact (often silent) failure behaviour.
#
# Four failure classes:
#   A. Stuck tool loop     — LLM never finishes → max_tool_rounds → silent empty reply
#   B. Unknown tool        — LLM requests a tool that isn't registered → error in observation
#   C. Incoherent finish   — LLM ignores the tool result and writes wrong state anyway
#   D. Parser robustness   — Malformed LLM JSON: missing fields, null values, wrong action
# ---------------------------------------------------------------------------

class TestReActFailureModes:

    # ------------------------------------------------------------------
    # A. Stuck tool loop
    # ------------------------------------------------------------------

    def test_stuck_tool_loop_gives_silent_empty_reply(self):
        """When the LLM keeps requesting tools past max_tool_rounds, the loop
        manufactures an empty finish — no reply, state unchanged.

        This is a silent failure: the caller receives result.reply == "" with
        no indication that anything went wrong. In production this appears as
        an object that stops responding mid-task.
        """
        brain = MockBrain()
        for i in range(10):
            brain.script("agent", LLMResponse(
                updated_state={"progress": i},
                reply="",
                tool_calls=[ToolCall(id=f"t{i}", tool="lookup", arguments={"step": i})],
            ))

        executor = MockToolExecutor()
        for _ in range(10):
            executor.script("still no answer")
        reg = ToolRegistry()
        reg.register("lookup", executor)

        obj = LLMObject(_make_defn(), brain, tool_registry=reg, max_tool_rounds=3)
        obj.set_state({"original": True})
        result = obj.process_message(_domain_msg("find the answer"))

        # Silent failure: empty reply, state reverts to what it was before the message
        assert result.reply == "", "Stuck loop should produce empty reply, not a partial answer"
        assert result.state_after == {"original": True}, (
            "State should be unchanged when loop is cut short — "
            "the LLM never produced a real finish"
        )
        assert len(executor.call_log) == 3, "Should have stopped executing tools at max_tool_rounds"

    # ------------------------------------------------------------------
    # B. Unknown tool
    # ------------------------------------------------------------------

    def test_unknown_tool_error_is_injected_as_observation(self):
        """When the LLM requests a tool that isn't registered, the registry
        returns an error result. That error is injected into the messages as
        '[Tool result for {id}]: \nError: Unknown tool: ...'

        The LLM can see this error in its next call and potentially recover.
        If it doesn't, the error is invisible to the caller.
        """
        brain = MockBrain()
        brain.script("agent", LLMResponse(
            updated_state={}, reply="",
            tool_calls=[ToolCall(id="t1", tool="nonexistent_tool", arguments={"q": "data"})],
        ))
        brain.script("agent", LLMResponse(
            updated_state={"recovered": False}, reply="I could not find the tool.",
        ))

        # Registry exists but does NOT have "nonexistent_tool" registered
        reg = ToolRegistry()
        reg.register("other_tool", MockToolExecutor())

        obj = LLMObject(_make_defn(), brain, tool_registry=reg)
        result = obj.process_message(_domain_msg("use the tool"))

        # The error reaches the LLM as an observation
        second_msgs = brain.call_log[1].messages
        user_turns = [m for m in second_msgs if m["role"] == "user"]
        error_turns = [m for m in user_turns if "Unknown tool: nonexistent_tool" in m["content"]]
        assert len(error_turns) == 1, (
            "Unknown tool error should be injected as a user turn so the LLM can observe it"
        )
        assert "[Tool result for t1]" in error_turns[0]["content"]

        # The reply reflects whatever the LLM scripted — framework does not add its own error message
        assert result.reply == "I could not find the tool."

    # ------------------------------------------------------------------
    # C. Incoherent finish
    # ------------------------------------------------------------------

    def test_incoherent_finish_is_accepted_without_validation(self):
        """The framework does not validate that the LLM's finish is coherent
        with the tool result it observed. If the LLM ignores the observation
        and writes wrong state, that wrong state is committed.

        This documents a fundamental limitation: the framework provides the
        observation but cannot enforce that the LLM uses it.
        """
        brain = MockBrain()
        brain.script("agent", LLMResponse(
            updated_state={}, reply="",
            tool_calls=[ToolCall(id="t1", tool="price_lookup", arguments={"sku": "X1"})],
        ))
        # LLM ignores "price: 99.00" in the tool result and hallucinates a different value
        brain.script("agent", LLMResponse(
            updated_state={"sku": "X1", "price": 42.0},  # wrong — tool returned 99.00
            reply="The price is $42.",                    # hallucinated
        ))

        executor = MockToolExecutor()
        executor.script("price: 99.00")
        reg = ToolRegistry()
        reg.register("price_lookup", executor)

        obj = LLMObject(_make_defn(), brain, tool_registry=reg)
        result = obj.process_message(_domain_msg("what is the price of X1?"))

        # Framework commits the incoherent state without complaint
        assert result.state_after == {"sku": "X1", "price": 42.0}
        assert result.reply == "The price is $42."

        # The correct value WAS in the messages — the LLM just didn't use it
        second_msgs = brain.call_log[1].messages
        all_content = " ".join(m["content"] for m in second_msgs)
        assert "price: 99.00" in all_content, "Tool result with correct value was present in context"

    # ------------------------------------------------------------------
    # D. Parser robustness
    # ------------------------------------------------------------------

    def test_parse_missing_action_defaults_to_finish(self):
        """LLM omits the 'action' field entirely — parser defaults to 'finish'."""
        step = _parse_react_step({"thought": "I am done", "finish": {"reply": "here"}})
        assert step.action == "finish"
        assert step.finish.reply == "here"

    def test_parse_unknown_action_treated_as_finish(self):
        """LLM returns an unrecognised action string (e.g. 'complete', 'FINISH').
        The parser's if/else treats anything that isn't 'tool_call' as finish.
        """
        for bad_action in ("FINISH", "complete", "done", "respond", ""):
            step = _parse_react_step({
                "thought": "wrapping up",
                "action": bad_action,
                "finish": {"reply": "result", "updated_state": '{"k": 1}'},
            })
            assert step.action == "finish", f"action={bad_action!r} should fall through to finish"
            assert step.finish.reply == "result"

    def test_parse_null_tool_call_object_produces_empty_tool_call(self):
        """LLM returns action='tool_call' but tool_call is null.
        Parser substitutes an empty dict, producing a ToolCall with blank fields.
        The ToolRegistry will return 'Unknown tool: ' for the empty tool name.
        """
        step = _parse_react_step({"thought": "calling", "action": "tool_call", "tool_call": None})
        assert step.action == "tool_call"
        assert step.tool_call.tool == ""
        assert step.tool_call.arguments == {}

    def test_parse_missing_finish_block_produces_empty_reply(self):
        """LLM returns action='finish' but omits the finish block.
        Parser substitutes an empty dict → reply='', state='', no outgoing messages.
        Object gets an empty reply committed silently.
        """
        step = _parse_react_step({"thought": "done", "action": "finish"})
        assert step.action == "finish"
        assert step.finish.reply == ""
        assert step.finish.updated_state == ""
        assert step.finish.outgoing_messages == []

    def test_parse_null_outgoing_messages_does_not_crash(self):
        """LLM returns outgoing_messages: null instead of [] or omitting it.
        The 'or []' guard in the parser prevents a TypeError on iteration.
        """
        step = _parse_react_step({
            "action": "finish",
            "finish": {"reply": "ok", "outgoing_messages": None},
        })
        assert step.finish.outgoing_messages == []

    def test_parse_tool_call_missing_arguments_defaults_to_empty_dict(self):
        """LLM returns a tool_call without the 'arguments' key.
        Parser defaults to {} — the tool executor receives an empty arguments dict.
        """
        step = _parse_react_step({
            "action": "tool_call",
            "tool_call": {"id": "t1", "tool": "lookup"},
        })
        assert step.tool_call.arguments == {}
