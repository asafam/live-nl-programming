"""Tests for MockService, MockRegistry, and MockInProcessExecutor."""
from src.lnl.mocks import MockRegistry, MockService


class TestMockService:
    def test_scripted_response(self):
        svc = MockService(name="email")
        svc.script_response("send", {"status": "sent"})

        result = svc.handle_call("send", {"to": "user@example.com"})

        assert result == {"status": "sent"}
        assert len(svc.recordings) == 1
        assert svc.recordings[0].method == "send"
        assert svc.recordings[0].args == {"to": "user@example.com"}

    def test_multiple_scripted_responses_consumed_in_order(self):
        svc = MockService(name="api")
        svc.script_response("get", {"page": 1})
        svc.script_response("get", {"page": 2})

        r1 = svc.handle_call("get")
        r2 = svc.handle_call("get")

        assert r1 == {"page": 1}
        assert r2 == {"page": 2}

    def test_fallback_returns_state(self):
        svc = MockService(name="db")
        svc.set_state("count", 42)

        result = svc.handle_call("query")

        assert result == {"count": 42}

    def test_state_operations(self):
        svc = MockService(name="store")
        svc.set_state("key", "value")
        assert svc.get_state("key") == "value"
        assert svc.get_state("missing", "default") == "default"

    def test_clear_recordings(self):
        svc = MockService(name="svc")
        svc.handle_call("method")
        assert len(svc.recordings) == 1
        svc.clear_recordings()
        assert len(svc.recordings) == 0


class TestMockRegistry:
    def test_add_and_get_service(self):
        reg = MockRegistry()
        svc = reg.add_service("email")
        assert reg.get_service("email") is svc
        assert reg.get_service("missing") is None

    def test_handle_call_routes_to_service(self):
        reg = MockRegistry()
        svc = reg.add_service("api")
        svc.script_response("get", {"data": 1})

        result = reg.handle_call("api", "get")
        assert result == {"data": 1}

    def test_unknown_service_raises(self):
        reg = MockRegistry()
        import pytest
        with pytest.raises(KeyError, match="Unknown service"):
            reg.handle_call("nonexistent", "method")

    def test_scheduled_events(self):
        reg = MockRegistry()
        reg.schedule_event(step=2, target="sensor", content="temperature=30")
        reg.schedule_event(step=2, target="alarm", content="check")
        reg.schedule_event(step=3, target="sensor", content="temperature=35")

        events_1 = reg.advance()  # step 1
        assert len(events_1) == 0

        events_2 = reg.advance()  # step 2
        assert len(events_2) == 2
        targets = {e.target for e in events_2}
        assert targets == {"sensor", "alarm"}

        events_3 = reg.advance()  # step 3
        assert len(events_3) == 1
        assert events_3[0].target == "sensor"

    def test_all_recordings(self):
        reg = MockRegistry()
        reg.add_service("a")
        reg.add_service("b")
        reg.handle_call("a", "method1")
        reg.handle_call("b", "method2")

        recs = reg.all_recordings()
        assert len(recs["a"]) == 1
        assert len(recs["b"]) == 1


# ── MockInProcessExecutor scripted_match_responses tests ──────────────────────

class TestMockInProcessExecutorMatchResponses:
    def _make_def(self, **kwargs):
        from src.data.schema import MockToolDef
        return MockToolDef(
            tool_name="slack.send_message",
            description="Send a Slack message.",
            arguments_schema={"type": "object", "properties": {"channel": {"type": "string"}}},
            response_template="fallback response",
            **kwargs,
        )

    def _make_call(self, id="t1", args=None):
        from src.lnl.types import ToolCall
        return ToolCall(id=id, tool="slack.send_message", arguments=args or {"channel": "general"})

    def test_arg_match_takes_priority_over_response_template(self):
        from src.data.schema import ScriptedMatchResponse
        from src.lnl.tools import MockInProcessExecutor
        executor = MockInProcessExecutor(self._make_def(
            scripted_match_responses=[
                ScriptedMatchResponse(match={"channel": "urgent"}, response="URGENT handled"),
            ],
        ))
        result = executor.execute(self._make_call(args={"channel": "urgent-alerts"}), {})
        assert result.output == "URGENT handled"

    def test_index_scripted_takes_priority_over_match(self):
        from src.data.schema import ScriptedMatchResponse
        from src.lnl.tools import MockInProcessExecutor
        executor = MockInProcessExecutor(self._make_def(
            scripted_responses=["scripted #1"],
            scripted_match_responses=[
                ScriptedMatchResponse(match={"channel": ".*"}, response="match response"),
            ],
        ))
        result = executor.execute(self._make_call(args={"channel": "general"}), {})
        assert result.output == "scripted #1"

    def test_falls_back_to_response_template_when_no_match(self):
        from src.data.schema import ScriptedMatchResponse
        from src.lnl.tools import MockInProcessExecutor
        executor = MockInProcessExecutor(self._make_def(
            scripted_match_responses=[
                ScriptedMatchResponse(match={"channel": "urgent"}, response="URGENT"),
            ],
        ))
        result = executor.execute(self._make_call(args={"channel": "general"}), {})
        assert result.output == "fallback response"

    def test_first_matching_entry_wins(self):
        from src.data.schema import ScriptedMatchResponse
        from src.lnl.tools import MockInProcessExecutor
        executor = MockInProcessExecutor(self._make_def(
            scripted_match_responses=[
                ScriptedMatchResponse(match={"channel": "deals"}, response="deals match"),
                ScriptedMatchResponse(match={"channel": ".*"}, response="catch-all"),
            ],
        ))
        result = executor.execute(self._make_call(args={"channel": "deals"}), {})
        assert result.output == "deals match"

    def test_match_response_interpolation(self):
        from src.data.schema import ScriptedMatchResponse
        from src.lnl.tools import MockInProcessExecutor
        executor = MockInProcessExecutor(self._make_def(
            scripted_match_responses=[
                ScriptedMatchResponse(match={"channel": ".*"}, response="Sent to #{channel}"),
            ],
        ))
        result = executor.execute(self._make_call(args={"channel": "deals"}), {})
        assert result.output == "Sent to #deals"


# ── MockInProcessExecutor trigger tests ──────────────────────────────────────

class TestMockInProcessExecutorTriggers:
    """Tests for the cross-object event injection mechanism on MockInProcessExecutor.

    When a mock tool is called, MockToolTrigger entries dispatch events to other
    LNL objects via inject_event in the tool context — simulating real-world
    callbacks like "email sent → Slack message arrives in a channel."
    """

    def _make_def(self, triggers=None, match=None, **kwargs):
        from src.data.schema import MockToolDef
        return MockToolDef(
            tool_name="email.send",
            description="Send an email.",
            arguments_schema={"type": "object", "properties": {
                "to":      {"type": "string"},
                "subject": {"type": "string"},
                "body":    {"type": "string"},
            }},
            response_template="email_id: {call_index}",
            triggers=triggers or [],
            match=match or {},
            **kwargs,
        )

    def _make_call(self, args=None):
        from src.lnl.types import ToolCall
        return ToolCall(
            id="c1",
            tool="email.send",
            arguments=args or {"to": "alice@company.com", "subject": "Hello"},
        )

    def test_trigger_fires_inject_event_with_correct_target_message_and_source(self):
        """When the tool is called, inject_event is called with the declared target, interpolated message, and source."""
        from src.data.schema import MockToolTrigger
        from src.lnl.tools import MockInProcessExecutor

        injected = []
        executor = MockInProcessExecutor(self._make_def(triggers=[
            MockToolTrigger(
                target_object_id="slack-monitor",
                message_template="Email sent to {to} — subject: {subject}",
                source="slack",
            ),
        ]))

        executor.execute(
            self._make_call(args={"to": "alice@company.com", "subject": "Q2 report"}),
            {"inject_event": lambda t, m, s: injected.append((t, m, s))},
        )

        assert len(injected) == 1
        target, message, source = injected[0]
        assert target == "slack-monitor"
        assert "alice@company.com" in message
        assert "Q2 report" in message
        assert source == "slack"

    def test_trigger_message_template_interpolates_all_arg_fields(self):
        """All {arg_name} placeholders in message_template are replaced with tool call argument values."""
        from src.data.schema import MockToolTrigger
        from src.lnl.tools import MockInProcessExecutor

        injected = []
        executor = MockInProcessExecutor(self._make_def(triggers=[
            MockToolTrigger(
                target_object_id="notification-handler",
                message_template="New message in #deal-alerts: {subject} from {to}, body: {body}",
                source="slack",
            ),
        ]))

        executor.execute(
            self._make_call(args={"to": "bob@company.com", "subject": "Deal closed", "body": "Acme Corp signed"}),
            {"inject_event": lambda t, m, s: injected.append((t, m, s))},
        )

        _, message, _ = injected[0]
        assert "bob@company.com" in message
        assert "Deal closed" in message
        assert "Acme Corp signed" in message
        assert "#deal-alerts" in message

    def test_trigger_message_template_includes_call_index(self):
        """{call_index} is available in message_template and increments per call."""
        from src.data.schema import MockToolTrigger
        from src.lnl.tools import MockInProcessExecutor

        injected = []

        def record(t, m, s):
            injected.append(m)

        executor = MockInProcessExecutor(self._make_def(triggers=[
            MockToolTrigger(
                target_object_id="audit-log",
                message_template="Email #{call_index} dispatched to {to}",
                source="audit",
            ),
        ]))

        executor.execute(self._make_call(args={"to": "alice@company.com", "subject": "First"}), {"inject_event": record})
        executor.execute(self._make_call(args={"to": "bob@company.com", "subject": "Second"}), {"inject_event": record})

        assert "Email #1 dispatched to alice@company.com" in injected[0]
        assert "Email #2 dispatched to bob@company.com" in injected[1]

    def test_trigger_fires_when_tool_level_match_passes(self):
        """Trigger fires when the tool-level match condition is satisfied."""
        from src.data.schema import MockToolTrigger
        from src.lnl.tools import MockInProcessExecutor

        injected = []
        executor = MockInProcessExecutor(self._make_def(
            match={"to": r"@company\.com$"},
            triggers=[MockToolTrigger(
                target_object_id="internal-notifier",
                message_template="Internal email sent to {to}",
                source="internal",
            )],
        ))

        executor.execute(
            self._make_call(args={"to": "alice@company.com", "subject": "Internal memo"}),
            {"inject_event": lambda t, m, s: injected.append((t, m, s))},
        )

        assert len(injected) == 1
        assert "alice@company.com" in injected[0][1]

    def test_trigger_suppressed_when_tool_level_match_fails(self):
        """Trigger does not fire when the tool-level match condition is not met."""
        from src.data.schema import MockToolTrigger
        from src.lnl.tools import MockInProcessExecutor

        injected = []
        executor = MockInProcessExecutor(self._make_def(
            match={"to": r"@company\.com$"},
            triggers=[MockToolTrigger(
                target_object_id="internal-notifier",
                message_template="Internal email sent to {to}",
                source="internal",
            )],
        ))

        # External address — match fails, trigger must not fire
        executor.execute(
            self._make_call(args={"to": "vendor@external.com", "subject": "Order confirm"}),
            {"inject_event": lambda t, m, s: injected.append((t, m, s))},
        )

        assert len(injected) == 0

    def test_multiple_triggers_all_fire_on_single_call(self):
        """All MockToolTrigger entries fire when a single tool call occurs."""
        from src.data.schema import MockToolTrigger
        from src.lnl.tools import MockInProcessExecutor

        injected = []
        executor = MockInProcessExecutor(self._make_def(triggers=[
            MockToolTrigger(target_object_id="slack-monitor",  message_template="Email to {to}", source="slack"),
            MockToolTrigger(target_object_id="crm-updater",    message_template="Contact {to} emailed", source="crm"),
            MockToolTrigger(target_object_id="audit-log",      message_template="Outbound: {to}", source="audit"),
        ]))

        executor.execute(self._make_call(), {"inject_event": lambda t, m, s: injected.append((t, m, s))})

        targets = [t for t, _, _ in injected]
        assert len(injected) == 3
        assert "slack-monitor" in targets
        assert "crm-updater" in targets
        assert "audit-log" in targets

    def test_trigger_graceful_when_inject_event_absent_from_context(self):
        """No exception is raised when inject_event is not provided in the tool context."""
        from src.data.schema import MockToolTrigger
        from src.lnl.tools import MockInProcessExecutor

        executor = MockInProcessExecutor(self._make_def(triggers=[
            MockToolTrigger(target_object_id="slack-monitor", message_template="Email to {to}", source="slack"),
        ]))

        # Must not raise; tool still returns a response
        result = executor.execute(self._make_call(), {})
        assert result.output is not None

    def test_trigger_dispatch_recorded_in_call_log(self):
        """Each trigger dispatch is captured in the executor's call_log for traceability."""
        from src.data.schema import MockToolTrigger
        from src.lnl.tools import MockInProcessExecutor

        executor = MockInProcessExecutor(self._make_def(triggers=[
            MockToolTrigger(target_object_id="slack-monitor", message_template="Email to {to}", source="slack"),
        ]))

        executor.execute(self._make_call(), {"inject_event": lambda *a: None})

        assert len(executor.call_log) == 1
        log = executor.call_log[0]
        assert "triggered" in log
        assert log["triggered"][0]["target"] == "slack-monitor"
        assert "alice@company.com" in log["triggered"][0]["message"]

    def test_no_trigger_log_entry_when_match_fails(self):
        """call_log has no 'triggered' key when the tool-level match suppresses the trigger."""
        from src.data.schema import MockToolTrigger
        from src.lnl.tools import MockInProcessExecutor

        executor = MockInProcessExecutor(self._make_def(
            match={"to": r"@company\.com$"},
            triggers=[MockToolTrigger(target_object_id="notifier", message_template="Hi {to}", source="slack")],
        ))

        executor.execute(
            self._make_call(args={"to": "external@other.com", "subject": "Hi"}),
            {"inject_event": lambda *a: None},
        )

        assert "triggered" not in executor.call_log[0]
