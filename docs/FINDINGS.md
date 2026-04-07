# Evaluation Findings Log

Tracks experiment iterations, pass rate changes, root cause analyses, and fixes applied.

---

## Iteration 1 — Baseline

**Date:** 2026-04-06
**Eval files:** `outputs/data/zapier/20260406_051248/runs/test_cases_eval_20260406_06*.jsonl`
**Dataset:** 83 samples, steps-only (no modifications)
**Pass rate: ~59%**
**Distribution:** 30 samples at 0%, 42 at 100% (strong bimodal)

### Root Causes Identified

#### 1. PENDING over-use — chain stalls (highest impact)

Objects treated fire-and-forget peer sends (writing to Notion, posting to Slack) as request-reply interactions. The object would send a message to a write service, set a `_pending_reply_from` flag in its state, and stop — waiting for a reply that write services never send. The chain would stall permanently.

**Evidence:** 30 samples at 0% pass rate. Reasoning consistently read: "policy receives event, pending intent to write, but no write executed."

**Files changed:**
- `src/lnl/brain.py` — Rewrote `_PEER_INTERACTION_LOOP` constant with explicit fire-and-forget vs request-reply distinction. Fire-and-forget (notify/write/forward): send and finish immediately, never set PENDING. Request-reply (query): send question, set PENDING, wait for answer. Default is fire-and-forget.
- `config/prompts/lnl/object.yaml` — Updated "Act on This Message Only" section to clarify: PENDING applies only when waiting for a peer's answer; after fire-and-forget sends, finish immediately.

#### 2. max_tool_rounds too low

`config/lnl/system.yaml` set `max_tool_rounds: 5`. Complex object chains required: data lookup → validate → compute → write → notify = potentially 4+ tool calls in a single object invocation. Hitting the limit silently produced an empty finish with no signal — the object appeared to complete but produced no output.

**Evidence:** `inventory` sample wrote reorder quantity 17 instead of 100 — likely a tool-round abort mid-computation.

**Files changed:**
- `config/lnl/system.yaml` — `max_tool_rounds: 5 → 8`

#### 3. Expectation writer ignores numeric thresholds

`write_expectations.yaml` Step 3 instructed: "determine routing logic from the Object System, not the Reference Data." Correct for channel/recipient routing decisions. But numeric thresholds (e.g., `score < 15 → escalate`) also live in object behavior descriptions, and the expectation writer was not evaluating whether the event data actually satisfied the threshold. It was writing "escalate" in the expected action regardless of the actual score.

**Evidence:** `call-coach` — all 5+ S001 failures were scores 23–24/30. The escalation threshold is `< 15`. The system correctly skipped escalation; the expectation incorrectly said it should happen.

**Files changed:**
- `config/prompts/data-gen/write_expectations.yaml` — Added "Threshold evaluation" instruction to Step 3: must evaluate whether event data satisfies the condition; only include conditional output if threshold IS met.

#### 4. Mock tool data field mismatches

`_generate_mock_tool_data()` in `generate_samples.py` generated data using only the tool name and generic step context. The function `_infer_data_hint()` only handled knowledge bases, org directories, and policy stores — no patterns for HubSpot, Salesforce, Airtable, Zapier Tables, Zendesk (the most common platforms in the dataset). This caused status field mismatches: mock data would generate `status: "captured"` but the expectation (and object behavior) expected `status: "new"`. Objects stored whatever field names the mock data used, causing downstream mismatches.

**Files changed:**
- `src/data/generate_samples.py` — Expanded `_infer_data_hint()` with platform-specific patterns (HubSpot/CRM, Salesforce, Zendesk/ticketing, Airtable/Zapier Tables/spreadsheets, Slack message stores, generic data stores). Added guidance to the mock data generation prompt: "Use field names and value formats that match how the behavior description refers to the data — downstream objects will access fields by exact name."

#### 5. Steps sparsity — single entry-point for multi-trigger automations

54 of 83 samples had exactly 1 step. The upstream cause: `identify_objects.yaml` didn't distinguish automations that respond to a single event type from those that naturally respond to multiple independent trigger types (e.g., `ticket.created` vs `ticket.resolved` in Zendesk both triggering different behavior chains). The data gen LLM was generating a single entry-point object even when the automation should have had two separate entry-points.

**Impact:** Single-step samples could only be tested on the S001 baseline event. Multi-trigger automations were under-tested.

**Files changed:**
- `config/prompts/data-gen/identify_objects.yaml` — Updated Q1 in mandatory checklist: "One entry-point service object per DISTINCT event type. If the automation responds to multiple independent trigger types that each start a different workflow, create a separate entry-point object for each. Never assign multiple unrelated event types to a single entry-point object's `event_sources`."
- `config/prompts/data-gen/write_steps.yaml` — Replaced "Most automations have only 1–2 steps" sentence with: "Write one step per DISTINCT external trigger type."

#### 6. Step text missing data the downstream chain needs

Step inputs (the text of trigger events) were often incomplete. Objects further down the chain needed data (thread IDs for messaging platforms, row identifiers for database operations, condition-satisfying values) that wasn't present in the trigger. Objects would invent or hallucinate the missing data, leading to mismatches with expectations.

**Files changed:**
- `config/prompts/data-gen/write_steps.yaml` — Added carry-all-data rule: "The step `text` must include every piece of content that downstream objects will need. Include conversation/thread identifiers for messaging platforms. Satisfy all conditions required by the expected output."

---

## Iteration 2 — Post Round-1 Fixes

**Date:** 2026-04-06
**Eval file:** `outputs/data/zapier/20260406_115945/runs/test_cases_eval_20260406_125451.jsonl`
**Dataset:** 85 samples (new pipeline run), steps-only
**Pass rate: 63.7%** (+4.7pp vs Iteration 1)
**Distribution:** 45 at 100%, 24 at 0%, 14 partial (16.9%) — bimodal persists

### What Improved

- 0% samples dropped: 30 → 24 (PENDING fix had partial effect)
- 100% samples grew: 42 → 45
- `call-coach` false failures resolved (threshold evaluation fix)

### Root Causes Identified (Remaining)

Deep trace of 20 zero-pass samples revealed three distinct chain-stall sub-patterns:

#### Sub-Pattern A: Read service mistyped as write service → PENDING forever (2–3 cases)

Business logic objects send a query to what should be a knowledge base or data lookup service. That service was generated as a write service with "Do not reply to incoming messages." The querying object correctly sets PENDING — and waits forever for an answer that never comes.

**Traced examples:**
- `slack-changelog`: `changelog-policy` sends a "please check for a duplicate entry" query to `notion-changelog`. `notion-changelog` was generated as a write service. Object state: `{"pending": {"step": "await_duplicate_check"}}`. Chain stalls permanently. No Notion entry, no Slack notification posted.
- `automate-hr-support`: `hr-triage-policy` queries `faq-knowledge-base` for an answer to a parental leave question. `faq-knowledge-base` has "Do not reply." No FAQ answer ever delivered to Maya Chen.

**Root cause:** `identify_objects.yaml` did not clearly distinguish objects that hold data and must answer lookup queries (knowledge bases, Notion databases used as FAQ stores, policy document stores) from pure write-only services.

**Files changed:**
- `config/prompts/data-gen/identify_objects.yaml` — Added "Important" note under Read services: any service that holds persistent data AND must answer lookup queries MUST be a read service with a `_data` tool, NOT a write service. Explicitly states: "Do NOT mark a data-holding lookup service as a write service — doing so permanently stalls the chain."

#### Sub-Pattern B: Sequential "after X confirms" behavior chains (4–5 cases)

Behavior descriptions were generated with sequential confirmation language: "send to gmail-drafts → *after gmail confirms* → send to hubspot-tasks → *after hubspot confirms* → notify Slack." The architecture is fire-and-forget for write services. The object sends to the first peer (correctly, fire-and-forget), then stops — because the behavior told it to wait for a confirmation that never comes.

**Traced example:**
- `automate-sales-follow-up-emails`: `call-follow-up-composer` sends to `gmail-drafts`. Behavior says: "After the Gmail draft link is confirmed, send instructions to hubspot-tasks. After the HubSpot task link is confirmed, send a notification payload to slack-notifications." `gmail-drafts` never confirms. Chain terminates at `gmail-drafts`. No HubSpot task, no Slack notification.

**Root cause:** `identify_objects.yaml` did not prohibit "after X confirms" / "when X responds" language for write services in behavior descriptions.

**Files changed:**
- `config/prompts/data-gen/identify_objects.yaml` — Added Fan-out rule under the `behavior` field definition: behavior descriptions MUST NOT use sequential confirmation language for write services. If multiple peers must be notified, describe them as a simultaneous fan-out ("Send to X, Y, and Z"). If a downstream send genuinely needs data only a peer can provide (e.g., a generated document ID), model that peer as a read service, not a write service.

#### Sub-Pattern C: Multi-peer fan-out incomplete — only first peer notified (5+ cases)

Objects whose behavior requires notifying multiple peers simultaneously only produce a message to the first peer. Either the LLM stops after the first `outgoing_messages` entry, or the second send is blocked because the trigger text didn't carry a required identifier.

**Traced examples:**
- `expenses-tracker`: `expense-policy` should send to `finance-notifications` AND `expense-tracker-updater` simultaneously. Only `finance-notifications` is notified. Object state records: `"status_update_blocked_reason": "missing expense row identifier"` — the trigger text didn't include a row ID to match.
- `offline-conversion-tracking`: `platform-dispatcher` sends to four ad platforms but never sends the final summary to `sync-confirmation`. State shows `"finalized": false` — the completion condition is never triggered.

**Root cause (split):**
1. Step text missing required identifiers (row IDs, record keys) needed for the second peer send — addressed by carry-all-data rule but existing samples predate the fix.
2. Object prompt didn't explicitly require ALL peer messages to be enumerated in a single finish response.

**Files changed:**
- `config/prompts/lnl/object.yaml` — Added Fan-out bullet to Peer Communication section: "When your behavior requires notifying or writing to multiple peers simultaneously, include ALL of them in `outgoing_messages` in a single finish response. Never stop after the first peer message."

### Also Fixed This Iteration

#### Vacuous 100% for 0-event test cases

Test cases for scheduled/heartbeat triggers (e.g., `ai-generated-press-mentions`) have steps with `expect: null` per the write_steps rule "omit expect for scheduled triggers." Running `--steps-only` eval would run the step, produce 0 judged events, and record `pass_rate: 1.0` — making them appear as passing tests.

**Files changed:**
- `src/data/evaluate.py` — `pass_rate` is now `None` (not `1.0`) when there are no evaluable events; display shows `N/A`; excluded from mean pass rate computation.
- `src/data/evaluate_baseline.py` — Same fix.
- `src/data/schema.py` — `TestCaseResult.pass_rate: float → Optional[float]`

---

## Open Issue: Unnatural Identifiers in Mock Data

**Observed in:** `helpdesk-automation-template-slack-clickup` across multiple eval runs (Iterations 1 and 2 datasets)

Mock data generation produces artificial-looking identifiers: Slack user IDs like `U4821` or `U8821`, knowledge base article IDs like `KB-0092`, employee IDs like `EMP-4821`. These get embedded into step text and expectations.

**Example from eval:**
```
Expected: Direct message sent to user U4821 (priya.nair) with step-by-step
          resolution instructions (from KB-0092), AND a Slack notification
          posted to #support-queue assigning the ticket to Jordan Reyes.
Reason:   Evidence confirms a DM was sent to U4821/priya.nair with KB-0092
          step-by-step instructions, but there is no evidence of a Slack
          notification to #support-queue assigning the ticket to Jordan Reyes.
```

In this specific case the identifier format (`U4821` vs `priya.nair`) is NOT causing the failure (the judge correctly maps both to the same person). The failure is a separate chain-stall issue (Sub-Pattern C — second peer not notified).

**But the broader concern is:**
1. Identifiers like `U4821` look fake. Real Slack user IDs are 9 chars (`U01ABCDEF`). Short 4-digit IDs break immersion and could confuse LLM judges.
2. Inconsistency risk: if the mock tool data returns `U4821` but the step text references only `priya.nair`, a strict judge may not recognize them as the same entity.
3. IDs in expectations are over-specific: if the system sends the DM to `priya.nair` (correct) but the expectation anchors on `U4821`, a strict judge might fail a correct outcome.

**Fix applied:**
- `src/data/generate_samples.py` — Added identifier format guidance to `_generate_mock_tool_data` prompt: Slack user IDs must be 9 alphanumeric chars (e.g. `U01ABCDEF`, not `U4821`); ticket IDs should follow realistic conventions (`PROJ-1042`, not `PROJ-42`); prefer names over opaque IDs where possible.
- `config/prompts/data-gen/write_expectations.yaml` — Added "Use names as primary identifiers" rule: reference people by name, not system ID. Only include an ID if it is the sole identifier in the evidence or is itself the meaningful output. Over-specifying identifiers causes false failures when system and evidence use different forms.

**Priority:** Medium. Does not cause failures by itself in current evals, but adds noise and risks future false failures.

---

## Iteration 3 — Post Round-2 Fixes (Data Gen Regeneration)

**Date:** 2026-04-06
**Eval file:** `outputs/data/zapier/20260406_165557/runs/test_cases_eval_20260406_174841.jsonl`
**Dataset:** 84 samples (fresh pipeline run with all data gen fixes), steps-only
**Pass rate: 64.3%** (+0.6pp vs Iteration 2 — essentially flat)
**Distribution:** 45 at 100%, 23 at 0%, 11 partial, 6 null (no evaluable events)

### Net Result: Significant churn, no net progress

11 samples improved (0% → 100%), 10 samples regressed (100% → 0%), offsetting each other almost exactly. The data gen fixes helped simple workflows but destabilized some complex ones — new samples were generated with different mock data and object definitions, some of which are worse than before.

### What Improved (11 samples: 0% → 100%)

All are simpler single-action or form-submission workflows that benefited from the data gen fixes (better mock data field names, fan-out rules, read/write service distinction):

`expenses-tracker`, `form-jira`, `form-telegram`, `landing-page`, `lead-capture`, `employee-onboarding-manager`, `turn-granola-notes-into-tasks`, `form-google-contacts`, `automate-employment-verification-letters`, `team-meeting-notes-automation-fathom-ai-slack`, `connect-databricks-engagement-data-salesloft-signals`

### What Regressed (10 samples: 100% → 0%)

These are more complex workflows where the freshly generated mock data or object definitions introduced new issues:

| Sample | Root Cause |
|--------|-----------|
| `helpdesk-automation-template-slack-clickup` | Assignee stored as "Unassigned" instead of assigned IT specialist |
| `automated-incident-postmortem-reviews` | Document content unavailable at runtime, blocking LLM analysis |
| `form-zoho-crm` | Full name split incorrectly — "Sarah Mitchell" stored as Last Name instead of "Mitchell" |
| `round-robin-lead-assignment` | Stalled waiting for queue lookup (request-reply not completing) |
| `inventory` | Reorder quantity wrong (15 instead of 50) |
| `call-prep-guide` | Timeout at 180s |
| `engineering-work-intake-slack-jira` | Jira issue created but Slack thread reply and manager DM missing |
| `team-operations-portal` | Google Doc created but link delivery incomplete |
| `target-account-engagement-alert-rep-outreach-kit` | Validation error: missing email body in outreach sender |
| `canaries-employee-attrition-risk-prediction-mitigation` | Message ingested but not forwarded to scheduled batch processing |

**Pattern:** Regressions are mostly data quality issues in the freshly generated samples — wrong field values, wrong name splits, blocked document access — not systematic architectural failures. Each is a one-off sample-level defect.

### Priority Targets: All Unchanged at 0%

- **`slack-changelog-automation`** — Duplicate detection policy blocking writes (LLM found CHLOG-10041 as existing entry, discarded processing entirely instead of writing new entry and notifying)
- **`automate-hr-support-ai-helpdesk-assistant`** — FAQ retrieval marked as "outage," causing escalation instead of answer; no response posted back to Maya Chen
- **`automate-sales-follow-up-emails-gong-hubspot`** — Gmail draft and HubSpot task created but Slack notification missing and no links produced

### Diagnosis: Why the Data Gen Fixes Didn't Move the Needle on Priority Targets

The priority targets were failing due to chain-stall sub-patterns (A, B, C). The data gen fixes addressed the *prompt rules* that generate objects, but the **existing samples that were already at 0% were regenerated into similarly-structured (or differently-broken) objects**. The fundamental issue is that:

1. `slack-changelog` — the new `notion-changelog` object is still being queried with request-reply semantics that stall
2. `automate-hr-support` — FAQ knowledge base classification or runtime still broken
3. `automate-sales-follow-up-emails` — fan-out now sends to gmail-drafts but Slack notification still missing from the chain

### Key Learning

**Regenerating samples introduces variance in both directions.** For each fixed sample (simpler workflows), a complex one broke due to LLM randomness in data gen. Pass rate alone is not a stable metric when regenerating — sample-level churn obscures whether the fixes are working.

**Better evaluation strategy going forward:** Fix specific known-bad samples by re-running Stage 1 only for those samples (`--id <sample_id>`), preserving passing samples. Only regenerate the full dataset when the prompt changes are fundamental enough to justify the churn risk.

### Also Applied This Iteration (Runtime Changes — Not Measurable via Eval)

These changes improve the live runtime but cannot be measured in offline eval:

| File | Change |
|------|--------|
| `src/lnl/types.py` | `Message` gains `timestamp: datetime` field set at creation — single authoritative clock |
| `src/lnl/runtime.py` | Heartbeat content simplified to `"Heartbeat"`; timestamp comes from `message.timestamp` |
| `src/lnl/brain.py` | All messages prefixed `[system time: <UTC ISO>]` from `message.timestamp`; `_peer_interaction_loop()` is now a function injecting `pending_timeout_seconds` and `heartbeat_interval_seconds` from config |
| `src/lnl/object.py` | Accepts `pending_timeout_seconds` and `heartbeat_interval_seconds`, passes to prompt builder |
| `config/lnl/system.yaml` | `pending_max_retries` replaced with `pending_timeout_seconds: 90` (wall-clock, decoupled from heartbeat frequency) |
| `config/prompts/lnl/object.yaml` | PENDING recovery uses timestamp comparison, not retry count; `_pending.started_at` copied from message's `[system time: ...]` prefix |
| `src/data/evaluate.py` | `--tc ID[mod_type]` selector (e.g. `--tc TC001[temporal]`) for targeting specific variants without counting indices |

---

## Notes on Methodology

- **Steps-only eval** (`--steps-only`) is used for iteration speed: skips modifications, covers baseline behavior only. Gives clean signal on chain completeness without modification noise.
- **Bimodal distribution** is the key health indicator: high 0% count = systemic chain stall; high 100% count = healthy. Goal is to shift 0% samples into partial or full pass.
- **0-event TCs** (scheduled/heartbeat triggers with `expect: null`) are excluded from pass rate (show as `N/A`) to avoid inflating metrics.
- **Regenerating samples introduces churn.** For targeted fixes, prefer re-running Stage 1 only for specific failing samples (`--id`) rather than full regeneration. Full regeneration is only justified when prompt changes are fundamental.
- **Evidence of remaining chain stalls** post-fix: look for `_pending` block in object states, and message bus activity that stops before reaching write service objects.
- **Runtime changes** (heartbeat, PENDING recovery) cannot be measured via offline eval — they require live runtime with `heartbeat.enabled: true`.

- **Regenerating samples** is required to apply data-gen prompt fixes — runtime prompt fixes (`object.yaml`, `brain.py`) take effect immediately on the existing dataset.
