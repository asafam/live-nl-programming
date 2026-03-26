/**
 * openclaw-mock-external
 *
 * OpenClaw plugin that registers mock tools for external systems used in the
 * LNL baseline evaluation. All tool calls are forwarded to a local MockServer
 * (src/data/mock_server.py) which responds with scripted or LLM-generated
 * responses and optionally injects callbacks back into the agent session.
 *
 * Install:
 *   openclaw plugin install ./plugins/openclaw-mock-external
 *
 * Configuration (environment variables):
 *   LNL_MOCK_SERVER_URL   — MockServer base URL (default: http://localhost:18888)
 */

import { definePluginEntry } from "openclaw-sdk/plugin";
import { Type } from "openclaw-sdk/schema";

const MOCK_SERVER_URL =
  process.env.LNL_MOCK_SERVER_URL ?? "http://localhost:18888";

// ── Tool definitions ──────────────────────────────────────────────────────────

interface MockTool {
  name: string;
  description: string;
  params: Record<string, unknown>;
}

const MOCK_TOOLS: MockTool[] = [
  {
    name: "slack_send_message",
    description: "Send a message to a Slack channel or user.",
    params: {
      channel: Type.String({ description: "Channel name (e.g. #deal-desk) or user ID" }),
      message: Type.String({ description: "Message text to send" }),
    },
  },
  {
    name: "slack_list_channels",
    description: "List available Slack channels.",
    params: {},
  },
  {
    name: "slack_add_reaction",
    description: "Add an emoji reaction to a Slack message.",
    params: {
      message_id: Type.String({ description: "Slack message ID" }),
      emoji: Type.String({ description: "Emoji name without colons (e.g. white_check_mark)" }),
    },
  },
  {
    name: "slack_get_user",
    description: "Get Slack user profile information.",
    params: {
      user: Type.String({ description: "Slack user ID or display name" }),
    },
  },
  {
    name: "email_send",
    description: "Send an email to one or more recipients.",
    params: {
      to: Type.String({ description: "Recipient email address or name" }),
      subject: Type.String({ description: "Email subject line" }),
      body: Type.String({ description: "Email body text" }),
    },
  },
  {
    name: "email_list_inbox",
    description: "List emails in an inbox folder.",
    params: {
      folder: Type.Optional(Type.String({ description: "Folder name (default: inbox)" })),
    },
  },
  {
    name: "email_read",
    description: "Read an email by its message ID.",
    params: {
      message_id: Type.String({ description: "Email message ID" }),
    },
  },
  {
    name: "jira_create_issue",
    description: "Create a new Jira issue.",
    params: {
      project: Type.String({ description: "Jira project key (e.g. PROJ)" }),
      summary: Type.String({ description: "Issue summary / title" }),
      description: Type.Optional(Type.String({ description: "Issue description" })),
    },
  },
  {
    name: "jira_update_issue",
    description: "Update the status of an existing Jira issue.",
    params: {
      issue_id: Type.String({ description: "Jira issue ID (e.g. PROJ-123)" }),
      status: Type.String({ description: "New status (e.g. In Progress, Done)" }),
    },
  },
  {
    name: "jira_get_issue",
    description: "Get details of a Jira issue.",
    params: {
      issue_id: Type.String({ description: "Jira issue ID" }),
    },
  },
  {
    name: "jira_list_issues",
    description: "List Jira issues matching a query.",
    params: {
      project: Type.Optional(Type.String({ description: "Filter by project key" })),
      status: Type.Optional(Type.String({ description: "Filter by status" })),
    },
  },
  {
    name: "webhook_post",
    description: "Send an HTTP POST to an external webhook URL.",
    params: {
      url: Type.String({ description: "Webhook destination URL" }),
      payload: Type.Optional(Type.String({ description: "JSON payload body" })),
    },
  },
];

// ── Forward a tool call to MockServer ─────────────────────────────────────────

async function forwardToMockServer(
  method: string,
  args: Record<string, unknown>,
  sessionKey: string,
): Promise<string> {
  const body = JSON.stringify({ ...args, __session_key__: sessionKey });

  let resp: Response;
  try {
    resp = await fetch(`${MOCK_SERVER_URL}/tool/${method}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
  } catch (err) {
    // MockServer not running — return a safe fallback so the agent can continue
    return `(mock unavailable) ${method} called with args: ${body}`;
  }

  if (!resp.ok) {
    return `(mock error ${resp.status}) ${method}`;
  }

  const data = (await resp.json()) as { status: string; result: string };
  return data.result ?? "(no result)";
}

// ── Plugin entry point ────────────────────────────────────────────────────────

export default definePluginEntry({
  id: "lnl-mock-external",
  name: "LNL Mock External Systems",

  register(api) {
    for (const tool of MOCK_TOOLS) {
      const { name, description, params } = tool;

      api.registerTool(
        {
          name,
          description,
          parameters: Type.Object(params as Record<string, unknown>),
          async execute(
            _id: string,
            args: Record<string, unknown>,
            context: { sessionKey: string },
          ): Promise<{ content: Array<{ type: string; text: string }> }> {
            const result = await forwardToMockServer(
              name,
              args,
              context?.sessionKey ?? "default",
            );
            return { content: [{ type: "text", text: result }] };
          },
        },
        { optional: true },
      );
    }

    // Log all tool calls in debug mode
    api.registerHook("before_tool_call", async (event: unknown) => {
      const e = event as { tool?: { name?: string }; args?: unknown };
      if (e?.tool?.name && MOCK_TOOLS.some((t) => t.name === e.tool!.name)) {
        // Tool is one of ours — no interception needed, just pass through
      }
      return { block: false };
    });
  },
});
