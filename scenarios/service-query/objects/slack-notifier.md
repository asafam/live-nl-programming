# Slack Notifier

## Role

Records all notification messages sent through it. Acts as a write-side service representing Slack.

## State

No messages sent yet.

## Behavior

When a message is received, record it in state with the sender and content. Keep a running log of all messages.
