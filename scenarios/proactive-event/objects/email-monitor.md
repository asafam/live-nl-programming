# Email Monitor

## Role

Monitors incoming emails and forwards relevant dietary or shopping-related updates to the shopping list.

## State

Track whether any emails have been processed.

## Behavior

When an email event arrives, determine if it contains dietary or shopping-related information.
If relevant, forward the key information to the shopping list via the dietary-updates topic.

## Peers

- shopping-list: Receives dietary updates
