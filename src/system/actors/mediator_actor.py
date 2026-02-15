"""Mediator Actor - Rule-based orchestration actor for live NL programming system."""

from __future__ import annotations
from typing import Optional, Dict, List
from .base import Actor
from src.system.llm.base import AbstractLLM


class MediatorActor(Actor):
    """
    Mediator Actor functions as a rule engine that translates signals into requests.

    Unlike domain actors that hold data, Mediators hold LOGIC (rules).
    They watch other actors and orchestrate interactions based on configurable rules.

    Key characteristics:
    - Listens to signals from specified actors
    - Evaluates signals against rules
    - Sends requests to target actors when rules match
    - Handles compensating transactions on failures
    """

    def __init__(
        self,
        name: str,
        llm: AbstractLLM,
        purpose: str,
        initial_rules: Optional[List[Dict]] = None,
        listening_to: Optional[List[str]] = None,
        enable_self_reflection: bool = False,
    ) -> None:
        """
        Initialize a Mediator actor.

        Args:
            name: Actor name
            llm: Language model for processing
            purpose: Natural language description of mediator's purpose
            initial_rules: List of rule dictionaries with trigger_condition and action_instruction
            listening_to: List of actor names this mediator watches
            enable_self_reflection: Whether to enable self-reflection on state changes
        """
        # Build the mediator system prompt
        system_prompt = self._build_system_prompt()

        # Initialize state with rules and listening configuration
        initial_state = {
            "purpose": purpose,
            "listening_to": listening_to or [],
            "rules": initial_rules or [],
            "constraints": []
        }

        # Get reflection prompt from actor.yaml config
        import os
        import yaml
        base_config_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        actor_prompt_path = os.path.join(base_config_dir, 'config', 'prompts', 'actor.yaml')
        with open(actor_prompt_path, 'r') as f:
            actor_config = yaml.safe_load(f)
        reflection_prompt = actor_config.get('reflection_prompt', '')

        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            initial_state=initial_state,
            reflection_prompt=reflection_prompt,
            enable_self_reflection=enable_self_reflection,
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the mediator actor."""
        return """=== IDENTITY ===

You are a **Mediator Actor** in a live NL system.
Your Name: {actor_name}
Your Purpose: {purpose}

CRITICAL: You DO NOT hold domain state. You hold LOGIC state (Rules). Your job is to translate SIGNALS into REQUESTS.

=== STATE ===

Your current state:
{state_schema}

State structure:
{
  "listening_to": ["ActorName1", "ActorName2"],  // The actors you watch
  "rules": [
    {
      "trigger_condition": "Natural language description of the event to catch",
      "action_instruction": "Natural language description of what to command",
      "target_actor": "Name of the actor to send the request to"
    }
  ]
}

=== CONSTRAINTS ===

You must adhere to the following constraints and rules:
{constraints}

=== EVENT HANDLING PROTOCOLS ===

You function as a Rule Engine. You will receive messages from the system.

1. **INPUT: PROCESSING SIGNALS (EVENT messages)**
   - When you receive an EVENT message from a source actor:
   - **Evaluation:** Compare the message content against your `rules` list.
   - **Logic:**
     * Does the signal match any `trigger_condition` in your rules?
     * If YES: Generate a specific `REQUEST` for the target actor defined in that rule.
     * If NO: Acknowledge but take no action.
   - **Pattern Matching:** Use semantic understanding to match signals to trigger conditions.
     The signal doesn't need to be an exact match, but should convey the same meaning.

2. **OUTPUT: SENDING REQUESTS**
   - Do not perform the action yourself. You are just mediating.
   - Send a message of type `REQUEST` to the relevant target actor specified in the matched rule.
   - **Format:** Include context about which rule was triggered and what action to take.
   - Example: "Based on your rule, please perform the specified action with the provided data"

3. **HANDLING FAILURES (Compensating Transactions)**
   - If the Target Actor replies with a FAILURE (e.g., "Cannot apply update"):
   - Check if you have a compensating/rollback rule defined
   - If yes, trigger the compensation logic
   - Otherwise, report the failure to the User or Coordinator
   - Example: If TargetActor fails, send REQUEST to SourceActor: "Rollback the last change"

4. **RULE MANAGEMENT (REQUEST messages)**
   - When you receive a REQUEST to add/modify/remove rules:
   - Update your `rules` state accordingly
   - Confirm the change to the requester

=== RELATIONSHIPS ===

You are aware of these related actors:
{related_actors}

You may send messages to any actor, but you primarily send REQUEST messages to actors
specified in your rules when their trigger conditions are met.

=== OUTGOING MESSAGES ===

When sending messages to other actors, you MUST specify the 'message_type':

- "REQUEST": You want another actor to do something (most common for mediators)
- "RESPONSE": You are acknowledging/replying to an incoming REQUEST or EVENT

CRITICAL: When you receive an EVENT that matches a rule, send a REQUEST to the target actor.
Do NOT send EVENT messages - you are translating events into requests.

=== FORMAT ===

Respond with a JSON object containing:

- response: Natural language response to the user/message sender (string)
- state_updates: Array of state field updates, each with 'key' (string) and 'value_json' (JSON-encoded string)
- messages: Array of messages to send to other actors, each with 'to' (actor name), 'message' (content), and 'message_type' ("REQUEST", "EVENT", or "RESPONSE")
- purpose_updates: Array of new purposes to accumulate (strings)
- constraints: Array of new constraints or rules to remember (strings)
"""

    @property
    def system_prompt(self) -> str:
        """Build system prompt including base prompt with state substitution."""
        import json
        prompt = self.base_system_prompt

        # Get state components
        purpose = self.state.get('purpose', '')
        constraints = self.state.get('constraints', [])
        state_schema = json.dumps(self.state, indent=2)

        # Format constraints
        if constraints:
            constraints_text = "\n".join(f"- {c}" for c in constraints)
        else:
            constraints_text = "(none)"

        # Substitute placeholders
        prompt = prompt.replace("{actor_name}", self.name)
        prompt = prompt.replace("{purpose}", purpose)
        prompt = prompt.replace("{state_schema}", state_schema)
        prompt = prompt.replace("{constraints}", constraints_text)

        # Add related actors context
        known_ctx = self._format_known_actors()
        related_actors_text = known_ctx if known_ctx else "(none)"
        prompt = prompt.replace("{related_actors}", related_actors_text)

        return prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Allow direct setting of system prompt (for backward compatibility)."""
        self.base_system_prompt = value

    def add_rule(self, trigger_condition: str, action_instruction: str, target_actor: str) -> None:
        """
        Add a new rule to the mediator.

        Args:
            trigger_condition: Natural language description of when to trigger
            action_instruction: Natural language description of what action to take
            target_actor: Name of the actor to send the request to
        """
        rule = {
            "trigger_condition": trigger_condition,
            "action_instruction": action_instruction,
            "target_actor": target_actor
        }

        current_rules = self.state.get("rules", [])
        current_rules.append(rule)
        self.set_state_field("rules", current_rules)
        print(f"[Mediator] {self.name}: Added rule: {trigger_condition} -> {action_instruction} (to {target_actor})")

    def remove_rule(self, rule_index: int) -> None:
        """
        Remove a rule by its index.

        Args:
            rule_index: Index of the rule to remove
        """
        current_rules = self.state.get("rules", [])
        if 0 <= rule_index < len(current_rules):
            removed_rule = current_rules.pop(rule_index)
            self.set_state_field("rules", current_rules)
            print(f"[Mediator] {self.name}: Removed rule at index {rule_index}: {removed_rule}")
        else:
            print(f"[Mediator] {self.name}: Invalid rule index {rule_index}")

    def add_listening_target(self, actor_name: str) -> None:
        """
        Add an actor to the list of actors this mediator listens to.

        Args:
            actor_name: Name of the actor to listen to
        """
        listening_to = self.state.get("listening_to", [])
        if actor_name not in listening_to:
            listening_to.append(actor_name)
            self.set_state_field("listening_to", listening_to)
            print(f"[Mediator] {self.name}: Now listening to {actor_name}")

    def remove_listening_target(self, actor_name: str) -> None:
        """
        Remove an actor from the list of actors this mediator listens to.

        Args:
            actor_name: Name of the actor to stop listening to
        """
        listening_to = self.state.get("listening_to", [])
        if actor_name in listening_to:
            listening_to.remove(actor_name)
            self.set_state_field("listening_to", listening_to)
            print(f"[Mediator] {self.name}: Stopped listening to {actor_name}")
