import json
import time
from dotenv import load_dotenv

from src.actors.coordinator_actor import CoordinatorActor
from src.actors.mock_email_monitor import MockEmailMonitor
from src.llm.openai_client import OpenAIChatLLM
from src.message_bus import MessageBus
from tests.utils import llm_assert_state

def test_proactive_events():
    print("\n" + "="*80)
    print("TEST: Proactive Events - Constraint Memory and Enforcement")
    print("="*80)

    load_dotenv()

    bus = MessageBus()

    def llm_factory(name: str):
        return OpenAIChatLLM()

    coordinator = CoordinatorActor(llm=llm_factory("Coordinator"), llm_factory=llm_factory)
    bus.register(coordinator)

    # 1. Setup: Create ShoppingList with meat
    print("\n[STEP 1] Creating ShoppingList with meat")
    print("-" * 80)
    bus.send_request(from_actor="User", to_actor="Coordinator", message="Create a shopping list with 2 steaks")

    # 2. Create the mock email listener first (simulating the external email system)
    print("\n[STEP 2] Setting up email monitor")
    print("-" * 80)
    email_monitor = MockEmailMonitor(name="EmailMonitor")
    bus.register(email_monitor)
    print("✓ EmailMonitor created")

    # 3. User requests orchestration: emails from EmailMonitor should update ShoppingList
    print("\n[STEP 3] User requests orchestration: when emails arrive, update shopping list")
    print("-" * 80)
    bus.send_request(from_actor="User", to_actor="Coordinator",
                     message="When EmailMonitor receives emails from Sarah about shopping plans and meal preparation, update the ShoppingList accordingly")
    print("✓ Orchestration requested")

    # Check if a mediator was created
    mediator_names = [name for name in bus.actors if "mediator" in name.lower() or "email" in name.lower() and name != "EmailMonitor"]
    if mediator_names:
        print(f"✓ Mediator(s) created: {mediator_names}")
        for mediator_name in mediator_names:
            mediator = bus.actors[mediator_name]
            if hasattr(mediator, 'state') and 'listening_to' in mediator.state:
                print(f"  - {mediator_name} listening to: {mediator.state['listening_to']}")
                print(f"  - {mediator_name} has {len(mediator.state.get('rules', []))} rule(s)")

    # Validate setup
    shopping_actor_name = next((name for name in bus.actors if "shopping" in name.lower()), None)
    if shopping_actor_name:
        state = bus.actors[shopping_actor_name].state
        print(f"✓ ShoppingList created with state: {state}")
        # Handle both list of dicts and dict of items, and 'shopping_list' key
        items = state.get("items") or state.get("shopping_list") or []
        if isinstance(items, dict):
            has_steak = "steak" in items or "steaks" in items
        else:
            # Handle list of dicts or list of strings
            has_steak = False
            for item in items:
                if isinstance(item, dict):
                    # Check both 'name' and 'item' keys for flexibility
                    item_name = item.get("name", "") or item.get("item", "")
                    if "steak" in item_name.lower():
                        has_steak = True
                        break
                elif isinstance(item, str):
                    if "steak" in item.lower():
                        has_steak = True
                        break
        assert has_steak, "Steaks should be in the list"
        print(f"✓ Verified: Shopping list contains steaks")

    # 4. Simulate interactions to trigger email monitor
    print("\n[STEP 4] Simulating interactions to trigger email event")
    print("-" * 80)
    print("Adding items to trigger email monitor...")
    bus.send_request(from_actor="User", to_actor="Coordinator", message="Add 1 milk")
    email_monitor.listen() # 1
    print(f"  Interaction count: {email_monitor.state['interaction_count']}/{email_monitor.trigger_count}")

    bus.send_request(from_actor="User", to_actor="Coordinator", message="Add 1 bread")
    email_monitor.listen() # 2 -> Trigger!
    print(f"  Interaction count: {email_monitor.state['interaction_count']}/{email_monitor.trigger_count}")
    print("  → Email from Sarah received: She's going vegan!")

    # 5. Validate impact of the event
    print("\n[STEP 5] Validating event impact (constraint memory)")
    print("-" * 80)
    
    shopping_actor_name = next((name for name in bus.actors if "shopping" in name.lower()), None)
    if shopping_actor_name:
        state = bus.actors[shopping_actor_name].state
        print(f"ShoppingList state after email: {state}")

        # Check if constraints were added
        constraints = state.get("constraints", [])
        print(f"\nConstraints stored: {constraints}")

        # Let's verify if "vegan" or "no meat" is in constraints
        has_vegan_constraint = any("vegan" in c.lower() or "meat" in c.lower() for c in constraints)
        if not has_vegan_constraint:
             print("⚠ WARNING: No explicit vegan constraint found. Checking if steaks were removed at least.")
        else:
            print(f"✓ Vegan constraint detected and saved")

        # Check if steaks were removed
        items = state.get("items") or state.get("shopping_list") or []
        if isinstance(items, dict):
            has_steak = "steak" in items or "steaks" in items
        else:
            has_steak = False
            for item in items:
                if isinstance(item, dict):
                    # Check both 'name' and 'item' keys for flexibility
                    item_name = item.get("name", "") or item.get("item", "")
                    if "steak" in item_name.lower():
                        has_steak = True
                        break
                elif isinstance(item, str):
                    if "steak" in item.lower():
                        has_steak = True
                        break
        assert not has_steak, "Steaks should have been removed"
        print(f"✓ Steaks successfully removed from shopping list")
        print(f"  Current items: {items}")

    else:
        assert False, "Shopping list actor not found"

    # 6. Verify constraint memory
    print("\n[STEP 6] Verifying constraint memory (should reject chicken)")
    print("-" * 80)
    response = bus.send_request(from_actor="User", to_actor="Coordinator", message="Add 1 package of chicken wings for $10")
    print(f"Response from system: {response}")
    
    if shopping_actor_name:
        state = bus.actors[shopping_actor_name].state
        print(f"\nShoppingList state after chicken attempt: {state}")

        items = state.get("items") or state.get("shopping_list") or []
        has_chicken = False
        if isinstance(items, dict):
            has_chicken = "chicken" in str(items).lower()
        else:
            for item in items:
                if isinstance(item, dict):
                    # Check both 'name' and 'item' keys for flexibility
                    item_name = item.get("name", "") or item.get("item", "")
                    if "chicken" in item_name.lower():
                        has_chicken = True
                        break
                elif isinstance(item, str):
                    if "chicken" in item.lower():
                        has_chicken = True
                        break

        assert not has_chicken, "Chicken wings should NOT be added due to vegan constraint"
        print(f"✓ Chicken wings correctly rejected due to vegan constraint")
        print(f"  Final items: {items}")

    print("\n" + "="*80)
    print("✓ TEST PASSED: Proactive Events - Constraint Memory and Enforcement")
    print("="*80)

def test_irrelevant_event():
    """Test that irrelevant events (D. IRRELEVANCE) don't cause state changes"""
    print("\n" + "="*80)
    print("TEST: Irrelevant Event - No State Changes")
    print("="*80)

    load_dotenv()

    bus = MessageBus()

    def llm_factory(name: str):
        return OpenAIChatLLM()

    coordinator = CoordinatorActor(llm=llm_factory("Coordinator"), llm_factory=llm_factory)
    bus.register(coordinator)

    # 1. Setup: Create BudgetManager and ShoppingList
    print("\n[STEP 1] Creating BudgetManager with $50")
    print("-" * 80)
    bus.send_request(from_actor="User", to_actor="Coordinator", message="Create a budget, call it BudgetManager, and set it to $50")
    print("✓ BudgetManager created")

    print("\n[STEP 2] Creating ShoppingList")
    print("-" * 80)
    bus.send_request(from_actor="User", to_actor="Coordinator", message="Create a shopping list, call it ShoppingList, to track items with quantity, unit price, and name")
    print("✓ ShoppingList created")

    print("\n[STEP 3] Configuring shopping list to coordinate with budget")
    print("-" * 80)
    bus.send_request(from_actor="User", to_actor="Coordinator", message="Configure the ShoppingList to notify the BudgetManager whenever items are added or removed")
    print("✓ Coordination configured")

    print("\n[STEP 4] Adding items to shopping list")
    print("-" * 80)
    bus.send_request(from_actor="User", to_actor="Coordinator", message="Add 2 apples at $1 each")
    print("  ✓ Added 2 apples at $1 each")
    bus.send_request(from_actor="User", to_actor="Coordinator", message="Add 1 milk at $4")
    print("  ✓ Added 1 milk at $4")

    # Capture state before irrelevant event
    print("\n[STEP 5] Capturing state before irrelevant event")
    print("-" * 80)
    shopping_list_state_before = bus.actors["ShoppingList"].state.copy() if "ShoppingList" in bus.actors else None
    budget_manager_state_before = bus.actors["BudgetManager"].state.copy() if "BudgetManager" in bus.actors else None

    print(f"  ShoppingList state: {shopping_list_state_before}")
    print(f"  BudgetManager state: {budget_manager_state_before}")

    # Send irrelevant event
    print("\n[STEP 6] Sending irrelevant event")
    print("-" * 80)
    print("Sending: 'The weather is nice today'")
    response = bus.send_request(from_actor="User", to_actor="Coordinator", message="The weather is nice today")
    print(f"Response: {response}")

    # Validate no state changes occurred
    print("\n[STEP 7] Validating that irrelevant event caused no state changes")
    print("-" * 80)

    if "ShoppingList" in bus.actors:
        shopping_list_state_after = bus.actors["ShoppingList"].state
        print(f"  ShoppingList state after: {shopping_list_state_after}")

        # Compare items list (ignore purpose/constraints which are metadata)
        items_before = shopping_list_state_before.get("items", [])
        items_after = shopping_list_state_after.get("items", [])
        assert items_before == items_after, f"ShoppingList items should not change on irrelevant event. Before: {items_before}, After: {items_after}"
        print("  ✓ ShoppingList items unchanged")

    if "BudgetManager" in bus.actors:
        budget_manager_state_after = bus.actors["BudgetManager"].state
        print(f"  BudgetManager state after: {budget_manager_state_after}")

        budget_before = budget_manager_state_before.get("budget")
        budget_after = budget_manager_state_after.get("budget")
        assert budget_before == budget_after, f"BudgetManager budget should not change on irrelevant event. Before: {budget_before}, After: {budget_after}"
        print("  ✓ BudgetManager budget unchanged")

    print("\n" + "="*80)
    print("✓ TEST PASSED: Irrelevant Event - Actors correctly ignored irrelevant information")
    print("="*80)

def test_affirmative_non_conflicting_event():
    """Test that affirmative events that don't conflict with current state don't cause changes.

    This tests the scenario where we send a message like 'Ok to eat meat' when the shopping list
    already has meat items. Since this doesn't conflict with the current state, it should be
    treated as irrelevant and cause no changes (neither CONFLICT nor ADJUSTMENT).
    """
    print("\n" + "="*80)
    print("TEST: Affirmative Non-Conflicting Event - No State Changes")
    print("="*80)

    load_dotenv()

    bus = MessageBus()

    def llm_factory(name: str):
        return OpenAIChatLLM()

    coordinator = CoordinatorActor(llm=llm_factory("Coordinator"), llm_factory=llm_factory)
    bus.register(coordinator)

    # 1. Setup: Create ShoppingList with meat items
    print("\n[STEP 1] Creating ShoppingList with meat items")
    print("-" * 80)
    bus.send_request(from_actor="User", to_actor="Coordinator", message="Create a shopping list, call it ShoppingList, to track items with quantity, unit price, and name")
    print("✓ ShoppingList created")

    print("\n[STEP 2] Adding meat items to shopping list")
    print("-" * 80)
    bus.send_request(from_actor="User", to_actor="Coordinator", message="Add 2 steaks at $10 each")
    print("  ✓ Added 2 steaks at $10 each")
    bus.send_request(from_actor="User", to_actor="Coordinator", message="Add 1 chicken breast at $5")
    print("  ✓ Added 1 chicken breast at $5")

    # Capture state before affirmative event
    print("\n[STEP 3] Capturing state before affirmative event")
    print("-" * 80)
    shopping_list_state_before = bus.actors["ShoppingList"].state.copy() if "ShoppingList" in bus.actors else None

    print(f"  ShoppingList state: {shopping_list_state_before}")

    # Verify we have meat items
    items_field = "items" if "items" in shopping_list_state_before else "shopping_list"
    items = shopping_list_state_before.get(items_field, [])
    has_meat_before = False
    for item in items:
        if isinstance(item, dict):
            name = item.get("name", "").lower()
            if "steak" in name or "chicken" in name or "beef" in name or "pork" in name:
                has_meat_before = True
                break

    assert has_meat_before, "Should have meat items before the test"
    print(f"  ✓ Confirmed meat items: {items}")

    # Send affirmative non-conflicting event via email monitor
    print("\n[STEP 4] Sending affirmative non-conflicting event via email")
    print("-" * 80)
    email_monitor = MockEmailMonitor(
        name="EmailMonitor",
        email_message="New email from Sarah: 'Just confirming - it's totally fine to buy meat products.'",
        trigger_count=0  # Trigger immediately
    )
    bus.register(email_monitor)
    print("Email message: 'Just confirming - it's totally fine to buy meat products.'")
    email_monitor.listen()  # Trigger the email event
    print("  ✓ Email event triggered")

    # Validate no state changes occurred
    print("\n[STEP 5] Validating that affirmative event caused no state changes")
    print("-" * 80)

    if "ShoppingList" in bus.actors:
        shopping_list_state_after = bus.actors["ShoppingList"].state
        print(f"  ShoppingList state after: {shopping_list_state_after}")

        # Compare items list
        items_before = shopping_list_state_before.get(items_field, [])
        items_after = shopping_list_state_after.get(items_field, [])

        # Verify items are unchanged
        assert items_before == items_after, f"ShoppingList items should not change on affirmative non-conflicting event. Before: {items_before}, After: {items_after}"
        print(f"  ✓ ShoppingList items unchanged: {items_after}")

        # Verify no new constraints were added
        constraints_before = shopping_list_state_before.get("constraints", [])
        constraints_after = shopping_list_state_after.get("constraints", [])

        # Allow for minor differences in constraint wording, but shouldn't add meat restrictions
        meat_restriction_keywords = ["vegan", "vegetarian", "no meat", "avoid meat", "exclude meat"]
        has_meat_restriction = any(
            any(keyword in constraint.lower() for keyword in meat_restriction_keywords)
            for constraint in constraints_after
        )

        assert not has_meat_restriction, f"Should not add meat restrictions when affirming 'ok to eat meat'. Constraints: {constraints_after}"
        print(f"  ✓ No meat restrictions added")
        print(f"    Constraints: {constraints_after if constraints_after else '(none)'}")

    print("\n" + "="*80)
    print("✓ TEST PASSED: Affirmative Non-Conflicting Event - Actors correctly ignored redundant affirmation")
    print("="*80)

if __name__ == "__main__":
    test_proactive_events()
