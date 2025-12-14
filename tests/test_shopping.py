#!/usr/bin/env python3

import json
import os
from dotenv import load_dotenv

from src.actors.coordinator_actor import CoordinatorActor
from src.llm.openai_client import OpenAIChatLLM
from src.message_bus import MessageBus
from tests.utils import get_validator_llm, llm_assert_state

def test_shopping_list():
    load_dotenv()

    bus = MessageBus()

    def llm_factory(name: str):
        return OpenAIChatLLM(model="gpt-4o", temperature=0.0)

    coordinator = CoordinatorActor(llm=llm_factory("Coordinator"), llm_factory=llm_factory)
    bus.register(coordinator)

    # Test 1: Create BudgetManager with $50
    print("Test 1: Creating BudgetManager with $50")
    response1 = bus.send(from_actor="User", to_actor="Coordinator", message="Create a budget and set it to $50")
    print(f"Response: {response1}")

    # Validate BudgetManager created and budget set
    print("\nValidating BudgetManager creation and budget setup...")
    if "BudgetManager" in bus.actors:
        state = bus.actors["BudgetManager"].state
        prompt = f"Inspect this state for a BudgetManager actor: {json.dumps(state)}. Does it have a budget amount? Answer only 'yes' or 'no'."
        llm_assert_state(state, prompt, "BudgetManager setup failed")
    print("BudgetManager validation passed.")

    # Test 2: Create ShoppingList
    print("\nTest 2: Creating ShoppingList")
    response2 = bus.send(from_actor="User", to_actor="Coordinator", message="Create a shopping list to track items with quantity, unit price, and name")
    print(f"Response: {response2}")

    # Validate ShoppingList created
    print("\nValidating ShoppingList creation...")
    if "ShoppingList" in bus.actors:
        state = bus.actors["ShoppingList"].state
        prompt = f"Inspect this state for a ShoppingList actor: {json.dumps(state)}. Does it have a list of items? Answer only 'yes' or 'no'."
        llm_assert_state(state, prompt, "ShoppingList creation failed")
    print("ShoppingList validation passed.")

    # Test 2.5: Configure shopping list to coordinate with budget
    print("\nTest 2.5: Configuring shopping list to coordinate with budget")
    response2_5 = bus.send(from_actor="User", to_actor="Coordinator", message="Updated the budget automatically when items are added or removed")
    print(f"Response: {response2_5}")

    # Test 3: Add affordable items
    print("\nTest 3: Adding affordable items to shopping list")
    response3 = bus.send(from_actor="User", to_actor="Coordinator", message="Add 2 apples at $1 each")
    print(f"Response: {response3}")

    response4 = bus.send(from_actor="User", to_actor="Coordinator", message="Add 1 milk at $4")
    print(f"Response: {response4}")

    # Validate items added
    print("\nValidating items addition...")
    if "ShoppingList" in bus.actors:
        state = bus.actors["ShoppingList"].state
        prompt = f"Inspect this state for a ShoppingList actor: {json.dumps(state)}. Does it have apples and milk in the items list? Answer only 'yes' or 'no'."
        llm_assert_state(state, prompt, "Items addition failed")
    print("Items addition validation passed.")

    # Test 4: Query shopping list
    print("\nTest 4: Querying shopping list")
    response5 = bus.send(from_actor="User", to_actor="Coordinator", message="What is in my shopping list?")
    print(f"Response: {response5}")

    # Test 5: Query budget
    print("\nTest 5: Querying current budget")
    response6 = bus.send(from_actor="User", to_actor="Coordinator", message="Where do we stand with the budget?")
    print(f"Response: {response6}")

    # Validate budget updated
    print("\nValidating budget update...")
    if "BudgetManager" in bus.actors:
        state = bus.actors["BudgetManager"].state
        prompt = f"Inspect this state for a BudgetManager actor: {json.dumps(state)}. Does it have expenses reflecting the added items? Answer only 'yes' or 'no'."
        llm_assert_state(state, prompt, "Budget update failed")
    print("Budget update validation passed.")

    # Test 6: Try to add expensive item that exceeds budget
    print("\nTest 6: Trying to add expensive item")
    response7 = bus.send(from_actor="User", to_actor="Coordinator", message="Add 10 steaks at $5 each")
    print(f"Response: {response7}")

    # Test 7: Remove item
    print("\nTest 7: Removing an item")
    response8 = bus.send(from_actor="User", to_actor="Coordinator", message="Remove milk from the shopping list")
    print(f"Response: {response8}")

    # Validate item removed
    print("\nValidating item removal...")
    if "ShoppingList" in bus.actors:
        state = bus.actors["ShoppingList"].state
        prompt = f"Inspect this state for a ShoppingList actor: {json.dumps(state)}. Is milk removed from the items? Answer only 'yes' or 'no'."
        llm_assert_state(state, prompt, "Item removal failed")
    print("Item removal validation passed.")

    # Test 8: Query budget again
    print("\nTest 8: Querying budget after removal")
    response9 = bus.send(from_actor="User", to_actor="Coordinator", message="What is the current budget?")
    print(f"Response: {response9}")

    # Inspect states with independent LLM

    if "ShoppingList" in bus.actors:
        state = bus.actors["ShoppingList"].state
        print(f"\nShoppingList state: {state}")
        prompt = f"Inspect this state for a ShoppingList actor: {json.dumps(state)}. Does it have a list of items, where each item has name, quantity, and unit_price? Answer only 'yes' or 'no'."
        print("Final ShoppingList validation...")
        llm_assert_state(state, prompt, "ShoppingList validation failed")
        print("Final ShoppingList validation passed.")

    if "BudgetManager" in bus.actors:
        state = bus.actors["BudgetManager"].state
        print(f"BudgetManager state: {state}")
        prompt = f"Inspect this state for a BudgetManager actor: {json.dumps(state)}. Does it have a budget amount (could be 'budget' or 'total_budget') and a list of expenses? Answer only 'yes' or 'no'."
        print("Final BudgetManager validation...")
        llm_assert_state(state, prompt, "BudgetManager validation failed")
        print("Final BudgetManager validation passed.")

if __name__ == "__main__":
    test_shopping_list()

def test_actor_creation():
    load_dotenv()

    validator_llm = get_validator_llm()

    bus = MessageBus()

    def llm_factory(name: str):
        return OpenAIChatLLM(model="gpt-4o", temperature=0.0)

    coordinator = CoordinatorActor(llm=llm_factory("Coordinator"), llm_factory=llm_factory)
    bus.register(coordinator)

    # Test creating a budget actor
    print("Test: Creating a BudgetManager with $50")
    response = bus.send(from_actor="User", to_actor="Coordinator", message="Create a budget and set it to $50")
    print(f"Response: {response}")

    # Check if BudgetManager was created with correct state
    if "BudgetManager" in bus.actors:
        state = bus.actors["BudgetManager"].state
        print(f"BudgetManager state: {state}")
        assert "total_budget" in state
        assert state["total_budget"] == 50
        assert "expenses" in state
        assert isinstance(state["expenses"], list)
        print("BudgetManager created successfully with correct state")
    else:
        print("BudgetManager not created")
        assert False, "BudgetManager should have been created"

if __name__ == "__main__":
    test_actor_creation()