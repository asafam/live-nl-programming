from src.llm.openai_client import OpenAIChatLLM
from src.llm.base import system_message, user_message

def get_validator_llm():
    return OpenAIChatLLM(model="gpt-4o", temperature=0.0)

def llm_assert_state(state, prompt, error_message="State validation failed"):
    llm = get_validator_llm()
    messages = [system_message("You are a state validator. Answer only 'yes' or 'no'."), user_message(prompt)]
    response = llm.generate_text(messages)
    assert response.strip().lower() == "yes", f"{error_message}: {response}"