"""
Tests for the Planner module.
"""
import pytest
from unittest.mock import MagicMock
from ..core.planner import Planner
from ..utils.schema import ActionPlan

# A mock response from the LLM for planner
MOCK_PLAN_RESPONSE = '''{
    "query": "What is the capital of France and its population?",
    "reasoning": "First, I need to find the capital of France. Then, I need to find the population of that city. I will use web search for both steps.",
    "plan": [
        {
            "id": 1,
            "action": "find_capital",
            "tool": "web_search",
            "input": {"query": "capital of France"},
            "expected_output_schema": "A string containing the name of the city."
        },
        {
            "id": 2,
            "action": "find_population",
            "tool": "web_search",
            "input": {"query": "population of Paris"},
            "expected_output_schema": "A string containing the population number."
        }
    ]
}'''

@pytest.fixture
def mock_llm_client():
    """Fixture for a mock LLM client."""
    client = MagicMock()
    # Configure the mock to return a specific structure
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = MOCK_PLAN_RESPONSE
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_response
    return client

def test_generate_plan_successfully(mock_llm_client):
    """
    Tests if the planner can successfully generate a valid ActionPlan.
    """
    # TODO: Expand this test to cover more complex queries and edge cases.
    available_tools = ["web_search", "calculator"]
    planner = Planner(llm_client=mock_llm_client, available_tools=available_tools)
    query = "What is the capital of France and its population?"

    action_plan = planner.generate_plan(query)

    assert isinstance(action_plan, ActionPlan)
    assert action_plan.query == query
    assert len(action_plan.plan) == 2
    assert action_plan.plan[0].id == 1
    assert action_plan.plan[0].tool == "web_search"
    assert action_plan.plan[1].action == "find_population"

    # Verify that the LLM was called correctly
    mock_llm_client.chat.completions.create.assert_called_once()

def test_generate_plan_with_invalid_json(mock_llm_client):
    """
    Tests the planner's behavior when the LLM returns malformed JSON.
    """
    # TODO: Implement this test case.
    # It should check that a ValueError is raised or a fallback is triggered.
    mock_message = MagicMock()
    mock_message.content = '{"plan": [invalid json]'
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_llm_client.chat.completions.create.return_value.choices = [mock_choice]

    planner = Planner(llm_client=mock_llm_client, available_tools=["web_search"])
    with pytest.raises(ValueError, match="Planner failed to generate a valid JSON plan."):
        planner.generate_plan("test query")
