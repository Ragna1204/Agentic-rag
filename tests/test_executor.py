"""
Tests for the Executor module.
"""
import pytest
from unittest.mock import MagicMock
from ..core.executor import Executor
from ..utils.schema import ActionPlan, ActionStep

@pytest.fixture
def mock_tools():
    """Fixture for mock tools."""
    mock_search_tool = MagicMock()
    mock_search_tool.execute.return_value = "Paris"

    mock_calc_tool = MagicMock()
    mock_calc_tool.execute.return_value = "25"

    return {
        "web_search": mock_search_tool,
        "calculator": mock_calc_tool
    }

def test_execute_plan_successfully(mock_tools):
    """
    Tests if the executor can run a valid plan step-by-step.
    """
    # TODO: Add tests for plans with dependencies between steps.
    executor = Executor(tools=mock_tools)
    plan = ActionPlan(
        query="Test query",
        reasoning="Test reasoning",
        plan=[
            ActionStep(id=1, action="search", tool="web_search", input={"query": "capital of France"}, expected_output_schema="string"),
            ActionStep(id=2, action="calculate", tool="calculator", input={"expression": "5*5"}, expected_output_schema="string")
        ]
    )

    results = executor.execute_plan(plan)

    assert 1 in results
    assert 2 in results
    assert results[1] == "Paris"
    assert results[2] == "25"

    # Check that the correct tools were called with the correct inputs
    mock_tools["web_search"].execute.assert_called_once_with(query="capital of France")
    mock_tools["calculator"].execute.assert_called_once_with(expression="5*5")

def test_execute_plan_with_missing_tool(mock_tools):
    """
    Tests the executor's behavior when a tool specified in the plan is not available.
    """
    # TODO: Implement this test case.
    executor = Executor(tools=mock_tools)
    plan = ActionPlan(
        query="Test query",
        reasoning="Test reasoning",
        plan=[
            ActionStep(id=1, action="unknown_action", tool="non_existent_tool", input={}, expected_output_schema="any")
        ]
    )

    results = executor.execute_plan(plan)

    assert 1 in results
    assert "error" in results[1]
    assert "not found" in results[1]["error"]
