"""
The Planner module for the agentic RAG pipeline.

This module is responsible for taking a user query and generating a structured,
step-by-step action plan. The plan is represented as a JSON object that can be
executed by the Executor module.
"""
import json
from typing import List, Dict, Any
from ..utils.schema import ActionPlan
from ..utils.prompts import get_planner_prompt
from ..utils.logging import get_logger

logger = get_logger(__name__)

class Planner:
    """
    Generates an action plan to answer a user's query.
    """
    def __init__(self, llm_client, available_tools: List[str]):
        """
        Initializes the Planner.

        Args:
            llm_client: An instance of an LLM client (e.g., OpenAI's client).
            available_tools (List[str]): A list of tool names available for planning.
        """
        # TODO: The LLM client should be a generic interface, not tied to OpenAI.
        # This would allow swapping different LLMs (e.g., from Hugging Face, Anthropic).
        self.llm_client = llm_client
        self.available_tools = available_tools
        logger.info(f"Planner initialized with tools: {self.available_tools}")

    def generate_plan(self, query: str) -> ActionPlan:
        """
        Generates a step-by-step plan to address the user's query.

        Args:
            query (str): The user's input query.

        Returns:
            ActionPlan: A Pydantic model instance of the generated plan.
        """
        logger.debug(f"Generating plan for query: '{query}'")
        prompt = get_planner_prompt(query, self.available_tools)

        try:
            # TODO: Add retry logic and error handling for the LLM call.
            # What happens if the LLM returns malformed JSON or fails to respond?
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo", # This should be configurable
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            plan_json_str = response.choices[0].message.content
            logger.debug(f"LLM returned plan: {plan_json_str}")

            # Parse the JSON and validate with Pydantic
            plan_dict = json.loads(plan_json_str)
            action_plan = ActionPlan(**plan_dict)

            logger.info(f"Successfully generated plan with {len(action_plan.plan)} steps.")
            return action_plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM response: {e}")
            # TODO: Implement a fallback mechanism. Maybe try a simpler prompt or return an error.
            raise ValueError("Planner failed to generate a valid JSON plan.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during plan generation: {e}")
            raise

# Example usage:
# if __name__ == '__main__':
#     from openai import OpenAI
#     import os
#
#     # This is a mock client for demonstration.
#     class MockLLMClient:
#         class MockChat:
#             class MockCompletions:
#                 def create(self, *args, **kwargs):
#                     class MockChoice:
#                         class MockMessage:
#                             content = '''{
#                                 "query": "What is the capital of France and its population?",
#                                 "reasoning": "I will first find the capital and then its population.",
#                                 "plan": [
#                                     {"id": 1, "action": "find_capital", "tool": "web_search", "input": {"query": "capital of France"}, "expected_output_schema": "string"},
#                                     {"id": 2, "action": "find_population", "tool": "web_search", "input": {"query": "population of Paris"}, "expected_output_schema": "string"}
#                                 ]
#                             }'''
#                         message = MockMessage()
#                     return type("MockResponse", (), {"choices": [MockChoice()]})()
#             completions = MockCompletions()
#         chat = MockChat()
#
#     planner = Planner(llm_client=MockLLMClient(), available_tools=["web_search", "calculator"])
#     plan = planner.generate_plan("What is the capital of France and its population?")
#     print(plan.model_dump_json(indent=2))
