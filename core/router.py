"""
Query Router to decide the execution path.

This module contains the logic to determine whether a user's query can be
answered with a simple, direct RAG pipeline or if it requires the more
complex, multi-step agentic approach.
"""
from ..utils.logging import get_logger

logger = get_logger(__name__)

class Router:
    """
    A simple router that decides the processing path for a query.
    """
    def __init__(self):
        """
        Initializes the Router.
        """
        # TODO: The routing logic is currently very basic. A more advanced implementation
        # could use an LLM call to classify the query's intent and complexity.
        # For example, train a classifier or use few-shot prompting to determine if a query
        # is "simple_lookup" vs "multi_step_reasoning".
        logger.info("Router initialized.")

    def decide_path(self, query: str) -> str:
        """
        Decides whether to use the 'simple_rag' or 'agentic' path.

        Args:
            query (str): The user's input query.

        Returns:
            str: Either "simple_rag" or "agentic".
        """
        logger.debug(f"Routing query: '{query}'")

        # Simple heuristic: if the query contains keywords that suggest complexity,
        # comparison, or a sequence of actions, use the agentic path.
        agentic_keywords = [
            "compare", "vs", "what are the differences", "pros and cons",
            "step-by-step", "how to", "what if", "and", "then", "first", "second"
        ]

        query_lower = query.lower()
        if any(keyword in query_lower for keyword in agentic_keywords):
            logger.info("Query routed to 'agentic' path.")
            return "agentic"

        # Another heuristic: if the query is very long, it might need a plan.
        if len(query.split()) > 20:
            logger.info("Long query routed to 'agentic' path.")
            return "agentic"

        logger.info("Query routed to 'simple_rag' path.")
        return "simple_rag"

# Example usage:
# if __name__ == '__main__':
#     router = Router()
#     query1 = "What is quantum error correction?"
#     query2 = "Compare the pros and cons of surface codes and LDPC codes for quantum computing."
#     print(f"Query: '{query1}' -> Path: {router.decide_path(query1)}")
#     print(f"Query: '{query2}' -> Path: {router.decide_path(query2)}")
