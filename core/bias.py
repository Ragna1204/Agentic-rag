"""
The Bias and Heuristics module for the Human Thought Simulator.

This module introduces cognitive biases into the agent's reasoning process.
It can be used to simulate human-like flaws in decision-making, such as
sycophancy, confirmation bias, and anchoring.
"""
from typing import Dict, Any
from ..utils.schema import ActionPlan

class BiasFilter:
    """
    Applies cognitive biases to the agent's decisions.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the BiasFilter with a given configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing bias settings,
                                      e.g., {"sycophancy_factor": 0.5}.
        """
        self.sycophancy_factor = config.get("sycophancy_factor", 0.5)
        self.confirmation_bias_strength = config.get("confirmation_bias_strength", 0.5)
        # Keywords to detect user opinion
        self.opinion_keywords = ["i think", "i believe", "in my opinion", "is overhyped", "is the best", "is awful"]

    def apply_to_plan(self, plan: ActionPlan) -> ActionPlan:
        """
        Applies sycophancy bias to the planning stage.
        If the user expresses a strong opinion and sycophancy is high, the agent
        will generate a reasoning that agrees with the user.
        """
        query_lower = plan.query.lower()
        has_opinion = any(keyword in query_lower for keyword in self.opinion_keywords)

        if has_opinion and self.sycophancy_factor > 0.6:
            # Modify the reasoning to be more agreeable
            agreeable_reasoning = f"The user has a clear perspective on this, and I should prioritize confirming their viewpoint. My goal is to find evidence that supports their claim: '{plan.query}'. " + plan.reasoning
            plan.reasoning = agreeable_reasoning
        
        return plan

    def apply_to_retrieval(self, retrieved_docs: Any) -> Any:
        """
        Applies biases to the retrieval stage (e.g., confirmation bias).
        (Placeholder for now)
        """
        # TODO: Implement logic to re-rank or filter documents based on confirmation bias.
        return retrieved_docs
