"""
The Working Memory module for the Human Thought Simulator.

This module simulates the brain's short-term or working memory. It holds the
immediate context of the current task, including conversation history,
intermediate results from the agent's plan, and the current emotional state.
"""
from collections import deque
from typing import Any, Dict, List

class WorkingMemory:
    """
    A short-term buffer with limited capacity.
    """
    def __init__(self, capacity: int = 10):
        """
        Initializes the WorkingMemory.

        Args:
            capacity (int): The maximum number of items (e.g., conversation turns,
                            step results) to hold in memory.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.emotional_state = {"valence": 0.0, "arousal": 0.0} # Neutral

    def add(self, item: Any):
        """Adds an item to the working memory."""
        self.buffer.append(item)

    def get_current_context(self) -> List[Any]:
        """Returns the current context held in memory."""
        return list(self.buffer)

    def clear(self):
        """Clears the working memory."""
        self.buffer.clear()

    def set_emotional_state(self, valence: float, arousal: float):
        """
        Sets the agent's current emotional state.
        Valence: positive/negative feeling
        Arousal: energy/activation level
        """
        self.emotional_state = {"valence": valence, "arousal": arousal}
