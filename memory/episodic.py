"""
The Episodic Memory module for the Human Thought Simulator.

This module logs events, conversations, and outcomes over time, creating a
chronological record of the agent's experiences.
"""
import datetime
import json
import os
from typing import List, Dict, Any

class EpisodicMemory:
    """
    Stores and retrieves time-stamped event logs.
    """
    def __init__(self, log_file: str = "storage/episodic_memory.log"):
        """
        Initializes the EpisodicMemory.

        Args:
            log_file (str): The path to the log file where memories are stored.
        """
        self.log_file = log_file
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Logs a new event to episodic memory.

        Args:
            event_type (str): The type of event (e.g., "query", "plan", "final_answer").
            data (Dict[str, Any]): The data associated with the event.
        """
        timestamp = datetime.datetime.now().isoformat()
        # A simple default serializer for objects that are not JSON serializable
        def default_serializer(o):
            if hasattr(o, 'model_dump'): # For Pydantic models
                return o.model_dump()
            return f"<non-serializable: {type(o).__name__}>"

        log_entry = {"timestamp": timestamp, "event_type": event_type, "data": data}
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry, default=default_serializer) + '\n')

    def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent episodes from the log file.
        """
        if not os.path.exists(self.log_file):
            return []
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Get the last `limit` lines
            recent_lines = lines[-limit:]
            
            episodes = []
            for line in recent_lines:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    # Handle cases where a line is not valid JSON
                    continue
            return episodes
        except IOError:
            return []

