"""
The Reflection and Consolidation module for the Human Thought Simulator.

This module is responsible for the agent's "metacognition" - its ability to
think about its own thoughts, learn from past experiences, and form new,
abstract concepts.
"""
from typing import List, Dict, Any
from collections import Counter
import re

class Reflection:
    """
    Manages the consolidation of episodic memories into semantic knowledge.
    """
    def __init__(self, episodic_memory, semantic_memory, llm_client):
        """
        Initializes the Reflection module.

        Args:
            episodic_memory: An instance of the episodic memory store.
            semantic_memory: An instance of the semantic memory store.
            llm_client: An instance of an LLM client.
        """
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.llm_client = llm_client

    def consolidate_memories(self) -> Dict[str, Any]:
        """
        Performs the "dreaming" or consolidation process.

        1. Fetches recent episodic memories.
        2. Identifies recurring themes or topics.
        3. Generates a new, generalized insight.
        4. Stores the new insight in semantic memory.
        """
        recent_episodes = self.episodic_memory.get_recent_episodes(limit=20)
        
        if len(recent_episodes) < 5: # Don't reflect on too few memories
            return {"status": "Not enough new memories to consolidate."}

        # 1. Extract queries from episodes
        queries = [episode['data']['query'] for episode in recent_episodes if episode['event_type'] in ['agentic_query', 'simple_query'] and 'query' in episode['data']]
        if not queries:
            return {"status": "No queries found in recent memories."}

        # 2. Find the most common topic using simple keyword analysis
        all_text = " ".join(queries)
        words = re.findall(r'\b\w{4,15}\b', all_text.lower()) # Find reasonably sized words
        # A simple list of stopwords
        stopwords = set(["what", "who", "when", "where", "why", "how", "the", "and", "a", "an", "is", "of", "for", "in", "on", "to"])
        words = [word for word in words if word not in stopwords]
        
        if not words:
            return {"status": "No significant topics found in queries."}

        most_common_topic = Counter(words).most_common(1)[0][0]

        # 3. Use an LLM to generate an insight about the topic
        prompt = f"Based on the fact that a user has repeatedly asked about '{most_common_topic}', what is a single, concise, abstract insight or summary that can be learned? The user's queries were: {queries}. Formulate a general statement. For example, if the topic is 'RAG', a good insight would be 'Retrieval-Augmented Generation is a technique to reduce hallucinations in LLMs by providing external knowledge.'"

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo", # Should be configurable
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100,
            )
            new_insight = response.choices[0].message.content.strip()
        except Exception as e:
            return {"status": "Failed to generate insight from LLM.", "error": str(e)}

        # 4. Store the new insight in semantic memory
        self.semantic_memory.add_concept(new_insight)

        return {
            "status": "Consolidation complete.",
            "topic": most_common_topic,
            "new_insight": new_insight
        }
