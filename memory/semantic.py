"""
The Semantic Memory module for the Human Thought Simulator.

This module stores generalized concepts and abstract knowledge, similar to
human semantic memory. It's built on top of the existing retriever
infrastructure (e.g., vector stores like FAISS).
"""
from typing import Any

class SemanticMemory:
    """
    Manages the storage and retrieval of abstract concepts.
    """
    def __init__(self, retriever: Any):
        """
        Initializes the SemanticMemory.

        Args:
            retriever: An instance of a retriever (e.g., DenseRetriever,
                       HybridRetriever) that manages the underlying vector store.
        """
        self.retriever = retriever

    def add_concept(self, concept_text: str):
        """
        Embeds and stores a new concept in the vector store.

        Args:
            concept_text (str): The textual description of the new concept.
        """
        new_concept = {"content": concept_text, "source": "reflection"}
        
        # Check for the method to add documents on the retriever
        if hasattr(self.retriever, 'add_documents') and callable(getattr(self.retriever, 'add_documents')):
            self.retriever.add_documents([new_concept])
        elif hasattr(self.retriever, 'add') and callable(getattr(self.retriever, 'add')):
            # This would be for a dense retriever, assuming it has an `add` method
            self.retriever.add([new_concept])
        else:
            # Log a warning if no method to add documents is found
            logger = get_logger(__name__)
            logger.warning("The configured retriever does not have an `add_documents` or `add` method. Cannot add new concepts to semantic memory.")

    def retrieve_related_concepts(self, query: str, top_k: int = 3) -> Any:
        """
        Retrieves concepts related to a given query.
        """
        return self.retriever.retrieve(query, top_k=top_k)
