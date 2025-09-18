"""
Hybrid Retriever that combines dense and sparse search results.

This module implements a hybrid retrieval strategy that leverages both semantic
(dense) and keyword-based (sparse) search. It fetches results from both retrievers,
then uses a reranker model to produce a final, more relevant list of documents.
"""
from typing import List
from cross_encoder import CrossEncoder
from .dense import DenseRetriever
from .sparse import BM25Retriever
from ..utils.schema import DocumentChunk
from ..utils.logging import get_logger

logger = get_logger(__name__)

class HybridRetriever:
    """
    Combines results from dense and sparse retrievers and reranks them.
    """
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: BM25Retriever, reranker_model_name: str, alpha: float = 0.5):
        """
        Initializes the HybridRetriever.

        Args:
            dense_retriever (DenseRetriever): An instance of the dense retriever.
            sparse_retriever (BM25Retriever): An instance of the sparse retriever.
            reranker_model_name (str): The name of the cross-encoder model for reranking.
            alpha (float): The weight for combining dense and sparse scores (not used with reranker).
        """
        logger.info("Initializing hybrid retriever.")
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha
        try:
            # TODO: The cross-encoder model can be slow. For production, this might need to be
            # served as a separate microservice.
            self.reranker = CrossEncoder(reranker_model_name)
            logger.info(f"Cross-encoder model '{reranker_model_name}' loaded.")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            self.reranker = None

    def retrieve(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Retrieves and reranks documents from both dense and sparse retrievers.

        Args:
            query (str): The user's query.
            top_k (int): The final number of documents to return after reranking.

        Returns:
            List[DocumentChunk]: A list of the most relevant document chunks.
        """
        logger.debug(f"Performing hybrid retrieval for query: '{query}'")

        # 1. Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k * 2)

        # 2. Combine and deduplicate results
        combined_results = {doc.source: doc for doc in dense_results}
        for doc in sparse_results:
            if doc.source not in combined_results:
                combined_results[doc.source] = doc

        candidate_docs = list(combined_results.values())
        if not candidate_docs:
            logger.warning("No documents found by either dense or sparse retriever.")
            return []

        logger.info(f"Found {len(candidate_docs)} unique candidates for reranking.")

        # 3. Rerank using the cross-encoder
        if self.reranker:
            pairs = [[query, doc.content] for doc in candidate_docs]
            try:
                scores = self.reranker.predict(pairs)
                logger.debug(f"Reranking scores: {scores}")

                # Assign new scores and sort
                for doc, score in zip(candidate_docs, scores):
                    doc.score = float(score) # Update score with the reranker's score
                    doc.metadata['retriever'] = 'hybrid_reranked'

                reranked_docs = sorted(candidate_docs, key=lambda x: x.score, reverse=True)
                final_results = reranked_docs[:top_k]
                logger.info(f"Returning {len(final_results)} reranked documents.")
                return final_results

            except Exception as e:
                logger.error(f"Error during reranking: {e}. Falling back to score fusion.")
                # Fallback to simple score fusion if reranking fails
                return self._fuse_and_sort(dense_results, sparse_results, top_k)
        else:
            # Fallback if no reranker is available
            logger.warning("No reranker model loaded. Using simple score fusion.")
            return self._fuse_and_sort(dense_results, sparse_results, top_k)

    def _fuse_and_sort(self, dense: List[DocumentChunk], sparse: List[DocumentChunk], top_k: int) -> List[DocumentChunk]:
        """
        A simple fallback method to combine scores using Reciprocal Rank Fusion (RRF)
        or a weighted average. Here we just use a placeholder.
        """
        # TODO: Implement a proper score normalization and fusion technique like RRF.
        # This placeholder just combines, sorts by score, and returns top_k.
        all_docs = {doc.source: doc for doc in dense}
        for doc in sparse:
            if doc.source not in all_docs:
                all_docs[doc.source] = doc
            else:
                # Simple weighted average of scores (assuming scores are somewhat normalized)
                # This is a naive approach.
                dense_score = all_docs[doc.source].score
                sparse_score = doc.score
                all_docs[doc.source].score = (self.alpha * dense_score) + ((1 - self.alpha) * sparse_score)
                all_docs[doc.source].metadata['retriever'] = 'hybrid_fused'

        sorted_docs = sorted(list(all_docs.values()), key=lambda x: x.score, reverse=True)
        return sorted_docs[:top_k]
