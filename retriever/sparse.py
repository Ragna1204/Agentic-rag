"""
Sparse Retriever using keyword-based algorithms like BM25.

This module provides a retriever that scores documents based on term frequency
and inverse document frequency. It's effective for queries where specific
keywords are important. This implementation uses the `rank-bm25` library for an
in-memory BM25 index. An Elasticsearch-based retriever is also sketched out.
"""
from rank_bm25 import BM25Okapi
from typing import List
from ..utils.schema import DocumentChunk
from ..utils.logging import get_logger

logger = get_logger(__name__)

class BM25Retriever:
    """
    A sparse retriever using the BM25 algorithm.
    """
    def __init__(self, corpus: List[dict]):
        """
        Initializes the BM25Retriever.

        Args:
            corpus (List[dict]): A list of documents, where each dict has 'source' and 'content'.
        """
        self.corpus = []
        self.documents = []
        self.bm25 = None
        if corpus:
            self.add_documents(corpus)

    def add_documents(self, new_corpus: List[dict]):
        """
        Adds new documents to the corpus and rebuilds the BM25 index.
        """
        logger.info(f"Adding {len(new_corpus)} new documents to BM25 index.")
        self.corpus.extend(new_corpus)
        self.documents = [doc['content'] for doc in self.corpus]
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index rebuilt with {len(self.documents)} total documents.")

    def retrieve(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Retrieves the top_k most relevant documents for a given query.

        Args:
            query (str): The user's query.
            top_k (int): The number of documents to retrieve.

        Returns:
            List[DocumentChunk]: A list of relevant document chunks.
        """
        if not self.bm25:
            logger.error("BM25 retriever is not initialized. Cannot retrieve.")
            return []

        logger.debug(f"Retrieving top {top_k} documents for query: '{query}'")
        try:
            tokenized_query = query.lower().split()
            doc_scores = self.bm25.get_scores(tokenized_query)

            # Get top_k scores and their indices
            top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]

            results = []
            for i in top_indices:
                original_doc = self.corpus[i]
                results.append(
                    DocumentChunk(
                        source=original_doc.get("source", "unknown"),
                        content=original_doc.get("content", ""),
                        score=doc_scores[i],
                        metadata={"retriever": "sparse"}
                    )
                )
            logger.info(f"Retrieved {len(results)} documents with sparse retriever.")
            return results
        except Exception as e:
            logger.error(f"Error during sparse retrieval: {e}")
            return []


# TODO: Implement an Elasticsearch-based sparse retriever for production use cases.
# from elasticsearch import Elasticsearch
# class ElasticsearchRetriever:
#     def __init__(self, host: str, port: int, index_name: str):
#         logger.info(f"Initializing Elasticsearch retriever for index: {index_name}")
#         self.es = Elasticsearch([{'host': host, 'port': port}])
#         self.index_name = index_name
#         if not self.es.ping():
#             raise ConnectionError("Could not connect to Elasticsearch.")

#     def retrieve(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
#         logger.debug(f"Querying Elasticsearch index '{self.index_name}' with query: '{query}'")
#         # TODO: Implement the actual search query using the Elasticsearch Python client.
#         # The query should use BM25 scoring.
#         # search_body = {
#         #     "query": {
#         #         "match": {
#         #             "content": query
#         #         }
#         #     }
#         # }
#         # response = self.es.search(index=self.index_name, body=search_body, size=top_k)
#         # ... process hits and return DocumentChunk list ...
#         return []
