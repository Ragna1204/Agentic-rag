"""
Tests for the Retriever modules (Dense, Sparse, Hybrid).
"""
import pytest
from unittest.mock import MagicMock, patch
from ..retriever.dense import DenseRetriever
from ..retriever.sparse import BM25Retriever
from ..retriever.hybrid import HybridRetriever
from ..utils.schema import DocumentChunk

# Mock data for retrievers
MOCK_CORPUS = [
    {"source": "doc1", "content": "Quantum computing promises to revolutionize science."},
    {"source": "doc2", "content": "The capital of France is Paris, a city of lights."},
    {"source": "doc3", "content": "BM25 is a great algorithm for keyword search."}
]

@pytest.fixture
def sparse_retriever():
    """Fixture for an in-memory BM25 retriever."""
    return BM25Retriever(corpus=MOCK_CORPUS)

def test_bm25_retriever(sparse_retriever):
    """
    Tests that the BM25 retriever finds documents based on keywords.
    """
    # TODO: Add more assertions to check scores and ranking order.
    query = "keyword search algorithm"
    results = sparse_retriever.retrieve(query, top_k=1)
    assert len(results) == 1
    assert results[0].source == "doc3"

@patch('faiss.read_index')
@patch('sentence_transformers.SentenceTransformer')
def test_dense_retriever(mock_sentencetransformer, mock_read_index):
    """
    Tests the dense retriever's logic.
    """
    # TODO: This test is complex to set up. It needs to mock the FAISS index
    # and the sentence transformer model outputs. This is a placeholder.
    
    # Setup mock model and index
    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = [[0.1, 0.2, 0.3]]
    mock_sentencetransformer.return_value = mock_model_instance

    mock_index_instance = MagicMock()
    # Distances and indices returned by faiss.search
    mock_index_instance.search.return_value = ([[0.1]], [[0]])
    mock_read_index.return_value = mock_index_instance

    # Mock the document map loading
    with patch.object(DenseRetriever, '_load_document_map', return_value={0: MOCK_CORPUS[0]}):
        dense_retriever = DenseRetriever(model_name="mock-model", index_path="mock-path")
        query = "quantum science"
        results = dense_retriever.retrieve(query, top_k=1)

        assert len(results) == 1
        assert results[0].source == "doc1"
        mock_model_instance.encode.assert_called_once_with([query])
        mock_index_instance.search.assert_called_once()


def test_hybrid_retriever_reranking():
    """
    Tests the hybrid retriever with a mock reranker.
    """
    # TODO: Implement this test. It should check that:
    # 1. Both dense and sparse retrievers are called.
    # 2. The results are combined and passed to the reranker.
    # 3. The final results are sorted according to the reranker's scores.
    
    # Mock retrievers
    mock_dense = MagicMock(spec=DenseRetriever)
    mock_dense.retrieve.return_value = [DocumentChunk(source="doc1", content="content1", score=0.8)]

    mock_sparse = MagicMock(spec=BM25Retriever)
    mock_sparse.retrieve.return_value = [DocumentChunk(source="doc2", content="content2", score=20.0)]

    # Mock reranker
    with patch('cross_encoder.CrossEncoder') as mock_cross_encoder:
        mock_reranker_instance = MagicMock()
        # Reranker gives higher score to the sparse result
        mock_reranker_instance.predict.return_value = [0.1, 0.9] 
        mock_cross_encoder.return_value = mock_reranker_instance

        hybrid_retriever = HybridRetriever(
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            reranker_model_name="mock-reranker"
        )
        
        query = "test query"
        results = hybrid_retriever.retrieve(query, top_k=2)

        assert len(results) == 2
        # Check that the result from sparse retriever is now ranked first
        assert results[0].source == "doc2"
        assert results[0].score == 0.9 
        mock_dense.retrieve.assert_called_once()
        mock_sparse.retrieve.assert_called_once()
        mock_reranker_instance.predict.assert_called_once()
