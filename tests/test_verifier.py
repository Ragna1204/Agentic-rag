"""
Tests for the Verifier module.
"""
import pytest
from unittest.mock import MagicMock
from ..core.verifier import Verifier
from ..utils.schema import DocumentChunk, VerificationResult

@pytest.fixture
def mock_llm_client():
    """Fixture for a mock LLM client for the verifier."""
    client = MagicMock()
    # Default mock response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '''{"verdict": "SUPPORTS", "reasoning": "The evidence explicitly states the claim."}'''
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_response
    return client

def test_verify_claim_supported(mock_llm_client):
    """
    Tests a successful verification where the evidence supports the claim.
    """
    # TODO: Add tests for "CONTRADICTS" and "INSUFFICIENT_EVIDENCE" verdicts.
    verifier = Verifier(llm_client=mock_llm_client)
    claim = "The sky is blue."
    evidence = [
        DocumentChunk(source="doc1.txt", content="The sky appears blue due to Rayleigh scattering.", score=0.9)
    ]

    result = verifier.verify(claim, evidence)

    assert isinstance(result, VerificationResult)
    assert result.verdict == "SUPPORTS"
    assert result.reasoning == "The evidence explicitly states the claim."
    mock_llm_client.chat.completions.create.assert_called_once()

def test_verify_with_no_evidence(mock_llm_client):
    """
    Tests that the verifier returns INSUFFICIENT_EVIDENCE if no evidence is provided.
    """
    # TODO: Implement this test case.
    verifier = Verifier(llm_client=mock_llm_client)
    claim = "The sky is blue."
    result = verifier.verify(claim, [])

    assert result.verdict == "INSUFFICIENT_EVIDENCE"
    # Ensure the LLM was not called
    mock_llm_client.chat.completions.create.assert_not_called()

def test_extract_claims():
    """
    Tests the claim extraction helper function.
    """
    # TODO: Make this test more robust, testing edge cases like text with no periods.
    verifier = Verifier(llm_client=MagicMock())
    text = "The capital of France is Paris. Paris is a beautiful city. It has a famous tower."
    claims = verifier.extract_claims(text)
    assert len(claims) == 3
    assert claims[0] == "The capital of France is Paris."
    assert claims[1] == "Paris is a beautiful city."
