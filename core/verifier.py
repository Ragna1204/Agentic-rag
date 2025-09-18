"""
The Verifier module for the agentic RAG pipeline.

This module is responsible for fact-checking claims generated during the
execution process. It takes a claim and a set of retrieved documents (evidence)
and uses an LLM to determine if the evidence supports or contradicts the claim.
"""
import json
from typing import List
from ..utils.schema import DocumentChunk, VerificationResult
from ..utils.prompts import get_verifier_prompt
from ..utils.logging import get_logger

logger = get_logger(__name__)

class Verifier:
    """
    Verifies claims against a body of evidence.
    """
    def __init__(self, llm_client):
        """
        Initializes the Verifier.

        Args:
            llm_client: An instance of an LLM client (e.g., OpenAI's client).
        """
        # TODO: The LLM client should be a generic interface.
        self.llm_client = llm_client
        logger.info("Verifier initialized.")

    def verify(self, claim: str, evidence_chunks: List[DocumentChunk]) -> VerificationResult:
        """
        Verifies a single claim against a list of evidence chunks.

        Args:
            claim (str): The claim to be verified (e.g., a sentence from a summary).
            evidence_chunks (List[DocumentChunk]): A list of document chunks to use as evidence.

        Returns:
            VerificationResult: A Pydantic model instance of the verification outcome.
        """
        if not evidence_chunks:
            logger.warning(f"No evidence provided to verify claim: '{claim}'")
            return VerificationResult(
                verdict="INSUFFICIENT_EVIDENCE",
                evidence=[],
                reasoning="No evidence was provided for verification."
            )

        logger.debug(f"Verifying claim: '{claim}'")
        evidence_text = "\n\n".join([f"Source: {chunk.source}\nContent: {chunk.content}" for chunk in evidence_chunks])
        prompt = get_verifier_prompt(claim=claim, evidence=evidence_text)

        try:
            # TODO: Add retry logic and error handling for the LLM call.
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo", # Should be configurable
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            result_json_str = response.choices[0].message.content
            logger.debug(f"LLM returned verification result: {result_json_str}")

            result_dict = json.loads(result_json_str)

            # We associate the original evidence with the result.
            verification_result = VerificationResult(
                verdict=result_dict.get("verdict", "INSUFFICIENT_EVIDENCE"),
                reasoning=result_dict.get("reasoning", "No reasoning provided."),
                evidence=evidence_chunks
            )
            logger.info(f"Verification result for claim: {verification_result.verdict}")
            return verification_result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from verifier LLM response: {e}")
            return VerificationResult(
                verdict="INSUFFICIENT_EVIDENCE",
                evidence=evidence_chunks,
                reasoning="Failed to parse verification response from LLM."
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred during verification: {e}")
            return VerificationResult(
                verdict="INSUFFICIENT_EVIDENCE",
                evidence=evidence_chunks,
                reasoning=f"An unexpected error occurred: {e}"
            )

    def extract_claims(self, text: str) -> List[str]:
        """
        Extracts individual, verifiable claims from a block of text.
        This is a helper function that could be used before calling `verify`.

        Args:
            text (str): The text to extract claims from (e.g., a summary).

        Returns:
            List[str]: A list of individual claims (sentences).
        """
        # TODO: This is a very naive implementation. A more advanced approach would use
        # an LLM or a fine-tuned NLP model to identify factual statements worth verifying.
        # For now, we just split by sentences.
        import re
        sentences = re.split(r'(?<=[.!?]) +', text)
        claims = [s.strip() for s in sentences if s.strip()]
        logger.info(f"Extracted {len(claims)} potential claims from text.")
        return claims
