"""
The Aggregator module for the agentic RAG pipeline.

This module is responsible for the final step of the process: synthesizing all
the collected and verified information into a single, coherent, and well-sourced
answer for the user.
"""
from typing import List, Dict, Any
from ..utils.schema import FinalAnswer, DocumentChunk, VerificationResult
from ..utils.prompts import get_aggregator_prompt
from ..utils.logging import get_logger

logger = get_logger(__name__)

class Aggregator:
    """
    Aggregates information from previous steps into a final answer.
    """
    def __init__(self, llm_client):
        """
        Initializes the Aggregator.

        Args:
            llm_client: An instance of an LLM client (e.g., OpenAI's client).
        """
        # TODO: The LLM client should be a generic interface.
        self.llm_client = llm_client
        logger.info("Aggregator initialized.")

    def synthesize_answer(
        self,
        query: str,
        step_results: Dict[int, Any],
        verification_results: List[VerificationResult]
    ) -> FinalAnswer:
        """
        Synthesizes the final answer from the outputs of the execution and verification steps.

        Args:
            query (str): The original user query.
            step_results (Dict[int, Any]): The outputs from the Executor.
            verification_results (List[VerificationResult]): The outputs from the Verifier.

        Returns:
            FinalAnswer: The final, structured answer.
        """
        logger.debug("Synthesizing final answer.")

        # 1. Consolidate summaries and sources
        summaries = self._extract_summaries(step_results)
        all_sources = self._collect_sources(step_results)

        # 2. Generate the narrative answer using an LLM
        prompt = get_aggregator_prompt(query, summaries, verification_results)
        try:
            # TODO: Add retry logic and error handling.
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo", # Should be configurable
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            final_narrative = response.choices[0].message.content
            logger.info("Successfully synthesized final narrative.")
        except Exception as e:
            logger.error(f"Failed to synthesize final answer from LLM: {e}")
            final_narrative = "Error: Could not generate a final answer due to an internal error."

        # 3. Calculate confidence score
        confidence = self._calculate_confidence(verification_results)

        # 4. Identify unverified claims
        unverified = [res.reasoning for res in verification_results if res.verdict == "INSUFFICIENT_EVIDENCE"]

        return FinalAnswer(
            answer=final_narrative,
            sources=list(all_sources.values()), # Deduplicated sources
            confidence_score=confidence,
            unverified_claims=unverified
        )

    def _extract_summaries(self, step_results: Dict[int, Any]) -> List[str]:
        """Helper to pull out string summaries from step results."""
        # TODO: This is a naive extraction. It should be more robust, perhaps based on
        # the `expected_output_schema` from the plan.
        summaries = []
        for result in step_results.values():
            if isinstance(result, str):
                summaries.append(result)
            elif isinstance(result, list) and result and isinstance(result[0], str):
                summaries.extend(result)
        return summaries

    def _collect_sources(self, step_results: Dict[int, Any]) -> Dict[str, DocumentChunk]:
        """Helper to collect and deduplicate all source documents."""
        sources = {}
        for result in step_results.values():
            if isinstance(result, list) and all(isinstance(item, DocumentChunk) for item in result):
                for doc in result:
                    if doc.source not in sources:
                        sources[doc.source] = doc
        return sources

    def _calculate_confidence(self, verification_results: List[VerificationResult]) -> float:
        """
        Calculates a confidence score based on the verification results.
        """
        # TODO: This is a simple heuristic. A more advanced model could be trained.
        if not verification_results:
            return 0.5 # Neutral confidence if no verification was done

        supported_count = sum(1 for res in verification_results if res.verdict == "SUPPORTS")
        total_verifications = len(verification_results)

        if total_verifications == 0:
            return 0.5

        confidence = supported_count / total_verifications
        logger.info(f"Calculated confidence score: {confidence:.2f}")
        return confidence
