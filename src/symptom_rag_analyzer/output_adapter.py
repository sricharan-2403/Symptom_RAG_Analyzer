"""Convert internal Symptom RAG results to the common agent output contract."""

from __future__ import annotations

from typing import Any

from symptom_rag_analyzer.orchestrator import ClinicalTextClarifierOutput, SymptomRAGResult


class SymptomRAGOutputAdapter:
	"""Adapt a successful internal RAG result without adding reasoning or fusion."""

	def adapt(
		self,
		clinical_context: ClinicalTextClarifierOutput,
		result: SymptomRAGResult,
	) -> dict[str, Any]:
		"""Return the JSON-compatible common output for an internal RAG result."""
		if not isinstance(clinical_context, ClinicalTextClarifierOutput):
			raise TypeError("clinical_context must be a ClinicalTextClarifierOutput instance")
		if not isinstance(result, SymptomRAGResult):
			raise TypeError("result must be a SymptomRAGResult instance")

		return {
			"agent": "symptom_rag",
			"status": "success",
			"query_context": {
				"diseases": list(clinical_context.diseases),
				"symptoms": list(clinical_context.symptoms),
				"medications": list(clinical_context.medications),
				"tests": list(clinical_context.tests),
				"procedures": [],
			},
			"diagnostic_candidates": [
				{
					"condition": candidate.condition,
					"rank": candidate.rank,
					"justification": candidate.justification,
					"supporting_evidence": list(candidate.supporting_evidence),
				}
				for candidate in result.answer.diagnostic_candidates
			],
			"evidence": [
				{
					"content": evidence.text,
					"source": evidence.filename,
					"source_type": evidence.source_type,
					"location": (
						f"Page {evidence.page_number}"
						if evidence.page_number is not None
						else ""
					),
					"relevance_score": evidence.score,
				}
				for evidence in result.evidence
			],
			"metadata": {"evidence_count": len(result.evidence)},
			"limitations": result.answer.limitations,
			"error": None,
		}