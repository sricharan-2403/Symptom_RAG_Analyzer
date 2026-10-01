"""Build prompt context from retrieved evidence."""

from symptom_rag_analyzer.retrieval.search import RetrievedEvidence


class EvidenceContextBuilder:
	"""Format retrieved evidence as deterministic prompt context."""

	def build_context(self, evidence: list[RetrievedEvidence]) -> str:
		"""Return formatted evidence blocks in their original order."""
		if not evidence:
			return ""

		blocks = []
		for index, item in enumerate(evidence, start=1):
			blocks.append(
				"\n".join(
					[
						f"[Evidence {index}]",
						f"Source: {item.filename}",
						f"Source type: {item.source_type}",
						f"Page: {item.page_number}",
						f"Similarity: {item.score}",
						f"Text: {item.text}",
					]
				)
			)

		return "\n\n".join(blocks)
