"""Application-level retrieval for biomedical document chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from symptom_rag_analyzer.embeddings.model import BiomedicalEmbeddingModel
from symptom_rag_analyzer.vector_store.qdrant_store import QdrantVectorStore


@dataclass
class RetrievedEvidence:
	"""A retrieved chunk with its similarity score and source information."""

	text: str
	score: float
	filename: str
	page_number: int | None
	chunk_index: int
	source_type: str
	metadata: dict[str, Any] = field(default_factory=dict)


class BiomedicalRetriever:
	"""Coordinate query embedding and vector search for biomedical chunks."""

	def __init__(
		self,
		embedding_model: BiomedicalEmbeddingModel | None = None,
		vector_store: QdrantVectorStore | None = None,
	) -> None:
		self.embedding_model = (
			embedding_model if embedding_model is not None else BiomedicalEmbeddingModel()
		)
		self.vector_store = (
			vector_store if vector_store is not None else QdrantVectorStore(vector_size=768)
		)

	def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
		"""Return the highest-scoring evidence for a query."""
		if not isinstance(query, str):
			raise ValueError("query must be a string")
		if not query.strip():
			raise ValueError("query cannot be empty or whitespace")
		if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
			raise ValueError("top_k must be a positive integer")

		query_embedding = self.embedding_model.embed_text(query)
		results = self.vector_store.search(query_embedding, top_k=top_k)
		return [self._to_evidence(result) for result in results]

	@staticmethod
	def _to_evidence(result: Any) -> RetrievedEvidence:
		payload = result.payload or {}
		document_metadata = payload.get("document_metadata", {})
		chunk_metadata = payload.get("chunk_metadata", {})
		metadata = dict(payload.get("metadata", {}))
		metadata.update(document_metadata)
		metadata.update(chunk_metadata)

		return RetrievedEvidence(
			text=payload.get("text", ""),
			score=result.score,
			filename=payload.get("filename", ""),
			page_number=payload.get("page_number"),
			chunk_index=payload.get("chunk_index", 0),
			source_type=payload.get("source_type", "unknown"),
			metadata=metadata,
		)
