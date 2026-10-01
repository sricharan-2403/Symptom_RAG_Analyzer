"""Application interface for sending clinical context and evidence to the reasoning LLM."""

import json
import os
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT, build_prompt


DEFAULT_MODEL = "gemini-3.6-flash"

_RAG_ANSWER_SCHEMA = {
	"type": "object",
	"properties": {
		"diagnostic_candidates": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"condition": {"type": "string"},
					"rank": {"type": "integer"},
					"justification": {"type": "string"},
					"supporting_evidence": {
						"type": "array",
						"items": {"type": "integer"},
					},
				},
				"required": [
					"condition",
					"rank",
					"justification",
					"supporting_evidence",
				],
			},
		},
		"limitations": {"type": "string"},
	},
	"required": ["diagnostic_candidates", "limitations"],
}


@dataclass
class DiagnosticCandidate:
	"""One preliminary diagnostic hypothesis supported by retrieved evidence."""

	condition: str
	rank: int
	justification: str
	supporting_evidence: list[int]


@dataclass
class RAGAnswer:
	"""Internal Answerer output, not yet the final common Data Fusion output."""

	diagnostic_candidates: list[DiagnosticCandidate]
	limitations: str


class BiomedicalAnswerer:
	"""Send prepared clinical context to Gemini and validate its structured response."""

	def __init__(self, model: str = DEFAULT_MODEL) -> None:
		"""Initialize the Gemini client using the GEMINI_API_KEY environment variable."""
		load_dotenv()
		api_key = os.environ.get("GEMINI_API_KEY")
		if not api_key or not api_key.strip():
			raise ValueError("GEMINI_API_KEY environment variable is required.")
		if not isinstance(model, str) or not model.strip():
			raise ValueError("model must be a non-empty string.")

		self.model = model
		try:
			self.client = genai.Client(api_key=api_key)
		except Exception as error:
			raise RuntimeError("Failed to initialize the Gemini client.") from error

	def answer(
		self,
		clinical_context: str,
		evidence_context: str,
	) -> RAGAnswer:
		"""Generate and validate an internal RAG answer from supplied context."""
		if not isinstance(clinical_context, str):
			raise TypeError("clinical_context must be a string.")
		if not isinstance(evidence_context, str):
			raise TypeError("evidence_context must be a string.")
		if not clinical_context.strip():
			raise ValueError("clinical_context cannot be empty or whitespace.")

		prompt = build_prompt(clinical_context, evidence_context)
		try:
			response = self.client.models.generate_content(
				model=self.model,
				contents=prompt,
				config=types.GenerateContentConfig(
					systemInstruction=SYSTEM_PROMPT,
					responseMimeType="application/json",
					responseSchema=_RAG_ANSWER_SCHEMA,
				),
			)
			return self._parse_response(response.text)
		except (TypeError, ValueError) as error:
			if isinstance(error, (TypeError, ValueError)) and str(error).startswith(
				"Invalid Gemini response:"
			):
				raise
			raise RuntimeError("Gemini request failed.") from error
		except Exception as error:
			raise RuntimeError("Gemini request failed.") from error

	@staticmethod
	def _parse_response(response_text: str) -> RAGAnswer:
		"""Validate Gemini JSON and convert it to the internal RAGAnswer model."""
		try:
			payload: Any = json.loads(response_text)
		except (TypeError, json.JSONDecodeError) as error:
			raise ValueError("Invalid Gemini response: response was not valid JSON.") from error

		if not isinstance(payload, dict):
			raise ValueError("Invalid Gemini response: expected a JSON object.")

		candidates = payload.get("diagnostic_candidates")
		limitations = payload.get("limitations")
		if not isinstance(candidates, list):
			raise ValueError("Invalid Gemini response: diagnostic_candidates must be a list.")
		if not isinstance(limitations, str):
			raise ValueError("Invalid Gemini response: limitations must be a string.")

		validated_candidates = []
		for candidate in candidates:
			if not isinstance(candidate, dict):
				raise ValueError("Invalid Gemini response: each candidate must be an object.")
			condition = candidate.get("condition")
			rank = candidate.get("rank")
			justification = candidate.get("justification")
			supporting_evidence = candidate.get("supporting_evidence")
			if not isinstance(condition, str) or not condition.strip():
				raise ValueError("Invalid Gemini response: condition must be non-empty.")
			if not isinstance(rank, int) or isinstance(rank, bool):
				raise ValueError("Invalid Gemini response: rank must be an integer.")
			if not isinstance(justification, str) or not justification.strip():
				raise ValueError("Invalid Gemini response: justification must be non-empty.")
			if not isinstance(supporting_evidence, list) or any(
				not isinstance(item, int) or isinstance(item, bool)
				for item in supporting_evidence
			):
				raise ValueError(
					"Invalid Gemini response: supporting_evidence must contain integers."
				)
			validated_candidates.append(
				DiagnosticCandidate(
					condition=condition,
					rank=rank,
					justification=justification,
					supporting_evidence=supporting_evidence,
				)
			)

		return RAGAnswer(
			diagnostic_candidates=validated_candidates,
			limitations=limitations,
		)
