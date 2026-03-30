from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticCheck:
    lexical_overlap: float
    embedding_similarity: float
    contradiction_flag: bool
    semantic_score: float
    passed: bool
    backend: str


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def lexical_overlap(reference: str, candidate: str) -> float:
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(ref_tokens | cand_tokens)


def _negation_contradiction(reference: str, candidate: str) -> bool:
    negation_markers = (" not ", " never ", " cannot ", " can't ", " no ", " none ", " invalid ")
    ref_lower = f" {reference.lower()} "
    cand_lower = f" {candidate.lower()} "
    ref_neg = any(marker in ref_lower for marker in negation_markers)
    cand_neg = any(marker in cand_lower for marker in negation_markers)
    return ref_neg != cand_neg


class SemanticEvaluator:
    def __init__(
        self,
        min_semantic_score: float = 0.55,
        min_lexical_overlap: float = 0.34,
        allow_download: bool = False,
    ) -> None:
        self.min_semantic_score = min_semantic_score
        self.min_lexical_overlap = min_lexical_overlap
        self.allow_download = allow_download
        self._embedder = None
        self._embedder_name = "sentence-transformers/all-MiniLM-L6-v2"

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        if not self.allow_download:
            self._embedder = False
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._embedder = False
            return self._embedder
        try:
            self._embedder = SentenceTransformer(self._embedder_name, local_files_only=True)
        except Exception:
            if not self.allow_download:
                self._embedder = False
                return self._embedder
            self._embedder = SentenceTransformer(self._embedder_name)
        return self._embedder

    def _embedding_similarity(self, reference: str, candidate: str) -> float:
        embedder = self._load_embedder()
        if embedder is False:
            return 0.0
        import numpy as np

        ref_vec, cand_vec = embedder.encode([reference, candidate], normalize_embeddings=True)
        return float(np.dot(ref_vec, cand_vec))

    def evaluate(self, reference: str, candidate: str) -> SemanticCheck:
        lexical = round(lexical_overlap(reference, candidate), 4)
        contradiction = _negation_contradiction(reference, candidate)
        embedding = 0.0
        backend = "lexical"
        try:
            embedding = round(self._embedding_similarity(reference, candidate), 4)
            if not math.isclose(embedding, 0.0):
                backend = "hybrid"
        except Exception:
            embedding = 0.0
            backend = "lexical"
        length_ratio = min(len(reference), len(candidate)) / max(1, max(len(reference), len(candidate)))
        if backend == "hybrid":
            score = round((0.4 * lexical) + (0.5 * embedding) + (0.1 * length_ratio), 4)
        else:
            score = round((0.8 * lexical) + (0.2 * length_ratio), 4)
        threshold = self.min_semantic_score if backend == "hybrid" else min(0.45, self.min_semantic_score)
        passed = lexical >= min(0.22, self.min_lexical_overlap) and score >= threshold and not contradiction
        return SemanticCheck(
            lexical_overlap=lexical,
            embedding_similarity=embedding,
            contradiction_flag=contradiction,
            semantic_score=score,
            passed=passed,
            backend=backend,
        )
