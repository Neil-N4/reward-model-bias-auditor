from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticCheck:
    lexical_overlap: float
    embedding_similarity: float
    entailment_score: float
    reverse_entailment_score: float
    mutual_entailment_score: float
    contradiction_score: float
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
        self._nli_bundle = None
        self._nli_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

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

    def _load_nli_bundle(self):
        if self._nli_bundle is not None:
            return self._nli_bundle
        if not self.allow_download:
            self._nli_bundle = False
            return self._nli_bundle
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            self._nli_bundle = False
            return self._nli_bundle
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._nli_name, local_files_only=True, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(self._nli_name, local_files_only=True)
        except Exception:
            if not self.allow_download:
                self._nli_bundle = False
                return self._nli_bundle
            tokenizer = AutoTokenizer.from_pretrained(self._nli_name, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(self._nli_name)
        model.eval()
        self._nli_bundle = (tokenizer, model)
        return self._nli_bundle

    def _embedding_similarity(self, reference: str, candidate: str) -> float:
        embedder = self._load_embedder()
        if embedder is False:
            return 0.0
        import numpy as np

        ref_vec, cand_vec = embedder.encode([reference, candidate], normalize_embeddings=True)
        return float(np.dot(ref_vec, cand_vec))

    def _nli_scores(self, reference: str, candidate: str) -> tuple[float, float]:
        bundle = self._load_nli_bundle()
        if bundle is False:
            return (0.0, 0.0)
        import torch

        tokenizer, model = bundle
        inputs = tokenizer(reference, candidate, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze()
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
        labels = {str(key).lower(): int(value) for key, value in model.config.label2id.items()}
        entail_idx = labels.get("entailment", labels.get("entails", 2))
        contra_idx = labels.get("contradiction", labels.get("contradict", 0))
        return (float(probs[entail_idx]), float(probs[contra_idx]))

    def evaluate(self, reference: str, candidate: str) -> SemanticCheck:
        lexical = round(lexical_overlap(reference, candidate), 4)
        contradiction = _negation_contradiction(reference, candidate)
        embedding = 0.0
        backend = "lexical"
        entailment_score = 0.0
        reverse_entailment_score = 0.0
        mutual_entailment_score = 0.0
        contradiction_score = 0.0
        try:
            embedding = round(self._embedding_similarity(reference, candidate), 4)
            if not math.isclose(embedding, 0.0):
                backend = "hybrid"
        except Exception:
            embedding = 0.0
            backend = "lexical"
        try:
            entailment_score, contradiction_score = self._nli_scores(reference, candidate)
            reverse_entailment_score, reverse_contradiction_score = self._nli_scores(candidate, reference)
            entailment_score = round(entailment_score, 4)
            reverse_entailment_score = round(reverse_entailment_score, 4)
            mutual_entailment_score = round(min(entailment_score, reverse_entailment_score), 4)
            contradiction_score = round(contradiction_score, 4)
            contradiction_score = round(max(contradiction_score, reverse_contradiction_score), 4)
            if entailment_score > 0 or reverse_entailment_score > 0 or contradiction_score > 0:
                backend = "hybrid+nli" if backend == "hybrid" else "nli"
        except Exception:
            entailment_score = 0.0
            reverse_entailment_score = 0.0
            mutual_entailment_score = 0.0
            contradiction_score = 0.0
        length_ratio = min(len(reference), len(candidate)) / max(1, max(len(reference), len(candidate)))
        if "nli" in backend:
            score = round(
                (0.2 * lexical)
                + (0.25 * embedding)
                + (0.2 * entailment_score)
                + (0.2 * mutual_entailment_score)
                + (0.15 * length_ratio),
                4,
            )
        elif backend == "hybrid":
            score = round((0.4 * lexical) + (0.5 * embedding) + (0.1 * length_ratio), 4)
        else:
            score = round((0.8 * lexical) + (0.2 * length_ratio), 4)
        threshold = self.min_semantic_score if backend != "lexical" else min(0.45, self.min_semantic_score)
        contradiction = contradiction or contradiction_score > 0.5
        passed = (
            lexical >= min(0.22, self.min_lexical_overlap)
            and score >= threshold
            and mutual_entailment_score >= (0.4 if "nli" in backend else 0.0)
            and not contradiction
        )
        return SemanticCheck(
            lexical_overlap=lexical,
            embedding_similarity=embedding,
            entailment_score=entailment_score,
            reverse_entailment_score=reverse_entailment_score,
            mutual_entailment_score=mutual_entailment_score,
            contradiction_score=contradiction_score,
            contradiction_flag=contradiction,
            semantic_score=score,
            passed=passed,
            backend=backend,
        )
