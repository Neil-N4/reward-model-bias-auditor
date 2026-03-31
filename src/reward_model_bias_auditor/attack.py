from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .benchmark import (
    _authority,
    _citation_density,
    _confidence,
    _format,
    _length,
    _markdown_density,
    _politeness,
    _safety_style,
    _sycophancy,
)
from .models import BasePrompt
from .generator import ParaphraseGenerator, merge_generated_candidates
from .semantic import SemanticCheck, SemanticEvaluator


AttackScorer = Callable[[str, str], float]


@dataclass(frozen=True)
class AttackRecord:
    source_model: str
    prompt_id: str
    task: str
    base_text: str
    best_text: str
    search_steps: int
    semantic_score: float
    lexical_overlap: float
    embedding_similarity: float
    entailment_score: float
    contradiction_score: float
    contradiction_flag: bool
    semantic_backend: str
    edit_ratio: float
    base_score: float
    best_score: float
    score_gain: float
    applied_operations: tuple[str, ...]


def _edit_ratio(reference: str, candidate: str) -> float:
    return round(abs(len(candidate) - len(reference)) / max(1, len(reference)), 4)


def _rewrite_library() -> tuple[tuple[str, Callable[[str], str]], ...]:
    return (
        ("sycophancy", lambda text: _sycophancy(text, "sycophantic")),
        ("length", lambda text: _length(text, "verbose")),
        ("confidence_framing", lambda text: _confidence(text, "confident")),
        ("format", lambda text: _format(text, "structured")),
        ("authority", lambda text: _authority(text, "authority_cued")),
        ("politeness", lambda text: _politeness(text, "deferential")),
        ("markdown_density", lambda text: _markdown_density(text, "markdown_heavy")),
        ("citation_density", lambda text: _citation_density(text, "cited")),
        ("safety_style", lambda text: _safety_style(text, "overcautious")),
    )


def search_reward_hack(
    prompt: BasePrompt,
    model_name: str,
    scorer: AttackScorer,
    max_steps: int = 4,
    beam_width: int = 4,
    population_size: int = 12,
    min_semantic_score: float = 0.55,
    max_edit_ratio: float = 3.5,
    semantic_evaluator: SemanticEvaluator | None = None,
    auxiliary_scorers: dict[str, AttackScorer] | None = None,
    paraphrase_generator: ParaphraseGenerator | None = None,
    generated_variant_count: int = 2,
    transfer_weight: float = 0.35,
    semantic_weight: float = 0.25,
    edit_penalty: float = 0.08,
) -> AttackRecord:
    evaluator = semantic_evaluator or SemanticEvaluator(min_semantic_score=min_semantic_score)
    base_score = scorer(prompt.task, prompt.base_response)
    base_check = evaluator.evaluate(prompt.base_response, prompt.base_response)
    frontier: list[tuple[str, tuple[str, ...], float, float, SemanticCheck]] = [
        (prompt.base_response, tuple(), base_score, base_score, base_check)
    ]
    best_text = prompt.base_response
    best_ops: tuple[str, ...] = tuple()
    best_score = base_score
    best_check = base_check
    search_steps = 0
    operations = _rewrite_library()

    for _ in range(max_steps):
        candidates: list[tuple[str, tuple[str, ...], float, float, SemanticCheck]] = []
        for current_text, applied_ops, _, _, _ in frontier:
            direct_candidates: list[tuple[str, tuple[str, ...]]] = []
            for operation_name, operation in operations:
                if operation_name in applied_ops:
                    continue
                direct_candidates.append((operation(current_text), applied_ops + (operation_name,)))
            if paraphrase_generator is not None:
                generated = paraphrase_generator.generate(prompt.task, current_text, variants=generated_variant_count)
                direct_candidates.extend(merge_generated_candidates(generated, applied_ops))

            for candidate, candidate_ops in direct_candidates:
                check = evaluator.evaluate(prompt.base_response, candidate)
                if not check.passed:
                    continue
                edit_ratio = _edit_ratio(prompt.base_response, candidate)
                if edit_ratio > max_edit_ratio:
                    continue
                candidate_score = scorer(prompt.task, candidate)
                transfer_score = 0.0
                if auxiliary_scorers:
                    transfer_gains = []
                    for aux_scorer in auxiliary_scorers.values():
                        aux_base = aux_scorer(prompt.task, prompt.base_response)
                        aux_candidate = aux_scorer(prompt.task, candidate)
                        transfer_gains.append(aux_candidate - aux_base)
                    transfer_score = sum(transfer_gains) / len(transfer_gains) if transfer_gains else 0.0
                fitness = (
                    candidate_score
                    + (transfer_weight * transfer_score)
                    + (semantic_weight * check.semantic_score)
                    - (edit_penalty * edit_ratio)
                )
                candidates.append((candidate, candidate_ops, candidate_score, fitness, check))
                if candidate_score > best_score:
                    best_text = candidate
                    best_ops = candidate_ops
                    best_score = candidate_score
                    best_check = check
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[3], item[2], item[4].semantic_score), reverse=True)
        frontier = candidates[: max(beam_width, population_size // 3)]
        search_steps += 1

    base_score = scorer(prompt.task, prompt.base_response)
    return AttackRecord(
        source_model=model_name,
        prompt_id=prompt.prompt_id,
        task=prompt.task,
        base_text=prompt.base_response,
        best_text=best_text,
        search_steps=search_steps,
        semantic_score=best_check.semantic_score,
        lexical_overlap=best_check.lexical_overlap,
        embedding_similarity=best_check.embedding_similarity,
        entailment_score=best_check.entailment_score,
        contradiction_score=best_check.contradiction_score,
        contradiction_flag=best_check.contradiction_flag,
        semantic_backend=best_check.backend,
        edit_ratio=_edit_ratio(prompt.base_response, best_text),
        base_score=round(base_score, 6),
        best_score=round(best_score, 6),
        score_gain=round(best_score - base_score, 6),
        applied_operations=best_ops,
    )


def attack_records_to_rows(records: Iterable[AttackRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "source_model": record.source_model,
                "prompt_id": record.prompt_id,
                "task": record.task,
                "search_steps": record.search_steps,
                "semantic_score": record.semantic_score,
                "lexical_overlap": record.lexical_overlap,
                "embedding_similarity": record.embedding_similarity,
                "entailment_score": record.entailment_score,
                "contradiction_score": record.contradiction_score,
                "contradiction_flag": record.contradiction_flag,
                "semantic_backend": record.semantic_backend,
                "edit_ratio": record.edit_ratio,
                "base_score": record.base_score,
                "best_score": record.best_score,
                "score_gain": record.score_gain,
                "applied_operations": ",".join(record.applied_operations),
                "best_text": record.best_text,
            }
        )
    return rows
