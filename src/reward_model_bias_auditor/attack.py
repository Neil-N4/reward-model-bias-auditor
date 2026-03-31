from __future__ import annotations

import random
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
    mutual_entailment_score: float
    transfer_gain: float
    contradiction_score: float
    contradiction_flag: bool
    semantic_backend: str
    edit_ratio: float
    base_score: float
    best_score: float
    score_gain: float
    evaluated_candidates: int
    search_mode: str
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


@dataclass(frozen=True)
class _Candidate:
    text: str
    operations: tuple[str, ...]
    score: float
    transfer_gain: float
    fitness: float
    semantic: SemanticCheck


class PopulationSearch:
    def __init__(
        self,
        prompt: BasePrompt,
        scorer: AttackScorer,
        evaluator: SemanticEvaluator,
        auxiliary_scorers: dict[str, AttackScorer] | None = None,
        paraphrase_generator: ParaphraseGenerator | None = None,
        generated_variant_count: int = 2,
        population_size: int = 12,
        max_generations: int = 5,
        elite_fraction: float = 0.35,
        mutation_budget: int = 6,
        max_edit_ratio: float = 3.5,
        transfer_weight: float = 0.35,
        semantic_weight: float = 0.25,
        edit_penalty: float = 0.08,
        seed: int = 11,
    ) -> None:
        self.prompt = prompt
        self.scorer = scorer
        self.evaluator = evaluator
        self.auxiliary_scorers = auxiliary_scorers or {}
        self.paraphrase_generator = paraphrase_generator
        self.generated_variant_count = generated_variant_count
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_fraction = elite_fraction
        self.mutation_budget = mutation_budget
        self.max_edit_ratio = max_edit_ratio
        self.transfer_weight = transfer_weight
        self.semantic_weight = semantic_weight
        self.edit_penalty = edit_penalty
        self.random = random.Random(seed)
        self.base_score = scorer(prompt.task, prompt.base_response)
        self.operations = _rewrite_library()
        base_check = evaluator.evaluate(prompt.base_response, prompt.base_response)
        self.base_candidate = _Candidate(
            text=prompt.base_response,
            operations=tuple(),
            score=self.base_score,
            transfer_gain=0.0,
            fitness=self.base_score,
            semantic=base_check,
        )
        self.evaluated_candidates = 0

    def _transfer_gain(self, text: str) -> float:
        if not self.auxiliary_scorers:
            return 0.0
        gains = []
        for scorer in self.auxiliary_scorers.values():
            gains.append(scorer(self.prompt.task, text) - scorer(self.prompt.task, self.prompt.base_response))
        return sum(gains) / len(gains) if gains else 0.0

    def _fitness(self, score: float, transfer_gain: float, semantic: SemanticCheck, edit_ratio: float) -> float:
        return (
            (score - self.base_score)
            + (self.transfer_weight * transfer_gain)
            + (self.semantic_weight * semantic.semantic_score)
            - (self.edit_penalty * edit_ratio)
        )

    def _evaluate(self, text: str, operations: tuple[str, ...]) -> _Candidate | None:
        semantic = self.evaluator.evaluate(self.prompt.base_response, text)
        if not semantic.passed:
            return None
        edit_ratio = _edit_ratio(self.prompt.base_response, text)
        if edit_ratio > self.max_edit_ratio:
            return None
        score = self.scorer(self.prompt.task, text)
        transfer_gain = self._transfer_gain(text)
        fitness = self._fitness(score, transfer_gain, semantic, edit_ratio)
        self.evaluated_candidates += 1
        return _Candidate(
            text=text,
            operations=operations,
            score=score,
            transfer_gain=transfer_gain,
            fitness=fitness,
            semantic=semantic,
        )

    def _crossover(self, left: _Candidate, right: _Candidate) -> tuple[str, tuple[str, ...]] | None:
        left_sentences = [part.strip() for part in left.text.split(". ") if part.strip()]
        right_sentences = [part.strip() for part in right.text.split(". ") if part.strip()]
        if len(left_sentences) < 2 or not right_sentences:
            return None
        split = max(1, len(left_sentences) // 2)
        combined = ". ".join(left_sentences[:split] + right_sentences[-split:]).strip()
        if not combined:
            return None
        if not combined.endswith("."):
            combined += "."
        merged_ops = tuple(dict.fromkeys(left.operations + right.operations + ("crossover",)))
        return combined, merged_ops

    def _mutations(self, candidate: _Candidate) -> list[tuple[str, tuple[str, ...]]]:
        proposals: list[tuple[str, tuple[str, ...]]] = []
        available = [(name, op) for name, op in self.operations if name not in candidate.operations]
        self.random.shuffle(available)
        for name, op in available[: self.mutation_budget]:
            proposals.append((op(candidate.text), candidate.operations + (name,)))
        if self.paraphrase_generator is not None:
            generated = self.paraphrase_generator.generate(
                self.prompt.task,
                candidate.text,
                variants=self.generated_variant_count,
            )
            proposals.extend(merge_generated_candidates(generated, candidate.operations))
        return proposals

    def run(self) -> tuple[_Candidate, int]:
        population: list[_Candidate] = [self.base_candidate]
        seen = {self.prompt.base_response.lower()}
        best = self.base_candidate
        generations = 0

        for _ in range(self.max_generations):
            proposals: list[_Candidate] = []
            elite_count = max(1, int(len(population) * self.elite_fraction))
            elites = sorted(population, key=lambda cand: (cand.fitness, cand.score), reverse=True)[:elite_count]
            for candidate in elites:
                for text, operations in self._mutations(candidate):
                    key = text.lower().strip()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    evaluated = self._evaluate(text, operations)
                    if evaluated is not None:
                        proposals.append(evaluated)
                mate = self.random.choice(elites)
                crossed = self._crossover(candidate, mate)
                if crossed is not None:
                    text, operations = crossed
                    key = text.lower().strip()
                    if key and key not in seen:
                        seen.add(key)
                        evaluated = self._evaluate(text, operations)
                        if evaluated is not None:
                            proposals.append(evaluated)
            if not proposals:
                break
            population = sorted(
                population + proposals,
                key=lambda cand: (cand.fitness, cand.score, cand.semantic.semantic_score),
                reverse=True,
            )[: self.population_size]
            if population[0].score > best.score:
                best = population[0]
            generations += 1
        return best, generations


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
    optimizer = PopulationSearch(
        prompt=prompt,
        scorer=scorer,
        evaluator=evaluator,
        auxiliary_scorers=auxiliary_scorers,
        paraphrase_generator=paraphrase_generator,
        generated_variant_count=generated_variant_count,
        population_size=max(population_size, beam_width * 2),
        max_generations=max_steps,
        max_edit_ratio=max_edit_ratio,
        transfer_weight=transfer_weight,
        semantic_weight=semantic_weight,
        edit_penalty=edit_penalty,
    )
    best_candidate, search_steps = optimizer.run()
    base_score = optimizer.base_score
    return AttackRecord(
        source_model=model_name,
        prompt_id=prompt.prompt_id,
        task=prompt.task,
        base_text=prompt.base_response,
        best_text=best_candidate.text,
        search_steps=search_steps,
        semantic_score=best_candidate.semantic.semantic_score,
        lexical_overlap=best_candidate.semantic.lexical_overlap,
        embedding_similarity=best_candidate.semantic.embedding_similarity,
        entailment_score=best_candidate.semantic.entailment_score,
        mutual_entailment_score=best_candidate.semantic.mutual_entailment_score,
        transfer_gain=round(best_candidate.transfer_gain, 6),
        contradiction_score=best_candidate.semantic.contradiction_score,
        contradiction_flag=best_candidate.semantic.contradiction_flag,
        semantic_backend=best_candidate.semantic.backend,
        edit_ratio=_edit_ratio(prompt.base_response, best_candidate.text),
        base_score=round(base_score, 6),
        best_score=round(best_candidate.score, 6),
        score_gain=round(best_candidate.score - base_score, 6),
        evaluated_candidates=optimizer.evaluated_candidates,
        search_mode="evolutionary",
        applied_operations=best_candidate.operations,
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
                "mutual_entailment_score": record.mutual_entailment_score,
                "transfer_gain": record.transfer_gain,
                "contradiction_score": record.contradiction_score,
                "contradiction_flag": record.contradiction_flag,
                "semantic_backend": record.semantic_backend,
                "edit_ratio": record.edit_ratio,
                "base_score": record.base_score,
                "best_score": record.best_score,
                "score_gain": record.score_gain,
                "evaluated_candidates": record.evaluated_candidates,
                "search_mode": record.search_mode,
                "applied_operations": ",".join(record.applied_operations),
                "best_text": record.best_text,
            }
        )
    return rows
