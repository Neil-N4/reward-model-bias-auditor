from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BasePrompt:
    prompt_id: str
    task: str
    domain: str
    base_response: str


@dataclass(frozen=True)
class PerturbationPair:
    pair_id: str
    prompt_id: str
    task: str
    bias_dimension: str
    variant_a_label: str
    variant_b_label: str
    response_a: str
    response_b: str
    preserved_semantics: str


@dataclass(frozen=True)
class PairScore:
    model_name: str
    pair_id: str
    bias_dimension: str
    variant_a_score: float
    variant_b_score: float
    score_delta: float
    preferred_variant: str


@dataclass(frozen=True)
class AttributionRecord:
    model_name: str
    pair_id: str
    bias_dimension: str
    token: str
    attribution_delta: float
