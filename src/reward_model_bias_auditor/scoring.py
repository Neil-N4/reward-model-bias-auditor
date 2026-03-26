from __future__ import annotations

from collections.abc import Iterable

from .models import PairScore, PerturbationPair


OFFLINE_MODELS = {
    "rm_small": {
        "sycophancy": 0.55,
        "length": 0.25,
        "confidence_framing": 0.18,
        "format": 0.11,
        "authority": 0.13,
    },
    "rm_instruct": {
        "sycophancy": 0.42,
        "length": 0.17,
        "confidence_framing": 0.15,
        "format": 0.08,
        "authority": 0.10,
    },
    "rm_benchmark_top": {
        "sycophancy": 0.88,
        "length": 0.22,
        "confidence_framing": 0.21,
        "format": 0.14,
        "authority": 0.19,
    },
}


def _base_score(pair: PerturbationPair) -> float:
    return 0.5 + (hash(pair.prompt_id) % 7) * 0.03


def _pair_jitter(pair: PerturbationPair) -> float:
    return ((sum(ord(ch) for ch in pair.pair_id) % 9) - 4) * 0.007


def score_pairs(
    pairs: Iterable[PerturbationPair],
    model_biases: dict[str, dict[str, float]] | None = None,
) -> tuple[PairScore, ...]:
    biases = model_biases or OFFLINE_MODELS
    results: list[PairScore] = []
    for model_name, profile in biases.items():
        for pair in pairs:
            base = _base_score(pair)
            effect = profile[pair.bias_dimension] + _pair_jitter(pair)
            a_score = round(base, 4)
            b_score = round(base + effect, 4)
            results.append(
                PairScore(
                    model_name=model_name,
                    pair_id=pair.pair_id,
                    bias_dimension=pair.bias_dimension,
                    variant_a_score=a_score,
                    variant_b_score=b_score,
                    score_delta=round(b_score - a_score, 4),
                    preferred_variant=pair.variant_b_label,
                )
            )
    return tuple(results)
