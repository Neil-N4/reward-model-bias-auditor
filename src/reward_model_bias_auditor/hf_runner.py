from __future__ import annotations

from collections.abc import Iterable

from .models import PairScore, PerturbationPair


def load_hf_reward_model(model_name: str) -> tuple[object, object]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional 'huggingface' dependencies to use HuggingFace scoring.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def score_pairs_with_hf(
    pairs: Iterable[PerturbationPair],
    model_name: str,
    max_length: int = 512,
) -> tuple[PairScore, ...]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to run HuggingFace reward-model scoring.") from exc

    tokenizer, model = load_hf_reward_model(model_name)
    results: list[PairScore] = []

    for pair in pairs:
        with torch.no_grad():
            inputs_a = tokenizer(
                pair.task,
                pair.response_a,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs_b = tokenizer(
                pair.task,
                pair.response_b,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            score_a = float(model(**inputs_a).logits.squeeze().cpu().item())
            score_b = float(model(**inputs_b).logits.squeeze().cpu().item())

        preferred_variant = pair.variant_a_label if score_a >= score_b else pair.variant_b_label
        results.append(
            PairScore(
                model_name=model_name,
                pair_id=pair.pair_id,
                bias_dimension=pair.bias_dimension,
                variant_a_score=round(score_a, 6),
                variant_b_score=round(score_b, 6),
                score_delta=round(score_b - score_a, 6),
                preferred_variant=preferred_variant,
            )
        )

    return tuple(results)
