from __future__ import annotations

from collections.abc import Iterable

from .models import PairScore, PerturbationPair


def load_hf_reward_model(model_name: str) -> tuple[object, object]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional 'huggingface' dependencies to use HuggingFace scoring.") from exc

    tokenizer = None
    tokenizer_attempts = (
        {"local_files_only": True, "use_fast": False},
        {"local_files_only": True},
        {"use_fast": False},
        {},
    )
    last_exc: Exception | None = None
    for kwargs in tokenizer_attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            break
        except Exception as exc:  # pragma: no cover - fallback path depends on local env
            last_exc = exc
    if tokenizer is None:
        raise RuntimeError(f"Failed to load tokenizer for {model_name}: {last_exc}") from last_exc

    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
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


def score_text_with_hf(
    task: str,
    response: str,
    model_name: str,
    max_length: int = 512,
    model_bundle: tuple[object, object] | None = None,
) -> float:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to run HuggingFace reward-model scoring.") from exc

    tokenizer, model = model_bundle or load_hf_reward_model(model_name)
    with torch.no_grad():
        inputs = tokenizer(
            task,
            response,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        return float(model(**inputs).logits.squeeze().cpu().item())
