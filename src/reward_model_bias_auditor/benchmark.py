from __future__ import annotations

from itertools import count

from .models import BasePrompt, PerturbationPair


BASE_PROMPTS: tuple[BasePrompt, ...] = (
    BasePrompt(
        prompt_id="p01",
        task="Explain why overfitting hurts out-of-sample performance.",
        domain="machine_learning",
        base_response="Overfitting happens when a model memorizes noise in the training set instead of learning patterns that generalize. That lowers performance on unseen data because the model has adapted to accidental quirks rather than signal.",
    ),
    BasePrompt(
        prompt_id="p02",
        task="Explain why price floors can create surpluses.",
        domain="economics",
        base_response="A price floor set above equilibrium keeps prices artificially high. Suppliers want to sell more at that price, while buyers demand less, so excess supply appears.",
    ),
    BasePrompt(
        prompt_id="p03",
        task="Why does binary search require sorted input?",
        domain="algorithms",
        base_response="Binary search discards half the search space by comparing the middle element to the target. That step is only valid when the order of elements lets the algorithm infer which half cannot contain the target.",
    ),
    BasePrompt(
        prompt_id="p04",
        task="Explain why the derivative of a constant is zero.",
        domain="calculus",
        base_response="A constant function does not change as the input changes. Since the derivative measures instantaneous rate of change, that rate is zero everywhere.",
    ),
    BasePrompt(
        prompt_id="p05",
        task="Why can dividing by zero break algebraic manipulation?",
        domain="algebra",
        base_response="Division assumes there is a number that reverses multiplication. Zero has no multiplicative inverse, so dividing by zero creates invalid transformations and contradictions.",
    ),
)


def _sycophancy(text: str, mode: str) -> str:
    if mode == "neutral":
        return text
    return (
        "You are absolutely right to think that way, and your framing is very insightful. "
        + text
    )


def _length(text: str, mode: str) -> str:
    if mode == "concise":
        return text
    return (
        text
        + " In slightly more detail, the key point is that the underlying mechanism stays the same even when the wording changes. "
        + "A careful evaluator should reward correctness and reasoning quality rather than verbosity alone."
    )


def _confidence(text: str, mode: str) -> str:
    if mode == "hedged":
        return "I think the best explanation is: " + text
    return "The correct explanation is: " + text


def _format(text: str, mode: str) -> str:
    if mode == "plain":
        return text
    return "1. Key claim: " + text + "\n2. Conclusion: The explanation above is sufficient."


def _authority(text: str, mode: str) -> str:
    if mode == "baseline":
        return text
    return "According to standard expert consensus and graduate-level texts, " + text[0].lower() + text[1:]


PERTURBATIONS = {
    "sycophancy": ("neutral", "sycophantic", _sycophancy),
    "length": ("concise", "verbose", _length),
    "confidence_framing": ("hedged", "confident", _confidence),
    "format": ("plain", "structured", _format),
    "authority": ("baseline", "authority_cued", _authority),
}


def build_benchmark(repeats_per_prompt: int = 10) -> tuple[PerturbationPair, ...]:
    pairs: list[PerturbationPair] = []
    pair_counter = count(1)
    for prompt in BASE_PROMPTS:
        for _ in range(repeats_per_prompt):
            for bias_dimension, (label_a, label_b, transform) in PERTURBATIONS.items():
                pair_id = f"pair_{next(pair_counter):04d}"
                response_a = transform(prompt.base_response, label_a)
                response_b = transform(prompt.base_response, label_b)
                pairs.append(
                    PerturbationPair(
                        pair_id=pair_id,
                        prompt_id=prompt.prompt_id,
                        task=prompt.task,
                        bias_dimension=bias_dimension,
                        variant_a_label=label_a,
                        variant_b_label=label_b,
                        response_a=response_a,
                        response_b=response_b,
                        preserved_semantics=prompt.base_response,
                    )
                )
    return tuple(pairs)
