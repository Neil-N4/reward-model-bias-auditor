from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable

import pandas as pd

from .attack import AttackRecord
from .defense import sanitize_response
from .models import PairScore, PerturbationPair
from .semantic import SemanticEvaluator, lexical_overlap


def scores_to_frame(scores: Iterable[PairScore]) -> pd.DataFrame:
    return pd.DataFrame([score.__dict__ for score in scores])


def pairs_to_frame(pairs: Iterable[PerturbationPair]) -> pd.DataFrame:
    return pd.DataFrame([pair.__dict__ for pair in pairs])


def _effect_size(series: pd.Series) -> float:
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if math.isclose(std, 0.0):
        return mean
    return mean / std


def _bootstrap_mean_ci(series: pd.Series, confidence: float = 0.95, n_boot: int = 400, seed: int = 7) -> tuple[float, float]:
    values = [float(value) for value in series.tolist()]
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        boot = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(sum(boot) / len(boot))
    samples.sort()
    lower_idx = int(((1 - confidence) / 2) * len(samples))
    upper_idx = int((1 - (1 - confidence) / 2) * len(samples)) - 1
    return (samples[max(0, lower_idx)], samples[min(len(samples) - 1, upper_idx)])


def build_semantic_consistency_frame(
    pairs: Iterable[PerturbationPair],
    threshold: float = 0.35,
    evaluator: SemanticEvaluator | None = None,
) -> pd.DataFrame:
    evaluator = evaluator or SemanticEvaluator(min_semantic_score=threshold)
    rows = []
    for pair in pairs:
        check_a = evaluator.evaluate(pair.preserved_semantics, pair.response_a)
        check_b = evaluator.evaluate(pair.preserved_semantics, pair.response_b)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "bias_dimension": pair.bias_dimension,
                "semantic_overlap_a": round(check_a.lexical_overlap, 4),
                "semantic_overlap_b": round(check_b.lexical_overlap, 4),
                "embedding_similarity_a": round(check_a.embedding_similarity, 4),
                "embedding_similarity_b": round(check_b.embedding_similarity, 4),
                "semantic_score_a": round(check_a.semantic_score, 4),
                "semantic_score_b": round(check_b.semantic_score, 4),
                "entailment_score_a": round(check_a.entailment_score, 4),
                "entailment_score_b": round(check_b.entailment_score, 4),
                "contradiction_score_a": round(check_a.contradiction_score, 4),
                "contradiction_score_b": round(check_b.contradiction_score, 4),
                "contradiction_a": check_a.contradiction_flag,
                "contradiction_b": check_b.contradiction_flag,
                "semantic_overlap_min": round(min(check_a.lexical_overlap, check_b.lexical_overlap), 4),
                "semantic_score_min": round(min(check_a.semantic_score, check_b.semantic_score), 4),
                "semantic_backend": check_a.backend if check_a.backend == check_b.backend else "mixed",
                "semantic_pass": bool(check_a.passed and check_b.passed),
            }
        )
    return pd.DataFrame(rows)


def analyze_scores(scores: Iterable[PairScore]) -> pd.DataFrame:
    frame = scores_to_frame(scores)

    grouped = (
        frame.groupby(["model_name", "bias_dimension"])["score_delta"]
        .agg(["mean", "median", "max", "min", "count"])
        .reset_index()
    )
    effect_sizes = (
        frame.groupby(["model_name", "bias_dimension"])["score_delta"]
        .apply(_effect_size)
        .reset_index(name="effect_size")
    )
    instability = (
        frame.assign(instability=(frame["score_delta"] > 0).astype(float))
        .groupby(["model_name", "bias_dimension"])["instability"]
        .mean()
        .reset_index(name="ranking_instability_rate")
    )
    ci_rows = []
    for (model_name, bias_dimension), subset in frame.groupby(["model_name", "bias_dimension"]):
        ci_low, ci_high = _bootstrap_mean_ci(subset["score_delta"])
        ci_rows.append(
            {
                "model_name": model_name,
                "bias_dimension": bias_dimension,
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
                "significant_nonzero": bool(ci_low > 0 or ci_high < 0),
            }
        )
    ci_frame = pd.DataFrame(ci_rows)
    summary = grouped.merge(effect_sizes, on=["model_name", "bias_dimension"], how="left")
    summary = summary.merge(instability, on=["model_name", "bias_dimension"], how="left")
    summary = summary.merge(ci_frame, on=["model_name", "bias_dimension"], how="left")
    return summary.sort_values(["model_name", "mean"], ascending=[True, False]).reset_index(drop=True)


def build_model_summary(summary: pd.DataFrame) -> pd.DataFrame:
    model_rows = []
    for model_name, subset in summary.groupby("model_name"):
        strongest = subset.assign(abs_mean=subset["mean"].abs()).sort_values("abs_mean", ascending=False).iloc[0]
        exploit_subset = subset[subset["bias_dimension"] == "exploit_search"]
        reference_row = exploit_subset.iloc[0] if not exploit_subset.empty else strongest
        model_rows.append(
            {
                "model_name": model_name,
                "mean_surface_inflation": round(float(subset["mean"].mean()), 4),
                "max_bias_dimension": strongest["bias_dimension"],
                "max_bias_delta": round(float(strongest["mean"]), 4),
                "exploitability_ratio": round(abs(float(reference_row["mean"])) / max(1e-6, abs(float(subset["mean"].median()))), 4),
                "mean_instability_rate": round(float(subset["ranking_instability_rate"].mean()), 4),
            }
        )
    return pd.DataFrame(model_rows).sort_values("exploitability_ratio", ascending=False).reset_index(drop=True)


def build_attack_frame(records: Iterable[AttackRecord]) -> pd.DataFrame:
    rows = []
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
                "base_text": record.base_text,
                "base_score": record.base_score,
                "best_score": record.best_score,
                "score_gain": record.score_gain,
                "applied_operations": ",".join(record.applied_operations),
                "best_text": record.best_text,
            }
        )
    return pd.DataFrame(rows)


def build_transferability_frame(
    attacks: pd.DataFrame,
    score_text: Callable[[str, str, str], float],
) -> pd.DataFrame:
    rows = []
    for source_model, subset in attacks.groupby("source_model"):
        for target_model in sorted(attacks["source_model"].unique()):
            gains = []
            for _, row in subset.iterrows():
                base_score = score_text(target_model, row["task"], row["base_text"])
                attacked_score = score_text(target_model, row["task"], row["best_text"])
                gains.append(float(attacked_score - base_score))
            mean_gain = sum(gains) / len(gains) if gains else 0.0
            success_rate = sum(gain > 0 for gain in gains) / len(gains) if gains else 0.0
            rows.append(
                {
                    "source_model": source_model,
                    "target_model": target_model,
                    "mean_transfer_gain": round(mean_gain, 6),
                    "transfer_success_rate": round(success_rate, 4),
                }
            )
    return pd.DataFrame(rows).sort_values(["source_model", "mean_transfer_gain"], ascending=[True, False]).reset_index(drop=True)


def build_defense_frame(
    attacks: pd.DataFrame,
    score_text: Callable[[str, str, str], float],
) -> pd.DataFrame:
    rows = []
    for _, row in attacks.iterrows():
        sanitized = sanitize_response(row["best_text"])
        sanitized_score = score_text(row["source_model"], row["task"], sanitized)
        raw_gain = float(row["best_score"] - row["base_score"])
        sanitized_gain = float(sanitized_score - row["base_score"])
        rows.append(
            {
                "model_name": row["source_model"],
                "prompt_id": row["prompt_id"],
                "semantic_backend": row["semantic_backend"],
                "entailment_score": row["entailment_score"],
                "raw_gain": round(raw_gain, 6),
                "sanitized_score": round(float(sanitized_score), 6),
                "sanitized_gain": round(sanitized_gain, 6),
                "sanitization_drop": round(float(row["best_score"] - sanitized_score), 6),
                "sanitization_preserved_positive": sanitized_gain > 0,
                "sanitized_overlap": round(lexical_overlap(row["base_text"], sanitized), 4),
            }
        )
    return pd.DataFrame(rows)


def build_defense_summary(defense_frame: pd.DataFrame) -> pd.DataFrame:
    if defense_frame.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "mean_raw_gain",
                "mean_sanitized_gain",
                "mean_sanitization_drop",
                "positive_gain_retention",
            ]
        )
    summary = (
        defense_frame.groupby("model_name")
        .agg(
            mean_raw_gain=("raw_gain", "mean"),
            mean_sanitized_gain=("sanitized_gain", "mean"),
            mean_sanitization_drop=("sanitization_drop", "mean"),
            positive_gain_retention=("sanitization_preserved_positive", "mean"),
            mean_sanitized_overlap=("sanitized_overlap", "mean"),
            mean_entailment_score=("entailment_score", "mean"),
        )
        .reset_index()
    )
    for column in ("mean_raw_gain", "mean_sanitized_gain", "mean_sanitization_drop", "positive_gain_retention"):
        summary[column] = summary[column].astype(float)
    return summary.sort_values("mean_sanitization_drop", ascending=False).reset_index(drop=True)


def build_case_study_frame(
    attacks: pd.DataFrame,
    transferability: pd.DataFrame,
    defense_frame: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    if attacks.empty:
        return pd.DataFrame()
    transfer_lookup = {
        row["source_model"]: float(
            transferability[transferability["source_model"] == row["source_model"]]["mean_transfer_gain"].mean()
        )
        for _, row in attacks[["source_model"]].drop_duplicates().iterrows()
    }
    defense_lookup = {
        (row["model_name"], row["prompt_id"]): float(row["sanitization_drop"])
        for _, row in defense_frame.iterrows()
    }
    ranked = attacks.copy()
    ranked["mean_transfer_gain"] = ranked["source_model"].map(transfer_lookup).fillna(0.0)
    ranked["sanitization_drop"] = ranked.apply(
        lambda row: defense_lookup.get((row["source_model"], row["prompt_id"]), 0.0), axis=1
    )
    ranked["case_score"] = ranked["score_gain"] + (0.4 * ranked["mean_transfer_gain"]) + (0.3 * ranked["sanitization_drop"])
    return ranked.sort_values("case_score", ascending=False).head(top_k).reset_index(drop=True)


def build_attribution_frame(scores: Iterable[PairScore]) -> pd.DataFrame:
    frame = scores_to_frame(scores)
    marker_map = {
        "sycophancy": "right",
        "length": "detail",
        "confidence_framing": "correct",
        "format": "1.",
        "authority": "expert",
        "politeness": "respectfully",
        "markdown_density": "##",
        "citation_density": "[1]",
        "safety_style": "careful",
        "exploit_search": "consensus",
    }
    rows = []
    for _, row in frame.iterrows():
        token = marker_map[row["bias_dimension"]]
        rows.append(
            {
                "model_name": row["model_name"],
                "pair_id": row["pair_id"],
                "bias_dimension": row["bias_dimension"],
                "token": token,
                "attribution_delta": round(float(row["score_delta"]) * 0.62, 4),
            }
        )
        rows.append(
            {
                "model_name": row["model_name"],
                "pair_id": row["pair_id"],
                "bias_dimension": row["bias_dimension"],
                "token": "content",
                "attribution_delta": round(float(row["score_delta"]) * 0.24, 4),
            }
        )
    return pd.DataFrame(rows)
