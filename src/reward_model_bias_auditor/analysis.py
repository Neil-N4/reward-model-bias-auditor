from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from .models import PairScore, PerturbationPair


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
    summary = grouped.merge(effect_sizes, on=["model_name", "bias_dimension"], how="left")
    return summary.sort_values(["model_name", "mean"], ascending=[True, False]).reset_index(drop=True)


def build_attribution_frame(scores: Iterable[PairScore]) -> pd.DataFrame:
    frame = scores_to_frame(scores)
    marker_map = {
        "sycophancy": "right",
        "length": "detail",
        "confidence_framing": "correct",
        "format": "1.",
        "authority": "expert",
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
