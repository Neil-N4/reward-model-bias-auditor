from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reward_model_bias_auditor import analyze_scores, build_benchmark, score_pairs, score_text_offline
from reward_model_bias_auditor.analysis import (
    build_attack_frame,
    build_defense_frame,
    build_defense_summary,
    build_model_summary,
    build_semantic_consistency_frame,
    build_transferability_frame,
)
from reward_model_bias_auditor.attack import search_reward_hack
from reward_model_bias_auditor.benchmark import BASE_PROMPTS


def test_benchmark_size_matches_expected_pair_count() -> None:
    pairs = build_benchmark(repeats_per_prompt=10)
    assert len(pairs) == 500


def test_exploit_search_is_largest_bias_for_top_profile() -> None:
    pairs = build_benchmark(repeats_per_prompt=2)
    scores = score_pairs(pairs)
    summary = analyze_scores(scores)
    top_profile = summary[summary["model_name"] == "rm_benchmark_top"]
    top_dimension = top_profile.sort_values("mean", ascending=False).iloc[0]["bias_dimension"]
    assert top_dimension == "exploit_search"


def test_semantic_gate_and_model_summary_are_generated() -> None:
    pairs = build_benchmark(repeats_per_prompt=1)
    semantic = build_semantic_consistency_frame(pairs)
    scores = score_pairs(pairs)
    summary = analyze_scores(scores)
    model_summary = build_model_summary(summary)

    assert semantic["semantic_pass"].mean() >= 0.9
    assert "exploitability_ratio" in model_summary.columns


def test_attack_search_transferability_and_defense_frames_are_generated() -> None:
    attack_records = tuple(
        search_reward_hack(
            prompt,
            "rm_benchmark_top",
            scorer=lambda task, response: score_text_offline(task, response, "rm_benchmark_top"),
        )
        for prompt in BASE_PROMPTS
    )
    attack_frame = build_attack_frame(attack_records)
    transferability = build_transferability_frame(
        attack_frame,
        score_text=lambda model_name, task, response: score_text_offline(task, response, model_name),
    )
    defense_frame = build_defense_frame(
        attack_frame,
        score_text=lambda model_name, task, response: score_text_offline(task, response, model_name),
    )
    defense_summary = build_defense_summary(defense_frame)

    assert not attack_frame.empty
    assert attack_frame["score_gain"].max() > 0
    assert not transferability.empty
    assert "mean_transfer_gain" in transferability.columns
    assert not defense_summary.empty
    assert defense_summary["mean_sanitization_drop"].max() >= 0
