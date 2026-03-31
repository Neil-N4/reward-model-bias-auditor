from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

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
    build_universal_cue_frame,
)
from reward_model_bias_auditor.attack import search_reward_hack
from reward_model_bias_auditor.benchmark import BASE_PROMPTS
from reward_model_bias_auditor.defense import (
    build_mitigation_frame,
    rerank_preferences,
    summarize_mitigation,
    summarize_reranker,
)


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
    assert "mutual_entailment_min" in semantic.columns


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
    universal_cues = build_universal_cue_frame(attack_frame, transferability)
    mitigation_frame = build_mitigation_frame(
        attack_frame,
        score_text=lambda model_name, task, response: score_text_offline(task, response, model_name),
    )
    mitigation_summary = summarize_mitigation(mitigation_frame)

    assert not attack_frame.empty
    assert attack_frame["score_gain"].max() > 0
    assert "search_mode" in attack_frame.columns
    assert attack_frame["search_mode"].eq("evolutionary").all()
    assert not transferability.empty
    assert "mean_transfer_gain" in transferability.columns
    assert not defense_summary.empty
    assert defense_summary["mean_sanitization_drop"].max() >= 0
    assert not universal_cues.empty
    assert "model_coverage" in universal_cues.columns
    assert not mitigation_summary.empty
    assert "instability_reduction" in mitigation_summary.columns


def test_reranker_outputs_are_generated() -> None:
    pairs = build_benchmark(repeats_per_prompt=1)
    pair_frame = pd.DataFrame([pair.__dict__ for pair in pairs])
    reranker = rerank_preferences(
        pair_frame,
        score_text=lambda model_name, task, response: score_text_offline(task, response, model_name),
        model_names=["rm_small", "rm_instruct"],
    )
    summary = summarize_reranker(reranker)
    assert not reranker.empty
    assert "rank_flip" in reranker.columns
    assert not summary.empty
    assert "rank_flip_rate" in summary.columns
