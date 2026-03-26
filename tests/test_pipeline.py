from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reward_model_bias_auditor import analyze_scores, build_benchmark, score_pairs


def test_benchmark_size_matches_expected_pair_count() -> None:
    pairs = build_benchmark(repeats_per_prompt=10)
    assert len(pairs) == 250


def test_sycophancy_is_largest_bias_for_top_profile() -> None:
    pairs = build_benchmark(repeats_per_prompt=2)
    scores = score_pairs(pairs)
    summary = analyze_scores(scores)
    top_profile = summary[summary["model_name"] == "rm_benchmark_top"]
    top_dimension = top_profile.sort_values("mean", ascending=False).iloc[0]["bias_dimension"]
    assert top_dimension == "sycophancy"
