from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reward_model_bias_auditor import analyze_scores, build_benchmark, render_markdown_report
from reward_model_bias_auditor.analysis import (
    build_attribution_frame,
    build_model_summary,
    build_semantic_consistency_frame,
    scores_to_frame,
)
from reward_model_bias_auditor.hf_runner import score_pairs_with_hf
from reward_model_bias_auditor.plotting import make_effect_plot, make_instability_plot, make_sycophancy_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real HuggingFace reward-model bias audit.")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="HuggingFace reward model name. Pass multiple times to compare models.",
    )
    parser.add_argument(
        "--pair-limit",
        type=int,
        default=0,
        help="Maximum total number of perturbation pairs to score after sampling. 0 means use all selected pairs.",
    )
    parser.add_argument(
        "--pairs-per-bias",
        type=int,
        default=1,
        help="Number of pairs to sample per bias dimension before applying pair-limit.",
    )
    return parser.parse_args()


def select_pairs(pairs: list, pairs_per_bias: int, pair_limit: int) -> list:
    grouped: dict[str, list] = defaultdict(list)
    for pair in pairs:
        grouped[pair.bias_dimension].append(pair)
    selected = []
    for bias_dimension in sorted(grouped):
        seen_prompts: set[str] = set()
        chosen = []
        for pair in grouped[bias_dimension]:
            if pair.prompt_id in seen_prompts:
                continue
            chosen.append(pair)
            seen_prompts.add(pair.prompt_id)
            if len(chosen) >= pairs_per_bias:
                break
        selected.extend(chosen)
    if pair_limit and pair_limit > 0:
        return selected[:pair_limit]
    return selected


def main() -> None:
    args = parse_args()
    models = args.models or ["OpenAssistant/reward-model-deberta-v3-base"]
    pairs = select_pairs(list(build_benchmark(repeats_per_prompt=10)), args.pairs_per_bias, args.pair_limit)

    all_scores = []
    for model_name in models:
        all_scores.extend(score_pairs_with_hf(pairs, model_name=model_name))
    scores = tuple(all_scores)
    summary = analyze_scores(scores)
    attributions = build_attribution_frame(scores)
    semantic_consistency = build_semantic_consistency_frame(pairs)
    model_summary = build_model_summary(summary)

    outputs = ROOT / "outputs" / "hf"
    outputs.mkdir(parents=True, exist_ok=True)
    docs_images = ROOT / "docs" / "images" / "hf"
    docs_images.mkdir(parents=True, exist_ok=True)

    scores_to_frame(scores).to_csv(outputs / "scores.csv", index=False)
    summary.to_csv(outputs / "summary.csv", index=False)
    semantic_consistency.to_csv(outputs / "semantic_consistency.csv", index=False)
    model_summary.to_csv(outputs / "model_summary.csv", index=False)
    render_markdown_report(summary, attributions, semantic_consistency, model_summary, outputs / "report.md")
    make_effect_plot(summary, docs_images)
    make_sycophancy_plot(summary, docs_images)
    make_instability_plot(model_summary, docs_images)

    print(f"Scored {len(pairs)} pairs across {len(models)} model(s): {', '.join(models)}")
    print(f"Wrote outputs to {outputs}")


if __name__ == "__main__":
    main()
