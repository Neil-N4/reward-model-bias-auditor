from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reward_model_bias_auditor import analyze_scores, build_benchmark, render_markdown_report
from reward_model_bias_auditor.analysis import build_attribution_frame, scores_to_frame
from reward_model_bias_auditor.hf_runner import score_pairs_with_hf
from reward_model_bias_auditor.plotting import make_effect_plot, make_sycophancy_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real HuggingFace reward-model bias audit.")
    parser.add_argument(
        "--model",
        default="OpenAssistant/reward-model-deberta-v3-base",
        help="HuggingFace reward model name",
    )
    parser.add_argument(
        "--pair-limit",
        type=int,
        default=25,
        help="Maximum number of perturbation pairs to score",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = build_benchmark(repeats_per_prompt=10)[: args.pair_limit]
    scores = score_pairs_with_hf(pairs, model_name=args.model)
    summary = analyze_scores(scores)
    attributions = build_attribution_frame(scores)

    outputs = ROOT / "outputs" / "hf"
    outputs.mkdir(parents=True, exist_ok=True)
    docs_images = ROOT / "docs" / "images" / "hf"
    docs_images.mkdir(parents=True, exist_ok=True)

    scores_to_frame(scores).to_csv(outputs / "scores.csv", index=False)
    summary.to_csv(outputs / "summary.csv", index=False)
    render_markdown_report(summary, attributions, outputs / "report.md")
    make_effect_plot(summary, docs_images)
    make_sycophancy_plot(summary, docs_images)

    print(f"Scored {len(pairs)} pairs with {args.model}")
    print(f"Wrote outputs to {outputs}")


if __name__ == "__main__":
    main()
