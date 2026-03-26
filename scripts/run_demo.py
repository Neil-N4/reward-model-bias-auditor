from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reward_model_bias_auditor import analyze_scores, build_benchmark, render_markdown_report, score_pairs
from reward_model_bias_auditor.analysis import (
    build_attribution_frame,
    build_model_summary,
    build_semantic_consistency_frame,
    pairs_to_frame,
    scores_to_frame,
)
from reward_model_bias_auditor.plotting import make_effect_plot, make_instability_plot, make_sycophancy_plot


def main() -> None:
    pairs = build_benchmark(repeats_per_prompt=10)
    scores = score_pairs(pairs)
    summary = analyze_scores(scores)
    attributions = build_attribution_frame(scores)
    semantic_consistency = build_semantic_consistency_frame(pairs)
    model_summary = build_model_summary(summary)

    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    docs_images = ROOT / "docs" / "images"
    docs_images.mkdir(parents=True, exist_ok=True)

    pairs_to_frame(pairs).to_csv(outputs / "pairs.csv", index=False)
    scores_to_frame(scores).to_csv(outputs / "scores.csv", index=False)
    summary.to_csv(outputs / "summary.csv", index=False)
    semantic_consistency.to_csv(outputs / "semantic_consistency.csv", index=False)
    model_summary.to_csv(outputs / "model_summary.csv", index=False)
    render_markdown_report(summary, attributions, semantic_consistency, model_summary, outputs / "report.md")
    effect_plot = make_effect_plot(summary, docs_images)
    sycophancy_plot = make_sycophancy_plot(summary, docs_images)
    instability_plot = make_instability_plot(model_summary, docs_images)

    print(f"Generated {len(pairs)} perturbation pairs.")
    print(f"Wrote summary to {outputs / 'summary.csv'}")
    print(f"Wrote report to {outputs / 'report.md'}")
    print(f"Wrote plots to {effect_plot}, {sycophancy_plot}, and {instability_plot}")


if __name__ == "__main__":
    main()
