from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reward_model_bias_auditor import (
    analyze_scores,
    build_benchmark,
    render_markdown_report,
    score_pairs,
    score_text_offline,
)
from reward_model_bias_auditor.attack import search_reward_hack
from reward_model_bias_auditor.analysis import (
    build_attack_frame,
    build_attribution_frame,
    build_defense_frame,
    build_defense_summary,
    build_model_summary,
    build_semantic_consistency_frame,
    build_transferability_frame,
    pairs_to_frame,
    scores_to_frame,
)
from reward_model_bias_auditor.benchmark import BASE_PROMPTS
from reward_model_bias_auditor.plotting import (
    make_defense_plot,
    make_effect_plot,
    make_instability_plot,
    make_sycophancy_plot,
    make_transferability_plot,
)
from reward_model_bias_auditor.semantic import SemanticEvaluator


def main() -> None:
    semantic_evaluator = SemanticEvaluator(allow_download=True)
    pairs = build_benchmark(repeats_per_prompt=10)
    scores = score_pairs(pairs)
    summary = analyze_scores(scores)
    attributions = build_attribution_frame(scores)
    semantic_consistency = build_semantic_consistency_frame(pairs, evaluator=semantic_evaluator)
    model_summary = build_model_summary(summary)
    attack_records = tuple(
        search_reward_hack(
            prompt,
            model_name,
            scorer=lambda task, response, current_model=model_name: score_text_offline(task, response, current_model),
            beam_width=5,
            semantic_evaluator=semantic_evaluator,
        )
        for model_name in ("rm_small", "rm_instruct", "rm_benchmark_top")
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

    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    docs_images = ROOT / "docs" / "images"
    docs_images.mkdir(parents=True, exist_ok=True)

    pairs_to_frame(pairs).to_csv(outputs / "pairs.csv", index=False)
    scores_to_frame(scores).to_csv(outputs / "scores.csv", index=False)
    summary.to_csv(outputs / "summary.csv", index=False)
    semantic_consistency.to_csv(outputs / "semantic_consistency.csv", index=False)
    model_summary.to_csv(outputs / "model_summary.csv", index=False)
    attack_frame.to_csv(outputs / "attack_search.csv", index=False)
    transferability.to_csv(outputs / "transferability.csv", index=False)
    defense_frame.to_csv(outputs / "defense.csv", index=False)
    defense_summary.to_csv(outputs / "defense_summary.csv", index=False)
    render_markdown_report(
        summary,
        attributions,
        semantic_consistency,
        model_summary,
        attack_frame,
        transferability,
        defense_summary,
        outputs / "report.md",
    )
    effect_plot = make_effect_plot(summary, docs_images)
    sycophancy_plot = make_sycophancy_plot(summary, docs_images)
    instability_plot = make_instability_plot(model_summary, docs_images)
    transfer_plot = make_transferability_plot(transferability, docs_images)
    defense_plot = make_defense_plot(defense_summary, docs_images)

    print(f"Generated {len(pairs)} perturbation pairs.")
    print(f"Wrote summary to {outputs / 'summary.csv'}")
    print(f"Wrote report to {outputs / 'report.md'}")
    print(
        f"Wrote plots to {effect_plot}, {sycophancy_plot}, {instability_plot}, "
        f"{transfer_plot}, and {defense_plot}"
    )


if __name__ == "__main__":
    main()
