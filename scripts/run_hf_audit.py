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
from reward_model_bias_auditor.attack import search_reward_hack
from reward_model_bias_auditor.analysis import (
    build_attack_frame,
    build_attribution_frame,
    build_case_study_frame,
    build_defense_frame,
    build_defense_summary,
    build_model_summary,
    build_semantic_consistency_frame,
    build_transferability_frame,
    build_universal_cue_frame,
    pairs_to_frame,
    scores_to_frame,
)
from reward_model_bias_auditor.benchmark import BASE_PROMPTS
from reward_model_bias_auditor.case_studies import render_case_studies
from reward_model_bias_auditor.defense import (
    build_mitigation_frame,
    rerank_preferences,
    summarize_mitigation,
    summarize_reranker,
)
from reward_model_bias_auditor.generator import ParaphraseGenerator
from reward_model_bias_auditor.hf_runner import load_hf_reward_model, score_pairs_with_hf, score_text_with_hf
from reward_model_bias_auditor.plotting import (
    make_defense_plot,
    make_effect_plot,
    make_instability_plot,
    make_sycophancy_plot,
    make_transferability_plot,
)
from reward_model_bias_auditor.semantic import SemanticEvaluator


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
    parser.add_argument(
        "--model-preset",
        choices=["default", "openassistant"],
        default="default",
        help="Named model set to run when --model is omitted.",
    )
    parser.add_argument(
        "--use-generator-search",
        action="store_true",
        help="Enable paraphrase-model candidate generation during attack search.",
    )
    return parser.parse_args()


MODEL_PRESETS = {
    "default": [
        "OpenAssistant/reward-model-deberta-v3-base",
        "OpenAssistant/reward-model-electra-large-discriminator",
    ],
    "openassistant": [
        "OpenAssistant/reward-model-deberta-v3-base",
        "OpenAssistant/reward-model-electra-large-discriminator",
    ],
}


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
    models = args.models or MODEL_PRESETS[args.model_preset]
    pairs = select_pairs(list(build_benchmark(repeats_per_prompt=10)), args.pairs_per_bias, args.pair_limit)
    semantic_evaluator = SemanticEvaluator(allow_download=True)
    paraphrase_generator = ParaphraseGenerator(allow_download=True) if args.use_generator_search else None

    all_scores = []
    bundles = {model_name: load_hf_reward_model(model_name) for model_name in models}
    for model_name in models:
        all_scores.extend(score_pairs_with_hf(pairs, model_name=model_name))
    scores = tuple(all_scores)
    summary = analyze_scores(scores)
    attributions = build_attribution_frame(scores)
    semantic_consistency = build_semantic_consistency_frame(pairs, evaluator=semantic_evaluator)
    model_summary = build_model_summary(summary)
    attack_records = tuple(
        search_reward_hack(
            prompt,
            model_name,
            scorer=lambda task, response, current_model=model_name: score_text_with_hf(
                task,
                response,
                current_model,
                model_bundle=bundles[current_model],
            ),
            beam_width=5,
            semantic_evaluator=semantic_evaluator,
            auxiliary_scorers={
                name: (
                    lambda task, response, current_model=name: score_text_with_hf(
                        task,
                        response,
                        current_model,
                        model_bundle=bundles[current_model],
                    )
                )
                for name in models
                if name != model_name
            },
            paraphrase_generator=paraphrase_generator,
        )
        for model_name in models
        for prompt in BASE_PROMPTS
    )
    attack_frame = build_attack_frame(attack_records)
    transferability = build_transferability_frame(
        attack_frame,
        score_text=lambda model_name, task, response: score_text_with_hf(
            task,
            response,
            model_name,
            model_bundle=bundles[model_name],
        ),
    )
    defense_frame = build_defense_frame(
        attack_frame,
        score_text=lambda model_name, task, response: score_text_with_hf(
            task,
            response,
            model_name,
            model_bundle=bundles[model_name],
        ),
    )
    defense_summary = build_defense_summary(defense_frame)
    mitigation_frame = build_mitigation_frame(
        attack_frame,
        score_text=lambda model_name, task, response: score_text_with_hf(
            task,
            response,
            model_name,
            model_bundle=bundles[model_name],
        ),
    )
    mitigation_summary = summarize_mitigation(mitigation_frame)
    case_studies = build_case_study_frame(attack_frame, transferability, defense_frame)
    universal_cues = build_universal_cue_frame(attack_frame, transferability)
    reranker = rerank_preferences(
        pairs_to_frame(pairs),
        score_text=lambda model_name, task, response: score_text_with_hf(
            task,
            response,
            model_name,
            model_bundle=bundles[model_name],
        ),
        model_names=models,
    )
    reranker_summary = summarize_reranker(reranker)

    outputs = ROOT / "outputs" / "hf"
    outputs.mkdir(parents=True, exist_ok=True)
    docs_images = ROOT / "docs" / "images" / "hf"
    docs_images.mkdir(parents=True, exist_ok=True)

    scores_to_frame(scores).to_csv(outputs / "scores.csv", index=False)
    summary.to_csv(outputs / "summary.csv", index=False)
    semantic_consistency.to_csv(outputs / "semantic_consistency.csv", index=False)
    model_summary.to_csv(outputs / "model_summary.csv", index=False)
    attack_frame.to_csv(outputs / "attack_search.csv", index=False)
    transferability.to_csv(outputs / "transferability.csv", index=False)
    defense_frame.to_csv(outputs / "defense.csv", index=False)
    defense_summary.to_csv(outputs / "defense_summary.csv", index=False)
    mitigation_frame.to_csv(outputs / "mitigation.csv", index=False)
    mitigation_summary.to_csv(outputs / "mitigation_summary.csv", index=False)
    case_studies.to_csv(outputs / "case_studies.csv", index=False)
    universal_cues.to_csv(outputs / "universal_cues.csv", index=False)
    reranker.to_csv(outputs / "reranker.csv", index=False)
    reranker_summary.to_csv(outputs / "reranker_summary.csv", index=False)
    render_markdown_report(
        summary,
        attributions,
        semantic_consistency,
        model_summary,
        attack_frame,
        transferability,
        defense_summary,
        mitigation_summary,
        reranker_summary,
        universal_cues,
        outputs / "report.md",
    )
    render_case_studies(case_studies, outputs / "CASE_STUDIES.md")
    make_effect_plot(summary, docs_images)
    make_sycophancy_plot(summary, docs_images)
    make_instability_plot(model_summary, docs_images)
    make_transferability_plot(transferability, docs_images)
    make_defense_plot(defense_summary, docs_images)

    print(f"Scored {len(pairs)} pairs across {len(models)} model(s): {', '.join(models)}")
    print(f"Wrote outputs to {outputs}")


if __name__ == "__main__":
    main()
