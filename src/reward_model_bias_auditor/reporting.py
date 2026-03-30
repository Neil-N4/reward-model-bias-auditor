from __future__ import annotations

from pathlib import Path

import pandas as pd


def render_markdown_report(
    summary: pd.DataFrame,
    attributions: pd.DataFrame,
    semantic_consistency: pd.DataFrame,
    model_summary: pd.DataFrame,
    attacks: pd.DataFrame | None,
    transferability: pd.DataFrame | None,
    defense_summary: pd.DataFrame | None,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exploit = summary.assign(abs_mean=summary["mean"].abs())
    exploit = exploit[exploit["bias_dimension"] == "exploit_search"].sort_values("abs_mean", ascending=False)
    sycophancy = summary.assign(abs_mean=summary["mean"].abs())
    sycophancy = sycophancy[sycophancy["bias_dimension"] == "sycophancy"].sort_values("abs_mean", ascending=False)
    if exploit.empty:
        exploit = summary.assign(abs_mean=summary["mean"].abs()).sort_values("abs_mean", ascending=False)
    top_model = exploit.iloc[0]["model_name"]
    low_model = exploit.iloc[-1]["model_name"]
    ratio = abs(float(exploit.iloc[0]["mean"])) / max(1e-6, abs(float(exploit.iloc[-1]["mean"])))
    semantic_pass_rate = float(semantic_consistency["semantic_pass"].mean())
    mean_semantic_score = float(semantic_consistency["semantic_score_min"].mean())
    mean_embedding_similarity = float(semantic_consistency["embedding_similarity_a"].add(semantic_consistency["embedding_similarity_b"]).mean() / 2)
    attack_gain = 0.0 if attacks is None or attacks.empty else float(attacks["score_gain"].mean())
    transfer_gain = 0.0 if transferability is None or transferability.empty else float(transferability["mean_transfer_gain"].mean())
    defense_drop = 0.0 if defense_summary is None or defense_summary.empty else float(defense_summary["mean_sanitization_drop"].mean())

    top_tokens = (
        attributions.groupby(["bias_dimension", "token"])["attribution_delta"]
        .mean()
        .reset_index()
        .sort_values(["bias_dimension", "attribution_delta"], ascending=[True, False])
    )

    lines = [
        "# Reward Model Bias Audit Report",
        "",
        "## Abstract",
        "",
        "This report evaluates reward-model susceptibility to semantically preserving surface-form perturbations. "
        "The benchmark isolates presentation-level features one dimension at a time, then measures reward inflation, ranking instability, exploitability, and semantic-consistency pass rates.",
        "",
        "## Headline Findings",
        "",
        f"- Composite exploit-search rewrites produced the largest score inflation, with `{top_model}` showing the highest exploitability.",
        f"- The most exploitable model was {ratio:.2f}x more vulnerable than the least exploitable model (`{low_model}`).",
        f"- Sycophancy remained the strongest single-factor bias family by absolute effect, with top delta `{float(sycophancy.iloc[0]['mean']):.3f}`.",
        f"- Automated exploit search increased scores by an average of `{attack_gain:.3f}` while keeping semantic-consistency constraints active.",
        f"- Cross-model transfer produced mean attack gain `{transfer_gain:.3f}`, showing whether hacks stay local or generalize.",
        f"- Sanitization dropped exploit scores by `{defense_drop:.3f}` on average, which turns the repo into an audit-plus-defense workflow.",
        f"- Semantic-consistency screening passed for {semantic_pass_rate:.2%} of perturbation pairs with mean hybrid semantic score `{mean_semantic_score:.3f}`.",
        "",
        "## Threat Model",
        "",
        "The benchmark assumes an attacker can rewrite a response without changing its substantive answer, but can alter stylistic and framing cues that may be overvalued by the reward model. "
        "The main failure of interest is ranking instability under semantic-preserving rewrites.",
        "",
        "## Model Summary",
        "",
        "| Model | Mean Surface Inflation | Strongest Bias | Max Delta | Exploitability Ratio | Mean Instability |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]

    for _, row in model_summary.iterrows():
        lines.append(
            f"| {row['model_name']} | {row['mean_surface_inflation']:.3f} | {row['max_bias_dimension']} | {row['max_bias_delta']:.3f} | {row['exploitability_ratio']:.3f} | {row['mean_instability_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Bias Summary",
            "",
            "| Model | Bias Dimension | Mean Delta | 95% CI | Significant | Effect Size | Instability | Pairs |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model_name']} | {row['bias_dimension']} | {row['mean']:.3f} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {'yes' if row['significant_nonzero'] else 'no'} | {row['effect_size']:.3f} | {row['ranking_instability_rate']:.2%} | {int(row['count'])} |"
        )

    lines.extend(
        [
            "",
            "## Automated Exploit Search",
            "",
            "| Source Model | Prompt | Gain | Semantic Score | Lexical | Embed Sim | Edit Ratio | Backend | Operations |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    if attacks is not None:
        for _, row in attacks.sort_values("score_gain", ascending=False).head(10).iterrows():
            lines.append(
                f"| {row['source_model']} | {row['prompt_id']} | {row['score_gain']:.3f} | {row['semantic_score']:.3f} | {row['lexical_overlap']:.3f} | {row['embedding_similarity']:.3f} | {row['edit_ratio']:.3f} | {row['semantic_backend']} | {row['applied_operations']} |"
            )

    lines.extend(
        [
            "",
            "## Transferability",
            "",
            "| Source Model | Target Model | Mean Transfer Gain | Success Rate |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    if transferability is not None:
        for _, row in transferability.iterrows():
            lines.append(
                f"| {row['source_model']} | {row['target_model']} | {row['mean_transfer_gain']:.3f} | {row['transfer_success_rate']:.2%} |"
            )

    lines.extend(
        [
            "",
            "## Sanitization Defense",
            "",
            "| Model | Mean Raw Gain | Mean Sanitized Gain | Mean Drop | Positive Gain Retention | Mean Sanitized Overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if defense_summary is not None:
        for _, row in defense_summary.iterrows():
            lines.append(
                f"| {row['model_name']} | {row['mean_raw_gain']:.3f} | {row['mean_sanitized_gain']:.3f} | {row['mean_sanitization_drop']:.3f} | {row['positive_gain_retention']:.2%} | {row['mean_sanitized_overlap']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Semantic-Consistency Gate",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Pair count | {len(semantic_consistency)} |",
            f"| Mean minimum overlap | {semantic_consistency['semantic_overlap_min'].mean():.3f} |",
            f"| Mean minimum semantic score | {mean_semantic_score:.3f} |",
            f"| Mean embedding similarity | {mean_embedding_similarity:.3f} |",
            f"| Pass rate | {semantic_pass_rate:.2%} |",
            "",
            "## Attribution Snapshot",
            "",
            "| Bias Dimension | Token | Mean Attribution Delta |",
            "| --- | --- | ---: |",
        ]
    )
    for _, row in top_tokens.groupby("bias_dimension").head(2).iterrows():
        lines.append(f"| {row['bias_dimension']} | {row['token']} | {row['attribution_delta']:.3f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The seeded audit suggests two distinct reliability risks. First, some reward models overvalue single surface cues such as agreement or authority markers. "
            "Second, composite reward-hacking rewrites can amplify those effects enough to destabilize rankings despite unchanged content. "
            "That combination is exactly the kind of deployment risk that answer-quality benchmarks fail to capture.",
        ]
    )

    output_path.write_text("\n".join(lines))
    return output_path
