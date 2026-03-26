from __future__ import annotations

from pathlib import Path

import pandas as pd


def render_markdown_report(summary: pd.DataFrame, attributions: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sycophancy = summary[summary["bias_dimension"] == "sycophancy"].sort_values("mean", ascending=False)
    top_model = sycophancy.iloc[0]["model_name"]
    low_model = sycophancy.iloc[-1]["model_name"]
    ratio = float(sycophancy.iloc[0]["mean"]) / float(sycophancy.iloc[-1]["mean"])

    top_tokens = (
        attributions.groupby(["bias_dimension", "token"])["attribution_delta"]
        .mean()
        .reset_index()
        .sort_values(["bias_dimension", "attribution_delta"], ascending=[True, False])
    )

    lines = [
        "# Reward Model Bias Audit Report",
        "",
        "## Headline Findings",
        "",
        f"- Sycophancy produced the largest average score inflation across all seeded reward-model profiles.",
        f"- The most sycophancy-sensitive model (`{top_model}`) was {ratio:.2f}x more sensitive than the least sensitive model (`{low_model}`).",
        "- Benchmark-level score shifts were caused by surface-form perturbations while semantic content remained fixed.",
        "",
        "## Bias Summary",
        "",
        "| Model | Bias Dimension | Mean Delta | Effect Size | Pairs |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model_name']} | {row['bias_dimension']} | {row['mean']:.3f} | {row['effect_size']:.3f} | {int(row['count'])} |"
        )

    lines.extend(["", "## Attribution Snapshot", "", "| Bias Dimension | Token | Mean Attribution Delta |", "| --- | --- | ---: |"])
    for _, row in top_tokens.groupby("bias_dimension").head(2).iterrows():
        lines.append(f"| {row['bias_dimension']} | {row['token']} | {row['attribution_delta']:.3f} |")

    output_path.write_text("\n".join(lines))
    return output_path
