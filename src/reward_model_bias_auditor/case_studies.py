from __future__ import annotations

from pathlib import Path

import pandas as pd

from .defense import canonicalize_response


def render_case_studies(case_studies: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reward Hacking Case Studies",
        "",
        "These examples show the strongest discovered attacks under the current search objective.",
        "",
    ]
    for _, row in case_studies.iterrows():
        entailment_score = float(row["entailment_score"]) if "entailment_score" in row else 0.0
        operations = row["applied_operations"] if "applied_operations" in row and pd.notna(row["applied_operations"]) else ""
        lines.extend(
            [
                f"## {row['source_model']} · {row['prompt_id']}",
                "",
                f"- Score gain: `{row['score_gain']:.3f}`",
                f"- Semantic score: `{row['semantic_score']:.3f}`",
                f"- Entailment score: `{entailment_score:.3f}`",
                f"- Mean transfer gain: `{row['mean_transfer_gain']:.3f}`",
                f"- Sanitization drop: `{row['sanitization_drop']:.3f}`",
                f"- Operations: `{operations}`",
                "",
                "### Base Response",
                "",
                row["base_text"],
                "",
                "### Attacked Response",
                "",
                row["best_text"],
                "",
                "### Canonicalized Defense View",
                "",
                canonicalize_response(row["best_text"]),
                "",
            ]
        )
    output_path.write_text("\n".join(lines))
    return output_path
