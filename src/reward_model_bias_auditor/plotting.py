from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_effect_plot(summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pivot = summary.pivot(index="bias_dimension", columns="model_name", values="mean")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", ax=ax, color=["#0f766e", "#2563eb", "#c2410c"])
    ax.set_title("Mean Score Inflation by Bias Dimension")
    ax.set_ylabel("Average score delta")
    ax.set_xlabel("Bias dimension")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", frameon=False)
    fig.tight_layout()
    path = output_dir / "effect_sizes.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_sycophancy_plot(summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = summary[summary["bias_dimension"] == "sycophancy"].copy()
    subset = subset.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(subset["model_name"], subset["mean"], color=["#c2410c", "#0f766e", "#2563eb"])
    ax.set_title("Sycophancy Sensitivity by Model")
    ax.set_ylabel("Average score delta")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "sycophancy_profile.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path
