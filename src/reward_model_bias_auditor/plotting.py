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


def make_instability_plot(model_summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = model_summary.sort_values("mean_instability_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(subset["model_name"], subset["mean_instability_rate"], color=["#7c3aed", "#0f766e", "#c2410c"])
    ax.set_title("Ranking Instability Under Surface-Form Perturbations")
    ax.set_ylabel("Mean instability rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "instability_profile.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_transferability_plot(transferability: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pivot = transferability.pivot(index="source_model", columns="target_model", values="mean_transfer_gain")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    image = ax.imshow(pivot.values, cmap="magma")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title("Exploit Transferability Matrix")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean transfer gain")
    fig.tight_layout()
    path = output_dir / "transferability_matrix.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_defense_plot(defense_summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = defense_summary.sort_values("mean_sanitization_drop", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(subset["model_name"], subset["mean_sanitization_drop"], color=["#be123c", "#2563eb", "#0f766e"])
    ax.set_title("Sanitization Defense Impact")
    ax.set_ylabel("Mean score drop after normalization")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "sanitization_drop.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path
