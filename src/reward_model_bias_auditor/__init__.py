"""Reward model bias auditor package."""

from .benchmark import build_benchmark
from .analysis import analyze_scores
from .scoring import score_pairs
from .reporting import render_markdown_report

__all__ = ["analyze_scores", "build_benchmark", "render_markdown_report", "score_pairs"]
