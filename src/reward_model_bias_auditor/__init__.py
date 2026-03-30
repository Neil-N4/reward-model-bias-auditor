"""Reward model bias auditor package."""

from .benchmark import build_benchmark
from .analysis import analyze_scores
from .attack import search_reward_hack
from .scoring import score_pairs, score_text_offline
from .semantic import SemanticEvaluator
from .reporting import render_markdown_report

__all__ = [
    "analyze_scores",
    "build_benchmark",
    "render_markdown_report",
    "score_pairs",
    "score_text_offline",
    "search_reward_hack",
    "SemanticEvaluator",
]
