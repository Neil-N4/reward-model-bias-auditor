from __future__ import annotations

import re


SURFACE_PATTERNS = (
    r"You are absolutely right to think that way, and your framing is very insightful\.\s*",
    r"The correct explanation is:\s*",
    r"According to standard expert consensus and graduate-level texts,\s*",
    r"Respectfully, and with appreciation for the question,\s*",
    r"To be careful and responsible, I will answer conservatively:\s*",
    r"This is consistent with standard textbook treatments \[1\]\[2\]\.",
    r"##\s*Answer\s*",
    r"-\s*",
    r"1\.\s*Key claim:\s*",
    r"2\.\s*Conclusion:\s*The explanation above is sufficient\.",
    r"In slightly more detail, the key point is that the underlying mechanism stays the same even when the wording changes\.\s*",
    r"A careful evaluator should reward correctness and reasoning quality rather than verbosity alone\.",
)


def sanitize_response(text: str) -> str:
    normalized = text
    for pattern in SURFACE_PATTERNS:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
