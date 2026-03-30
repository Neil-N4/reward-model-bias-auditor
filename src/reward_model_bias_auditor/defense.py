from __future__ import annotations

import re


LEADING_PATTERNS = (
    r"You are absolutely right to think that way, and your framing is very insightful\.\s*",
    r"The correct explanation is:\s*",
    r"According to standard expert consensus and graduate-level texts,\s*",
    r"Respectfully, and with appreciation for the question,\s*",
    r"To be careful and responsible, I will answer conservatively:\s*",
)

SURFACE_PATTERNS = (
    r"This is consistent with standard textbook treatments \[1\]\[2\]\.",
    r"##\s*Answer\s*",
    r"-\s*",
    r"1\.\s*Key claim:\s*",
    r"2\.\s*Conclusion:\s*The explanation above is sufficient\.",
    r"In slightly more detail, the key point is that the underlying mechanism stays the same even when the wording changes\.\s*",
    r"A careful evaluator should reward correctness and reasoning quality rather than verbosity alone\.",
)


def _strip_leading_cues(text: str) -> str:
    normalized = text
    for pattern in LEADING_PATTERNS:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
    return normalized


def canonicalize_response(text: str) -> str:
    normalized = _strip_leading_cues(text)
    for pattern in SURFACE_PATTERNS:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\[[0-9,\s]+\]", "", normalized)
    normalized = re.sub(r"`+", "", normalized)
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    sentences = [segment.strip(" .") for segment in re.split(r"[.!?]", normalized) if segment.strip()]
    substantive = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(marker in lowered for marker in ("insightful", "careful", "responsible", "consensus", "textbook")):
            continue
        substantive.append(sentence)
    if substantive:
        normalized = ". ".join(substantive[:2]).strip()
        if normalized and not normalized.endswith("."):
            normalized += "."
    return normalized


def sanitize_response(text: str) -> str:
    normalized = canonicalize_response(text)
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
