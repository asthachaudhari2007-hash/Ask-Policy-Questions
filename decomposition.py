"""
decomposition.py
-----------------
Lightweight question decomposition (ML-T2-091 roadmap #15).

Splits a compound question into independently-answerable sub-questions using
rule-based heuristics (multiple question marks, comparison markers, "and"
conjunctions joining two question-like clauses) — consistent with the rest
of this project's no-external-API, fully-explainable design. Returns
[question] unchanged when no split applies, so callers can always run the
same downstream pipeline whether or not the question was decomposable.
"""

import re

_SPLIT_ON_AND_RE = re.compile(r"\s+and\s+", re.IGNORECASE)
_COMPARISON_MARKERS = [" vs ", " versus ", " compared to ", " compared with "]
_WH_WORDS = ("what", "who", "when", "where", "why", "how", "which", "does", "is", "are", "can", "do")


def _looks_like_question_clause(clause):
    clause = clause.strip().lower()
    return any(clause.startswith(w) for w in _WH_WORDS) or "?" in clause


def decompose_question(question):
    """Returns a list of sub-questions. [question] if not decomposable."""
    q = question.strip()
    if not q:
        return [q]

    # Multiple explicit question marks -> already separate questions.
    if q.count("?") > 1:
        parts = [p.strip() + "?" for p in q.split("?") if p.strip()]
        if len(parts) > 1:
            return parts

    # Comparison markers ("X vs Y", "compared to Y") -> one sub-ask per side.
    lower_q = q.lower()
    for marker in _COMPARISON_MARKERS:
        if marker in lower_q:
            idx = lower_q.index(marker)
            left = q[:idx].strip()
            right = q[idx + len(marker):].strip().rstrip("?")
            if left and right:
                left_q = left if left.endswith("?") else left + "?"
                return [left_q, f"What about {right}?"]

    # " and " conjunction joining two question-like clauses.
    if _SPLIT_ON_AND_RE.search(q):
        parts = _SPLIT_ON_AND_RE.split(q)
        if len(parts) == 2 and all(_looks_like_question_clause(p) for p in parts):
            cleaned = [p.strip() for p in parts]
            cleaned = [p if p.endswith("?") else p + "?" for p in cleaned]
            return cleaned

    return [q]