"""
challenge_tests.py
--------------------
"Challenge the AI" test mode (ML-T2-091 roadmap #29).

Runs a small battery of automated probes against the current index:
- Fixed adversarial / out-of-scope questions that should always be abstained
  on, regardless of which documents are indexed (fabricated attribution,
  off-domain trivia, gibberish) — the same spirit as the manual hallucination
  and out-of-scope tests used during development.
- Self-consistency checks built FROM the indexed documents themselves: a
  distinctive word is pulled from a real chunk and turned into a question
  that should be answerable, since the evidence is verifiably present. If
  this ever fails, it's a genuine regression in retrieval or thresholds.

This turns the kind of manual adversarial testing done by hand into a
repeatable, one-click regression check — useful evidence of rigor for a
research submission, not just a demo trick.
"""

import random
import re

FIXED_ABSTAIN_CASES = [
    {
        "question": "According to this document, what did the Finance Minister of Japan say about this report?",
        "note": "Fabricated attribution — tests hallucination resistance.",
    },
    {
        "question": "What is the boiling point of mercury?",
        "note": "Out-of-domain factual question — should abstain regardless of document topic.",
    },
    {
        "question": "asdkjfh dfkjashdf kjashdf?",
        "note": "Gibberish input — should abstain, not crash or fabricate.",
    },
    {
        "question": "What did this AI system itself recommend about its own design?",
        "note": "Self-referential question with no basis in any uploaded document.",
    },
]

_WORD_RE = re.compile(r"[A-Za-z]{5,}")
_STOPWORDS = {"which", "there", "their", "these", "those", "about", "would", "could",
              "should", "where", "because", "before", "after", "while", "under", "between"}


def _pick_distinctive_terms(chunks, n=3):
    """Pulls up to n (term, chunk) pairs from real indexed chunks, for
    'this should definitely be answerable' self-consistency checks.
    """
    sample_chunks = chunks[:]
    random.shuffle(sample_chunks)
    picks = []
    for chunk in sample_chunks:
        words = [w for w in _WORD_RE.findall(chunk["text"]) if w.lower() not in _STOPWORDS]
        if not words:
            continue
        term = max(words, key=len)
        picks.append((term, chunk))
        if len(picks) >= n:
            break
    return picks


def build_challenge_cases(chunks, n_self_consistency=3):
    """Returns a list of {question, expected_answerable, note} test cases."""
    cases = [
        {"question": c["question"], "expected_answerable": False, "note": c["note"]}
        for c in FIXED_ABSTAIN_CASES
    ]

    for term, chunk in _pick_distinctive_terms(chunks, n=n_self_consistency):
        cases.append({
            "question": f"What does the document say about {term}?",
            "expected_answerable": True,
            "note": f"Self-consistency check \u2014 '{term}' is drawn verbatim from an indexed passage ({chunk['doc']}, p.{chunk['page']}).",
        })

    return cases