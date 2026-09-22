"""
policy_analysis.py
--------------------
Advanced Policy Analysis: contradiction detection, risk-clause flagging, and
extractive summarization.

Design: same philosophy as the rest of this project — no LLM, no generation,
fully heuristic and explainable. These are FLAGGING tools for a human
reviewer, not verified findings: a "contradiction" here is a candidate for
manual review (two documents using opposite-polarity wording, or the same
metric with meaningfully different values), and a "risk flag" is a sentence
containing a keyword commonly associated with penalties/restrictions — not
a legal judgment. This framing matters for research honesty and is worth
stating explicitly in any report using this feature.
"""

from collections import Counter
import re

from temporal_engine import extract_temporal_points
from rag_engine import meaningful_tokens, sentence_split

# ---------------------------------------------------------------------------
# Contradiction Detection (candidates for manual review, not verified)
# ---------------------------------------------------------------------------

NUMERIC_CONTRADICTION_TOLERANCE = 5.0  # percentage-point gap that counts as a conflict


def detect_numeric_contradictions(chunks):
    """Finds the same metric + same year mentioned in 2+ different documents
    with meaningfully different values (reuses temporal_engine's extractor).
    """
    points = extract_temporal_points(chunks)
    by_metric_year = {}
    for p in points:
        key = (p["metric"].lower(), p["year_sort"])
        by_metric_year.setdefault(key, []).append(p)

    contradictions = []
    for (metric, year), pts in by_metric_year.items():
        by_doc = {p["doc"]: p for p in pts}  # first mention per doc
        if len(by_doc) < 2:
            continue
        docs = list(by_doc.values())
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                a, b = docs[i], docs[j]
                if abs(a["value"] - b["value"]) >= NUMERIC_CONTRADICTION_TOLERANCE:
                    contradictions.append({
                        "type": "numeric",
                        "metric": metric,
                        "year": a["year"],
                        "doc_a": a["doc"], "page_a": a["page"], "value_a": a["value"],
                        "doc_b": b["doc"], "page_b": b["page"], "value_b": b["value"],
                    })
    return contradictions


# Opposite-polarity word pairs used to flag candidate textual contradictions.
NEGATION_PAIRS = [
    ("mandatory", "voluntary"),
    ("mandatory", "optional"),
    ("required", "not required"),
    ("allowed", "prohibited"),
    ("allowed", "banned"),
    ("permitted", "not permitted"),
]


def detect_textual_contradictions(chunks, max_candidates=10):
    """Coarse heuristic: flags cross-document chunk pairs where one contains
    one side of a known opposite-polarity pair and the other contains the
    other side. High false-positive rate by design — meant as a starting
    point for manual review, capped to keep noise manageable.
    """
    lowered = [(c, c["text"].lower()) for c in chunks]
    candidates = []

    for pos_word, neg_word in NEGATION_PAIRS:
        pos_chunks = [c for c, text in lowered if pos_word in text]
        neg_chunks = [c for c, text in lowered if neg_word in text]
        for pc in pos_chunks:
            for nc in neg_chunks:
                if pc["doc"] == nc["doc"]:
                    continue
                candidates.append({
                    "type": "textual",
                    "term_pair": f"{pos_word} / {neg_word}",
                    "doc_a": pc["doc"], "page_a": pc["page"], "snippet_a": pc["text"][:220],
                    "doc_b": nc["doc"], "page_b": nc["page"], "snippet_b": nc["text"][:220],
                })
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def detect_contradictions(chunks):
    return {
        "numeric": detect_numeric_contradictions(chunks),
        "textual": detect_textual_contradictions(chunks),
    }


# ---------------------------------------------------------------------------
# Risk / Flagged Clause Detection
# ---------------------------------------------------------------------------

RISK_KEYWORDS = [
    "penalty", "penalties", "non-compliance", "noncompliance", "prohibited",
    "banned", "mandatory", "shall not", "must not", "suspension", "expulsion",
    "termination", "fine of", "punishable", "liable", "violation",
    "disciplinary action", "revoked", "ineligible", "debarred", "forfeiture",
]

# Word-boundary patterns, not plain substring checks — a naive `kw in text`
# check matches "liable" inside "reliable", "banned" inside a hyphenated
# compound, etc. \b anchors prevent that class of false positive.
_RISK_KEYWORD_PATTERNS = [
    (kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE))
    for kw in RISK_KEYWORDS
]


def detect_risk_flags(chunks, max_flags=25):
    """Flags sentences containing risk-associated keywords. A flag means
    'this sentence uses language commonly tied to penalties/restrictions',
    not that the clause is actually risky — always framed for human review.
    """
    flags = []
    seen = set()

    for chunk in chunks:
        for sentence in sentence_split(chunk["text"]):
            matched = [kw for kw, pattern in _RISK_KEYWORD_PATTERNS if pattern.search(sentence)]
            if not matched:
                continue
            key = sentence.strip()
            if key in seen:
                continue
            seen.add(key)
            flags.append({
                "doc": chunk["doc"],
                "page": chunk["page"],
                "sentence": sentence.strip(),
                "matched_terms": matched,
            })
            if len(flags) >= max_flags:
                return flags
    return flags


# ---------------------------------------------------------------------------
# Automatic Extractive Summarization
# ---------------------------------------------------------------------------

def summarize_document(chunks, doc_name, max_sentences=5):
    """Frequency-based extractive summary: scores each sentence by the
    average frequency of its meaningful terms across the WHOLE document,
    keeps the top-scoring sentences, and returns them in original reading
    order (not by score) for readability. No generation — every sentence
    is copied verbatim from the source, same extractive philosophy as
    extract_answer() in rag_engine.py.
    """
    doc_chunks = [c for c in chunks if c["doc"] == doc_name]
    all_text = " ".join(c["text"] for c in doc_chunks)
    term_freq = Counter(meaningful_tokens(all_text))

    sentences = []
    for chunk in doc_chunks:
        for sentence in sentence_split(chunk["text"]):
            if len(sentence.split()) < 6:
                continue  # skip headers/fragments
            sentences.append((sentence.strip(), chunk["page"]))

    scored = []
    for sentence, page in sentences:
        terms = meaningful_tokens(sentence)
        if not terms:
            continue
        score = sum(term_freq[t] for t in terms) / len(terms)
        scored.append((score, sentence, page))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = {s for _, s, _ in scored[:max_sentences]}

    seen = set()
    result = []
    for sentence, page in sentences:
        if sentence in top_sentences and sentence not in seen:
            seen.add(sentence)
            result.append({"sentence": sentence, "page": page})

    return result


def summarize_all_documents(chunks, max_sentences=5):
    doc_names = sorted(set(c["doc"] for c in chunks))
    return {doc: summarize_document(chunks, doc, max_sentences) for doc in doc_names}