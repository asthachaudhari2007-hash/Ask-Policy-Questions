"""
rag_engine.py
--------------
Implements the core of ML-T2-091: retrieval + evidence-aware answerability.

Design choices (mirroring the Stage 1 Research Brief):
- No internet / no external embedding API is assumed available, so "dense retrieval"
  is approximated with TF-IDF + cosine similarity (brief's Approach B), and BM25
  (Approach A) is implemented from scratch as the lexical baseline. This keeps
  indexing and retrieval fast (no model download, no encode step) at some cost
  to matching paraphrases that share no vocabulary with the query.
- The answerability decision (Approach D, "hybrid evidence-aware classifier") is
  currently a threshold-based hybrid of retrieval score + query-term coverage.
  This is intentionally a placeholder: the brief explicitly says thresholds are
  hypotheses to be tested once labeled data exists. Swap `decide_answerability`
  for a trained classifier later without touching the rest of the pipeline.
- Answers are extractive (exact sentences from the source), so there is no
  hallucination risk and no LLM dependency.
- Confidence labels and the decision trace make the answerability gate's
  reasoning inspectable step-by-step, rather than a single opaque bool.
"""

import os
import re
import glob
import math
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROCESSED_FOLDER = "data/processed"

# --- Tunable "hypotheses" (see brief section 10 — thresholds to be validated) ---
CHUNK_WORDS = 140
CHUNK_OVERLAP = 30
TOP_K = 5
SCORE_THRESHOLD = 0.18       # min hybrid retrieval score to even consider answerable
COVERAGE_THRESHOLD = 0.45    # min fraction of query terms that must appear in top evidence
MARGIN_MIN = 0.03            # top score must beat the rest by at least this much

# Hybrid retrieval mix: lexical (BM25) vs lexical-weighted (TF-IDF/cosine).
TFIDF_WEIGHT = 0.5
BM25_WEIGHT = 0.5

STOPWORDS = set("""a an the is are was were be been being of in on at to for from by with
about as it this that these those and or if then than but not no can may might should
would could will shall do does did have has had i you he she they we what when where why
how which who whom your our their his her its there here""".split())

# Question-framing / meta words: these describe HOW a question is asked, not
# WHAT content it needs. Treating them as "required evidence" unfairly
# penalizes coverage for correctly-matched passages (e.g. "What does the
# acronym GER stand for?" shouldn't require the passage to contain the word
# "acronym" or "stand"). Kept separate from STOPWORDS for clarity, but merged
# into the same filtering set used by meaningful_tokens().
META_QUESTION_WORDS = set("""acronym abbreviation stand stands mean means meaning
according report document text mentioned mention state states says say term terms
refer refers referred referring define defines defined definition explain explains
explained describe describes described""".split())

STOPWORDS = STOPWORDS | META_QUESTION_WORDS

_word_re = re.compile(r"[a-zA-Z]+")


def tokenize(text):
    return [w.lower() for w in _word_re.findall(text) if len(w) > 1]


def meaningful_tokens(text):
    return [w for w in tokenize(text) if w not in STOPWORDS]


# ---------------------------------------------------------------------------
# Parsing + chunking
# ---------------------------------------------------------------------------

def parse_pages(txt_path):
    """Splits a processed .txt file (with '--- PAGE n ---' markers) into (page_num, text)."""
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = re.split(r"--- PAGE (\d+) ---", content)
    pages = []
    # parts = ['', '1', text1, '2', text2, ...]
    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        text = parts[i + 1].strip()
        if text:
            pages.append((page_num, text))
    return pages


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_words - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def sentence_split(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


# ---------------------------------------------------------------------------
# BM25 (from scratch — no external package needed)
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, tokenized_corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = tokenized_corpus
        self.N = len(tokenized_corpus)
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0
        self.df = Counter()
        self.term_freqs = []
        for doc in tokenized_corpus:
            tf = Counter(doc)
            self.term_freqs.append(tf)
            for term in tf:
                self.df[term] += 1
        self.idf = {
            term: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

    def get_scores(self, query_tokens):
        scores = np.zeros(self.N)
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf_doc in enumerate(self.term_freqs):
                f = tf_doc.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


def _minmax(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Index over all processed documents
# ---------------------------------------------------------------------------

class DocumentIndex:
    def __init__(self):
        self.chunks = []  # list of dicts: {doc, page, text}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bm25 = None
        self.tokenized_chunks = []

    def is_empty(self):
        return len(self.chunks) == 0

    def build(self, processed_folder=PROCESSED_FOLDER):
        self.chunks = []
        for txt_path in sorted(glob.glob(os.path.join(processed_folder, "*.txt"))):
            doc_name = os.path.splitext(os.path.basename(txt_path))[0]
            for page_num, page_text in parse_pages(txt_path):
                for chunk in chunk_text(page_text):
                    self.chunks.append({
                        "doc": doc_name,
                        "page": page_num,
                        "text": chunk,
                    })

        if not self.chunks:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.bm25 = None
            self.tokenized_chunks = []
            return

        texts = [c["text"] for c in self.chunks]
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        self.tokenized_chunks = [tokenize(t) for t in texts]
        self.bm25 = BM25(self.tokenized_chunks)

    def retrieve(self, query, top_k=TOP_K):
        if self.is_empty():
            return []

        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix @ query_vec.T).toarray().ravel()

        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        hybrid = TFIDF_WEIGHT * _minmax(tfidf_scores) + BM25_WEIGHT * _minmax(bm25_scores)

        order = np.argsort(-hybrid)[:top_k]
        results = []
        for idx in order:
            c = self.chunks[idx]
            results.append({
                "doc": c["doc"],
                "page": c["page"],
                "text": c["text"],
                "score": float(hybrid[idx]),
                "tfidf_score": float(tfidf_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
            })
        return results


# ---------------------------------------------------------------------------
# Confidence labeling + decision trace
# ---------------------------------------------------------------------------

def confidence_label(score):
    """Maps the raw hybrid retrieval score into a human-readable confidence band.

    Bands are derived from SCORE_THRESHOLD the same way the threshold itself is
    treated elsewhere in this file: as a hypothesis to refine once labeled data
    exists, not a fixed cutoff.
    """
    if score >= SCORE_THRESHOLD + 0.15:
        return "High"
    elif score >= SCORE_THRESHOLD:
        return "Medium"
    elif score >= SCORE_THRESHOLD * 0.5:
        return "Low"
    else:
        return "Very Low"


def build_decision_trace(top_score, coverage, margin, margin_required=True, margin_passed=None):
    """Step-by-step record of each check the answerability gate performed.

    Exposing this (rather than only the final bool) is what makes the
    "why can't I answer" explanation possible on the frontend.

    margin_required=False is used when the top passage already covers 100%
    of the query's terms: a near-zero margin against another equally strong
    match in that case means the term appears in multiple places in the
    corpus (corroborating evidence), not that the match is ambiguous. Without
    this, single-keyword questions about a common term regularly fail the
    margin check even with perfect coverage — a real failure mode found via
    the self-consistency checks in "Challenge the AI" test mode.
    """
    if margin_passed is None:
        margin_passed = margin >= MARGIN_MIN

    if margin_required:
        margin_explanation = "Is the top passage clearly better than the runner-up, or is the match ambiguous?"
    else:
        margin_explanation = (
            "Skipped: the top passage already covers 100% of the required terms, so a tight margin "
            "against another equally strong match reflects corroborating evidence, not ambiguity."
        )

    return [
        {
            "check": "Retrieval score",
            "value": round(top_score, 3),
            "threshold": SCORE_THRESHOLD,
            "passed": bool(top_score >= SCORE_THRESHOLD),
            "explanation": "Is the top passage a strong enough lexical/keyword match to the question?",
        },
        {
            "check": "Query term coverage",
            "value": round(coverage, 3),
            "threshold": COVERAGE_THRESHOLD,
            "passed": bool(coverage >= COVERAGE_THRESHOLD),
            "explanation": "Does the passage actually mention most of the key terms in the question?",
        },
        {
            "check": "Margin over next-best passage",
            "value": round(margin, 3),
            "threshold": MARGIN_MIN if margin_required else 0.0,
            "passed": bool(margin_passed),
            "explanation": margin_explanation,
        },
    ]


# ---------------------------------------------------------------------------
# Answerability decision + extractive answer
# ---------------------------------------------------------------------------

def query_coverage(query, evidence_text):
    q_terms = set(meaningful_tokens(query))
    if not q_terms:
        return 0.0, set(), set()
    e_terms = set(meaningful_tokens(evidence_text))
    covered = q_terms & e_terms
    return len(covered) / len(q_terms), q_terms, covered


def decide_answerability(query, results):
    """Returns a dict describing the answerable/abstain decision, matching brief section 08."""
    if not results:
        return {
            "answerable": False,
            "confidence": 0.0,
            "confidence_label": "None",
            "coverage": 0.0,
            "reason": "No documents have been indexed yet, or none were retrieved.",
            "missing_terms": [],
            "evidence": [],
            "decision_trace": [],
        }

    top = results[0]
    coverage, q_terms, covered = query_coverage(query, top["text"])
    missing_terms = sorted(q_terms - covered)
    margin = top["score"] - results[1]["score"] if len(results) > 1 else top["score"]

    # See build_decision_trace's docstring: when coverage is already
    # complete, a tight margin against another equally strong match is
    # corroborating evidence (the term appears in multiple places), not
    # genuine ambiguity, so the margin gate is skipped in that case.
    margin_required = coverage < 0.999
    margin_passed = (margin >= MARGIN_MIN) if margin_required else True

    trace = build_decision_trace(top["score"], coverage, margin, margin_required, margin_passed)
    answerable = all(step["passed"] for step in trace)

    if answerable:
        reason = "Top-ranked evidence covers most of the question's key terms with a strong retrieval score."
    elif not trace[0]["passed"]:
        reason = "No passage in the collection is a strong enough semantic/lexical match to this question."
    elif not trace[1]["passed"]:
        reason = (
            "The closest passage is topically related but does not mention: "
            + (", ".join(missing_terms[:6]) if missing_terms else "one or more required details")
            + ". This looks like an incomplete-evidence case, not a true match."
        )
    else:
        reason = "The top passage is not clearly better than the next best match, so the evidence is ambiguous."

    return {
        "answerable": answerable,
        "confidence": round(top["score"], 3),
        "confidence_label": confidence_label(top["score"]),
        "coverage": round(coverage, 3),
        "reason": reason,
        "missing_terms": missing_terms,
        "evidence": results,
        "decision_trace": trace,
    }


_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_LOW_INFO_MARKERS = ("retrieved from", "source:", "sourced from")


def _is_low_information(sentence):
    """Flags citation/URL fragments and other low-information sentences that
    make poor extractive answers even when they happen to contain the query
    term (e.g. a URL like 'https://parakh.education.gov.in' contains the
    literal word 'parakh', but is not an informative answer about PARAKH).
    """
    if _URL_RE.search(sentence):
        return True
    lower = sentence.lower()
    if any(marker in lower for marker in _LOW_INFO_MARKERS):
        return True
    if len(sentence.split()) < 6:
        return True
    return False


def extract_answer(query, top_chunk_text, max_sentences=2):
    """Pick the sentence(s) in the top chunk with the most query-term
    overlap, preferring substantive sentences over short citation/URL
    fragments that happen to contain the query term.
    """
    q_terms = set(meaningful_tokens(query))
    sentences = sentence_split(top_chunk_text)
    if not sentences:
        return top_chunk_text

    scored = []
    for s in sentences:
        s_terms = set(meaningful_tokens(s))
        overlap = len(q_terms & s_terms)
        scored.append((overlap, s))
    scored.sort(key=lambda x: x[0], reverse=True)

    substantive_matches = [s for ov, s in scored if ov > 0 and not _is_low_information(s)]
    any_matches = [s for ov, s in scored if ov > 0]

    if substantive_matches:
        best = substantive_matches[:max_sentences]
    elif any_matches:
        # Only a low-information fragment (citation, URL) matched the query
        # term. Pair it with the passage's first substantive sentence so the
        # answer isn't just a bare citation with no actual content.
        first_substantive = next((s for s in sentences if not _is_low_information(s)), None)
        best = ([first_substantive] if first_substantive else []) + any_matches[:1]
    else:
        best = [sentences[0]]

    return " ".join(best)