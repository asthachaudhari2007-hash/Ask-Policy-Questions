"""
lang_support.py
-----------------
Lightweight Hindi / Hinglish query support for PolicyGuard AI (ML-T2-091).

Design choice (consistent with rag_engine.py's philosophy): no internet, no
external translation API. Instead this module:
  1. Detects whether a question is English, Hindi (Devanagari script), or
     Hinglish (Hindi written in Roman script).
  2. Maps recognized policy-domain terms (Hindi/Hinglish -> English) using a
     hand-built glossary, so the existing BM25 + TF-IDF retrieval (which only
     understands the English-language document corpus) can still find the
     right passages.
  3. Returns what it translated, so the decision is transparent rather than
     a black box — this is intentionally a small, auditable dictionary, not
     a general-purpose translator. Extend TERM_GLOSSARY as needed; it is a
     placeholder in the same sense SCORE_THRESHOLD is in rag_engine.py.

This does NOT translate the final English answer back into Hindi — the
underlying documents are English, so the extractive answer stays in English.
Translating the *question* is enough to retrieve the right evidence.
"""

import re

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_WORD_RE = re.compile(r"[\u0900-\u097Fa-zA-Z]+")

# Common Hindi function/discourse words, written in Roman script. Presence of
# any of these (with no Devanagari) is treated as a strong Hinglish signal.
HINGLISH_MARKERS = {
    "kya", "hai", "hain", "ka", "ki", "ke", "kaise", "kyun", "kyu", "nahi",
    "kar", "sakte", "sakta", "sakti", "kab", "kaun", "kahan", "kitna",
    "kitne", "mujhe", "hamare", "humare", "aap", "apka", "apke", "iske",
    "uske", "hoga", "hogi", "chahiye", "matlab", "wala", "wali",
}


def detect_language(text):
    """Returns 'hindi', 'hinglish', or 'english'."""
    if _DEVANAGARI_RE.search(text):
        return "hindi"

    tokens = {t.lower() for t in re.findall(r"[a-zA-Z]+", text)}
    if tokens & HINGLISH_MARKERS:
        return "hinglish"

    return "english"


# ---------------------------------------------------------------------------
# Domain glossary (Hindi + Hinglish -> English)
# ---------------------------------------------------------------------------
# Keep keys lowercase for Roman-script entries; Devanagari entries are
# matched as-is. Extend this as you see real questions fail to retrieve.

TERM_GLOSSARY = {
    # attendance
    "उपस्थिति": "attendance", "hazri": "attendance", "upasthiti": "attendance",

    # leave / medical
    "छुट्टी": "leave", "chhutti": "leave", "avkash": "leave", "अवकाश": "leave",
    "बीमारी": "illness medical", "bimari": "illness medical",

    # fee / waiver
    "फीस": "fee", "fees": "fee", "shulk": "fee", "शुल्क": "fee",
    "छूट": "waiver exemption", "choot": "waiver exemption",

    # student / teacher
    "छात्र": "student", "chhaatra": "student", "vidyarthi": "student", "विद्यार्थी": "student",
    "शिक्षक": "teacher", "shikshak": "teacher", "adhyapak": "teacher", "अध्यापक": "teacher",

    # school / education / policy
    "विद्यालय": "school", "vidyalaya": "school", "school": "school",
    "शिक्षा": "education", "shiksha": "education",
    "नीति": "policy", "niti": "policy",
    "नियम": "rule", "niyam": "rule",

    # admission / exam / dropout
    "दाखिला": "admission", "daakhila": "admission", "admission": "admission",
    "परीक्षा": "exam", "pariksha": "exam", "exam": "exam",
    "ड्रॉपआउट": "dropout",

    # hostel / scholarship
    "छात्रावास": "hostel", "chhaatravas": "hostel",
    "छात्रवृत्ति": "scholarship", "chhaatravritti": "scholarship", "scholarship": "scholarship",

    # government / recommendation
    "सरकार": "government", "sarkar": "government",
    "सिफारिश": "recommendation", "sifarish": "recommendation",
}


def translate_query_terms(query):
    """Maps recognized Hindi/Hinglish domain terms to English and appends them
    to the query so the English-language retrieval index can match on them.

    Returns (augmented_query, matched_terms) where matched_terms is a list of
    (original_token, english_gloss) pairs actually recognized — shown to the
    user so the translation step stays transparent.
    """
    tokens = _WORD_RE.findall(query)
    matched_terms = []
    extra_terms = []

    for tok in tokens:
        key = tok if _DEVANAGARI_RE.search(tok) else tok.lower()
        if key in TERM_GLOSSARY:
            gloss = TERM_GLOSSARY[key]
            matched_terms.append((tok, gloss))
            extra_terms.append(gloss)

    # Keep the original query (so proper nouns like "NEP", "GER" survive)
    # and append the English glosses so BM25/TF-IDF can match the corpus.
    augmented_query = query + " " + " ".join(extra_terms) if extra_terms else query
    return augmented_query, matched_terms