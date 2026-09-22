"""
temporal_engine.py
--------------------
Extracts year-tagged numeric mentions from indexed document chunks
(ML-T2-091 roadmap #17 Temporal Policy Analysis + #18 Trend Visualization).

Design: purely regex-based pattern matching over already-chunked text — no
new model, no external API. Looks for a year or year-range near a percentage
or plain number, tagged with whatever metric keyword appears nearby (GER,
PTR, dropout, etc.), so a question like "how has GER changed over time" can
be turned into a small time series and charted.

This is intentionally a heuristic extractor, not a guaranteed-correct one —
same "hypothesis to validate" spirit as the thresholds in rag_engine.py.
False positives (e.g. a page number that happens to sit near a year) are
possible; the metric-keyword requirement below is what keeps noise down.
"""

import re

_YEAR_RE = r"(\d{4}(?:-\d{2,4})?)"
_NUMBER_RE = r"(\d{1,3}(?:\.\d+)?)\s?%?"

_YEAR_THEN_NUMBER = re.compile(_YEAR_RE + r"[^\d%]{0,25}?" + _NUMBER_RE)
_NUMBER_THEN_YEAR = re.compile(_NUMBER_RE + r"[^\d%]{0,25}?" + _YEAR_RE)

# Metric keywords used both to tag a data point and to keep noise down —
# a bare number near a year with no recognizable metric label is skipped.
METRIC_KEYWORDS = [
    "GER", "gross enrolment ratio", "PTR", "pupil-teacher ratio",
    "dropout", "enrolment", "enrollment", "attendance", "literacy rate",
]


def _nearest_metric(text, position, window=80):
    start = max(0, position - window)
    end = min(len(text), position + window)
    snippet = text[start:end].lower()
    for kw in METRIC_KEYWORDS:
        if kw.lower() in snippet:
            return kw
    return None


def extract_temporal_points(chunks, metric_filter=None):
    """chunks: list of {doc, page, text} dicts (DocumentIndex.chunks).

    Returns a list of {metric, year, year_sort, value, doc, page} dicts,
    deduplicated by (metric, year, value), sorted by year.
    """
    points = []
    seen = set()

    for chunk in chunks:
        text = chunk["text"]

        for pattern in (_YEAR_THEN_NUMBER, _NUMBER_THEN_YEAR):
            for m in pattern.finditer(text):
                groups = m.groups()
                if pattern is _YEAR_THEN_NUMBER:
                    year_str, value_str = groups[0], groups[1]
                else:
                    value_str, year_str = groups[0], groups[1]

                try:
                    value = float(value_str)
                except (TypeError, ValueError):
                    continue

                try:
                    year_num = int(year_str[:4])
                except ValueError:
                    continue
                if year_num < 1990 or year_num > 2035:
                    continue  # filters out obvious false positives (page refs etc.)

                metric = _nearest_metric(text, m.start())
                if not metric:
                    continue  # unlabeled numbers are too noisy to keep
                if metric_filter and metric_filter.lower() not in metric.lower():
                    continue

                key = (metric, year_str, value_str)
                if key in seen:
                    continue
                seen.add(key)

                points.append({
                    "metric": metric,
                    "year": year_str,
                    "year_sort": year_num,
                    "value": value,
                    "doc": chunk["doc"],
                    "page": chunk["page"],
                })

    points.sort(key=lambda p: p["year_sort"])
    return points