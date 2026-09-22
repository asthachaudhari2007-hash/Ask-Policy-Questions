# PolicyGuard AI

**Evidence-Aware Policy Intelligence** — an internship project (ML-T2-091, Learn Depth Academy Track 2) that answers policy questions from uploaded PDF documents, but only when the evidence genuinely supports an answer. When it doesn't, the system says so and explains why, instead of guessing.

> Ask Policy Questions. Get Evidence.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Tunable Parameters](#tunable-parameters)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)

---

## Overview

Most PDF question-answering tools will always produce *an* answer, even when the source document doesn't actually contain one — a core driver of hallucination in retrieval-augmented systems. PolicyGuard AI is built around a different premise: **a policy Q&A system should be able to say "I don't have enough evidence to answer that."**

Every question is passed through an **evidence-aware answerability gate** before any answer is shown. The gate checks retrieval strength, query-term coverage, and result-margin confidence — all three must pass before the system will answer. Answers themselves are **extractive** (verbatim sentences copied from the source), never generated, so there is no hallucination risk in the answer text itself.

The system is entirely **self-contained**: no LLM API calls, no external translation service, no internet dependency at inference time (aside from optional one-time setup). Every design choice — chunking, hybrid retrieval, threshold-based answerability, contradiction/risk heuristics — is intentionally simple and explainable, in line with the project's research brief.

## Key Features

### Core Pipeline
- PDF upload & automatic text extraction (page-level, via PyMuPDF)
- Intelligent chunking (140-word chunks, 30-word overlap)
- Hybrid retrieval: BM25 (lexical, implemented from scratch) + TF-IDF/cosine similarity
- Evidence-aware answerability gate (retrieval score, term coverage, and margin thresholds)
- Extractive, citation-grounded question answering with page-level source references
- Hallucination protection via abstention when evidence is insufficient

### Advanced AI Features
- Confidence score with human-readable labels (High / Medium / Low / Very Low)
- "Why can't I answer?" plain-language explanation
- Step-by-step decision trace (every gate check shown with pass/fail and reasoning)
- Evidence highlighting (query terms highlighted inline in retrieved passages)
- Question decomposition (splits compound questions into independently-answered parts)
- Cross-document comparison (same question answered separately per uploaded document)
- Temporal policy analysis (extracts year-tagged figures from text via pattern matching)
- Policy trend visualization (interactive multi-metric line chart, no external chart library)

### Advanced Policy Analysis
- Contradiction detection (numeric: same metric/year with conflicting values across documents; textual: opposite-polarity phrasing candidates)
- Risk / flagged clause detection (keyword-based, word-boundary matched)
- Automatic extractive document summarization (frequency-scored sentence selection)

*(All three are explicitly framed as candidates for human review, not verified conclusions — see [Known Limitations](#known-limitations).)*

### Accessibility & Interaction
- Full English / Hindi UI toggle
- Hindi and Hinglish question support via a domain-specific glossary that augments retrieval queries
- Voice answer output (Web Speech API text-to-speech)
- Responsive, step-guided workflow with locked states until prerequisites are met

### Testing & Reporting
- "Challenge the AI" test mode — a one-click adversarial + self-consistency regression suite
- Question History panel with live session statistics (answerable/abstained counts, average confidence)
- Session export as a formatted PDF report (via ReportLab) or CSV

## How It Works

```
PDF Upload → Text Extraction → Chunking
                                   │
                                   ▼
                    Hybrid Retrieval (BM25 + TF-IDF)
                                   │
                                   ▼
        Evidence-Aware Answerability Gate (3 checks, all must pass)
              1. Retrieval score ≥ threshold
              2. Query-term coverage ≥ threshold
              3. Margin over runner-up sufficient (skipped if coverage = 100%)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
              ANSWERABLE                    INSUFFICIENT EVIDENCE
       Extractive answer +              Abstain + explanation of
       citations + confidence           what evidence is missing
```

Hindi/Hinglish questions are translated at the query level (domain-glossary term substitution) before retrieval, since the underlying document corpus is English.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| PDF processing | PyMuPDF (`fitz`) |
| Retrieval | BM25 (custom implementation), scikit-learn `TfidfVectorizer` |
| PDF report export | ReportLab |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Voice output | Web Speech API (browser-native) |
| Charts | Hand-built inline SVG (no charting library) |

No LLM API, no vector database, no external translation service.

## Project Structure

```
policy-answerability/
├── app.py                  # Flask routes: upload, ask, compare, trends, export, challenge, policy-analysis
├── rag_engine.py            # Chunking, BM25, hybrid retrieval, answerability gate, extractive answering
├── lang_support.py          # Hindi/Hinglish detection + domain-glossary query translation
├── decomposition.py         # Compound question splitting
├── temporal_engine.py       # Year-tagged figure extraction for trend analysis
├── policy_analysis.py       # Contradiction detection, risk flagging, extractive summarization
├── challenge_tests.py       # Adversarial + self-consistency test case generation
├── report_export.py         # Session-history-to-PDF report builder
├── templates/
│   └── index.html            # Single-page UI (upload, ask, compare, trends, analysis, history)
├── static/
│   └── css/style.css
├── uploads/                  # Uploaded PDFs (gitignored)
└── data/processed/           # Extracted plain-text per document (gitignored)
```

## Setup & Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd policy-answerability

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

The app runs at `http://127.0.0.1:5000` by default.

## Usage

1. **Upload** a policy PDF on the home page — text is extracted and indexed automatically.
2. **Ask a question** in English, Hindi, or Hinglish. The system retrieves evidence and either answers with citations or explains why it's abstaining.
3. **Compare across documents** once 2+ PDFs are uploaded, to see per-document answerability side by side.
4. **View policy trends** by picking a metric (GER, PTR, dropout, etc.) to chart year-over-year figures found in your documents.
5. **Run Advanced Policy Analysis** to surface candidate contradictions, risk-flagged clauses, and per-document summaries.
6. **Run Challenge Tests** to regression-check the answerability gate against adversarial and self-consistency probes.
7. **Review your session** in the Question History panel, and export it as a PDF or CSV report.

## API Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Renders the main UI |
| `/upload` | POST | Uploads and indexes a PDF |
| `/ask` | POST | Answers a question (with decomposition if compound) |
| `/compare` | POST | Answers a question separately per indexed document |
| `/trends` | POST | Returns year-tagged data points for a chosen metric |
| `/policy-analysis` | POST | Returns contradictions, risk flags, and summaries |
| `/challenge` | POST | Runs the adversarial + self-consistency test suite |
| `/export/pdf` | POST | Generates a PDF report from a submitted session history |

## Tunable Parameters

All thresholds live in `rag_engine.py` and are treated as explicit hypotheses to validate, not fixed truths:

| Parameter | Default | Meaning |
|---|---|---|
| `CHUNK_WORDS` | 140 | Words per chunk |
| `CHUNK_OVERLAP` | 30 | Overlap between consecutive chunks |
| `SCORE_THRESHOLD` | 0.18 | Minimum hybrid retrieval score to consider answering |
| `COVERAGE_THRESHOLD` | 0.45 | Minimum fraction of query terms that must appear in evidence |
| `MARGIN_MIN` | 0.03 | Minimum score gap over the runner-up passage |
| `BM25_WEIGHT` / TF-IDF weight | 0.5 / 0.5 | Hybrid retrieval mix |

## Known Limitations

Stated explicitly, as acknowledged design trade-offs rather than oversights:

- **Extractive only** — answers are copied verbatim from source text; there is no generative rewriting or synthesis across passages.
- **English-only source documents** — Hindi/Hinglish support translates the *question* via a domain glossary; it does not translate document content.
- **Heuristic thresholds** — the answerability gate's thresholds are hand-tuned starting points, not learned from labeled data.
- **Regex-based temporal extraction** — year/value extraction can miss non-standard formats or produce occasional false positives.
- **Contradiction and risk detection are candidate-generators, not verified findings** — both use coarse keyword/heuristic matching and are explicitly surfaced as "for manual review," with a meaningful false-positive rate by design.
- **Chunk-boundary effects** — since chunking is word-count based rather than sentence-aware, a chunk can occasionally start mid-sentence, which can affect which exact sentence is extracted as an answer.

## Roadmap

Potential future additions: a document library/management UI, PDF integrity validation on upload, voice question input, dark/light mode, and a dedicated evidence explorer for browsing the indexed corpus directly.

## Acknowledgments

Built for ML-T2-091, Learn Depth Academy Track 2. Test documents used during development include a NITI Aayog-style report on the school education system in India and an NEP 2020 implementation summary.