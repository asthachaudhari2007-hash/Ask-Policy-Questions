from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import io
import fitz  # PyMuPDF

from rag_engine import (
    DocumentIndex, decide_answerability, extract_answer,
    SCORE_THRESHOLD, COVERAGE_THRESHOLD, MARGIN_MIN,
)
from lang_support import detect_language, translate_query_terms
from decomposition import decompose_question
from temporal_engine import extract_temporal_points, METRIC_KEYWORDS
from report_export import build_session_report_pdf
from challenge_tests import build_challenge_cases
from policy_analysis import detect_contradictions, detect_risk_flags, summarize_all_documents

app = Flask(__name__)

# Folder configuration
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "data/processed"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create folders automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# In-memory retrieval index, rebuilt whenever a new document is processed
INDEX = DocumentIndex()
INDEX.build(PROCESSED_FOLDER)  # picks up any docs already processed from a previous run


@app.route("/")
def home():
    return render_template("index.html", indexed_chunks=len(INDEX.chunks))


@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return redirect(url_for("home"))

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("home"))

    if not file.filename.lower().endswith(".pdf"):
        return redirect(url_for("home"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    document = fitz.open(filepath)
    extracted_text = []
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")
        extracted_text.append({"page": page_number, "text": text})
    document.close()

    output_filename = os.path.splitext(file.filename)[0] + ".txt"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for page_data in extracted_text:
            output_file.write(f"\n--- PAGE {page_data['page']} ---\n\n")
            output_file.write(page_data["text"])

    INDEX.build(PROCESSED_FOLDER)

    return render_template(
        "index.html",
        success=True,
        filename=file.filename,
        pages=len(extracted_text),
        message="PDF uploaded and text extracted successfully.",
        indexed_chunks=len(INDEX.chunks)
    )


def answer_single_question(question):
    """Runs the full retrieve -> decide -> extract pipeline for ONE question.

    Shared by /ask (single questions) and the decomposition path (each
    sub-question runs through this same function), so behavior stays
    identical whether or not the original question was split.
    """
    detected_language = detect_language(question)
    if detected_language != "english":
        retrieval_query, matched_terms = translate_query_terms(question)
    else:
        retrieval_query, matched_terms = question, []

    results = INDEX.retrieve(retrieval_query)
    decision = decide_answerability(retrieval_query, results)

    response = {
        "question": question,
        "detected_language": detected_language,
        "translated_query": retrieval_query if detected_language != "english" else None,
        "matched_terms": [{"original": o, "english": e} for o, e in matched_terms],
        "answerable": decision["answerable"],
        "confidence": decision["confidence"],
        "confidence_label": decision["confidence_label"],
        "coverage": decision.get("coverage", 0),
        "reason": decision["reason"],
        "missing_terms": decision["missing_terms"],
        "decision_trace": decision["decision_trace"],
        "thresholds": {
            "score_threshold": SCORE_THRESHOLD,
            "coverage_threshold": COVERAGE_THRESHOLD,
            "margin_min": MARGIN_MIN,
        },
        "evidence": [
            {"doc": e["doc"], "page": e["page"], "text": e["text"], "score": round(e["score"], 3)}
            for e in decision["evidence"]
        ],
    }

    if decision["answerable"]:
        response["answer"] = extract_answer(retrieval_query, decision["evidence"][0]["text"])

    return response


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    if INDEX.is_empty():
        return jsonify({"error": "No documents have been uploaded yet. Upload a policy PDF first."}), 400

    # --- Question Decomposition ---
    # A compound question ("What is GER and how has it changed?") is split
    # into independently-answerable parts; each part runs the identical
    # single-question pipeline so results stay directly comparable.
    sub_questions = decompose_question(question)

    if len(sub_questions) > 1:
        sub_answers = [answer_single_question(sq) for sq in sub_questions]
        response = {
            "question": question,
            "decomposed": True,
            "sub_questions": sub_answers,
            "answerable": all(sa["answerable"] for sa in sub_answers),
        }
        return jsonify(response)

    response = answer_single_question(question)
    response["decomposed"] = False
    return jsonify(response)


@app.route("/compare", methods=["POST"])
def compare_documents():
    """Cross-Document Comparison (#16): answers the same question separately
    against each indexed document, so the user can see where documents
    agree, disagree, or where only one of them has relevant evidence.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    if INDEX.is_empty():
        return jsonify({"error": "No documents have been uploaded yet. Upload a policy PDF first."}), 400

    documents = sorted(set(c["doc"] for c in INDEX.chunks))
    if len(documents) < 2:
        return jsonify({"error": "Upload at least two documents to compare across them."}), 400

    detected_language = detect_language(question)
    if detected_language != "english":
        retrieval_query, _ = translate_query_terms(question)
    else:
        retrieval_query = question

    # Pull a generous pool so every document gets a fair chance to appear,
    # then split by document and run the SAME answerability gate on each
    # document's own slice — so "insufficient evidence" is judged per-doc.
    pool_size = max(30, len(documents) * 10)
    pooled_results = INDEX.retrieve(retrieval_query, top_k=pool_size)

    per_doc = []
    for doc in documents:
        doc_results = [r for r in pooled_results if r["doc"] == doc]
        decision = decide_answerability(retrieval_query, doc_results)
        entry = {
            "doc": doc,
            "answerable": decision["answerable"],
            "confidence": decision["confidence"],
            "confidence_label": decision["confidence_label"],
            "coverage": decision.get("coverage", 0),
            "reason": decision["reason"],
        }
        if decision["answerable"] and decision["evidence"]:
            entry["answer"] = extract_answer(retrieval_query, decision["evidence"][0]["text"])
            entry["page"] = decision["evidence"][0]["page"]
        per_doc.append(entry)

    return jsonify({"question": question, "documents": per_doc})


@app.route("/trends", methods=["POST"])
def trends():
    """Temporal Policy Analysis + Trend Visualization (#17/#18): extracts a
    small year -> value series for a chosen metric from the indexed
    documents, for the frontend to render as a line chart.
    """
    data = request.get_json(silent=True) or {}
    metric = (data.get("metric") or "").strip()

    if INDEX.is_empty():
        return jsonify({"error": "No documents have been uploaded yet. Upload a policy PDF first."}), 400

    points = extract_temporal_points(INDEX.chunks, metric_filter=metric or None)

    return jsonify({
        "metric": metric,
        "available_metrics": METRIC_KEYWORDS,
        "points": points,
    })


@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    """Export Research Report (#33/#34): turns the client-side Question
    History session into a downloadable, formatted PDF. The session data
    itself is generated and held in the browser; this endpoint only
    formats whatever it's sent.
    """
    data = request.get_json(silent=True) or {}
    session = data.get("session") or []

    if not session:
        return jsonify({"error": "No questions in this session to export yet."}), 400

    pdf_bytes = build_session_report_pdf(session)

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="policyguard_session_report.pdf",
    )


@app.route("/challenge", methods=["POST"])
def challenge():
    """"Challenge the AI" Test Mode (#29): runs a battery of adversarial and
    self-consistency probes against the current index and reports pass/fail
    for each, turning manual robustness testing into a one-click check.
    """
    if INDEX.is_empty():
        return jsonify({"error": "No documents have been uploaded yet. Upload a policy PDF first."}), 400

    cases = build_challenge_cases(INDEX.chunks)
    results = []
    for case in cases:
        response = answer_single_question(case["question"])
        passed = response["answerable"] == case["expected_answerable"]
        results.append({
            "question": case["question"],
            "note": case["note"],
            "expected_answerable": case["expected_answerable"],
            "actual_answerable": response["answerable"],
            "confidence_label": response["confidence_label"],
            "passed": passed,
            "reason": response.get("reason"),
            "answer": response.get("answer"),
        })

    passed_count = sum(1 for r in results if r["passed"])
    return jsonify({
        "total": len(results),
        "passed": passed_count,
        "results": results,
    })


@app.route("/policy-analysis", methods=["POST"])
def policy_analysis():
    """Advanced Policy Analysis: contradiction detection, risk-clause
    flagging, and extractive summarization. All heuristic and explainable —
    results are candidates for human review, not verified conclusions.
    """
    if INDEX.is_empty():
        return jsonify({"error": "No documents have been uploaded yet. Upload a policy PDF first."}), 400

    contradictions = detect_contradictions(INDEX.chunks)
    risks = detect_risk_flags(INDEX.chunks)
    summaries = summarize_all_documents(INDEX.chunks)

    return jsonify({
        "contradictions": contradictions,
        "risks": risks,
        "summaries": summaries,
    })


if __name__ == "__main__":
    app.run(debug=True)