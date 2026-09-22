"""
report_export.py
------------------
Builds a downloadable PDF report from a client-submitted question-history
session (ML-T2-091 roadmap #33/#34: Export Research Report / Evidence-Based
PDF Report).

The session data itself (questions, verdicts, confidence, evidence) is
generated during normal use of the app and lives in the browser's Question
History panel; this module only formats whatever the frontend sends as a
clean, readable PDF. No new retrieval or answerability logic here.
"""

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable


def _esc(text):
    return escape(str(text)) if text is not None else ""


def build_session_report_pdf(session):
    """session: list of dicts shaped like a frontend history entry:
    {question, answerable, confidence_label, confidence, coverage, reason,
    answer, evidence: [{doc, page, text, score}, ...]}.

    Returns raw PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], alignment=TA_CENTER, fontSize=18)
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10,
                                 textColor="#666666", spaceAfter=18)
    section_style = ParagraphStyle("QSection", parent=styles["Heading2"], fontSize=12,
                                    spaceBefore=14, spaceAfter=4, textColor="#3730a3")
    label_style = ParagraphStyle("Label", parent=styles["Normal"], fontSize=9.5,
                                  textColor="#374151", spaceAfter=2, leading=13)
    body_style = ParagraphStyle("Body2", parent=styles["Normal"], fontSize=9.5, leading=13.5, spaceAfter=6)
    evidence_style = ParagraphStyle("Evidence", parent=styles["Normal"], fontSize=8.5, leading=12,
                                     textColor="#4b5563", leftIndent=12, spaceAfter=4)

    story = []
    story.append(Paragraph("PolicyGuard AI &mdash; Research Session Report", title_style))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')} &middot; {len(session)} question(s) in this session",
        meta_style
    ))

    total = len(session)
    answerable_count = sum(1 for s in session if s.get("answerable"))
    abstained_count = total - answerable_count
    confidences = [s.get("confidence") for s in session if isinstance(s.get("confidence"), (int, float))]
    avg_conf = (sum(confidences) / len(confidences) * 100) if confidences else None

    summary_line = f"Answerable: {answerable_count} &nbsp;&nbsp;|&nbsp;&nbsp; Abstained: {abstained_count}"
    if avg_conf is not None:
        summary_line += f" &nbsp;&nbsp;|&nbsp;&nbsp; Avg. confidence: {avg_conf:.1f}%"
    story.append(Paragraph(summary_line, label_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color="#e5e7eb"))

    for i, entry in enumerate(session, start=1):
        q = _esc(entry.get("question", ""))
        verdict = "Answerable" if entry.get("answerable") else "Insufficient Evidence \u2014 Abstained"
        conf_label = _esc(entry.get("confidence_label", "N/A"))

        story.append(Paragraph(f"{i}. {q}", section_style))
        story.append(Paragraph(f"<b>Verdict:</b> {verdict} &nbsp;&nbsp; <b>Confidence:</b> {conf_label}", label_style))

        if entry.get("answerable"):
            story.append(Paragraph(f"<b>Answer:</b> {_esc(entry.get('answer', ''))}", body_style))
        else:
            story.append(Paragraph(f"<b>Why:</b> {_esc(entry.get('reason', ''))}", body_style))

        evidence = entry.get("evidence") or []
        if evidence:
            story.append(Paragraph("<b>Evidence:</b>", label_style))
            for e in evidence[:3]:
                doc_name = _esc(e.get("doc", ""))
                page = e.get("page", "")
                text = _esc(e.get("text", ""))[:400]
                story.append(Paragraph(f"[{doc_name} \u2014 p.{page}] {text}", evidence_style))

        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", color="#f3f4f6"))

    doc.build(story)
    return buffer.getvalue()