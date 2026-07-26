from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import io

# ── Colour Palette ─────────────────────────────────────────────────────────
BLUE       = colors.HexColor("#38bdf8")
DARK       = colors.HexColor("#0f172a")
DARK_MID   = colors.HexColor("#1e293b")
SLATE      = colors.HexColor("#334155")
MUTED      = colors.HexColor("#64748b")
WHITE      = colors.white
GREEN      = colors.HexColor("#22c55e")
YELLOW     = colors.HexColor("#eab308")
ORANGE     = colors.HexColor("#f97316")
RED        = colors.HexColor("#ef4444")


def score_colour(score):
    if score >= 90: return GREEN
    if score >= 70: return YELLOW
    if score >= 50: return ORANGE
    return RED


def score_grade(score):
    if score >= 90: return "PASS — High Integrity"
    if score >= 70: return "PASS — Moderate Integrity"
    if score >= 50: return "FLAGGED — Review Required"
    return "FAIL — Integrity Compromised"


def calc_score(alerts):
    score = 100
    for a in alerts:
        t = (a.get("violation_type") or "").upper()
        if   "IMPERSONATION"  in t: score -= 20
        elif "UNAUTHORIZED"   in t: score -= 15
        elif "OBJECT_DETECTED" in t: score -= 10
        elif "LOOKING"        in t: score -= 5
        elif "NO FACE"        in t: score -= 5
    return max(0, score)


# ═══════════════════════════════════════════════════════════════════════════
def generate_report(student: dict, alerts: list) -> bytes:
    """
    Generates a PDF exam integrity report for a single student.
    Returns the PDF as bytes for download or storage.
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
        title=f"AIProctor Report — {student.get('registration_number', '')}",
    )

    styles  = getSampleStyleSheet()
    story   = []
    score   = calc_score(alerts)
    grade   = score_grade(score)
    s_colour = score_colour(score)

    # ── Header ────────────────────────────────────────────────────────────
    header_style = ParagraphStyle(
        "header",
        fontSize=22,
        textColor=BLUE,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "sub",
        fontSize=10,
        textColor=MUTED,
        fontName="Helvetica",
        alignment=TA_CENTER,
        spaceAfter=2,
    )

    story.append(Paragraph("AIProctor", header_style))
    story.append(Paragraph(
        "Examination Integrity Report", sub_style
    ))
    story.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%d %B %Y, %H:%M UTC')}",
        sub_style
    ))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(
        width="100%", thickness=2,
        color=BLUE, spaceAfter=0.4*cm
    ))

    # ── Student Info Table ────────────────────────────────────────────────
    label_style = ParagraphStyle(
        "lbl",
        fontSize=8,
        textColor=MUTED,
        fontName="Helvetica-Bold",
        spaceAfter=2,
    )
    value_style = ParagraphStyle(
        "val",
        fontSize=11,
        textColor=WHITE,
        fontName="Helvetica-Bold",
    )

    info_data = [
        [
            Paragraph("STUDENT NAME",        label_style),
            Paragraph("REGISTRATION NUMBER", label_style),
            Paragraph("DEPARTMENT",          label_style),
        ],
        [
            Paragraph(student.get("full_name",           "—"), value_style),
            Paragraph(student.get("registration_number", "—"), value_style),
            Paragraph(student.get("department",          "—"), value_style),
        ],
        [
            Paragraph("EXAM ID",    label_style),
            Paragraph("STATUS",     label_style),
            Paragraph("EXAM DATE",  label_style),
        ],
        [
            Paragraph(student.get("exam_id", "—") or "—", value_style),
            Paragraph(
                (student.get("status") or "completed").upper(),
                value_style
            ),
            Paragraph(
                datetime.utcnow().strftime("%d %B %Y"), value_style
            ),
        ],
    ]

    info_table = Table(info_data, colWidths=[5.7*cm, 5.7*cm, 5.7*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), DARK_MID),
        ("ROWBACKGROUND",(0,0),(-1,-1),   DARK_MID),
        ("BOX",         (0, 0), (-1, -1), 1, SLATE),
        ("INNERGRID",   (0, 0), (-1, -1), 0.5, SLATE),
        ("TOPPADDING",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0,0),(-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",(0, 0), (-1, -1), 12),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Integrity Score Box ───────────────────────────────────────────────
    score_style = ParagraphStyle(
        "score",
        fontSize=36,
        textColor=s_colour,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    grade_style = ParagraphStyle(
        "grade",
        fontSize=12,
        textColor=s_colour,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    score_label = ParagraphStyle(
        "score_lbl",
        fontSize=8,
        textColor=MUTED,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    score_data = [[
        Paragraph("INTEGRITY SCORE", score_label),
        Paragraph("GRADE",           score_label),
        Paragraph("TOTAL VIOLATIONS",score_label),
    ],[
        Paragraph(f"{score}%",        score_style),
        Paragraph(grade,              grade_style),
        Paragraph(str(len(alerts)),   score_style),
    ]]

    score_table = Table(score_data, colWidths=[5.7*cm, 5.7*cm, 5.7*cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), DARK),
        ("BOX",          (0, 0), (-1, -1), 2, s_colour),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, SLATE),
        ("TOPPADDING",   (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 12),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Score Deduction Key ───────────────────────────────────────────────
    key_title = ParagraphStyle(
        "key_title",
        fontSize=9,
        textColor=MUTED,
        fontName="Helvetica-Bold",
        spaceBefore=4,
        spaceAfter=6,
    )
    story.append(Paragraph("SCORING DEDUCTIONS", key_title))

    key_data = [
        ["Violation Type",          "Deduction", "Occurrences", "Total Deducted"],
    ]

    # Count occurrences per type
    type_counts = {}
    for a in alerts:
        t = a.get("violation_type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    deduction_map = {
        "IMPERSONATION":    20,
        "UNAUTHORIZED":     15,
        "OBJECT_DETECTED":  10,
        "LOOKING":           5,
        "NO FACE":           5,
    }

    for vtype, count in type_counts.items():
        deduction = 0
        t_upper   = vtype.upper()
        for key, val in deduction_map.items():
            if key in t_upper:
                deduction = val
                break
        total_ded = min(deduction * count, 100)
        key_data.append([
            vtype,
            f"-{deduction} pts",
            str(count),
            f"-{total_ded} pts"
        ])

    key_table = Table(
        key_data,
        colWidths=[8.5*cm, 2.5*cm, 3*cm, 3.1*cm]
    )
    key_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  DARK_MID),
        ("BACKGROUND",   (0, 1), (-1, -1), DARK),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",    (0, 1), (-1, -1), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("BOX",          (0, 0), (-1, -1), 1, SLATE),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, SLATE),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [DARK, DARK_MID]),
    ]))
    story.append(key_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Violation Timeline ────────────────────────────────────────────────
    if alerts:
        timeline_title = ParagraphStyle(
            "tl_title",
            fontSize=9,
            textColor=MUTED,
            fontName="Helvetica-Bold",
            spaceBefore=4,
            spaceAfter=6,
        )
        story.append(Paragraph("VIOLATION TIMELINE", timeline_title))

        tl_data = [["#", "Timestamp", "Violation Type", "Objects"]]

        for i, a in enumerate(alerts[:50], 1):  # cap at 50 rows
            ts = a.get("timestamp", "")
            try:
                dt  = datetime.fromisoformat(ts)
                ts  = dt.strftime("%H:%M:%S")
            except Exception:
                pass

            vtype   = a.get("violation_type", "—")
            objects = ", ".join(a.get("objects", [])) or "—"

            # Skip session end records in timeline
            if "EXAM_ENDED" in vtype.upper():
                continue

            tl_data.append([
                str(i), ts, vtype, objects
            ])

        tl_table = Table(
            tl_data,
            colWidths=[1*cm, 3*cm, 10*cm, 3.1*cm]
        )
        tl_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  DARK_MID),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  BLUE),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("TEXTCOLOR",     (0, 1), (-1, -1), WHITE),
            ("BOX",           (0, 0), (-1, -1), 1, SLATE),
            ("INNERGRID",     (0, 0), (-1, -1), 0.5, SLATE),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [DARK, DARK_MID]),
            ("ALIGN",         (0, 0), (1, -1),  "CENTER"),
        ]))
        story.append(tl_table)

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(
        width="100%", thickness=1,
        color=SLATE, spaceAfter=0.3*cm
    ))
    footer_style = ParagraphStyle(
        "footer",
        fontSize=8,
        textColor=MUTED,
        fontName="Helvetica",
        alignment=TA_CENTER,
    )
    story.append(Paragraph(
        "This report was generated automatically by the AIProctor "
        "Examination Monitoring System. "
        "It is confidential and intended for authorised institutional "
        "use only.",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()