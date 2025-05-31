import matplotlib
matplotlib.use('Agg')
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
from flask import send_file
from wordcloud import WordCloud
from datetime import datetime
from src.app.data_store import dialogue_data
import io


def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    elements = []

    # Cover Title
    elements.append(Spacer(1, 40))
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, textColor=colors.HexColor("#2c3e50"))
    elements.append(Paragraph("Law Enforcement Empathy Interaction Summary", title_style))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%b %d, %Y at %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 30))

    if not dialogue_data['dialogue']:
        elements.append(Paragraph("No data available.", styles['Normal']))
        doc.build(elements)
        buffer.seek(0)
        return send_file(buffer, download_name="report.pdf", as_attachment=True, mimetype="application/pdf")

    # De-duplicate dialogue entries
    seen = set()
    dialogue_scores = []
    for text, score in zip(dialogue_data['dialogue'], dialogue_data['empathy_scores']):
        if text not in seen:
            seen.add(text)
            dialogue_scores.append((text, score))

    # Dialogue Table
    elements.append(Paragraph("Dialogue & Empathy Scores", styles['Heading2']))
    elements.append(Spacer(1, 12))

    table_data = [["Dialogue", "Empathy Score"]]
    max_len = 80
    for text, score in dialogue_scores:
        trimmed = text if len(text) <= max_len else text[:max_len] + "..."
        table_data.append([trimmed, f"{score:.2f}"])

    table = Table(table_data, colWidths=[360, 90])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # Empathy Score Graph
    graph_buffer = io.BytesIO()
    plt.figure(figsize=(6, 3))
    plt.plot(range(len(dialogue_data['empathy_scores'])), dialogue_data['empathy_scores'], marker='o', linestyle='-')
    plt.xlabel("Dialogue Turn")
    plt.ylabel("Empathy Score")
    plt.title("Empathy Score Progression")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_buffer, format="PNG")
    plt.close()
    graph_buffer.seek(0)

    elements.append(Paragraph("Empathy Score Graph", styles['Heading2']))
    elements.append(Spacer(1, 12))
    graph_img = Image(graph_buffer, width=6 * inch, height=3 * inch)
    graph_img.hAlign = 'CENTER'
    elements.append(graph_img)
    elements.append(Spacer(1, 20))

    # Word Cloud
    text_data = " ".join(dialogue_data['dialogue'])
    wordcloud = WordCloud(width=600, height=300, background_color="white").generate(text_data)
    wc_buffer = io.BytesIO()
    wordcloud.to_image().save(wc_buffer, format="PNG")
    wc_buffer.seek(0)

    elements.append(Paragraph("Empathy Word Cloud", styles['Heading2']))
    elements.append(Spacer(1, 12))
    wordcloud_img = Image(wc_buffer, width=6 * inch, height=3 * inch)
    wordcloud_img.hAlign = 'CENTER'
    elements.append(wordcloud_img)

    # Build and Return PDF
    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        download_name='Empathy_Analysis_Report.pdf',
        as_attachment=True,
        mimetype='application/pdf'
    )
