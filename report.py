from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(data):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []

    for row in data:
        text = f"{row['Time']} - {row['Status']} - Warnings: {row['Warnings']}"
        content.append(Paragraph(text, styles["Normal"]))

    doc.build(content)