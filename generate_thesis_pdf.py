"""
Grivora AI — Thesis PDF Generator
Generates a professional, academic-quality PDF from the WORKFLOW.md content.
"""

import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, Color, white, black
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable, Preformatted, Flowable,
    Frame, PageTemplate, BaseDocTemplate, NextPageTemplate
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPDF
import os
import datetime

# ─── Color Palette ───────────────────────────────────────────────────────────
COLORS = {
    'primary':      HexColor('#1a1a2e'),    # Deep navy
    'secondary':    HexColor('#16213e'),    # Dark blue
    'accent':       HexColor('#0f3460'),    # Royal blue
    'highlight':    HexColor('#e94560'),    # Coral red
    'accent_blue':  HexColor('#2563eb'),    # Bright blue
    'accent_purple':HexColor('#7c3aed'),    # Purple
    'accent_cyan':  HexColor('#0891b2'),    # Cyan
    'success':      HexColor('#059669'),    # Green
    'warning':      HexColor('#d97706'),    # Amber
    'text_dark':    HexColor('#1a1a2e'),    # Main text
    'text_medium':  HexColor('#4a5568'),    # Secondary text
    'text_light':   HexColor('#718096'),    # Tertiary text
    'bg_light':     HexColor('#f8fafc'),    # Light background
    'bg_code':      HexColor('#1e293b'),    # Code background
    'bg_code_text': HexColor('#e2e8f0'),    # Code text
    'border':       HexColor('#e2e8f0'),    # Borders
    'table_header': HexColor('#1e40af'),    # Table header bg
    'table_alt':    HexColor('#f1f5f9'),    # Table alt row
    'white':        HexColor('#ffffff'),
    'cover_gradient_start': HexColor('#0f172a'),
    'cover_gradient_end':   HexColor('#1e3a5f'),
}


# ─── Custom Flowables ────────────────────────────────────────────────────────

class GradientRect(Flowable):
    """A gradient rectangle for decorative purposes."""
    def __init__(self, width, height, color1, color2, horizontal=True):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color1 = color1
        self.color2 = color2
        self.horizontal = horizontal

    def draw(self):
        canvas = self.canv
        steps = 50
        for i in range(steps):
            t = i / float(steps)
            r = self.color1.red + (self.color2.red - self.color1.red) * t
            g = self.color1.green + (self.color2.green - self.color1.green) * t
            b = self.color1.blue + (self.color2.blue - self.color1.blue) * t
            canvas.setFillColor(Color(r, g, b))
            if self.horizontal:
                x = self.width * i / steps
                w = self.width / steps + 1
                canvas.rect(x, 0, w, self.height, fill=1, stroke=0)
            else:
                y = self.height * i / steps
                h = self.height / steps + 1
                canvas.rect(0, y, self.width, h, fill=1, stroke=0)


class SectionHeader(Flowable):
    """A styled section header with accent bar."""
    def __init__(self, text, width, level=1):
        Flowable.__init__(self)
        self.text = text
        self.width = width
        self.level = level
        self.height = 40 if level == 1 else 30

    def draw(self):
        canvas = self.canv
        if self.level == 1:
            # Accent bar
            canvas.setFillColor(COLORS['accent_blue'])
            canvas.roundRect(0, 0, 4, self.height - 5, 2, fill=1, stroke=0)
            # Section text
            canvas.setFillColor(COLORS['primary'])
            canvas.setFont("Helvetica-Bold", 16)
            canvas.drawString(14, self.height - 22, self.text)
            # Underline
            canvas.setStrokeColor(COLORS['border'])
            canvas.setLineWidth(0.5)
            canvas.line(14, -2, self.width - 20, -2)
        else:
            canvas.setFillColor(COLORS['accent_purple'])
            canvas.roundRect(0, 4, 3, self.height - 12, 1.5, fill=1, stroke=0)
            canvas.setFillColor(COLORS['secondary'])
            canvas.setFont("Helvetica-Bold", 13)
            canvas.drawString(12, self.height - 20, self.text)


class CodeBlock(Flowable):
    """A styled code block with dark background."""
    MAX_HEIGHT = 650  # Max height to fit in a page frame

    def __init__(self, code_text, width, title=None):
        Flowable.__init__(self)
        self.code_text = code_text
        self.block_width = width
        self.title = title
        self.line_height = 11
        self.padding = 12
        lines = code_text.split('\n')
        self.num_lines = len(lines)
        self.height = self.num_lines * self.line_height + self.padding * 2
        if title:
            self.height += 20
        # Cap height
        if self.height > self.MAX_HEIGHT:
            self.height = self.MAX_HEIGHT

    def draw(self):
        canvas = self.canv
        # Background
        canvas.setFillColor(COLORS['bg_code'])
        canvas.roundRect(0, 0, self.block_width - 30, self.height, 6, fill=1, stroke=0)

        y_offset = self.height - self.padding
        if self.title:
            canvas.setFillColor(COLORS['accent_cyan'])
            canvas.setFont("Helvetica-Bold", 8)
            canvas.drawString(self.padding, y_offset - 4, self.title)
            y_offset -= 18

        # Code text
        canvas.setFillColor(COLORS['bg_code_text'])
        canvas.setFont("Courier", 7.5)
        for line in self.code_text.split('\n'):
            if y_offset - 8 < self.padding:
                break  # Stop if we'd overflow
            # Truncate long lines
            if len(line) > 95:
                line = line[:92] + '...'
            canvas.drawString(self.padding, y_offset - 8, line)
            y_offset -= self.line_height


def split_code_block(code_text, width, title=None, max_lines=50):
    """Split a large code block into multiple smaller CodeBlock flowables."""
    lines = code_text.split('\n')
    if len(lines) <= max_lines:
        return [CodeBlock(code_text, width, title)]

    blocks = []
    for i in range(0, len(lines), max_lines):
        chunk = '\n'.join(lines[i:i + max_lines])
        t = title if i == 0 else (f"{title} (cont.)" if title else "(continued)")
        blocks.append(CodeBlock(chunk, width, t))
    return blocks


class QuoteBlock(Flowable):
    """A styled blockquote."""
    def __init__(self, text, width):
        Flowable.__init__(self)
        self.text = text
        self.block_width = width
        self.height = 40

    def draw(self):
        canvas = self.canv
        # Left accent bar
        canvas.setFillColor(COLORS['accent_blue'])
        canvas.roundRect(0, 0, 3, self.height, 1.5, fill=1, stroke=0)
        # Background
        canvas.setFillColor(HexColor('#eff6ff'))
        canvas.roundRect(8, 0, self.block_width - 50, self.height, 4, fill=1, stroke=0)
        # Text
        canvas.setFillColor(COLORS['accent'])
        canvas.setFont("Helvetica-Oblique", 10)
        canvas.drawString(16, self.height / 2 - 4, self.text)


class CoverPage(Flowable):
    """A full cover page flowable."""
    def __init__(self, width, height):
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def draw(self):
        canvas = self.canv

        # Background gradient (dark navy)
        steps = 80
        for i in range(steps):
            t = i / float(steps)
            r = 0.059 + (0.118 - 0.059) * t
            g = 0.090 + (0.227 - 0.090) * t
            b = 0.165 + (0.373 - 0.165) * t
            canvas.setFillColor(Color(r, g, b))
            y_pos = self.height * (1 - (i + 1) / steps)
            h = self.height / steps + 1
            canvas.rect(0, y_pos, self.width, h, fill=1, stroke=0)

        # Decorative geometric elements
        canvas.setFillColor(Color(1, 1, 1, 0.03))
        canvas.circle(self.width * 0.8, self.height * 0.7, 200, fill=1, stroke=0)
        canvas.circle(self.width * 0.1, self.height * 0.2, 150, fill=1, stroke=0)

        # Top accent line
        canvas.setStrokeColor(COLORS['highlight'])
        canvas.setLineWidth(3)
        canvas.line(60, self.height - 60, self.width - 60, self.height - 60)

        # Thin secondary line
        canvas.setStrokeColor(COLORS['accent_cyan'])
        canvas.setLineWidth(1)
        canvas.line(60, self.height - 68, self.width - 60, self.height - 68)

        # Main title
        canvas.setFillColor(white)
        canvas.setFont("Helvetica-Bold", 36)
        canvas.drawString(60, self.height - 180, "Grivora AI")

        # Subtitle
        canvas.setFillColor(COLORS['accent_cyan'])
        canvas.setFont("Helvetica", 16)
        canvas.drawString(60, self.height - 215, "Production Workflow Documentation")

        # Description
        canvas.setFillColor(Color(1, 1, 1, 0.75))
        canvas.setFont("Helvetica", 12)
        canvas.drawString(60, self.height - 260, "Complete System Architecture, Methodology & Technical Reference")

        # Decorative divider
        canvas.setStrokeColor(COLORS['highlight'])
        canvas.setLineWidth(2)
        canvas.line(60, self.height - 290, 200, self.height - 290)

        # Module boxes
        modules = [
            ("Data Analysis", COLORS['accent_blue']),
            ("Predictive ML", COLORS['accent_purple']),
            ("Business Intelligence", COLORS['accent_cyan']),
            ("Auto Dashboard", COLORS['success']),
            ("AI Chat", COLORS['warning']),
        ]
        y_start = self.height - 350
        for i, (name, color) in enumerate(modules):
            x = 60 + (i % 3) * 160
            y = y_start - (i // 3) * 50
            canvas.setFillColor(color)
            canvas.setFillAlpha(0.15)
            canvas.roundRect(x, y, 145, 38, 6, fill=1, stroke=0)
            canvas.setFillAlpha(1.0)
            canvas.setStrokeColor(color)
            canvas.setLineWidth(1)
            canvas.roundRect(x, y, 145, 38, 6, fill=0, stroke=1)
            canvas.setFillColor(white)
            canvas.setFont("Helvetica-Bold", 10)
            canvas.drawCentredString(x + 72.5, y + 14, name)

        # Technology badge
        canvas.setFillColor(Color(1, 1, 1, 0.08))
        canvas.roundRect(60, self.height - 500, self.width - 120, 55, 8, fill=1, stroke=0)
        canvas.setFillColor(Color(1, 1, 1, 0.6))
        canvas.setFont("Helvetica", 9)
        canvas.drawString(75, self.height - 465, "Built with: Python 3.10+ • Flask • Google Gemini 2.5 Flash • scikit-learn • Chart.js • Pandas")
        canvas.drawString(75, self.height - 480, "Formats: CSV • Excel • JSON • Parquet • XML • ODS • 38 REST API Endpoints")

        # Bottom section
        canvas.setStrokeColor(Color(1, 1, 1, 0.2))
        canvas.setLineWidth(0.5)
        canvas.line(60, 100, self.width - 60, 100)

        canvas.setFillColor(Color(1, 1, 1, 0.5))
        canvas.setFont("Helvetica", 10)
        canvas.drawString(60, 75, "Version 1.0")
        canvas.drawString(60, 58, f"Generated: {datetime.datetime.now().strftime('%B %d, %Y')}")

        canvas.setFillColor(Color(1, 1, 1, 0.3))
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(self.width - 60, 75, "Full Architecture Reference")
        canvas.drawRightString(self.width - 60, 58, "Confidential — Internal Documentation")


# ─── PDF Builder ─────────────────────────────────────────────────────────────

def parse_markdown_to_elements(md_text, page_width):
    """Parse the WORKFLOW.md content and convert to reportlab flowables."""
    elements = []
    lines = md_text.split('\n')
    content_width = page_width - 2 * inch

    styles = getSampleStyleSheet()

    # Custom styles
    style_body = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=15,
        textColor=COLORS['text_dark'],
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )

    style_h1 = ParagraphStyle(
        'H1Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=26,
        textColor=COLORS['primary'],
        spaceBefore=24,
        spaceAfter=12,
    )

    style_h2 = ParagraphStyle(
        'H2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=20,
        textColor=COLORS['accent'],
        spaceBefore=18,
        spaceAfter=8,
    )

    style_h3 = ParagraphStyle(
        'H3Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=COLORS['accent_purple'],
        spaceBefore=14,
        spaceAfter=6,
    )

    style_h4 = ParagraphStyle(
        'H4Custom',
        parent=styles['Heading4'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=COLORS['accent_blue'],
        spaceBefore=10,
        spaceAfter=4,
    )

    style_bullet = ParagraphStyle(
        'BulletCustom',
        parent=style_body,
        leftIndent=20,
        bulletIndent=8,
        spaceAfter=3,
    )

    style_sub_bullet = ParagraphStyle(
        'SubBulletCustom',
        parent=style_body,
        leftIndent=40,
        bulletIndent=28,
        spaceAfter=2,
        fontSize=9.5,
    )

    style_table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        textColor=COLORS['white'],
        alignment=TA_CENTER,
        leading=11,
    )

    style_table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        textColor=COLORS['text_dark'],
        alignment=TA_LEFT,
        leading=10,
    )

    i = 0
    in_code_block = False
    code_lines = []
    code_title = None
    skip_main_title = True  # Skip first H1 (we have cover page)

    # Table accumulator
    table_rows = []
    in_table = False

    def flush_table():
        nonlocal table_rows, in_table
        if table_rows and len(table_rows) >= 2:
            # Remove separator row (row 1 with dashes)
            header = table_rows[0]
            data_rows = table_rows[2:] if len(table_rows) > 2 else []

            num_cols = len(header)
            col_width = (content_width - 10) / num_cols

            # Build table data with Paragraphs
            t_data = []
            # Header row
            h_row = [Paragraph(escape_html(c.strip()), style_table_header) for c in header]
            t_data.append(h_row)

            for row in data_rows:
                # Pad row if needed
                while len(row) < num_cols:
                    row.append('')
                r = [Paragraph(escape_html(c.strip()), style_table_cell) for c in row[:num_cols]]
                t_data.append(r)

            if t_data:
                col_widths = [col_width] * num_cols
                t = Table(t_data, colWidths=col_widths)
                t_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), COLORS['table_header']),
                    ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 8.5),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('TOPPADDING', (0, 0), (-1, 0), 8),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
                    ('TOPPADDING', (0, 1), (-1, -1), 5),
                    ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ROUNDEDCORNERS', [4, 4, 4, 4]),
                ]
                # Alternate row colors
                for row_idx in range(1, len(t_data)):
                    if row_idx % 2 == 0:
                        t_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), COLORS['table_alt']))

                t.setStyle(TableStyle(t_style))
                elements.append(Spacer(1, 6))
                elements.append(t)
                elements.append(Spacer(1, 8))

        table_rows = []
        in_table = False

    def escape_html(text):
        """Escape special chars for reportlab XML."""
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        # Handle inline code
        text = re.sub(r'`([^`]+)`', r'<font face="Courier" size="8" color="#e94560">\1</font>', text)
        # Handle bold
        text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
        # Handle italic
        text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
        return text

    while i < len(lines):
        line = lines[i]

        # ── Code blocks ──
        if line.strip().startswith('```'):
            if in_code_block:
                # End code block
                code_text = '\n'.join(code_lines)
                if code_text.strip():
                    blocks = split_code_block(code_text, content_width, code_title)
                    elements.append(Spacer(1, 4))
                    for blk in blocks:
                        elements.append(blk)
                        elements.append(Spacer(1, 4))
                    elements.append(Spacer(1, 2))
                in_code_block = False
                code_lines = []
                code_title = None
            else:
                # Start code block
                in_code_block = True
                lang = line.strip().replace('```', '').strip()
                code_title = lang if lang else None
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # ── Tables ──
        if '|' in line and line.strip().startswith('|'):
            cells = [c.strip() for c in line.strip().split('|')]
            # Remove empty first/last from split
            if cells and cells[0] == '':
                cells = cells[1:]
            if cells and cells[-1] == '':
                cells = cells[:-1]

            if not in_table:
                in_table = True
                table_rows = []

            table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            flush_table()

        # ── Horizontal rules ──
        if line.strip() == '---' or line.strip() == '***':
            elements.append(Spacer(1, 8))
            elements.append(HRFlowable(
                width="100%",
                thickness=1,
                color=COLORS['border'],
                spaceBefore=4,
                spaceAfter=4,
            ))
            elements.append(Spacer(1, 4))
            i += 1
            continue

        # ── Empty lines ──
        if line.strip() == '':
            i += 1
            continue

        # ── Headings ──
        if line.startswith('# ') and not line.startswith('## '):
            if skip_main_title:
                skip_main_title = False
                i += 1
                continue
            text = line[2:].strip()
            elements.append(Spacer(1, 16))
            sh = SectionHeader(text, content_width, level=1)
            elements.append(sh)
            elements.append(Spacer(1, 10))
            i += 1
            continue

        if line.startswith('## '):
            text = line[3:].strip()
            # Major sections get a page break (sections, not subsections)
            if text.startswith('SECTION'):
                elements.append(PageBreak())
                elements.append(SectionHeader(text, content_width, level=1))
                elements.append(Spacer(1, 10))
            elif text.startswith('APPENDIX'):
                elements.append(PageBreak())
                elements.append(SectionHeader(text, content_width, level=1))
                elements.append(Spacer(1, 10))
            else:
                elements.append(SectionHeader(text, content_width, level=1))
                elements.append(Spacer(1, 8))
            i += 1
            continue

        if line.startswith('### '):
            text = line[4:].strip()
            elements.append(Spacer(1, 6))
            elements.append(SectionHeader(text, content_width, level=2))
            elements.append(Spacer(1, 4))
            i += 1
            continue

        if line.startswith('#### '):
            text = line[5:].strip()
            elements.append(Paragraph(escape_html(text), style_h4))
            i += 1
            continue

        # ── Blockquotes ──
        if line.startswith('>'):
            text = line.lstrip('> ').strip()
            text = text.strip('*').strip()
            qb = QuoteBlock(text, content_width)
            elements.append(Spacer(1, 4))
            elements.append(qb)
            elements.append(Spacer(1, 6))
            i += 1
            continue

        # ── Bullet points ──
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            indent_level = len(line) - len(line.lstrip())
            text = line.strip()[2:].strip()
            if indent_level > 2:
                bullet_char = '◦'
                style = style_sub_bullet
            else:
                bullet_char = '•'
                style = style_bullet
            elements.append(Paragraph(
                f'{bullet_char}  {escape_html(text)}',
                style
            ))
            i += 1
            continue

        # ── Numbered lists ──
        if re.match(r'^\s*\d+\.\s', line):
            text = re.sub(r'^\s*(\d+)\.\s', r'\1. ', line.strip())
            elements.append(Paragraph(escape_html(text), style_bullet))
            i += 1
            continue

        # ── Regular paragraph ──
        text = line.strip()
        if text:
            # Check if it's an italicized footer line
            if text.startswith('*') and text.endswith('*') and not text.startswith('**'):
                clean = text.strip('*').strip()
                footer_style = ParagraphStyle(
                    'Footer',
                    parent=style_body,
                    fontName='Helvetica-Oblique',
                    fontSize=9,
                    textColor=COLORS['text_light'],
                    alignment=TA_CENTER,
                    spaceBefore=12,
                )
                elements.append(Paragraph(escape_html(clean), footer_style))
            else:
                elements.append(Paragraph(escape_html(text), style_body))

        i += 1

    # Flush any remaining table
    if in_table:
        flush_table()

    return elements


def draw_cover_page(canvas_obj, doc):
    """Draw the cover page directly on canvas (first page only)."""
    page_width, page_height = A4
    canvas_obj.saveState()

    # Background gradient (dark navy)
    steps = 80
    for i in range(steps):
        t = i / float(steps)
        r = 0.059 + (0.118 - 0.059) * t
        g = 0.090 + (0.227 - 0.090) * t
        b = 0.165 + (0.373 - 0.165) * t
        canvas_obj.setFillColor(Color(r, g, b))
        y_pos = page_height * (1 - (i + 1) / steps)
        h = page_height / steps + 1
        canvas_obj.rect(0, y_pos, page_width, h, fill=1, stroke=0)

    # Decorative geometric elements
    canvas_obj.setFillColor(Color(1, 1, 1, 0.03))
    canvas_obj.circle(page_width * 0.8, page_height * 0.7, 200, fill=1, stroke=0)
    canvas_obj.circle(page_width * 0.1, page_height * 0.2, 150, fill=1, stroke=0)

    # Top accent line
    canvas_obj.setStrokeColor(COLORS['highlight'])
    canvas_obj.setLineWidth(3)
    canvas_obj.line(60, page_height - 60, page_width - 60, page_height - 60)

    # Thin secondary line
    canvas_obj.setStrokeColor(COLORS['accent_cyan'])
    canvas_obj.setLineWidth(1)
    canvas_obj.line(60, page_height - 68, page_width - 60, page_height - 68)

    # Main title
    canvas_obj.setFillColor(white)
    canvas_obj.setFont("Helvetica-Bold", 36)
    canvas_obj.drawString(60, page_height - 180, "Grivora AI")

    # Subtitle
    canvas_obj.setFillColor(COLORS['accent_cyan'])
    canvas_obj.setFont("Helvetica", 16)
    canvas_obj.drawString(60, page_height - 215, "Production Workflow Documentation")

    # Description
    canvas_obj.setFillColor(Color(1, 1, 1, 0.75))
    canvas_obj.setFont("Helvetica", 12)
    canvas_obj.drawString(60, page_height - 260, "Complete System Architecture, Methodology & Technical Reference")

    # Decorative divider
    canvas_obj.setStrokeColor(COLORS['highlight'])
    canvas_obj.setLineWidth(2)
    canvas_obj.line(60, page_height - 290, 200, page_height - 290)

    # Module boxes
    modules = [
        ("Data Analysis", COLORS['accent_blue']),
        ("Predictive ML", COLORS['accent_purple']),
        ("Business Intelligence", COLORS['accent_cyan']),
        ("Auto Dashboard", COLORS['success']),
        ("AI Chat", COLORS['warning']),
    ]
    y_start = page_height - 350
    for i, (name, color) in enumerate(modules):
        x = 60 + (i % 3) * 160
        y = y_start - (i // 3) * 50
        canvas_obj.setFillColor(color)
        canvas_obj.setFillAlpha(0.15)
        canvas_obj.roundRect(x, y, 145, 38, 6, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1.0)
        canvas_obj.setStrokeColor(color)
        canvas_obj.setLineWidth(1)
        canvas_obj.roundRect(x, y, 145, 38, 6, fill=0, stroke=1)
        canvas_obj.setFillColor(white)
        canvas_obj.setFont("Helvetica-Bold", 10)
        canvas_obj.drawCentredString(x + 72.5, y + 14, name)

    # Technology badge
    canvas_obj.setFillColor(Color(1, 1, 1, 0.08))
    canvas_obj.roundRect(60, page_height - 500, page_width - 120, 55, 8, fill=1, stroke=0)
    canvas_obj.setFillColor(Color(1, 1, 1, 0.6))
    canvas_obj.setFont("Helvetica", 9)
    canvas_obj.drawString(75, page_height - 465, "Built with: Python 3.10+ | Flask | Google Gemini 2.5 Flash | scikit-learn | Chart.js | Pandas")
    canvas_obj.drawString(75, page_height - 480, "Formats: CSV | Excel | JSON | Parquet | XML | ODS | 38 REST API Endpoints")

    # Bottom section
    canvas_obj.setStrokeColor(Color(1, 1, 1, 0.2))
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(60, 100, page_width - 60, 100)

    canvas_obj.setFillColor(Color(1, 1, 1, 0.5))
    canvas_obj.setFont("Helvetica", 10)
    canvas_obj.drawString(60, 75, "Version 1.0")
    canvas_obj.drawString(60, 58, f"Generated: {datetime.datetime.now().strftime('%B %d, %Y')}")

    canvas_obj.setFillColor(Color(1, 1, 1, 0.3))
    canvas_obj.setFont("Helvetica", 9)
    canvas_obj.drawRightString(page_width - 60, 75, "Full Architecture Reference")
    canvas_obj.drawRightString(page_width - 60, 58, "Confidential - Internal Documentation")

    canvas_obj.restoreState()


def add_page_number(canvas_obj, doc):
    """Add page numbers and header/footer to each page (except cover)."""
    page_num = doc.page
    if page_num == 1:
        return  # Skip cover page

    canvas_obj.saveState()

    # Header line
    canvas_obj.setStrokeColor(COLORS['border'])
    canvas_obj.setLineWidth(0.5)
    canvas_obj.line(inch, A4[1] - 0.6 * inch, A4[0] - inch, A4[1] - 0.6 * inch)

    # Header text
    canvas_obj.setFillColor(COLORS['text_light'])
    canvas_obj.setFont("Helvetica", 7.5)
    canvas_obj.drawString(inch, A4[1] - 0.53 * inch, "Grivora AI - Production Workflow Documentation")
    canvas_obj.drawRightString(A4[0] - inch, A4[1] - 0.53 * inch, "Technical Reference v1.0")

    # Footer line
    canvas_obj.setStrokeColor(COLORS['border'])
    canvas_obj.line(inch, 0.6 * inch, A4[0] - inch, 0.6 * inch)

    # Page number (centered)
    canvas_obj.setFillColor(COLORS['accent_blue'])
    canvas_obj.setFont("Helvetica-Bold", 9)
    canvas_obj.drawCentredString(A4[0] / 2, 0.4 * inch, f"- {page_num} -")

    # Footer text
    canvas_obj.setFillColor(COLORS['text_light'])
    canvas_obj.setFont("Helvetica", 7)
    canvas_obj.drawString(inch, 0.4 * inch, "Confidential")
    canvas_obj.drawRightString(A4[0] - inch, 0.4 * inch, datetime.datetime.now().strftime('%B %Y'))

    canvas_obj.restoreState()


def on_first_page(canvas_obj, doc):
    """Handler for the first page - draws cover page."""
    draw_cover_page(canvas_obj, doc)


def on_later_pages(canvas_obj, doc):
    """Handler for subsequent pages - adds header/footer."""
    add_page_number(canvas_obj, doc)


def generate_pdf():
    """Main PDF generation function."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(script_dir, 'WORKFLOW.md')
    pdf_path = os.path.join(script_dir, 'Grivora_AI_Thesis.pdf')

    print(f"[*] Reading: {md_path}")
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    print("[*] Building PDF document...")

    # Create document
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        leftMargin=inch,
        rightMargin=inch,
        title="Grivora AI - Production Workflow Documentation",
        author="Grivora AI Team",
        subject="Complete System Architecture, Methodology & Technical Reference",
        creator="Grivora AI PDF Generator",
    )

    elements = []
    page_width = A4[0]
    page_height = A4[1]

    # ── Cover Page (first page is blank content, cover drawn via canvas callback) ──
    # Small spacer - the actual cover is rendered by on_first_page canvas callback
    elements.append(Spacer(1, 1))
    elements.append(PageBreak())

    # ── Table of Contents Page ──
    toc_item_style = ParagraphStyle(
        'TOCItem',
        fontName='Helvetica',
        fontSize=11,
        leading=22,
        textColor=COLORS['text_dark'],
        leftIndent=0,
    )

    elements.append(SectionHeader("Table of Contents", page_width - 2 * inch, level=1))
    elements.append(Spacer(1, 16))

    # TOC entries
    toc_entries = [
        ("Section 1", "Platform Overview & Vision"),
        ("Section 2", "Full System Architecture"),
        ("Section 3", "Technology Stack"),
        ("Section 4", "Directory Structure"),
        ("Section 5", "Data Ingestion Pipeline"),
        ("Section 6", "Module 01: Data Analysis (DA)"),
        ("Section 7", "Module 02: Predictive Modeling & Analysis (PMA)"),
        ("Section 8", "Module 03: Business Intelligence & Analytics (BIA)"),
        ("Section 9", "Module 04: Auto Dashboard"),
        ("Section 10", "AI Agent System & LLM Layer"),
        ("Section 11", "REST API Reference"),
        ("Section 12", "Frontend Architecture"),
        ("Section 13", "Security & Configuration"),
        ("Section 14", "Data Flow Diagrams"),
        ("Section 15", "Production Deployment Guide"),
        ("Section 16", "Error Handling Strategy"),
        ("Section 17", "Performance Considerations"),
        ("Appendix A", "File Size & Format Quick Reference"),
        ("Appendix B", "Agent Decision Logic"),
        ("Appendix C", "KPI Detection Keywords"),
    ]

    for section, title in toc_entries:
        # Section number in accent color + title
        color = COLORS["accent_blue"]
        if section.startswith("Appendix"):
            color = COLORS["accent_purple"]
        text = f'<font color="{color}" face="Helvetica-Bold">{section}</font>  --  {title}'
        elements.append(Paragraph(text, toc_item_style))

    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="60%", thickness=1, color=COLORS['border'], spaceAfter=8))
    elements.append(PageBreak())

    # ── Parse and add main content ──
    print("[*] Parsing markdown content...")
    content_elements = parse_markdown_to_elements(md_content, page_width)
    elements.extend(content_elements)

    # ── Build PDF ──
    print("[*] Rendering PDF pages...")
    doc.build(elements, onFirstPage=on_first_page, onLaterPages=on_later_pages)

    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"\n[OK] PDF generated successfully!")
    print(f"[>] Location: {pdf_path}")
    print(f"[>] Size: {file_size_mb:.2f} MB")

    return pdf_path


if __name__ == '__main__':
    generate_pdf()

