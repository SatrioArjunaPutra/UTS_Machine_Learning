"""Helper reportlab: gaya teks, tabel dari CSV, gambar. Font DejaVu dari matplotlib
(selalu tersedia) supaya simbol seperti ≥, ×, −, ² tercetak benar."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer, Table, TableStyle

_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(_FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(_FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                              boldItalic="DejaVu-Bold")

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
TEXT_W = PAGE_W - 2 * MARGIN
ACCENT = colors.HexColor("#1f4e79")

ST = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=17, leading=21, spaceAfter=6, textColor=ACCENT),
    "sub": ParagraphStyle("sub", fontName="DejaVu", fontSize=9.5, leading=13, textColor=colors.HexColor("#444")),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=13, leading=16, spaceBefore=8, spaceAfter=4, keepWithNext=1,
                         textColor=ACCENT),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2, keepWithNext=1),
    "p": ParagraphStyle("p", fontName="DejaVu", fontSize=9.3, leading=12.3, alignment=TA_JUSTIFY, spaceAfter=3),
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=9.3, leading=12.3, leftIndent=10,
                             bulletIndent=2, spaceAfter=1),
    "cap": ParagraphStyle("cap", fontName="DejaVu-Oblique", fontSize=7.6, leading=9.5,
                          textColor=colors.HexColor("#555"), spaceAfter=5),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=7.0, leading=8.4),
    "cellb": ParagraphStyle("cellb", fontName="DejaVu-Bold", fontSize=7.0, leading=8.4, textColor=colors.white),
}


def _esc(t: str) -> str:
    """Escape '<' yang bukan tag (mis. 'latih < 1 Jul') agar parser XML reportlab tidak gagal."""
    return re.sub(r"<(?=[\s\d=−-])", "&lt;", t)


def H1(t): return Paragraph(_esc(t), ST["h1"])
def H2(t): return Paragraph(_esc(t), ST["h2"])
def P(t): return Paragraph(_esc(t), ST["p"])
def CAP(t): return Paragraph(_esc(t), ST["cap"])
def BUL(items): return [Paragraph(_esc(i), ST["bullet"], bulletText="•") for i in items]


def fmt(v, nd=3):
    """Format angka gaya Indonesia: ribuan '.', desimal ','."""
    if isinstance(v, bool):
        return "ya" if v else "tidak"
    if isinstance(v, float):
        if pd.isna(v):
            return "–"
        if abs(v) >= 1000:
            return f"{v:,.0f}".replace(",", ".")
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v:.0f}"
        return s.replace(".", ",")
    if isinstance(v, int):
        return f"{v:,}".replace(",", ".")
    return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fit_words(df: pd.DataFrame, widths: list, font: float, nd: int) -> list:
    """Lebarkan kolom agar kata terpanjang (header maupun isi) tidak terpotong di tengah;
    kekurangannya diambil dari kolom yang masih longgar. Bila total kebutuhan melebihi
    lebar halaman, lebar semula dipakai."""
    pad = 6 + 1
    need = []
    for j, c in enumerate(df.columns):
        words = [(w, "DejaVu-Bold") for w in str(c).replace("_", " ").split()]
        words += [(w, "DejaVu") for v in df.iloc[:, j].tolist() for w in fmt(v, nd).split()]
        need.append(max((pdfmetrics.stringWidth(w.replace("&amp;", "&"), f, font) for w, f in words), default=0) + pad)
    if sum(need) > TEXT_W:
        return widths
    out = [max(w, n) for w, n in zip(widths, need)]
    extra = sum(out) - TEXT_W
    slack = [o - n for o, n in zip(out, need)]
    tot = sum(slack)
    return [o - extra * s / tot for o, s in zip(out, slack)] if extra > 0 and tot > 0 else out


def table(df: pd.DataFrame, widths=None, nd=3, highlight_rows=(), font=7.0):
    font = font * 1.08
    cell = ParagraphStyle("c", parent=ST["cell"], fontSize=font, leading=font * 1.2)
    head = ParagraphStyle("h", parent=ST["cellb"], fontSize=font, leading=font * 1.2)
    data = [[Paragraph(str(c).replace("_", " "), head) for c in df.columns]]
    for _, r in df.iterrows():
        data.append([Paragraph(fmt(v, nd), cell) for v in r.tolist()])
    if widths is None:
        widths = [TEXT_W / len(df.columns)] * len(df.columns)
    else:
        s = sum(widths)
        widths = [w * TEXT_W / s for w in widths]
    widths = _fit_words(df, widths, font, nd)
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
             ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d0da")),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    for i in highlight_rows:
        style.append(("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor("#fff2cc")))
    t.setStyle(TableStyle(style))
    return t


def img(path: Path, width_frac: float = 1.0, max_h_cm: float = 9.0):
    from PIL import Image as PILImage
    w_px, h_px = PILImage.open(path).size
    w = TEXT_W * width_frac
    h = w * h_px / w_px
    if h > max_h_cm * cm:
        h = max_h_cm * cm
        w = h * w_px / h_px
    return Image(str(path), width=w, height=h)


def figure(path: Path, caption: str, width_frac=1.0, max_h_cm=9.0):
    return KeepTogether([img(path, width_frac, max_h_cm), CAP(caption)])


def side_by_side(paths, caption, max_h_cm=6.0):
    n = len(paths)
    ims = [img(p, 1.0 / n - 0.02, max_h_cm) for p in paths]
    t = Table([ims], colWidths=[TEXT_W / n] * n)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return KeepTogether([t, CAP(caption)])


SP = lambda h=4: Spacer(1, h)
