import os
from io import BytesIO
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display

def ar(text):
    if text is None: text = ""
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)

def fmt_currency(x):
    try: return f"{float(x):,.0f} ﷼"
    except: return "-"

def ensure_pdf_fonts(reg_path, bold_path):
    try:
        pdfmetrics.registerFont(TTFont("Cairo", reg_path))
        pdfmetrics.registerFont(TTFont("Cairo-Bold", bold_path))
        return True
    except: return False

def make_pdf_report(payload, logo_path, reg_path, bold_path):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    ok_fonts = ensure_pdf_fonts(reg_path, bold_path)
    f_reg, f_bold = ("Cairo", "Cairo-Bold") if ok_fonts else ("Helvetica", "Helvetica-Bold")
    
    # صفحة 1: الملخص التنفيذي
    c.setFont(f_bold, 16)
    c.drawRightString(W-2*cm, H-2*cm, ar("تقرير تقييم عقار بلدي"))
    c.setFont(f_reg, 10)
    c.drawRightString(W-2*cm, H-2.8*cm, ar(f"تاريخ التقرير: {payload.get('report_date', '')}"))
    
    rows = [("الاستخدام", payload.get('target_use')), ("المساحة", f"{payload.get('land_area', 0)} م2"),
            ("قيمة المتبقي", fmt_currency(payload.get('residual', 0))), ("الإيجار المقترح", fmt_currency(payload.get('rent_est', 0)))]
    
    y = H - 5*cm
    for k, v in rows:
        c.setFont(f_bold, 11)
        c.drawRightString(W-2*cm, y, ar(k))
        c.setFont(f_reg, 11)
        c.drawString(2*cm, y, ar(v))
        y -= 1*cm
    
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

def make_excel_report(payload):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame([payload]).to_excel(writer, sheet_name="Summary")
    out.seek(0)
    return out.read()
