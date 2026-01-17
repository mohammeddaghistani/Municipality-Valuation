# =========================================================
# Municipality Valuation – Full Integrated App
# =========================================================

import os
import math
import uuid
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import arabic_reshaper
from bidi.algorithm import get_display


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Municipality Valuation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# Paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
FONT_REG = os.path.join(BASE_DIR, "fonts", "Cairo-Regular.ttf")
FONT_BOLD = os.path.join(BASE_DIR, "fonts", "Cairo-Bold.ttf")


# =========================================================
# Helpers – Arabic
# =========================================================
def ar(txt):
    if txt is None:
        return ""
    reshaped = arabic_reshaper.reshape(str(txt))
    return get_display(reshaped)


def fmt_currency(x):
    try:
        return f"{float(x):,.0f} ﷼"
    except:
        return "-"


# =========================================================
# Fonts
# =========================================================
def ensure_pdf_fonts():
    try:
        pdfmetrics.registerFont(TTFont("Cairo", FONT_REG))
        pdfmetrics.registerFont(TTFont("Cairo-Bold", FONT_BOLD))
        return True
    except:
        return False


# =========================================================
# Distance (Haversine)
# =========================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# =========================================================
# Session State
# =========================================================
if "data_bank" not in st.session_state:
    st.session_state.data_bank = pd.DataFrame(columns=[
        "رقم العقد", "اسم المشروع", "النشاط", "اسم الحي",
        "القيمة السنوية للعقد", "Latitude", "Longitude"
    ])

if "report_no" not in st.session_state:
    st.session_state.report_no = f"MV-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


# =========================================================
# Header
# =========================================================
st.image(LOGO_PATH, width=160)
st.markdown("## 🏛️ منظومة التقييم الاستثماري البلدي")
st.caption("Decision Support System – Residual Method + Spatial Analysis")

st.divider()

# =========================================================
# Inputs
# =========================================================
c1, c2, c3 = st.columns(3)

with c1:
    land_area = st.number_input("المساحة (م2)", value=2500)

with c2:
    target_use = st.selectbox(
        "الاستخدام المستهدف",
        ["تجاري/إداري", "سياحي/ترفيهي", "خدمي/صحي"]
    )

with c3:
    coords_txt = st.text_input("إحداثيات الموقع (lat,lon)", "24.7136,46.6753")

try:
    lat0, lon0 = [float(x.strip()) for x in coords_txt.split(",")]
    site_coords = (lat0, lon0)
except:
    site_coords = None


# =========================================================
# Financials
# =========================================================
st.markdown("### 💰 معطيات التقييم")

f1, f2, f3 = st.columns(3)

with f1:
    gdv = st.number_input("القيمة التطويرية النهائية (GDV)", value=15_000_000)

with f2:
    cost = st.number_input("إجمالي التكاليف", value=9_000_000)

with f3:
    margin = st.slider("هامش الربح %", 10, 30, 20) / 100

residual = gdv - (cost + gdv * margin)
rent_est = max(residual * 0.08, 0)
rent_m2 = rent_est / land_area if land_area else 0

st.metric("الإيجار السنوي المقترح", fmt_currency(rent_est))
st.metric("سعر المتر الإيجاري", f"{rent_m2:,.2f} ﷼ / م2")

# =========================================================
# Scenarios
# =========================================================
scenarios = pd.DataFrame([
    {"السيناريو": "محافظ", "Rent": rent_est * 0.9},
    {"السيناريو": "أساسي", "Rent": rent_est},
    {"السيناريو": "متفائل", "Rent": rent_est * 1.1},
])

rent_min = scenarios["Rent"].min()
rent_max = scenarios["Rent"].max()

# =========================================================
# Comparable deals (dummy example if empty)
# =========================================================
if st.session_state.data_bank.empty:
    st.session_state.data_bank = pd.DataFrame([
        {"رقم العقد": "A1", "اسم المشروع": "مشروع 1", "النشاط": "تجاري",
         "اسم الحي": "الملقا", "القيمة السنوية للعقد": 900000,
         "Latitude": lat0+0.01, "Longitude": lon0+0.01},
        {"رقم العقد": "A2", "اسم المشروع": "مشروع 2", "النشاط": "تجاري",
         "اسم الحي": "الياسمين", "القيمة السنوية للعقد": 1_050_000,
         "Latitude": lat0-0.01, "Longitude": lon0-0.02},
    ])

bank = st.session_state.data_bank.copy()
bank["المسافة (كم)"] = bank.apply(
    lambda r: haversine_km(lat0, lon0, r["Latitude"], r["Longitude"]),
    axis=1
)

comps = bank.sort_values("المسافة (كم)").head(10)

# =========================================================
# Recommendation
# =========================================================
vals = comps["القيمة السنوية للعقد"]
q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
rec_low = max(q1, rent_min)
rec_high = min(q3, rent_max)

confidence = min(95, int(50 + len(comps)*4))

st.success(
    f"✅ التوصية النهائية: من {fmt_currency(rec_low)} إلى {fmt_currency(rec_high)}"
)
st.info(f"📊 درجة الثقة: {confidence}%")

# =========================================================
# PyDeck Map
# =========================================================
st.markdown("### 🗺️ الخريطة التحليلية")

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=comps,
        get_position="[Longitude, Latitude]",
        get_radius=200,
        get_fill_color=[0, 140, 255],
        pickable=True,
    ),
    pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([{"Latitude": lat0, "Longitude": lon0}]),
        get_position="[Longitude, Latitude]",
        get_radius=300,
        get_fill_color=[255, 191, 0],
    )
]

view = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=13, pitch=35)

st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view))

# =========================================================
# Static map for PDF
# =========================================================
MAP_IMG = os.path.join(DATA_DIR, "map.png")
plt.figure(figsize=(5,5))
plt.scatter(comps["Longitude"], comps["Latitude"], c="blue", s=80)
plt.scatter(lon0, lat0, c="gold", s=200, marker="*")
plt.savefig(MAP_IMG, dpi=200)
plt.close()

# =========================================================
# PDF
# =========================================================
def make_pdf():
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    ensure_pdf_fonts()
    w, h = A4

    c.setFont("Cairo-Bold", 18)
    c.drawRightString(w-2*cm, h-2*cm, ar("تقرير تقييم عقار بلدي"))

    c.setFont("Cairo", 11)
    c.drawRightString(w-2*cm, h-3*cm, ar(f"رقم التقرير: {st.session_state.report_no}"))
    c.drawRightString(w-2*cm, h-3.7*cm, ar(f"التاريخ: {datetime.now().strftime('%Y-%m-%d')}"))

    y = h-5*cm
    for k, v in [
        ("الاستخدام", target_use),
        ("الإيجار المقترح", fmt_currency(rent_est)),
        ("سعر المتر", f"{rent_m2:,.2f}"),
        ("التوصية", f"{fmt_currency(rec_low)} – {fmt_currency(rec_high)}"),
        ("درجة الثقة", f"{confidence}%"),
    ]:
        c.drawRightString(w-2*cm, y, ar(k))
        c.drawString(2*cm, y, ar(v))
        y -= 0.8*cm

    if os.path.exists(MAP_IMG):
        c.drawImage(MAP_IMG, 2*cm, y-8*cm, w-4*cm, 7*cm)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


if st.button("📄 تحميل تقرير PDF"):
    pdf = make_pdf()
    st.download_button(
        "⬇️ تنزيل PDF",
        data=pdf,
        file_name=f"{st.session_state.report_no}.pdf",
        mime="application/pdf"
    )
