import os
import math
import uuid
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import arabic_reshaper
from io import BytesIO
from datetime import datetime
from bidi.algorithm import get_display
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# =========================================================
# 1. الإعدادات العامة والمسارات
# =========================================================
st.set_page_config(page_title="Municipality Valuation System", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# تأكد من وجود الخطوط في هذا المسار لتجنب الأخطاء
FONT_REG = os.path.join(BASE_DIR, "fonts", "Cairo-Regular.ttf")
FONT_BOLD = os.path.join(BASE_DIR, "fonts", "Cairo-Bold.ttf")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

# =========================================================
# 2. الدوال المساعدة (Helpers)
# =========================================================

def ar(txt):
    """معالجة النصوص العربية للعرض الصحيح"""
    if not txt: return ""
    return get_display(arabic_reshaper.reshape(str(txt)))

def fmt_currency(x):
    try: return f"{float(x):,.0f} ﷼"
    except: return "-"

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calc_confidence_score(comps_df):
    if comps_df.empty: return {"percent": 0, "level": "منخفضة", "text": "لا توجد بيانات"}
    n = len(comps_df)
    mean_dist = comps_df["المسافة (كم)"].mean()
    vals = pd.to_numeric(comps_df["القيمة السنوية للعقد"], errors="coerce").dropna()
    if vals.empty: return None
    
    score_n = min(n / 10, 1.0)
    score_d = max(0, 1 - (mean_dist / 5))
    iqr = vals.quantile(0.75) - vals.quantile(0.25)
    score_v = max(0, 1 - (iqr / vals.median() if vals.median() != 0 else 1))
    
    pct = int(round((0.4 * score_n + 0.35 * score_d + 0.25 * score_v) * 100))
    level = "عالية" if pct >= 80 else "متوسطة" if pct >= 60 else "محدودة"
    return {"percent": pct, "level": level, "text": f"درجة الثقة {pct}% ({level})"}

# =========================================================
# 3. اختيار الصفقات والتوصية
# =========================================================

def select_comparable_deals(bank_df, site_coords, target_activity, top_n=10):
    if bank_df.empty or not site_coords: return pd.DataFrame()
    lat0, lon0 = site_coords
    df = bank_df.copy()
    df["المسافة (كم)"] = df.apply(lambda r: haversine_km(lat0, lon0, float(r["Latitude"]), float(r["Longitude"])), axis=1)
    
    # فلترة بسيطة (يمكن تحسينها حسب الحاجة)
    selected = df.sort_values("المسافة (كم)").head(top_n)
    return selected

def recommend_rent_advanced(comps_df, scenario_min, scenario_max):
    if comps_df.empty: return None
    vals = pd.to_numeric(comps_df["القيمة السنوية للعقد"], errors="coerce").dropna()
    med = vals.median()
    low = max(vals.quantile(0.25), scenario_min)
    high = min(vals.quantile(0.75), scenario_max)
    
    return {
        "low": low, "median": med, "high": high,
        "text": f"النطاق الموصى به: {fmt_currency(low)} - {fmt_currency(high)}"
    }

# =========================================================
# 4. واجهة Streamlit (UI)
# =========================================================

if "data_bank" not in st.session_state:
    # بيانات وهمية للتجربة
    st.session_state.data_bank = pd.DataFrame([
        {"رقم العقد": "101", "اسم المشروع": "برج السلام", "النشاط": "تجاري", "اسم الحي": "الملقا", "القيمة السنوية للعقد": 1200000, "Latitude": 24.714, "Longitude": 46.676},
        {"رقم العقد": "102", "اسم المشروع": "مجمع ريادة", "النشاط": "تجاري", "اسم الحي": "النخيل", "القيمة السنوية للعقد": 950000, "Latitude": 24.712, "Longitude": 46.674},
    ])

st.title("🏛️ منظومة التقييم الاستثماري البلدي")

with st.sidebar:
    st.header("إعدادات الموقع")
    coords_txt = st.text_input("إحداثيات (lat,lon)", "24.7136,46.6753")
    land_area = st.number_input("المساحة (م2)", value=2500)
    target_use = st.selectbox("الاستخدام", ["تجاري", "إداري", "سياحي"])

# معالجة الإحداثيات
try:
    lat0, lon0 = [float(x.strip()) for x in coords_txt.split(",")]
    site_coords = (lat0, lon0)
except:
    st.error("خطأ في تنسيق الإحداثيات")
    site_coords = None

# الحسابات المالية (الطريقة المتبقية)
st.subheader("💰 التحليل المالي")
col1, col2, col3 = st.columns(3)
with col1: gdv = st.number_input("القيمة التطويرية (GDV)", value=15_000_000)
with col2: cost = st.number_input("إجمالي التكاليف", value=9_000_000)
with col3: margin = st.slider("هامش الربح %", 10, 30, 20) / 100

residual = gdv - (cost + gdv * margin)
rent_est = max(residual * 0.08, 0)
rent_min, rent_max = rent_est * 0.9, rent_est * 1.1

# التحليل المكاني
comps_df = select_comparable_deals(st.session_state.data_bank, site_coords, target_use)
rec = recommend_rent_advanced(comps_df, rent_min, rent_max)
conf = calc_confidence_score(comps_df)

# عرض النتائج
if rec:
    st.success(f"✅ {rec['text']}")
    st.info(f"📊 {conf['text']}")

# الخريطة
st.subheader("🗺️ الخريطة التحليلية")
if site_coords:
    view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=14)
    site_layer = pdk.Layer("ScatterplotLayer", data=[{"lat": lat0, "lon": lon0}], get_position="[lon, lat]", get_radius=100, get_fill_color=[255, 0, 0])
    comp_layer = pdk.Layer("ScatterplotLayer", data=comps_df, get_position="[Longitude, Latitude]", get_radius=80, get_fill_color=[0, 0, 255])
    st.pydeck_chart(pdk.Deck(layers=[site_layer, comp_layer], initial_view_state=view_state))

# =========================================================
# 5. تصدير PDF (مختصر)
# =========================================================
def make_pdf():
    buf = BytesIO()
    pdfmetrics.registerFont(TTFont("Cairo", FONT_REG))
    c = canvas.Canvas(buf, pagesize=A4)
    # رسم النصوص (استخدم دالة ar للتعريب)
    c.setFont("Cairo", 14)
    c.drawRightString(19*cm, 27*cm, ar("تقرير تقييم عقاري"))
    c.setFont("Cairo", 10)
    c.drawRightString(19*cm, 26*cm, ar(f"المساحة: {land_area} م2"))
    c.drawRightString(19*cm, 25*cm, ar(f"التوصية: {rec['text'] if rec else ''}"))
    c.showPage()
    c.save()
    return buf.getvalue()

if st.button("📄 إصدار تقرير PDF"):
    pdf_data = make_pdf()
    st.download_button("تنزيل التقرير", pdf_data, "report.pdf", "application/pdf")
