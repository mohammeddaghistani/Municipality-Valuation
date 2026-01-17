# =========================================================
# app.py — Municipality Valuation (Modular)
# Requires:
#   core_valuation.py
#   report_engine.py
# Optional (for PDF map image):
#   maps_engine.py  (make_static_map_image)
# Files:
#   logo.png
#   fonts/Cairo-Regular.ttf
#   fonts/Cairo-Bold.ttf
# =========================================================

import os
import re
from io import BytesIO
from datetime import datetime
import uuid

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

from core_valuation import (
    residual_value,
    estimated_rent,
    build_scenarios,
    sensitivity_matrix,
    select_comparable_deals,
    recommend_rent_advanced,
    calc_confidence_score,
)

# maps_engine is optional (only used for static map image for PDF)
try:
    from maps_engine import make_static_map_image
except Exception:
    make_static_map_image = None

from report_engine import (
    make_pdf_report,
    make_excel_report,
)

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Municipality Valuation System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# Paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOGO_FILE = os.path.join(BASE_DIR, "logo.png")
CAIRO_REG = os.path.join(BASE_DIR, "fonts", "Cairo-Regular.ttf")
CAIRO_BOLD = os.path.join(BASE_DIR, "fonts", "Cairo-Bold.ttf")

BANK_CSV = os.path.join(DATA_DIR, "data_bank.csv")
MAP_IMG_PATH = os.path.join(DATA_DIR, "map_static.png")

# =========================================================
# Branding CSS (mdaghistani style)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;800&display=swap');

:root{
  --bg:#070A0F;
  --panel:#0E1420;
  --panel2:#101B2A;
  --text:#F2E9D3;
  --muted:#9AA6B2;
  --gold:#B08A3A;
  --gold2:#8E6B2A;
  --green:#80A070;
  --border:rgba(255,255,255,0.08);
  --shadow: 0 18px 28px -20px rgba(0,0,0,0.65);
}

html, body, [class*="css"]{
  font-family:'Cairo', sans-serif;
  direction: rtl;
  text-align: right;
}

.stApp{
  background:
    radial-gradient(1200px 700px at 70% -10%, rgba(176,138,58,0.18), transparent 55%),
    radial-gradient(900px 600px at 10% 0%, rgba(128,160,112,0.10), transparent 55%),
    linear-gradient(180deg, var(--bg) 0%, #05070B 100%);
}

.block-container{ padding-top: 1.0rem; padding-bottom: 2.2rem; }

.main-header{
  background:
    radial-gradient(circle at 20% 25%, rgba(176,138,58,0.20), transparent 48%),
    radial-gradient(circle at 85% 20%, rgba(128,160,112,0.14), transparent 52%),
    linear-gradient(135deg, #0A0F18 0%, #0E1420 55%, #0A0F18 100%);
  border:1px solid var(--border);
  padding: 26px 22px;
  border-radius: 18px;
  color: var(--text);
  text-align: center;
  margin-bottom: 18px;
  box-shadow: var(--shadow);
  position: relative;
  overflow:hidden;
}
.main-header .badge{
  display:inline-block;
  margin-top:10px;
  padding:6px 12px;
  border-radius: 999px;
  background: rgba(176,138,58,0.10);
  border: 1px solid rgba(176,138,58,0.25);
  color: var(--text);
  font-size: 0.88rem;
}

.glass-card{
  background: linear-gradient(180deg, rgba(14,20,32,0.92) 0%, rgba(10,15,24,0.92) 100%);
  border: 1px solid var(--border);
  padding: 18px 16px;
  border-radius: 18px;
  box-shadow: var(--shadow);
  margin-bottom: 14px;
}

.metric-card{
  background: linear-gradient(180deg, rgba(16,27,42,0.95) 0%, rgba(14,20,32,0.95) 100%);
  border: 1px solid rgba(176,138,58,0.18);
  border-right: 6px solid var(--gold);
  padding: 14px 12px;
  border-radius: 14px;
  box-shadow: 0 10px 22px -18px rgba(0,0,0,0.75);
  text-align:center;
}
.metric-label{ color: var(--muted); font-size: 0.92rem; margin-bottom: 6px; }
.metric-value{ color: var(--text); font-size: 1.4rem; font-weight: 900; letter-spacing:0.2px; }
.metric-value span{ color: var(--gold); }

.stTabs [data-baseweb="tab-list"]{ gap: 10px; }
.stTabs [data-baseweb="tab"]{
  font-weight: 800;
  color: var(--muted);
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(14,20,32,0.75);
}
.stTabs [aria-selected="true"]{
  color: var(--text) !important;
  background: rgba(176,138,58,0.16) !important;
  border: 1px solid rgba(176,138,58,0.35) !important;
}

.stButton > button{
  background: linear-gradient(135deg, rgba(176,138,58,0.95) 0%, rgba(142,107,42,0.95) 100%) !important;
  color: #0B0E14 !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 12px !important;
  font-weight: 900 !important;
}
.stButton > button:hover{
  filter: brightness(1.05);
  border: 1px solid rgba(176,138,58,0.55) !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Utilities
# =========================================================
BANK_COLS = [
    "رقم العقد", "اسم المشروع", "النشاط الرئيسي", "النشاط الفرعي", "النشاط", "اسم الحي",
    "القيمة السنوية للعقد", "القيمة الإجمالية لكامل مدةاللعقد", "مدة العقد",
    "نسبة الزيادة السنوية لكل خمسة سنوات", "نسبةفترة التجهيز والإنشاء",
    "تاريخ الصفقة(المقصود بها توقيع العقد)", "اسم المستثمر الحالي",
    "رابط الموقع", "Latitude", "Longitude",
    "سعر المتر"
]

def ensure_bank_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in BANK_COLS:
        if c not in df.columns:
            df[c] = None
    return df[BANK_COLS]

def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def fmt_currency(x):
    try:
        return f"{float(x):,.0f} ﷼"
    except Exception:
        return "-"

def build_deal_key(row: dict) -> str:
    cn = normalize_text(row.get("رقم العقد", ""))
    if cn:
        return f"CN|{cn}"
    parts = [
        normalize_text(row.get("اسم المشروع", "")),
        normalize_text(row.get("اسم الحي", "")),
        normalize_text(row.get("رابط الموقع", "")),
        normalize_text(row.get("النشاط", "")),
        normalize_text(row.get("القيمة السنوية للعقد", "")),
    ]
    return "FP|" + "|".join(parts)

def extract_lat_lng(url: str):
    if not url:
        return None
    m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)", url)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"[?&]ll=(-?\d+\.\d+),(-?\d+\.\d+)", url)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None

def parse_coords(coords_txt: str):
    if not coords_txt:
        return None
    try:
        parts = [p.strip() for p in coords_txt.split(",")]
        if len(parts) != 2:
            return None
        lat = float(parts[0])
        lng = float(parts[1])
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return None
        return (lat, lng)
    except Exception:
        return None

def load_bank_from_disk() -> pd.DataFrame:
    if os.path.exists(BANK_CSV):
        try:
            df = pd.read_csv(BANK_CSV, encoding="utf-8-sig")
            return ensure_bank_cols(df)
        except Exception:
            pass
    return ensure_bank_cols(pd.DataFrame())

def save_bank_to_disk(df: pd.DataFrame) -> bool:
    try:
        df2 = ensure_bank_cols(df)
        df2.to_csv(BANK_CSV, index=False, encoding="utf-8-sig")
        return True
    except Exception:
        return False

def import_deals_from_excel(uploaded_file) -> dict:
    df = pd.read_excel(uploaded_file)
    df.columns = [normalize_text(c) for c in df.columns]

    expected = [
        "رقم العقد", "اسم المشروع",
        "القيمة الإجمالية لكامل مدةاللعقد",
        "القيمة السنوية للعقد",
        "نسبة الزيادة السنوية لكل خمسة سنوات",
        "مدة العقد",
        "نسبةفترة التجهيز والإنشاء",
        "تاريخ الصفقة(المقصود بها توقيع العقد)",
        "النشاط الرئيسي", "النشاط الفرعي",
        "اسم الحي",
        "اسم المستثمر الحالي",
        "رابط الموقع"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = None

    bank = ensure_bank_cols(st.session_state.data_bank.copy())
    existing = set(bank.apply(lambda r: build_deal_key(r.to_dict()), axis=1).tolist())

    added = 0
    skipped = 0
    missing_geo = 0
    new_rows = []

    for _, r in df.iterrows():
        row = {c: r.get(c, None) for c in df.columns}
        main_act = normalize_text(row.get("النشاط الرئيسي", ""))
        sub_act = normalize_text(row.get("النشاط الفرعي", ""))
        act = main_act if not sub_act else f"{main_act} - {sub_act}"
        map_link = normalize_text(row.get("رابط الموقع", ""))

        coords = extract_lat_lng(map_link) if map_link else None

        new = {
            "رقم العقد": normalize_text(row.get("رقم العقد", "")) or None,
            "اسم المشروع": row.get("اسم المشروع", None),
            "النشاط الرئيسي": main_act or None,
            "النشاط الفرعي": sub_act or None,
            "النشاط": act or None,
            "اسم الحي": row.get("اسم الحي", None),
            "القيمة السنوية للعقد": row.get("القيمة السنوية للعقد", None),
            "القيمة الإجمالية لكامل مدةاللعقد": row.get("القيمة الإجمالية لكامل مدةاللعقد", None),
            "مدة العقد": row.get("مدة العقد", None),
            "نسبة الزيادة السنوية لكل خمسة سنوات": row.get("نسبة الزيادة السنوية لكل خمسة سنوات", None),
            "نسبةفترة التجهيز والإنشاء": row.get("نسبةفترة التجهيز والإنشاء", None),
            "تاريخ الصفقة(المقصود بها توقيع العقد)": row.get("تاريخ الصفقة(المقصود بها توقيع العقد)", None),
            "اسم المستثمر الحالي": row.get("اسم المستثمر الحالي", None),
            "رابط الموقع": map_link or None,
            "Latitude": coords[0] if coords else None,
            "Longitude": coords[1] if coords else None,
        }

        k = build_deal_key(new)
        if k in existing:
            skipped += 1
            continue

        if not coords:
            missing_geo += 1

        existing.add(k)
        new_rows.append(new)
        added += 1

    if new_rows:
        bank = pd.concat([bank, pd.DataFrame(new_rows)], ignore_index=True)
        st.session_state.data_bank = ensure_bank_cols(bank)

    return {"total": int(len(df)), "added": int(added), "skipped": int(skipped), "missing_geo": int(missing_geo)}

def deals_summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"count": 0, "geo_count": 0, "annual_min": None, "annual_median": None, "annual_max": None}
    geo = df.dropna(subset=["Latitude", "Longitude"])
    annual = pd.to_numeric(df["القيمة السنوية للعقد"], errors="coerce").dropna()
    return {
        "count": int(len(df)),
        "geo_count": int(len(geo)),
        "annual_min": float(annual.min()) if not annual.empty else None,
        "annual_median": float(annual.median()) if not annual.empty else None,
        "annual_max": float(annual.max()) if not annual.empty else None,
    }

def template_excel_bytes() -> bytes:
    out = BytesIO()
    cols = [
        "رقم العقد","اسم المشروع","القيمة الإجمالية لكامل مدةاللعقد","القيمة السنوية للعقد",
        "نسبة الزيادة السنوية لكل خمسة سنوات","مدة العقد","نسبةفترة التجهيز والإنشاء",
        "تاريخ الصفقة(المقصود بها توقيع العقد)","النشاط الرئيسي","النشاط الفرعي",
        "اسم الحي","اسم المستثمر الحالي","رابط الموقع"
    ]
    df = pd.DataFrame([{c: "" for c in cols}])
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Template", index=False)
    out.seek(0)
    return out.read()

def build_map_layers(site_coords, comps_df: pd.DataFrame):
    layers = []
    if site_coords is None:
        return layers

    site_df = pd.DataFrame([{
        "lat": float(site_coords[0]),
        "lng": float(site_coords[1]),
        "tooltip": "📍 العقار المستهدف"
    }])

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=site_df,
            get_position='[lng, lat]',
            get_radius=90,
            pickable=True,
            opacity=0.95,
            auto_highlight=True,
        )
    )

    if comps_df is not None and isinstance(comps_df, pd.DataFrame) and not comps_df.empty:
        df = comps_df.copy()
        if "Latitude" in df.columns and "Longitude" in df.columns:
            df = df.dropna(subset=["Latitude", "Longitude"]).copy()
            if not df.empty:
                df["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
                df["lng"] = pd.to_numeric(df["Longitude"], errors="coerce")
                df = df.dropna(subset=["lat", "lng"]).copy()
                if not df.empty:
                    df["tooltip"] = df.apply(
                        lambda r: f"🏷️ {normalize_text(r.get('اسم المشروع',''))}\n"
                                  f"📌 {normalize_text(r.get('اسم الحي',''))}\n"
                                  f"💰 {fmt_currency(r.get('القيمة السنوية للعقد', None))}",
                        axis=1
                    )
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=df,
                            get_position='[lng, lat]',
                            get_radius=70,
                            pickable=True,
                            opacity=0.75,
                            auto_highlight=True,
                        )
                    )
    return layers

# =========================================================
# Session init
# =========================================================
if "data_bank" not in st.session_state:
    st.session_state.data_bank = load_bank_from_disk()

if st.session_state.get("data_bank") is None or st.session_state.data_bank.empty:
    st.session_state.data_bank = load_bank_from_disk()

if "report_no" not in st.session_state:
    st.session_state.report_no = f"MV-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

if "report_date" not in st.session_state:
    st.session_state.report_date = datetime.now().strftime("%Y-%m-%d %H:%M")

# =========================================================
# Header
# =========================================================
st.markdown(
    "<div class='main-header'>"
    "<h1 style='margin:0; font-weight:900;'>🏛️ منظومة التقييم الاستثماري البلدي</h1>"
    "<div class='badge'>M. DAGHISTANI | Elite Business Strategy</div>"
    "</div>",
    unsafe_allow_html=True
)

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "💎 التقييم + الخريطة التحليلية",
    "📈 الحساسية + السيناريوهات",
    "📂 بنك الصفقات المرجعي",
    "📄 التقارير والتصدير",
])

# =========================================================
# TAB 1 — Valuation + Analytical Map (Makkah Default)
# =========================================================
with tab1:
    col_a, col_b = st.columns([1, 1.25])

    with col_a:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📍 موقع العقار المستهدف - مكة")
        coords_txt = st.text_input(
            "Lat,Lng",
            value="21.4225,39.8262",
            help="القيمة الافتراضية: مكة المكرمة (المسجد الحرام)"
        )

        coords = parse_coords(coords_txt)

        st.divider()
        st.subheader("💰 معطيات التقييم (Residual)")
        land_area = st.number_input("المساحة الإجمالية (م2)", value=1000.0, min_value=1.0)
        target_use = st.selectbox(
            "الاستخدام المستهدف",
            ["سكني حجاج/فندقي", "تجاري/إداري", "سياحي/ترفيهي", "خدمي/صحي"]
        )

        total_gdv = st.number_input("القيمة التطويرية النهائية (ريال)", value=50_000_000.0, step=500000.0)
        total_cost = st.number_input("تكاليف الإنشاء والرسوم (ريال)", value=30_000_000.0, step=500000.0)
        p_margin = st.slider("هامش الربح المستهدف (%)", 10, 40, 25) / 100.0
        cap_rate = st.slider("معدل الرسملة (Cap Rate) %", 4, 10, 6) / 100.0

        residual = residual_value(total_gdv, total_cost, p_margin)
        rent_est = estimated_rent(residual, cap_rate=cap_rate)
        rent_per_m2 = rent_est / land_area if land_area else 0.0
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>القيمة المتبقية للأرض</div>"
                f"<div class='metric-value'><span>{fmt_currency(residual)}</span></div></div>",
                unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>الدخل السنوي المتوقع</div>"
                f"<div class='metric-value'><span>{fmt_currency(rent_est)}</span></div></div>",
                unsafe_allow_html=True
            )
        with k3:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>سعر المتر الإيجاري</div>"
                f"<div class='metric-value'><span>{rent_per_m2:,.0f} ﷼</span></div></div>",
                unsafe_allow_html=True
            )

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🗺️ الخارطة الاستثمارية - مكة المكرمة")

        bank_df = ensure_bank_cols(st.session_state.data_bank)

        comps_for_map = pd.DataFrame()
        if coords and not bank_df.empty:
            try:
                comps_for_map = select_comparable_deals(
                    bank_df=bank_df,
                    site_coords=coords,
                    target_activity=target_use,
                    top_n=10,
                    min_same_activity=5
                )
            except Exception:
                comps_for_map = pd.DataFrame()

        layers = build_map_layers(coords, comps_for_map)

        if coords:
            view_state = pdk.ViewState(
                latitude=float(coords[0]),
                longitude=float(coords[1]),
                zoom=13,
                pitch=45
            )

            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=view_state,
                    map_style="carto-darkmatter",
                    layers=layers,
                    tooltip={"text": "{tooltip}"}
                ),
                use_container_width=True
            )
        else:
            st.warning("⚠️ صيغة الإحداثيات غير صحيحة. مثال: 21.4225,39.8262")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 2 — Sensitivity + Scenarios
# =========================================================
with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📈 مصفوفة الحساسية 5×5 (GDV × Cost)")
    st.caption("تعرض أثر تغير GDV والتكلفة على الإيجار السنوي (حسب Cap Rate).")

    sens = sensitivity_matrix(
        gdv=total_gdv,
        cost=total_cost,
        margin=p_margin,
        cap_rate=cap_rate,
        steps=(-0.2, -0.1, 0, 0.1, 0.2)
    )

    st.dataframe(
        sens.style.format("{:,.0f}").background_gradient(axis=None),
        use_container_width=True
    )

    st.divider()
    st.subheader("🧪 السيناريوهات (محافظ / أساسي / متفائل)")
    scen_df = build_scenarios(rent_est)
    st.dataframe(
        scen_df.rename(columns={"scenario": "السيناريو", "rent": "الإيجار"}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3 — Deals Bank (Excel import + manual add + save)
# =========================================================
with tab3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📂 بنك الصفقات المرجعي")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("💾 حفظ البنك", use_container_width=True):
            ok = save_bank_to_disk(st.session_state.data_bank)
            st.success("تم حفظ البنك ✅" if ok else "تعذر الحفظ ❌")

    with c2:
        if st.button("📥 استرجاع من الملف", use_container_width=True):
            st.session_state.data_bank = load_bank_from_disk()
            st.success("تم الاسترجاع ✅")

    with c3:
        st.download_button(
            "⬇️ تنزيل قالب Excel",
            data=template_excel_bytes(),
            file_name="deals_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    st.divider()
    st.subheader("📤 استيراد صفقات من Excel")
    up = st.file_uploader("ارفع ملف Excel (xlsx)", type=["xlsx"])
    if up is not None:
        try:
            r = import_deals_from_excel(up)
            st.success(
                f"✅ تمت القراءة | الإجمالي: {r['total']} | المضاف: {r['added']} | "
                f"المكرر: {r['skipped']} | بدون إحداثيات: {r['missing_geo']}"
            )
        except Exception as e:
            st.error(f"خطأ أثناء الاستيراد: {e}")

    st.divider()
    st.subheader("➕ إضافة صفقة يدويًا")
    with st.form("manual_deal", clear_on_submit=True):
        m1, m2, m3 = st.columns(3)
        cn = m1.text_input("رقم العقد (اختياري)")
        proj = m2.text_input("اسم المشروع")
        dist = m3.text_input("اسم الحي")

        m4, m5, m6 = st.columns(3)
        act_main = m4.text_input("النشاط الرئيسي")
        act_sub = m5.text_input("النشاط الفرعي")
        annual = m6.number_input("القيمة السنوية للعقد", min_value=0.0, step=10000.0)

        map_link = st.text_input("رابط الموقع (Google Maps)")
        lat = st.number_input("Latitude", value=0.0, format="%.6f")
        lng = st.number_input("Longitude", value=0.0, format="%.6f")

        ok_add = st.form_submit_button("حفظ الصفقة")

    if ok_add:
        act = act_main.strip() if not act_sub.strip() else f"{act_main.strip()} - {act_sub.strip()}"
        coords_m = None
        if map_link.strip():
            coords_m = extract_lat_lng(map_link.strip())
        if lat != 0.0 or lng != 0.0:
            coords_m = (lat, lng)

        new = {
            "رقم العقد": cn.strip() or None,
            "اسم المشروع": proj.strip() or None,
            "النشاط الرئيسي": act_main.strip() or None,
            "النشاط الفرعي": act_sub.strip() or None,
            "النشاط": act or None,
            "اسم الحي": dist.strip() or None,
            "القيمة السنوية للعقد": float(annual),
            "رابط الموقع": map_link.strip() or None,
            "Latitude": coords_m[0] if coords_m else None,
            "Longitude": coords_m[1] if coords_m else None,
        }
        df = ensure_bank_cols(st.session_state.data_bank)
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        st.session_state.data_bank = ensure_bank_cols(df)
        st.success("تمت الإضافة ✅")

    st.divider()
    st.subheader("📊 سجل الصفقات")
    st.dataframe(ensure_bank_cols(st.session_state.data_bank), use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 4 — Reports (PDF + Excel)
# =========================================================
with tab4:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📄 إصدار تقرير (PDF) + تصدير Excel")

    coords_r = parse_coords(coords_txt)
    bank_df = ensure_bank_cols(st.session_state.data_bank)

    comps_df = pd.DataFrame()
    if coords_r and not bank_df.empty:
        try:
            comps_df = select_comparable_deals(
                bank_df=bank_df,
                site_coords=coords_r,
                target_activity=target_use,
                top_n=10,
                min_same_activity=5
            )
        except Exception:
            comps_df = pd.DataFrame()

    scen_df = build_scenarios(rent_est)
    rent_min = float(scen_df["rent"].min())
    rent_max = float(scen_df["rent"].max())
    rent_range_txt = f"{fmt_currency(rent_min)} إلى {fmt_currency(rent_max)}"

    rec = recommend_rent_advanced(comps_df, rent_min, rent_max) if (comps_df is not None and not comps_df.empty) else None
    conf = calc_confidence_score(comps_df) if (comps_df is not None and not comps_df.empty) else None
    sens2 = sensitivity_matrix(total_gdv, total_cost, p_margin, cap_rate=cap_rate)

    map_ok = False
    if make_static_map_image and coords_r and comps_df is not None and not comps_df.empty:
        try:
            map_ok = bool(make_static_map_image(comps_df, coords_r, MAP_IMG_PATH))
        except Exception:
            map_ok = False

    payload = {
        "REG_YEAR": "2026",
        "report_no": st.session_state.report_no,
        "report_date": st.session_state.report_date,
        "target_use": target_use,
        "land_area": float(land_area),
        "coords_txt": coords_txt,
        "total_gdv": float(total_gdv),
        "total_cost": float(total_cost),
        "p_margin": float(p_margin),
        "residual": float(residual),
        "rent_est": float(rent_est),
        "rent_per_m2": float(rent_per_m2),
        "rent_range_txt": rent_range_txt,
        "grace_years": 2.0,
        "qr_url": "",
        "recommendation": rec,
        "confidence": conf,
        "comps_df": comps_df.to_dict(orient="records") if comps_df is not None and not comps_df.empty else [],
        "scenarios_df": scen_df.to_dict(orient="records"),
        "sensitivity_df": sens2,
        "map_image_path": MAP_IMG_PATH if (map_ok and os.path.exists(MAP_IMG_PATH)) else None,
        "deals_summary": deals_summary(bank_df),
    }

    b1, b2 = st.columns(2)
    with b1:
        if st.button("📄 توليد PDF متعدد الصفحات", use_container_width=True):
            try:
                pdf_bytes = make_pdf_report(
                    payload=payload,
                    logo_path=LOGO_FILE,
                    cairo_regular_path=CAIRO_REG,
                    cairo_bold_path=CAIRO_BOLD
                )
                st.download_button(
                    "⬇️ تنزيل PDF",
                    data=pdf_bytes,
                    file_name=f"{payload['report_no']}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"تعذر توليد PDF: {e}")

    with b2:
        if st.button("⬇️ تصدير Excel متعدد الشيتات", use_container_width=True):
            try:
                xbytes = make_excel_report(payload)
                st.download_button(
                    "⬇️ تنزيل Excel",
                    data=xbytes,
                    file_name=f"{payload['report_no']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"تعذر توليد Excel: {e}")

    st.divider()
    st.subheader("✅ ملخص جاهزية التقرير")
    s = deals_summary(bank_df)
    st.write(f"- عدد الصفقات في البنك: **{s['count']}**")
    st.write(f"- صفقات بإحداثيات: **{s['geo_count']}**")
    if rec:
        st.write(f"- التوصية: **{rec.get('text','')}**")
    if conf:
        st.write(f"- درجة الثقة: **{conf.get('text','')}**")

    st.markdown("</div>", unsafe_allow_html=True)
