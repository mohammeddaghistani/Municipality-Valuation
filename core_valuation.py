# =========================================================
# core_valuation.py
# Core Valuation & Decision Engine
# =========================================================

import math
import pandas as pd
import numpy as np

def residual_value(gdv, cost, margin):
    return gdv - (cost + gdv * margin)

def estimated_rent(residual, cap_rate=0.08):
    return max(residual * cap_rate, 0)

def build_scenarios(base_rent):
    return pd.DataFrame([
        {"scenario": "محافظ", "rent": base_rent * 0.90},
        {"scenario": "أساسي", "rent": base_rent},
        {"scenario": "متفائل", "rent": base_rent * 1.10},
    ])

def sensitivity_matrix(gdv, cost, margin, cap_rate=0.08, steps=(-0.2, -0.1, 0, 0.1, 0.2)):
    rows = []
    for dg in steps:
        for dc in steps:
            g = gdv * (1 + dg)
            c = cost * (1 + dc)
            r = residual_value(g, c, margin)
            rent = estimated_rent(r, cap_rate)
            rows.append({"GDV_%": int(dg * 100), "Cost_%": int(dc * 100), "Rent": rent})
    df = pd.DataFrame(rows)
    return df.pivot(index="GDV_%", columns="Cost_%", values="Rent")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def select_comparable_deals(bank_df, site_coords, target_activity, top_n=10, min_same_activity=5):
    if bank_df.empty or not site_coords: return pd.DataFrame()
    lat0, lon0 = site_coords
    df = bank_df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["distance_km"] = df.apply(lambda r: haversine_km(lat0, lon0, float(r["Latitude"]), float(r["Longitude"])), axis=1)
    df["main_activity"] = df["النشاط"].astype(str).str.split("-").str[0].str.strip()
    target_main = target_activity.split("/")[0].strip()
    same = df[df["main_activity"] == target_main].sort_values("distance_km")
    other = df[df["main_activity"] != target_main].sort_values("distance_km")
    sel = same.head(top_n) if len(same) >= min_same_activity else pd.concat([same, other.head(top_n - len(same))])
    sel = sel.sort_values("distance_km").head(top_n)
    sel["distance_km"] = sel["distance_km"].round(3)
    return sel

def recommend_rent_advanced(comps_df, scenario_min, scenario_max):
    if comps_df.empty or "القيمة السنوية للعقد" not in comps_df: return None
    vals = pd.to_numeric(comps_df["القيمة السنوية للعقد"], errors="coerce").dropna()
    if vals.empty: return None
    q1, med, q3 = vals.quantile(0.25), vals.median(), vals.quantile(0.75)
    low, high = max(q1, scenario_min), min(q3, scenario_max)
    if high <= low: low, high = max(med * 0.9, scenario_min), min(med * 1.1, scenario_max)
    return {"low": low, "median": med, "high": high, "count": len(vals), 
            "text": f"يوصى بإيجار {low:,.0f} – {high:,.0f} ريال بناء على {len(vals)} صفقات"}

def calc_confidence_score(comps_df):
    if comps_df.empty: return None
    n = len(comps_df)
    mean_dist = comps_df["distance_km"].mean()
    vals = pd.to_numeric(comps_df["القيمة السنوية للعقد"], errors="coerce").dropna()
    if vals.empty: return None
    iqr, med = vals.quantile(0.75) - vals.quantile(0.25), vals.median()
    score = (0.4 * min(n/10, 1.0)) + (0.35 * max(0, 1-(mean_dist/5))) + (0.25 * max(0, 1-(iqr/med if med else 0)))
    pct = int(round(score * 100))
    level = "عالية" if pct >= 80 else "متوسطة" if pct >= 60 else "محدودة"
    return {"percent": pct, "level": level, "text": f"درجة الثقة {pct}% ({level})"}
