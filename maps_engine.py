import os
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt

def build_pydeck_layers(comps_df, site_coords, recommendation=None):
    layers = []
    if site_coords:
        site_df = pd.DataFrame([{"lat": float(site_coords[0]), "lon": float(site_coords[1])}])
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=site_df, get_position="[lon, lat]",
            get_radius=120, get_fill_color=[255, 191, 0], pickable=True
        ))
    if comps_df is not None and not comps_df.empty:
        df = comps_df.copy()
        df["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df["annual_num"] = pd.to_numeric(df.get("القيمة السنوية للعقد", 0), errors="coerce")
        df = df.dropna(subset=["lat", "lon"])
        if not df.empty:
            max_val = max(float(df["annual_num"].max()), 1.0)
            df["color"] = df["distance_km"].apply(lambda d: [255, 215, 0] if d <= 1 else ([0, 160, 220] if d <= 3 else [150, 150, 150]))
            df["radius"] = df["annual_num"].apply(lambda v: 80 + (float(v) / max_val) * 220)
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=df, get_position="[lon, lat]",
                get_radius="radius", get_fill_color="color", pickable=True
            ))
    return layers

def make_static_map_image(comps_df, site_coords, out_path):
    if comps_df is None or comps_df.empty or not site_coords: return False
    df = comps_df.copy()
    df["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    if df.empty: return False
    plt.figure(figsize=(6, 6))
    plt.scatter(df["lon"], df["lat"], s=100, c="blue", alpha=0.5)
    plt.scatter([float(site_coords[1])], [float(site_coords[0])], s=300, c="gold", marker="*")
    plt.title("Map of Comparable Deals")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return True

def pydeck_view_state(site_coords, zoom=13, pitch=35):
    if not site_coords: return pdk.ViewState(latitude=24.71, longitude=46.67, zoom=11)
    return pdk.ViewState(latitude=float(site_coords[0]), longitude=float(site_coords[1]), zoom=zoom, pitch=pitch)
