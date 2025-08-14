import logging
import os
from typing import Dict, List, Optional

import ee
import folium
import pandas as pd


def _add_ee_layer(m: folium.Map, img: ee.Image, vis: dict, name: str, opacity: float = 1.0) -> None:
    try:
        if img is None:
            logging.warning("Skipping EE layer '%s': None image.", name)
            return
        map_id = ee.Image(img).getMapId(vis)
        folium.TileLayer(
            tiles=map_id["tile_fetcher"].url_format,
            attr="Google Earth Engine",
            overlay=True,
            control=True,
            name=name,
            opacity=opacity,
        ).add_to(m)
    except Exception as exc:
        logging.error("Failed to add EE layer '%s': %s", name, exc)


def create_map(
    roi: ee.Geometry,
    lst_img: ee.Image,
    ndvi_img: ee.Image,
    ndbi_img: ee.Image,
    lc_img: ee.Image,
    topo_img: ee.Image,
    df: pd.DataFrame,
    ordered_zones: List[str],
    zone_colors: Dict[str, str],
    outputs_dir: str,
    start_date: str,
    end_date: str,
    city_name: str,
) -> Optional[str]:
    try:
        centroid = roi.centroid().coordinates().getInfo()
        center = [centroid[1], centroid[0]]
    except Exception:
        center = [0.0, 0.0]

    m = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap")

    lst_min = float(df["LST_Celsius"].quantile(0.02))
    lst_max = float(df["LST_Celsius"].quantile(0.98))
    lst_vis = {"min": lst_min, "max": lst_max, "palette": ["#000080", "#0000FF", "#00FFFF", "#FFFF00", "#FF8000", "#FF0000", "#800000"]}
    ndvi_vis = {"min": -0.2, "max": 0.8, "palette": ["#8B4513", "#FFFFFF", "#00FF00"]}
    ndbi_vis = {"min": -0.5, "max": 0.5, "palette": ["#00FF00", "#FFFF00", "#FF0000"]}
    lc_vis = {"min": 10, "max": 95, "palette": ["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4", "#f0f0f0", "#0064c8", "#0096ff", "#6e6e6e"]}

    _add_ee_layer(m, lst_img, lst_vis, f"LST ({start_date}..{end_date})")
    _add_ee_layer(m, ndvi_img, ndvi_vis, "NDVI")
    _add_ee_layer(m, ndbi_img, ndbi_vis, "NDBI")
    _add_ee_layer(m, lc_img, lc_vis, "ESA WorldCover")
    if topo_img is not None:
        elev_vis = {"min": float(df["Elevation"].min()), "max": float(df["Elevation"].max()), "palette": ["#006633", "#E6E699", "#B36B00"]}
        slope_vis = {"min": 0.0, "max": float(df["Slope"].quantile(0.98)), "palette": ["#FFFFFF", "#B3B3B3", "#000000"]}
        _add_ee_layer(m, topo_img.select("Elevation"), elev_vis, "Elevation")
        _add_ee_layer(m, topo_img.select("Slope"), slope_vis, "Slope")

    # Boundary overlay
    try:
        roi_geojson = roi.getInfo()
        folium.GeoJson(
            roi_geojson,
            name="Study Area",
            style_function=lambda x: {"color": "yellow", "fillOpacity": 0.0, "weight": 3, "dashArray": "10,10"},
        ).add_to(m)
    except Exception:
        pass

    # Sampled points layer (UHI Zones)
    if {"latitude", "longitude", "LST_Zone"}.issubset(df.columns):
        subset = df.sample(n=min(2000, len(df)), random_state=42) if len(df) > 2000 else df
        grp = folium.FeatureGroup(name="Sampled Points (UHI Zones)")
        for _, row in subset.iterrows():
            color = zone_colors.get(row["LST_Zone"], "gray")
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=2,
                color="black",
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"LST: {row['LST_Celsius']:.1f}°C, Zone: {row['LST_Zone']}, LC: {row.get('Land_Cover_Type','')}",
            ).add_to(grp)
        grp.add_to(m)

    # Hot/Cold spots layer
    if "Hot_Spot_Class" in df.columns and df["Hot_Spot_Class"].notna().any():
        hs = df[df["Hot_Spot_Class"].isin(["Hot Spot", "Cold Spot"])]
        hs = hs if not hs.empty else df.sample(n=min(200, len(df)), random_state=42)
        grp = folium.FeatureGroup(name="LST Hot/Cold Spots")
        color_map = {"Hot Spot": "red", "Cold Spot": "blue", "Not Significant": "lightgray"}
        for _, row in hs.iterrows():
            col = color_map.get(row["Hot_Spot_Class"], "gray")
            tip = f"LST: {row['LST_Celsius']:.1f}°C, Class: {row['Hot_Spot_Class']}"
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color="black",
                weight=1,
                fill=True,
                fill_color=col,
                fill_opacity=0.8,
                tooltip=tip,
            ).add_to(grp)
        grp.add_to(m)

    folium.LayerControl(position="topright").add_to(m)

    path = os.path.join(outputs_dir, f"{city_name.replace(' ', '_').lower()}_enhanced_uhi_map.html")
    try:
        m.save(path)
        logging.info("Saved interactive map: %s", path)
        return path
    except Exception as exc:
        logging.error("Failed to save map: %s", exc)
        return None

