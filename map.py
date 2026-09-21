import logging
import os
import re
from html import escape
from typing import Dict, List, Optional

import ee
import folium
import pandas as pd
from folium import plugins


def _add_ee_layer(
    m: folium.Map,
    img: ee.Image,
    vis: dict,
    name: str,
    opacity: float = 1.0,
    shown: bool = True,
) -> None:
    if img is None:
        return
    map_id = img.getMapId(vis)
    folium.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        overlay=True,
        control=True,
        name=name,
        opacity=opacity,
        show=shown,
    ).add_to(m)


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
    initial_zoom: int = 9,
) -> Optional[str]:

    try:
        center = roi.centroid().coordinates().getInfo()[::-1]
    except Exception:
        center = [float(df.latitude.mean()), float(df.longitude.mean())]

    m = folium.Map(location=center, zoom_start=initial_zoom, tiles="OpenStreetMap")
    title = f"{escape(city_name)} | {start_date} to {end_date} (end exclusive)"
    m.get_root().html.add_child(
        folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:8px">{title}</div>'
        )
    )

    lst_min, lst_max = (
        df["LST_Celsius"].quantile(0.02),
        df["LST_Celsius"].quantile(0.98),
    )
    lst_vis = {
        "min": float(lst_min),
        "max": float(lst_max) if lst_max > lst_min else float(lst_min + 1),
        "palette": ["#000080", "#00FFFF", "#FFFF00", "#FF8000", "#FF0000"],
    }
    ndvi_vis = {"min": -0.2, "max": 0.8, "palette": ["#8B4513", "#FFFFFF", "#00FF00"]}
    ndbi_vis = {"min": -0.5, "max": 0.5, "palette": ["#00FF00", "#FFFF00", "#FF0000"]}
    # WorldCover class codes are sparse; remap to consecutive palette positions.
    lc_img = lc_img.remap(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], list(range(11))
    )
    lc_vis = {
        "min": 0,
        "max": 10,
        "palette": [
            "006400",
            "ffbb22",
            "ffff4c",
            "f096ff",
            "fa0000",
            "b4b4b4",
            "f0f0f0",
            "0064c8",
            "0096a0",
            "00cf75",
            "fae6a0",
        ],
    }

    _add_ee_layer(m, lst_img, lst_vis, "Land Surface Temperature", shown=True)
    _add_ee_layer(m, ndvi_img, ndvi_vis, "Vegetation Index (NDVI)", shown=False)
    _add_ee_layer(m, ndbi_img, ndbi_vis, "Built-up Index (NDBI)", shown=False)
    _add_ee_layer(m, lc_img, lc_vis, "Land Cover", shown=False)

    if topo_img is not None and {"Elevation", "Slope"}.issubset(df.columns):
        elev_min = float(df["Elevation"].min())
        elev_max = float(df["Elevation"].max())
        elev_vis = {
            "min": elev_min,
            "max": elev_max,
            "palette": ["#006633", "#E6E699", "#B36B00"],
        }
        _add_ee_layer(
            m, topo_img.select("Elevation"), elev_vis, "Elevation", shown=False
        )

    if {"latitude", "longitude", "LST_Zone"}.issubset(df.columns):
        subset = df.sample(n=min(2000, len(df)), random_state=42)
        grp = folium.FeatureGroup(name="Relative LST Zones (Sampled Points)")
        for _, row in subset.iterrows():
            folium.CircleMarker(
                [row["latitude"], row["longitude"]],
                radius=2,
                color=zone_colors.get(row["LST_Zone"], "gray"),
                fill=True,
                fill_opacity=0.7,
                tooltip=f"LST: {row['LST_Celsius']:.1f}°C, Zone: {row['LST_Zone']}",
            ).add_to(grp)
        grp.add_to(m)

    if "Hot_Spot_Class" in df.columns and df["Hot_Spot_Class"].notna().any():
        hs = df[df["Hot_Spot_Class"].isin(["Hot Spot", "Cold Spot"])]
        grp = folium.FeatureGroup(name="Significant Hot & Cold Spots")
        color_map = {"Hot Spot": "red", "Cold Spot": "blue"}
        for _, row in hs.iterrows():
            folium.CircleMarker(
                [row["latitude"], row["longitude"]],
                radius=3,
                weight=2,
                color=color_map.get(row["Hot_Spot_Class"]),
                fill=False,
                tooltip=f"Class: {row['Hot_Spot_Class']}",
            ).add_to(grp)
        grp.add_to(m)

    plugins.Fullscreen().add_to(m)
    plugins.MiniMap(toggle_display=True).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding: 10px; opacity: 0.9;">
     <b>Legend</b><br><hr>
     <b>Land Surface Temperature</b><br>
     <i style="background: linear-gradient(to right, #000080, #00FFFF, #FFFF00, #FF0000); display: block; height: 10px; width: 100%;"></i>
     <span style="float: left;">Cool</span><span style="float: right;">Hot</span><br><hr>
     <b>Sampled Points</b><br>
    """
    for zone, color in zone_colors.items():
        legend_html += f'<i style="background:{color}; width: 15px; height: 15px; border-radius: 50%; float: left; margin-right: 5px;"></i>{zone}<br>'
    legend_html += """
     <hr><b>Spatial Clusters</b><br>
     <i style="background:transparent; width: 15px; height: 15px; border-radius: 50%; border: 2px solid red; float: left; margin-right: 5px;"></i>Hot Spot<br>
     <i style="background:transparent; width: 15px; height: 15px; border-radius: 50%; border: 2px solid blue; float: left; margin-right: 5px;"></i>Cold Spot<br>
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    slug = re.sub(r"[^a-z0-9_-]+", "_", city_name.lower()).strip("_") or "study_area"
    path = os.path.join(outputs_dir, f"{slug}_{start_date[:4]}_interactive_map.html")
    m.save(path)
    logging.info("Saved enhanced interactive map: %s", path)
    return path
