import logging
import os
from typing import Dict, List, Optional

import ee
import folium
from folium import plugins
import pandas as pd
import geemap.foliumap as geemap


def _add_ee_layer(m: folium.Map, img: ee.Image, vis: dict, name: str, opacity: float = 1.0, shown: bool = True) -> None:
    try:
        if img is None: return
        geemap.ee_tile_layer(img, vis, name, shown=shown, opacity=opacity).add_to(m)
    except Exception:
        # Fallback for stability if geemap fails
        try:
            map_id = ee.Image(img).getMapId(vis)
            folium.TileLayer(
                tiles=map_id["tile_fetcher"].url_format, attr="Google Earth Engine",
                overlay=True, control=True, name=name, opacity=opacity, show=shown,
            ).add_to(m)
        except Exception as e:
            logging.error("Failed to add EE layer '%s' with both geemap and fallback: %s", name, e)

def create_map(
    roi: ee.Geometry, lst_img: ee.Image, ndvi_img: ee.Image, ndbi_img: ee.Image,
    lc_img: ee.Image, topo_img: ee.Image, df: pd.DataFrame, ordered_zones: List[str],
    zone_colors: Dict[str, str], outputs_dir: str, start_date: str, end_date: str, city_name: str,
) -> Optional[str]:
    
    try:
        center = roi.centroid().coordinates().getInfo()[::-1]
    except Exception:
        center = [0.0, 0.0]

    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")
    folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)
    folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google', name='Google Satellite').add_to(m)

    lst_min, lst_max = df["LST_Celsius"].quantile(0.02), df["LST_Celsius"].quantile(0.98)
    lst_vis = {"min": lst_min, "max": lst_max, "palette": ['#000080', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000']}
    ndvi_vis = {"min": -0.2, "max": 0.8, "palette": ['#8B4513', '#FFFFFF', '#00FF00']}
    ndbi_vis = {"min": -0.5, "max": 0.5, "palette": ['#00FF00', '#FFFF00', '#FF0000']}
    lc_vis = {"palette": ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000', '#b4b4b4', '#0064c8', '#0096ff', '#6e6e6e']}

    _add_ee_layer(m, lst_img, lst_vis, f"Land Surface Temperature", shown=True)
    _add_ee_layer(m, ndvi_img, ndvi_vis, "Vegetation Index (NDVI)", shown=False)
    _add_ee_layer(m, ndbi_img, ndbi_vis, "Built-up Index (NDBI)", shown=False)
    _add_ee_layer(m, lc_img, lc_vis, "Land Cover", shown=False)
    
    if topo_img and {"Elevation", "Slope"}.issubset(df.columns):
        # FIX: Explicitly cast NumPy min/max values to standard Python floats.
        elev_min = float(df["Elevation"].min())
        elev_max = float(df["Elevation"].max())
        elev_vis = {"min": elev_min, "max": elev_max, "palette": ['#006633', '#E6E699', '#B36B00']}
        _add_ee_layer(m, topo_img.select("Elevation"), elev_vis, "Elevation", shown=False)

    if {"latitude", "longitude", "LST_Zone"}.issubset(df.columns):
        subset = df.sample(n=min(2000, len(df)), random_state=42)
        grp = folium.FeatureGroup(name="UHI Zones (Sampled Points)")
        for _, row in subset.iterrows():
            folium.CircleMarker(
                [row["latitude"], row["longitude"]], radius=2,
                color=zone_colors.get(row["LST_Zone"], "gray"), fill=True, fill_opacity=0.7,
                tooltip=f"LST: {row['LST_Celsius']:.1f}°C, Zone: {row['LST_Zone']}"
            ).add_to(grp)
        grp.add_to(m)

    if "Hot_Spot_Class" in df.columns and df["Hot_Spot_Class"].notna().any():
        hs = df[df["Hot_Spot_Class"].isin(["Hot Spot", "Cold Spot"])]
        grp = folium.FeatureGroup(name="Significant Hot & Cold Spots")
        color_map = {"Hot Spot": "red", "Cold Spot": "blue"}
        for _, row in hs.iterrows():
            folium.CircleMarker(
                [row["latitude"], row["longitude"]], radius=3, weight=2,
                color=color_map.get(row["Hot_Spot_Class"]), fill=False,
                tooltip=f"Class: {row['Hot_Spot_Class']}"
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
    path = os.path.join(outputs_dir, f"{city_name.replace(' ', '_').lower()}_interactive_map.html")
    m.save(path)
    logging.info("Saved enhanced interactive map: %s", path)
    return path
