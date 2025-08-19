import logging
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from esda.getisord import G_Local
from libpysal import weights


def run_hot_spot_analysis(df: pd.DataFrame, k_neighbors: int = 8) -> pd.DataFrame:
    """
    Performs Getis-Ord Gi* hot spot analysis on LST values.
    """
    df = df.copy()
    if not {"latitude", "longitude"}.issubset(df.columns) or len(df) < 3:
        df["Gi_ZScore"] = np.nan
        df["Gi_PValue"] = np.nan
        df["Hot_Spot_Class"] = "Not Enough Data"
        return df

    coords = df[["longitude", "latitude"]].values
    try:
        w = weights.KNN(coords, k=k_neighbors)
        w.transform = "R"
        if sum(w.cardinalities.values()) == 0:
            logging.warning("No neighbors found for any points with KNN(k=%d).", k_neighbors)
            df["Gi_ZScore"] = np.nan
            df["Gi_PValue"] = np.nan
            df["Hot_Spot_Class"] = "No Neighbors"
            return df
            
        lisa = G_Local(df["LST_Celsius"].values, w)
        df["Gi_ZScore"] = lisa.z_sim
        df["Gi_PValue"] = lisa.p_sim
        df["Hot_Spot_Class"] = "Not Significant"
        
        df.loc[(df["Gi_ZScore"] > 1.96) & (df["Gi_PValue"] < 0.05), "Hot_Spot_Class"] = "Hot Spot"
        df.loc[(df["Gi_ZScore"] < -1.96) & (df["Gi_PValue"] < 0.05), "Hot_Spot_Class"] = "Cold Spot"
        logging.info("Hot spot analysis complete. Found %d hot spots and %d cold spots.", 
                     (df["Hot_Spot_Class"] == "Hot Spot").sum(),
                     (df["Hot_Spot_Class"] == "Cold Spot").sum())
        return df
    except Exception as exc:
        logging.error("Hot spot analysis failed: %s", exc)
        df["Gi_ZScore"] = np.nan
        df["Gi_PValue"] = np.nan
        df["Hot_Spot_Class"] = "Analysis Error"
        return df


def plot_hot_spots(df: pd.DataFrame, save_path: str) -> None:
    """
    Plots and saves the spatial distribution of hot/cold spots to a specific file path.
    """
    plt.figure(figsize=(11, 9))
    
    hue_order = ["Hot Spot", "Cold Spot", "Not Significant", "Not Enough Data", "No Neighbors", "Analysis Error"]
    palette = {
        "Hot Spot": "darkred", "Cold Spot": "darkblue", "Not Significant": "lightgray",
        "Not Enough Data": "darkgrey", "No Neighbors": "black", "Analysis Error": "purple",
    }
    
    sns.scatterplot(
        data=df, x="longitude", y="latitude", hue="Hot_Spot_Class",
        hue_order=hue_order, palette=palette, s=25, alpha=0.8,
        edgecolor="w", linewidth=0.3,
    )
    plt.title("LST Hot/Cold Spots (Getis-Ord Gi*)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # FIX: Use the 'save_path' directly instead of constructing a new path.
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved hot spot analysis plot: %s", save_path)
