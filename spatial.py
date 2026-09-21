import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from esda.getisord import G_Local
from libpysal import weights
from pyproj import CRS, Transformer
from scipy.stats import false_discovery_control


def run_hot_spot_analysis(df: pd.DataFrame, k_neighbors: int = 8) -> pd.DataFrame:
    """
    Performs Getis-Ord Gi* hot spot analysis on LST values.
    """
    df = df.copy()
    if (
        not {"latitude", "longitude"}.issubset(df.columns)
        or len(df) < 4
        or df["LST_Celsius"].nunique() < 2
    ):
        df["Gi_ZScore"] = np.nan
        df["Gi_PValue"] = np.nan
        df["Gi_QValue"] = np.nan
        df["Hot_Spot_Class"] = "Not Enough Data"
        return df

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={df.latitude.mean()} +lon_0={df.longitude.mean()} +datum=WGS84 +units=m"
    )
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    coords = np.column_stack(
        transformer.transform(df.longitude.to_numpy(), df.latitude.to_numpy())
    )
    try:
        # Leave at least one point outside each neighborhood including self.
        w = weights.KNN(coords, k=min(k_neighbors, len(df) - 2))
        if sum(w.cardinalities.values()) == 0:
            logging.warning(
                "No neighbors found for any points with KNN(k=%d).", k_neighbors
            )
            df["Gi_ZScore"] = np.nan
            df["Gi_PValue"] = np.nan
            df["Gi_QValue"] = np.nan
            df["Hot_Spot_Class"] = "No Neighbors"
            return df

        # Gi* includes self. Kelvin keeps this ratio statistic on a positive scale.
        # Use deterministic, two-sided normal-approximation inference with BH FDR.
        lisa = G_Local(
            df["LST_Celsius"].values + 273.15,
            w,
            star=True,
            transform="B",
            permutations=0,
        )
        df["Gi_ZScore"] = lisa.Zs
        df["Gi_PValue"] = np.minimum(1.0, 2 * lisa.p_norm)
        valid = np.isfinite(df["Gi_PValue"])
        df["Gi_QValue"] = np.nan
        if valid.any():
            df.loc[valid, "Gi_QValue"] = false_discovery_control(
                df.loc[valid, "Gi_PValue"].to_numpy()
            )
        df["Hot_Spot_Class"] = "Not Significant"

        df.loc[(df["Gi_ZScore"] > 0) & (df["Gi_QValue"] < 0.05), "Hot_Spot_Class"] = (
            "Hot Spot"
        )
        df.loc[(df["Gi_ZScore"] < 0) & (df["Gi_QValue"] < 0.05), "Hot_Spot_Class"] = (
            "Cold Spot"
        )
        df.loc[~valid, "Hot_Spot_Class"] = "Not Enough Data"
        logging.info(
            "Hot spot analysis complete. Found %d hot spots and %d cold spots.",
            (df["Hot_Spot_Class"] == "Hot Spot").sum(),
            (df["Hot_Spot_Class"] == "Cold Spot").sum(),
        )
        return df
    except Exception as exc:
        logging.error("Hot spot analysis failed: %s", exc)
        df["Gi_ZScore"] = np.nan
        df["Gi_PValue"] = np.nan
        df["Gi_QValue"] = np.nan
        df["Hot_Spot_Class"] = "Analysis Error"
        return df


def plot_hot_spots(df: pd.DataFrame, save_path: str) -> None:
    """
    Plots and saves the spatial distribution of hot/cold spots to a specific file path.
    """
    plt.figure(figsize=(11, 9))

    hue_order = [
        "Hot Spot",
        "Cold Spot",
        "Not Significant",
        "Not Enough Data",
        "No Neighbors",
        "Analysis Error",
    ]
    palette = {
        "Hot Spot": "darkred",
        "Cold Spot": "darkblue",
        "Not Significant": "lightgray",
        "Not Enough Data": "darkgrey",
        "No Neighbors": "black",
        "Analysis Error": "purple",
    }

    sns.scatterplot(
        data=df,
        x="longitude",
        y="latitude",
        hue="Hot_Spot_Class",
        hue_order=hue_order,
        palette=palette,
        s=25,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.3,
    )
    plt.title("LST Hot/Cold Spots (Gi*, normal approximation, BH q < 0.05)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved hot spot analysis plot: %s", save_path)
