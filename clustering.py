import logging
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


ZONE_COLORS = {
    "Cool_Zone": "#0000FF",
    "Mild_Zone": "#FFFF00",
    "Hot_Zone": "#FF0000",
    "Mild_Zone_1": "#87CEEB",
    "Mild_Zone_2": "#FFA07A",
}


def choose_optimal_clusters(df: pd.DataFrame, candidate_k: List[int], random_state: int) -> int:
    """
    Determines the optimal number of clusters using the silhouette score method.
    """
    if len(df) < 3:
        logging.warning("Not enough samples for silhouette-based selection. Defaulting to 2 clusters.")
        return 2
    scores = []
    valid_k = []
    logging.info("Determining optimal clusters using silhouette score...")
    for k in candidate_k:
        if k <= 1 or k >= len(df):
            continue
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(df[["LST_Celsius"]])
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(df[["LST_Celsius"]], labels)
            scores.append(score)
            valid_k.append(k)
            logging.info("  k=%d -> silhouette score=%.4f", k, score)
        except Exception as exc:
            logging.debug("Silhouette calculation failed for k=%d: %s", k, exc)
    if not scores:
        logging.warning("Silhouette score selection failed for all k values. Defaulting to k=3.")
        return 3
    
    best_k = valid_k[int(np.argmax(scores))]
    logging.info("Optimal number of clusters found: k=%d", best_k)
    return best_k


def kmeans_and_label(df: pd.DataFrame, k: int, random_state: int) -> Tuple[pd.DataFrame, List[str], Dict[str, str], np.ndarray]:
    """
    Performs K-Means clustering and labels the data with descriptive UHI zone names.
    """
    k = max(2, min(k, len(df) - 1)) if len(df) > 2 else 2
    
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(df[["LST_Celsius"]])
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)
    
    if k == 2:
        zone_names = ["Cool_Zone", "Hot_Zone"]
    elif k == 3:
        zone_names = ["Cool_Zone", "Mild_Zone", "Hot_Zone"]
    elif k == 4:
        zone_names = ["Cool_Zone", "Mild_Zone_1", "Mild_Zone_2", "Hot_Zone"]
    else:
        zone_names = [f"Zone_{i+1}" for i in range(k)]
        
    id_to_name = {cluster_id: zone_names[i] for i, cluster_id in enumerate(order)}
    
    df = df.copy()
    df["LST_Cluster"] = labels
    df["LST_Zone"] = df["LST_Cluster"].map(id_to_name)
    
    zone_colors = {name: ZONE_COLORS.get(name, "#A9A9A9") for name in zone_names}
    ordered_zones = [id_to_name[idx] for idx in order]
    
    logging.info("UHI zones identified: %s", dict(df["LST_Zone"].value_counts()))
    return df, ordered_zones, zone_colors, centers


def plot_uhi_distribution(df: pd.DataFrame, ordered_zones: List[str], zone_colors: Dict[str, str], save_path: str) -> None:
    """
    Plots and saves the UHI zone distributions to a specified file path.
    """
    plt.figure(figsize=(14, 6))
    
    # Box plot of LST by zone
    plt.subplot(1, 2, 1)
    # FIX: Address the FutureWarning by assigning the x-variable to hue and disabling the legend it creates.
    sns.boxplot(data=df, x="LST_Zone", y="LST_Celsius", order=ordered_zones, palette=zone_colors, hue="LST_Zone", legend=False)
    plt.title("LST Distribution by UHI Zone")
    plt.xlabel("UHI Zone")
    plt.ylabel("LST (°C)")
    
    # Pie chart of zone distribution
    plt.subplot(1, 2, 2)
    counts = df["LST_Zone"].value_counts().reindex(ordered_zones, fill_value=0)
    colors = [zone_colors.get(z, "grey") for z in counts.index]
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=90)
    plt.title("UHI Zone Proportions")
    
    plt.tight_layout()
    # FIX: Use the 'save_path' directly, instead of trying to construct a new path.
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved UHI distribution plot: %s", save_path)
