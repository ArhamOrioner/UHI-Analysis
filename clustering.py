import logging
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


def choose_optimal_clusters(
    df: pd.DataFrame, candidate_k: List[int], random_state: int
) -> int:
    """
    Determines the optimal number of clusters using the silhouette score method.
    """
    distinct = df["LST_Celsius"].nunique()
    if distinct < 2 or len(df) < 3:
        return max(1, min(distinct, len(df)))
    scores = []
    valid_k = []
    logging.info("Determining optimal clusters using silhouette score...")
    for k in candidate_k:
        if k <= 1 or k >= len(df) or k > distinct:
            continue
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(df[["LST_Celsius"]])
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(
                df[["LST_Celsius"]],
                labels,
                sample_size=min(2000, len(df)),
                random_state=random_state,
            )
            scores.append(score)
            valid_k.append(k)
            logging.info("  k=%d -> silhouette score=%.4f", k, score)
        except Exception as exc:
            logging.debug("Silhouette calculation failed for k=%d: %s", k, exc)
    if not scores:
        logging.warning(
            "Silhouette selection unavailable; using up to two distinct zones."
        )
        return min(2, distinct, len(df))

    best_k = valid_k[int(np.argmax(scores))]
    logging.info("Optimal number of clusters found: k=%d", best_k)
    return best_k


def kmeans_and_label(
    df: pd.DataFrame, k: int, random_state: int
) -> Tuple[pd.DataFrame, List[str], Dict[str, str], np.ndarray]:
    """
    Performs K-Means clustering and labels the data with descriptive UHI zone names.
    """
    if df.empty:
        raise ValueError("Cannot cluster an empty sample")
    k = max(1, min(k, len(df), df["LST_Celsius"].nunique()))

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(df[["LST_Celsius"]])
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)

    if k == 1:
        zone_names = ["Uniform_Zone"]
    elif k == 2:
        zone_names = ["Cool_Zone", "Hot_Zone"]
    elif k == 3:
        zone_names = ["Cool_Zone", "Mild_Zone", "Hot_Zone"]
    elif k == 4:
        zone_names = ["Cool_Zone", "Mild_Zone_1", "Mild_Zone_2", "Hot_Zone"]
    else:
        zone_names = [f"Zone_{i + 1}" for i in range(k)]

    id_to_name = {cluster_id: zone_names[i] for i, cluster_id in enumerate(order)}

    df = df.copy()
    df["LST_Cluster"] = labels
    df["LST_Zone"] = df["LST_Cluster"].map(id_to_name)

    from matplotlib.colors import to_hex

    zone_colors = {
        name: ZONE_COLORS.get(name, to_hex(plt.get_cmap("coolwarm")(i / max(1, k - 1))))
        for i, name in enumerate(zone_names)
    }
    ordered_zones = [id_to_name[idx] for idx in order]

    logging.info("UHI zones identified: %s", dict(df["LST_Zone"].value_counts()))
    return df, ordered_zones, zone_colors, centers


def plot_uhi_distribution(
    df: pd.DataFrame,
    ordered_zones: List[str],
    zone_colors: Dict[str, str],
    save_path: str,
) -> None:
    """
    Plots and saves the UHI zone distributions to a specified file path.
    """
    plt.figure(figsize=(14, 6))

    # Box plot of LST by zone
    plt.subplot(1, 2, 1)
    sns.boxplot(
        data=df,
        x="LST_Zone",
        y="LST_Celsius",
        order=ordered_zones,
        palette=zone_colors,
        hue="LST_Zone",
        legend=False,
    )
    plt.title("LST Distribution by Relative Temperature Zone")
    plt.xlabel("Relative Temperature Zone")
    plt.ylabel("LST (°C)")

    # Pie chart of zone distribution
    plt.subplot(1, 2, 2)
    counts = df["LST_Zone"].value_counts().reindex(ordered_zones, fill_value=0)
    colors = [zone_colors.get(z, "grey") for z in counts.index]
    plt.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    plt.title("Sample Proportions by Temperature Zone")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved UHI distribution plot: %s", save_path)
