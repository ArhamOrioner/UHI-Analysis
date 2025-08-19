import logging
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_eda_plots(df: pd.DataFrame, outputs_dir: str) -> None:
    sns.set_style("whitegrid")

    # --- Distributions Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    sns.histplot(df, x="LST_Celsius", kde=True, ax=axes[0, 0], color="skyblue", edgecolor="black")
    axes[0, 0].set_title("Distribution of Land Surface Temperature", fontsize=16)
    axes[0, 0].set_xlabel("Land Surface Temperature (°C)")
    
    sns.histplot(df, x="NDVI", kde=True, ax=axes[0, 1], color="lightgreen", edgecolor="black")
    axes[0, 1].set_title("Distribution of Vegetation Index (NDVI)", fontsize=16)
    axes[0, 1].set_xlabel("Vegetation Index (NDVI)")
    
    sns.histplot(df, x="NDBI", kde=True, ax=axes[1, 0], color="lightcoral", edgecolor="black")
    axes[1, 0].set_title("Distribution of Built-up Index (NDBI)", fontsize=16)
    axes[1, 0].set_xlabel("Built-up Index (NDBI)")

    palette = {"Trees":"#228B22", "Shrubland":"#DAA520", "Grassland":"#ADFF2F", "Cropland":"#FFD700", "Built-up":"#A52A2A", "Bare_Vegetation":"#C0C0C0", "Water":"#4682B4", "Wetland":"#8FBC8F", "Mangroves":"#556B2F"}
    present = df["Land_Cover_Type"].dropna().unique().tolist()
    filtered_palette = {k: v for k, v in palette.items() if k in present}

    # FIX for FutureWarning: Add hue and disable the legend it creates.
    sns.boxplot(data=df, x="Land_Cover_Type", y="LST_Celsius", ax=axes[1, 1], palette=filtered_palette, hue="Land_Cover_Type", legend=False)
    
    # FIX for ValueError: Use plt.setp to apply rotation and alignment to the tick labels.
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha="right")
    
    axes[1, 1].set_title("Temperature by Land Cover Type", fontsize=16)
    axes[1, 1].set_xlabel("Land Cover Type")
    axes[1, 1].set_ylabel("Land Surface Temperature (°C)")
    
    footer_text = ("LST: Land Surface Temperature. The temperature of the ground's surface.\n"
                   "NDVI (Vegetation Index): Higher values indicate more green vegetation.\n"
                   "NDBI (Built-up Index): Higher values indicate more buildings and impervious surfaces.")
    plt.figtext(0.5, 0.01, footer_text, ha="center", fontsize=10, color='gray')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(outputs_dir, "feature_distributions.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved feature distributions plot: %s", path)

    # --- Correlation Plot ---
    numerical_cols = [c for c in ["LST_Celsius", "NDVI", "NDBI", "Elevation", "Slope", "Aspect"] if c in df.columns]
    plot_df = df[numerical_cols].rename(columns={
        "LST_Celsius": "Temperature", "NDVI": "Vegetation", "NDBI": "Built-up",
    })
    corr = plot_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="viridis", center=0, square=True, linewidths=0.5, fmt=".2f")
    plt.title("Correlation Matrix of Key Variables", fontsize=16)
    path = os.path.join(outputs_dir, "correlation_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved correlation matrix plot: %s", path)

    # --- Scatter Relationships ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.scatterplot(df, x="NDVI", y="LST_Celsius", hue="Land_Cover_Type", ax=axes[0], s=20, alpha=0.7)
    axes[0].set_title("Temperature vs. Vegetation", fontsize=16)
    axes[0].set_xlabel("Vegetation Index (NDVI)")
    axes[0].set_ylabel("Land Surface Temperature (°C)")

    sns.scatterplot(df, x="NDBI", y="LST_Celsius", hue="Land_Cover_Type", ax=axes[1], s=20, alpha=0.7)
    axes[1].set_title("Temperature vs. Built-up Areas", fontsize=16)
    axes[1].set_xlabel("Built-up Index (NDBI)")
    axes[1].set_ylabel("Land Surface Temperature (°C)")
    
    plt.tight_layout()
    path = os.path.join(outputs_dir, "scatter_relationships.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved scatter relationships plot: %s", path)
