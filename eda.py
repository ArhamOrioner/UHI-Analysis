import logging
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_eda_plots(df: pd.DataFrame, outputs_dir: str) -> None:
    sns.set_style("whitegrid")

    # Distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.histplot(df, x="LST_Celsius", kde=True, ax=axes[0, 0], color="skyblue", edgecolor="black")
    axes[0, 0].set_title("Distribution of LST")
    sns.histplot(df, x="NDVI", kde=True, ax=axes[0, 1], color="lightgreen", edgecolor="black")
    axes[0, 1].set_title("Distribution of NDVI")
    sns.histplot(df, x="NDBI", kde=True, ax=axes[1, 0], color="lightcoral", edgecolor="black")
    axes[1, 0].set_title("Distribution of NDBI")

    palette = {
        "Trees": "#228B22",
        "Shrubland": "#DAA520",
        "Grassland": "#ADFF2F",
        "Cropland": "#FFD700",
        "Built-up": "#A52A2A",
        "Bare_Vegetation": "#C0C0C0",
        "Snow_Ice": "#ADD8E6",
        "Water": "#4682B4",
        "Wetland": "#8FBC8F",
        "Mangroves": "#556B2F",
    }
    present = df["Land_Cover_Type"].dropna().unique().tolist()
    filtered_palette = {k: v for k, v in palette.items() if k in present}
    sns.boxplot(data=df, x="Land_Cover_Type", y="LST_Celsius", ax=axes[1, 1], palette=filtered_palette)
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].set_title("LST by Land Cover Type")
    plt.tight_layout()
    path = os.path.join(outputs_dir, "feature_distributions.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved %s", path)

    # Correlation
    numerical_cols = [c for c in ["LST_Celsius", "NDVI", "NDBI", "Elevation", "Slope", "Aspect"] if c in df.columns]
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="viridis", center=0, square=True, linewidths=0.5, fmt=".2f")
    plt.title("Correlation Matrix")
    path = os.path.join(outputs_dir, "correlation_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved %s", path)

    # Scatter relationships
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.scatterplot(df, x="NDVI", y="LST_Celsius", hue="Land_Cover_Type", ax=axes[0], s=20, alpha=0.7)
    axes[0].set_title("LST vs NDVI")
    sns.scatterplot(df, x="NDBI", y="LST_Celsius", hue="Land_Cover_Type", ax=axes[1], s=20, alpha=0.7)
    axes[1].set_title("LST vs NDBI")
    plt.tight_layout()
    path = os.path.join(outputs_dir, "scatter_relationships.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved %s", path)