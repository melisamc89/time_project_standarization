#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot SI position vs SI time from the full Structure Index CSV,
and count number of rows per session.

Outputs
-------
- Scatter plots:
    * colored by dataset
    * colored by depth
    * colored by recording type
- CSV with number of rows per session
- Bar plot of rows per session
- TXT summary
- PNG and SVG figures
"""

import os
import timeit
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

SI_SESSION_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/structure_index_umap/structure_index_umap_per_session.csv"
OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/structure_index_pos_vs_time"

PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
TXT_DIR = os.path.join(OUTPUT_DIR, "txt")


# =============================================================================
# Parameters
# =============================================================================

X_COL = "si_position"
Y_COL = "si_time"

POINT_SIZE = 60
POINT_ALPHA = 0.85
ADD_MOUSE_LABELS = False   # set True if you want mouse names on scatter points
MAX_SESSION_LABELS_IN_BARPLOT = 100  # avoid unreadable x-axis if too many sessions


# =============================================================================
# Colors
# =============================================================================

DEPTH_COLOR_MAP = {
    "superficial": "#9900ff",
    "deep": "#cc9900",
    "whole population": "gray",
    "whole_population": "gray",
    "whole": "gray",
    "all": "gray",
    "unknown": "gray",
}

RECORDING_COLOR_MAP = {
    "calcium": "#1f77b4",
    "ephys": "#d62728",
    "unknown": "gray",
}

DEFAULT_CATEGORY_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def canonicalize_depth(x):
    if pd.isna(x):
        return "unknown"
    x = str(x).strip().lower()
    x = x.replace("-", " ").replace("__", "_")

    if x in ["sup", "superficial", "superficial cells"]:
        return "superficial"
    if x in ["deep", "deep cells"]:
        return "deep"
    if x in ["whole", "whole population", "whole_population", "all"]:
        return "whole population"
    return x


def canonicalize_recording_type(x):
    if pd.isna(x):
        return "unknown"
    x = str(x).strip().lower()
    if "calcium" in x:
        return "calcium"
    if "ephys" in x:
        return "ephys"
    return "unknown"


def get_dataset_color_map(categories):
    categories = list(categories)
    cmap = {}
    for i, cat in enumerate(categories):
        cmap[cat] = DEFAULT_CATEGORY_COLORS[i % len(DEFAULT_CATEGORY_COLORS)]
    return cmap


def save_fig(fig, png_path, svg_path):
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(df, color_col, color_map, title, basename, add_mouse_labels=False):
    fig, ax = plt.subplots(figsize=(7, 6))

    categories = [c for c in pd.unique(df[color_col]) if pd.notna(c)]

    for cat in categories:
        sub = df[df[color_col] == cat]
        ax.scatter(
            sub[X_COL],
            sub[Y_COL],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=color_map.get(cat, "gray"),
            edgecolors="black",
            linewidths=0.35,
            label=str(cat),
        )

        if add_mouse_labels and "mouse_name" in sub.columns:
            for _, row in sub.iterrows():
                ax.text(
                    row[X_COL] + 0.003,
                    row[Y_COL] + 0.003,
                    str(row["mouse_name"]),
                    fontsize=7,
                    alpha=0.85,
                )

    ax.set_xlabel("SI position")
    ax.set_ylabel("SI time")
    ax.set_title(title)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    save_fig(
        fig,
        os.path.join(PNG_DIR, basename + ".png"),
        os.path.join(SVG_DIR, basename + ".svg"),
    )


def plot_rows_per_session_barplot(session_counts):
    fig, ax = plt.subplots(figsize=(max(10, 0.22 * len(session_counts)), 5))

    x = np.arange(len(session_counts))
    y = session_counts["n_rows"].values

    ax.bar(x, y)
    ax.set_ylabel("Number of rows")
    ax.set_title("Number of rows per session")

    if len(session_counts) <= MAX_SESSION_LABELS_IN_BARPLOT:
        ax.set_xticks(x)
        ax.set_xticklabels(session_counts["session_name"].astype(str).values, rotation=90)
    else:
        ax.set_xticks([])

    fig.tight_layout()

    save_fig(
        fig,
        os.path.join(PNG_DIR, "rows_per_session_barplot.png"),
        os.path.join(SVG_DIR, "rows_per_session_barplot.svg"),
    )


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)
    ensure_dir(CSV_DIR)
    ensure_dir(TXT_DIR)

    if not os.path.isfile(SI_SESSION_CSV):
        raise FileNotFoundError(f"Missing SI CSV: {SI_SESSION_CSV}")

    t0 = timeit.default_timer()

    print("==============================================================")
    print("SI position vs SI time")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")

    df = pd.read_csv(SI_SESSION_CSV)

    needed_cols = [X_COL, Y_COL, "session_name"]
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # clean metadata
    if "dataset_name" not in df.columns:
        df["dataset_name"] = "unknown_dataset"

    if "depth_type" in df.columns:
        df["depth_group"] = df["depth_type"].apply(canonicalize_depth)
    else:
        df["depth_group"] = "unknown"

    if "recording_type" in df.columns:
        df["recording_group"] = df["recording_type"].apply(canonicalize_recording_type)
    else:
        df["recording_group"] = "unknown"

    # keep rows with valid SI position and SI time
    valid_mask = df[X_COL].notna() & df[Y_COL].notna()
    df_plot = df.loc[valid_mask].copy()

    print(f"Loaded rows: {len(df)}")
    print(f"Rows with valid {X_COL} and {Y_COL}: {len(df_plot)}")
    print("Available datasets:", sorted([x for x in pd.unique(df_plot["dataset_name"]) if pd.notna(x)]))
    print("Available depth groups:", sorted([x for x in pd.unique(df_plot["depth_group"]) if pd.notna(x)]))
    print("Available recording groups:", sorted([x for x in pd.unique(df_plot["recording_group"]) if pd.notna(x)]))

    # -------------------------------------------------------------------------
    # scatter by dataset
    # -------------------------------------------------------------------------
    dataset_categories = sorted([x for x in pd.unique(df_plot["dataset_name"]) if pd.notna(x)])
    dataset_color_map = get_dataset_color_map(dataset_categories)

    plot_scatter(
        df=df_plot,
        color_col="dataset_name",
        color_map=dataset_color_map,
        title="SI position vs SI time | colored by dataset",
        basename="si_position_vs_time_by_dataset",
        add_mouse_labels=ADD_MOUSE_LABELS,
    )

    # -------------------------------------------------------------------------
    # scatter by depth
    # -------------------------------------------------------------------------
    depth_color_map = {
        "deep": DEPTH_COLOR_MAP["deep"],
        "superficial": DEPTH_COLOR_MAP["superficial"],
        "whole population": DEPTH_COLOR_MAP["whole population"],
        "unknown": DEPTH_COLOR_MAP["unknown"],
    }

    plot_scatter(
        df=df_plot,
        color_col="depth_group",
        color_map=depth_color_map,
        title="SI position vs SI time | colored by depth",
        basename="si_position_vs_time_by_depth",
        add_mouse_labels=ADD_MOUSE_LABELS,
    )

    # -------------------------------------------------------------------------
    # scatter by recording type
    # -------------------------------------------------------------------------
    plot_scatter(
        df=df_plot,
        color_col="recording_group",
        color_map=RECORDING_COLOR_MAP,
        title="SI position vs SI time | colored by recording type",
        basename="si_position_vs_time_by_recording_type",
        add_mouse_labels=ADD_MOUSE_LABELS,
    )

    # -------------------------------------------------------------------------
    # count rows per session
    # -------------------------------------------------------------------------
    session_counts = (
        df.groupby("session_name", as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
        .sort_values(["n_rows", "session_name"], ascending=[False, True])
        .reset_index(drop=True)
    )

    session_counts_csv = os.path.join(CSV_DIR, "rows_per_session.csv")
    session_counts.to_csv(session_counts_csv, index=False)

    plot_rows_per_session_barplot(session_counts)

    # -------------------------------------------------------------------------
    # summary txt
    # -------------------------------------------------------------------------
    txt_path = os.path.join(TXT_DIR, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SI position vs SI time summary\n")
        f.write("====================================================\n")
        f.write(f"Input CSV: {SI_SESSION_CSV}\n")
        f.write(f"Total rows in CSV: {len(df)}\n")
        f.write(f"Rows with valid {X_COL} and {Y_COL}: {len(df_plot)}\n\n")

        f.write("Dataset counts:\n")
        f.write(df_plot["dataset_name"].value_counts(dropna=False).to_string())
        f.write("\n\n")

        f.write("Depth counts:\n")
        f.write(df_plot["depth_group"].value_counts(dropna=False).to_string())
        f.write("\n\n")

        f.write("Recording type counts:\n")
        f.write(df_plot["recording_group"].value_counts(dropna=False).to_string())
        f.write("\n\n")

        f.write("Rows per session:\n")
        f.write(session_counts.to_string(index=False))
        f.write("\n")

    elapsed = timeit.default_timer() - t0
    print(f"Saved rows-per-session CSV: {session_counts_csv}")
    print(f"Finished in {elapsed:.2f} s")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()