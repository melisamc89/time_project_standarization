#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Intra-Dataset Clustering, Refined Time/Position/Direction Assignment,
Animal-Level Cell Counts, Two-Way ANOVA on Information Values with Post-Hoc Tests,
Error Bars & Dots Plotting, Text Reports, and Dual PNG/SVG Export Pipeline.
"""

import os
import re
import json
import pickle
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore, ttest_ind, chi2_contingency, chisquare
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from openTSNE import TSNE

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# =============================================================================
# Paths & Settings
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"

MI_CSV = os.path.join(PROJECT_ROOT, "analysis", "mutual_information_behavior_and_lfp",
                      "mutual_information_per_neuron.csv")
MASTER_INDEX_CSV = os.path.join(PROJECT_ROOT, "data", "metadata", "master_dataset_index.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "mir_comprehensive_twoway_anova")

MIR_FEATURE_COLUMNS = [
    "mi_position",
    "mi_position_x_direction",
    "mi_direction",
    "mi_speed",
    "mi_time",
    "mi_inner_time",
    "mi_trial_id",
]

MIR_FEATURE_TITLES = [
    "PositionMIR",
    "DirPositionMIR",
    "MovDirMIR",
    "SpeedMIR",
    "TimeMIR",
    "InnerTimeMIR",
    "TrialIDMIR",
]

N_CLUSTERS = 3
UNASSIGNED_LABEL = 10
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 50

TSNE_PERPLEXITY = 50
TSNE_RANDOM_STATE = 42
TSNE_N_JOBS = 8
TSNE_INITIALIZATION = "pca"
TSNE_METRIC = "euclidean"

SCATTER_SIZE = 40
FEATURE_CMAP = "coolwarm"
MOUSE_CMAP = "rainbow"

MIR_VMIN_Z = -1.0
MIR_VMAX_Z = 3.5

CLUSTER_COLOR_MAP = {
    2: "#ec8ef8ff",  # Position
    1: "#66cef4ff",  # Direction
    0: "#bce784ff",  # Time
    10: "#bbbcc0ff",  # Unassigned
}

CLUSTER_COLOR_MAP_STR = {
    "2": "#ec8ef8ff",
    "1": "#66cef4ff",
    "0": "#bce784ff",
    "10": "#bbbcc0ff",
}

CLUSTER_NAME_MAP_STR = {
    "0": "Time",
    "1": "Direction",
    "2": "Position",
    "10": "Unassigned",
}


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def save_figure_both(fig, png_dir, svg_dir, out_base):
    ensure_dir(png_dir)
    ensure_dir(svg_dir)
    png_path = os.path.join(png_dir, out_base + ".png")
    svg_path = os.path.join(svg_dir, out_base + ".svg")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(svg_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")


def save_text_file(stats_dir, out_base, text):
    ensure_dir(stats_dir)
    out_path = os.path.join(stats_dir, out_base + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] Saved: {out_path}")


def rowwise_zscore(X):
    Xz = zscore(X, axis=1, nan_policy="omit")
    Xz = np.where(np.isfinite(Xz), Xz, 0.0)
    return Xz


def build_mouse_index(df):
    mice_sorted = sorted(df["mouse_name"].dropna().astype(str).unique().tolist())
    mouse_to_idx = {m: i for i, m in enumerate(mice_sorted)}
    out = df.copy()
    out["mouse_index"] = out["mouse_name"].astype(str).map(mouse_to_idx)
    return out, mice_sorted


# =============================================================================
# Refined Semantic Mapping & Clustering
# =============================================================================

def infer_intra_dataset_semantic_mapping(df_dataset, raw_labels):
    """
    Time cluster: higher info in trial_id and time simultaneously,
                  lower info in position, direction, and position_x_direction.
    Position cluster: higher info in position.
    Direction cluster: remaining.
    """
    tmp = df_dataset.copy()
    tmp["raw_label"] = raw_labels

    cluster_ids = sorted(tmp["raw_label"].unique().tolist())
    if len(cluster_ids) < N_CLUSTERS:
        return {cid: cid for cid in cluster_ids}

    scores = {}
    for cid in cluster_ids:
        sub = tmp[tmp["raw_label"] == cid]
        pos = sub["mi_position"].mean()
        time = sub["mi_time"].mean()
        trial = sub["mi_trial_id"].mean()
        dir_val = sub["mi_direction"].mean()
        pos_dir = sub["mi_position_x_direction"].mean() if "mi_position_x_direction" in sub.columns else 0.0

        # Time score definition
        time_score = (time + trial) - (pos + dir_val + pos_dir)
        # Position score definition
        pos_score = pos - (time + trial)

        scores[cid] = {"time_score": time_score, "pos_score": pos_score, "pos": pos}

    # 1. Time cluster has highest time_score
    time_cid = max(cluster_ids, key=lambda c: scores[c]["time_score"])

    # 2. Position cluster has highest pos among remaining
    remaining_after_time = [c for c in cluster_ids if c != time_cid]
    pos_cid = max(remaining_after_time, key=lambda c: scores[c]["pos"]) if remaining_after_time else time_cid

    # 3. Direction gets the final remaining cluster
    remaining_after_pos = [c for c in remaining_after_time if c != pos_cid]
    dir_cid = remaining_after_pos[0] if remaining_after_pos else (
        remaining_after_time[0] if remaining_after_time else time_cid)

    return {
        pos_cid: 2,  # Position
        time_cid: 0,  # Time
        dir_cid: 1,  # Direction
    }


def cluster_single_dataset(df_dataset):
    X_raw = df_dataset[MIR_FEATURE_COLUMNS].astype(float).to_numpy()
    valid_mask = np.all(np.isfinite(X_raw), axis=1)
    df_sub = df_dataset.loc[valid_mask].copy().reset_index(drop=True)
    X_raw = X_raw[valid_mask]

    if len(df_sub) < N_CLUSTERS:
        raise ValueError("Too few valid neurons for clustering.")

    X_z = rowwise_zscore(X_raw)
    z_cols = [f"z_{c}" for c in MIR_FEATURE_COLUMNS]
    for j, col in enumerate(z_cols):
        df_sub[col] = X_z[:, j]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_z)

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
    )
    raw_labels = kmeans.fit_predict(X_scaled)

    remap_dict = infer_intra_dataset_semantic_mapping(df_sub, raw_labels)
    semantic_labels = np.array([remap_dict.get(int(l), UNASSIGNED_LABEL) for l in raw_labels], dtype=int)

    tsne = TSNE(
        n_components=2,
        perplexity=min(TSNE_PERPLEXITY, max(5, len(df_sub) // 3)),
        initialization=TSNE_INITIALIZATION,
        metric=TSNE_METRIC,
        n_jobs=TSNE_N_JOBS,
        random_state=TSNE_RANDOM_STATE,
        verbose=False,
    )
    embedding = np.asarray(tsne.fit(X_scaled))

    df_sub["cluster_initial"] = raw_labels
    df_sub["cluster_id"] = semantic_labels
    df_sub["tsne_1"] = embedding[:, 0]
    df_sub["tsne_2"] = embedding[:, 1]

    return df_sub


# =============================================================================
# Statistical Analyses & Plotting with Error Bars, Dots, and Two-Way ANOVA
# =============================================================================

def analyze_and_plot_subset(df_clustered, sub_png_dir, sub_svg_dir, sub_stats_dir, prefix):
    # 1. Embedding Grid Plot
    feature_cols = [f"z_{c}" for c in MIR_FEATURE_COLUMNS]
    df_plot, _ = build_mouse_index(df_clustered)

    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    axs = axs.ravel()
    sns.scatterplot(data=df_plot, x="tsne_1", y="tsne_2", hue="cluster_id", palette=CLUSTER_COLOR_MAP, ax=axs[0],
                    legend=False, s=SCATTER_SIZE)
    axs[0].set_title("tSNE by Cluster ID")
    sns.scatterplot(data=df_plot, x="tsne_1", y="tsne_2", hue="mouse_index", palette=MOUSE_CMAP, ax=axs[1],
                    legend=False, s=SCATTER_SIZE)
    axs[1].set_title("tSNE by Mouse ID")

    for i in range(7):
        sc = axs[i + 2].scatter(df_plot["tsne_1"], df_plot["tsne_2"], c=df_plot[feature_cols[i]], cmap=FEATURE_CMAP,
                                s=SCATTER_SIZE, vmin=MIR_VMIN_Z, vmax=MIR_VMAX_Z)
        axs[i + 2].set_title(f"tSNE by {MIR_FEATURE_TITLES[i]}")
        plt.colorbar(sc, ax=axs[i + 2])
    for ax in axs:
        ax.grid(False)
    plt.tight_layout()
    save_figure_both(fig, sub_png_dir, sub_svg_dir, f"{prefix}_clusters_grid")

    # 2. Animal-Level Cell Counts Statistics & Barplot with Dots & Error Bars
    animal_counts = df_clustered.groupby(["mouse_name", "cluster_id"]).size().reset_index(name="cell_count")
    animal_counts["cluster_str"] = animal_counts["cluster_id"].astype(int).astype(str)
    animal_counts["cluster_name"] = animal_counts["cluster_str"].map(CLUSTER_NAME_MAP_STR)

    order = [c for c in ["0", "1", "2"] if c in animal_counts["cluster_str"].unique()]
    cluster_names_order = [CLUSTER_NAME_MAP_STR[c] for c in order]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(
        data=animal_counts, x="cluster_name", y="cell_count", order=cluster_names_order,
        palette=[CLUSTER_COLOR_MAP_STR[c] for c in order], capsize=0.1, errwidth=1.5, ax=ax, alpha=0.85
    )
    sns.stripplot(
        data=animal_counts, x="cluster_name", y="cell_count", order=cluster_names_order,
        color="black", alpha=0.7, size=6, jitter=0.15, ax=ax
    )
    ax.set_title(f"{prefix}: Animal-Level Cell Counts per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Neurons per Animal")
    plt.tight_layout()
    save_figure_both(fig, sub_png_dir, sub_svg_dir, f"{prefix}_animal_level_cell_counts")

    # One-Way ANOVA & Tukey HSD for Cell Counts
    anova_text = f"--- ONE-WAY ANOVA & TUKEY HSD: CELL COUNTS ({prefix}) ---\n\n"
    try:
        model = ols('cell_count ~ C(cluster_name)', data=animal_counts).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_text += anova_table.to_string() + "\n\n"

        tukey = pairwise_tukeyhsd(endog=animal_counts["cell_count"], groups=animal_counts["cluster_name"], alpha=0.05)
        anova_text += str(tukey) + "\n"
    except Exception as e:
        anova_text += f"ANOVA/Tukey failed: {e}\n"
    save_text_file(sub_stats_dir, f"{prefix}_cell_counts_stats", anova_text)

    # 3. Information Value Statistics: Two-Way ANOVA (Cluster x Feature) & Barplot
    melted_info = df_clustered[df_clustered["cluster_id"] != UNASSIGNED_LABEL].melt(
        id_vars=["mouse_name", "cluster_id"], value_vars=MIR_FEATURE_COLUMNS,
        var_name="feature", value_name="MIR_value"
    )
    melted_info["cluster_str"] = melted_info["cluster_id"].astype(int).astype(str)
    melted_info["cluster_name"] = melted_info["cluster_str"].map(CLUSTER_NAME_MAP_STR)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(
        data=melted_info, x="feature", y="MIR_value", hue="cluster_name",
        palette=[CLUSTER_COLOR_MAP_STR[c] for c in order], capsize=0.1, errwidth=1.5, ax=ax, alpha=0.85
    )
    ax.set_xticklabels(MIR_FEATURE_TITLES, rotation=25)
    ax.set_title(f"{prefix}: Mean Information Values by Feature & Cluster")
    ax.set_xlabel("Behavioral Feature")
    ax.set_ylabel("Mutual Information Rate (MIR)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    save_figure_both(fig, sub_png_dir, sub_svg_dir, f"{prefix}_information_values_barplot")

    # Two-Way ANOVA for Information Values: MIR_value ~ C(cluster_name) * C(feature)
    info_stats_text = f"--- TWO-WAY ANOVA & TUKEY HSD: INFORMATION VALUES ({prefix}) ---\n\n"
    try:
        model_info = ols('MIR_value ~ C(cluster_name) * C(feature)', data=melted_info).fit()
        anova_table_info = sm.stats.anova_lm(model_info, typ=2)
        info_stats_text += "Two-Way ANOVA Table:\n" + anova_table_info.to_string() + "\n\n"

        # Tukey HSD post-hoc test across interaction groups (Cluster x Feature)
        melted_info["group"] = melted_info["cluster_name"] + "_" + melted_info["feature"]
        tukey_info = pairwise_tukeyhsd(endog=melted_info["MIR_value"], groups=melted_info["group"], alpha=0.05)
        info_stats_text += "Tukey HSD Post-Hoc Test (Cluster x Feature groups):\n" + str(tukey_info) + "\n"
    except Exception as e:
        info_stats_text += f"Two-Way ANOVA/Tukey failed: {e}\n"
    save_text_file(sub_stats_dir, f"{prefix}_information_values_twoway_anova_stats", info_stats_text)


# =============================================================================
# Main Execution Loop
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)

    if not os.path.isfile(MI_CSV):
        raise FileNotFoundError(f"MI CSV not found: {MI_CSV}")
    if not os.path.isfile(MASTER_INDEX_CSV):
        raise FileNotFoundError(f"Master index CSV not found: {MASTER_INDEX_CSV}")

    df_mi = pd.read_csv(MI_CSV)
    df_master = pd.read_csv(MASTER_INDEX_CSV)

    join_cols = ["dataset_name", "mouse_name", "session_name", "depth_type"]
    add_cols = [c for c in ["modality", "paradigm", "model", "int_i", "int_f"] if c in df_master.columns]
    df_all = df_mi.merge(df_master[join_cols + add_cols].drop_duplicates(), on=join_cols, how="left")
    df_all = add_neuron_uid(df_all)

    base_datasets = [d for d in df_all["dataset_name"].dropna().astype(str).unique() if
                     "lt_rot" not in d.lower() and "intervention" not in d.lower()]

    subsets_to_process = {}
    for ds in base_datasets:
        subsets_to_process[ds] = df_all[df_all["dataset_name"].astype(str).str.lower() == ds.lower()].copy()

    # Split lt_rot into lt and rot explicitly[cite: 2]
    lt_rot_df = df_all[df_all["dataset_name"].astype(str).str.lower().str.contains("lt_rot")].copy()
    if not lt_rot_df.empty:
        lt_mask = lt_rot_df["session_name"].astype(str).str.lower().str.contains("lt") & ~lt_rot_df[
            "session_name"].astype(str).str.lower().str.contains("rot")
        rot_mask = lt_rot_df["session_name"].astype(str).str.lower().str.contains("rot") & ~lt_rot_df[
            "session_name"].astype(str).str.lower().str.contains("lt")
        subsets_to_process["lt_rot_LT"] = lt_rot_df[lt_mask].copy()
        subsets_to_process["lt_rot_ROT"] = lt_rot_df[rot_mask].copy()

    # Handle intervention dataset (excluding sessions containing 'combined')
    intervention_df = df_all[df_all["dataset_name"].astype(str).str.lower().str.contains("intervention")].copy()
    if not intervention_df.empty:
        clean_intervention = intervention_df[
            ~intervention_df["session_name"].astype(str).str.lower().str.contains("combined")].copy()
        clean_intervention["int_i_clean"] = clean_intervention["int_i"].fillna(0)
        clean_intervention["int_f_clean"] = clean_intervention["int_f"].fillna(0)
        subsets_to_process["intervention_clean"] = clean_intervention

    for subset_name, df_subset in subsets_to_process.items():
        if len(df_subset) < N_CLUSTERS:
            print(f"[WARNING] Subset {subset_name} has too few neurons ({len(df_subset)}). Skipping.")
            continue

        print(f"\n==========================================")
        print(f"Processing Subset: {subset_name} (n={len(df_subset)})")
        print(f"==========================================")

        sub_out_dir = os.path.join(OUTPUT_DIR, sanitize_for_filename(subset_name))
        sub_data_dir = os.path.join(sub_out_dir, "data")
        sub_png_dir = os.path.join(sub_out_dir, "png")
        sub_svg_dir = os.path.join(sub_out_dir, "svg")
        sub_stats_dir = os.path.join(sub_out_dir, "stats")
        ensure_dir(sub_data_dir)
        ensure_dir(sub_png_dir)
        ensure_dir(sub_svg_dir)
        ensure_dir(sub_stats_dir)

        try:
            df_clustered = cluster_single_dataset(df_subset)
        except Exception as e:
            print(f"[WARNING] Clustering failed for {subset_name}: {e}")
            continue

        out_csv = os.path.join(sub_data_dir, f"{sanitize_for_filename(subset_name)}_labeled_neurons.csv")
        df_clustered.to_csv(out_csv, index=False)
        print(f"[OK] Saved: {out_csv}")

        # Run analysis, plots, and stats
        analyze_and_plot_subset(df_clustered, sub_png_dir, sub_svg_dir, sub_stats_dir,
                                sanitize_for_filename(subset_name))

        print(f"[OK] Finished subset: {subset_name}")

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()