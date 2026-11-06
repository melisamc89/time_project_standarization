#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Intra-Dataset Clustering, Refined Time/Position/Direction Assignment,
Animal-Level Cell Counts with Dots & Error Bars, Two-Way ANOVA with Cleaned Post-Hoc Tests,
LFP Feature Analysis (Excluding Ultra-Ultra-Slow & Ultra-Slow Bands),
Dedicated LFP Waveform, Envelope, and Phase Bar Plots with Scatter Dots & Error Bars,
One-Sample t-Tests against Zero for LFP Features, Dual Heatmaps (Original & Sorted),
Matrix t-SNE Projections, Text Reports, and Dual PNG/SVG Export.
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
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import scipy.cluster.hierarchy as sch

from scipy.stats import zscore, ttest_ind, ttest_1samp, chi2_contingency, chisquare
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
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

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "mir_comprehensive_lfp_filtered_pipeline")

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

# Excluded ultra_ultra_slow and ultra_slow as requested
LFP_BAND_NAMES = [
    "slow",
    "delta",
    "theta",
    "gamma",
    "ripple",
    "mua",
]

LFP_WAVEFORM_COLUMNS = [f"mi_lfp_{band}" for band in LFP_BAND_NAMES]
LFP_ENVELOPE_COLUMNS = [f"mi_lfp_{band}_envelope" for band in LFP_BAND_NAMES]
LFP_PHASE_COLUMNS = [f"mi_lfp_{band}_phase" for band in LFP_BAND_NAMES]

ALL_LFP_COLUMNS = LFP_WAVEFORM_COLUMNS + LFP_ENVELOPE_COLUMNS + LFP_PHASE_COLUMNS

N_CLUSTERS = 3
UNASSIGNED_LABEL = 10
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 50
N_PERMUTATIONS = 500

TSNE_PERPLEXITY = 30
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
    2: "#ec8ef8ff",  # Position (Magenta/Purple)
    1: "#66cef4ff",  # Direction (Blue)
    0: "#bce784ff",  # Time (Green)
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

DEPTH_COLOR_MAP = {
    "superficial": "#9900ff",
    "deep": "#cc9900",
    "whole population": "gray",
}


# =============================================================================
# Helpers & Custom Colormaps
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
    fig.savefig(svg_path, bbox_inches="tight")
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


def add_neuron_uid(df):
    out = df.copy()
    out["neuron_uid"] = (
            out["dataset_name"].astype(str) + "__" +
            out["mouse_name"].astype(str) + "__" +
            out["session_name"].astype(str) + "__" +
            out["depth_type"].astype(str) + "__row" +
            out["row_index"].astype(str) + "__neuron" +
            out["neuron_index"].astype(str)
    )
    return out


def build_mouse_index(df):
    mice_sorted = sorted(df["mouse_name"].dropna().astype(str).unique().tolist())
    mouse_to_idx = {m: i for i, m in enumerate(mice_sorted)}
    out = df.copy()
    out["mouse_index"] = out["mouse_name"].astype(str).map(mouse_to_idx)
    return out, mice_sorted


def get_cluster_custom_colormap(hex_color):
    if hex_color.endswith("ff") and len(hex_color) == 9:
        hex_color = hex_color[:7]
    return LinearSegmentedColormap.from_list("custom_cluster_cmap", ["#ffffff", hex_color])


def pval_to_star(pval):
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


# =============================================================================
# Hierarchical Sorting & Permutation Test Helpers
# =============================================================================

def get_hierarchically_sorted_matrix(corr_df):
    mat = corr_df.to_numpy()
    dist_matrix = np.clip(1.0 - mat, 0.0, 2.0)
    np.fill_diagonal(dist_matrix, 0.0)

    condensed_dist = squareform(dist_matrix, checks=False)
    linkage_matrix = sch.linkage(condensed_dist, method='average')
    sorted_idx = sch.leaves_list(linkage_matrix)

    features = corr_df.columns.tolist()
    sorted_features = [features[i] for i in sorted_idx]

    return corr_df.loc[sorted_features, sorted_features]


def permutation_test_correlation_matrix(df_data, feature_cols, n_perms=500):
    rng = np.random.default_rng(42)
    sub = df_data[feature_cols].astype(float).dropna()
    if len(sub) < 3:
        return np.eye(len(feature_cols)), np.ones((len(feature_cols), len(feature_cols)))

    obs_corr = sub.corr(method="pearson").to_numpy()
    obs_corr = np.nan_to_num(obs_corr, nan=0.0)
    n_features = len(feature_cols)

    perm_max = np.zeros(n_perms)
    data_arr = sub.to_numpy()

    for p in range(n_perms):
        perm_arr = data_arr.copy()
        for j in range(n_features):
            rng.shuffle(perm_arr[:, j])
        perm_corr = np.corrcoef(perm_arr, rowvar=False)
        perm_corr = np.nan_to_num(perm_corr, nan=0.0)
        perm_max[p] = np.max(np.abs(perm_corr))

    p_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            val = abs(obs_corr[i, j])
            p_matrix[i, j] = (np.sum(perm_max >= val) + 1) / (n_perms + 1)

    return obs_corr, p_matrix


# =============================================================================
# Refined Semantic Mapping & Clustering
# =============================================================================

def infer_intra_dataset_semantic_mapping(df_dataset, raw_labels):
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

        time_score = (time + trial) - (pos + dir_val + pos_dir)
        pos_score = pos - (time + trial)

        scores[cid] = {"time_score": time_score, "pos_score": pos_score, "pos": pos}

    time_cid = max(cluster_ids, key=lambda c: scores[c]["time_score"])
    remaining_after_time = [c for c in cluster_ids if c != time_cid]
    pos_cid = max(remaining_after_time, key=lambda c: scores[c]["pos"]) if remaining_after_time else time_cid
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
# t-SNE on Correlation Matrices Helper
# =============================================================================

def plot_correlation_matrix_tsne(corr_matrix, feature_names, sub_png_dir, sub_svg_dir, prefix, label_name, plot_color):
    if corr_matrix.shape[0] < 3:
        return

    tsne = TSNE(
        n_components=2,
        perplexity=min(10, max(2, corr_matrix.shape[0] // 3)),
        initialization="pca",
        metric="euclidean",
        random_state=42,
        verbose=False
    )
    embedding = np.asarray(tsne.fit(corr_matrix))

    if plot_color.endswith("ff") and len(plot_color) == 9:
        plot_color = plot_color[:7]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], color=plot_color, s=100, edgecolor='black', linewidth=1.2, alpha=0.9)

    for idx, name in enumerate(feature_names):
        short_name = name.replace("mi_lfp_", "").replace("mi_", "")
        ax.annotate(short_name, (embedding[idx, 0] + 0.5, embedding[idx, 1] + 0.5), fontsize=8, fontweight='bold')

    ax.set_title(f"{prefix}: Correlation Matrix t-SNE ({label_name})", fontsize=11, fontweight='bold',
                 color=plot_color if plot_color != "#bbcc0" else "black")
    ax.set_xlabel("t-SNE Dim 1", fontsize=10)
    ax.set_ylabel("t-SNE Dim 2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    save_figure_both(fig, sub_png_dir, sub_svg_dir, f"{prefix}_{sanitize_for_filename(label_name)}_matrix_tsne")


# =============================================================================
# Complete Analysis & Plotting Pipeline
# =============================================================================

def analyze_and_plot_subset(df_clustered, sub_png_dir, sub_svg_dir, sub_stats_dir, prefix):
    # 1. Standard Embedding Grid Plot (Behavior)
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

    available_lfp_cols = [c for c in ALL_LFP_COLUMNS if c in df_clustered.columns]

    # 2. Dedicated LFP Category Bar Plots (Waveforms, Envelopes, Phases) with Scatter Dots & Error Bars
    order = [c for c in ["0", "1", "2"] if c in df_clustered["cluster_id"].astype(int).astype(str).unique()]

    lfp_categories = {
        "Waveform": [c for c in available_lfp_cols if c in LFP_WAVEFORM_COLUMNS],
        "Envelope": [c for c in available_lfp_cols if c in LFP_ENVELOPE_COLUMNS],
        "Phase": [c for c in available_lfp_cols if c in LFP_PHASE_COLUMNS],
    }

    lfp_anova_report = f"--- LFP CATEGORY TWO-WAY ANOVAs & ONE-SAMPLE T-TESTS ({prefix}) ---\n\n"

    for cat_name, cat_cols in lfp_categories.items():
        if not cat_cols:
            continue

        melted_cat = df_clustered[df_clustered["cluster_id"] != UNASSIGNED_LABEL].melt(
            id_vars=["mouse_name", "cluster_id"], value_vars=cat_cols,
            var_name="lfp_feature", value_name="MIR_value"
        )
        melted_cat["cluster_str"] = melted_cat["cluster_id"].astype(int).astype(str)
        melted_cat["cluster_name"] = melted_cat["cluster_str"].map(CLUSTER_NAME_MAP_STR)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=melted_cat, x="lfp_feature", y="MIR_value", hue="cluster_name",
            palette=[CLUSTER_COLOR_MAP_STR[c] for c in order if c in melted_cat["cluster_str"].unique()],
            capsize=0.1, err_kws={'linewidth': 1.5}, ax=ax, alpha=0.85
        )
        sns.stripplot(
            data=melted_cat, x="lfp_feature", y="MIR_value", hue="cluster_name",
            palette=[CLUSTER_COLOR_MAP_STR[c] for c in order if c in melted_cat["cluster_str"].unique()],
            dodge=True, color="black", alpha=0.4, size=3, jitter=0.1, ax=ax, legend=False
        )
        ax.set_xticks(range(len(cat_cols)))
        ax.set_xticklabels([c.replace("mi_lfp_", "") for c in cat_cols], rotation=25, ha="right")
        ax.set_title(f"{prefix}: LFP {cat_name} Information by Band & Cluster", fontsize=11, fontweight='bold')
        ax.set_xlabel("LFP Feature")
        ax.set_ylabel("Mutual Information Rate (MIR)")
        plt.legend(title="Cluster")
        plt.tight_layout()
        save_figure_both(fig, sub_png_dir, sub_svg_dir, f"{prefix}_lfp_{cat_name.lower()}_barplot")

        # Two-Way ANOVA & Cleaned Post-Hoc Tukey HSD for this LFP Category
        lfp_anova_report += f"=== LFP {cat_name} Two-Way ANOVA ===\n"
        try:
            clean_cat = melted_cat.dropna(subset=['MIR_value']).copy()
            model_cat = ols('MIR_value ~ C(cluster_name) * C(lfp_feature)', data=clean_cat).fit()
            anova_table_cat = sm.stats.anova_lm(model_cat, typ=2)
            lfp_anova_report += anova_table_cat.to_string() + "\n\n"

            clean_cat["group"] = clean_cat["cluster_name"] + "_" + clean_cat["lfp_feature"]
            group_counts = clean_cat.groupby("group")["MIR_value"].count()
            valid_groups = group_counts[group_counts >= 2].index
            filtered_tukey_df = clean_cat[clean_cat["group"].isin(valid_groups)]

            tukey_cat = pairwise_tukeyhsd(
                endog=filtered_tukey_df["MIR_value"],
                groups=filtered_tukey_df["group"],
                alpha=0.05
            )
            lfp_anova_report += "Tukey HSD Post-Hoc Test:\n" + str(tukey_cat) + "\n\n"
        except Exception as e:
            lfp_anova_report += f"ANOVA/Tukey failed: {e}\n\n"

    # One-Sample t-Tests against Zero for all LFP features
    lfp_anova_report += "=== One-Sample t-Tests against Zero (LFP Features) ===\n"
    for lfp_col in available_lfp_cols:
        valid_vals = df_clustered[lfp_col].dropna()
        if len(valid_vals) > 1:
            t_stat, p_val = ttest_1samp(valid_vals, 0.0)
            lfp_anova_report += f"{lfp_col}: mean={valid_vals.mean():.5f}, t={t_stat:.4f}, p={p_val:.4e} ({pval_to_star(p_val)})\n"

    save_text_file(sub_stats_dir, f"{prefix}_lfp_filtered_stats", lfp_anova_report)

    corr_features = MIR_FEATURE_COLUMNS + available_lfp_cols

    # 3. Complete Population Correlation Heatmaps (Original & Hierarchically Sorted)
    obs_corr_pop, _ = permutation_test_correlation_matrix(df_clustered, corr_features, n_perms=N_PERMUTATIONS)
    corr_df_pop = pd.DataFrame(obs_corr_pop, index=corr_features, columns=corr_features)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df_pop, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
                cbar_kws={'label': 'Pearson r (Original)'})
    ax.set_title(f"{prefix}: Complete Population Correlation Matrix (Original Order)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_figure_both(fig, sub_png_dir, sub_svg_dir, f"{prefix}_complete_population_original_correlation")
    corr_df_pop.to_csv(os.path.join(sub_stats_dir, f"{prefix}_complete_population_original_matrix.csv"))

    sorted_corr_pop = get_hierarchically_sorted_matrix(corr_df_pop)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sorted_corr_pop, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
                cbar_kws={'label': 'Pearson r (Sorted)'})
    ax.set_title(f"{prefix}: Complete Population Correlation Matrix (Hierarchically Sorted)", fontsize=12,
                 fontweight='bold')
    plt.tight_layout()
    save_figure_both(fig, sub_png_dir, sub_svg_dir, f"{prefix}_complete_population_sorted_correlation")
    sorted_corr_pop.to_csv(os.path.join(sub_stats_dir, f"{prefix}_complete_population_sorted_matrix.csv"))

    plot_correlation_matrix_tsne(obs_corr_pop, corr_features, sub_png_dir, sub_svg_dir, prefix, "Complete_Population",
                                 "#2b2b2b")

    # 4. Cluster-by-Cluster Correlation Heatmaps (Original & Sorted with Custom Colormap)
    cluster_ids_present = sorted(df_clustered["cluster_id"].unique().tolist())
    corr_stats_text = f"--- CLUSTER-BY-CLUSTER DUAL HEATMAP CORRELATIONS ({prefix}) ---\n\n"

    for cid in cluster_ids_present:
        if cid == UNASSIGNED_LABEL:
            continue
        c_str = str(int(cid))
        c_name = CLUSTER_NAME_MAP_STR.get(c_str, f"Cluster_{cid}")
        c_hex = CLUSTER_COLOR_MAP_STR.get(c_str, "#bbbcc0ff")
        custom_cmap = get_cluster_custom_colormap(c_hex)

        sub_c = df_clustered[df_clustered["cluster_id"] == cid].copy()
        if len(sub_c) < 5:
            continue

        obs_corr_c, _ = permutation_test_correlation_matrix(sub_c, corr_features, n_perms=N_PERMUTATIONS)
        corr_df_c = pd.DataFrame(obs_corr_c, index=corr_features, columns=corr_features)

        # Original
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_df_c, annot=False, cmap=custom_cmap, vmin=0, vmax=1, ax=ax,
                    cbar_kws={'label': f'Pearson r ({c_name})'})
        ax.set_title(f"{prefix} | Cluster: {c_name} (Original Order, n={len(sub_c)})",
                     color=c_hex if c_hex != "#bbbcc0ff" else "black", fontsize=12, fontweight='bold')
        plt.tight_layout()
        orig_base = f"{prefix}_cluster_{c_name.lower()}_original_correlation"
        save_figure_both(fig, sub_png_dir, sub_svg_dir, orig_base)
        corr_df_c.to_csv(os.path.join(sub_stats_dir, f"{orig_base}_matrix.csv"))

        # Sorted
        sorted_corr_c = get_hierarchically_sorted_matrix(corr_df_c)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(sorted_corr_c, annot=False, cmap=custom_cmap, vmin=0, vmax=1, ax=ax,
                    cbar_kws={'label': f'Pearson r ({c_name})'})
        ax.set_title(f"{prefix} | Cluster: {c_name} (Hierarchically Sorted, n={len(sub_c)})",
                     color=c_hex if c_hex != "#bbbcc0ff" else "black", fontsize=12, fontweight='bold')
        plt.tight_layout()
        sorted_base = f"{prefix}_cluster_{c_name.lower()}_sorted_correlation"
        save_figure_both(fig, sub_png_dir, sub_svg_dir, sorted_base)
        sorted_corr_c.to_csv(os.path.join(sub_stats_dir, f"{sorted_base}_matrix.csv"))

        plot_correlation_matrix_tsne(obs_corr_c, corr_features, sub_png_dir, sub_svg_dir, prefix, c_name, c_hex)

        corr_stats_text += f"Cluster: {c_name} (Color: {c_hex}, n={len(sub_c)})\n"
        corr_stats_text += f"Saved original and sorted matrices for {c_name}.\n\n"

    save_text_file(sub_stats_dir, f"{prefix}_cluster_correlation_stats", corr_stats_text)

    # 5. Animal-Level Cell Counts Barplot with Dots & Error Bars
    animal_counts = df_clustered.groupby(["mouse_name", "cluster_id"]).size().reset_index(name="cell_count")
    animal_counts["cluster_str"] = animal_counts["cluster_id"].astype(int).astype(str)
    animal_counts["cluster_name"] = animal_counts["cluster_str"].map(CLUSTER_NAME_MAP_STR)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(
        data=animal_counts, x="cluster_name", y="cell_count", order=[CLUSTER_NAME_MAP_STR[c] for c in order],
        palette=[CLUSTER_COLOR_MAP_STR[c] for c in order], capsize=0.1, err_kws={'linewidth': 1.5}, ax=ax, alpha=0.85
    )
    sns.stripplot(
        data=animal_counts, x="cluster_name", y="cell_count", order=[CLUSTER_NAME_MAP_STR[c] for c in order],
        color="black", alpha=0.7, size=6, jitter=0.15, ax=ax
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CLUSTER_NAME_MAP_STR[c] for c in order])
    ax.set_title(f"{prefix}: Animal-Level Cell Counts per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Neurons per Animal")
    plt.tight_layout()
    save_figure_both(fig, sub_png_dir, sub_svg_dir, f"{prefix}_animal_level_cell_counts")


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

    lt_rot_df = df_all[df_all["dataset_name"].astype(str).str.lower().str.contains("lt_rot")].copy()
    if not lt_rot_df.empty:
        lt_mask = lt_rot_df["session_name"].astype(str).str.lower().str.contains("lt") & ~lt_rot_df[
            "session_name"].astype(str).str.lower().str.contains("rot")
        rot_mask = lt_rot_df["session_name"].astype(str).str.lower().str.contains("rot") & ~lt_rot_df[
            "session_name"].astype(str).str.lower().str.contains("lt")
        subsets_to_process["lt_rot_LT"] = lt_rot_df[lt_mask].copy()
        subsets_to_process["lt_rot_ROT"] = lt_rot_df[rot_mask].copy()

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

        analyze_and_plot_subset(df_clustered, sub_png_dir, sub_svg_dir, sub_stats_dir,
                                sanitize_for_filename(subset_name))

        print(f"[OK] Finished subset: {subset_name}")

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()