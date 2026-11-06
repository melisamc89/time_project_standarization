#!/usr/init/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Multi-Clustering Pipeline:
- Position Rule: Median (mi_position + mi_position_x_direction) -> ID 2 (Pink)
- Time Rule: Maximize (Time Score - Position/Direction Score) -> ID 0 (Green)
- Direction Rule: Remaining -> ID 1 (Cyan)
- Includes Standardized 3x3 LFP Suites (Linear & Log CWT), Cluster Regressions,
  and Cluster-Colored Log-Scale Bivariate Contour Plots.
- Generates dual t-SNE plots (with and without colorbars) per configuration.
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

from scipy.stats import zscore, ttest_ind, chi2_contingency, chisquare, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from openTSNE import TSNE

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.colors import LogNorm, Normalize

# =============================================================================
# Paths & Configuration
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"

MI_CSV = os.path.join(PROJECT_ROOT, "analysis", "mutual_information_behavior_and_lfp",
                      "mutual_information_per_neuron.csv")
MASTER_INDEX_CSV = os.path.join(PROJECT_ROOT, "data", "metadata", "master_dataset_index.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "mir_comprehensive_twoway_anova_lfp_per_behavior_detailed")

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

# Cluster Color and Name Maps (Position = Pink/2, Direction = Cyan/1, Time = Green/0)
CLUSTER_COLOR_MAP = {
    2: "#ec8ef8ff",  # Position (Pink)
    1: "#66cef4ff",  # Direction (Cyan)
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

BAND_ORDER = ["slow", "delta", "theta", "gamma", "ripple", "mua"]
BAND_FREQS = {
    "slow": 0.5,
    "delta": 2.0,
    "theta": 10.0,
    "gamma": 65.0,
    "ripple": 175.0,
    "mua": 650.0
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


def discover_lfp_columns(df):
    cols = df.columns.tolist()
    return sorted([c for c in cols if c.startswith("mi_lfp_") or c.startswith("mi_cwt_")])


def parse_lfp_feature_metadata(feat_name):
    clean = feat_name.replace("mi_", "")
    modality = "waveform"
    band = clean.replace("lfp_", "").replace("cwt_", "")

    if "_envelope" in clean:
        modality = "envelope"
        band = clean.replace("lfp_", "").replace("_envelope", "")
    elif "_phase" in clean:
        modality = "phase"
        band = clean.replace("lfp_", "").replace("_phase", "")
    elif clean.startswith("cwt_"):
        modality = "cwt"
        band = clean.replace("cwt_", "")
    elif clean.startswith("lfp_"):
        modality = "waveform"
        band = clean.replace("lfp_", "")
    return band, modality


# =============================================================================
# Clustering & Refined Semantic Mapping Rule (Time Cluster Minimizes Pos/Dir Info)
# =============================================================================

def infer_intra_dataset_semantic_mapping(df_dataset, raw_labels):
    """
    Semantic mapping rules:
    1) Position cluster (ID 2, Pink): Max median (mi_position + mi_position_x_direction).
    2) Time cluster (ID 0, Green): Maximize (Time Score - Position/Direction Score) among remaining.
       Ensures time cluster has high time info and minimum information about position, direction, or directed positions.
    3) Direction cluster (ID 1, Cyan): The remaining cluster.
    """
    tmp = df_dataset.copy()
    tmp["raw_label"] = raw_labels

    cluster_ids = sorted(tmp["raw_label"].unique().tolist())
    if len(cluster_ids) < N_CLUSTERS:
        return {cid: cid for cid in cluster_ids}

    cluster_scores = {}
    for cid in cluster_ids:
        sub = tmp[tmp["raw_label"] == cid]

        # Position and Direction information components
        pos_score = np.median(sub["mi_position"] + sub["mi_position_x_direction"])
        dir_score = np.median(sub["mi_direction"]) if "mi_direction" in sub.columns else 0.0
        pos_dir_total = pos_score + dir_score

        # Time information components
        time_score = np.median(sub["mi_time"] + sub["mi_trial_id"]) if "mi_trial_id" in sub.columns else np.median(
            sub["mi_time"])

        # Net time score: rewards high time info while penalizing position/direction info
        net_time_score = time_score - pos_dir_total

        cluster_scores[cid] = {
            "pos": pos_score,
            "net_time": net_time_score
        }

    # 1. Position Cluster: Highest position score
    pos_cid = max(cluster_ids, key=lambda c: cluster_scores[c]["pos"])
    remaining_after_pos = [c for c in cluster_ids if c != pos_cid]

    if remaining_after_pos:
        # 2. Time Cluster: Highest net time score (maximizing time while minimizing position/direction info)
        time_cid = max(remaining_after_pos, key=lambda c: cluster_scores[c]["net_time"])
    else:
        time_cid = pos_cid

    # 3. Direction Cluster: Whatever remains
    remaining_after_time = [c for c in remaining_after_pos if c != time_cid]
    dir_cid = remaining_after_time[0] if remaining_after_time else (
        remaining_after_pos[0] if remaining_after_pos else pos_cid)

    return {pos_cid: 2, time_cid: 0, dir_cid: 1}


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

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
    raw_labels = kmeans.fit_predict(X_scaled)

    remap_dict = infer_intra_dataset_semantic_mapping(df_sub, raw_labels)
    semantic_labels = np.array([remap_dict.get(int(l), UNASSIGNED_LABEL) for l in raw_labels], dtype=int)

    tsne = TSNE(
        n_components=2, perplexity=min(TSNE_PERPLEXITY, max(5, len(df_sub) // 3)),
        initialization=TSNE_INITIALIZATION, metric=TSNE_METRIC, n_jobs=TSNE_N_JOBS,
        random_state=TSNE_RANDOM_STATE, verbose=False,
    )
    embedding = np.asarray(tsne.fit(X_scaled))

    df_sub["cluster_initial"] = raw_labels
    df_sub["cluster_id"] = semantic_labels
    df_sub["tsne_1"] = embedding[:, 0]
    df_sub["tsne_2"] = embedding[:, 1]

    return df_sub


# =============================================================================
# Plotting & Advanced Analyses
# =============================================================================

def plot_cluster_colored_log_contours(df_plot, out_png_dir, out_svg_dir, prefix):
    plot_df = df_plot[df_plot["cluster_id"] != UNASSIGNED_LABEL].copy()
    if plot_df.empty:
        return

    for col in ["mi_position", "mi_time", "mi_cwt_ripple", "mi_cwt_mua", "mi_cwt_slow"]:
        if col in plot_df.columns:
            plot_df[col] = np.maximum(plot_df[col], 1e-4)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [
        ("mi_position", "mi_cwt_ripple", "PositionMIR vs CWT Ripple Power"),
        ("mi_position", "mi_cwt_mua", "PositionMIR vs CWT MUA Power"),
        ("mi_time", "mi_cwt_slow", "TimeMIR vs CWT Slow Power")
    ]

    cluster_name_lookup = {0: "Time", 1: "Direction", 2: "Position"}

    for idx, (x_col, y_col, title) in enumerate(pairs):
        ax = axes[idx]
        if x_col not in plot_df.columns or y_col not in plot_df.columns:
            continue

        sns.scatterplot(
            data=plot_df, x=x_col, y=y_col, hue="cluster_id",
            palette=CLUSTER_COLOR_MAP, s=35, alpha=0.8, ax=ax, edgecolor="none", legend=(idx == 2)
        )
        sns.kdeplot(
            data=plot_df, x=x_col, y=y_col,
            levels=5, color="black", linewidths=1.2, ax=ax
        )

        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(x_col.replace("mi_", "").capitalize(), fontsize=10)
        ax.set_ylabel(y_col.replace("mi_", "CWT ").replace("_", " ").capitalize() + " (Log Scale)", fontsize=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        if idx == 2 and ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()
            mapped_labels = [cluster_name_lookup.get(int(float(l)), l) for l in labels]
            ax.legend(handles=handles, labels=mapped_labels, title="Cluster", fontsize=9, title_fontsize=10)

    plt.tight_layout()
    save_figure_both(fig, out_png_dir, out_svg_dir, f"{prefix}_cluster_colored_log_contours")


def plot_lfp_embedding_suite(df_plot, out_png_dir, out_svg_dir, out_base_prefix):
    lfp_cols = discover_lfp_columns(df_plot)
    if not lfp_cols:
        return

    modalities = {"waveform": [], "envelope": [], "phase": [], "cwt": []}
    for c in lfp_cols:
        band, mod = parse_lfp_feature_metadata(c)
        if mod in modalities:
            modalities[mod].append((band, c))

    for mod, items in modalities.items():
        if not items:
            continue

        scales_to_plot = ["log", "linear"] if mod == "cwt" else ["linear"]

        for scale_type in scales_to_plot:
            for with_colorbar in [True, False]:
                fig, axs = plt.subplots(3, 3, figsize=(12, 10))
                axs = axs.ravel()

                for idx, (band, col) in enumerate(items):
                    if idx >= len(axs):
                        break
                    ax = axs[idx]
                    vals = df_plot[col]

                    if mod == "cwt" and scale_type == "log":
                        valid_vals = vals[vals > 0]
                        vmin = valid_vals.min() if len(valid_vals) > 0 else 1e-4
                        vmax = valid_vals.max() if len(valid_vals) > 0 else 1.0
                        norm = LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
                        cmap = "viridis"
                        scale_suffix = "log"
                    elif mod == "phase":
                        norm = Normalize(vmin=-np.pi, vmax=np.pi)
                        cmap = "twilight"
                        scale_suffix = "linear"
                    else:
                        p2, p98 = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
                        norm = Normalize(vmin=p2, vmax=max(p98, p2 + 1e-3))
                        cmap = "viridis"
                        scale_suffix = "linear"

                    sc = ax.scatter(
                        df_plot["tsne_1"],
                        df_plot["tsne_2"],
                        c=vals,
                        cmap=cmap,
                        norm=norm,
                        s=SCATTER_SIZE,
                        alpha=0.85,
                    )
                    ax.set_title(f"tSNE by {band.capitalize()} ({mod}) [{scale_suffix}]", fontsize=10)
                    ax.grid(False)

                    if with_colorbar:
                        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)

                for j in range(idx + 1, len(axs)):
                    axs[j].axis("off")

                scale_label = f"({scale_type.capitalize()} Scale)" if mod == "cwt" else ""
                cb_label = "with Colorbar" if with_colorbar else "without Colorbar"
                plt.suptitle(f"t-SNE Embeddings Colored by LFP {mod.capitalize()}s {scale_label} [{cb_label}]",
                             fontsize=12, y=0.98)
                plt.tight_layout()

                grid_suffix = f"_lfp_{mod}_embedding_grid_{scale_type}" if mod == "cwt" else f"_lfp_{mod}_embedding_grid_standardized"
                cb_suffix = "_with_cb" if with_colorbar else "_no_cb"
                save_figure_both(fig, out_png_dir, out_svg_dir, out_base_prefix + grid_suffix + cb_suffix)


def plot_lfp_feature_cluster_barplots(df_plot, out_png_dir, out_svg_dir, stats_dir, out_base_prefix):
    plot_df = df_plot[df_plot["cluster_id"] != UNASSIGNED_LABEL].copy()
    lfp_cols = discover_lfp_columns(plot_df)
    if not lfp_cols:
        return

    cluster_names_order = ["Direction", "Time", "Position"]
    palette_colors = [
        CLUSTER_COLOR_MAP_STR["1"],  # Direction (Cyan)
        CLUSTER_COLOR_MAP_STR["0"],  # Time (Green)
        CLUSTER_COLOR_MAP_STR["2"]  # Position (Pink)
    ]

    for col in lfp_cols:
        band, mod = parse_lfp_feature_metadata(col)
        sub_df = plot_df[["cluster_id", col]].dropna().copy()
        sub_df["cluster_str"] = sub_df["cluster_id"].astype(int).astype(str)
        sub_df["cluster_name"] = sub_df["cluster_str"].map(CLUSTER_NAME_MAP_STR)
        sub_df["feature_val"] = sub_df[col]

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(
            data=sub_df, x="cluster_name", y="feature_val", order=cluster_names_order,
            hue="cluster_name", hue_order=cluster_names_order,
            palette=palette_colors, capsize=0.1, err_kws={'linewidth': 1.5}, ax=ax, alpha=0.85, legend=False
        )
        sns.stripplot(
            data=sub_df, x="cluster_name", y="feature_val", order=cluster_names_order,
            color="black", alpha=0.6, size=5, jitter=0.15, ax=ax
        )
        ax.set_title(f"LFP {mod.capitalize()} ({band}) by Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel(f"MI Value ({band} {mod})")
        plt.tight_layout()

        feat_slug = sanitize_for_filename(col)
        save_figure_both(fig, out_png_dir, out_svg_dir, f"{out_base_prefix}_barplot_{feat_slug}")


def analyze_per_behavior_vs_lfp_regression(df_plot, out_png_dir, out_svg_dir, stats_dir, out_base_prefix):
    plot_df = df_plot[df_plot["cluster_id"] != UNASSIGNED_LABEL].copy()
    lfp_cols = discover_lfp_columns(plot_df)
    if not lfp_cols:
        return

    all_reg_results = []
    cluster_reg_results = []

    for beh_col, beh_title in zip(MIR_FEATURE_COLUMNS, MIR_FEATURE_TITLES):
        beh_results = []
        cluster_beh_results = {0: [], 1: [], 2: []}

        for lfp_col in lfp_cols:
            band, mod = parse_lfp_feature_metadata(lfp_col)
            sub = plot_df[[beh_col, lfp_col, "cluster_id"]].dropna().copy()
            if len(sub) < 10:
                continue

            X = sub[[beh_col]].values
            y = sub[lfp_col].values
            reg = LinearRegression().fit(X, y)
            r_val, p_val = pearsonr(sub[beh_col], sub[lfp_col])

            res_dict = {
                "behavioral_feature": beh_col,
                "behavioral_title": beh_title,
                "lfp_feature": lfp_col,
                "band": band,
                "modality": mod,
                "slope": reg.coef_[0],
                "intercept": reg.intercept_,
                "r_value": r_val,
                "p_value": p_val,
            }
            beh_results.append(res_dict)
            all_reg_results.append(res_dict)

            for cid in [0, 1, 2]:
                sub_c = sub[sub["cluster_id"] == cid]
                if len(sub_c) >= 5:
                    r_c, p_c = pearsonr(sub_c[beh_col], sub_c[lfp_col])
                    c_dict = {
                        "cluster_id": cid,
                        "cluster_name": CLUSTER_NAME_MAP_STR[str(cid)],
                        "behavioral_feature": beh_col,
                        "behavioral_title": beh_title,
                        "lfp_feature": lfp_col,
                        "band": band,
                        "modality": mod,
                        "r_value": r_c,
                        "p_value": p_c,
                    }
                    cluster_beh_results[cid].append(c_dict)
                    cluster_reg_results.append(c_dict)

        beh_df = pd.DataFrame(beh_results)
        if not beh_df.empty and "modality" in beh_df.columns:
            pivot_r = beh_df.pivot(index="band", columns="modality", values="r_value").fillna(0)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(pivot_r, annot=True, fmt=".3f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
            ax.set_title(f"Pearson r ({beh_title} vs LFP Features) [Global]", fontsize=10)
            plt.tight_layout()
            beh_slug = sanitize_for_filename(beh_col)
            save_figure_both(fig, out_png_dir, out_svg_dir, f"{out_base_prefix}_correl_heatmap_{beh_slug}")

            cwt_sub = beh_df[beh_df["modality"] == "cwt"].copy()
            if not cwt_sub.empty:
                cwt_sub["freq"] = cwt_sub["band"].map(BAND_FREQS)
                cwt_sub = cwt_sub.sort_values("freq")

                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.plot(cwt_sub["freq"], cwt_sub["r_value"], marker="o", linestyle="-", color="black", linewidth=2,
                        label="Global", alpha=0.8)

                for cid in [0, 1, 2]:
                    c_df = pd.DataFrame(cluster_beh_results[cid])
                    if not c_df.empty and "modality" in c_df.columns:
                        c_cwt = c_df[c_df["modality"] == "cwt"].copy()
                        if not c_cwt.empty:
                            c_cwt["freq"] = c_cwt["band"].map(BAND_FREQS)
                            c_cwt = c_cwt.sort_values("freq")
                            c_name = CLUSTER_NAME_MAP_STR[str(cid)]
                            c_color = CLUSTER_COLOR_MAP[cid]
                            ax.plot(c_cwt["freq"], c_cwt["r_value"], marker="s", linestyle="--", color=c_color,
                                    linewidth=1.8, label=f"{c_name}")

                ax.set_xscale("log")
                ax.set_xlabel("Frequency Band Center (Hz)", fontsize=10)
                ax.set_ylabel("Pearson r (CWT Power vs " + beh_title + ")", fontsize=10)
                ax.set_title(f"CWT Pearson r vs Frequency Band ({beh_title})", fontsize=11)
                ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
                ax.set_xticks([0.5, 2.0, 10.0, 65.0, 175.0, 650.0])
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                ax.legend(title="Population / Cluster", fontsize=9)
                plt.tight_layout()
                beh_slug = sanitize_for_filename(beh_col)
                save_figure_both(fig, out_png_dir, out_svg_dir, f"{out_base_prefix}_cwt_pearson_vs_freq_{beh_slug}")


def analyze_and_plot_subset(df_clustered, sub_png_dir, sub_svg_dir, sub_stats_dir, prefix):
    feature_cols = [f"z_{c}" for c in MIR_FEATURE_COLUMNS]
    df_plot, _ = build_mouse_index(df_clustered)

    for with_colorbar in [True, False]:
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
            if with_colorbar:
                plt.colorbar(sc, ax=axs[i + 2])
        for ax in axs:
            ax.grid(False)
        plt.tight_layout()
        cb_suffix = "_with_cb" if with_colorbar else "_no_cb"
        save_figure_both(fig, sub_png_dir, sub_svg_dir, f"{prefix}_clusters_grid" + cb_suffix)

    plot_lfp_embedding_suite(df_plot, sub_png_dir, sub_svg_dir, prefix)
    plot_cluster_colored_log_contours(df_clustered, sub_png_dir, sub_svg_dir, prefix)

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

    plot_lfp_feature_cluster_barplots(df_clustered, sub_png_dir, sub_svg_dir, sub_stats_dir, prefix)
    analyze_per_behavior_vs_lfp_regression(df_clustered, sub_png_dir, sub_svg_dir, sub_stats_dir, prefix)


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)

    if not os.path.isfile(MI_CSV):
        raise FileNotFoundError(f"MI CSV not found: {MI_CSV}")
    if not os.path.isfile(MASTER_INDEX_CSV):
        raise FileNotFoundError(f"Master index CSV not found: {MASTER_INDEX_CSV}")

    try:
        df_mi = pd.read_csv(MI_CSV, engine="python", on_bad_lines="skip")
    except Exception:
        df_mi = pd.read_csv(MI_CSV, on_bad_lines="skip")

    try:
        df_master = pd.read_csv(MASTER_INDEX_CSV, engine="python", on_bad_lines="skip")
    except Exception:
        df_master = pd.read_csv(MASTER_INDEX_CSV, on_bad_lines="skip")

    join_cols = ["dataset_name", "mouse_name", "session_name", "depth_type"]
    add_cols = [c for c in ["modality", "paradigm", "model", "int_i", "int_f"] if c in df_master.columns]
    df_all = df_mi.merge(df_master[join_cols + add_cols].drop_duplicates(), on=join_cols, how="left")

    subsets_to_process = {}

    base_datasets = [d for d in df_all["dataset_name"].dropna().astype(str).unique() if
                     "lt_rot" not in d.lower() and "intervention" not in d.lower() and "egrin" not in d.lower()]

    for ds in base_datasets:
        subsets_to_process[ds] = df_all[df_all["dataset_name"].astype(str).str.lower() == ds.lower()].copy()

    egrin_mask = df_all["dataset_name"].astype(str).str.lower().str.contains("egrin")
    if egrin_mask.any():
        subsets_to_process["lt_egrin"] = df_all[egrin_mask].copy()

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