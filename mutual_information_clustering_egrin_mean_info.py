#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot Mean MI Values per Cluster (Behavior + LFP Envelopes & Phases) for lt_egrin.

Generates:
1. Figure 1: Mean MI for Behaviors (Position, Time) + 8 LFP Envelopes per cluster.
2. Figure 2: Mean MI for Behaviors (Position, Time) + 8 LFP Phases per cluster.
3. Figure 3: Combined 2-panel comparative layout (Envelopes on Top, Phases on Bottom).

Formatting:
- Grouped bar plot with mean ± SEM error bars.
- Jittered scatter points overlaid on top of each bar showing individual neuron MI.
- Clusters: Time (#bce784ff), Direction (#66cef4ff), Position (#ec8ef8ff), Unassigned (#bbbcc0ff).
- Clean scientific styling (no top/right spines, labeled ticks).
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Directories & Settings
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
TRANSFER_DIR = os.path.join(PROJECT_ROOT, "analysis", "opentsne_universal_transfer_and_lfp", "transfer", "egrin_all")
PLOTS_DIR = os.path.join(TRANSFER_DIR, "plots")
TABLES_DIR = os.path.join(TRANSFER_DIR, "tables")

CLUSTER_COLORS = {
    "Time": "#bce784ff",  # Light Green
    "Direction": "#66cef4ff",  # Sky Blue
    "Position": "#ec8ef8ff",  # Violet / Orchid
    "Unassigned": "#bbbcc0ff",  # Gray
}

CLUSTER_ORDER = ["Time", "Direction", "Position", "Unassigned"]

BANDS = [
    "ultra_ultra_slow",
    "ultra_slow",
    "slow",
    "delta",
    "theta",
    "gamma",
    "ripple",
    "mua",
]

BAND_DISPLAY_NAMES = [
    "U-U-Slow",
    "U-Slow",
    "Slow",
    "Delta",
    "Theta",
    "Gamma",
    "Ripple",
    "MUA",
]


# =============================================================================
# Plotting Helper
# =============================================================================

def render_grouped_bar_scatter(
        df,
        feature_keys,
        feature_labels,
        title,
        ylabel,
        save_path,
        ylim=None,
        jitter_scale=0.08,
):
    """
    Renders grouped bar plots with SEM error bars and individual neuron scatter dots.

    feature_keys: list of column names in df (e.g., ['mi_position', 'mi_time', 'mi_lfp_theta_envelope'])
    feature_labels: list of clean display strings for x-axis ticks
    """
    n_features = len(feature_keys)
    n_clusters = len(CLUSTER_ORDER)
    x_indices = np.arange(n_features)

    bar_width = 0.80 / n_clusters

    fig, ax = plt.subplots(figsize=(max(14.0, n_features * 1.3), 5.5), dpi=300)

    for c_idx, cluster_name in enumerate(CLUSTER_ORDER):
        c_sub = df[df["cluster_label"] == cluster_name]
        c_color = CLUSTER_COLORS[cluster_name]

        # Shift per cluster bar
        offset = (c_idx - (n_clusters - 1) / 2.0) * bar_width
        cluster_x = x_indices + offset

        means = []
        sems = []

        for f_idx, feat in enumerate(feature_keys):
            if feat in c_sub.columns:
                vals = c_sub[feat].dropna().values
            else:
                vals = np.array([])

            if len(vals) > 0:
                m = np.mean(vals)
                s = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                means.append(m)
                sems.append(s)

                # Overlaid individual neuron scatter points with horizontal jitter
                jitter = np.random.normal(0, jitter_scale * bar_width, size=len(vals))
                ax.scatter(
                    cluster_x[f_idx] + jitter,
                    vals,
                    color=c_color,
                    edgecolors="#333333",
                    linewidth=0.3,
                    s=14,
                    alpha=0.45,
                    zorder=3,
                )
            else:
                means.append(0.0)
                sems.append(0.0)

        # Main bars
        ax.bar(
            cluster_x,
            means,
            yerr=sems,
            width=bar_width,
            color=c_color,
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.85,
            label=cluster_name,
            zorder=2,
            capsize=3,
            error_kw={"elinewidth": 1.2, "ecolor": "#222222", "capthick": 1.0},
        )

    # Style
    ax.set_xticks(x_indices)
    ax.set_xticklabels(feature_labels, fontsize=11, rotation=25, ha="right")
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=13, pad=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)

    if ylim is not None:
        ax.set_ylim(ylim)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Divider line separating Behavioral MI from LFP Band MIs
    ax.axvline(1.5, color="#555555", linestyle="--", lw=1.2, alpha=0.7, zorder=1)

    # Text labels for feature blocks
    y_top = ax.get_ylim()[1]
    ax.text(0.5, y_top * 0.94, "Behavior", ha="center", va="top", fontsize=11, fontweight="bold", color="#333333")
    ax.text((n_features + 1.5) / 2.0, y_top * 0.94, "LFP Spectral Bands", ha="center", va="top", fontsize=11,
            fontweight="bold", color="#333333")

    ax.legend(title="Cluster", frameon=True, fontsize=10, loc="upper right")
    fig.tight_layout()

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Plot -> {save_path}")


# =============================================================================
# Main Script
# =============================================================================

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Locate labeled neuron CSV produced during transfer
    csv_candidates = [
        os.path.join(TRANSFER_DIR, "egrin_all_labeled_neurons.csv"),
        os.path.join(TRANSFER_DIR, "lt_egrin_labeled_neurons.csv"),
    ]
    csv_path = None
    for c in csv_candidates:
        if os.path.isfile(c):
            csv_path = c
            break

    if csv_path is None:
        # Fallback to any labeled CSV matching egrin
        matches = glob.glob(os.path.join(PROJECT_ROOT, "analysis", "**", "*egrin*labeled_neurons*.csv"), recursive=True)
        if matches:
            csv_path = matches[0]
        else:
            raise FileNotFoundError(f"Could not locate labeled neurons CSV for egrin under {PROJECT_ROOT}/analysis")

    print(f"Loading egrin clustered neurons from: {csv_path}")
    df_egrin = pd.read_csv(csv_path)

    # -------------------------------------------------------------------------
    # 1. Feature Setup: Position, Time, and 8 LFP Envelopes
    # -------------------------------------------------------------------------
    env_keys = ["mi_position", "mi_time"] + [f"mi_lfp_{b}_envelope" for b in BANDS]
    env_labels = ["Position", "Time"] + [f"{b_name} Env" for b_name in BAND_DISPLAY_NAMES]

    out_env = os.path.join(PLOTS_DIR, "lt_egrin_mean_mi_behavior_and_lfp_envelopes_per_cluster.png")
    render_grouped_bar_scatter(
        df=df_egrin,
        feature_keys=env_keys,
        feature_labels=env_labels,
        title="lt_egrin | Mean Information: Behavior & LFP Amplitude Envelopes per Cluster",
        ylabel="Mutual Information (bits / nat)",
        save_path=out_env,
    )

    # -------------------------------------------------------------------------
    # 2. Feature Setup: Position, Time, and 8 LFP Phases
    # -------------------------------------------------------------------------
    phase_keys = ["mi_position", "mi_time"] + [f"mi_lfp_{b}_phase" for b in BANDS]
    phase_labels = ["Position", "Time"] + [f"{b_name} Phase" for b_name in BAND_DISPLAY_NAMES]

    out_phase = os.path.join(PLOTS_DIR, "lt_egrin_mean_mi_behavior_and_lfp_phases_per_cluster.png")
    render_grouped_bar_scatter(
        df=df_egrin,
        feature_keys=phase_keys,
        feature_labels=phase_labels,
        title="lt_egrin | Mean Information: Behavior & LFP Instantaneous Phases per Cluster",
        ylabel="Mutual Information (bits / nat)",
        save_path=out_phase,
    )

    # -------------------------------------------------------------------------
    # 3. Combined Dual-Panel Plot (Envelopes Top, Phases Bottom)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(16.0, 10.0), sharex=False, dpi=300)
    fig.subplots_adjust(hspace=0.45)

    panels = [
        (axes[0], env_keys, env_labels, "LFP Amplitude Envelopes", "mi_lfp_{b}_envelope"),
        (axes[1], phase_keys, phase_labels, "LFP Instantaneous Phases", "mi_lfp_{b}_phase"),
    ]

    n_features = len(env_keys)
    n_clusters = len(CLUSTER_ORDER)
    x_indices = np.arange(n_features)
    bar_width = 0.80 / n_clusters

    for ax, f_keys, f_labels, sub_title, _ in panels:
        for c_idx, cluster_name in enumerate(CLUSTER_ORDER):
            c_sub = df_egrin[df_egrin["cluster_label"] == cluster_name]
            c_color = CLUSTER_COLORS[cluster_name]
            offset = (c_idx - (n_clusters - 1) / 2.0) * bar_width
            cluster_x = x_indices + offset

            means, sems = [], []
            for f_idx, feat in enumerate(f_keys):
                vals = c_sub[feat].dropna().values if feat in c_sub.columns else np.array([])
                if len(vals) > 0:
                    means.append(np.mean(vals))
                    sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
                    jitter = np.random.normal(0, 0.08 * bar_width, size=len(vals))
                    ax.scatter(
                        cluster_x[f_idx] + jitter,
                        vals,
                        color=c_color,
                        edgecolors="#333333",
                        linewidth=0.3,
                        s=12,
                        alpha=0.40,
                        zorder=3,
                    )
                else:
                    means.append(0.0)
                    sems.append(0.0)

            ax.bar(
                cluster_x,
                means,
                yerr=sems,
                width=bar_width,
                color=c_color,
                edgecolor="#222222",
                linewidth=0.8,
                alpha=0.85,
                label=cluster_name,
                zorder=2,
                capsize=3,
                error_kw={"elinewidth": 1.2, "ecolor": "#222222", "capthick": 1.0},
            )

        ax.set_xticks(x_indices)
        ax.set_xticklabels(f_labels, fontsize=10, rotation=22, ha="right")
        ax.set_ylabel("Mutual Information", fontsize=11, labelpad=6)
        ax.set_title(f"Position & Time vs. {sub_title} per Cluster", fontsize=12, fontweight="bold", pad=8)
        ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
        ax.axvline(1.5, color="#555555", linestyle="--", lw=1.2, alpha=0.7, zorder=1)

        y_max = ax.get_ylim()[1]
        ax.text(0.5, y_max * 0.93, "Behavior", ha="center", va="top", fontsize=10, fontweight="bold", color="#333333")
        ax.text((n_features + 1.5) / 2.0, y_max * 0.93, "LFP Spectral Bands", ha="center", va="top", fontsize=10,
                fontweight="bold", color="#333333")

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        if ax == axes[0]:
            ax.legend(title="Cluster", frameon=True, fontsize=9, loc="upper right")

    fig.suptitle("lt_egrin | Neuronal Information Representation: Behavior vs. LFP Spectral Modalities", fontsize=14,
                 y=0.98, fontweight="bold")
    out_comb = os.path.join(PLOTS_DIR, "lt_egrin_mean_mi_combined_behavior_and_lfp_suite.png")
    fig.savefig(out_comb, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Combined Suite Plot -> {out_comb}")

    # -------------------------------------------------------------------------
    # 4. Summary Table Export (Mean ± SEM per Feature per Cluster)
    # -------------------------------------------------------------------------
    all_summary_keys = ["mi_position", "mi_time"] + [f"mi_lfp_{b}_envelope" for b in BANDS] + [f"mi_lfp_{b}_phase" for b
                                                                                               in BANDS]
    summary_records = []

    for feat in all_summary_keys:
        row = {"feature": feat}
        for cl in CLUSTER_ORDER:
            sub = df_egrin[df_egrin["cluster_label"] == cl]
            vals = sub[feat].dropna().values if feat in sub.columns else np.array([])
            if len(vals) > 0:
                row[f"{cl}_mean"] = np.mean(vals)
                row[f"{cl}_sem"] = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                row[f"{cl}_n"] = len(vals)
            else:
                row[f"{cl}_mean"] = np.nan
                row[f"{cl}_sem"] = np.nan
                row[f"{cl}_n"] = 0
        summary_records.append(row)

    summary_df = pd.DataFrame(summary_records)
    table_path = os.path.join(TABLES_DIR, "lt_egrin_cluster_mi_means_and_sems.csv")
    summary_df.to_csv(table_path, index=False)
    print(f"[OK] Saved Statistics Table -> {table_path}")


if __name__ == "__main__":
    main()