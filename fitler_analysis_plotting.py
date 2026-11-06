#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive bar plotting and statistical comparison (Mann-Whitney U: Deep vs Superficial)
for MEB expansion, Structure Index (SI), and Persistent Homology (TDA H1 metrics)
across filter sigmas [4, 8, 16].

Author: Adapted for Melisa Beatriz Maidana Capitán / Emanuel Sánchez Aimar
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# =============================================================================
# PATHS & CONFIGURATION
# =============================================================================

POSSIBLE_ROOTS = [
    Path("/home/hippolab/Documents/melisa/time_project_standarization"),
    Path("/home/melisamc/Documents/time_project_standarization"),
]
BASE_STD_DIR = next((str(p) for p in POSSIBLE_ROOTS if p.exists()), str(POSSIBLE_ROOTS[0]))

SAVE_ROOT = os.path.join(BASE_STD_DIR, "analysis", "filter_influence_combined_complete")
PLOTS_DIR = os.path.join(SAVE_ROOT, "summary_bar_plots_stats_complete")
os.makedirs(PLOTS_DIR, exist_ok=True)

AREA_PALETTE = {
    "superficial": "#9900ff",
    "deep": "#cc9900",
    "whole population": "#666666"
}


# =============================================================================
# LOAD DATA SAFELY
# =============================================================================

def load_analysis_data():
    meb_path = os.path.join(SAVE_ROOT, "filter_analysis_meb_all_sessions.csv")
    si_path = os.path.join(SAVE_ROOT, "filter_analysis_si_all_sessions.csv")
    tda_path = os.path.join(SAVE_ROOT, "filter_analysis_tda_all_sessions.csv")

    def safe_read_csv(path):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                return pd.read_csv(path)
            except Exception as e:
                print(f"[WARNING] Could not parse {path}: {e}")
        return pd.DataFrame()

    return safe_read_csv(meb_path), safe_read_csv(si_path), safe_read_csv(tda_path)


def save_stats_txt(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(str(line) + "\n")


# =============================================================================
# 1. MEB STATS & BAR PLOT
# =============================================================================

def plot_and_stat_meb(df_meb):
    if df_meb.empty or "meb_expansion" not in df_meb.columns:
        print("[WARNING] MEB dataframe is empty or missing required columns.")
        return

    datasets = sorted(df_meb["dataset_name"].unique())
    stats_lines = []

    for ds in datasets:
        sub = df_meb[(df_meb["dataset_name"] == ds) & (df_meb["depth_type"].isin(["deep", "superficial"]))].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        sns.barplot(
            data=sub, x="filter_sigma", y="meb_expansion", hue="depth_type",
            palette=AREA_PALETTE, errorbar="sd", capsize=0.1, ax=ax
        )

        ax.set_title(f"MEB Expansion: Deep vs Superficial | Dataset: {ds}", fontsize=11, fontweight="bold")
        ax.set_ylabel("MEB Expansion (Norm)", fontsize=10)
        ax.set_xlabel("Filter Sigma (frames)", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        stats_lines.append(f"=== DATASET: {ds} | MEB EXPANSION ===")
        sigmas = sorted(sub["filter_sigma"].unique())
        y_max = sub["meb_expansion"].max() if not sub["meb_expansion"].isnull().all() else 1.0

        for i, sig in enumerate(sigmas):
            sub_sig = sub[sub["filter_sigma"] == sig]
            deep_vals = sub_sig[sub_sig["depth_type"] == "deep"]["meb_expansion"].dropna().values
            sup_vals = sub_sig[sub_sig["depth_type"] == "superficial"]["meb_expansion"].dropna().values

            if len(deep_vals) >= 2 and len(sup_vals) >= 2:
                stat, pval = mannwhitneyu(deep_vals, sup_vals, alternative="two-sided")
                stats_lines.append(f"  Sigma {sig}: U={stat:.3f}, p={pval:.6g} (Deep mean: {np.mean(deep_vals):.4f}, Sup mean: {np.mean(sup_vals):.4f})")
                if pval < 0.05:
                    ax.text(i, y_max * 1.05, "*", ha="center", va="bottom", fontsize=14, fontweight="bold", color="black")
            else:
                stats_lines.append(f"  Sigma {sig}: Insufficient data for Mann-Whitney U")

        stats_lines.append("")
        sns.despine(ax=ax)
        plt.tight_layout()

        fig.savefig(os.path.join(PLOTS_DIR, f"MEB_bar_{ds}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(PLOTS_DIR, f"MEB_bar_{ds}.svg"), bbox_inches="tight")
        plt.close(fig)

    save_stats_txt(os.path.join(PLOTS_DIR, "MEB_stats_deep_vs_sup.txt"), stats_lines)
    print("[OK] MEB bar plots and statistics saved.")


# =============================================================================
# 2. STRUCTURE INDEX (SI) STATS & BAR PLOT
# =============================================================================

def plot_and_stat_si(df_si):
    if df_si.empty or "si" not in df_si.columns:
        print("[WARNING] SI dataframe is empty or missing required columns.")
        return

    datasets = sorted(df_si["dataset_name"].unique())
    behaviors = sorted(df_si["behavioral_label"].unique())
    stats_lines = []

    for ds in datasets:
        sub_ds = df_si[(df_si["dataset_name"] == ds) & (df_si["depth_type"].isin(["deep", "superficial"]))].copy()
        if sub_ds.empty:
            continue

        stats_lines.append(f"=== DATASET: {ds} | STRUCTURE INDEX (SI) ===")

        for beh in behaviors:
            sub_b = sub_ds[sub_ds["behavioral_label"] == beh]
            if sub_b.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
            sns.barplot(
                data=sub_b, x="filter_sigma", y="si", hue="depth_type",
                palette=AREA_PALETTE, errorbar="sd", capsize=0.1, ax=ax
            )

            ax.set_title(f"SI ({beh}): Deep vs Superficial | Dataset: {ds}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Structure Index", fontsize=10)
            ax.set_xlabel("Filter Sigma (frames)", fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

            stats_lines.append(f"--- Behavior: {beh} ---")
            sigmas = sorted(sub_b["filter_sigma"].unique())
            y_max = sub_b["si"].max() if not sub_b["si"].isnull().all() else 1.0

            for i, sig in enumerate(sigmas):
                sub_sig = sub_b[sub_b["filter_sigma"] == sig]
                deep_vals = sub_sig[sub_sig["depth_type"] == "deep"]["si"].dropna().values
                sup_vals = sub_sig[sub_sig["depth_type"] == "superficial"]["si"].dropna().values

                if len(deep_vals) >= 2 and len(sup_vals) >= 2:
                    stat, pval = mannwhitneyu(deep_vals, sup_vals, alternative="two-sided")
                    stats_lines.append(f"  Sigma {sig}: U={stat:.3f}, p={pval:.6g} (Deep mean: {np.mean(deep_vals):.4f}, Sup mean: {np.mean(sup_vals):.4f})")
                    if pval < 0.05:
                        ax.text(i, min(1.0, y_max * 1.05), "*", ha="center", va="bottom", fontsize=14, fontweight="bold", color="black")
                else:
                    stats_lines.append(f"  Sigma {sig}: Insufficient data")

            stats_lines.append("")
            sns.despine(ax=ax)
            plt.tight_layout()

            safe_beh = beh.replace(",", "_").replace("(", "").replace(")", "")
            fig.savefig(os.path.join(PLOTS_DIR, f"SI_bar_{ds}_{safe_beh}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(PLOTS_DIR, f"SI_bar_{ds}_{safe_beh}.svg"), dpi=300, bbox_inches="tight")
            plt.close(fig)

    save_stats_txt(os.path.join(PLOTS_DIR, "SI_stats_deep_vs_sup.txt"), stats_lines)
    print("[OK] SI bar plots and statistics saved.")


# =============================================================================
# 3. PERSISTENT HOMOLOGY (TDA H1) STATS & BAR PLOTS
# =============================================================================

def plot_and_stat_tda(df_tda):
    if df_tda.empty or "umap_h1_max_lifetime" not in df_tda.columns:
        print("[WARNING] TDA dataframe is empty or missing required columns.")
        return

    datasets = sorted(df_tda["dataset_name"].unique())
    stats_lines = []

    metrics_to_plot = [
        ("umap_h1_max_lifetime", "H1 Max Lifetime"),
        ("umap_h1_ratio", "H1 Lifetime Ratio (L1 / L2)")
    ]

    for ds in datasets:
        sub = df_tda[(df_tda["dataset_name"] == ds) & (df_tda["depth_type"].isin(["deep", "superficial"]))].copy()
        if sub.empty:
            continue

        stats_lines.append(f"=== DATASET: {ds} | PERSISTENT HOMOLOGY (TDA) ===")

        for metric_col, metric_label in metrics_to_plot:
            if metric_col not in sub.columns:
                continue

            fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
            sns.barplot(
                data=sub, x="filter_sigma", y=metric_col, hue="depth_type",
                palette=AREA_PALETTE, errorbar="sd", capsize=0.1, ax=ax
            )

            ax.set_title(f"{metric_label}: Deep vs Superficial | Dataset: {ds}", fontsize=11, fontweight="bold")
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_xlabel("Filter Sigma (frames)", fontsize=10)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

            stats_lines.append(f"--- Metric: {metric_label} ---")
            sigmas = sorted(sub["filter_sigma"].unique())
            y_max = sub[metric_col].max() if not sub[metric_col].isnull().all() else 1.0

            for i, sig in enumerate(sigmas):
                sub_sig = sub[sub["filter_sigma"] == sig]
                deep_vals = sub_sig[sub_sig["depth_type"] == "deep"][metric_col].dropna().values
                sup_vals = sub_sig[sub_sig["depth_type"] == "superficial"][metric_col].dropna().values

                if len(deep_vals) >= 2 and len(sup_vals) >= 2:
                    stat, pval = mannwhitneyu(deep_vals, sup_vals, alternative="two-sided")
                    stats_lines.append(f"  Sigma {sig}: U={stat:.3f}, p={pval:.6g} (Deep mean: {np.mean(deep_vals):.4f}, Sup mean: {np.mean(sup_vals):.4f})")
                    if pval < 0.05:
                        ax.text(i, y_max * 1.05, "*", ha="center", va="bottom", fontsize=14, fontweight="bold", color="black")
                else:
                    stats_lines.append(f"  Sigma {sig}: Insufficient data")

            stats_lines.append("")
            sns.despine(ax=ax)
            plt.tight_layout()

            safe_metric = metric_col.replace("_", "")
            fig.savefig(os.path.join(PLOTS_DIR, f"TDA_bar_{ds}_{safe_metric}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(PLOTS_DIR, f"TDA_bar_{ds}_{safe_metric}.svg"), dpi=300, bbox_inches="tight")
            plt.close(fig)

    save_stats_txt(os.path.join(PLOTS_DIR, "TDA_stats_deep_vs_sup.txt"), stats_lines)
    print("[OK] TDA bar plots and statistics saved.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading analysis results for complete bar plots and Mann-Whitney U tests...")
    df_meb, df_si, df_tda = load_analysis_data()

    print(f"Generating complete statistical bar plots in: {PLOTS_DIR}")
    plot_and_stat_meb(df_meb)
    plot_and_stat_si(df_si)
    plot_and_stat_tda(df_tda)
    print(f"\n[OK] Complete statistical bar plotting pipeline finished successfully!")


if __name__ == "__main__":
    main()