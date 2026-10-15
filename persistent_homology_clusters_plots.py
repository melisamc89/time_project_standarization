#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-Animal Statistical Summary and Visualization of Persistent Homology.

Compares:
- Original Neural Space (cleanSignal) vs. 3D UMAP Space
- Groups: Position, Time, Direction, Unassigned, All
- Metrics: H1 Prominence Ratio (L1 / L2) and Max H1 Lifetime

Generates:
1. Two-panel comparative boxplot figure (Original Space vs. UMAP Space).
2. Paired animal trajectory plots across clusters.
3. Full statistical reports (Descriptives, Tests vs. 1.0, FDR-corrected Pairwise Contrasts).
4. All figures saved in PNG (300 DPI) and SVG.
"""

from __future__ import annotations

import os
import itertools
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
from statsmodels.stats.multitest import multipletests

# =============================================================================
# 1. PATHS & SETTINGS
# =============================================================================

POSSIBLE_ROOTS = [
    Path("/home/hippolab/Documents/melisa/time_project_standarization"),
    Path("/home/melisamc/Documents/time_project_standarization"),
]
PROJECT_ROOT = next((p for p in POSSIBLE_ROOTS if p.exists()), POSSIBLE_ROOTS[0])

# Locate the output directory produced by the replication pipeline
TDA_DIR = PROJECT_ROOT / "analysis" / "tda_julio_exact_replication_mem_safe"
if not TDA_DIR.exists():
    TDA_DIR = PROJECT_ROOT / "analysis" / "tda_julio_exact_replication"

TABLES_DIR = TDA_DIR / "tables"
STATS_DIR = TDA_DIR / "stats"
PLOTS_DIR = TDA_DIR / "summary_plots"

for p in [STATS_DIR, PLOTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CSV_PATH = TABLES_DIR / "tda_julio_replication_summary.csv"

CLUSTER_COLORS = {
    "Position":   "#ec8ef8ff",    # Violet / Orchid
    "Time":       "#bce784ff",    # Light Green
    "Direction":  "#66cef4ff",    # Sky Blue
    "Unassigned": "#bbbcc0ff",    # Gray
    "All":        "#111111",      # Deep Charcoal
}

CLUSTER_ORDER = ["Position", "Time", "Direction", "Unassigned", "All"]
FUNCTIONAL_CLUSTERS = ["Position", "Time", "Direction", "Unassigned"]


# =============================================================================
# 2. STATISTICAL HELPERS
# =============================================================================

def pval_to_star(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def compute_descriptives(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    records = []
    for grp in CLUSTER_ORDER:
        sub = df[df["group"] == grp][metric_col].dropna().values
        n = len(sub)
        if n > 0:
            mean = np.mean(sub)
            sd = np.std(sub, ddof=1) if n > 1 else 0.0
            sem = sd / np.sqrt(n) if n > 1 else 0.0
            median = np.median(sub)
            q25, q75 = np.percentile(sub, [25, 75])
            iqr = q75 - q25
            min_v = np.min(sub)
            max_v = np.max(sub)
        else:
            mean = sd = sem = median = q25 = q75 = iqr = min_v = max_v = np.nan

        records.append({
            "Group": grp,
            "N": n,
            "Mean": mean,
            "SEM": sem,
            "SD": sd,
            "Median": median,
            "IQR": iqr,
            "25th": q25,
            "75th": q75,
            "Min": min_v,
            "Max": max_v,
        })
    return pd.DataFrame(records)


def run_null_tests_vs_unity(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    records = []
    for grp in CLUSTER_ORDER:
        vals = df[df["group"] == grp][metric_col].dropna().values
        n = len(vals)
        if n >= 3:
            t_stat, p_t_two = stats.ttest_1samp(vals, popmean=1.0)
            p_t = p_t_two / 2.0 if t_stat > 0 else (1.0 - (p_t_two / 2.0))

            diffs = vals - 1.0
            if np.all(diffs == 0):
                w_stat, p_w = 0.0, 1.0
            else:
                try:
                    w_res = stats.wilcoxon(diffs, alternative="greater")
                    w_stat, p_w = w_res.statistic, w_res.pvalue
                except Exception:
                    w_stat, p_w = np.nan, np.nan
        else:
            t_stat, p_t, w_stat, p_w = np.nan, np.nan, np.nan, np.nan

        records.append({
            "Group": grp,
            "N": n,
            "t_stat": t_stat,
            "p_val_ttest": p_t,
            "wilcoxon_stat": w_stat,
            "p_val_wilcoxon": p_w,
            "star": pval_to_star(p_t),
        })
    return pd.DataFrame(records)


def run_pairwise_comparisons(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    sub_clusters = FUNCTIONAL_CLUSTERS
    pairs = list(itertools.combinations(sub_clusters, 2))
    records = []

    for g1, g2 in pairs:
        v1_df = df[df["group"] == g1][["mouse", metric_col]].rename(columns={metric_col: "v1"}).dropna()
        v2_df = df[df["group"] == g2][["mouse", metric_col]].rename(columns={metric_col: "v2"}).dropna()
        merged = pd.merge(v1_df, v2_df, on="mouse", how="inner")

        n_paired = len(merged)
        if n_paired >= 3:
            d = merged["v1"] - merged["v2"]
            t_pair, p_pair = stats.ttest_rel(merged["v1"], merged["v2"])
            try:
                w_res = stats.wilcoxon(d)
                w_stat, p_w = w_res.statistic, w_res.pvalue
            except Exception:
                w_stat, p_w = np.nan, np.nan
            dz = np.mean(d) / np.std(d, ddof=1) if np.std(d, ddof=1) > 0 else 0.0
        else:
            t_pair, p_pair, w_stat, p_w, dz = np.nan, np.nan, np.nan, np.nan, np.nan

        records.append({
            "Contrast": f"{g1} vs. {g2}",
            "N_animals": n_paired,
            "Mean_Diff": np.mean(merged["v1"] - merged["v2"]) if n_paired > 0 else np.nan,
            "Cohen_dz": dz,
            "t_stat": t_pair,
            "p_paired_t": p_pair,
            "wilcoxon_stat": w_stat,
            "p_wilcoxon": p_w,
        })

    df_pairs = pd.DataFrame(records)
    valid_mask = df_pairs["p_paired_t"].notna()
    if valid_mask.sum() > 0:
        _, fdr_p, _, _ = multipletests(df_pairs.loc[valid_mask, "p_paired_t"], method="fdr_bh")
        df_pairs.loc[valid_mask, "p_fdr_bh"] = fdr_p
        df_pairs["star_fdr"] = df_pairs["p_fdr_bh"].apply(pval_to_star)
    else:
        df_pairs["p_fdr_bh"] = np.nan
        df_pairs["star_fdr"] = "n/a"

    return df_pairs


# =============================================================================
# 3. PLOTTING SUITE
# =============================================================================

def plot_dual_space_summary_boxplots(df: pd.DataFrame, save_base: Path):
    """Side-by-side comparison of H1 Prominence Ratio: Original Space vs. UMAP Space."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.0), dpi=300, sharey=False)
    groups = [g for g in CLUSTER_ORDER if g in df["group"].unique()]
    x_positions = np.arange(len(groups))

    panels = [
        {"ax": axes[0], "col": "orig_ratio_L1_L2", "title": "A. Original Neural Space (cleanSignal)"},
        {"ax": axes[1], "col": "umap_ratio_L1_L2", "title": "B. 3D UMAP Manifold Space"},
    ]

    for p in panels:
        ax = p["ax"]
        col = p["col"]

        box_data = [df[df["group"] == g][col].dropna().values for g in groups]

        bp = ax.boxplot(
            box_data,
            positions=x_positions,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="none", linewidth=1.8),
            whiskerprops=dict(linewidth=1.6),
            capprops=dict(linewidth=1.6),
            medianprops=dict(linewidth=2.0),
        )

        for idx, g in enumerate(groups):
            c = CLUSTER_COLORS[g]
            bp["boxes"][idx].set_edgecolor(c)
            bp["whiskers"][2 * idx].set_color(c)
            bp["whiskers"][2 * idx + 1].set_color(c)
            bp["caps"][2 * idx].set_color(c)
            bp["caps"][2 * idx + 1].set_color(c)
            bp["medians"][idx].set_color(c)

        np.random.seed(42)
        for idx, g in enumerate(groups):
            vals = df[df["group"] == g][col].dropna().values
            c = CLUSTER_COLORS[g]
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(
                x_positions[idx] + jitter,
                vals,
                color=c,
                s=52,
                alpha=0.9,
                zorder=4,
                edgecolors="#222222",
                linewidth=0.6,
            )

        # Baseline noise threshold
        ax.axhline(1.0, linestyle="--", color="black", linewidth=1.2, alpha=0.7)
        ax.text(x_positions[-1] - 0.45, 1.03, "Noise Floor (1.0)", fontsize=8.5, color="#555555", style="italic")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(groups, fontsize=10, fontweight="bold", rotation=15)
        ax.set_ylabel(r"H1 Prominence Ratio ($L_1 / L_2$)", fontsize=11, fontweight="bold")
        ax.set_title(p["title"], fontsize=12, fontweight="bold", pad=10)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.set_ylim(bottom=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.3)
        ax.spines["bottom"].set_linewidth(1.3)
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle("Persistent Homology: H1 Loop Prominence Across Spaces & Functional Clusters",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig.savefig(save_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(save_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Dual Space Boxplot -> {save_base.name}")


def plot_animal_paired_trajectories(df: pd.DataFrame, save_base: Path):
    """Plots paired profiles across functional clusters for individual animals."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=300, sharey=False)
    func_groups = [g for g in FUNCTIONAL_CLUSTERS if g in df["group"].unique()]
    x_func = np.arange(len(func_groups))

    panels = [
        {"ax": axes[0], "col": "orig_ratio_L1_L2", "title": "A. Original Neural Space Trajectories"},
        {"ax": axes[1], "col": "umap_ratio_L1_L2", "title": "B. 3D UMAP Space Trajectories"},
    ]

    for p in panels:
        ax = p["ax"]
        col = p["col"]
        piv = df.pivot(index="mouse", columns="group", values=col)[func_groups]

        for mouse_id, row in piv.iterrows():
            ax.plot(x_func, row.values, marker="o", markersize=6, lw=1.6, alpha=0.8, label=str(mouse_id))

        ax.axhline(1.0, linestyle="--", color="black", linewidth=1.2, alpha=0.7)
        ax.set_xticks(x_func)
        ax.set_xticklabels(func_groups, fontsize=10, fontweight="bold", rotation=15)
        ax.set_ylabel(r"H1 Prominence Ratio ($L_1 / L_2$)", fontsize=11, fontweight="bold")
        ax.set_title(p["title"], fontsize=12, fontweight="bold", pad=10)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.legend(title="Animal", fontsize=8.5, frameon=True, loc="upper right")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.3)
        ax.spines["bottom"].set_linewidth(1.3)
        ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()
    fig.savefig(save_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(save_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Animal Trajectory Plot -> {save_base.name}")


# =============================================================================
# 4. MAIN STATISTICAL SUMMARY GENERATOR
# =============================================================================

def run_summary_analysis():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing summary CSV:\n{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Loaded {len(df)} records from {CSV_PATH.name}")

    # 1. Descriptive Tables
    desc_orig = compute_descriptives(df, "orig_ratio_L1_L2")
    desc_orig.to_csv(STATS_DIR / "descriptives_prominence_original_space.csv", index=False)

    desc_umap = compute_descriptives(df, "umap_ratio_L1_L2")
    desc_umap.to_csv(STATS_DIR / "descriptives_prominence_umap_space.csv", index=False)

    # 2. One-Sample Tests vs Unity (Noise floor = 1.0)
    null_orig = run_null_tests_vs_unity(df, "orig_ratio_L1_L2")
    null_orig.to_csv(STATS_DIR / "tests_vs_unity_original_space.csv", index=False)

    null_umap = run_null_tests_vs_unity(df, "umap_ratio_L1_L2")
    null_umap.to_csv(STATS_DIR / "tests_vs_unity_umap_space.csv", index=False)

    # 3. Pairwise Contrasts across clusters
    pairs_orig = run_pairwise_comparisons(df, "orig_ratio_L1_L2")
    pairs_orig.to_csv(STATS_DIR / "pairwise_contrasts_original_space.csv", index=False)

    pairs_umap = run_pairwise_comparisons(df, "umap_ratio_L1_L2")
    pairs_umap.to_csv(STATS_DIR / "pairwise_contrasts_umap_space.csv", index=False)

    # 4. Generate Figures
    plot_dual_space_summary_boxplots(df, PLOTS_DIR / "tda_prominence_ratio_dual_space_summary")
    plot_animal_paired_trajectories(df, PLOTS_DIR / "tda_animal_paired_trajectories")

    # 5. Export Comprehensive Text Report
    report_file = STATS_DIR / "tda_comprehensive_statistical_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TDA H1 PROMINENCE RATIO: STATISTICAL SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. DESCRIPTIVE STATISTICS: ORIGINAL NEURAL SPACE\n")
        f.write(desc_orig.to_string(index=False) + "\n\n")

        f.write("2. DESCRIPTIVE STATISTICS: 3D UMAP SPACE\n")
        f.write(desc_umap.to_string(index=False) + "\n\n")

        f.write("3. TESTS AGAINST NOISE FLOOR (H0: Prominence Ratio <= 1.0)\n")
        f.write("--- Original Space ---\n")
        f.write(null_orig.to_string(index=False) + "\n\n")
        f.write("--- UMAP Space ---\n")
        f.write(null_umap.to_string(index=False) + "\n\n")

        f.write("4. PAIRWISE CONTRASTS ACROSS CLUSTERS (Benjamini-Hochberg FDR)\n")
        f.write("--- Original Space ---\n")
        f.write(pairs_orig.to_string(index=False) + "\n\n")
        f.write("--- UMAP Space ---\n")
        f.write(pairs_umap.to_string(index=False) + "\n")

    print(f"\n[OK] Full summary report exported to: {report_file}")

    # Print summary tables to console
    print("\n" + "=" * 90)
    print("ORIGINAL NEURAL SPACE - DESCRIPTIVES & TEST VS. 1.0")
    print("=" * 90)
    summary_orig_view = pd.merge(
        desc_orig[["Group", "N", "Mean", "SEM", "Median", "IQR"]],
        null_orig[["Group", "t_stat", "p_val_ttest", "star"]],
        on="Group"
    )
    print(summary_orig_view.to_string(index=False))

    print("\n" + "=" * 90)
    print("3D UMAP SPACE - DESCRIPTIVES & TEST VS. 1.0")
    print("=" * 90)
    summary_umap_view = pd.merge(
        desc_umap[["Group", "N", "Mean", "SEM", "Median", "IQR"]],
        null_umap[["Group", "t_stat", "p_val_ttest", "star"]],
        on="Group"
    )
    print(summary_umap_view.to_string(index=False))


if __name__ == "__main__":
    run_summary_analysis()