#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lt_egrin Structure Index Advanced Plotting & Statistical Pipeline:
- Narrower bar plot with ANOVA stats annotation on top
- Scatter plots (Time vs Slow, Position vs MUA) with unshaded regression lines (PNG & SVG)
- Regression coefficients report (CSV)
- One-way ANOVA & Tukey HSD post-hoc for Complete Population (Gray, No Grid, PNG & SVG)
- Two-Way ANOVA & Tukey HSD post-hoc for Functional Clusters (No Grid, PNG & SVG)
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(".")
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "structure_index_complete_and_clusters_lt_egrin"
FIGURES_DIR = OUTPUT_DIR / "summary_plots"
STATS_DIR = OUTPUT_DIR / "stats_reports"
SUMMARY_CSV = OUTPUT_DIR / "structure_index_manifold_summary_lt_egrin.csv"
PER_SESSION_CSV = OUTPUT_DIR / "structure_index_manifold_per_session_lt_egrin.csv"

CLUSTER_PALETTE = {
    "complete_population": "#7f7f7f",  # Gray
    "Time Cluster": "#bce784ff",  # Green
    "Direction Cluster": "#66cef4ff",  # Blue
    "Position Cluster": "#ec8ef8ff",  # Purple/Magenta
}


def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_fig_both(fig, base_path):
    ensure_dir(Path(base_path).parent)
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base_path}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {base_path}.png and {base_path}.svg")


def main():
    ensure_dir(FIGURES_DIR)
    ensure_dir(STATS_DIR)

    if not SUMMARY_CSV.is_file() or not PER_SESSION_CSV.is_file():
        print(f"[ERROR] Required summary or per-session CSV files not found in {OUTPUT_DIR}.")
        return

    df_summary = pd.read_csv(SUMMARY_CSV)
    df_session = pd.read_csv(PER_SESSION_CSV)

    # 1. Two-Way ANOVA for Barplot Annotations
    anova_text = "Two-Way ANOVA stats unavailable"
    try:
        model_bar = ols('structure_index ~ C(neural_space) + C(feature_name) + C(neural_space):C(feature_name)',
                        data=df_summary).fit()
        anova_table_bar = sm.stats.anova_lm(model_bar, typ=2)
        space_f = anova_table_bar.loc['C(neural_space)', 'F']
        space_p = anova_table_bar.loc['C(neural_space)', 'PR(>F)']
        feat_f = anova_table_bar.loc['C(feature_name)', 'F']
        feat_p = anova_table_bar.loc['C(feature_name)', 'PR(>F)']
        anova_text = (
            f"Two-Way ANOVA Summary:\n"
            f"• Neural Space: F = {space_f:.2f}, p = {space_p:.2e}\n"
            f"• Feature: F = {feat_f:.2f}, p = {feat_p:.2e}"
        )
    except Exception as e:
        print(f"[WARN] ANOVA annotation failed: {e}")

    # Bar Plot (Narrower width)
    plt.figure(figsize=(11, 5.0))
    ax = sns.barplot(
        data=df_summary,
        x="feature_name",
        y="structure_index",
        hue="neural_space",
        palette=CLUSTER_PALETTE,
        errorbar="se",
        capsize=0.1,
        err_kws={"linewidth": 1.2}
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.ylabel("Structure Index (SI)", fontsize=11, fontweight="bold")
    plt.xlabel("Neural & Behavioral Features", fontsize=11, fontweight="bold")
    plt.title("Structure Index Comparison: Complete Population (Gray) vs. Functional Clusters", fontsize=12,
              fontweight="bold", pad=15)
    plt.ylim(0.0, 1.0)
    ax.text(
        0.02, 0.95, anova_text,
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    ax.grid(False)
    plt.legend(title="Neural Subspace", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    save_fig_both(plt.gcf(), FIGURES_DIR / "lt_egrin_structure_index_barplot_colored")

    # Scatter Helper (Unshaded Regressions, PNG & SVG)
    regression_stats_report = []

    def plot_scatter(sub_features, y_feat, title_prefix, filename_prefix):
        n_feats = len(sub_features)
        fig, axes = plt.subplots(1, n_feats, figsize=(4.2 * n_feats, 4.8), sharey=True)
        if n_feats == 1:
            axes = [axes]

        for ax, sfeat in zip(axes, sub_features):
            for cluster_name, color in CLUSTER_PALETTE.items():
                col_x = f"{cluster_name}__{sfeat}"
                col_y = f"{cluster_name}__{y_feat}"
                if col_x not in df_session.columns or col_y not in df_session.columns:
                    continue
                sub_df = df_session[[col_x, col_y]].dropna()
                if len(sub_df) < 3:
                    continue

                sns.scatterplot(data=sub_df, x=col_x, y=col_y, color=color, s=60, alpha=0.85, ax=ax, label=cluster_name)
                x_vals = sub_df[col_x].values
                y_vals = sub_df[col_y].values

                if len(np.unique(x_vals)) > 1 and len(np.unique(y_vals)) > 1:
                    sns.regplot(data=sub_df, x=col_x, y=col_y, scatter=False, color=color, ax=ax, ci=None,
                                line_kws={"linewidth": 1.8, "alpha": 0.9})
                    r_val, p_val = stats.pearsonr(x_vals, y_vals)
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                else:
                    r_val, p_val, slope, intercept = np.nan, np.nan, np.nan, np.nan

                regression_stats_report.append({
                    "analysis": filename_prefix,
                    "feature": sfeat,
                    "target": y_feat,
                    "neural_space": cluster_name,
                    "slope": slope,
                    "intercept": intercept,
                    "pearson_r": r_val,
                    "p_value": p_val,
                    "n_samples": len(sub_df)
                })

            clean_feat_name = sfeat.replace("_", " ").title()
            ax.set_title(f"{clean_feat_name}", fontsize=10, fontweight="bold")
            ax.set_xlabel(sfeat, fontsize=9)
            ax.set_ylim(0.0, 1.0)
            if ax == axes[0]:
                ax.set_ylabel(f"{y_feat.replace('_', ' ').title()} Structure Index", fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("")
            ax.grid(False)
            if ax == axes[-1]:
                ax.legend(title="Neural Subspace", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=8)

        plt.suptitle(f"{title_prefix}: Scatter & Unshaded Regressions", fontsize=12, fontweight="bold", y=1.03)
        plt.tight_layout()
        save_fig_both(plt.gcf(), FIGURES_DIR / filename_prefix)

    # 2. Scatter Time vs Slow
    plot_scatter(
        sub_features=['lfp_waveform_slow', 'lfp_envelope_slow', 'lfp_phase_slow', 'cwt_power_slow'],
        y_feat="time",
        title_prefix="Time Structure Index vs. Slow Band Features",
        filename_prefix="lt_egrin_time_vs_slow"
    )

    # 3. Scatter Position vs MUA
    plot_scatter(
        sub_features=['lfp_waveform_mua', 'lfp_envelope_mua', 'lfp_phase_mua', 'cwt_power_mua'],
        y_feat="position",
        title_prefix="Position Structure Index vs. MUA Band Features",
        filename_prefix="lt_egrin_position_vs_mua"
    )

    pd.DataFrame(regression_stats_report).to_csv(STATS_DIR / "regression_coefficients_report.csv", index=False)

    # 4. Complete Population Boxplot (Gray, No Grid, Post-hoc reported)
    target_lfp_features = ['complete_population__lfp_envelope_slow', 'complete_population__cwt_power_slow',
                           'complete_population__cwt_power_delta', 'complete_population__cwt_power_theta',
                           'complete_population__cwt_power_gamma', 'complete_population__cwt_power_ripple',
                           'complete_population__cwt_power_mua']
    melted_complete = pd.melt(df_session[[c for c in target_lfp_features if c in df_session.columns]],
                              var_name="feature_name", value_name="structure_index").dropna()

    if not melted_complete.empty:
        groups = [group["structure_index"].values for _, group in melted_complete.groupby("feature_name")]
        if len(groups) > 1:
            f_val, p_val = stats.f_oneway(*groups)
            print(f"\n[ONE-WAY ANOVA] Complete Population: F = {f_val:.4f}, p = {p_val:.4e}")
            tukey_complete = pairwise_tukeyhsd(endog=melted_complete["structure_index"],
                                               groups=melted_complete["feature_name"], alpha=0.05)
            print("\n--- Tukey HSD Post-hoc (Complete Population) ---")
            print(tukey_complete)
            with open(STATS_DIR / "tukey_complete_population.txt", "w") as f:
                f.write(str(tukey_complete))

        plt.figure(figsize=(10, 5))
        ax1 = sns.boxplot(data=melted_complete, x="feature_name", y="structure_index", color="#7f7f7f")
        ax1.grid(False)
        plt.xticks(rotation=45, ha="right")
        plt.title("Structure Index Distribution: Slow Envelope & CWT Bands (Complete Population - Gray)", fontsize=11,
                  fontweight="bold")
        plt.xlabel("LFP / CWT Feature", fontsize=10)
        plt.ylabel("Structure Index (SI)", fontsize=10)
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        save_fig_both(plt.gcf(), FIGURES_DIR / "complete_population_lfp_cwt_boxplot")

    # 5. Clusters Boxplot (Two-Way ANOVA & Post-hoc, No Grid, Significance Annotations)
    cluster_subspaces = ["Time Cluster", "Direction Cluster", "Position Cluster"]
    cluster_features_to_test = ['lfp_envelope_slow', 'cwt_power_slow', 'cwt_power_delta', 'cwt_power_theta',
                                'cwt_power_gamma', 'cwt_power_ripple', 'cwt_power_mua']
    cluster_data_rows = []
    for _, row in df_session.iterrows():
        for cspace in cluster_subspaces:
            for feat in cluster_features_to_test:
                col_name = f"{cspace}__{feat}"
                if col_name in df_session.columns and pd.notna(row[col_name]):
                    cluster_data_rows.append({"cluster": cspace, "feature": feat, "structure_index": row[col_name]})

    df_clusters_anova = pd.DataFrame(cluster_data_rows)
    if not df_clusters_anova.empty:
        model = ols('structure_index ~ C(cluster) + C(feature) + C(cluster):C(feature)', data=df_clusters_anova).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print("\n--- Two-Way ANOVA Table (Clusters vs Features) ---")
        print(anova_table)
        anova_table.to_csv(STATS_DIR / "twoway_anova_clusters_features.csv")

        tukey_cluster = pairwise_tukeyhsd(endog=df_clusters_anova["structure_index"],
                                          groups=df_clusters_anova["cluster"], alpha=0.05)
        tukey_feat = pairwise_tukeyhsd(endog=df_clusters_anova["structure_index"], groups=df_clusters_anova["feature"],
                                       alpha=0.05)
        print("\n--- Tukey HSD Post-hoc: Cluster Effect ---")
        print(tukey_cluster)
        print("\n--- Tukey HSD Post-hoc: Feature Effect ---")
        print(tukey_feat)

        with open(STATS_DIR / "tukey_clusters_posthoc.txt", "w") as f:
            f.write("=== CLUSTER EFFECT ===\n" + str(tukey_cluster) + "\n\n=== FEATURE EFFECT ===\n" + str(tukey_feat))

        plt.figure(figsize=(12, 6))
        ax2 = sns.boxplot(data=df_clusters_anova, x="feature", y="structure_index", hue="cluster",
                          palette=CLUSTER_PALETTE)
        ax2.grid(False)
        plt.xticks(rotation=45, ha="right")
        plt.title("Structure Index Across Clusters for Slow Envelope & CWT Bands", fontsize=11, fontweight="bold")
        plt.xlabel("LFP / CWT Feature", fontsize=10)
        plt.ylabel("Structure Index (SI)", fontsize=10)
        plt.ylim(0.0, 1.0)
        ax2.text(
            0.02, 0.95,
            f"Two-Way ANOVA Cluster p = {anova_table.loc['C(cluster)', 'PR(>F)']:.2e}\nFeature p = {anova_table.loc['C(feature)', 'PR(>F)']:.2e}",
            transform=ax2.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray')
        )
        plt.legend(title="Neural Subspace", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        save_fig_both(plt.gcf(), FIGURES_DIR / "clusters_lfp_cwt_boxplot")


if __name__ == "__main__":
    main()