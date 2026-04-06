#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting + statistics for Structure Index results across datasets.

This script reads:
1) the per-session Structure Index CSV
2) the original analysis index CSV

and generates:

A) learning dataset
   - Enumerates sessions 1..4 per animal by session datetime
   - Plots mean SI over animals vs session number for:
       * position
       * position_direction
       * direction
       * time
   - Main version:
       * scatter + lines
       * gray std band
       * stats text on the side
   - Clean version:
       * same plot but without stats text
       * mouse names visible near scatter points
   - One-way ANOVA saved to txt

B) lt_rot dataset
   - LT only: deep vs superficial boxplots
   - ROT only: deep vs superficial boxplots
   - LT vs ROT within deep
   - LT vs ROT within superficial
   - Combined LT/ROT x depth plot with 2-way ANOVA
   - Main version: stats text on the side
   - Clean version: no stats text, mouse names next to points
   - Stats saved to txt

   LT/ROT condition is inferred ONLY from session_name:
       if "lt" in session_name -> lt
       if "rot" in session_name -> rot

C) ephys dataset
   - Compare whole population, deep, superficial
   - All pairwise comparisons
   - Significance bars with stars:
       * whole vs deep -> two-sided Welch t-test
       * whole vs superficial -> two-sided Welch t-test
       * deep vs superficial -> paired t-test
   - Main version: stats text on the side
   - Clean version: no stats text, mouse names next to points
   - Stats saved to txt

D) egrin dataset
   - Compare whole population, deep, superficial
   - All pairwise comparisons
   - Main + clean versions
   - Stats saved to txt

Outputs:
- PNG + SVG in:
    OUTPUT_DIR/png
    OUTPUT_DIR/svg
    OUTPUT_DIR/png_clean
    OUTPUT_DIR/svg_clean
- TXT stats in:
    OUTPUT_DIR/stats_txt
"""

import os
import re
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


# =============================================================================
# Paths
# =============================================================================

SI_SESSION_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/structure_index_umap/structure_index_umap_per_session.csv"
ANALYSIS_INDEX_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/umap_pca_per_row/umap_pca_analysis_index.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/structure_index_umap_plots"

PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")
PNG_CLEAN_DIR = os.path.join(OUTPUT_DIR, "png_clean")
SVG_CLEAN_DIR = os.path.join(OUTPUT_DIR, "svg_clean")
TXT_DIR = os.path.join(OUTPUT_DIR, "stats_txt")


# =============================================================================
# Style / colors
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

MEASURE_LABELS = {
    "si_position": "Position",
    "si_position_direction": "Position + direction",
    "si_direction": "Direction",
    "si_time": "Time",
}

LEARNING_MEASURES = [
    "si_position",
    "si_position_direction",
    "si_direction",
    "si_time",
]

LTROT_MEASURES = [
    "si_position",
    "si_position_direction",
    "si_direction",
    "si_time",
]

EPHYS_EGRIN_MEASURES = [
    "si_position",
    "si_position_direction",
    "si_direction",
    "si_time",
]

LEARNING_DATASET_NAME = "learning"
LTROT_DATASET_NAME = "lt_rot"
EPHYS_DATASET_NAME = "lt_ephys_mice"
EGRIN_DATASET_NAME = "lt_egrin"
DREADDS_DATASET_NAME = "dreadds"

RNG = np.random.default_rng(42)


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


def save_figure(fig, basename, clean=False):
    if clean:
        png_path = os.path.join(PNG_CLEAN_DIR, basename + ".png")
        svg_path = os.path.join(SVG_CLEAN_DIR, basename + ".svg")
    else:
        png_path = os.path.join(PNG_DIR, basename + ".png")
        svg_path = os.path.join(SVG_DIR, basename + ".svg")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")


def save_text(text, basename):
    txt_path = os.path.join(TXT_DIR, basename + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)


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


def get_depth_color(depth):
    return DEPTH_COLOR_MAP.get(depth, "gray")


def parse_session_datetime(session_name):
    """
    Expects strings containing:
        YYYYMMDD_HHMMSS
    """
    if pd.isna(session_name):
        return pd.NaT

    s = str(session_name)
    m = re.search(r"(\d{8})_(\d{6})", s)
    if m is None:
        return pd.NaT

    try:
        return pd.to_datetime(m.group(1) + "_" + m.group(2), format="%Y%m%d_%H%M%S")
    except Exception:
        return pd.NaT


def infer_ltrot_condition_from_session_name(row):
    """
    Infer LT vs ROT only from session_name.
    """
    session_name = str(row.get("session_name", "")).strip().lower()

    if "lt" in session_name:
        return "lt"
    elif "rot" in session_name:
        return "rot"
    else:
        return None


def format_float(x, nd=6):
    if x is None:
        return "nan"
    try:
        if np.isnan(x):
            return "nan"
    except Exception:
        pass
    return f"{x:.{nd}f}"


def safe_unique_nonan(series):
    return [x for x in pd.unique(series) if pd.notna(x)]


def cohens_d_independent(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    na = len(a)
    nb = len(b)
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled


def pvalue_to_stars(p):
    if p is None or np.isnan(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def welch_ttest_pvalue(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    res = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided", nan_policy="omit")
    return res.pvalue


def paired_ttest_pvalue(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    res = stats.ttest_rel(a, b, alternative="two-sided", nan_policy="omit")
    return res.pvalue


def add_significance_bar(ax, x1, x2, y, h, text, fontsize=11):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.4, c="black")
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", fontsize=fontsize)


def run_pairwise_tests(group_a, group_b, label_a="A", label_b="B"):
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    lines = []
    lines.append(f"Comparison: {label_a} vs {label_b}")
    lines.append(f"n_{label_a} = {len(a)}")
    lines.append(f"n_{label_b} = {len(b)}")
    lines.append(f"mean_{label_a} = {format_float(np.mean(a) if len(a) else np.nan)}")
    lines.append(f"std_{label_a}  = {format_float(np.std(a, ddof=1) if len(a) > 1 else np.nan)}")
    lines.append(f"mean_{label_b} = {format_float(np.mean(b) if len(b) else np.nan)}")
    lines.append(f"std_{label_b}  = {format_float(np.std(b, ddof=1) if len(b) > 1 else np.nan)}")
    lines.append(f"cohens_d ({label_a}-{label_b}) = {format_float(cohens_d_independent(a, b))}")

    if len(a) >= 2 and len(b) >= 2:
        t_two = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided", nan_policy="omit")
        t_greater = stats.ttest_ind(a, b, equal_var=False, alternative="greater", nan_policy="omit")
        t_less = stats.ttest_ind(a, b, equal_var=False, alternative="less", nan_policy="omit")

        lines.append("")
        lines.append("Welch t-tests")
        lines.append(f"two-sided: statistic={format_float(t_two.statistic)}, p={format_float(t_two.pvalue)}")
        lines.append(f"{label_a} > {label_b}: statistic={format_float(t_greater.statistic)}, p={format_float(t_greater.pvalue)}")
        lines.append(f"{label_a} < {label_b}: statistic={format_float(t_less.statistic)}, p={format_float(t_less.pvalue)}")
    else:
        lines.append("")
        lines.append("Welch t-tests")
        lines.append("Not enough samples.")

    if len(a) >= 1 and len(b) >= 1:
        try:
            mw_two = stats.mannwhitneyu(a, b, alternative="two-sided")
            mw_greater = stats.mannwhitneyu(a, b, alternative="greater")
            mw_less = stats.mannwhitneyu(a, b, alternative="less")

            lines.append("")
            lines.append("Mann-Whitney U tests")
            lines.append(f"two-sided: statistic={format_float(mw_two.statistic)}, p={format_float(mw_two.pvalue)}")
            lines.append(f"{label_a} > {label_b}: statistic={format_float(mw_greater.statistic)}, p={format_float(mw_greater.pvalue)}")
            lines.append(f"{label_a} < {label_b}: statistic={format_float(mw_less.statistic)}, p={format_float(mw_less.pvalue)}")
        except Exception as e:
            lines.append("")
            lines.append("Mann-Whitney U tests")
            lines.append(f"Failed: {e}")
    else:
        lines.append("")
        lines.append("Mann-Whitney U tests")
        lines.append("Not enough samples.")

    return "\n".join(lines)


def run_paired_tests(group_a, group_b, label_a="A", label_b="B"):
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    lines = []
    lines.append(f"Paired comparison: {label_a} vs {label_b}")
    lines.append(f"n_pairs = {len(a)}")
    lines.append(f"mean_{label_a} = {format_float(np.mean(a) if len(a) else np.nan)}")
    lines.append(f"std_{label_a}  = {format_float(np.std(a, ddof=1) if len(a) > 1 else np.nan)}")
    lines.append(f"mean_{label_b} = {format_float(np.mean(b) if len(b) else np.nan)}")
    lines.append(f"std_{label_b}  = {format_float(np.std(b, ddof=1) if len(b) > 1 else np.nan)}")
    lines.append(f"mean_diff ({label_a}-{label_b}) = {format_float(np.mean(a - b) if len(a) else np.nan)}")

    if len(a) >= 2:
        t_two = stats.ttest_rel(a, b, alternative="two-sided", nan_policy="omit")
        t_greater = stats.ttest_rel(a, b, alternative="greater", nan_policy="omit")
        t_less = stats.ttest_rel(a, b, alternative="less", nan_policy="omit")

        lines.append("")
        lines.append("Paired t-tests")
        lines.append(f"two-sided: statistic={format_float(t_two.statistic)}, p={format_float(t_two.pvalue)}")
        lines.append(f"{label_a} > {label_b}: statistic={format_float(t_greater.statistic)}, p={format_float(t_greater.pvalue)}")
        lines.append(f"{label_a} < {label_b}: statistic={format_float(t_less.statistic)}, p={format_float(t_less.pvalue)}")
    else:
        lines.append("")
        lines.append("Paired t-tests")
        lines.append("Not enough paired samples.")

    if len(a) >= 1:
        try:
            w_two = stats.wilcoxon(a, b, alternative="two-sided")
            w_greater = stats.wilcoxon(a, b, alternative="greater")
            w_less = stats.wilcoxon(a, b, alternative="less")

            lines.append("")
            lines.append("Wilcoxon signed-rank tests")
            lines.append(f"two-sided: statistic={format_float(w_two.statistic)}, p={format_float(w_two.pvalue)}")
            lines.append(f"{label_a} > {label_b}: statistic={format_float(w_greater.statistic)}, p={format_float(w_greater.pvalue)}")
            lines.append(f"{label_a} < {label_b}: statistic={format_float(w_less.statistic)}, p={format_float(w_less.pvalue)}")
        except Exception as e:
            lines.append("")
            lines.append("Wilcoxon signed-rank tests")
            lines.append(f"Failed: {e}")
    else:
        lines.append("")
        lines.append("Wilcoxon signed-rank tests")
        lines.append("Not enough paired samples.")

    return "\n".join(lines)


def run_one_way_anova(df, value_col, factor_col):
    out_lines = []
    sub = df[[value_col, factor_col]].dropna().copy()

    out_lines.append(f"One-way ANOVA for {value_col} ~ {factor_col}")
    out_lines.append(f"N total = {len(sub)}")

    groups = []
    levels = sorted(safe_unique_nonan(sub[factor_col]))
    for lv in levels:
        x = sub.loc[sub[factor_col] == lv, value_col].values
        x = x[np.isfinite(x)]
        groups.append(x)
        out_lines.append(
            f"Level {lv}: n={len(x)}, mean={format_float(np.mean(x) if len(x) else np.nan)}, "
            f"std={format_float(np.std(x, ddof=1) if len(x) > 1 else np.nan)}"
        )

    valid_groups = [g for g in groups if len(g) > 0]
    if len(valid_groups) >= 2 and all(len(g) >= 2 for g in valid_groups):
        f_stat, p_val = stats.f_oneway(*valid_groups)
        out_lines.append("")
        out_lines.append(f"scipy.stats.f_oneway: F={format_float(f_stat)}, p={format_float(p_val)}")
    else:
        out_lines.append("")
        out_lines.append("scipy.stats.f_oneway: not enough data.")

    try:
        model = smf.ols(f"{value_col} ~ C({factor_col})", data=sub).fit()
        anova_tbl = anova_lm(model, typ=2)
        out_lines.append("")
        out_lines.append("statsmodels ANOVA table:")
        out_lines.append(anova_tbl.to_string())
    except Exception as e:
        out_lines.append("")
        out_lines.append(f"statsmodels ANOVA failed: {e}")

    return "\n".join(out_lines)


def run_two_way_anova(df, value_col, factor_a, factor_b):
    out_lines = []
    sub = df[[value_col, factor_a, factor_b]].dropna().copy()
    out_lines.append(f"Two-way ANOVA for {value_col} ~ C({factor_a}) * C({factor_b})")
    out_lines.append(f"N total = {len(sub)}")

    try:
        model = smf.ols(
            f"{value_col} ~ C({factor_a}) + C({factor_b}) + C({factor_a}):C({factor_b})",
            data=sub
        ).fit()
        anova_tbl = anova_lm(model, typ=2)
        out_lines.append("")
        out_lines.append(anova_tbl.to_string())
    except Exception as e:
        out_lines.append("")
        out_lines.append(f"2-way ANOVA failed: {e}")

    return "\n".join(out_lines)


def build_pairwise_summary_for_groups(data_dict):
    labels = list(data_dict.keys())
    chunks = []
    for a, b in itertools.combinations(labels, 2):
        chunks.append(run_pairwise_tests(data_dict[a], data_dict[b], label_a=a, label_b=b))
        chunks.append("\n" + "-" * 80 + "\n")
    return "\n".join(chunks)


def get_paired_deep_superficial_values(df, measure):
    """
    Pair deep and superficial values by mouse_name + session_name.
    """
    sub = df[df["depth_type"].isin(["deep", "superficial"])].copy()
    sub = sub[["mouse_name", "session_name", "depth_type", measure]].dropna().copy()

    pivot = sub.pivot_table(
        index=["mouse_name", "session_name"],
        columns="depth_type",
        values=measure,
        aggfunc="mean",
    ).reset_index()

    pivot = pivot.dropna(subset=["deep", "superficial"]).copy()

    deep_vals = pivot["deep"].values
    superficial_vals = pivot["superficial"].values

    return deep_vals, superficial_vals, pivot


def add_boxplot_with_points(
    ax,
    data_dict,
    color_dict=None,
    title=None,
    ylabel="Structure Index",
    stat_text=None,
    point_label_dict=None,
):
    labels = list(data_dict.keys())
    values = [np.asarray(data_dict[k]["values"], dtype=float) if isinstance(data_dict[k], dict) else np.asarray(data_dict[k], dtype=float)
              for k in labels]
    positions = np.arange(1, len(labels) + 1)

    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        boxprops=dict(color="black"),
    )

    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(color_dict.get(label, "lightgray") if color_dict is not None else "lightgray")
        patch.set_alpha(0.85)

    for i, lab in enumerate(labels, start=1):
        if isinstance(data_dict[lab], dict):
            vals = np.asarray(data_dict[lab]["values"], dtype=float)
            names = data_dict[lab].get("names", None)
        else:
            vals = np.asarray(data_dict[lab], dtype=float)
            names = None

        mask = np.isfinite(vals)
        vals = vals[mask]
        if names is not None:
            names = np.asarray(names, dtype=object)[mask]

        if len(vals) == 0:
            continue

        jitter = RNG.normal(loc=0.0, scale=0.05, size=len(vals))
        xs = np.full(len(vals), i) + jitter

        ax.scatter(
            xs,
            vals,
            s=30,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            color=color_dict.get(lab, "gray") if color_dict is not None else "gray",
            zorder=3,
        )

        if point_label_dict is not None and point_label_dict.get(lab, False) and names is not None:
            y_range = max(np.nanmax(vals) - np.nanmin(vals), 1e-3)
            dy = 0.01 * y_range
            for x, y, n in zip(xs, vals, names):
                ax.text(x + 0.02, y + dy, str(n), fontsize=7, alpha=0.9)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if stat_text is not None:
        ax.text(
            1.02, 0.98, stat_text,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
        )


def make_learning_session_number(df):
    df = df.copy()
    df["session_dt"] = df["session_name"].apply(parse_session_datetime)

    out = []
    for mouse_name, sub in df.groupby("mouse_name", sort=False):
        sub = sub.sort_values(["session_dt", "session_name"]).copy()
        sub["session_number"] = np.arange(1, len(sub) + 1)
        out.append(sub)

    out = pd.concat(out, axis=0, ignore_index=True)
    return out


# =============================================================================
# Plotting functions
# =============================================================================

def plot_learning_curves(df_learning):
    for measure in LEARNING_MEASURES:
        sub = df_learning[["mouse_name", "session_name", "session_number", measure]].dropna().copy()

        per_mouse = (
            sub.groupby(["mouse_name", "session_number"], as_index=False)[measure]
            .mean()
            .copy()
        )

        curve = (
            per_mouse.groupby("session_number")[measure]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("session_number")
        )

        stats_text = run_one_way_anova(per_mouse, measure, "session_number")

        for clean in [False, True]:
            fig, ax = plt.subplots(figsize=(6.5, 4.8))

            for mouse_name, mouse_df in per_mouse.groupby("mouse_name"):
                mouse_df = mouse_df.sort_values("session_number")
                ax.plot(
                    mouse_df["session_number"],
                    mouse_df[measure],
                    "-o",
                    alpha=0.35,
                    color="lightgray",
                    linewidth=1.5,
                    markersize=5,
                )

                if clean:
                    for _, r in mouse_df.iterrows():
                        ax.text(
                            r["session_number"] + 0.03,
                            r[measure],
                            str(mouse_name),
                            fontsize=7,
                            alpha=0.9,
                        )

            x = curve["session_number"].values.astype(float)
            y = curve["mean"].values.astype(float)
            s = curve["std"].fillna(0.0).values.astype(float)

            ax.fill_between(x, y - s, y + s, color="gray", alpha=0.25, label="mean ± std")
            ax.plot(x, y, "-o", color="black", linewidth=2.2, markersize=7, label="mean")

            ax.set_xlabel("Session number")
            ax.set_ylabel("Structure Index")
            ax.set_title(f"learning | {MEASURE_LABELS[measure]}")
            ax.set_xticks(sorted(curve["session_number"].unique()))
            ax.legend(frameon=False)

            if not clean:
                ax.text(
                    1.02, 0.98,
                    "\n".join(stats_text.splitlines()[:8]),
                    transform=ax.transAxes,
                    va="top", ha="left",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                )

            fig.tight_layout()
            base = f"learning_curve_{sanitize_for_filename(measure)}"
            save_figure(fig, base, clean=clean)
            plt.close(fig)

        save_text(stats_text, f"learning_curve_{sanitize_for_filename(measure)}_stats")


def plot_two_group_comparison(
    df,
    group_col,
    group_a,
    group_b,
    measure,
    title,
    basename,
    color_map=None,
    label_points=False,
):
    sub = df[[group_col, measure, "mouse_name"]].dropna().copy()
    sub = sub[sub[group_col].isin([group_a, group_b])].copy()

    a_vals = sub.loc[sub[group_col] == group_a, measure].values
    b_vals = sub.loc[sub[group_col] == group_b, measure].values
    a_names = sub.loc[sub[group_col] == group_a, "mouse_name"].astype(str).values
    b_names = sub.loc[sub[group_col] == group_b, "mouse_name"].astype(str).values

    stats_text = run_pairwise_tests(a_vals, b_vals, label_a=group_a, label_b=group_b)

    for clean in [False, True]:
        fig, ax = plt.subplots(figsize=(5.8, 4.8))

        data_dict = {
            group_a: {"values": a_vals, "names": a_names},
            group_b: {"values": b_vals, "names": b_names},
        }

        colors = None
        if color_map is not None:
            colors = {
                group_a: color_map.get(group_a, "gray"),
                group_b: color_map.get(group_b, "gray"),
            }

        add_boxplot_with_points(
            ax=ax,
            data_dict=data_dict,
            color_dict=colors,
            title=title,
            ylabel="Structure Index",
            stat_text=None if clean else "\n".join(stats_text.splitlines()[:12]),
            point_label_dict={group_a: clean and label_points, group_b: clean and label_points},
        )

        fig.tight_layout()
        save_figure(fig, basename, clean=clean)
        plt.close(fig)

    save_text(stats_text, basename + "_stats")


def plot_ltrot_all(df_ltrot):
    # LT only / ROT only: deep vs superficial
    for condition in ["lt", "rot"]:
        sub_cond = df_ltrot[df_ltrot["condition"] == condition].copy()
        sub_cond = sub_cond[sub_cond["depth_type"].isin(["deep", "superficial"])].copy()

        for measure in LTROT_MEASURES:
            title = f"{LTROT_DATASET_NAME} | {condition.upper()} only | {MEASURE_LABELS[measure]}"
            basename = f"ltrot_{condition}_deep_vs_superficial_{sanitize_for_filename(measure)}"
            plot_two_group_comparison(
                df=sub_cond,
                group_col="depth_type",
                group_a="deep",
                group_b="superficial",
                measure=measure,
                title=title,
                basename=basename,
                color_map={
                    "deep": get_depth_color("deep"),
                    "superficial": get_depth_color("superficial"),
                },
                label_points=True,
            )

    # LT vs ROT within deep and within superficial
    for depth in ["deep", "superficial"]:
        sub_depth = df_ltrot[df_ltrot["depth_type"] == depth].copy()
        sub_depth = sub_depth[sub_depth["condition"].isin(["lt", "rot"])].copy()

        for measure in LTROT_MEASURES:
            title = f"{LTROT_DATASET_NAME} | {depth} | LT vs ROT | {MEASURE_LABELS[measure]}"
            basename = f"ltrot_{depth}_lt_vs_rot_{sanitize_for_filename(measure)}"
            plot_two_group_comparison(
                df=sub_depth,
                group_col="condition",
                group_a="lt",
                group_b="rot",
                measure=measure,
                title=title,
                basename=basename,
                color_map={"lt": get_depth_color(depth), "rot": get_depth_color(depth)},
                label_points=True,
            )

    # Combined LT/ROT x depth
    sub = df_ltrot[df_ltrot["depth_type"].isin(["deep", "superficial"])].copy()
    sub = sub[sub["condition"].isin(["lt", "rot"])].copy()

    for measure in LTROT_MEASURES:
        two_way_text = run_two_way_anova(sub, measure, "condition", "depth_type")

        data_dict = {
            "lt_deep": {
                "values": sub.loc[(sub["condition"] == "lt") & (sub["depth_type"] == "deep"), measure].dropna().values,
                "names": sub.loc[(sub["condition"] == "lt") & (sub["depth_type"] == "deep"), "mouse_name"].astype(str).values,
            },
            "lt_superficial": {
                "values": sub.loc[(sub["condition"] == "lt") & (sub["depth_type"] == "superficial"), measure].dropna().values,
                "names": sub.loc[(sub["condition"] == "lt") & (sub["depth_type"] == "superficial"), "mouse_name"].astype(str).values,
            },
            "rot_deep": {
                "values": sub.loc[(sub["condition"] == "rot") & (sub["depth_type"] == "deep"), measure].dropna().values,
                "names": sub.loc[(sub["condition"] == "rot") & (sub["depth_type"] == "deep"), "mouse_name"].astype(str).values,
            },
            "rot_superficial": {
                "values": sub.loc[(sub["condition"] == "rot") & (sub["depth_type"] == "superficial"), measure].dropna().values,
                "names": sub.loc[(sub["condition"] == "rot") & (sub["depth_type"] == "superficial"), "mouse_name"].astype(str).values,
            },
        }

        color_dict = {
            "lt_deep": get_depth_color("deep"),
            "lt_superficial": get_depth_color("superficial"),
            "rot_deep": get_depth_color("deep"),
            "rot_superficial": get_depth_color("superficial"),
        }

        for clean in [False, True]:
            fig, ax = plt.subplots(figsize=(7.4, 5.0))

            add_boxplot_with_points(
                ax=ax,
                data_dict=data_dict,
                color_dict=color_dict,
                title=f"{LTROT_DATASET_NAME} | LT/ROT × depth | {MEASURE_LABELS[measure]}",
                ylabel="Structure Index",
                stat_text=None if clean else "\n".join(two_way_text.splitlines()[:10]),
                point_label_dict={
                    "lt_deep": clean,
                    "lt_superficial": clean,
                    "rot_deep": clean,
                    "rot_superficial": clean,
                },
            )

            fig.tight_layout()
            basename = f"ltrot_combined_2way_{sanitize_for_filename(measure)}"
            save_figure(fig, basename, clean=clean)
            plt.close(fig)

        save_text(two_way_text, f"ltrot_combined_2way_{sanitize_for_filename(measure)}_stats")


def plot_three_depth_dataset(df_dataset, dataset_label):
    """
    ephys:
      - whole vs deep: unpaired Welch
      - whole vs superficial: unpaired Welch
      - deep vs superficial: paired t-test
      - add significance bars with stars

    egrin:
      - all pairwise unpaired tests
      - no significance bars by default
    """
    sub = df_dataset.copy()
    sub["depth_type"] = sub["depth_type"].apply(canonicalize_depth)
    sub = sub[sub["depth_type"].isin(["whole population", "deep", "superficial"])].copy()

    for measure in EPHYS_EGRIN_MEASURES:
        whole_vals = sub.loc[sub["depth_type"] == "whole population", measure].dropna().values
        whole_names = sub.loc[sub["depth_type"] == "whole population", "mouse_name"].astype(str).values

        deep_vals = sub.loc[sub["depth_type"] == "deep", measure].dropna().values
        deep_names = sub.loc[sub["depth_type"] == "deep", "mouse_name"].astype(str).values

        sup_vals = sub.loc[sub["depth_type"] == "superficial", measure].dropna().values
        sup_names = sub.loc[sub["depth_type"] == "superficial", "mouse_name"].astype(str).values

        data_dict = {
            "whole population": {"values": whole_vals, "names": whole_names},
            "deep": {"values": deep_vals, "names": deep_names},
            "superficial": {"values": sup_vals, "names": sup_names},
        }

        stat_blocks = []

        stat_blocks.append(
            run_pairwise_tests(
                whole_vals,
                deep_vals,
                label_a="whole population",
                label_b="deep",
            )
        )
        stat_blocks.append("\n" + "-" * 80 + "\n")

        stat_blocks.append(
            run_pairwise_tests(
                whole_vals,
                sup_vals,
                label_a="whole population",
                label_b="superficial",
            )
        )
        stat_blocks.append("\n" + "-" * 80 + "\n")

        if dataset_label == EPHYS_DATASET_NAME:
            deep_paired, sup_paired, _ = get_paired_deep_superficial_values(sub, measure)
            stat_blocks.append(
                run_paired_tests(
                    deep_paired,
                    sup_paired,
                    label_a="deep",
                    label_b="superficial",
                )
            )
        else:
            stat_blocks.append(
                run_pairwise_tests(
                    deep_vals,
                    sup_vals,
                    label_a="deep",
                    label_b="superficial",
                )
            )

        stat_text = "\n".join(stat_blocks)

        for clean in [False, True]:
            fig, ax = plt.subplots(figsize=(6.9, 5.3))

            add_boxplot_with_points(
                ax=ax,
                data_dict=data_dict,
                color_dict={
                    "whole population": get_depth_color("whole population"),
                    "deep": get_depth_color("deep"),
                    "superficial": get_depth_color("superficial"),
                },
                title=f"{dataset_label} | depth comparison | {MEASURE_LABELS[measure]}",
                ylabel="Structure Index",
                stat_text=None if clean else "\n".join(stat_text.splitlines()[:16]),
                point_label_dict={
                    "whole population": clean,
                    "deep": clean,
                    "superficial": clean,
                },
            )

            if dataset_label == EPHYS_DATASET_NAME:
                xpos = {
                    "whole population": 1,
                    "deep": 2,
                    "superficial": 3,
                }

                all_vals = np.concatenate([
                    np.asarray(whole_vals, dtype=float),
                    np.asarray(deep_vals, dtype=float),
                    np.asarray(sup_vals, dtype=float),
                ])
                all_vals = all_vals[np.isfinite(all_vals)]

                if len(all_vals) > 0:
                    y_min = np.min(all_vals)
                    y_max = np.max(all_vals)
                else:
                    y_min, y_max = 0.0, 1.0

                y_range = max(y_max - y_min, 1e-3)
                base_y = y_max + 0.05 * y_range
                step = 0.10 * y_range
                h = 0.03 * y_range

                p_whole_deep = welch_ttest_pvalue(whole_vals, deep_vals)
                add_significance_bar(
                    ax=ax,
                    x1=xpos["whole population"],
                    x2=xpos["deep"],
                    y=base_y,
                    h=h,
                    text=pvalue_to_stars(p_whole_deep),
                    fontsize=11,
                )

                p_whole_sup = welch_ttest_pvalue(whole_vals, sup_vals)
                add_significance_bar(
                    ax=ax,
                    x1=xpos["whole population"],
                    x2=xpos["superficial"],
                    y=base_y + step,
                    h=h,
                    text=pvalue_to_stars(p_whole_sup),
                    fontsize=11,
                )

                deep_paired, sup_paired, _ = get_paired_deep_superficial_values(sub, measure)
                p_deep_sup_paired = paired_ttest_pvalue(deep_paired, sup_paired)
                add_significance_bar(
                    ax=ax,
                    x1=xpos["deep"],
                    x2=xpos["superficial"],
                    y=base_y + 2 * step,
                    h=h,
                    text=pvalue_to_stars(p_deep_sup_paired),
                    fontsize=11,
                )

                ax.set_ylim(top=base_y + 3 * step + 2 * h)

            fig.tight_layout()
            basename = f"{sanitize_for_filename(dataset_label)}_depth_comparison_{sanitize_for_filename(measure)}"
            save_figure(fig, basename, clean=clean)
            plt.close(fig)

        save_text(stat_text, f"{sanitize_for_filename(dataset_label)}_depth_comparison_{sanitize_for_filename(measure)}_stats")

def infer_dreadds_session_type(row):
    """
    Infer DREADDs session type from session_name.

    Expected labels:
    - lt_veh
    - lt_rot
    - lt_cno
    - rot_cno

    Also accepts rot_veh if present in the data, but the requested
    combined comparison uses:
        lt_veh + lt_rot + lt_cno   vs   rot_cno
    """
    session_name = str(row.get("session_name", "")).strip().lower()

    if "lt_veh" in session_name:
        return "lt_veh"
    elif "lt_rot" in session_name:
        return "lt_rot"
    elif "lt_cno" in session_name:
        return "lt_cno"
    elif "rot_cno" in session_name:
        return "rot_cno"
    elif "rot_veh" in session_name:
        return "rot_veh"
    else:
        return None


def plot_dreadds_deep_vs_superficial_per_session(df_dreadds):
    """
    For each DREADDs session type:
    make deep vs superficial boxplots for all requested SI measures.
    Exports normal + clean versions and saves stats txt.
    """
    session_order = ["lt_veh", "lt_rot", "lt_cno", "rot_cno", "rot_veh"]

    for sess_type in session_order:
        sub_sess = df_dreadds[df_dreadds["dreadds_session_type"] == sess_type].copy()
        if len(sub_sess) == 0:
            continue

        sub_sess = sub_sess[sub_sess["depth_type"].isin(["deep", "superficial"])].copy()
        if len(sub_sess) == 0:
            continue

        for measure in EPHYS_EGRIN_MEASURES:
            title = f"{DREADDS_DATASET_NAME} | {sess_type} | deep vs superficial | {MEASURE_LABELS[measure]}"
            basename = f"dreadds_{sess_type}_deep_vs_superficial_{sanitize_for_filename(measure)}"

            plot_two_group_comparison(
                df=sub_sess,
                group_col="depth_type",
                group_a="deep",
                group_b="superficial",
                measure=measure,
                title=title,
                basename=basename,
                color_map={
                    "deep": get_depth_color("deep"),
                    "superficial": get_depth_color("superficial"),
                },
                label_points=True,
            )


def plot_dreadds_ltfamily_vs_rotcno(df_dreadds):
    """
    Combine:
        lt_veh + lt_rot + lt_cno
    and compare against:
        rot_cno

    Do this for:
        - all (deep + superficial together)
        - deep only
        - superficial only
    """
    lt_family = ["lt_veh", "lt_rot", "lt_cno"]
    reference = "rot_cno"

    df_use = df_dreadds.copy()
    df_use = df_use[df_use["dreadds_session_type"].isin(lt_family + [reference])].copy()

    # -------------------------------------------------------------------------
    # all
    # -------------------------------------------------------------------------
    df_all = df_use.copy()
    df_all["comparison_group"] = np.where(
        df_all["dreadds_session_type"].isin(lt_family),
        "lt_family",
        "rot_cno",
    )

    for measure in EPHYS_EGRIN_MEASURES:
        title = f"{DREADDS_DATASET_NAME} | lt_family vs rot_cno | all | {MEASURE_LABELS[measure]}"
        basename = f"dreadds_ltfamily_vs_rotcno_all_{sanitize_for_filename(measure)}"

        plot_two_group_comparison(
            df=df_all,
            group_col="comparison_group",
            group_a="lt_family",
            group_b="rot_cno",
            measure=measure,
            title=title,
            basename=basename,
            color_map={
                "lt_family": "gray",
                "rot_cno": "black",
            },
            label_points=True,
        )

    # -------------------------------------------------------------------------
    # deep / superficial
    # -------------------------------------------------------------------------
    for depth in ["deep", "superficial"]:
        df_depth = df_use[df_use["depth_type"] == depth].copy()
        if len(df_depth) == 0:
            continue

        df_depth["comparison_group"] = np.where(
            df_depth["dreadds_session_type"].isin(lt_family),
            "lt_family",
            "rot_cno",
        )

        for measure in EPHYS_EGRIN_MEASURES:
            title = f"{DREADDS_DATASET_NAME} | lt_family vs rot_cno | {depth} | {MEASURE_LABELS[measure]}"
            basename = f"dreadds_ltfamily_vs_rotcno_{depth}_{sanitize_for_filename(measure)}"

            plot_two_group_comparison(
                df=df_depth,
                group_col="comparison_group",
                group_a="lt_family",
                group_b="rot_cno",
                measure=measure,
                title=title,
                basename=basename,
                color_map={
                    "lt_family": get_depth_color(depth),
                    "rot_cno": get_depth_color(depth),
                },
                label_points=True,
            )

# =============================================================================
# Loading / merging
# =============================================================================

def load_and_merge_tables():
    if not os.path.isfile(SI_SESSION_CSV):
        raise FileNotFoundError(f"Missing SI CSV: {SI_SESSION_CSV}")
    if not os.path.isfile(ANALYSIS_INDEX_CSV):
        raise FileNotFoundError(f"Missing analysis index CSV: {ANALYSIS_INDEX_CSV}")

    si_df = pd.read_csv(SI_SESSION_CSV)
    analysis_df = pd.read_csv(ANALYSIS_INDEX_CSV)

    if "analysis_file_path" in si_df.columns and "analysis_file_path" in analysis_df.columns:
        merged = si_df.merge(
            analysis_df,
            on="analysis_file_path",
            how="left",
            suffixes=("", "_analysis"),
        )
    else:
        if "row_index" in si_df.columns:
            analysis_df = analysis_df.reset_index().rename(columns={"index": "row_index"})
            merged = si_df.merge(
                analysis_df,
                on="row_index",
                how="left",
                suffixes=("", "_analysis"),
            )
        else:
            merged = si_df.copy()

    for col in ["dataset_name", "mouse_name", "session_name", "depth_type"]:
        alt = f"{col}_analysis"
        if alt in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged[alt])
            else:
                merged[col] = merged[alt]

    merged["depth_type"] = merged["depth_type"].apply(canonicalize_depth)
    return merged


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)
    ensure_dir(PNG_CLEAN_DIR)
    ensure_dir(SVG_CLEAN_DIR)
    ensure_dir(TXT_DIR)

    df = load_and_merge_tables()

    print("==============================================================")
    print("Structure Index plotting/statistics")
    print("==============================================================")
    print(f"Loaded rows: {len(df)}")
    print("Available dataset_name values:")
    print(sorted(safe_unique_nonan(df["dataset_name"])))
    print("")

    # -------------------------------------------------------------------------
    # learning
    # -------------------------------------------------------------------------
    df_learning = df[df["dataset_name"] == LEARNING_DATASET_NAME].copy()
    if len(df_learning) > 0:
        df_learning = make_learning_session_number(df_learning)
        df_learning = df_learning[df_learning["session_number"].isin([1, 2, 3, 4])].copy()

        print(f"[learning] rows = {len(df_learning)}")
        print(f"[learning] mice = {sorted(safe_unique_nonan(df_learning['mouse_name']))}")
        plot_learning_curves(df_learning)
    else:
        print(f"[learning] No rows found for dataset_name == '{LEARNING_DATASET_NAME}'")

    # -------------------------------------------------------------------------
    # lt_rot
    # -------------------------------------------------------------------------
    df_ltrot = df[df["dataset_name"] == LTROT_DATASET_NAME].copy()
    if len(df_ltrot) > 0:
        df_ltrot["condition"] = df_ltrot.apply(infer_ltrot_condition_from_session_name, axis=1)
        print(f"[lt_rot] rows = {len(df_ltrot)}")
        print(f"[lt_rot] inferred conditions = {sorted(safe_unique_nonan(df_ltrot['condition']))}")
        print(f"[lt_rot] depth types = {sorted(safe_unique_nonan(df_ltrot['depth_type']))}")
        plot_ltrot_all(df_ltrot)
    else:
        print(f"[lt_rot] No rows found for dataset_name == '{LTROT_DATASET_NAME}'")

    # -------------------------------------------------------------------------
    # ephys
    # -------------------------------------------------------------------------
    df_ephys = df[df["dataset_name"] == EPHYS_DATASET_NAME].copy()
    if len(df_ephys) > 0:
        print(f"[ephys] rows = {len(df_ephys)}")
        print(f"[ephys] depth types = {sorted(safe_unique_nonan(df_ephys['depth_type']))}")
        plot_three_depth_dataset(df_ephys, EPHYS_DATASET_NAME)
    else:
        print(f"[ephys] No rows found for dataset_name == '{EPHYS_DATASET_NAME}'")

    # -------------------------------------------------------------------------
    # egrin
    # -------------------------------------------------------------------------
    df_egrin = df[df["dataset_name"] == EGRIN_DATASET_NAME].copy()
    if len(df_egrin) > 0:
        print(f"[egrin] rows = {len(df_egrin)}")
        print(f"[egrin] depth types = {sorted(safe_unique_nonan(df_egrin['depth_type']))}")
        plot_three_depth_dataset(df_egrin, EGRIN_DATASET_NAME)
    else:
        print(f"[egrin] No rows found for dataset_name == '{EGRIN_DATASET_NAME}'")

    # -------------------------------------------------------------------------
    # dreadds
    # -------------------------------------------------------------------------
    df_dreadds = df[df["dataset_name"] == DREADDS_DATASET_NAME].copy()
    if len(df_dreadds) > 0:
        df_dreadds["dreadds_session_type"] = df_dreadds.apply(infer_dreadds_session_type, axis=1)

        print(f"[dreadds] rows = {len(df_dreadds)}")
        print(f"[dreadds] session types = {sorted(safe_unique_nonan(df_dreadds['dreadds_session_type']))}")
        print(f"[dreadds] depth types = {sorted(safe_unique_nonan(df_dreadds['depth_type']))}")

        plot_dreadds_deep_vs_superficial_per_session(df_dreadds)
        plot_dreadds_ltfamily_vs_rotcno(df_dreadds)
    else:
        print(f"[dreadds] No rows found for dataset_name == '{DREADDS_DATASET_NAME}'")

    print("")
    print("Done.")
    print(f"Main figures:  {PNG_DIR} and {SVG_DIR}")
    print(f"Clean figures: {PNG_CLEAN_DIR} and {SVG_CLEAN_DIR}")
    print(f"Stats txt:     {TXT_DIR}")


if __name__ == "__main__":
    main()