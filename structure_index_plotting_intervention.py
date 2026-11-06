#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting & Statistics Suite for the Intervention Dataset (Standardization Project).

Reads:
  - structure_index_neural_space_per_session.csv
  - OR structure_index_dual_space_per_session.csv
  (auto-detects whichever file contains dataset_name == 'intervention')

Generates:
  - Multi-panel Basal vs. Sinusoidal comparisons grouped by (int_i, int_f)
  - Paired mouse trajectories + boxplots across:
      * Position (si_position)
      * Continuous Time (si_time)
      * Trial / Inner Time (si_inner_time)
      * Direction (si_direction)
      * Position + Direction (si_position_direction)
  - Two-tailed and one-tailed paired hypothesis tests + effect sizes (Cohen's dz)
  - Stacked significance brackets on top (NO horizontal gridlines or guidelines)

Saves into:
  - PROJECT_ROOT/analysis/structure_index_comparison_plots/intervention/
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# 1. PATHS & SETTINGS
# =============================================================================

POSSIBLE_ROOTS = [
    Path("/home/hippolab/Documents/melisa/time_project_standarization"),
    Path("/home/melisamc/Documents/time_project_standarization"),
]
PROJECT_ROOT = next((p for p in POSSIBLE_ROOTS if p.is_dir()), POSSIBLE_ROOTS[0])

# Primary input tables from the standardization project
SI_NEURAL_CSV = PROJECT_ROOT / "analysis" / "structure_index_neural_space" / "structure_index_neural_space_per_session.csv"
SI_DUAL_CSV = PROJECT_ROOT / "analysis" / "structure_index_dual_space" / "structure_index_dual_space_per_session.csv"

# Destination directories in the standardization project
OUTPUT_BASE = PROJECT_ROOT / "analysis" / "structure_index_comparison_plots" / "intervention"
PNG_DIR = OUTPUT_BASE / "png"
SVG_DIR = OUTPUT_BASE / "svg"
PNG_CLEAN_DIR = OUTPUT_BASE / "png_clean"
SVG_CLEAN_DIR = OUTPUT_BASE / "svg_clean"
STATS_DIR = OUTPUT_BASE / "stats_txt"

for p in [PNG_DIR, SVG_DIR, PNG_CLEAN_DIR, SVG_CLEAN_DIR, STATS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Variables to compare
MEASURES = [
    "si_position",
    "si_time",
    "si_inner_time",
    "si_direction",
    "si_position_direction",
]

MEASURE_LABELS = {
    "si_position": "Position",
    "si_time": "Continuous Time",
    "si_inner_time": "Trial / Inner Time",
    "si_trial_id": "Trial ID",
    "si_direction": "Direction",
    "si_position_direction": "Position + Direction",
}

BOX_COLORS = {
    "basal": "#1f77b4",       # Muted Blue
    "sinusoidal": "#ff7f0e",  # Safety Orange
}

DATE_PARAMS_FALLBACK = {
    "1707": (3.0, 0.17),
    "2309": (1.5, 0.17),
    "2509": (1.5, 0.17),
    "0110": (1.5, 1.0),
    "0910": (1.5, 1.0),
}


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def extract_base_session(session_name: str) -> str:
    s = str(session_name).strip()
    for suff in ["_basal", "_sinusoidal", "_combined"]:
        if s.endswith(suff):
            return s[:-len(suff)]
    return s


def parse_date_str(session_name: str) -> str:
    m = re.search(r"(\d{4})", str(session_name))
    return m.group(1) if m else ""


def pvalue_to_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return "n/a"
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else np.nan


def add_significance_bracket(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str):
    """Draws a clean bracket without horizontal guidelines."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black", clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h * 1.12, text, ha="center", va="bottom", fontsize=10, fontweight="bold")


def get_mouse_palette(mice: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    unique_mice = sorted(list(set(mice)))
    cmap = mpl.colormaps["tab10"]
    return {m: cmap(i % 10) for i, m in enumerate(unique_mice)}


def sanitize_for_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))


# =============================================================================
# 3. STATISTICAL REPORT ENGINE (PAIRED TESTS)
# =============================================================================

def run_paired_tests(basal_vals: np.ndarray, sinus_vals: np.ndarray, label: str) -> Tuple[str, float]:
    a = np.asarray(basal_vals, dtype=float)
    b = np.asarray(sinus_vals, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    diff = a - b

    n = len(a)
    mean_b, std_b = (float(np.mean(a)), float(np.std(a, ddof=1))) if n > 1 else (np.nan, np.nan)
    mean_s, std_s = (float(np.mean(b)), float(np.std(b, ddof=1))) if n > 1 else (np.nan, np.nan)
    dz = cohens_dz(a, b)

    lines = [
        f"VARIABLE: {label}",
        f"  Matched Sessions (N): {n}",
        f"  Basal:      Mean = {mean_b:.4f} | SD = {std_b:.4f}",
        f"  Sinusoidal: Mean = {mean_s:.4f} | SD = {std_s:.4f}",
        f"  Pairwise Diff (Basal - Sinus): Mean = {np.mean(diff):+.4f} | SD = {np.std(diff, ddof=1):.4f}",
        f"  Effect Size (Cohen's dz): {dz:.4f}",
    ]

    p_two = np.nan
    if n >= 2:
        t_two = stats.ttest_rel(a, b, alternative="two-sided")
        t_gt = stats.ttest_rel(a, b, alternative="greater")
        t_lt = stats.ttest_rel(a, b, alternative="less")
        p_two = float(t_two.pvalue)

        lines.extend([
            "  Paired t-test:",
            f"    Two-sided:                 t = {t_two.statistic:+.4f}, p = {t_two.pvalue:.5e} ({pvalue_to_stars(t_two.pvalue)})",
            f"    One-sided (Basal > Sinus): t = {t_gt.statistic:+.4f}, p = {t_gt.pvalue:.5e} ({pvalue_to_stars(t_gt.pvalue)})",
            f"    One-sided (Basal < Sinus): t = {t_lt.statistic:+.4f}, p = {t_lt.pvalue:.5e} ({pvalue_to_stars(t_lt.pvalue)})",
        ])

    if n >= 3:
        try:
            w_two = stats.wilcoxon(a, b, alternative="two-sided")
            w_gt = stats.wilcoxon(a, b, alternative="greater")
            w_lt = stats.wilcoxon(a, b, alternative="less")
            lines.extend([
                "  Wilcoxon Signed-Rank test:",
                f"    Two-sided:                 W = {w_two.statistic:.1f}, p = {w_two.pvalue:.5e} ({pvalue_to_stars(w_two.pvalue)})",
                f"    One-sided (Basal > Sinus): W = {w_gt.statistic:.1f}, p = {w_gt.pvalue:.5e} ({pvalue_to_stars(w_gt.pvalue)})",
                f"    One-sided (Basal < Sinus): W = {w_lt.statistic:.1f}, p = {w_lt.pvalue:.5e} ({pvalue_to_stars(w_lt.pvalue)})",
            ])
        except Exception as exc:
            lines.append(f"  Wilcoxon failed: {exc}")

    lines.append("")
    return "\n".join(lines), p_two


# =============================================================================
# 4. PLOTTING SUITE
# =============================================================================

def plot_intervention_group(df_group: pd.DataFrame, int_i: float, int_f: float, space_label: str):
    tag = f"int_i_{int_i}_int_f_{int_f}".replace(".", "p")
    group_title = f"Intervention: int_i = {int_i}, int_f = {int_f} ({space_label})"

    # Pair basal and sinusoidal by mouse and base_session
    piv = df_group.pivot_table(
        index=["mouse_name", "base_session"],
        columns="paradigm",
        values=MEASURES,
        aggfunc="first"
    )

    if "basal" not in piv.columns.levels[1] or "sinusoidal" not in piv.columns.levels[1]:
        print(f"[SKIP] Group int_i={int_i}, int_f={int_f} lacks both basal and sinusoidal recordings.")
        return

    n_sessions = len(piv)
    all_mice = [str(m) for m in piv.index.get_level_values("mouse_name").unique()]
    mouse_colors = get_mouse_palette(all_mice)

    valid_measures = [m for m in MEASURES if (m, "basal") in piv.columns and (m, "sinusoidal") in piv.columns]
    n_panels = len(valid_measures)
    if n_panels == 0:
        return

    stat_report_lines = [
        f"INTERVENTION STRUCTURE INDEX REPORT: {group_title}",
        "=" * 80,
        f"Data Space: {space_label}",
        f"Parameter Set: int_i = {int_i}, int_f = {int_f}",
        f"Paired Sessions: N = {n_sessions}\n",
        "=" * 80,
    ]

    for clean in [False, True]:
        fig, axes = plt.subplots(1, n_panels, figsize=(4.8 * n_panels, 5.2), dpi=300)
        if n_panels == 1:
            axes = [axes]

        for ax_idx, m_col in enumerate(valid_measures):
            ax = axes[ax_idx]
            sub_m = piv[m_col].dropna(subset=["basal", "sinusoidal"]).copy()

            b_vals = sub_m["basal"].values
            s_vals = sub_m["sinusoidal"].values
            mice_list = [str(m) for m in sub_m.index.get_level_values("mouse_name")]

            label_name = MEASURE_LABELS.get(m_col, m_col)
            stat_txt, p_val = run_paired_tests(b_vals, s_vals, label_name)
            if not clean:
                stat_report_lines.append(stat_txt)

            x_pos = [1.0, 2.0]
            labels = ["Basal", "Sinusoidal"]

            # Paired animal lines
            for i in range(len(b_vals)):
                m_color = mouse_colors.get(mice_list[i], "gray")
                ax.plot(x_pos, [b_vals[i], s_vals[i]], color=m_color, alpha=0.85, lw=1.6, zorder=3)
                ax.scatter(x_pos[0], b_vals[i], color=m_color, s=50, edgecolors="k", lw=0.6, zorder=4)
                ax.scatter(x_pos[1], s_vals[i], color=m_color, s=50, edgecolors="k", lw=0.6, zorder=4)

                if clean:
                    ax.text(x_pos[1] + 0.05, s_vals[i], str(mice_list[i]), fontsize=6.5, alpha=0.85, va="center")

            # Boxplots
            bp = ax.boxplot(
                [b_vals, s_vals],
                positions=x_pos,
                widths=0.45,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(color="black", linewidth=1.2),
                capprops=dict(color="black", linewidth=1.2),
                boxprops=dict(linewidth=1.2),
            )
            bp["boxes"][0].set_facecolor(BOX_COLORS["basal"])
            bp["boxes"][0].set_alpha(0.25)
            bp["boxes"][1].set_facecolor(BOX_COLORS["sinusoidal"])
            bp["boxes"][1].set_alpha(0.25)

            # Significance bracket on top
            all_pts = np.concatenate([b_vals, s_vals])
            y_max = np.max(all_pts) if len(all_pts) else 1.0
            y_min = np.min(all_pts) if len(all_pts) else 0.0
            y_range = max(y_max - y_min, 0.08)

            bracket_y = y_max + 0.08 * y_range
            star = pvalue_to_stars(p_val)
            add_significance_bracket(ax, x_pos[0], x_pos[1], bracket_y, 0.035 * y_range, star)

            ax.set_title(label_name, fontsize=11, fontweight="bold", pad=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
            ax.set_xlim(0.4, 2.6)
            ax.set_ylim(-0.02, bracket_y + 0.12 * y_range)

            # Strictly NO horizontal lines or grid lines
            ax.grid(False)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

            if ax_idx == 0:
                ax.set_ylabel(f"Structure Index ({space_label})", fontsize=10, fontweight="bold")

        # Animal legend
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markeredgecolor="k", markersize=8, label=f"Mouse {m}")
            for m, col in mouse_colors.items()
        ]
        axes[-1].legend(handles=handles, title="Mouse ID", loc="upper left", bbox_to_anchor=(1.04, 1.0), frameon=False, fontsize=9)

        fig.suptitle(f"{group_title} (N = {n_sessions} Sessions)", fontsize=13, fontweight="bold", y=0.99)
        fig.tight_layout(rect=[0.02, 0.02, 0.90, 0.95])

        out_name = f"intervention_{tag}_{sanitize_for_filename(space_label)}"
        target_dir = PNG_CLEAN_DIR if clean else PNG_DIR
        target_svg = SVG_CLEAN_DIR if clean else SVG_DIR

        fig.savefig(target_dir / f"{out_name}.png", bbox_inches="tight")
        fig.savefig(target_svg / f"{out_name}.svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED FIGURE] {target_dir.name}/{out_name}.png")

    stats_out = STATS_DIR / f"intervention_{tag}_{sanitize_for_filename(space_label)}_stats.txt"
    with open(stats_out, "w", encoding="utf-8") as f:
        f.write("\n".join(stat_report_lines))
    print(f"  [SAVED STATS]  {stats_out.name}")


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main():
    # Detect available CSV in standardization project by checking content
    candidate_files = [SI_NEURAL_CSV, SI_DUAL_CSV]

    csv_to_use = None
    df = None

    for cand in candidate_files:
        if cand.is_file():
            try:
                tmp_df = pd.read_csv(cand)
                if "dataset_name" in tmp_df.columns:
                    interv_rows = tmp_df[tmp_df["dataset_name"].astype(str).str.lower() == "intervention"]
                    if not interv_rows.empty:
                        csv_to_use = cand
                        df = interv_rows.copy()
                        break
            except Exception as e:
                print(f"[WARN] Failed reading {cand}: {e}")

    if df is None or df.empty:
        checked_str = "\n  ".join([str(c) for c in candidate_files])
        raise ValueError(
            f"No rows with dataset_name == 'intervention' found in candidate spreadsheets:\n  {checked_str}\n\n"
            "Please verify that structure_index_computation.py has finished running for the 'intervention' dataset."
        )

    print("=" * 80)
    print("INTERVENTION STRUCTURE INDEX PLOTTING & STATS (STANDARDIZATION PROJECT)")
    print("=" * 80)
    print(f"Selected Spreadsheet: {csv_to_use}")
    print(f"Intervention Records: {len(df)}")
    print(f"Output Directory:     {OUTPUT_BASE}\n")

    # Standardize metadata columns
    df["base_session"] = df["session_name"].apply(extract_base_session)
    df["date_str"] = df["session_name"].apply(parse_date_str)

    # Propagate (int_i, int_f) from sinusoidal to matching basal sessions
    sinus_subset = df[df["paradigm"] == "sinusoidal"]
    param_map = {}
    for _, r in sinus_subset.iterrows():
        key = (str(r["mouse_name"]), str(r["base_session"]))
        if pd.notna(r.get("int_i")) and pd.notna(r.get("int_f")):
            param_map[key] = (float(r["int_i"]), float(r["int_f"]))

    # Assign group parameters
    for idx, r in df.iterrows():
        key = (str(r["mouse_name"]), str(r["base_session"]))
        if key in param_map:
            df.loc[idx, "group_int_i"] = param_map[key][0]
            df.loc[idx, "group_int_f"] = param_map[key][1]
        elif str(r["date_str"]) in DATE_PARAMS_FALLBACK:
            fallback = DATE_PARAMS_FALLBACK[str(r["date_str"])]
            df.loc[idx, "group_int_i"] = fallback[0]
            df.loc[idx, "group_int_f"] = fallback[1]
        else:
            df.loc[idx, "group_int_i"] = r.get("int_i", 0.0)
            df.loc[idx, "group_int_f"] = r.get("int_f", 0.0)

    spaces = df["space"].unique().tolist() if "space" in df.columns else ["neural"]

    for sp in spaces:
        df_sp = df[df["space"] == sp].copy() if "space" in df.columns else df.copy()
        space_label = "Complete Neural Space" if sp == "neural" else "3D UMAP Space"
        print(f"\n>>> PROCESSING: [{space_label.upper()}] <<<")

        for (int_i, int_f), g_df in df_sp.groupby(["group_int_i", "group_int_f"]):
            print(f"--> Group int_i={int_i}, int_f={int_f} (N={len(g_df)} rows)...")
            plot_intervention_group(g_df, float(int_i), float(int_f), space_label)

    print("\n" + "=" * 80)
    print("ALL INTERVENTION PLOTS & STATS SUCCESSFULLY SAVED IN STANDARDIZATION PROJECT!")
    print(f"  Figures: {PNG_DIR} and {SVG_DIR}")
    print(f"  Clean:   {PNG_CLEAN_DIR} and {SVG_CLEAN_DIR}")
    print(f"  Stats:   {STATS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()