#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal-range aligned overlay plot with vertical shifts.

This script reads the saved lag-summary CSV from:
    LT_ROT_Scale_Effects_GlobalMean_Analysis

and generates:
    1) one figure for deep
    2) one figure for superficial
    3) one combined figure with deep and superficial together

Each figure has 2 panels:
    left  = lt
    right = rot

Each scale is plotted:
    x = actual temporal distance represented
    y = similarity + vertical offset

Combined figure:
    color = depth
    alpha = scale
    vertical shift = scale
    small depth offset separates deep and superficial within each scale band
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = (
    "/home/melisamc/Documents/time_project_standarization/"
    "LT_ROT_Scale_Effects_GlobalMean_Analysis"
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "csv",
    "lt_rot_scale_effect_lag_summary.csv",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "figures",
    "temporal_range_overlay_vertical_shift",
)

# =============================================================================
# PARAMETERS
# =============================================================================

# Must match the original analysis
SCALE_SPLITS = [1, 2, 4, 8, 16, 32]
N_TEMPORAL_BINS = 30

# Plot options
EXCLUDE_ZERO_LAG = True
USE_ABS_SIMILARITY = False

# If True, x-axis limits are inferred from data in each panel
AUTO_XLIM = False

# If AUTO_XLIM = False
FIXED_XMIN = 0.0
FIXED_XMAX = 50.0

# Vertical shift settings
VERTICAL_SHIFT_STEP = 0.20
SHIFT_HIGHEST_SCALE_AT_TOP = True

# Small separation between deep and superficial in combined plot
DEPTH_PAIR_SEPARATION = 0.0

# Y-axis padding after vertical shifting
Y_PADDING_LOW = 0.05
Y_PADDING_HIGH = 0.05

# Colors
DEEP_COLOR = "#D4A928"
SUPERFICIAL_COLOR = "#A020F0"

DEPTHS = ["deep", "superficial"]
ENV_ORDER = ["lt", "rot"]

ENV_LABELS = {
    "lt": "Environment A: lt",
    "rot": "Environment B: rot",
}


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def savefig_png_svg(fig, out_base):
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def normalize_depth(depth):
    depth = str(depth).lower()

    if depth in ["sup", "superficial"]:
        return "superficial"
    if "super" in depth:
        return "superficial"
    if "deep" in depth:
        return "deep"

    return depth


def normalize_environment(env):
    env = str(env).lower()

    if "rot" in env:
        return "rot"
    if "lt" in env:
        return "lt"

    return env


def get_required_columns():
    return [
        "mouse_name",
        "session_name",
        "depth_type",
        "environment",
        "scale_split",
        "lag",
        "similarity",
    ]


def get_scale_alpha_map(scale_splits):
    """
    Larger scales are lighter.
    """
    sorted_scales = sorted(scale_splits)
    default_alphas = np.linspace(1.0, 0.35, len(sorted_scales))

    return {
        s: float(a)
        for s, a in zip(sorted_scales, default_alphas)
    }


def get_scale_shift_map(scale_splits, step=0.08, highest_on_top=True):
    """
    Assign vertical offsets to scales.

    If highest_on_top=True:
        largest scale gets highest y-offset.
    """
    sorted_scales = sorted(scale_splits)

    if highest_on_top:
        order = sorted_scales[::-1]
    else:
        order = sorted_scales

    shift_map = {}
    for idx, s in enumerate(order):
        shift_map[s] = idx * step

    return shift_map


def compute_x_percent(lag_values, scale_split, n_temporal_bins):
    """
    Convert lag to actual temporal distance as % of full session.

    temporal distance fraction = lag / (scale_split * n_temporal_bins)
    """
    lag_values = np.asarray(lag_values, dtype=float)
    return 100.0 * lag_values / (scale_split * n_temporal_bins)


def prepare_dataframe(df):
    df = df.copy()

    df["depth_type"] = df["depth_type"].apply(normalize_depth)
    df["environment"] = df["environment"].apply(normalize_environment)

    if EXCLUDE_ZERO_LAG:
        df = df[df["lag"] > 0].copy()

    if USE_ABS_SIMILARITY:
        df["similarity_plot"] = df["similarity"].abs()
    else:
        df["similarity_plot"] = df["similarity"]

    return df


def apply_x_limits(ax, x_all):
    if AUTO_XLIM and len(x_all) > 0:
        xmin = max(0.0, np.nanmin(x_all))
        xmax = np.nanmax(x_all)

        xr = xmax - xmin
        if xr <= 0:
            xr = 1.0

        ax.set_xlim(xmin - 0.02 * xr, xmax + 0.04 * xr)
    else:
        ax.set_xlim(FIXED_XMIN, FIXED_XMAX)


def apply_common_ylim(axes, global_ymin, global_ymax):
    if not np.isfinite(global_ymin):
        global_ymin = -0.2
    if not np.isfinite(global_ymax):
        global_ymax = 1.0

    yr = global_ymax - global_ymin
    if yr <= 0:
        yr = 1.0

    for ax in axes:
        ax.set_ylim(
            global_ymin - Y_PADDING_LOW * yr,
            global_ymax + Y_PADDING_HIGH * yr,
        )


# =============================================================================
# INDIVIDUAL DEPTH PLOT
# =============================================================================

def plot_temporal_overlay_vertical_shift_for_depth(
        lag_df,
        depth,
        out_base,
):
    """
    One figure per depth.

    Two panels:
        left  = lt
        right = rot

    Curves:
        x = actual temporal distance represented
        y = similarity + vertical offset
    """
    depth = normalize_depth(depth)

    depth_color_map = {
        "deep": DEEP_COLOR,
        "superficial": SUPERFICIAL_COLOR,
    }

    if depth not in depth_color_map:
        raise ValueError(f"Unknown depth: {depth}")

    base_color = depth_color_map[depth]

    df = prepare_dataframe(lag_df)
    df = df[
        (df["depth_type"] == depth)
        & (df["environment"].isin(ENV_ORDER))
        & (df["scale_split"].isin(SCALE_SPLITS))
        ].copy()

    if len(df) == 0:
        print(f"[WARNING] No data available for depth={depth}")
        return

    alpha_map = get_scale_alpha_map(SCALE_SPLITS)
    shift_map = get_scale_shift_map(
        SCALE_SPLITS,
        step=VERTICAL_SHIFT_STEP,
        highest_on_top=SHIFT_HIGHEST_SCALE_AT_TOP,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=False)

    global_ymin = np.inf
    global_ymax = -np.inf

    for ax, env in zip(axes, ENV_ORDER):
        env_df = df[df["environment"] == env].copy()
        x_all = []

        for scale_split in sorted(SCALE_SPLITS):
            sub = env_df[env_df["scale_split"] == scale_split].copy()

            if len(sub) == 0:
                continue

            stats = (
                sub.groupby("lag", observed=True)["similarity_plot"]
                .agg(["mean", "sem", "count"])
                .reset_index()
                .sort_values("lag")
            )

            if len(stats) == 0:
                continue

            x_percent = compute_x_percent(
                lag_values=stats["lag"].values,
                scale_split=scale_split,
                n_temporal_bins=N_TEMPORAL_BINS,
            )

            y = stats["mean"].values
            yerr = stats["sem"].values

            shift = shift_map[scale_split]
            alpha = alpha_map[scale_split]

            y_shifted = y + shift

            ax.plot(
                x_percent,
                y_shifted,
                marker="o",
                markersize=4,
                linewidth=2.0,
                color=base_color,
                alpha=alpha,
                label=f"scale = {scale_split}",
            )

            ax.fill_between(
                x_percent,
                y_shifted - yerr,
                y_shifted + yerr,
                color=base_color,
                alpha=max(0.06, alpha * 0.15),
                linewidth=0,
            )

            ax.axhline(
                shift,
                color=base_color,
                alpha=max(0.10, alpha * 0.25),
                linewidth=0.8,
                linestyle="--",
            )

            if len(x_percent) > 0:
                ax.text(
                    x_percent.min(),
                    shift,
                    f"S={scale_split}",
                    color=base_color,
                    alpha=alpha,
                    fontsize=8,
                    va="bottom",
                    ha="left",
                )

            x_all.extend(x_percent.tolist())

            local_ymin = np.nanmin(y_shifted - yerr)
            local_ymax = np.nanmax(y_shifted + yerr)

            if np.isfinite(local_ymin):
                global_ymin = min(global_ymin, local_ymin)
            if np.isfinite(local_ymax):
                global_ymax = max(global_ymax, local_ymax)

        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Temporal distance (% of full session)")
        ax.grid(alpha=0.20)
        apply_x_limits(ax, x_all)

        ax.legend(frameon=False, fontsize=8, title="Pseudo-trials")

    apply_common_ylim(axes, global_ymin, global_ymax)

    axes[0].set_ylabel("Representational similarity + vertical shift")

    title_suffix = "absolute similarity" if USE_ABS_SIMILARITY else "signed similarity"

    fig.suptitle(
        f"{depth.capitalize()} | temporal-range aligned overlay with vertical shifts ({title_suffix})",
        fontsize=13,
    )

    fig.tight_layout()
    savefig_png_svg(fig, out_base)

    print(f"[OK] Saved: {out_base}.png/.svg")


# =============================================================================
# COMBINED DEEP + SUPERFICIAL PLOT
# =============================================================================

def plot_temporal_overlay_vertical_shift_both_depths(
        lag_df,
        out_base,
):
    """
    Combined figure with deep and superficial together.

    Two panels:
        left  = lt
        right = rot

    Encoding:
        color = depth
        alpha = scale
        vertical shift = scale
        small extra offset = deep vs superficial within each scale band
    """
    df = prepare_dataframe(lag_df)

    df = df[
        (df["depth_type"].isin(["deep", "superficial"]))
        & (df["environment"].isin(ENV_ORDER))
        & (df["scale_split"].isin(SCALE_SPLITS))
        ].copy()

    if len(df) == 0:
        print("[WARNING] No data available for combined deep/superficial plot")
        return

    depth_color_map = {
        "deep": DEEP_COLOR,
        "superficial": SUPERFICIAL_COLOR,
    }

    alpha_map = get_scale_alpha_map(SCALE_SPLITS)
    shift_map = get_scale_shift_map(
        SCALE_SPLITS,
        step=VERTICAL_SHIFT_STEP,
        highest_on_top=SHIFT_HIGHEST_SCALE_AT_TOP,
    )

    depth_offset_map = {
        "deep": -DEPTH_PAIR_SEPARATION / 2.0,
        "superficial": +DEPTH_PAIR_SEPARATION / 2.0,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=False)

    global_ymin = np.inf
    global_ymax = -np.inf

    for ax, env in zip(axes, ENV_ORDER):
        env_df = df[df["environment"] == env].copy()
        x_all = []

        for scale_split in sorted(SCALE_SPLITS):
            scale_df = env_df[env_df["scale_split"] == scale_split].copy()

            if len(scale_df) == 0:
                continue

            base_shift = shift_map[scale_split]
            alpha = alpha_map[scale_split]

            ax.axhline(
                base_shift,
                color="black",
                alpha=max(0.08, alpha * 0.18),
                linewidth=0.8,
                linestyle="--",
            )

            ax.text(
                0.0,
                base_shift,
                f"S={scale_split}",
                color="black",
                alpha=max(0.35, alpha),
                fontsize=8,
                va="bottom",
                ha="left",
            )

            for depth in ["deep", "superficial"]:
                sub = scale_df[scale_df["depth_type"] == depth].copy()

                if len(sub) == 0:
                    continue

                stats = (
                    sub.groupby("lag", observed=True)["similarity_plot"]
                    .agg(["mean", "sem", "count"])
                    .reset_index()
                    .sort_values("lag")
                )

                if len(stats) == 0:
                    continue

                x_percent = compute_x_percent(
                    lag_values=stats["lag"].values,
                    scale_split=scale_split,
                    n_temporal_bins=N_TEMPORAL_BINS,
                )

                y = stats["mean"].values
                yerr = stats["sem"].values

                y_shifted = y + base_shift + depth_offset_map[depth]

                ax.plot(
                    x_percent,
                    y_shifted,
                    marker="o",
                    markersize=4,
                    linewidth=2.0,
                    color=depth_color_map[depth],
                    alpha=alpha,
                )

                ax.fill_between(
                    x_percent,
                    y_shifted - yerr,
                    y_shifted + yerr,
                    color=depth_color_map[depth],
                    alpha=max(0.05, alpha * 0.12),
                    linewidth=0,
                )

                x_all.extend(x_percent.tolist())

                local_ymin = np.nanmin(y_shifted - yerr)
                local_ymax = np.nanmax(y_shifted + yerr)

                if np.isfinite(local_ymin):
                    global_ymin = min(global_ymin, local_ymin)
                if np.isfinite(local_ymax):
                    global_ymax = max(global_ymax, local_ymax)

        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Temporal distance (% of full session)")
        ax.grid(alpha=0.20)
        apply_x_limits(ax, x_all)

    apply_common_ylim(axes, global_ymin, global_ymax)

    axes[0].set_ylabel("Representational similarity + vertical shift")

    depth_handles = [
        Line2D([0], [0], color=DEEP_COLOR, lw=3, label="deep"),
        Line2D([0], [0], color=SUPERFICIAL_COLOR, lw=3, label="superficial"),
    ]

    scale_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            alpha=alpha_map[s],
            lw=3,
            marker="o",
            markersize=5,
            label=f"scale = {s}",
        )
        for s in sorted(SCALE_SPLITS)
    ]

    leg1 = axes[1].legend(
        handles=depth_handles,
        title="Depth",
        loc="upper right",
        frameon=False,
    )
    axes[1].add_artist(leg1)

    axes[1].legend(
        handles=scale_handles,
        title="Pseudo-trials",
        loc="center right",
        frameon=False,
    )

    title_suffix = "absolute similarity" if USE_ABS_SIMILARITY else "signed similarity"

    fig.suptitle(
        f"Deep + superficial | temporal-range aligned overlay with vertical shifts ({title_suffix})",
        fontsize=13,
    )

    fig.tight_layout()
    savefig_png_svg(fig, out_base)

    print(f"[OK] Saved combined plot: {out_base}.png/.svg")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found:\n{CSV_PATH}")

    lag_df = pd.read_csv(CSV_PATH)

    missing = [c for c in get_required_columns() if c not in lag_df.columns]
    if len(missing) > 0:
        raise KeyError(
            "Missing required columns in lag summary CSV:\n" + "\n".join(missing)
        )

    print("==============================================================")
    print("Temporal-range overlay with vertical shifts")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Scales: {SCALE_SPLITS}")
    print(f"N temporal bins: {N_TEMPORAL_BINS}")
    print(f"Exclude lag 0: {EXCLUDE_ZERO_LAG}")
    print(f"Use abs(similarity): {USE_ABS_SIMILARITY}")
    print(f"Vertical shift step: {VERTICAL_SHIFT_STEP}")
    print(f"Depth pair separation: {DEPTH_PAIR_SEPARATION}")
    print(f"Auto x-limits: {AUTO_XLIM}")
    print("")

    plot_temporal_overlay_vertical_shift_for_depth(
        lag_df=lag_df,
        depth="deep",
        out_base=os.path.join(
            OUTPUT_DIR,
            "temporal_range_overlay_vertical_shift_deep",
        ),
    )

    plot_temporal_overlay_vertical_shift_for_depth(
        lag_df=lag_df,
        depth="superficial",
        out_base=os.path.join(
            OUTPUT_DIR,
            "temporal_range_overlay_vertical_shift_superficial",
        ),
    )

    plot_temporal_overlay_vertical_shift_both_depths(
        lag_df=lag_df,
        out_base=os.path.join(
            OUTPUT_DIR,
            "temporal_range_overlay_vertical_shift_deep_and_superficial",
        ),
    )

    print("")
    print("Done.")


if __name__ == "__main__":
    main()


#################################################################33
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Animal-by-animal temporal-range aligned overlay plots with vertical shifts.

This script reads the saved lag-summary CSV from:
    LT_ROT_Scale_Effects_GlobalMean_Analysis

and generates:
    - one combined figure per animal

Each figure has 2 panels:
    left  = lt
    right = rot

Each panel shows:
    - deep and superficial together
    - color = depth
    - alpha = scale
    - vertical shift = scale
    - NO extra shift between deep and superficial
      (they share the same scale baseline)

This is the animal-by-animal version of the combined curves plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = (
    "/home/melisamc/Documents/time_project_standarization/"
    "LT_ROT_Scale_Effects_GlobalMean_Analysis"
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "csv",
    "lt_rot_scale_effect_lag_summary.csv",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "figures",
    "temporal_range_overlay_vertical_shift_per_animal",
)


# =============================================================================
# PARAMETERS
# =============================================================================

# Must match your original analysis
SCALE_SPLITS = [1, 2, 4, 8, 16, 32]
N_TEMPORAL_BINS = 30

# Plot options
EXCLUDE_ZERO_LAG = True
USE_ABS_SIMILARITY = False

# X-axis settings
AUTO_XLIM = False
FIXED_XMIN = 0.0
FIXED_XMAX = 50.0

# Vertical shift settings
VERTICAL_SHIFT_STEP = 0.20
SHIFT_HIGHEST_SCALE_AT_TOP = True

# Y padding
Y_PADDING_LOW = 0.05
Y_PADDING_HIGH = 0.05

# Colors
DEEP_COLOR = "#D4A928"
SUPERFICIAL_COLOR = "#A020F0"

DEPTHS = ["deep", "superficial"]
ENV_ORDER = ["lt", "rot"]

ENV_LABELS = {
    "lt": "Environment A: lt",
    "rot": "Environment B: rot",
}


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def savefig_png_svg(fig, out_base):
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def normalize_depth(depth):
    depth = str(depth).lower()

    if depth in ["sup", "superficial"]:
        return "superficial"
    if "super" in depth:
        return "superficial"
    if "deep" in depth:
        return "deep"

    return depth


def normalize_environment(env):
    env = str(env).lower()

    if "rot" in env:
        return "rot"
    if "lt" in env:
        return "lt"

    return env


def sanitize_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def get_required_columns():
    return [
        "mouse_name",
        "session_name",
        "depth_type",
        "environment",
        "scale_split",
        "lag",
        "similarity",
    ]


def get_scale_alpha_map(scale_splits):
    """
    Larger scales are lighter.
    """
    sorted_scales = sorted(scale_splits)
    default_alphas = np.linspace(1.0, 0.35, len(sorted_scales))

    return {
        s: float(a)
        for s, a in zip(sorted_scales, default_alphas)
    }


def get_scale_shift_map(scale_splits, step=0.08, highest_on_top=True):
    """
    Assign vertical offsets to scales.
    """
    sorted_scales = sorted(scale_splits)

    if highest_on_top:
        order = sorted_scales[::-1]
    else:
        order = sorted_scales

    shift_map = {}
    for idx, s in enumerate(order):
        shift_map[s] = idx * step

    return shift_map


def compute_x_percent(lag_values, scale_split, n_temporal_bins):
    """
    Convert lag to actual temporal distance as % of full session.
    """
    lag_values = np.asarray(lag_values, dtype=float)
    return 100.0 * lag_values / (scale_split * n_temporal_bins)


def prepare_dataframe(df):
    df = df.copy()

    df["depth_type"] = df["depth_type"].apply(normalize_depth)
    df["environment"] = df["environment"].apply(normalize_environment)

    if EXCLUDE_ZERO_LAG:
        df = df[df["lag"] > 0].copy()

    if USE_ABS_SIMILARITY:
        df["similarity_plot"] = df["similarity"].abs()
    else:
        df["similarity_plot"] = df["similarity"]

    return df


def apply_x_limits(ax, x_all):
    if AUTO_XLIM and len(x_all) > 0:
        xmin = max(0.0, np.nanmin(x_all))
        xmax = np.nanmax(x_all)

        xr = xmax - xmin
        if xr <= 0:
            xr = 1.0

        ax.set_xlim(xmin - 0.02 * xr, xmax + 0.04 * xr)
    else:
        ax.set_xlim(FIXED_XMIN, FIXED_XMAX)


def apply_common_ylim(axes, global_ymin, global_ymax):
    if not np.isfinite(global_ymin):
        global_ymin = -0.2
    if not np.isfinite(global_ymax):
        global_ymax = 1.0

    yr = global_ymax - global_ymin
    if yr <= 0:
        yr = 1.0

    for ax in axes:
        ax.set_ylim(
            global_ymin - Y_PADDING_LOW * yr,
            global_ymax + Y_PADDING_HIGH * yr,
        )


# =============================================================================
# MAIN PLOTTING FUNCTION: ONE FIGURE PER ANIMAL
# =============================================================================

def plot_temporal_overlay_vertical_shift_for_one_animal(
    lag_df,
    mouse_name,
    out_base,
):
    """
    One combined figure for one animal.

    Two panels:
        left  = lt
        right = rot

    Encoding:
        color = depth
        alpha = scale
        vertical shift = scale
        NO extra shift for deep vs superficial
    """
    df = prepare_dataframe(lag_df)

    df = df[
        (df["mouse_name"] == mouse_name)
        & (df["depth_type"].isin(["deep", "superficial"]))
        & (df["environment"].isin(ENV_ORDER))
        & (df["scale_split"].isin(SCALE_SPLITS))
    ].copy()

    if len(df) == 0:
        print(f"[WARNING] No data available for mouse: {mouse_name}")
        return

    depth_color_map = {
        "deep": DEEP_COLOR,
        "superficial": SUPERFICIAL_COLOR,
    }

    alpha_map = get_scale_alpha_map(SCALE_SPLITS)
    shift_map = get_scale_shift_map(
        SCALE_SPLITS,
        step=VERTICAL_SHIFT_STEP,
        highest_on_top=SHIFT_HIGHEST_SCALE_AT_TOP,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=False)

    global_ymin = np.inf
    global_ymax = -np.inf

    for ax, env in zip(axes, ENV_ORDER):
        env_df = df[df["environment"] == env].copy()
        x_all = []

        for scale_split in sorted(SCALE_SPLITS):
            scale_df = env_df[env_df["scale_split"] == scale_split].copy()

            if len(scale_df) == 0:
                continue

            base_shift = shift_map[scale_split]
            alpha = alpha_map[scale_split]

            # horizontal dashed baseline for this scale
            ax.axhline(
                base_shift,
                color="black",
                alpha=max(0.08, alpha * 0.18),
                linewidth=0.8,
                linestyle="--",
            )

            # scale label at the left side
            ax.text(
                0.0,
                base_shift,
                f"S={scale_split}",
                color="black",
                alpha=max(0.35, alpha),
                fontsize=8,
                va="bottom",
                ha="left",
            )

            for depth in ["deep", "superficial"]:
                sub = scale_df[scale_df["depth_type"] == depth].copy()

                if len(sub) == 0:
                    continue

                # For one animal, there may still be multiple rows per lag
                # (e.g. multiple sessions/entries), so keep mean±sem.
                stats = (
                    sub.groupby("lag", observed=True)["similarity_plot"]
                    .agg(["mean", "sem", "count"])
                    .reset_index()
                    .sort_values("lag")
                )

                if len(stats) == 0:
                    continue

                x_percent = compute_x_percent(
                    lag_values=stats["lag"].values,
                    scale_split=scale_split,
                    n_temporal_bins=N_TEMPORAL_BINS,
                )

                y = stats["mean"].values
                yerr = stats["sem"].fillna(0.0).values

                # IMPORTANT:
                # deep and superficial use the SAME shift for each scale
                y_shifted = y + base_shift

                ax.plot(
                    x_percent,
                    y_shifted,
                    marker="o",
                    markersize=4,
                    linewidth=2.0,
                    color=depth_color_map[depth],
                    alpha=alpha,
                )

                ax.fill_between(
                    x_percent,
                    y_shifted - yerr,
                    y_shifted + yerr,
                    color=depth_color_map[depth],
                    alpha=max(0.05, alpha * 0.12),
                    linewidth=0,
                )

                x_all.extend(x_percent.tolist())

                local_ymin = np.nanmin(y_shifted - yerr)
                local_ymax = np.nanmax(y_shifted + yerr)

                if np.isfinite(local_ymin):
                    global_ymin = min(global_ymin, local_ymin)
                if np.isfinite(local_ymax):
                    global_ymax = max(global_ymax, local_ymax)

        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Temporal distance (% of full session)")
        ax.grid(alpha=0.20)
        apply_x_limits(ax, x_all)

    apply_common_ylim(axes, global_ymin, global_ymax)

    axes[0].set_ylabel("Representational similarity + vertical shift")

    depth_handles = [
        Line2D([0], [0], color=DEEP_COLOR, lw=3, label="deep"),
        Line2D([0], [0], color=SUPERFICIAL_COLOR, lw=3, label="superficial"),
    ]

    scale_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            alpha=alpha_map[s],
            lw=3,
            marker="o",
            markersize=5,
            label=f"scale = {s}",
        )
        for s in sorted(SCALE_SPLITS)
    ]

    leg1 = axes[1].legend(
        handles=depth_handles,
        title="Depth",
        loc="upper right",
        frameon=False,
    )
    axes[1].add_artist(leg1)

    axes[1].legend(
        handles=scale_handles,
        title="Pseudo-trials",
        loc="center right",
        frameon=False,
    )

    title_suffix = "absolute similarity" if USE_ABS_SIMILARITY else "signed similarity"

    fig.suptitle(
        f"{mouse_name} | Deep + superficial | temporal-range aligned overlay with vertical shifts ({title_suffix})",
        fontsize=13,
    )

    fig.tight_layout()
    savefig_png_svg(fig, out_base)

    print(f"[OK] Saved animal plot: {out_base}.png/.svg")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found:\n{CSV_PATH}")

    lag_df = pd.read_csv(CSV_PATH)

    missing = [c for c in get_required_columns() if c not in lag_df.columns]
    if len(missing) > 0:
        raise KeyError(
            "Missing required columns in lag summary CSV:\n" + "\n".join(missing)
        )

    # Normalize once here to identify animals cleanly
    lag_df["depth_type"] = lag_df["depth_type"].apply(normalize_depth)
    lag_df["environment"] = lag_df["environment"].apply(normalize_environment)

    valid_df = lag_df[
        lag_df["depth_type"].isin(["deep", "superficial"])
        & lag_df["environment"].isin(ENV_ORDER)
        & lag_df["scale_split"].isin(SCALE_SPLITS)
    ].copy()

    mouse_list = sorted(valid_df["mouse_name"].dropna().unique().tolist())

    print("==============================================================")
    print("Animal-by-animal temporal-range overlay with vertical shifts")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Scales: {SCALE_SPLITS}")
    print(f"N temporal bins: {N_TEMPORAL_BINS}")
    print(f"Exclude lag 0: {EXCLUDE_ZERO_LAG}")
    print(f"Use abs(similarity): {USE_ABS_SIMILARITY}")
    print(f"Vertical shift step: {VERTICAL_SHIFT_STEP}")
    print(f"Auto x-limits: {AUTO_XLIM}")
    print(f"Animals found: {len(mouse_list)}")
    print("")

    for mouse_name in mouse_list:
        out_base = os.path.join(
            OUTPUT_DIR,
            f"{sanitize_filename(mouse_name)}_temporal_range_overlay_vertical_shift_deep_and_superficial"
        )

        plot_temporal_overlay_vertical_shift_for_one_animal(
            lag_df=lag_df,
            mouse_name=mouse_name,
            out_base=out_base,
        )

    print("")
    print("Done.")


if __name__ == "__main__":
    main()