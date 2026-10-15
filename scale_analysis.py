#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LT-ROT scale-effect RSA / PV-style global-mean analysis.

Goal
----
Proof of concept for scale effects in lt_rot only.

For each lt_rot session:
    Environment A = lt
    Environment B = rot

For each scale in SCALE_SPLITS:
    1. Split the full session into scale pseudo-trials.
       Example:
           scale = 2  -> two halves
           scale = 4  -> four quarters
           scale = 8  -> eight global blocks
           scale = 16 -> sixteen global blocks

    2. Inside each pseudo-trial, split into N_TEMPORAL_BINS relative time bins.
       Example:
           N_TEMPORAL_BINS = 20

    3. For each relative temporal bin k, combine the same bin across pseudo-trials:
           bin k from pseudo-trial 1
           bin k from pseudo-trial 2
           ...
       into one population vector.

    4. Optional position-balanced combination:
           Within each relative time bin, compute position-bin-specific
           population vectors and average across position bins. This gives
           one global-mean vector per relative temporal bin, but reduces
           bias from unequal position sampling.

    5. Compute 20 x 20 representational similarity matrix:
           sim[i, j] = corr(PV_i, PV_j)

    6. Summarize by |Δ temporal bin|.

Outputs
-------
/home/melisamc/Documents/time_project_standarization/LT_ROT_Scale_Effects_GlobalMean_Analysis

Main figures:
    - Per-scale Environment A/B curves: deep vs superficial.
    - Scale grid: rows = scale, columns = environment.
    - Mean similarity by scale.

Notes
-----
This is intentionally close to the global_mean RSA/PV logic:
    representation = mean population vector
    similarity = Pearson correlation between vectors
"""

import os
import csv
import re
import pickle
import timeit
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Paths
# =============================================================================

CSV_PATH = (
    "/home/melisamc/Documents/time_project_standarization/"
    "data/metadata/master_dataset_index.csv"
)

OUTPUT_DIR = (
    "/home/melisamc/Documents/time_project_standarization/"
    "LT_ROT_Scale_Effects_GlobalMean_Analysis"
)

SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
ROW_MATRIX_DIR = os.path.join(FIG_DIR, "row_matrices")
MEAN_MATRIX_DIR = os.path.join(FIG_DIR, "mean_matrices")
CURVE_DIR = os.path.join(FIG_DIR, "curves")
SCALE_GRID_DIR = os.path.join(FIG_DIR, "scale_grids")

SESSION_SUMMARY_CSV = os.path.join(CSV_DIR, "lt_rot_scale_effect_session_summary.csv")
MATRIX_ENTRIES_CSV = os.path.join(CSV_DIR, "lt_rot_scale_effect_matrix_entries.csv")
LAG_SUMMARY_CSV = os.path.join(CSV_DIR, "lt_rot_scale_effect_lag_summary.csv")
FAILED_ROWS_CSV = os.path.join(CSV_DIR, "failed_rows.csv")

# =============================================================================
# Parameters
# =============================================================================

# Scale effect: split each session into this many pseudo-trials.
# Include 1 as baseline: no pseudo-trial splitting, just one full session.
SCALE_SPLITS = [1, 2, 4, 8, 16, 32]

# For every scale, compute a 20 x 20 similarity matrix.
N_TEMPORAL_BINS = 200

# Position balancing while still producing a global-mean vector.
POSITION_BALANCED_GLOBAL_MEAN = True
N_POSITION_BINS = 20
MIN_SAMPLES_PER_POSITION_BIN = 2

# Minimum pseudo-trials needed for a position bin to contribute.
# 1 is permissive and works well for proof of concept.
# You can later try 2 or more.
MIN_PSEUDOTRIALS_PER_POSITION_BIN = 1

REMOVE_MOV_DIR_ZERO = True

# Z-scoring:
#   True: z-score each neuron across the valid session before analysis.
ZSCORE_CELLS = True

MIN_TIMEPOINTS_PER_TEMPORAL_BIN = 3
MIN_NEURONS = 5

SIMILARITY_METRIC = "pearson"

# Negative values:
#   For comparing with your clipped RSA/PV plots, keep this True.
#   For raw correlation geometry, set to False.
CLIP_NEGATIVE_SIMILARITY_IN_RESULTS = False
# For the compact scale summary:
# If negative similarities are not clipped, average abs(similarity)
# so positive and negative values do not cancel out.
SUMMARY_USE_ABS_WHEN_NOT_CLIPPED = True
# Heatmap color range
HEATMAP_VMIN = -0.20
HEATMAP_VMAX = 0.20

# Summary plot y-axis range
SUMMARY_YMIN = -0.2
SUMMARY_YMAX = 0.2

# Exclude lag 0 from summary plots.
# lag 0 is self-comparison and is usually trivially high.
EXCLUDE_ZERO_LAG_IN_PLOTS = True

# Deep/superficial color scheme.
DEEP_COLOR = "#D4A928"
SUPERFICIAL_COLOR = "#A020F0"

DEPTHS_TO_COMPARE = ["deep", "superficial"]

ENVIRONMENT_ORDER = ["lt", "rot"]
ENVIRONMENT_LABELS = {
    "lt": "Environment A: lt",
    "rot": "Environment B: rot",
}

# =============================================================================
# CSV schemas
# =============================================================================

SESSION_SUMMARY_COLUMNS = [
                              "dataset_name",
                              "mouse_name",
                              "session_name",
                              "depth_type",
                              "environment",
                              "standardized_file_path",
                              "row_index",
                              "scale_split",
                              "n_pseudo_trials",
                              "n_temporal_bins",
                              "position_balanced_global_mean",
                              "n_position_bins",
                              "n_neurons",
                              "n_timepoints_original",
                              "n_timepoints_after_filter",
                              "mean_similarity_all",
                              "mean_similarity_offdiag",
                          ] + [
                              f"lag_{i}" for i in range(N_TEMPORAL_BINS)
                          ] + [
                              "session_pickle_name",
                              "session_pickle_path",
                          ]

MATRIX_ENTRY_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "environment",
    "row_index",
    "scale_split",
    "bin_i",
    "bin_j",
    "lag",
    "similarity",
]

LAG_SUMMARY_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "environment",
    "row_index",
    "scale_split",
    "lag",
    "similarity",
]

FAILED_COLUMNS = [
    "row_index",
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "standardized_file_path",
    "scale_split",
    "error",
]


# =============================================================================
# General helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ensure_all_dirs():
    for path in [
        OUTPUT_DIR,
        SESSION_PKL_DIR,
        CSV_DIR,
        LOG_DIR,
        FIG_DIR,
        ROW_MATRIX_DIR,
        MEAN_MATRIX_DIR,
        CURVE_DIR,
        SCALE_GRID_DIR,
    ]:
        ensure_dir(path)


def load_pickle_standard(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle_standard(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def ensure_1d(x):
    x = np.asarray(x)

    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]

    return x.reshape(-1)


def ensure_2d(x):
    x = np.asarray(x)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    return x


def get_pos_1d(pos):
    pos = np.asarray(pos)

    if pos.ndim == 1:
        return pos

    if pos.ndim == 2:
        return pos[:, 0]

    raise ValueError(f"Unexpected pos shape: {pos.shape}")


def sanitize_for_filename(text):
    text = str(text)
    safe = []

    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")

    return "".join(safe)


def init_csv(csv_path, columns):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_row(csv_path, columns, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def append_many_csv_rows(csv_path, columns, rows):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        for row in rows:
            writer.writerow(row)


def savefig_png_svg(fig, out_base):
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# =============================================================================
# Metadata helpers
# =============================================================================

def normalize_depth(depth):
    depth = str(depth).lower()

    if depth in ["sup", "superficial"]:
        return "superficial"

    if "super" in depth:
        return "superficial"

    if "deep" in depth:
        return "deep"

    if "whole" in depth:
        return "whole_population"

    return depth


def is_lt_rot_dataset(dataset_name):
    text = str(dataset_name).lower()
    return "lt_rot" in text or ("lt" in text and "rot" in text)


def infer_environment_from_session(session_name):
    """
    Infer environment from session_name only.

    Environment A = lt
    Environment B = rot
    """
    session = str(session_name).lower()

    if "rot" in session:
        return "rot"

    if "lt" in session:
        return "lt"

    return "unknown"


# =============================================================================
# Data loading
# =============================================================================

def load_standardized_row(row_idx, row):
    standardized_file_path = row["standardized_file_path"]

    if not os.path.isfile(standardized_file_path):
        raise FileNotFoundError(
            f"Missing standardized file for row {row_idx}: "
            f"{standardized_file_path}"
        )

    data = load_pickle_standard(standardized_file_path)

    required_keys = ["signal", "pos", "mov_dir", "speed", "time"]

    for key in required_keys:
        if key not in data:
            raise KeyError(
                f"Row {row_idx}: missing key '{key}' in "
                f"{standardized_file_path}. Available keys: {list(data.keys())}"
            )

    signal = ensure_2d(data["signal"])
    pos = ensure_1d(get_pos_1d(data["pos"]))
    mov_dir = ensure_1d(data["mov_dir"])
    speed = ensure_1d(data["speed"])
    time = ensure_1d(data["time"])

    trial_id = (
        ensure_1d(data["trial_id"])
        if data.get("trial_id", None) is not None
        else None
    )

    inner_time = (
        ensure_1d(data["inner_time"])
        if data.get("inner_time", None) is not None
        else None
    )

    n_time = signal.shape[0]

    def check_len(x, name):
        if x is None:
            return None

        if len(x) != n_time:
            raise ValueError(
                f"Row {row_idx}: length mismatch for {name}. "
                f"Expected {n_time}, got {len(x)}"
            )

        return x

    pos = check_len(pos, "pos")
    mov_dir = check_len(mov_dir, "mov_dir")
    speed = check_len(speed, "speed")
    time = check_len(time, "time")

    if trial_id is not None:
        trial_id = check_len(trial_id, "trial_id")

    if inner_time is not None:
        inner_time = check_len(inner_time, "inner_time")

    behavior = {
        "pos": pos,
        "mov_dir": mov_dir,
        "speed": speed,
        "time": time,
        "trial_id": trial_id,
        "inner_time": inner_time,
    }

    return signal.astype(float), behavior


# =============================================================================
# Representation computation
# =============================================================================

def zscore_activity_cells(signal):
    mu = np.nanmean(signal, axis=0, keepdims=True)
    sd = np.nanstd(signal, axis=0, keepdims=True)

    sd[sd == 0] = np.nan

    z = (signal - mu) / sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    return z


def split_array_indices(indices, n_parts):
    indices = np.asarray(indices)
    return [x for x in np.array_split(indices, n_parts)]


def make_pseudo_trial_relative_bins(n_time, n_pseudo_trials, n_temporal_bins):
    """
    Split full recording into pseudo-trials, then each pseudo-trial into
    relative temporal bins.

    Returns
    -------
    pseudo_bins : list of list of arrays
        pseudo_bins[pseudo_trial_id][temporal_bin_id] = indices
        indices are relative to the filtered signal.
    """
    full_indices = np.arange(n_time)
    pseudo_trials = split_array_indices(full_indices, n_pseudo_trials)

    pseudo_bins = []

    for pseudo_idx in pseudo_trials:
        if len(pseudo_idx) < n_temporal_bins * MIN_TIMEPOINTS_PER_TEMPORAL_BIN:
            # Still split, but downstream bins may be invalid.
            bins = split_array_indices(pseudo_idx, n_temporal_bins)
        else:
            bins = split_array_indices(pseudo_idx, n_temporal_bins)

        pseudo_bins.append(bins)

    return pseudo_bins


def safe_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < MIN_NEURONS:
        return np.nan

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def compute_similarity(x, y):
    if SIMILARITY_METRIC == "pearson":
        return safe_corr(x, y)

    raise ValueError(f"Unknown SIMILARITY_METRIC: {SIMILARITY_METRIC}")


def compute_global_mean_vector(signal, indices):
    """
    Basic global-mean population vector for a set of time indices.
    """
    if len(indices) < MIN_TIMEPOINTS_PER_TEMPORAL_BIN:
        return np.full(signal.shape[1], np.nan)

    return np.nanmean(signal[indices, :], axis=0)


def compute_position_balanced_global_mean_vector(
        signal,
        position,
        pseudo_bins,
        temporal_bin_id,
        position_edges,
):
    """
    Compute one population vector for a relative temporal bin by combining
    the same temporal bin across pseudo-trials in a position-balanced way.

    The output is still a vector of length n_neurons.

    Logic:
        For each pseudo-trial:
            take its temporal bin k
            split samples by position bin
            compute mean vector per position bin

        Then:
            average across pseudo-trials inside each position bin
            average across position bins

    This reduces the chance that the representation is dominated by one
    pseudo-trial sampling a different part of the track.
    """
    n_neurons = signal.shape[1]

    pos_bin_vectors = []

    for b in range(len(position_edges) - 1):
        lo = position_edges[b]
        hi = position_edges[b + 1]

        vectors_this_pos_bin = []

        for pseudo_trial_bins in pseudo_bins:
            idx = pseudo_trial_bins[temporal_bin_id]

            if len(idx) < MIN_TIMEPOINTS_PER_TEMPORAL_BIN:
                continue

            pos_values = position[idx]

            if b == len(position_edges) - 2:
                mask = (pos_values >= lo) & (pos_values <= hi)
            else:
                mask = (pos_values >= lo) & (pos_values < hi)

            if np.sum(mask) >= MIN_SAMPLES_PER_POSITION_BIN:
                idx_pos = idx[mask]
                vectors_this_pos_bin.append(
                    np.nanmean(signal[idx_pos, :], axis=0)
                )

        if len(vectors_this_pos_bin) >= MIN_PSEUDOTRIALS_PER_POSITION_BIN:
            vectors_this_pos_bin = np.vstack(vectors_this_pos_bin)
            pos_bin_vectors.append(
                np.nanmean(vectors_this_pos_bin, axis=0)
            )

    if len(pos_bin_vectors) == 0:
        return np.full(n_neurons, np.nan)

    pos_bin_vectors = np.vstack(pos_bin_vectors)

    return np.nanmean(pos_bin_vectors, axis=0)


def compute_scale_representations(signal, position, scale_split):
    """
    Compute N_TEMPORAL_BINS representation vectors for one scale.

    Each representation is one global-mean population vector.
    """
    n_time = signal.shape[0]

    pseudo_bins = make_pseudo_trial_relative_bins(
        n_time=n_time,
        n_pseudo_trials=scale_split,
        n_temporal_bins=N_TEMPORAL_BINS,
    )

    position_edges = None

    if POSITION_BALANCED_GLOBAL_MEAN:
        pos_min = np.nanmin(position)
        pos_max = np.nanmax(position)

        if not np.isfinite(pos_min) or not np.isfinite(pos_max) or pos_min == pos_max:
            raise ValueError("Invalid position vector for position-balanced global mean.")

        position_edges = np.linspace(pos_min, pos_max, N_POSITION_BINS + 1)

    representations = []

    for temporal_bin_id in range(N_TEMPORAL_BINS):
        if POSITION_BALANCED_GLOBAL_MEAN:
            rep = compute_position_balanced_global_mean_vector(
                signal=signal,
                position=position,
                pseudo_bins=pseudo_bins,
                temporal_bin_id=temporal_bin_id,
                position_edges=position_edges,
            )
        else:
            pooled_indices = []

            for pseudo_trial_bins in pseudo_bins:
                pooled_indices.extend(pseudo_trial_bins[temporal_bin_id].tolist())

            pooled_indices = np.asarray(pooled_indices, dtype=int)
            rep = compute_global_mean_vector(signal, pooled_indices)

        representations.append(rep)

    return representations, pseudo_bins, position_edges


def compute_similarity_matrix_from_representations(representations):
    mat = np.full((N_TEMPORAL_BINS, N_TEMPORAL_BINS), np.nan)

    for i in range(N_TEMPORAL_BINS):
        for j in range(N_TEMPORAL_BINS):
            mat[i, j] = compute_similarity(representations[i], representations[j])

    if CLIP_NEGATIVE_SIMILARITY_IN_RESULTS:
        mat = np.where(
            np.isfinite(mat),
            np.clip(mat, 0.0, 1.0),
            np.nan,
        )

    return mat


def matrix_to_lag_summary(mat):
    out = {}

    for lag in range(N_TEMPORAL_BINS):
        vals = []

        for i in range(N_TEMPORAL_BINS):
            for j in range(N_TEMPORAL_BINS):
                if abs(i - j) == lag:
                    vals.append(mat[i, j])

        out[lag] = float(np.nanmean(vals))

    return out


def mean_offdiagonal(mat):
    vals = []

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i != j:
                vals.append(mat[i, j])

    return float(np.nanmean(vals))


# =============================================================================
# Plotting
# =============================================================================

def get_plot_lags(df):
    lags = sorted(df["lag"].dropna().unique())

    if EXCLUDE_ZERO_LAG_IN_PLOTS:
        lags = [lag for lag in lags if lag > 0]

    return lags


def plot_similarity_matrix(mat, title, out_base):
    fig, ax = plt.subplots(figsize=(5.6, 5.0))

    im = ax.imshow(
        np.clip(mat, HEATMAP_VMIN, HEATMAP_VMAX),
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        origin="upper",
        aspect="equal",
    )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Relative temporal bin")
    ax.set_ylabel("Relative temporal bin")

    ax.set_xticks(np.arange(N_TEMPORAL_BINS))
    ax.set_yticks(np.arange(N_TEMPORAL_BINS))
    ax.set_xticklabels([str(i + 1) for i in range(N_TEMPORAL_BINS)], fontsize=6)
    ax.set_yticklabels([str(i + 1) for i in range(N_TEMPORAL_BINS)], fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Representational similarity")

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_env_deep_sup_curves_for_scale(lag_df, scale_split, out_base):
    df = lag_df[
        (lag_df["scale_split"] == scale_split)
        & lag_df["environment"].isin(["lt", "rot"])
        & lag_df["depth_type"].isin(["deep", "superficial"])
        ].copy()

    if len(df) == 0:
        return

    lags = get_plot_lags(df)

    if len(lags) == 0:
        return

    depth_colors = {
        "deep": DEEP_COLOR,
        "superficial": SUPERFICIAL_COLOR,
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.6),
        sharey=True,
    )

    for ax, env in zip(axes, ENVIRONMENT_ORDER):
        env_df = df[df["environment"] == env].copy()

        for depth in DEPTHS_TO_COMPARE:
            sub = env_df[env_df["depth_type"] == depth].copy()

            if len(sub) == 0:
                continue

            stats = (
                sub.groupby("lag", observed=True)["similarity"]
                .agg(["mean", "sem", "count"])
                .reset_index()
            )

            y = []
            yerr = []

            for lag in lags:
                row = stats[stats["lag"] == lag]

                if len(row) == 0:
                    y.append(np.nan)
                    yerr.append(np.nan)
                else:
                    y.append(float(row["mean"].iloc[0]))
                    yerr.append(float(row["sem"].iloc[0]))

            y = np.asarray(y, dtype=float)
            yerr = np.asarray(yerr, dtype=float)

            ax.plot(
                lags,
                y,
                marker="o",
                linewidth=2.5,
                markersize=5,
                color=depth_colors[depth],
                label=f"{depth} (n={sub['mouse_name'].nunique()})",
            )

            ax.fill_between(
                lags,
                y - yerr,
                y + yerr,
                color=depth_colors[depth],
                alpha=0.20,
                linewidth=0,
            )

        ax.set_title(ENVIRONMENT_LABELS[env])
        ax.set_xlabel("|Δ temporal bin|")
        ax.set_ylim(SUMMARY_YMIN, SUMMARY_YMAX)
        ax.set_xticks(lags)
        ax.legend(frameon=False)

    axes[0].set_ylabel("Global-mean representational similarity")

    fig.suptitle(
        f"lt_rot scale effect | pseudo-trials = {scale_split} | "
        f"bins = {N_TEMPORAL_BINS}"
    )

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_scale_grid(lag_df, out_base):
    df = lag_df[
        lag_df["environment"].isin(["lt", "rot"])
        & lag_df["depth_type"].isin(["deep", "superficial"])
        ].copy()

    if len(df) == 0:
        return

    depth_colors = {
        "deep": DEEP_COLOR,
        "superficial": SUPERFICIAL_COLOR,
    }

    n_rows = len(SCALE_SPLITS)
    n_cols = len(ENVIRONMENT_ORDER)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.0, 2.6 * n_rows),
        sharex=True,
        sharey=True,
    )

    if n_rows == 1:
        axes = np.asarray([axes])

    for r, scale_split in enumerate(SCALE_SPLITS):
        for c, env in enumerate(ENVIRONMENT_ORDER):
            ax = axes[r, c]

            sub_env = df[
                (df["scale_split"] == scale_split)
                & (df["environment"] == env)
                ].copy()

            if len(sub_env) == 0:
                continue

            lags = get_plot_lags(sub_env)

            for depth in DEPTHS_TO_COMPARE:
                sub = sub_env[sub_env["depth_type"] == depth].copy()

                if len(sub) == 0:
                    continue

                stats = (
                    sub.groupby("lag", observed=True)["similarity"]
                    .agg(["mean", "sem"])
                    .reset_index()
                )

                y = []
                yerr = []

                for lag in lags:
                    row = stats[stats["lag"] == lag]

                    if len(row) == 0:
                        y.append(np.nan)
                        yerr.append(np.nan)
                    else:
                        y.append(float(row["mean"].iloc[0]))
                        yerr.append(float(row["sem"].iloc[0]))

                y = np.asarray(y, dtype=float)
                yerr = np.asarray(yerr, dtype=float)

                ax.plot(
                    lags,
                    y,
                    marker="o",
                    linewidth=2.0,
                    markersize=4,
                    color=depth_colors[depth],
                    label=depth,
                )

                ax.fill_between(
                    lags,
                    y - yerr,
                    y + yerr,
                    color=depth_colors[depth],
                    alpha=0.20,
                    linewidth=0,
                )

            if r == 0:
                ax.set_title(ENVIRONMENT_LABELS[env])

            if c == 0:
                ax.set_ylabel(f"S={scale_split}\nsimilarity")

            if r == n_rows - 1:
                ax.set_xlabel("|Δ temporal bin|")

            ax.set_ylim(SUMMARY_YMIN, SUMMARY_YMAX)

            if r == 0 and c == 1:
                ax.legend(frameon=False)

    fig.suptitle("Scale effect: global-mean similarity curves")
    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_scale_summary(lag_df, out_base):
    """
    Summary across scales:
        y = mean similarity magnitude across nonzero lags
        x = number of pseudo-trials

    If CLIP_NEGATIVE_SIMILARITY_IN_RESULTS = False, we optionally average
    abs(similarity), because signed positive/negative values can cancel.
    """
    df = lag_df[
        lag_df["environment"].isin(["lt", "rot"])
        & lag_df["depth_type"].isin(["deep", "superficial"])
        ].copy()

    if EXCLUDE_ZERO_LAG_IN_PLOTS:
        df = df[df["lag"] > 0].copy()

    if len(df) == 0:
        return

    # ---------------------------------------------------------------------
    # Important scale-summary correction
    # ---------------------------------------------------------------------
    if (not CLIP_NEGATIVE_SIMILARITY_IN_RESULTS) and SUMMARY_USE_ABS_WHEN_NOT_CLIPPED:
        df["similarity_for_summary"] = df["similarity"].abs()
        y_label = "Mean |similarity| across nonzero lags"
        title = "Scale effect summary: similarity magnitude"
    else:
        df["similarity_for_summary"] = df["similarity"]
        y_label = "Mean similarity across nonzero lags"
        title = "Scale effect summary"

    # First average within each mouse/session/scale over lags,
    # then average across mice.
    per_row = (
        df.groupby(
            [
                "mouse_name",
                "session_name",
                "depth_type",
                "environment",
                "row_index",
                "scale_split",
            ],
            observed=True,
        )["similarity_for_summary"]
        .mean()
        .reset_index()
    )

    depth_colors = {
        "deep": DEEP_COLOR,
        "superficial": SUPERFICIAL_COLOR,
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)

    for ax, env in zip(axes, ENVIRONMENT_ORDER):
        env_df = per_row[per_row["environment"] == env].copy()

        for depth in DEPTHS_TO_COMPARE:
            sub = env_df[env_df["depth_type"] == depth].copy()

            if len(sub) == 0:
                continue

            stats = (
                sub.groupby("scale_split", observed=True)["similarity_for_summary"]
                .agg(["mean", "sem", "count"])
                .reset_index()
            )

            x = stats["scale_split"].values
            y = stats["mean"].values
            yerr = stats["sem"].values

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.5,
                markersize=6,
                color=depth_colors[depth],
                label=f"{depth} (n={sub['mouse_name'].nunique()})",
            )

            ax.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=depth_colors[depth],
                alpha=0.20,
                linewidth=0,
            )

        ax.set_title(ENVIRONMENT_LABELS[env])
        ax.set_xlabel("Number of pseudo-trials")
        ax.set_xticks(SCALE_SPLITS)
        ax.set_ylim(SUMMARY_YMIN, SUMMARY_YMAX)
        ax.legend(frameon=False)

    axes[0].set_ylabel(y_label)

    fig.suptitle(title)
    fig.tight_layout()
    savefig_png_svg(fig, out_base)


# =============================================================================
# Row processing
# =============================================================================

def process_row_scale(row_idx, row, scale_split):
    signal, behavior = load_standardized_row(row_idx, row)

    dataset_name = row["dataset_name"]
    mouse_name = row["mouse_name"]
    session_name = row["session_name"]
    depth_type = normalize_depth(row["depth_type"])
    environment = infer_environment_from_session(session_name)

    if environment not in ["lt", "rot"]:
        raise ValueError(
            f"Could not infer lt/rot environment from session_name={session_name}"
        )

    n_timepoints_original = signal.shape[0]
    n_neurons = signal.shape[1]

    if n_neurons < MIN_NEURONS:
        raise ValueError(f"Too few neurons: {n_neurons}")

    if REMOVE_MOV_DIR_ZERO:
        valid_idx = behavior["mov_dir"] != 0
    else:
        valid_idx = np.ones(signal.shape[0], dtype=bool)

    signal_valid = signal[valid_idx, :]
    position_valid = behavior["pos"][valid_idx]

    n_timepoints_after_filter = signal_valid.shape[0]

    min_needed = scale_split * N_TEMPORAL_BINS * MIN_TIMEPOINTS_PER_TEMPORAL_BIN

    if n_timepoints_after_filter < min_needed:
        raise ValueError(
            f"Too few timepoints after filtering for scale={scale_split}: "
            f"{n_timepoints_after_filter}. Need at least {min_needed}."
        )

    if ZSCORE_CELLS:
        signal_valid = zscore_activity_cells(signal_valid)

    representations, pseudo_bins, position_edges = compute_scale_representations(
        signal=signal_valid,
        position=position_valid,
        scale_split=scale_split,
    )

    similarity_matrix = compute_similarity_matrix_from_representations(
        representations
    )

    lag_summary = matrix_to_lag_summary(similarity_matrix)

    mean_similarity_all = float(np.nanmean(similarity_matrix))
    mean_similarity_offdiag = mean_offdiagonal(similarity_matrix)

    session_base = (
        f"row_{row_idx:04d}_"
        f"{sanitize_for_filename(dataset_name)}_"
        f"{sanitize_for_filename(mouse_name)}_"
        f"{sanitize_for_filename(session_name)}_"
        f"{sanitize_for_filename(depth_type)}_"
        f"{environment}_scale_{scale_split}"
    )

    session_pickle_name = session_base + ".pkl"
    session_pickle_path = os.path.join(SESSION_PKL_DIR, session_pickle_name)

    session_payload = {
        "metadata": {
            "dataset_name": dataset_name,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "depth_type": depth_type,
            "environment": environment,
            "row_index": row_idx,
            "standardized_file_path": row["standardized_file_path"],
            "scale_split": scale_split,
            "n_pseudo_trials": scale_split,
            "n_temporal_bins": N_TEMPORAL_BINS,
            "position_balanced_global_mean": POSITION_BALANCED_GLOBAL_MEAN,
            "n_position_bins": N_POSITION_BINS,
            "n_neurons": n_neurons,
            "n_timepoints_original": n_timepoints_original,
            "n_timepoints_after_filter": n_timepoints_after_filter,
            "zscore_cells": ZSCORE_CELLS,
            "remove_mov_dir_zero": REMOVE_MOV_DIR_ZERO,
            "clip_negative_similarity": CLIP_NEGATIVE_SIMILARITY_IN_RESULTS,
        },
        "indices": {
            "valid_idx_mov_dir_nonzero": valid_idx,
            "pseudo_trial_relative_bins_after_filter": pseudo_bins,
        },
        "position_edges": position_edges,
        "similarity": {
            "matrix": similarity_matrix,
            "representations": representations,
            "lag_summary": lag_summary,
        },
    }

    save_pickle_standard(session_pickle_path, session_payload)

    session_row = {
        "dataset_name": dataset_name,
        "mouse_name": mouse_name,
        "session_name": session_name,
        "depth_type": depth_type,
        "environment": environment,
        "standardized_file_path": row["standardized_file_path"],
        "row_index": row_idx,
        "scale_split": scale_split,
        "n_pseudo_trials": scale_split,
        "n_temporal_bins": N_TEMPORAL_BINS,
        "position_balanced_global_mean": POSITION_BALANCED_GLOBAL_MEAN,
        "n_position_bins": N_POSITION_BINS,
        "n_neurons": n_neurons,
        "n_timepoints_original": n_timepoints_original,
        "n_timepoints_after_filter": n_timepoints_after_filter,
        "mean_similarity_all": mean_similarity_all,
        "mean_similarity_offdiag": mean_similarity_offdiag,
    }

    for lag in range(N_TEMPORAL_BINS):
        session_row[f"lag_{lag}"] = lag_summary.get(lag, np.nan)

    session_row["session_pickle_name"] = session_pickle_name
    session_row["session_pickle_path"] = session_pickle_path

    matrix_rows = []

    for i in range(N_TEMPORAL_BINS):
        for j in range(N_TEMPORAL_BINS):
            matrix_rows.append(
                {
                    "dataset_name": dataset_name,
                    "mouse_name": mouse_name,
                    "session_name": session_name,
                    "depth_type": depth_type,
                    "environment": environment,
                    "row_index": row_idx,
                    "scale_split": scale_split,
                    "bin_i": i + 1,
                    "bin_j": j + 1,
                    "lag": abs(i - j),
                    "similarity": similarity_matrix[i, j],
                }
            )

    lag_rows = []

    for lag, value in lag_summary.items():
        lag_rows.append(
            {
                "dataset_name": dataset_name,
                "mouse_name": mouse_name,
                "session_name": session_name,
                "depth_type": depth_type,
                "environment": environment,
                "row_index": row_idx,
                "scale_split": scale_split,
                "lag": lag,
                "similarity": value,
            }
        )

    plot_similarity_matrix(
        similarity_matrix,
        title=(
            f"{ENVIRONMENT_LABELS[environment]} | {depth_type}\n"
            f"{mouse_name} | pseudo-trials={scale_split}"
        ),
        out_base=os.path.join(ROW_MATRIX_DIR, session_base + "_matrix"),
    )

    print(
        f"[OK] row={row_idx} | scale={scale_split} | env={environment} | "
        f"mouse={mouse_name} | depth={depth_type} | "
        f"mean_offdiag={mean_similarity_offdiag:.4f}"
    )

    return session_row, matrix_rows, lag_rows


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_mean_matrix(matrix_df, scale_split, env, depth):
    sub = matrix_df[
        (matrix_df["scale_split"] == scale_split)
        & (matrix_df["environment"] == env)
        & (matrix_df["depth_type"] == depth)
        ].copy()

    if len(sub) == 0:
        return None

    mat = np.full((N_TEMPORAL_BINS, N_TEMPORAL_BINS), np.nan)

    for i in range(N_TEMPORAL_BINS):
        for j in range(N_TEMPORAL_BINS):
            cell = sub[
                (sub["bin_i"] == i + 1)
                & (sub["bin_j"] == j + 1)
                ]

            if len(cell) > 0:
                mat[i, j] = np.nanmean(cell["similarity"].values)

    return mat


def run_aggregations():
    matrix_df = pd.read_csv(MATRIX_ENTRIES_CSV)
    lag_df = pd.read_csv(LAG_SUMMARY_CSV)

    if len(matrix_df) == 0 or len(lag_df) == 0:
        print("No successful entries. Skipping aggregation.")
        return

    matrix_df = matrix_df[
        matrix_df["environment"].isin(["lt", "rot"])
        & matrix_df["depth_type"].isin(["deep", "superficial"])
        ].copy()

    lag_df = lag_df[
        lag_df["environment"].isin(["lt", "rot"])
        & lag_df["depth_type"].isin(["deep", "superficial"])
        ].copy()

    matrix_df.to_csv(
        os.path.join(CSV_DIR, "lt_rot_scale_effect_matrix_entries_deep_superficial.csv"),
        index=False,
    )

    lag_df.to_csv(
        os.path.join(CSV_DIR, "lt_rot_scale_effect_lag_summary_deep_superficial.csv"),
        index=False,
    )

    # Per-scale curves.
    for scale_split in SCALE_SPLITS:
        plot_env_deep_sup_curves_for_scale(
            lag_df,
            scale_split=scale_split,
            out_base=os.path.join(
                CURVE_DIR,
                f"scale_{scale_split}_environment_A_B_deep_vs_superficial_curves",
            ),
        )

    # Mean matrices per scale/env/depth.
    for scale_split in SCALE_SPLITS:
        for env in ENVIRONMENT_ORDER:
            for depth in DEPTHS_TO_COMPARE:
                mat = aggregate_mean_matrix(
                    matrix_df,
                    scale_split=scale_split,
                    env=env,
                    depth=depth,
                )

                if mat is None:
                    continue

                plot_similarity_matrix(
                    mat,
                    title=(
                        f"{ENVIRONMENT_LABELS[env]} | {depth}\n"
                        f"mean matrix | pseudo-trials={scale_split}"
                    ),
                    out_base=os.path.join(
                        MEAN_MATRIX_DIR,
                        f"scale_{scale_split}_{env}_{depth}_mean_similarity_matrix",
                    ),
                )

    # Grid with all scales.
    plot_scale_grid(
        lag_df,
        out_base=os.path.join(
            SCALE_GRID_DIR,
            "scale_effect_grid_environment_A_B_deep_superficial",
        ),
    )

    # Summary across scales.
    plot_scale_summary(
        lag_df,
        out_base=os.path.join(
            SCALE_GRID_DIR,
            "scale_effect_summary_mean_nonzero_lag_similarity",
        ),
    )

    print("Aggregations completed.")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_all_dirs()

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_cols = [
        "dataset_name",
        "mouse_name",
        "session_name",
        "depth_type",
        "standardized_file_path",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"Missing required column '{col}' in master dataset CSV."
            )

    # Only lt_rot dataset.
    df = df[df["dataset_name"].apply(is_lt_rot_dataset)].copy()
    df["environment"] = df["session_name"].apply(infer_environment_from_session)
    df["depth_type_norm"] = df["depth_type"].apply(normalize_depth)

    df = df[
        df["environment"].isin(["lt", "rot"])
        & df["depth_type_norm"].isin(["deep", "superficial"])
        ].copy()

    df = df.reset_index(drop=True)

    init_csv(SESSION_SUMMARY_CSV, SESSION_SUMMARY_COLUMNS)
    init_csv(MATRIX_ENTRIES_CSV, MATRIX_ENTRY_COLUMNS)
    init_csv(LAG_SUMMARY_CSV, LAG_SUMMARY_COLUMNS)
    init_csv(FAILED_ROWS_CSV, FAILED_COLUMNS)

    log_path = os.path.join(LOG_DIR, "lt_rot_scale_effect_analysis_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("LT-ROT Scale-Effect Global-Mean Analysis")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Rows to process: {len(df)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Scale splits: {SCALE_SPLITS}")
    print(f"N temporal bins per scale: {N_TEMPORAL_BINS}")
    print(f"Position-balanced global mean: {POSITION_BALANCED_GLOBAL_MEAN}")
    print(f"N position bins: {N_POSITION_BINS}")
    print(f"Remove mov_dir == 0: {REMOVE_MOV_DIR_ZERO}")
    print(f"Z-score cells: {ZSCORE_CELLS}")
    print(f"Clip negative similarity: {CLIP_NEGATIVE_SIMILARITY_IN_RESULTS}")
    print("")

    ok_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        for scale_split in SCALE_SPLITS:
            local_time = timeit.default_timer()

            try:
                session_row, matrix_rows, lag_rows = process_row_scale(
                    row_idx=row_idx,
                    row=row,
                    scale_split=scale_split,
                )

                append_csv_row(
                    SESSION_SUMMARY_CSV,
                    SESSION_SUMMARY_COLUMNS,
                    session_row,
                )

                append_many_csv_rows(
                    MATRIX_ENTRIES_CSV,
                    MATRIX_ENTRY_COLUMNS,
                    matrix_rows,
                )

                append_many_csv_rows(
                    LAG_SUMMARY_CSV,
                    LAG_SUMMARY_COLUMNS,
                    lag_rows,
                )

                ok_count += 1

                elapsed = timeit.default_timer() - local_time
                print(
                    f"[OK] row={row_idx} | scale={scale_split} "
                    f"finished in {elapsed:.2f} s\n"
                )

            except Exception as e:
                fail_count += 1

                failed_row = {
                    "row_index": row_idx,
                    "dataset_name": row.get("dataset_name", ""),
                    "mouse_name": row.get("mouse_name", ""),
                    "session_name": row.get("session_name", ""),
                    "depth_type": row.get("depth_type", ""),
                    "standardized_file_path": row.get("standardized_file_path", ""),
                    "scale_split": scale_split,
                    "error": str(e),
                }

                append_csv_row(
                    FAILED_ROWS_CSV,
                    FAILED_COLUMNS,
                    failed_row,
                )

                print(
                    f"[ERROR] row={row_idx} | scale={scale_split}: {e}\n"
                )

    print("==============================================================")
    print("Running aggregations")
    print("==============================================================")

    run_aggregations()

    total_elapsed = timeit.default_timer() - global_time

    print("==============================================================")
    print("Done")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Successful row-scale entries: {ok_count}")
    print(f"Failed row-scale entries: {fail_count}")
    print(f"Total elapsed time: {total_elapsed:.2f} s")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()

