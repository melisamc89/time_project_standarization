#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Representational Similarity Analysis using standardized raw data.

This script loads the standardized time-project dataset and computes RSA
matrices for each row/session.

Main idea
---------
For each row:
    signal: time x neurons

The recording is divided into temporal windows. For each window, we build a
representation vector. Then we compute a window x window similarity matrix.

RSA modes
---------
1) "global_mean":
    Representation of each window = mean activity of each neuron in that window.
    Vector shape: neurons.

2) "position_binned":
    Representation of each window = neuron x position-bin activity map.
    Vector shape after flattening: neurons * position_bins.

    This is recommended for internal drift because it controls better for
    spatial sampling: each time window is represented by its position-binned
    population code.

Outputs
-------
/home/melisamc/Documents/time_project_standarization/RSA_Analysis

Saves:
    - per-row RSA heatmaps
    - per-row pickle files
    - row-level CSV summaries
    - matrix-entry CSV
    - lag-summary CSV
    - learning aggregation with mouse-wise ordered sessions
    - learning bar plots and curve plots
    - lt_rot split into lt and rot using session_name
    - dreadds/dreads/egrin deep vs superficial comparisons
    - general deep/superficial comparisons
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
    "RSA_Analysis"
)

SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
ROW_HEATMAP_DIR = os.path.join(FIG_DIR, "row_heatmaps")
LEARNING_DIR = os.path.join(FIG_DIR, "learning")
LT_ROT_DIR = os.path.join(FIG_DIR, "lt_rot")
DEEP_SUP_DIR = os.path.join(FIG_DIR, "deep_superficial_all_datasets")


RSA_SESSION_CSV = os.path.join(CSV_DIR, "rsa_per_session.csv")
RSA_MATRIX_ENTRIES_CSV = os.path.join(CSV_DIR, "rsa_matrix_entries_per_session.csv")
RSA_LAG_CSV = os.path.join(CSV_DIR, "rsa_lag_summary_per_session.csv")
FAILED_ROWS_CSV = os.path.join(CSV_DIR, "failed_rows.csv")


# =============================================================================
# Parameters
# =============================================================================

MAX_ROWS = None

# Set to None to process all datasets.
# Example:
# DATASET_FILTER = ["learning", "lt_rot", "lt_ephys_mice", "dreadds", "egrin"]
DATASET_FILTER = None

# RSA setting.
# Options:
#   "global_mean", global
#   "position_binned"
RSA_MODE = "global_mean"

# Number of temporal windows in the RSA matrix.
N_WINDOWS = 5

# Requested change: 50 position bins.
N_POSITION_BINS = 50

MIN_SAMPLES_PER_POSITION_BIN = 2

REMOVE_MOV_DIR_ZERO = True

# "trial_normalized":
#     Split each trial into N_WINDOWS and pool equivalent windows across trials.
#
# "global_time":
#     Split the full recording timeline into N_WINDOWS.
WINDOW_MODE = "global_time"

ZSCORE_CELLS = True
MIN_TIMEPOINTS_PER_WINDOW = 3

# Similarity settings.
# Options:
#   "pearson"
#   "cosine"
SIMILARITY_METRIC = "pearson"

# For correlation RSA, [-1, 1] is natural.
HEATMAP_VMIN = -1
HEATMAP_VMAX = 1

# If True, negative similarities are clipped to 0 in saved results.
CLIP_NEGATIVE_SIMILARITY_IN_RESULTS = False

PROCESS_WHOLE_POPULATION = True


# =============================================================================
# Color schemes
# =============================================================================

DEEP_COLOR = "#D4A928"          # gold
SUPERFICIAL_COLOR = "#A020F0"   # purple


def get_learning_gray_palette(n_sessions):
    """
    Return n grayscale colors, darker to lighter.
    """
    if n_sessions <= 1:
        return ["#666666"]

    vals = np.linspace(0.25, 0.80, n_sessions)
    return [str(v) for v in vals]


# =============================================================================
# CSV schemas
# =============================================================================

RSA_SESSION_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "condition",
    "standardized_file_path",
    "row_index",
    "rsa_mode",
    "window_mode",
    "similarity_metric",
    "n_neurons",
    "n_timepoints_original",
    "n_timepoints_after_dir_filter",
    "n_windows",
    "n_position_bins",
    "mean_similarity_all",
    "mean_similarity_offdiag",
] + [
    f"lag_{i}" for i in range(N_WINDOWS)
] + [
    "session_pickle_name",
    "session_pickle_path",
]

RSA_MATRIX_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "condition",
    "row_index",
    "window_i",
    "window_j",
    "lag",
    "rsa_similarity",
]

RSA_LAG_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "condition",
    "row_index",
    "lag",
    "rsa_similarity",
]

FAILED_COLUMNS = [
    "row_index",
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "standardized_file_path",
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
        ROW_HEATMAP_DIR,
        LEARNING_DIR,
        LT_ROT_DIR,
        DEEP_SUP_DIR,
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


def base_name_from_row(row_idx, row):
    return (
        f"row_{row_idx:04d}_"
        f"{sanitize_for_filename(row['dataset_name'])}_"
        f"{sanitize_for_filename(row['mouse_name'])}_"
        f"{sanitize_for_filename(row['session_name'])}_"
        f"{sanitize_for_filename(row['depth_type'])}"
    )


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


def infer_condition(row):
    """
    Infer experimental condition.

    Critical rule for lt_rot:
        condition is determined from session_name, not dataset_name.
    """
    dataset = str(row.get("dataset_name", "")).lower()
    session = str(row.get("session_name", "")).lower()
    path = str(row.get("standardized_file_path", "")).lower()

    # lt_rot-specific: use session_name first.
    if "lt_rot" in dataset or ("lt" in dataset and "rot" in dataset):
        if "rot" in session:
            return "rot"
        if "lt" in session:
            return "lt"

        if "rot" in path:
            return "rot"
        if "lt" in path:
            return "lt"

        return "unknown_lt_rot_condition"

    session_path_text = " ".join([session, path])

    if "dreadds" in dataset or "dreads" in dataset:
        if "cno" in session_path_text:
            return "cno"
        if "saline" in session_path_text:
            return "saline"
        if "veh" in session_path_text or "vehicle" in session_path_text:
            return "vehicle"
        if "pre" in session_path_text:
            return "pre"
        if "post" in session_path_text:
            return "post"
        return "dreadds"

    if "egrin" in dataset:
        if "pre" in session_path_text:
            return "pre"
        if "post" in session_path_text:
            return "post"
        if "baseline" in session_path_text:
            return "baseline"
        if "stim" in session_path_text:
            return "stim"
        return "egrin"

    if "rot" in session:
        return "rot"

    if "lt" in session:
        return "lt"

    if "pre" in session_path_text:
        return "pre"

    if "post" in session_path_text:
        return "post"

    return str(row.get("session_name", "unknown_condition"))


# =============================================================================
# Learning session ordering
# =============================================================================

def extract_learning_datetime_from_session(session_name):
    """
    Extract datetime from learning session names.

    Expected patterns include:
        YYYYMMDD_HHMMSS
        YYYYMMDD-HHMMSS
        YYYYMMDD_HH_MM_SS
        YYYY_MM_DD_HH_MM_SS
        YYYY-MM-DD_HH-MM-SS
    """
    text = str(session_name)

    match = re.search(r"(\d{8})[_\-]?(\d{6})", text)
    if match is not None:
        date_part = match.group(1)
        time_part = match.group(2)

        try:
            return pd.to_datetime(date_part + time_part, format="%Y%m%d%H%M%S")
        except Exception:
            pass

    match = re.search(r"(\d{8})", text)
    if match is not None:
        date_part = match.group(1)

        try:
            return pd.to_datetime(date_part, format="%Y%m%d")
        except Exception:
            pass

    match = re.search(
        r"(\d{4})[_\-](\d{1,2})[_\-](\d{1,2})[_\-](\d{1,2})[_\-](\d{1,2})[_\-](\d{1,2})",
        text,
    )

    if match is not None:
        y, mo, d, h, mi, s = match.groups()

        try:
            return pd.to_datetime(f"{y}-{mo}-{d} {h}:{mi}:{s}")
        except Exception:
            pass

    match = re.search(r"(\d{4})[_\-](\d{1,2})[_\-](\d{1,2})", text)
    if match is not None:
        y, mo, d = match.groups()

        try:
            return pd.to_datetime(f"{y}-{mo}-{d}")
        except Exception:
            pass

    try:
        return pd.to_datetime(text)
    except Exception:
        return pd.NaT


def build_learning_session_mapping_by_mouse(learning_matrix_df):
    """
    Assign session_1, session_2, session_3, ... within each mouse.
    """
    session_df = (
        learning_matrix_df[["mouse_name", "session_name"]]
        .drop_duplicates()
        .copy()
    )

    session_df["session_datetime"] = session_df["session_name"].apply(
        extract_learning_datetime_from_session
    )

    session_df = session_df.sort_values(
        ["mouse_name", "session_datetime", "session_name"],
        na_position="last",
    ).reset_index(drop=True)

    session_df["session_order"] = session_df.groupby("mouse_name").cumcount() + 1
    session_df["learning_session"] = "session_" + session_df["session_order"].astype(str)

    return session_df


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
# RSA computation
# =============================================================================

def zscore_activity_cells(signal):
    """
    Z-score each neuron across time.

    Input:
        signal: time x neurons
    """
    mu = np.nanmean(signal, axis=0, keepdims=True)
    sd = np.nanstd(signal, axis=0, keepdims=True)

    sd[sd == 0] = np.nan

    z = (signal - mu) / sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    return z


def split_indices_equal(indices, n_windows):
    indices = np.asarray(indices)
    return [x for x in np.array_split(indices, n_windows)]


def make_global_window_indices(n_time, n_windows):
    all_idx = np.arange(n_time)
    return split_indices_equal(all_idx, n_windows)


def make_trial_normalized_window_indices(trial_id, n_windows):
    """
    Split every trial into n_windows and pool relative windows across trials.
    """
    trial_id = ensure_1d(trial_id)

    valid_mask = np.isfinite(trial_id)
    valid_trials = pd.unique(trial_id[valid_mask])

    pooled_windows = [[] for _ in range(n_windows)]

    for trial in valid_trials:
        trial_idx = np.where(trial_id == trial)[0]

        if len(trial_idx) < n_windows:
            continue

        trial_windows = split_indices_equal(trial_idx, n_windows)

        for window_idx in range(n_windows):
            pooled_windows[window_idx].extend(trial_windows[window_idx].tolist())

    pooled_windows = [np.asarray(x, dtype=int) for x in pooled_windows]

    return pooled_windows


def safe_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def safe_cosine_similarity(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)

    if nx == 0 or ny == 0:
        return np.nan

    return float(np.dot(x, y) / (nx * ny))


def compute_similarity(x, y):
    if SIMILARITY_METRIC == "pearson":
        return safe_corr(x, y)

    if SIMILARITY_METRIC == "cosine":
        return safe_cosine_similarity(x, y)

    raise ValueError(f"Unknown SIMILARITY_METRIC: {SIMILARITY_METRIC}")


def compute_global_mean_representations(signal, window_indices):
    """
    One vector per window:
        vector = mean activity per neuron.
    """
    reps = []

    for idx in window_indices:
        if len(idx) < MIN_TIMEPOINTS_PER_WINDOW:
            reps.append(np.full(signal.shape[1], np.nan))
        else:
            reps.append(np.nanmean(signal[idx, :], axis=0))

    return reps


def compute_position_binned_representations(signal, position, window_indices):
    """
    One vector per window:
        vector = flattened neuron x position-bin map.

    For each temporal window:
        For each position bin:
            compute mean population activity in that bin.

    Missing bins are filled with the neuron's mean activity inside the window.
    """
    position = ensure_1d(position)

    if len(position) != signal.shape[0]:
        raise ValueError(
            f"Position length {len(position)} does not match signal time "
            f"{signal.shape[0]}"
        )

    pos_min = np.nanmin(position)
    pos_max = np.nanmax(position)

    if not np.isfinite(pos_min) or not np.isfinite(pos_max) or pos_min == pos_max:
        raise ValueError("Invalid position vector for position-binned RSA.")

    bin_edges = np.linspace(pos_min, pos_max, N_POSITION_BINS + 1)

    reps = []

    for idx in window_indices:
        if len(idx) < MIN_TIMEPOINTS_PER_WINDOW:
            reps.append(np.full(signal.shape[1] * N_POSITION_BINS, np.nan))
            continue

        window_signal = signal[idx, :]
        window_pos = position[idx]

        tuning_map = np.full((signal.shape[1], N_POSITION_BINS), np.nan)

        for b in range(N_POSITION_BINS):
            lo = bin_edges[b]
            hi = bin_edges[b + 1]

            if b == N_POSITION_BINS - 1:
                mask = (window_pos >= lo) & (window_pos <= hi)
            else:
                mask = (window_pos >= lo) & (window_pos < hi)

            if np.sum(mask) >= MIN_SAMPLES_PER_POSITION_BIN:
                tuning_map[:, b] = np.nanmean(window_signal[mask, :], axis=0)

        # Fill missing position bins with each neuron's mean in this window.
        neuron_window_mean = np.nanmean(window_signal, axis=0)

        for b in range(N_POSITION_BINS):
            missing = ~np.isfinite(tuning_map[:, b])
            tuning_map[missing, b] = neuron_window_mean[missing]

        reps.append(tuning_map.reshape(-1))

    return reps


def compute_rsa_matrix(signal, behavior):
    """
    Compute window x window RSA matrix.
    """
    if ZSCORE_CELLS:
        signal = zscore_activity_cells(signal)

    n_time = signal.shape[0]

    if WINDOW_MODE == "trial_normalized" and behavior["trial_id"] is not None:
        window_indices = make_trial_normalized_window_indices(
            behavior["trial_id"],
            N_WINDOWS,
        )

        if any(len(idx) < MIN_TIMEPOINTS_PER_WINDOW for idx in window_indices):
            print(
                "[WARNING] trial_normalized windows have too few samples. "
                "Falling back to global_time windows."
            )
            window_indices = make_global_window_indices(n_time, N_WINDOWS)

    else:
        window_indices = make_global_window_indices(n_time, N_WINDOWS)

    if RSA_MODE == "global_mean":
        representations = compute_global_mean_representations(signal, window_indices)

    elif RSA_MODE == "position_binned":
        representations = compute_position_binned_representations(
            signal=signal,
            position=behavior["pos"],
            window_indices=window_indices,
        )

    else:
        raise ValueError(f"Unknown RSA_MODE: {RSA_MODE}")

    rsa_matrix = np.full((N_WINDOWS, N_WINDOWS), np.nan)

    for i in range(N_WINDOWS):
        for j in range(N_WINDOWS):
            rsa_matrix[i, j] = compute_similarity(representations[i], representations[j])

    if CLIP_NEGATIVE_SIMILARITY_IN_RESULTS:
        rsa_matrix = np.where(
            np.isfinite(rsa_matrix),
            np.clip(rsa_matrix, 0.0, 1.0),
            np.nan,
        )

    return rsa_matrix, representations, window_indices


def matrix_to_lag_summary(mat):
    """
    Average matrix entries by distance from diagonal.
    """
    out = {}

    n = mat.shape[0]

    for lag in range(n):
        vals = []

        for i in range(n):
            for j in range(n):
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

def savefig_png_svg(fig, out_base):
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_rsa_matrix(mat, title, out_base):
    plot_mat = np.clip(mat, HEATMAP_VMIN, HEATMAP_VMAX)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))

    im = ax.imshow(
        plot_mat,
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        origin="upper",
        aspect="equal",
    )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Window")
    ax.set_ylabel("Window")

    ax.set_xticks(np.arange(N_WINDOWS))
    ax.set_yticks(np.arange(N_WINDOWS))
    ax.set_xticklabels([str(i + 1) for i in range(N_WINDOWS)], fontsize=6)
    ax.set_yticklabels([str(i + 1) for i in range(N_WINDOWS)], fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Representational similarity")

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_learning_session_lag_barplot(df, title, out_base, session_order):
    """
    Learning bar plot:
        x-axis = lag
        bar color = ordered learning session, using gray tones.
    """
    if len(df) == 0:
        return

    lags = sorted([lag for lag in df["lag"].dropna().unique() if lag > 0])
    sessions = session_order

    means = (
        df.groupby(["learning_session", "lag"], observed=True)["rsa_similarity"]
        .mean()
        .reset_index()
    )

    sems = (
        df.groupby(["learning_session", "lag"], observed=True)["rsa_similarity"]
        .sem()
        .reset_index()
        .rename(columns={"rsa_similarity": "sem"})
    )

    stats = means.merge(sems, on=["learning_session", "lag"], how="left")

    x = np.arange(len(lags))
    width = 0.85 / max(len(sessions), 1)

    gray_colors = get_learning_gray_palette(len(sessions))

    fig, ax = plt.subplots(figsize=(11.0, 4.8))

    for k, session in enumerate(sessions):
        y = []
        yerr = []

        for lag in lags:
            sub = stats[
                (stats["learning_session"] == session)
                & (stats["lag"] == lag)
            ]

            if len(sub) == 0:
                y.append(np.nan)
                yerr.append(np.nan)
            else:
                y.append(float(sub["rsa_similarity"].iloc[0]))
                yerr.append(float(sub["sem"].iloc[0]))

        offset = (k - (len(sessions) - 1) / 2) * width

        ax.bar(
            x + offset,
            y,
            width=width,
            yerr=yerr,
            capsize=2,
            color=gray_colors[k],
            edgecolor="black",
            linewidth=0.7,
            label=session,
        )

    ax.set_title(title)
    ax.set_xlabel("|Δ window|")
    ax.set_ylabel("Representational similarity")
    ax.set_ylim(HEATMAP_VMIN, HEATMAP_VMAX)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(lag)) for lag in lags])
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_learning_session_lag_curves(df, title, out_base, session_order):
    """
    Learning curve plot:
        x-axis = lag
        line color = ordered learning session, using gray tones.
    """
    if len(df) == 0:
        return

    lags = sorted([lag for lag in df["lag"].dropna().unique() if lag > 0])
    sessions = session_order

    means = (
        df.groupby(["learning_session", "lag"], observed=True)["rsa_similarity"]
        .mean()
        .reset_index()
    )

    sems = (
        df.groupby(["learning_session", "lag"], observed=True)["rsa_similarity"]
        .sem()
        .reset_index()
        .rename(columns={"rsa_similarity": "sem"})
    )

    stats = means.merge(sems, on=["learning_session", "lag"], how="left")

    gray_colors = get_learning_gray_palette(len(sessions))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for k, session in enumerate(sessions):
        y = []
        yerr = []

        for lag in lags:
            sub = stats[
                (stats["learning_session"] == session)
                & (stats["lag"] == lag)
            ]

            if len(sub) == 0:
                y.append(np.nan)
                yerr.append(np.nan)
            else:
                y.append(float(sub["rsa_similarity"].iloc[0]))
                yerr.append(float(sub["sem"].iloc[0]))

        y = np.asarray(y, dtype=float)
        yerr = np.asarray(yerr, dtype=float)

        ax.plot(
            lags,
            y,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=gray_colors[k],
            label=session,
        )

        ax.fill_between(
            lags,
            y - yerr,
            y + yerr,
            color=gray_colors[k],
            alpha=0.20,
            linewidth=0,
        )

    ax.set_title(title)
    ax.set_xlabel("|Δ window|")
    ax.set_ylabel("Representational similarity")
    ax.set_ylim(HEATMAP_VMIN, HEATMAP_VMAX)
    ax.set_xticks(lags)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_deep_sup_lag_barplot(df, title, out_base):
    """
    Deep vs superficial bar plot using requested color scheme:
        deep = gold
        superficial = purple
    """
    df = df[df["depth_type"].isin(["deep", "superficial"])].copy()

    if len(df) == 0:
        return

    lags = sorted([lag for lag in df["lag"].dropna().unique() if lag > 0])
    depths = ["deep", "superficial"]

    depth_colors = {
        "deep": DEEP_COLOR,
        "superficial": SUPERFICIAL_COLOR,
    }

    means = (
        df.groupby(["depth_type", "lag"], observed=True)["rsa_similarity"]
        .mean()
        .reset_index()
    )

    sems = (
        df.groupby(["depth_type", "lag"], observed=True)["rsa_similarity"]
        .sem()
        .reset_index()
        .rename(columns={"rsa_similarity": "sem"})
    )

    stats = means.merge(sems, on=["depth_type", "lag"], how="left")

    x = np.arange(len(lags))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for k, depth in enumerate(depths):
        y = []
        yerr = []

        for lag in lags:
            sub = stats[
                (stats["depth_type"] == depth)
                & (stats["lag"] == lag)
            ]

            if len(sub) == 0:
                y.append(np.nan)
                yerr.append(np.nan)
            else:
                y.append(float(sub["rsa_similarity"].iloc[0]))
                yerr.append(float(sub["sem"].iloc[0]))

        offset = (k - 0.5) * width

        ax.bar(
            x + offset,
            y,
            width=width,
            yerr=yerr,
            capsize=2,
            color=depth_colors[depth],
            edgecolor="black",
            linewidth=0.8,
            label=depth,
        )

    ax.set_title(title)
    ax.set_xlabel("|Δ window|")
    ax.set_ylabel("Representational similarity")
    ax.set_ylim(HEATMAP_VMIN, HEATMAP_VMAX)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(lag)) for lag in lags])
    ax.legend(frameon=False)

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_deep_sup_lag_curves(df, title, out_base):
    """
    Deep vs superficial curve plot:
        deep = gold
        superficial = purple
    """
    df = df[df["depth_type"].isin(["deep", "superficial"])].copy()

    if len(df) == 0:
        return

    lags = sorted([lag for lag in df["lag"].dropna().unique() if lag > 0])
    depths = ["deep", "superficial"]

    depth_colors = {
        "deep": DEEP_COLOR,
        "superficial": SUPERFICIAL_COLOR,
    }

    means = (
        df.groupby(["depth_type", "lag"], observed=True)["rsa_similarity"]
        .mean()
        .reset_index()
    )

    sems = (
        df.groupby(["depth_type", "lag"], observed=True)["rsa_similarity"]
        .sem()
        .reset_index()
        .rename(columns={"rsa_similarity": "sem"})
    )

    stats = means.merge(sems, on=["depth_type", "lag"], how="left")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for depth in depths:
        y = []
        yerr = []

        for lag in lags:
            sub = stats[
                (stats["depth_type"] == depth)
                & (stats["lag"] == lag)
            ]

            if len(sub) == 0:
                y.append(np.nan)
                yerr.append(np.nan)
            else:
                y.append(float(sub["rsa_similarity"].iloc[0]))
                yerr.append(float(sub["sem"].iloc[0]))

        y = np.asarray(y, dtype=float)
        yerr = np.asarray(yerr, dtype=float)

        ax.plot(
            lags,
            y,
            marker="o",
            linewidth=2.5,
            markersize=6,
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

    ax.set_title(title)
    ax.set_xlabel("|Δ window|")
    ax.set_ylabel("Representational similarity")
    ax.set_ylim(HEATMAP_VMIN, HEATMAP_VMAX)
    ax.set_xticks(lags)
    ax.legend(frameon=False)

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


# =============================================================================
# Row processing
# =============================================================================

def process_row(row_idx, row):
    signal, behavior = load_standardized_row(row_idx, row)

    n_timepoints_original = signal.shape[0]
    n_neurons = signal.shape[1]

    if REMOVE_MOV_DIR_ZERO:
        valid_idx = behavior["mov_dir"] != 0
    else:
        valid_idx = np.ones(n_timepoints_original, dtype=bool)

    signal_valid = signal[valid_idx, :]

    behavior_valid = {}

    for key, value in behavior.items():
        if value is None:
            behavior_valid[key] = None
        else:
            behavior_valid[key] = value[valid_idx]

    n_timepoints_after_dir_filter = signal_valid.shape[0]

    if n_timepoints_after_dir_filter < N_WINDOWS * MIN_TIMEPOINTS_PER_WINDOW:
        raise ValueError(
            f"Too few timepoints after filtering: "
            f"{n_timepoints_after_dir_filter}"
        )

    rsa_matrix, representations, window_indices = compute_rsa_matrix(
        signal_valid,
        behavior_valid,
    )

    lag_summary = matrix_to_lag_summary(rsa_matrix)
    mean_similarity_all = float(np.nanmean(rsa_matrix))
    mean_similarity_offdiag = mean_offdiagonal(rsa_matrix)

    dataset_name = row["dataset_name"]
    mouse_name = row["mouse_name"]
    session_name = row["session_name"]
    depth_type = normalize_depth(row["depth_type"])
    condition = infer_condition(row)

    session_base = base_name_from_row(row_idx, row) + "_rsa"
    session_pickle_name = session_base + ".pkl"
    session_pickle_path = os.path.join(SESSION_PKL_DIR, session_pickle_name)

    session_payload = {
        "metadata": {
            "dataset_name": dataset_name,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "depth_type": depth_type,
            "condition": condition,
            "row_index": row_idx,
            "standardized_file_path": row["standardized_file_path"],
            "rsa_mode": RSA_MODE,
            "window_mode": WINDOW_MODE,
            "similarity_metric": SIMILARITY_METRIC,
            "n_neurons": n_neurons,
            "n_timepoints_original": n_timepoints_original,
            "n_timepoints_after_dir_filter": n_timepoints_after_dir_filter,
            "n_windows": N_WINDOWS,
            "n_position_bins": N_POSITION_BINS,
            "zscore_cells": ZSCORE_CELLS,
            "remove_mov_dir_zero": REMOVE_MOV_DIR_ZERO,
            "clip_negative_similarity": CLIP_NEGATIVE_SIMILARITY_IN_RESULTS,
        },
        "indices": {
            "valid_idx_mov_dir_nonzero": valid_idx,
            "window_indices_after_filter": window_indices,
        },
        "rsa": {
            "matrix": rsa_matrix,
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
        "condition": condition,
        "standardized_file_path": row["standardized_file_path"],
        "row_index": row_idx,
        "rsa_mode": RSA_MODE,
        "window_mode": WINDOW_MODE,
        "similarity_metric": SIMILARITY_METRIC,
        "n_neurons": n_neurons,
        "n_timepoints_original": n_timepoints_original,
        "n_timepoints_after_dir_filter": n_timepoints_after_dir_filter,
        "n_windows": N_WINDOWS,
        "n_position_bins": N_POSITION_BINS,
        "mean_similarity_all": mean_similarity_all,
        "mean_similarity_offdiag": mean_similarity_offdiag,
    }

    for lag in range(N_WINDOWS):
        session_row[f"lag_{lag}"] = lag_summary.get(lag, np.nan)

    session_row["session_pickle_name"] = session_pickle_name
    session_row["session_pickle_path"] = session_pickle_path

    matrix_rows = []

    for i in range(N_WINDOWS):
        for j in range(N_WINDOWS):
            matrix_rows.append(
                {
                    "dataset_name": dataset_name,
                    "mouse_name": mouse_name,
                    "session_name": session_name,
                    "depth_type": depth_type,
                    "condition": condition,
                    "row_index": row_idx,
                    "window_i": i + 1,
                    "window_j": j + 1,
                    "lag": abs(i - j),
                    "rsa_similarity": rsa_matrix[i, j],
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
                "condition": condition,
                "row_index": row_idx,
                "lag": lag,
                "rsa_similarity": value,
            }
        )

    heatmap_title = (
        f"{dataset_name}\n"
        f"{mouse_name}, {session_name}, {depth_type}\n"
        f"RSA: {RSA_MODE}, bins={N_POSITION_BINS}, {SIMILARITY_METRIC}"
    )

    heatmap_base = os.path.join(
        ROW_HEATMAP_DIR,
        session_base + "_matrix",
    )

    plot_rsa_matrix(
        rsa_matrix,
        title=heatmap_title,
        out_base=heatmap_base,
    )

    print(
        f"Row {row_idx} | dataset={dataset_name} | mouse={mouse_name} | "
        f"session={session_name} | depth={depth_type} | condition={condition} | "
        f"neurons={n_neurons} | mean_offdiag={mean_similarity_offdiag:.4f}"
    )

    return session_row, matrix_rows, lag_rows


# =============================================================================
# Aggregation helpers
# =============================================================================

def aggregate_mean_matrix_from_entries(df):
    mat = np.full((N_WINDOWS, N_WINDOWS), np.nan)

    for i in range(N_WINDOWS):
        for j in range(N_WINDOWS):
            sub = df[
                (df["window_i"] == i + 1)
                & (df["window_j"] == j + 1)
            ]

            if len(sub) == 0:
                mat[i, j] = np.nan
            else:
                mat[i, j] = np.nanmean(sub["rsa_similarity"].values)

    return mat


def run_learning_aggregation(matrix_df, lag_df):
    learning_mask = (
        matrix_df["dataset_name"].astype(str).str.lower().str.contains("learning")
        | matrix_df["session_name"].astype(str).str.lower().str.contains("learning")
    )

    learning_matrix_df = matrix_df[learning_mask].copy()

    if len(learning_matrix_df) == 0:
        print("No learning rows found. Skipping learning aggregation.")
        return

    learning_lag_df = lag_df[
        lag_df["row_index"].isin(learning_matrix_df["row_index"].unique())
    ].copy()

    mapping_df = build_learning_session_mapping_by_mouse(learning_matrix_df)

    learning_matrix_df = learning_matrix_df.merge(
        mapping_df[
            [
                "mouse_name",
                "session_name",
                "session_datetime",
                "session_order",
                "learning_session",
            ]
        ],
        on=["mouse_name", "session_name"],
        how="left",
    )

    learning_lag_df = learning_lag_df.merge(
        mapping_df[
            [
                "mouse_name",
                "session_name",
                "session_datetime",
                "session_order",
                "learning_session",
            ]
        ],
        on=["mouse_name", "session_name"],
        how="left",
    )

    mapping_df.to_csv(
        os.path.join(CSV_DIR, "learning_session_order_mapping_by_mouse.csv"),
        index=False,
    )

    learning_matrix_df.to_csv(
        os.path.join(CSV_DIR, "learning_matrix_entries_with_mousewise_ordered_sessions.csv"),
        index=False,
    )

    learning_lag_df.to_csv(
        os.path.join(CSV_DIR, "learning_lag_summary_with_mousewise_ordered_sessions.csv"),
        index=False,
    )

    max_order = int(np.nanmax(mapping_df["session_order"].values))
    ordered_sessions = [f"session_{i}" for i in range(1, max_order + 1)]

    for ordered_session in ordered_sessions:
        sub = learning_matrix_df[
            learning_matrix_df["learning_session"] == ordered_session
        ].copy()

        if len(sub) == 0:
            continue

        mean_mat = aggregate_mean_matrix_from_entries(sub)

        n_mice = sub["mouse_name"].nunique()
        n_rows = sub["row_index"].nunique()

        out_base = os.path.join(
            LEARNING_DIR,
            f"learning_{ordered_session}_mean_rsa_matrix_across_mice",
        )

        plot_rsa_matrix(
            mean_mat,
            title=(
                f"Learning {ordered_session}\n"
                f"mean RSA across mice | mice={n_mice}, rows={n_rows}"
            ),
            out_base=out_base,
        )

    plot_learning_session_lag_barplot(
        learning_lag_df,
        title="Learning dataset: RSA similarity by session and temporal distance",
        out_base=os.path.join(
            LEARNING_DIR,
            "learning_rsa_lag_barplot_by_mousewise_ordered_session",
        ),
        session_order=ordered_sessions,
    )

    plot_learning_session_lag_curves(
        learning_lag_df,
        title="Learning dataset: RSA similarity curves by session",
        out_base=os.path.join(
            LEARNING_DIR,
            "learning_rsa_lag_curves_by_mousewise_ordered_session",
        ),
        session_order=ordered_sessions,
    )

    learning_mouse_session_lag_summary = (
        learning_lag_df
        .groupby(
            [
                "mouse_name",
                "learning_session",
                "session_order",
                "lag",
            ],
            observed=True,
        )["rsa_similarity"]
        .mean()
        .reset_index()
    )

    learning_mouse_session_lag_summary.to_csv(
        os.path.join(CSV_DIR, "learning_mouse_session_rsa_lag_summary.csv"),
        index=False,
    )

    print("Learning RSA aggregation completed using mouse-wise session ordering.")


def run_lt_rot_aggregation(matrix_df, lag_df):
    ltrot_mask = (
        matrix_df["dataset_name"]
        .astype(str)
        .str.lower()
        .str.contains("lt_rot")
    )

    ltrot_matrix_df = matrix_df[ltrot_mask].copy()

    if len(ltrot_matrix_df) == 0:
        print("No lt_rot rows found. Skipping lt_rot aggregation.")
        return

    ltrot_matrix_df = ltrot_matrix_df[
        ltrot_matrix_df["depth_type"].isin(["deep", "superficial"])
    ].copy()

    ltrot_matrix_df = ltrot_matrix_df[
        ltrot_matrix_df["condition"].isin(["lt", "rot"])
    ].copy()

    if len(ltrot_matrix_df) == 0:
        print("lt_rot rows found, but no rows were classified as lt or rot.")
        print("Check session_name values in the CSV.")
        return

    ltrot_lag_df = lag_df[
        lag_df["row_index"].isin(ltrot_matrix_df["row_index"].unique())
        & lag_df["depth_type"].isin(["deep", "superficial"])
        & lag_df["condition"].isin(["lt", "rot"])
    ].copy()

    ltrot_matrix_df.to_csv(
        os.path.join(CSV_DIR, "lt_rot_rsa_matrix_entries_lt_and_rot_separated.csv"),
        index=False,
    )

    ltrot_lag_df.to_csv(
        os.path.join(CSV_DIR, "lt_rot_rsa_lag_summary_lt_and_rot_separated.csv"),
        index=False,
    )

    for condition in ["lt", "rot"]:
        condition_matrix_df = ltrot_matrix_df[
            ltrot_matrix_df["condition"] == condition
        ].copy()

        condition_lag_df = ltrot_lag_df[
            ltrot_lag_df["condition"] == condition
        ].copy()

        if len(condition_matrix_df) == 0:
            print(f"No lt_rot {condition} rows found.")
            continue

        print(
            f"lt_rot {condition}: "
            f"{condition_matrix_df['mouse_name'].nunique()} mice, "
            f"{condition_matrix_df['row_index'].nunique()} rows"
        )

        for depth in ["deep", "superficial"]:
            sub = condition_matrix_df[
                condition_matrix_df["depth_type"] == depth
            ].copy()

            if len(sub) == 0:
                print(f"No lt_rot {condition} {depth} rows found.")
                continue

            mean_mat = aggregate_mean_matrix_from_entries(sub)

            n_mice = sub["mouse_name"].nunique()
            n_rows = sub["row_index"].nunique()

            out_base = os.path.join(
                LT_ROT_DIR,
                f"lt_rot_{condition}_{depth}_mean_{N_WINDOWS}x{N_WINDOWS}_rsa_matrix",
            )

            plot_rsa_matrix(
                mean_mat,
                title=(
                    f"lt_rot {condition}, {depth}\n"
                    f"mean RSA across mice | mice={n_mice}, rows={n_rows}"
                ),
                out_base=out_base,
            )

        plot_deep_sup_lag_barplot(
            condition_lag_df,
            title=f"lt_rot {condition}: RSA deep vs superficial",
            out_base=os.path.join(
                LT_ROT_DIR,
                f"lt_rot_{condition}_deep_vs_superficial_rsa_lag_barplot",
            ),
        )

        plot_deep_sup_lag_curves(
            condition_lag_df,
            title=f"lt_rot {condition}: RSA deep vs superficial curves",
            out_base=os.path.join(
                LT_ROT_DIR,
                f"lt_rot_{condition}_deep_vs_superficial_rsa_lag_curves",
            ),
        )

    print("lt_rot RSA aggregation completed with lt and rot separated by session_name.")


def run_named_dataset_deep_sup_aggregation(matrix_df, lag_df, dataset_keyword, output_subdir_name):
    out_dir = os.path.join(FIG_DIR, output_subdir_name)
    ensure_dir(out_dir)

    dataset_mask = (
        matrix_df["dataset_name"]
        .astype(str)
        .str.lower()
        .str.contains(dataset_keyword.lower())
    )

    dataset_matrix_df = matrix_df[dataset_mask].copy()

    if len(dataset_matrix_df) == 0:
        print(f"No {dataset_keyword} rows found. Skipping.")
        return

    dataset_matrix_df = dataset_matrix_df[
        dataset_matrix_df["depth_type"].isin(["deep", "superficial"])
    ].copy()

    dataset_lag_df = lag_df[
        lag_df["row_index"].isin(dataset_matrix_df["row_index"].unique())
        & lag_df["depth_type"].isin(["deep", "superficial"])
    ].copy()

    if len(dataset_lag_df) == 0:
        print(f"{dataset_keyword} rows found, but no deep/superficial entries.")
        return

    dataset_matrix_df.to_csv(
        os.path.join(
            CSV_DIR,
            f"{output_subdir_name}_rsa_matrix_entries_deep_superficial.csv",
        ),
        index=False,
    )

    dataset_lag_df.to_csv(
        os.path.join(
            CSV_DIR,
            f"{output_subdir_name}_rsa_lag_summary_deep_superficial.csv",
        ),
        index=False,
    )

    conditions = sorted(dataset_matrix_df["condition"].dropna().unique())

    for condition in conditions:
        condition_matrix_df = dataset_matrix_df[
            dataset_matrix_df["condition"] == condition
        ].copy()

        condition_lag_df = dataset_lag_df[
            dataset_lag_df["condition"] == condition
        ].copy()

        if len(condition_matrix_df) == 0:
            continue

        for depth in ["deep", "superficial"]:
            sub = condition_matrix_df[
                condition_matrix_df["depth_type"] == depth
            ].copy()

            if len(sub) == 0:
                continue

            mean_mat = aggregate_mean_matrix_from_entries(sub)

            n_mice = sub["mouse_name"].nunique()
            n_rows = sub["row_index"].nunique()

            out_base = os.path.join(
                out_dir,
                f"{output_subdir_name}_{sanitize_for_filename(condition)}_"
                f"{depth}_mean_{N_WINDOWS}x{N_WINDOWS}_rsa_matrix",
            )

            plot_rsa_matrix(
                mean_mat,
                title=(
                    f"{output_subdir_name} {condition}, {depth}\n"
                    f"mean RSA across mice | mice={n_mice}, rows={n_rows}"
                ),
                out_base=out_base,
            )

        plot_deep_sup_lag_barplot(
            condition_lag_df,
            title=f"{output_subdir_name} {condition}: RSA deep vs superficial",
            out_base=os.path.join(
                out_dir,
                f"{output_subdir_name}_{sanitize_for_filename(condition)}_"
                f"deep_vs_superficial_rsa_lag_barplot",
            ),
        )

        plot_deep_sup_lag_curves(
            condition_lag_df,
            title=f"{output_subdir_name} {condition}: RSA deep vs superficial curves",
            out_base=os.path.join(
                out_dir,
                f"{output_subdir_name}_{sanitize_for_filename(condition)}_"
                f"deep_vs_superficial_rsa_lag_curves",
            ),
        )

    print(f"{dataset_keyword} RSA aggregation completed.")


def run_all_deep_sup_aggregation(matrix_df, lag_df):
    non_learning_mask = ~(
        matrix_df["dataset_name"].astype(str).str.lower().str.contains("learning")
        | matrix_df["session_name"].astype(str).str.lower().str.contains("learning")
    )

    ds_matrix_df = matrix_df[
        non_learning_mask
        & matrix_df["depth_type"].isin(["deep", "superficial"])
    ].copy()

    if len(ds_matrix_df) == 0:
        print("No non-learning deep/superficial rows found.")
        return

    ds_lag_df = lag_df[
        lag_df["row_index"].isin(ds_matrix_df["row_index"].unique())
        & lag_df["depth_type"].isin(["deep", "superficial"])
    ].copy()

    ds_matrix_df.to_csv(
        os.path.join(CSV_DIR, "all_deep_sup_rsa_matrix_entries.csv"),
        index=False,
    )

    ds_lag_df.to_csv(
        os.path.join(CSV_DIR, "all_deep_sup_rsa_lag_summary.csv"),
        index=False,
    )

    groups = (
        ds_matrix_df[["dataset_name", "condition"]]
        .drop_duplicates()
        .sort_values(["dataset_name", "condition"])
    )

    for _, group_row in groups.iterrows():
        dataset_name = group_row["dataset_name"]
        condition = group_row["condition"]

        group_matrix_df = ds_matrix_df[
            (ds_matrix_df["dataset_name"] == dataset_name)
            & (ds_matrix_df["condition"] == condition)
        ]

        group_lag_df = ds_lag_df[
            (ds_lag_df["dataset_name"] == dataset_name)
            & (ds_lag_df["condition"] == condition)
        ]

        for depth in ["deep", "superficial"]:
            sub = group_matrix_df[
                group_matrix_df["depth_type"] == depth
            ]

            if len(sub) == 0:
                continue

            mean_mat = aggregate_mean_matrix_from_entries(sub)

            out_base = os.path.join(
                DEEP_SUP_DIR,
                f"{sanitize_for_filename(dataset_name)}_"
                f"{sanitize_for_filename(condition)}_"
                f"{depth}_mean_{N_WINDOWS}x{N_WINDOWS}_rsa_matrix",
            )

            plot_rsa_matrix(
                mean_mat,
                title=f"{dataset_name} {condition}, {depth}",
                out_base=out_base,
            )

        plot_deep_sup_lag_barplot(
            group_lag_df,
            title=f"{dataset_name} {condition}: RSA deep vs superficial",
            out_base=os.path.join(
                DEEP_SUP_DIR,
                f"{sanitize_for_filename(dataset_name)}_"
                f"{sanitize_for_filename(condition)}_"
                f"deep_vs_superficial_rsa_lag_barplot",
            ),
        )

        plot_deep_sup_lag_curves(
            group_lag_df,
            title=f"{dataset_name} {condition}: RSA deep vs superficial curves",
            out_base=os.path.join(
                DEEP_SUP_DIR,
                f"{sanitize_for_filename(dataset_name)}_"
                f"{sanitize_for_filename(condition)}_"
                f"deep_vs_superficial_rsa_lag_curves",
            ),
        )

    print("All deep/superficial RSA aggregation completed.")


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

    if DATASET_FILTER is not None:
        df = df[df["dataset_name"].isin(DATASET_FILTER)].copy()

    if not PROCESS_WHOLE_POPULATION:
        df = df[
            ~df["depth_type"].astype(str).str.lower().str.contains("whole")
        ].copy()

    df = df.reset_index(drop=True)

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    init_csv(RSA_SESSION_CSV, RSA_SESSION_COLUMNS)
    init_csv(RSA_MATRIX_ENTRIES_CSV, RSA_MATRIX_COLUMNS)
    init_csv(RSA_LAG_CSV, RSA_LAG_COLUMNS)
    init_csv(FAILED_ROWS_CSV, FAILED_COLUMNS)

    log_path = os.path.join(LOG_DIR, "rsa_analysis_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Representational Similarity Analysis")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Rows to process: {len(df)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Session PKL dir: {SESSION_PKL_DIR}")
    print(f"RSA mode: {RSA_MODE}")
    print(f"Window mode: {WINDOW_MODE}")
    print(f"N windows: {N_WINDOWS}")
    print(f"N position bins: {N_POSITION_BINS}")
    print(f"Similarity metric: {SIMILARITY_METRIC}")
    print(f"Remove mov_dir == 0: {REMOVE_MOV_DIR_ZERO}")
    print(f"Z-score cells: {ZSCORE_CELLS}")
    print(f"Clip negative similarity in results: {CLIP_NEGATIVE_SIMILARITY_IN_RESULTS}")
    print("")

    ok_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        local_time = timeit.default_timer()

        try:
            session_row, matrix_rows, lag_rows = process_row(row_idx, row)

            append_csv_row(RSA_SESSION_CSV, RSA_SESSION_COLUMNS, session_row)
            append_many_csv_rows(RSA_MATRIX_ENTRIES_CSV, RSA_MATRIX_COLUMNS, matrix_rows)
            append_many_csv_rows(RSA_LAG_CSV, RSA_LAG_COLUMNS, lag_rows)

            ok_count += 1

            elapsed = timeit.default_timer() - local_time
            print(f"[OK] Row {row_idx} finished in {elapsed:.2f} s\n")

        except Exception as e:
            fail_count += 1

            failed_row = {
                "row_index": row_idx,
                "dataset_name": row.get("dataset_name", ""),
                "mouse_name": row.get("mouse_name", ""),
                "session_name": row.get("session_name", ""),
                "depth_type": row.get("depth_type", ""),
                "standardized_file_path": row.get("standardized_file_path", ""),
                "error": str(e),
            }

            append_csv_row(FAILED_ROWS_CSV, FAILED_COLUMNS, failed_row)

            print(f"[ERROR] Row {row_idx}: {e}\n")

    print("==============================================================")
    print("Running aggregations")
    print("==============================================================")

    matrix_df = pd.read_csv(RSA_MATRIX_ENTRIES_CSV)
    lag_df = pd.read_csv(RSA_LAG_CSV)

    if len(matrix_df) > 0:
        run_learning_aggregation(matrix_df, lag_df)

        run_lt_rot_aggregation(matrix_df, lag_df)

        run_named_dataset_deep_sup_aggregation(
            matrix_df,
            lag_df,
            dataset_keyword="dreadds",
            output_subdir_name="dreadds",
        )

        run_named_dataset_deep_sup_aggregation(
            matrix_df,
            lag_df,
            dataset_keyword="dreads",
            output_subdir_name="dreads",
        )

        run_named_dataset_deep_sup_aggregation(
            matrix_df,
            lag_df,
            dataset_keyword="egrin",
            output_subdir_name="egrin",
        )

        run_all_deep_sup_aggregation(matrix_df, lag_df)

    else:
        print("No matrix entries were generated. Skipping aggregations.")

    total_elapsed = timeit.default_timer() - global_time

    print("==============================================================")
    print("Done")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Successful entries: {ok_count}")
    print(f"Failed entries: {fail_count}")
    print(f"Total elapsed time: {total_elapsed:.2f} s")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()