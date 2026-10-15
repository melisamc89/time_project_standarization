#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Population Vector Drift Analysis using standardized raw data.

This script:
1) Reads:
   /home/melisamc/Documents/time_project_standarization/data/metadata/master_dataset_index.csv

2) For each row:
   - Loads row["standardized_file_path"]
   - Extracts:
        signal      : time x neurons
        pos
        mov_dir
        speed
        time
        trial_id    optional
        inner_time  optional
   - Removes mov_dir == 0
   - Splits trial/session into 10 chunks
   - Computes one population vector per chunk
   - Computes a 10 x 10 PV correlation matrix

3) Saves:
   - Per-row/session heatmaps
   - Per-row/session pickle
   - CSV summaries
   - Learning session mean matrices using mouse-wise session ordering
   - Learning lag bar plot with gray color scheme
   - lt_rot separated into lt and rot
   - dreadds/dreads/egrin deep vs superficial analyses
   - General deep/superficial analysis for all non-learning datasets

Output:
   /home/melisamc/Documents/time_project_standarization/PV_Drift_Analysis
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
    "PV_Drift_Analysis"
)

SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
ROW_HEATMAP_DIR = os.path.join(FIG_DIR, "row_heatmaps")
LEARNING_DIR = os.path.join(FIG_DIR, "learning")
LT_ROT_DIR = os.path.join(FIG_DIR, "lt_rot")
DEEP_SUP_DIR = os.path.join(FIG_DIR, "deep_superficial_all_datasets")


PV_SESSION_CSV = os.path.join(CSV_DIR, "pv_drift_per_session.csv")
PV_MATRIX_ENTRIES_CSV = os.path.join(CSV_DIR, "pv_matrix_entries_per_session.csv")
PV_LAG_CSV = os.path.join(CSV_DIR, "pv_lag_summary_per_session.csv")
FAILED_ROWS_CSV = os.path.join(CSV_DIR, "failed_rows.csv")


# =============================================================================
# Parameters
# =============================================================================

MAX_ROWS = None

# Set to None to process all datasets.
# Example:
# DATASET_FILTER = ["learning", "lt_rot", "lt_ephys_mice", "dreadds", "egrin"]
DATASET_FILTER = None

N_CHUNKS = 5

REMOVE_MOV_DIR_ZERO = True

# "trial_normalized":
#   Split each trial into 10 chunks, then pool equivalent relative chunks.
# "global_time":
#   Split the whole recording timeline into 10 chunks.
CHUNK_MODE = "global_time"

ZSCORE_CELLS = True
MIN_TIMEPOINTS_PER_CHUNK = 3

HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 1.0

# Clip negative correlations to 0 so matrices are directly comparable in [0,1].
CLIP_NEGATIVE_CORRELATIONS_IN_RESULTS = True

PROCESS_WHOLE_POPULATION = True


# =============================================================================
# Color schemes
# =============================================================================

# Deep vs superficial colors: matching the style you showed
DEEP_COLOR = "#D4A928"          # gold
SUPERFICIAL_COLOR = "#A020F0"   # purple

# Learning gray tones
def get_learning_gray_palette(n_sessions):
    """
    Return n grayscale colors, darker to lighter.
    """
    if n_sessions <= 1:
        return ["#666666"]

    # Avoid extreme black/white for visibility
    vals = np.linspace(0.25, 0.80, n_sessions)
    return [str(v) for v in vals]  # matplotlib grayscale strings


# =============================================================================
# CSV schemas
# =============================================================================

PV_SESSION_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "condition",
    "standardized_file_path",
    "row_index",
    "chunk_mode",
    "n_neurons",
    "n_timepoints_original",
    "n_timepoints_after_dir_filter",
    "n_chunks",
    "mean_corr_all",
    "mean_corr_offdiag",
] + [
    f"lag_{i}" for i in range(N_CHUNKS)
] + [
    "session_pickle_name",
    "session_pickle_path",
]

PV_MATRIX_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "condition",
    "row_index",
    "chunk_i",
    "chunk_j",
    "lag",
    "pv_corr",
]

PV_LAG_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "condition",
    "row_index",
    "lag",
    "pv_corr",
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

    Critical fix for lt_rot:
        condition is determined from session_name, not dataset_name.
    """
    dataset = str(row.get("dataset_name", "")).lower()
    session = str(row.get("session_name", "")).lower()
    path = str(row.get("standardized_file_path", "")).lower()

    # -------------------------------------------------------------------------
    # lt_rot-specific: use session_name only
    # -------------------------------------------------------------------------
    if "lt_rot" in dataset or ("lt" in dataset and "rot" in dataset):
        if "rot" in session:
            return "rot"
        if "lt" in session:
            return "lt"

        # fallback only if session_name failed
        if "rot" in path:
            return "rot"
        if "lt" in path:
            return "lt"

        return "unknown_lt_rot_condition"

    session_path_text = " ".join([session, path])

    # -------------------------------------------------------------------------
    # dreadds / dreads
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # egrin
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # generic fallback
    # -------------------------------------------------------------------------
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

    session_df["session_order"] = (
        session_df.groupby("mouse_name").cumcount() + 1
    )

    session_df["learning_session"] = (
        "session_" + session_df["session_order"].astype(str)
    )

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
# PV computation
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


def split_indices_equal(indices, n_chunks):
    indices = np.asarray(indices)
    return [x for x in np.array_split(indices, n_chunks)]


def make_global_chunk_indices(n_time, n_chunks):
    all_idx = np.arange(n_time)
    return split_indices_equal(all_idx, n_chunks)


def make_trial_normalized_chunk_indices(trial_id, n_chunks):
    """
    Split every trial into n_chunks and pool relative chunks across trials.
    """
    trial_id = ensure_1d(trial_id)

    valid_mask = np.isfinite(trial_id)
    valid_trials = pd.unique(trial_id[valid_mask])

    pooled_chunks = [[] for _ in range(n_chunks)]

    for trial in valid_trials:
        trial_idx = np.where(trial_id == trial)[0]

        if len(trial_idx) < n_chunks:
            continue

        trial_chunks = split_indices_equal(trial_idx, n_chunks)

        for chunk_idx in range(n_chunks):
            pooled_chunks[chunk_idx].extend(trial_chunks[chunk_idx].tolist())

    pooled_chunks = [np.asarray(x, dtype=int) for x in pooled_chunks]
    return pooled_chunks


def compute_chunk_population_vectors(signal, chunk_indices):
    """
    Compute one population vector per chunk.
    signal:
        time x neurons
    """
    vectors = []

    for idx in chunk_indices:
        if len(idx) < MIN_TIMEPOINTS_PER_CHUNK:
            vectors.append(np.full(signal.shape[1], np.nan))
        else:
            vectors.append(np.nanmean(signal[idx, :], axis=0))

    return vectors


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


def compute_pv_matrix(signal, behavior):
    """
    Compute N_CHUNKS x N_CHUNKS PV correlation matrix.
    """
    if ZSCORE_CELLS:
        signal = zscore_activity_cells(signal)

    n_time = signal.shape[0]

    if CHUNK_MODE == "trial_normalized" and behavior["trial_id"] is not None:
        chunk_indices = make_trial_normalized_chunk_indices(
            behavior["trial_id"],
            N_CHUNKS,
        )

        if any(len(idx) < MIN_TIMEPOINTS_PER_CHUNK for idx in chunk_indices):
            print(
                "[WARNING] trial_normalized chunks have too few samples. "
                "Falling back to global_time chunks."
            )
            chunk_indices = make_global_chunk_indices(n_time, N_CHUNKS)
    else:
        chunk_indices = make_global_chunk_indices(n_time, N_CHUNKS)

    pv_vectors = compute_chunk_population_vectors(signal, chunk_indices)

    pv_matrix = np.full((N_CHUNKS, N_CHUNKS), np.nan)

    for i in range(N_CHUNKS):
        for j in range(N_CHUNKS):
            pv_matrix[i, j] = safe_corr(pv_vectors[i], pv_vectors[j])

    if CLIP_NEGATIVE_CORRELATIONS_IN_RESULTS:
        pv_matrix = np.where(
            np.isfinite(pv_matrix),
            np.clip(pv_matrix, 0.0, 1.0),
            np.nan,
        )

    return pv_matrix, pv_vectors, chunk_indices


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
def get_nonzero_lags(df):
    """
    Return sorted lag values excluding lag 0.

    lag 0 is the diagonal/self-comparison and is usually trivially high.
    """
    return sorted([lag for lag in df["lag"].dropna().unique() if lag > 0])

def savefig_png_svg(fig, out_base):
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_pv_matrix(mat, title, out_base):
    plot_mat = np.clip(mat, HEATMAP_VMIN, HEATMAP_VMAX)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))

    im = ax.imshow(
        plot_mat,
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        origin="upper",
        aspect="equal",
    )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Trial chunk")
    ax.set_ylabel("Trial chunk")

    ax.set_xticks(np.arange(N_CHUNKS))
    ax.set_yticks(np.arange(N_CHUNKS))
    ax.set_xticklabels([str(i + 1) for i in range(N_CHUNKS)], fontsize=8)
    ax.set_yticklabels([str(i + 1) for i in range(N_CHUNKS)], fontsize=8)

    for i in range(N_CHUNKS):
        for j in range(N_CHUNKS):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PV correlation")

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_learning_session_lag_barplot(df, title, out_base, session_order):
    """
    Learning plot:
        x-axis   = lag
        bar color = session (gray tones)
    """
    if len(df) == 0:
        return

    lags = get_nonzero_lags(df)

    if len(lags) == 0:
        return

    sessions = session_order

    means = (
        df.groupby(["learning_session", "lag"], observed=True)["pv_corr"]
        .mean()
        .reset_index()
    )

    sems = (
        df.groupby(["learning_session", "lag"], observed=True)["pv_corr"]
        .sem()
        .reset_index()
        .rename(columns={"pv_corr": "sem"})
    )

    stats = means.merge(sems, on=["learning_session", "lag"], how="left")

    x = np.arange(len(lags))
    width = 0.85 / max(len(sessions), 1)

    gray_colors = get_learning_gray_palette(len(sessions))

    fig, ax = plt.subplots(figsize=(10.0, 4.8))

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
                y.append(float(sub["pv_corr"].iloc[0]))
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
    ax.set_xlabel("|Δ chunk|")
    ax.set_ylabel("PV correlation")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(lag)) for lag in lags])
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_deep_sup_lag_barplot(df, title, out_base):
    """
    Deep vs superficial plot using:
        deep         -> gold
        superficial  -> purple
    """
    df = df[df["depth_type"].isin(["deep", "superficial"])].copy()

    if len(df) == 0:
        return
    lags = get_nonzero_lags(df)

    if len(lags) == 0:
        return

    depths = ["deep", "superficial"]
    depth_colors = {
        "deep": DEEP_COLOR,
        "superficial": SUPERFICIAL_COLOR,
    }

    means = (
        df.groupby(["depth_type", "lag"], observed=True)["pv_corr"]
        .mean()
        .reset_index()
    )

    sems = (
        df.groupby(["depth_type", "lag"], observed=True)["pv_corr"]
        .sem()
        .reset_index()
        .rename(columns={"pv_corr": "sem"})
    )

    stats = means.merge(sems, on=["depth_type", "lag"], how="left")

    x = np.arange(len(lags))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

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
                y.append(float(sub["pv_corr"].iloc[0]))
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
    ax.set_xlabel("|Δ chunk|")
    ax.set_ylabel("PV correlation")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(lag)) for lag in lags])
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

    if n_timepoints_after_dir_filter < N_CHUNKS * MIN_TIMEPOINTS_PER_CHUNK:
        raise ValueError(
            f"Too few timepoints after filtering: "
            f"{n_timepoints_after_dir_filter}"
        )

    pv_matrix, pv_vectors, chunk_indices = compute_pv_matrix(
        signal_valid,
        behavior_valid,
    )

    lag_summary = matrix_to_lag_summary(pv_matrix)
    mean_corr_all = float(np.nanmean(pv_matrix))
    mean_corr_offdiag = mean_offdiagonal(pv_matrix)

    dataset_name = row["dataset_name"]
    mouse_name = row["mouse_name"]
    session_name = row["session_name"]
    depth_type = normalize_depth(row["depth_type"])
    condition = infer_condition(row)

    session_base = base_name_from_row(row_idx, row) + "_pv_drift"
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
            "chunk_mode": CHUNK_MODE,
            "n_neurons": n_neurons,
            "n_timepoints_original": n_timepoints_original,
            "n_timepoints_after_dir_filter": n_timepoints_after_dir_filter,
            "n_chunks": N_CHUNKS,
            "zscore_cells": ZSCORE_CELLS,
            "remove_mov_dir_zero": REMOVE_MOV_DIR_ZERO,
            "clip_negative_correlations": CLIP_NEGATIVE_CORRELATIONS_IN_RESULTS,
        },
        "indices": {
            "valid_idx_mov_dir_nonzero": valid_idx,
            "chunk_indices_after_filter": chunk_indices,
        },
        "pv": {
            "matrix": pv_matrix,
            "vectors": pv_vectors,
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
        "chunk_mode": CHUNK_MODE,
        "n_neurons": n_neurons,
        "n_timepoints_original": n_timepoints_original,
        "n_timepoints_after_dir_filter": n_timepoints_after_dir_filter,
        "n_chunks": N_CHUNKS,
        "mean_corr_all": mean_corr_all,
        "mean_corr_offdiag": mean_corr_offdiag,
    }

    for lag in range(N_CHUNKS):
        session_row[f"lag_{lag}"] = lag_summary.get(lag, np.nan)

    session_row["session_pickle_name"] = session_pickle_name
    session_row["session_pickle_path"] = session_pickle_path

    matrix_rows = []

    for i in range(N_CHUNKS):
        for j in range(N_CHUNKS):
            matrix_rows.append(
                {
                    "dataset_name": dataset_name,
                    "mouse_name": mouse_name,
                    "session_name": session_name,
                    "depth_type": depth_type,
                    "condition": condition,
                    "row_index": row_idx,
                    "chunk_i": i + 1,
                    "chunk_j": j + 1,
                    "lag": abs(i - j),
                    "pv_corr": pv_matrix[i, j],
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
                "pv_corr": value,
            }
        )

    heatmap_title = (
        f"{dataset_name}\n"
        f"{mouse_name}, {session_name}, {depth_type}"
    )

    heatmap_base = os.path.join(
        ROW_HEATMAP_DIR,
        session_base + "_matrix",
    )

    plot_pv_matrix(
        pv_matrix,
        title=heatmap_title,
        out_base=heatmap_base,
    )

    lag_print = " | ".join(
        [f"lag{lag}={lag_summary.get(lag, np.nan):.4f}" for lag in range(N_CHUNKS)]
    )

    print(
        f"Row {row_idx} | dataset={dataset_name} | mouse={mouse_name} | "
        f"session={session_name} | depth={depth_type} | condition={condition} | "
        f"neurons={n_neurons} | mean_offdiag={mean_corr_offdiag:.4f} | "
        f"{lag_print}"
    )

    return session_row, matrix_rows, lag_rows


# =============================================================================
# Aggregation helpers
# =============================================================================

def aggregate_mean_matrix_from_entries(df):
    mat = np.full((N_CHUNKS, N_CHUNKS), np.nan)

    for i in range(N_CHUNKS):
        for j in range(N_CHUNKS):
            sub = df[
                (df["chunk_i"] == i + 1)
                & (df["chunk_j"] == j + 1)
            ]

            if len(sub) == 0:
                mat[i, j] = np.nan
            else:
                mat[i, j] = np.nanmean(sub["pv_corr"].values)

    return mat


def run_learning_aggregation(matrix_df, lag_df):
    """
    Learning aggregation with mouse-wise session ordering.
    """
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
            f"learning_{ordered_session}_mean_matrix_across_mice",
        )

        plot_pv_matrix(
            mean_mat,
            title=(
                f"Learning {ordered_session}\n"
                f"mean across mice | mice={n_mice}, rows={n_rows}"
            ),
            out_base=out_base,
        )

    plot_learning_session_lag_barplot(
        learning_lag_df,
        title="Learning dataset: PV correlation by session and temporal-bin difference",
        out_base=os.path.join(
            LEARNING_DIR,
            "learning_lag_barplot_by_mousewise_ordered_session",
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
        )["pv_corr"]
        .mean()
        .reset_index()
    )

    learning_mouse_session_lag_summary.to_csv(
        os.path.join(CSV_DIR, "learning_mouse_session_lag_summary.csv"),
        index=False,
    )

    print("Learning aggregation completed using mouse-wise session ordering.")


def run_lt_rot_aggregation(matrix_df, lag_df):
    """
    For lt_rot dataset:
        - Select rows from dataset_name containing lt_rot
        - Use condition inferred from session_name:
            session_name contains 'lt'  -> condition='lt'
            session_name contains 'rot' -> condition='rot'
        - Plot lt and rot separately
    """
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
        os.path.join(CSV_DIR, "lt_rot_matrix_entries_lt_and_rot_separated.csv"),
        index=False,
    )

    ltrot_lag_df.to_csv(
        os.path.join(CSV_DIR, "lt_rot_lag_summary_lt_and_rot_separated.csv"),
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
                f"lt_rot_{condition}_{depth}_mean_{N_CHUNKS}x{N_CHUNKS}_matrix",
            )

            plot_pv_matrix(
                mean_mat,
                title=(
                    f"lt_rot {condition}, {depth}\n"
                    f"mean across mice | mice={n_mice}, rows={n_rows}"
                ),
                out_base=out_base,
            )

        plot_deep_sup_lag_barplot(
            condition_lag_df,
            title=f"lt_rot {condition}: deep vs superficial",
            out_base=os.path.join(
                LT_ROT_DIR,
                f"lt_rot_{condition}_deep_vs_superficial_lag_barplot",
            ),
        )

    print("lt_rot aggregation completed with lt and rot separated by session_name.")


def run_named_dataset_deep_sup_aggregation(matrix_df, lag_df, dataset_keyword, output_subdir_name):
    """
    Run deep/superficial PV-drift analysis for a named dataset.
    Examples:
        dataset_keyword = "dreadds"
        dataset_keyword = "dreads"
        dataset_keyword = "egrin"
    """
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
            f"{output_subdir_name}_matrix_entries_deep_superficial.csv",
        ),
        index=False,
    )

    dataset_lag_df.to_csv(
        os.path.join(
            CSV_DIR,
            f"{output_subdir_name}_lag_summary_deep_superficial.csv",
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
                f"{depth}_mean_{N_CHUNKS}x{N_CHUNKS}_matrix",
            )

            plot_pv_matrix(
                mean_mat,
                title=(
                    f"{output_subdir_name} {condition}, {depth}\n"
                    f"mean across mice | mice={n_mice}, rows={n_rows}"
                ),
                out_base=out_base,
            )

        plot_deep_sup_lag_barplot(
            condition_lag_df,
            title=f"{output_subdir_name} {condition}: deep vs superficial",
            out_base=os.path.join(
                out_dir,
                f"{output_subdir_name}_{sanitize_for_filename(condition)}_"
                f"deep_vs_superficial_lag_barplot",
            ),
        )

    print(f"{dataset_keyword} aggregation completed.")


def run_all_deep_sup_aggregation(matrix_df, lag_df):
    """
    General fallback deep/superficial aggregation for all non-learning datasets.
    """
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
        os.path.join(CSV_DIR, "all_deep_sup_matrix_entries.csv"),
        index=False,
    )

    ds_lag_df.to_csv(
        os.path.join(CSV_DIR, "all_deep_sup_lag_summary.csv"),
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
                f"{depth}_mean_{N_CHUNKS}x{N_CHUNKS}_matrix",
            )

            plot_pv_matrix(
                mean_mat,
                title=f"{dataset_name} {condition}, {depth}",
                out_base=out_base,
            )

        plot_deep_sup_lag_barplot(
            group_lag_df,
            title=f"{dataset_name} {condition}: deep vs superficial",
            out_base=os.path.join(
                DEEP_SUP_DIR,
                f"{sanitize_for_filename(dataset_name)}_"
                f"{sanitize_for_filename(condition)}_"
                f"deep_vs_superficial_lag_barplot",
            ),
        )

    print("All deep/superficial aggregation completed.")


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

    init_csv(PV_SESSION_CSV, PV_SESSION_COLUMNS)
    init_csv(PV_MATRIX_ENTRIES_CSV, PV_MATRIX_COLUMNS)
    init_csv(PV_LAG_CSV, PV_LAG_COLUMNS)
    init_csv(FAILED_ROWS_CSV, FAILED_COLUMNS)

    log_path = os.path.join(LOG_DIR, "pv_drift_analysis_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Population Vector Drift Analysis")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Rows to process: {len(df)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Session PKL dir: {SESSION_PKL_DIR}")
    print(f"Chunk mode: {CHUNK_MODE}")
    print(f"N chunks: {N_CHUNKS}")
    print(f"Remove mov_dir == 0: {REMOVE_MOV_DIR_ZERO}")
    print(f"Z-score cells: {ZSCORE_CELLS}")
    print(f"Clip negative correlations in results: {CLIP_NEGATIVE_CORRELATIONS_IN_RESULTS}")
    print("")

    ok_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        local_time = timeit.default_timer()

        try:
            session_row, matrix_rows, lag_rows = process_row(row_idx, row)

            append_csv_row(PV_SESSION_CSV, PV_SESSION_COLUMNS, session_row)
            append_many_csv_rows(PV_MATRIX_ENTRIES_CSV, PV_MATRIX_COLUMNS, matrix_rows)
            append_many_csv_rows(PV_LAG_CSV, PV_LAG_COLUMNS, lag_rows)

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

    matrix_df = pd.read_csv(PV_MATRIX_ENTRIES_CSV)
    lag_df = pd.read_csv(PV_LAG_CSV)

    if len(matrix_df) > 0:
        # Learning with mouse-wise ordered sessions and gray palette
        run_learning_aggregation(matrix_df, lag_df)

        # lt_rot split by session_name into lt and rot
        run_lt_rot_aggregation(matrix_df, lag_df)

        # Dedicated deep/superficial analyses
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

        # General fallback
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