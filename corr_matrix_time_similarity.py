#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correlation-matrix drift analysis across all datasets.

For each row/session:
    1. Load standardized signal: time x neurons
    2. Optionally remove mov_dir == 0
    3. Z-score neurons across the full valid session
    4. Split global time into N_CHUNKS
    5. For each chunk:
           compute neuron x neuron correlation matrix
    6. For each pair of chunks:
           vectorize upper triangle of both matrices
           compute similarity between correlation matrices
    7. Summarize similarity matrix by |Δ chunk|

This is a second-order representation analysis:
    chunk -> neuron-neuron correlation matrix

Outputs:
    /home/melisamc/Documents/time_project_standarization/CorrMatrix_Drift_Analysis

Plots:
    - per-row heatmaps
    - learning session-wise mean matrices and lag curves
    - lt_rot split into lt and rot, deep vs superficial
    - dreads/dreadds/egrin deep vs superficial
    - all non-learning deep/superficial comparisons

Important:
    lag 0 is saved in CSVs but excluded from summary plots.
    y-axis limits are computed automatically from plotted values.
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
# PATHS
# =============================================================================

CSV_PATH = (
    "/home/melisamc/Documents/time_project_standarization/"
    "data/metadata/master_dataset_index.csv"
)

OUTPUT_DIR = (
    "/home/melisamc/Documents/time_project_standarization/"
    "CorrMatrix_Drift_Analysis"
)

SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
ROW_HEATMAP_DIR = os.path.join(FIG_DIR, "row_heatmaps")
LEARNING_DIR = os.path.join(FIG_DIR, "learning")
LT_ROT_DIR = os.path.join(FIG_DIR, "lt_rot")
DEEP_SUP_DIR = os.path.join(FIG_DIR, "deep_superficial_all_datasets")

SESSION_CSV = os.path.join(CSV_DIR, "corr_matrix_drift_per_session.csv")
MATRIX_ENTRIES_CSV = os.path.join(CSV_DIR, "corr_matrix_similarity_entries.csv")
LAG_CSV = os.path.join(CSV_DIR, "corr_matrix_lag_summary.csv")
FAILED_ROWS_CSV = os.path.join(CSV_DIR, "failed_rows.csv")

# =============================================================================
# PARAMETERS
# =============================================================================

MAX_ROWS = None
DATASET_FILTER = None

# Global-time chunks
N_CHUNKS = 5

REMOVE_MOV_DIR_ZERO = True
ZSCORE_CELLS = True

# Correlation matrix estimation
MIN_TIMEPOINTS_PER_CHUNK = 10
MIN_NEURONS = 5

# Similarity between correlation matrices
# Main mode: Pearson correlation between upper triangles
MATRIX_SIMILARITY_METRIC = "pearson"

# Correlation matrix method
CORR_METHOD = "pearson"

# Save signed values by default.
CLIP_NEGATIVE_SIMILARITY_IN_RESULTS = False

# Plotting
EXCLUDE_ZERO_LAG_IN_PLOTS = True

# Heatmaps can stay fixed for comparability
HEATMAP_VMIN = -1.0
HEATMAP_VMAX = 1.0

# Automatic y-limit padding for summary plots
YLIM_PADDING_FRAC = 0.08

PROCESS_WHOLE_POPULATION = True

# =============================================================================
# COLORS
# =============================================================================

DEEP_COLOR = "#D4A928"
SUPERFICIAL_COLOR = "#A020F0"


def get_learning_gray_palette(n_sessions):
    if n_sessions <= 1:
        return ["#666666"]

    vals = np.linspace(0.25, 0.80, n_sessions)
    return [str(v) for v in vals]


# =============================================================================
# CSV SCHEMAS
# =============================================================================

SESSION_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "condition",
    "standardized_file_path",
    "row_index",
    "n_chunks",
    "corr_method",
    "matrix_similarity_metric",
    "n_neurons",
    "n_timepoints_original",
    "n_timepoints_after_filter",
    "mean_similarity_all",
    "mean_similarity_offdiag",
    "session_pickle_name",
    "session_pickle_path",
]

MATRIX_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "condition",
    "row_index",
    "chunk_i",
    "chunk_j",
    "lag",
    "corr_matrix_similarity",
]

LAG_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "condition",
    "row_index",
    "lag",
    "corr_matrix_similarity",
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
# GENERAL HELPERS
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


def load_pickle_standard(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle_standard(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def savefig_png_svg(fig, out_base):
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def sanitize_filename(text):
    text = str(text)
    safe = []

    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")

    return "".join(safe)


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
# METADATA HELPERS
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
    dataset = str(row.get("dataset_name", "")).lower()
    session = str(row.get("session_name", "")).lower()
    path = str(row.get("standardized_file_path", "")).lower()

    # lt_rot condition must come from session name
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


def extract_learning_datetime_from_session(session_name):
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

    try:
        return pd.to_datetime(text)
    except Exception:
        return pd.NaT


def build_learning_session_mapping_by_mouse(matrix_df):
    session_df = (
        matrix_df[["mouse_name", "session_name"]]
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
# DATA LOADING
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

    n_time = signal.shape[0]

    def check_len(x, name):
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

    behavior = {
        "pos": pos,
        "mov_dir": mov_dir,
        "speed": speed,
        "time": time,
    }

    return signal.astype(float), behavior


# =============================================================================
# CORRELATION-MATRIX DRIFT COMPUTATION
# =============================================================================

def zscore_activity_cells(signal):
    mu = np.nanmean(signal, axis=0, keepdims=True)
    sd = np.nanstd(signal, axis=0, keepdims=True)

    sd[sd == 0] = np.nan

    z = (signal - mu) / sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    return z


def split_indices_equal(n_time, n_chunks):
    indices = np.arange(n_time)
    return [x for x in np.array_split(indices, n_chunks)]


def compute_neuron_corr_matrix(signal_chunk):
    """
    signal_chunk: time x neurons
    """
    if signal_chunk.shape[0] < MIN_TIMEPOINTS_PER_CHUNK:
        return np.full((signal_chunk.shape[1], signal_chunk.shape[1]), np.nan)

    if signal_chunk.shape[1] < MIN_NEURONS:
        return np.full((signal_chunk.shape[1], signal_chunk.shape[1]), np.nan)

    if CORR_METHOD != "pearson":
        raise ValueError(f"Unknown CORR_METHOD: {CORR_METHOD}")

    corr_mat = np.corrcoef(signal_chunk, rowvar=False)

    return corr_mat


def vectorize_corr_matrix(corr_mat):
    corr_mat = np.asarray(corr_mat, dtype=float)

    if corr_mat.ndim != 2 or corr_mat.shape[0] != corr_mat.shape[1]:
        raise ValueError(f"Expected square matrix, got {corr_mat.shape}")

    n = corr_mat.shape[0]
    idx = np.triu_indices(n, k=1)

    return corr_mat[idx]


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


def compute_matrix_similarity(vec_a, vec_b):
    if MATRIX_SIMILARITY_METRIC == "pearson":
        return safe_corr(vec_a, vec_b)

    raise ValueError(f"Unknown MATRIX_SIMILARITY_METRIC: {MATRIX_SIMILARITY_METRIC}")


def compute_corr_matrix_drift(signal):
    """
    Returns:
        similarity_matrix: N_CHUNKS x N_CHUNKS
        corr_mats: list of neuron-neuron correlation matrices
        corr_vecs: list of upper-triangle vectors
        chunk_indices: list of indices
    """
    if ZSCORE_CELLS:
        signal = zscore_activity_cells(signal)

    chunk_indices = split_indices_equal(signal.shape[0], N_CHUNKS)

    corr_mats = []
    corr_vecs = []

    for idx in chunk_indices:
        signal_chunk = signal[idx, :]

        corr_mat = compute_neuron_corr_matrix(signal_chunk)
        corr_vec = vectorize_corr_matrix(corr_mat)

        corr_mats.append(corr_mat)
        corr_vecs.append(corr_vec)

    similarity_matrix = np.full((N_CHUNKS, N_CHUNKS), np.nan)

    for i in range(N_CHUNKS):
        for j in range(N_CHUNKS):
            similarity_matrix[i, j] = compute_matrix_similarity(
                corr_vecs[i],
                corr_vecs[j],
            )

    if CLIP_NEGATIVE_SIMILARITY_IN_RESULTS:
        similarity_matrix = np.where(
            np.isfinite(similarity_matrix),
            np.clip(similarity_matrix, 0.0, 1.0),
            np.nan,
        )

    return similarity_matrix, corr_mats, corr_vecs, chunk_indices


def matrix_to_lag_summary(mat):
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
# PLOTTING HELPERS
# =============================================================================

def get_nonzero_lags(df, lag_col="lag"):
    lags = sorted(df[lag_col].dropna().unique())

    if EXCLUDE_ZERO_LAG_IN_PLOTS:
        lags = [lag for lag in lags if lag > 0]

    return lags


def compute_auto_ylim_from_values(values, padding_frac=YLIM_PADDING_FRAC):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return -1.0, 1.0

    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))

    if ymin == ymax:
        pad = max(abs(ymin), 1.0) * 0.1
        return ymin - pad, ymax + pad

    yrange = ymax - ymin
    return ymin - padding_frac * yrange, ymax + padding_frac * yrange


def compute_ylim_from_plot_df(df, value_col="corr_matrix_similarity"):
    plot_df = df.copy()

    if EXCLUDE_ZERO_LAG_IN_PLOTS and "lag" in plot_df.columns:
        plot_df = plot_df[plot_df["lag"] > 0].copy()

    return compute_auto_ylim_from_values(plot_df[value_col].values)


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
    ax.set_xlabel("Global chunk")
    ax.set_ylabel("Global chunk")

    ax.set_xticks(np.arange(N_CHUNKS))
    ax.set_yticks(np.arange(N_CHUNKS))
    ax.set_xticklabels([str(i + 1) for i in range(N_CHUNKS)], fontsize=6)
    ax.set_yticklabels([str(i + 1) for i in range(N_CHUNKS)], fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation-matrix similarity")

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_learning_lag_curves(df, title, out_base, session_order):
    if len(df) == 0:
        return

    lags = get_nonzero_lags(df)
    if len(lags) == 0:
        return

    y_min, y_max = compute_ylim_from_plot_df(df)

    means = (
        df.groupby(["learning_session", "lag"], observed=True)["corr_matrix_similarity"]
        .mean()
        .reset_index()
    )

    sems = (
        df.groupby(["learning_session", "lag"], observed=True)["corr_matrix_similarity"]
        .sem()
        .reset_index()
        .rename(columns={"corr_matrix_similarity": "sem"})
    )

    stats = means.merge(sems, on=["learning_session", "lag"], how="left")
    gray_colors = get_learning_gray_palette(len(session_order))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for k, session in enumerate(session_order):
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
                y.append(float(sub["corr_matrix_similarity"].iloc[0]))
                yerr.append(float(sub["sem"].iloc[0]))

        y = np.asarray(y, dtype=float)
        yerr = np.nan_to_num(np.asarray(yerr, dtype=float), nan=0.0)

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

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("|Δ chunk|")
    ax.set_ylabel("Correlation-matrix similarity")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(lags)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_learning_lag_barplot(df, title, out_base, session_order):
    if len(df) == 0:
        return

    lags = get_nonzero_lags(df)
    if len(lags) == 0:
        return

    y_min, y_max = compute_ylim_from_plot_df(df)

    means = (
        df.groupby(["learning_session", "lag"], observed=True)["corr_matrix_similarity"]
        .mean()
        .reset_index()
    )

    sems = (
        df.groupby(["learning_session", "lag"], observed=True)["corr_matrix_similarity"]
        .sem()
        .reset_index()
        .rename(columns={"corr_matrix_similarity": "sem"})
    )

    stats = means.merge(sems, on=["learning_session", "lag"], how="left")

    x = np.arange(len(lags))
    width = 0.85 / max(len(session_order), 1)
    gray_colors = get_learning_gray_palette(len(session_order))

    fig, ax = plt.subplots(figsize=(11.0, 4.8))

    for k, session in enumerate(session_order):
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
                y.append(float(sub["corr_matrix_similarity"].iloc[0]))
                yerr.append(float(sub["sem"].iloc[0]))

        offset = (k - (len(session_order) - 1) / 2) * width

        ax.bar(
            x + offset,
            y,
            width=width,
            yerr=np.nan_to_num(yerr, nan=0.0),
            capsize=2,
            color=gray_colors[k],
            edgecolor="black",
            linewidth=0.7,
            label=session,
        )

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("|Δ chunk|")
    ax.set_ylabel("Correlation-matrix similarity")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(lag)) for lag in lags])
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_deep_sup_lag_curves(df, title, out_base):
    df = df[df["depth_type"].isin(["deep", "superficial"])].copy()

    if len(df) == 0:
        return

    lags = get_nonzero_lags(df)
    if len(lags) == 0:
        return

    y_min, y_max = compute_ylim_from_plot_df(df)

    depth_colors = {
        "deep": DEEP_COLOR,
        "superficial": SUPERFICIAL_COLOR,
    }

    means = (
        df.groupby(["depth_type", "lag"], observed=True)["corr_matrix_similarity"]
        .mean()
        .reset_index()
    )

    sems = (
        df.groupby(["depth_type", "lag"], observed=True)["corr_matrix_similarity"]
        .sem()
        .reset_index()
        .rename(columns={"corr_matrix_similarity": "sem"})
    )

    stats = means.merge(sems, on=["depth_type", "lag"], how="left")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for depth in ["deep", "superficial"]:
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
                y.append(float(sub["corr_matrix_similarity"].iloc[0]))
                yerr.append(float(sub["sem"].iloc[0]))

        y = np.asarray(y, dtype=float)
        yerr = np.nan_to_num(np.asarray(yerr, dtype=float), nan=0.0)

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

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("|Δ chunk|")
    ax.set_ylabel("Correlation-matrix similarity")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(lags)
    ax.legend(frameon=False)

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_deep_sup_lag_barplot(df, title, out_base):
    df = df[df["depth_type"].isin(["deep", "superficial"])].copy()

    if len(df) == 0:
        return

    lags = get_nonzero_lags(df)
    if len(lags) == 0:
        return

    y_min, y_max = compute_ylim_from_plot_df(df)

    depth_colors = {
        "deep": DEEP_COLOR,
        "superficial": SUPERFICIAL_COLOR,
    }

    means = (
        df.groupby(["depth_type", "lag"], observed=True)["corr_matrix_similarity"]
        .mean()
        .reset_index()
    )

    sems = (
        df.groupby(["depth_type", "lag"], observed=True)["corr_matrix_similarity"]
        .sem()
        .reset_index()
        .rename(columns={"corr_matrix_similarity": "sem"})
    )

    stats = means.merge(sems, on=["depth_type", "lag"], how="left")

    x = np.arange(len(lags))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for k, depth in enumerate(["deep", "superficial"]):
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
                y.append(float(sub["corr_matrix_similarity"].iloc[0]))
                yerr.append(float(sub["sem"].iloc[0]))

        offset = (k - 0.5) * width

        ax.bar(
            x + offset,
            y,
            width=width,
            yerr=np.nan_to_num(yerr, nan=0.0),
            capsize=2,
            color=depth_colors[depth],
            edgecolor="black",
            linewidth=0.8,
            label=depth,
        )

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("|Δ chunk|")
    ax.set_ylabel("Correlation-matrix similarity")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(lag)) for lag in lags])
    ax.legend(frameon=False)

    fig.tight_layout()
    savefig_png_svg(fig, out_base)


# =============================================================================
# ROW PROCESSING
# =============================================================================

def process_row(row_idx, row):
    signal, behavior = load_standardized_row(row_idx, row)

    n_timepoints_original = signal.shape[0]
    n_neurons = signal.shape[1]

    if n_neurons < MIN_NEURONS:
        raise ValueError(f"Too few neurons: {n_neurons}")

    if REMOVE_MOV_DIR_ZERO:
        valid_idx = behavior["mov_dir"] != 0
    else:
        valid_idx = np.ones(signal.shape[0], dtype=bool)

    signal_valid = signal[valid_idx, :]

    n_timepoints_after_filter = signal_valid.shape[0]

    if n_timepoints_after_filter < N_CHUNKS * MIN_TIMEPOINTS_PER_CHUNK:
        raise ValueError(
            f"Too few timepoints after filtering: "
            f"{n_timepoints_after_filter}. "
            f"Need at least {N_CHUNKS * MIN_TIMEPOINTS_PER_CHUNK}."
        )

    similarity_matrix, corr_mats, corr_vecs, chunk_indices = compute_corr_matrix_drift(
        signal_valid
    )

    lag_summary = matrix_to_lag_summary(similarity_matrix)
    mean_similarity_all = float(np.nanmean(similarity_matrix))
    mean_similarity_offdiag = mean_offdiagonal(similarity_matrix)

    dataset_name = row["dataset_name"]
    mouse_name = row["mouse_name"]
    session_name = row["session_name"]
    depth_type = normalize_depth(row["depth_type"])
    condition = infer_condition(row)

    session_base = (
        f"row_{row_idx:04d}_"
        f"{sanitize_filename(dataset_name)}_"
        f"{sanitize_filename(mouse_name)}_"
        f"{sanitize_filename(session_name)}_"
        f"{sanitize_filename(depth_type)}_corrmatrix_drift"
    )

    session_pickle_name = session_base + ".pkl"
    session_pickle_path = os.path.join(SESSION_PKL_DIR, session_pickle_name)

    payload = {
        "metadata": {
            "dataset_name": dataset_name,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "depth_type": depth_type,
            "condition": condition,
            "row_index": row_idx,
            "standardized_file_path": row["standardized_file_path"],
            "n_chunks": N_CHUNKS,
            "corr_method": CORR_METHOD,
            "matrix_similarity_metric": MATRIX_SIMILARITY_METRIC,
            "n_neurons": n_neurons,
            "n_timepoints_original": n_timepoints_original,
            "n_timepoints_after_filter": n_timepoints_after_filter,
            "zscore_cells": ZSCORE_CELLS,
            "remove_mov_dir_zero": REMOVE_MOV_DIR_ZERO,
            "clip_negative_similarity": CLIP_NEGATIVE_SIMILARITY_IN_RESULTS,
        },
        "indices": {
            "valid_idx_mov_dir_nonzero": valid_idx,
            "chunk_indices_after_filter": chunk_indices,
        },
        "correlation_matrix_drift": {
            "similarity_matrix": similarity_matrix,
            "corr_mats": corr_mats,
            "corr_vecs": corr_vecs,
            "lag_summary": lag_summary,
        },
    }

    save_pickle_standard(session_pickle_path, payload)

    session_row = {
        "dataset_name": dataset_name,
        "mouse_name": mouse_name,
        "session_name": session_name,
        "depth_type": depth_type,
        "condition": condition,
        "standardized_file_path": row["standardized_file_path"],
        "row_index": row_idx,
        "n_chunks": N_CHUNKS,
        "corr_method": CORR_METHOD,
        "matrix_similarity_metric": MATRIX_SIMILARITY_METRIC,
        "n_neurons": n_neurons,
        "n_timepoints_original": n_timepoints_original,
        "n_timepoints_after_filter": n_timepoints_after_filter,
        "mean_similarity_all": mean_similarity_all,
        "mean_similarity_offdiag": mean_similarity_offdiag,
        "session_pickle_name": session_pickle_name,
        "session_pickle_path": session_pickle_path,
    }

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
                    "corr_matrix_similarity": similarity_matrix[i, j],
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
                "corr_matrix_similarity": value,
            }
        )

    plot_similarity_matrix(
        similarity_matrix,
        title=(
            f"{dataset_name}\n"
            f"{mouse_name}, {session_name}, {depth_type}\n"
            f"Correlation-matrix drift, chunks={N_CHUNKS}"
        ),
        out_base=os.path.join(ROW_HEATMAP_DIR, session_base + "_matrix"),
    )

    print(
        f"Row {row_idx} | dataset={dataset_name} | mouse={mouse_name} | "
        f"session={session_name} | depth={depth_type} | condition={condition} | "
        f"neurons={n_neurons} | mean_offdiag={mean_similarity_offdiag:.4f}"
    )

    return session_row, matrix_rows, lag_rows


# =============================================================================
# AGGREGATION HELPERS
# =============================================================================

def aggregate_mean_matrix_from_entries(df):
    mat = np.full((N_CHUNKS, N_CHUNKS), np.nan)

    for i in range(N_CHUNKS):
        for j in range(N_CHUNKS):
            sub = df[
                (df["chunk_i"] == i + 1)
                & (df["chunk_j"] == j + 1)
                ]

            if len(sub) > 0:
                mat[i, j] = np.nanmean(sub["corr_matrix_similarity"].values)

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

        plot_similarity_matrix(
            mean_mat,
            title=(
                f"Learning {ordered_session}\n"
                f"mean correlation-matrix similarity | mice={n_mice}, rows={n_rows}"
            ),
            out_base=os.path.join(
                LEARNING_DIR,
                f"learning_{ordered_session}_mean_corr_matrix_similarity",
            ),
        )

    plot_learning_lag_barplot(
        learning_lag_df,
        title="Learning: correlation-matrix similarity by session",
        out_base=os.path.join(
            LEARNING_DIR,
            "learning_corr_matrix_lag_barplot",
        ),
        session_order=ordered_sessions,
    )

    plot_learning_lag_curves(
        learning_lag_df,
        title="Learning: correlation-matrix similarity curves by session",
        out_base=os.path.join(
            LEARNING_DIR,
            "learning_corr_matrix_lag_curves",
        ),
        session_order=ordered_sessions,
    )

    learning_lag_df.to_csv(
        os.path.join(CSV_DIR, "learning_corr_matrix_lag_summary_ordered_sessions.csv"),
        index=False,
    )

    print("Learning aggregation completed.")


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
        & ltrot_matrix_df["condition"].isin(["lt", "rot"])
        ].copy()

    ltrot_lag_df = lag_df[
        lag_df["row_index"].isin(ltrot_matrix_df["row_index"].unique())
        & lag_df["depth_type"].isin(["deep", "superficial"])
        & lag_df["condition"].isin(["lt", "rot"])
        ].copy()

    if len(ltrot_lag_df) == 0:
        print("lt_rot rows found, but no lt/rot deep/superficial entries.")
        return

    ltrot_matrix_df.to_csv(
        os.path.join(CSV_DIR, "lt_rot_corr_matrix_entries_lt_rot_split.csv"),
        index=False,
    )

    ltrot_lag_df.to_csv(
        os.path.join(CSV_DIR, "lt_rot_corr_matrix_lag_summary_lt_rot_split.csv"),
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

            plot_similarity_matrix(
                mean_mat,
                title=(
                    f"lt_rot {condition}, {depth}\n"
                    f"mean correlation-matrix similarity | mice={n_mice}, rows={n_rows}"
                ),
                out_base=os.path.join(
                    LT_ROT_DIR,
                    f"lt_rot_{condition}_{depth}_mean_corr_matrix_similarity",
                ),
            )

        plot_deep_sup_lag_barplot(
            condition_lag_df,
            title=f"lt_rot {condition}: correlation-matrix similarity deep vs superficial",
            out_base=os.path.join(
                LT_ROT_DIR,
                f"lt_rot_{condition}_deep_vs_superficial_corr_matrix_lag_barplot",
            ),
        )

        plot_deep_sup_lag_curves(
            condition_lag_df,
            title=f"lt_rot {condition}: correlation-matrix similarity deep vs superficial curves",
            out_base=os.path.join(
                LT_ROT_DIR,
                f"lt_rot_{condition}_deep_vs_superficial_corr_matrix_lag_curves",
            ),
        )

    print("lt_rot aggregation completed.")


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

            plot_similarity_matrix(
                mean_mat,
                title=(
                    f"{output_subdir_name} {condition}, {depth}\n"
                    f"mean correlation-matrix similarity | mice={n_mice}, rows={n_rows}"
                ),
                out_base=os.path.join(
                    out_dir,
                    f"{output_subdir_name}_{sanitize_filename(condition)}_{depth}_mean_corr_matrix_similarity",
                ),
            )

        plot_deep_sup_lag_barplot(
            condition_lag_df,
            title=f"{output_subdir_name} {condition}: correlation-matrix similarity deep vs superficial",
            out_base=os.path.join(
                out_dir,
                f"{output_subdir_name}_{sanitize_filename(condition)}_deep_vs_superficial_corr_matrix_lag_barplot",
            ),
        )

        plot_deep_sup_lag_curves(
            condition_lag_df,
            title=f"{output_subdir_name} {condition}: correlation-matrix similarity deep vs superficial curves",
            out_base=os.path.join(
                out_dir,
                f"{output_subdir_name}_{sanitize_filename(condition)}_deep_vs_superficial_corr_matrix_lag_curves",
            ),
        )

    print(f"{dataset_keyword} aggregation completed.")


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

            plot_similarity_matrix(
                mean_mat,
                title=f"{dataset_name} {condition}, {depth}",
                out_base=os.path.join(
                    DEEP_SUP_DIR,
                    f"{sanitize_filename(dataset_name)}_{sanitize_filename(condition)}_{depth}_mean_corr_matrix_similarity",
                ),
            )

        plot_deep_sup_lag_barplot(
            group_lag_df,
            title=f"{dataset_name} {condition}: correlation-matrix similarity deep vs superficial",
            out_base=os.path.join(
                DEEP_SUP_DIR,
                f"{sanitize_filename(dataset_name)}_{sanitize_filename(condition)}_deep_vs_superficial_corr_matrix_lag_barplot",
            ),
        )

        plot_deep_sup_lag_curves(
            group_lag_df,
            title=f"{dataset_name} {condition}: correlation-matrix similarity deep vs superficial curves",
            out_base=os.path.join(
                DEEP_SUP_DIR,
                f"{sanitize_filename(dataset_name)}_{sanitize_filename(condition)}_deep_vs_superficial_corr_matrix_lag_curves",
            ),
        )

    print("All deep/superficial aggregation completed.")


# =============================================================================
# MAIN
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
            raise KeyError(f"Missing required column '{col}' in master dataset CSV.")

    if DATASET_FILTER is not None:
        df = df[df["dataset_name"].isin(DATASET_FILTER)].copy()

    if not PROCESS_WHOLE_POPULATION:
        df = df[
            ~df["depth_type"].astype(str).str.lower().str.contains("whole")
        ].copy()

    df = df.reset_index(drop=True)

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    init_csv(SESSION_CSV, SESSION_COLUMNS)
    init_csv(MATRIX_ENTRIES_CSV, MATRIX_COLUMNS)
    init_csv(LAG_CSV, LAG_COLUMNS)
    init_csv(FAILED_ROWS_CSV, FAILED_COLUMNS)

    log_path = os.path.join(LOG_DIR, "corr_matrix_drift_analysis_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Correlation-Matrix Drift Analysis")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Rows to process: {len(df)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"N chunks: {N_CHUNKS}")
    print(f"Remove mov_dir == 0: {REMOVE_MOV_DIR_ZERO}")
    print(f"Z-score cells: {ZSCORE_CELLS}")
    print(f"Correlation method: {CORR_METHOD}")
    print(f"Matrix similarity metric: {MATRIX_SIMILARITY_METRIC}")
    print(f"Clip negative similarity: {CLIP_NEGATIVE_SIMILARITY_IN_RESULTS}")
    print(f"Exclude lag 0 in plots: {EXCLUDE_ZERO_LAG_IN_PLOTS}")
    print("")

    ok_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        local_time = timeit.default_timer()

        try:
            session_row, matrix_rows, lag_rows = process_row(row_idx, row)

            append_csv_row(SESSION_CSV, SESSION_COLUMNS, session_row)
            append_many_csv_rows(MATRIX_ENTRIES_CSV, MATRIX_COLUMNS, matrix_rows)
            append_many_csv_rows(LAG_CSV, LAG_COLUMNS, lag_rows)

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

    matrix_df = pd.read_csv(MATRIX_ENTRIES_CSV)
    lag_df = pd.read_csv(LAG_CSV)

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
