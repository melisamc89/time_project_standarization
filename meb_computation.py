#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UMAP lap-centroid visualization + original-space MEB/dispersion metrics

This script:
1) loads each standardized session
2) ALWAYS computes a full-lap trial_id from position only
3) computes one centroid per full lap in ORIGINAL signal space
4) fits UMAP on the full signal and projects lap centroids with umap_model.transform(...)
5) plots:
   - UMAP cloud in gray
   - lap centroids in YlGn_r
   - using previously saved UMAP view angles when available
6) computes dispersion metrics in ORIGINAL space only:
   - MEB radius
   - normalized MEB radius
   - distance-to-global-centroid metrics:
       raw, dimension-normalized, MEB-normalized
   - consecutive centroid distance metrics:
       raw, dimension-normalized, MEB-normalized
7) saves:
   - png + svg figure
   - pkl with all data
   - txt stats
   - fitted UMAP model
   - analysis index CSV
   - summary CSV with dataset/mouse/session/depth/modality + metrics

Notes
-----
- UMAP model is recomputed here and saved per session.
- MEB is computed ONLY in original space.
- Full-lap trial id is always recomputed from position.
"""

import os
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import umap
from sklearn.metrics import pairwise_distances


# =============================================================================
# Paths
# =============================================================================

CSV_PATH = "/home/melisamc/Documents/time_project_standarization/data/metadata/master_dataset_index.csv"

PREV_ANALYSIS_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/umap_pca_per_row"
PREV_ANALYSIS_INDEX_CSV = os.path.join(PREV_ANALYSIS_DIR, "umap_pca_analysis_index.csv")

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/meb_lap_centroids_umap"
PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")
TXT_DIR = os.path.join(OUTPUT_DIR, "stats_txt")
PKL_DIR = os.path.join(OUTPUT_DIR, "data_pkl")
MODEL_DIR = os.path.join(OUTPUT_DIR, "umap_models")

ANALYSIS_INDEX_CSV = os.path.join(OUTPUT_DIR, "meb_lap_centroids_analysis_index.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "meb_dispersion_summary.csv")

MAX_ROWS = None

# Optional filtering
DATASET_FILTER = None
# DATASET_FILTER = "learning"


# =============================================================================
# UMAP parameters
# =============================================================================

UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
UMAP_N_COMPONENTS = 3


# =============================================================================
# Full lap definition
# =============================================================================

HIGH_THR_PERCENTILE = 90
LOW_THR_PERCENTILE = 10


# =============================================================================
# Optional preprocessing
# =============================================================================

APPLY_GAUSSIAN_SMOOTHING = False
SMOOTH_SIGMA = 4

APPLY_OUTLIER_FILTER_FOR_SIGNAL_POINTS = False


# =============================================================================
# Plot parameters
# =============================================================================

FIGSIZE = (8, 7)
GRAY_POINT_SIZE = 3
GRAY_ALPHA = 0.15
CENTROID_POINT_SIZE = 70
CENTROID_ALPHA = 1.0

USE_PREVIOUS_VIEW_ANGLES = True
DEFAULT_ELEV = 20.0
DEFAULT_AZIM = -60.0


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_pickle_standard(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle_standard(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def sanitize_for_filename(text):
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


def base_name_from_row(row):
    return (
        f"{sanitize_for_filename(row['dataset_name'])}_"
        f"{sanitize_for_filename(row['mouse_name'])}_"
        f"{sanitize_for_filename(row['session_name'])}_"
        f"{sanitize_for_filename(row['depth_type'])}"
    )


def save_figure_both(fig, out_base):
    png_path = os.path.join(PNG_DIR, out_base + ".png")
    svg_path = os.path.join(SVG_DIR, out_base + ".svg")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")
    return png_path, svg_path


def save_text(text, out_base):
    txt_path = os.path.join(TXT_DIR, out_base + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] Saved: {txt_path}")
    return txt_path


def smooth_calcium_signals(calcium_data, sigma=4):
    from scipy.ndimage import gaussian_filter1d
    calcium_data = np.asarray(calcium_data)
    smoothed = np.zeros_like(calcium_data)
    for i in range(calcium_data.shape[1]):
        smoothed[:, i] = gaussian_filter1d(calcium_data[:, i], sigma=sigma)
    return smoothed


def filter_noisy_outliers(data):
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D, 1), axis=1) - 1
    noise_idx = np.where(nn_dist < np.percentile(nn_dist, 20))[0]
    signal_idx = np.where(nn_dist >= np.percentile(nn_dist, 20))[0]
    return noise_idx, signal_idx


def infer_modality_label(row, standardized_file_path, data_dict):
    """
    Prefer CSV column 'modality' if present.
    Otherwise infer from path/name.
    """
    if "modality" in row and pd.notna(row["modality"]):
        return str(row["modality"])

    path_lower = str(standardized_file_path).lower()
    dataset_lower = str(row.get("dataset_name", "")).lower()

    if "ephys" in path_lower or "ephys" in dataset_lower:
        return "ephys"
    if "calcium" in path_lower or "miniscope" in path_lower or "learning" in dataset_lower:
        return "calcium"

    return "unknown"


# =============================================================================
# Full-lap trial id
# =============================================================================

def compute_full_lap_trial_id(pos, high_thr=None, low_thr=None):
    """
    One trial = one full out-and-back cycle:
      cross above high_thr, then return below low_thr -> increment trial id.
    """
    pos = np.asarray(pos).ravel()

    if high_thr is None:
        high_thr = np.percentile(pos, HIGH_THR_PERCENTILE)
    if low_thr is None:
        low_thr = np.percentile(pos, LOW_THR_PERCENTILE)

    trial_id = np.zeros_like(pos, dtype=int)
    in_high = False
    current_id = 0

    for t in range(1, len(pos)):
        if (not in_high) and (pos[t] > high_thr):
            in_high = True
        elif in_high and (pos[t] < low_thr):
            current_id += 1
            in_high = False
        trial_id[t] = current_id

    return trial_id, float(high_thr), float(low_thr)


def compute_centroids_per_trial(signal, trial_id):
    signal = np.asarray(signal)
    trial_id = np.asarray(trial_id).astype(int).ravel()

    unique_tids = np.unique(trial_id)
    centroids = []
    valid_trials = []
    counts = []

    for tid in unique_tids:
        mask = (trial_id == tid)
        if not np.any(mask):
            continue
        center = np.nanmean(signal[mask], axis=0)
        if np.isnan(center).all():
            continue
        centroids.append(center)
        valid_trials.append(int(tid))
        counts.append(int(np.sum(mask)))

    if len(centroids) == 0:
        return np.empty((0, signal.shape[1])), np.array([], dtype=int), np.array([], dtype=int)

    centroids = np.vstack(centroids)
    valid_trials = np.asarray(valid_trials, dtype=int)
    counts = np.asarray(counts, dtype=int)
    return centroids, valid_trials, counts


def consecutive_distances(points):
    points = np.asarray(points)
    if len(points) < 2:
        return np.array([], dtype=float)
    diffs = points[1:] - points[:-1]
    return np.linalg.norm(diffs, axis=1)


# =============================================================================
# Exact MEB (Welzl)
# =============================================================================

def _ball_from(points):
    P = np.asarray(points, dtype=float)
    if len(P) == 0:
        return np.zeros(1), 0.0
    if len(P) == 1:
        return P[0], 0.0
    if len(P) == 2:
        c = (P[0] + P[1]) / 2.0
        r = np.linalg.norm(P[0] - c)
        return c, r

    A = 2 * (P[1:] - P[0])
    b = np.sum(P[1:] ** 2 - P[0] ** 2, axis=1)

    try:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        c = P.mean(axis=0)

    r = np.max(np.linalg.norm(P - c, axis=1))
    return c, r


def _is_in_ball(p, c, r, eps=1e-10):
    return np.linalg.norm(p - c) <= r + eps


def _welzl(P, R, d):
    if len(R) == d + 1 or len(P) == 0:
        c, r = _ball_from(R)
        return c, r

    p = P.pop()
    c, r = _welzl(P, R, d)

    if _is_in_ball(p, c, r):
        P.append(p)
        return c, r

    R.append(p)
    c2, r2 = _welzl(P, R, d)
    R.pop()
    P.append(p)
    return c2, r2


def minimum_enclosing_ball(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_points x n_dims)")

    n, d = X.shape
    if n == 0:
        return np.zeros(d), 0.0

    P = [x for x in X]
    rng = np.random.default_rng(123)
    rng.shuffle(P)
    c, r = _welzl(P, [], d)
    return c, float(r)


# =============================================================================
# Previous view angles
# =============================================================================

def load_previous_view_angles(prev_analysis_index_csv, base_name):
    if not os.path.isfile(prev_analysis_index_csv):
        return None, None, None

    prev_df = pd.read_csv(prev_analysis_index_csv)
    if "analysis_file_path" not in prev_df.columns:
        return None, None, None

    for _, row in prev_df.iterrows():
        analysis_file_path = row["analysis_file_path"]
        if not isinstance(analysis_file_path, str):
            continue
        analysis_file_name = os.path.basename(analysis_file_path)
        if analysis_file_name.startswith(base_name) and analysis_file_name.endswith("_embedding_behavior.pkl"):
            try:
                payload = load_pickle_standard(analysis_file_path)
                elev = payload["metadata"]["umap_view_angles"]["elev"]
                azim = payload["metadata"]["umap_view_angles"]["azim"]
                return float(elev), float(azim), analysis_file_path
            except Exception:
                return None, None, analysis_file_path

    return None, None, None


# =============================================================================
# Plotting
# =============================================================================

def style_3d_axis(ax, title):
    ax.set_title(title)
    ax.grid(False)

    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except Exception:
        pass

    try:
        ax.xaxis.pane.set_edgecolor("white")
        ax.yaxis.pane.set_edgecolor("white")
        ax.zaxis.pane.set_edgecolor("white")
    except Exception:
        pass

    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")


def plot_umap_with_lap_centroids(
    umap_embedding,
    umap_centroids,
    elev,
    azim,
    row,
    row_idx,
    meb_radius_norm
):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        umap_embedding[:, 2],
        c="lightgray",
        s=GRAY_POINT_SIZE,
        alpha=GRAY_ALPHA,
        rasterized=True
    )

    if len(umap_centroids) > 0:
        sc = ax.scatter(
            umap_centroids[:, 0],
            umap_centroids[:, 1],
            umap_centroids[:, 2],
            c=np.arange(len(umap_centroids)),
            cmap="YlGn_r",
            s=CENTROID_POINT_SIZE,
            alpha=CENTROID_ALPHA,
            edgecolors="black",
            linewidths=0.4
        )
        ax.plot(
            umap_centroids[:, 0],
            umap_centroids[:, 1],
            umap_centroids[:, 2],
            color="black",
            linewidth=1.0,
            alpha=0.6
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("lap index")

    title = (
        f"UMAP + lap centroids | row={row_idx} | "
        f"{row['dataset_name']} | {row['mouse_name']} | "
        f"{row['session_name']} | {row['depth_type']}\n"
        f"normalized MEB radius (orig space) = {meb_radius_norm:.6f}"
    )
    style_3d_axis(ax, title)
    ax.view_init(elev=elev, azim=azim)

    return fig


# =============================================================================
# Stats
# =============================================================================

def compute_dispersion_stats(centroids_orig, valid_trials):
    stats = {}

    if len(centroids_orig) == 0:
        stats["n_centroids"] = 0
        return stats

    n_dims = centroids_orig.shape[1]
    dim_norm = np.sqrt(n_dims)

    global_centroid = np.mean(centroids_orig, axis=0)
    d_to_global = np.linalg.norm(centroids_orig - global_centroid[None, :], axis=1)

    meb_center, meb_radius = minimum_enclosing_ball(centroids_orig)
    meb_radius_norm = meb_radius / dim_norm

    consec_d = consecutive_distances(centroids_orig)

    # normalized by dimensionality
    d_to_global_dimnorm = d_to_global / dim_norm
    consec_d_dimnorm = consec_d / dim_norm if len(consec_d) else np.array([], dtype=float)

    # normalized by MEB radius
    if meb_radius > 0:
        d_to_global_mebnorm = d_to_global / meb_radius
        consec_d_mebnorm = consec_d / meb_radius if len(consec_d) else np.array([], dtype=float)
    else:
        d_to_global_mebnorm = np.full_like(d_to_global, np.nan, dtype=float)
        consec_d_mebnorm = np.full_like(consec_d, np.nan, dtype=float) if len(consec_d) else np.array([], dtype=float)

    stats["n_centroids"] = int(len(centroids_orig))
    stats["n_dims"] = int(n_dims)
    stats["dim_norm_factor"] = float(dim_norm)

    stats["meb_center"] = meb_center
    stats["meb_radius"] = float(meb_radius)
    stats["meb_radius_norm"] = float(meb_radius_norm)

    stats["global_centroid"] = global_centroid

    # raw
    stats["dist_to_global"] = d_to_global
    stats["dist_to_global_mean"] = float(np.mean(d_to_global))
    stats["dist_to_global_std"] = float(np.std(d_to_global))
    stats["dist_to_global_max"] = float(np.max(d_to_global))

    stats["consecutive_distances"] = consec_d
    stats["consecutive_distances_mean"] = float(np.mean(consec_d)) if len(consec_d) else np.nan
    stats["consecutive_distances_std"] = float(np.std(consec_d)) if len(consec_d) else np.nan
    stats["consecutive_distances_sum"] = float(np.sum(consec_d)) if len(consec_d) else np.nan

    # dimension-normalized
    stats["dist_to_global_dimnorm"] = d_to_global_dimnorm
    stats["dist_to_global_dimnorm_mean"] = float(np.mean(d_to_global_dimnorm))
    stats["dist_to_global_dimnorm_std"] = float(np.std(d_to_global_dimnorm))
    stats["dist_to_global_dimnorm_max"] = float(np.max(d_to_global_dimnorm))

    stats["consecutive_distances_dimnorm"] = consec_d_dimnorm
    stats["consecutive_distances_dimnorm_mean"] = float(np.mean(consec_d_dimnorm)) if len(consec_d_dimnorm) else np.nan
    stats["consecutive_distances_dimnorm_std"] = float(np.std(consec_d_dimnorm)) if len(consec_d_dimnorm) else np.nan
    stats["consecutive_distances_dimnorm_sum"] = float(np.sum(consec_d_dimnorm)) if len(consec_d_dimnorm) else np.nan

    # MEB-normalized
    stats["dist_to_global_mebnorm"] = d_to_global_mebnorm
    stats["dist_to_global_mebnorm_mean"] = float(np.mean(d_to_global_mebnorm))
    stats["dist_to_global_mebnorm_std"] = float(np.std(d_to_global_mebnorm))
    stats["dist_to_global_mebnorm_max"] = float(np.max(d_to_global_mebnorm))

    stats["consecutive_distances_mebnorm"] = consec_d_mebnorm
    stats["consecutive_distances_mebnorm_mean"] = float(np.mean(consec_d_mebnorm)) if len(consec_d_mebnorm) else np.nan
    stats["consecutive_distances_mebnorm_std"] = float(np.std(consec_d_mebnorm)) if len(consec_d_mebnorm) else np.nan
    stats["consecutive_distances_mebnorm_sum"] = float(np.sum(consec_d_mebnorm)) if len(consec_d_mebnorm) else np.nan

    stats["lap_ids"] = valid_trials
    return stats


def stats_to_text(row, row_idx, modality_label, high_thr, low_thr, stats, prev_angles_source):
    lines = []
    lines.append("Lap-centroid dispersion statistics")
    lines.append("=" * 70)
    lines.append(f"row_idx: {row_idx}")
    lines.append(f"dataset_name: {row['dataset_name']}")
    lines.append(f"mouse_name: {row['mouse_name']}")
    lines.append(f"session_name: {row['session_name']}")
    lines.append(f"depth_type: {row['depth_type']}")
    lines.append(f"modality: {modality_label}")
    lines.append("")
    lines.append("trial_id_source: recomputed_full_lap_from_position")
    lines.append(f"high_thr: {high_thr}")
    lines.append(f"low_thr: {low_thr}")
    lines.append(f"previous_angle_source: {prev_angles_source}")
    lines.append("")

    if stats.get("n_centroids", 0) == 0:
        lines.append("No centroids available.")
        return "\n".join(lines)

    lines.append(f"n_centroids: {stats['n_centroids']}")
    lines.append(f"n_dims: {stats['n_dims']}")
    lines.append(f"dim_norm_factor: {stats['dim_norm_factor']:.10f}")
    lines.append("")

    lines.append(f"MEB radius (original space): {stats['meb_radius']:.10f}")
    lines.append(f"MEB radius normalized by sqrt(n_dims): {stats['meb_radius_norm']:.10f}")
    lines.append("")

    lines.append("Distance to global centroid (raw):")
    lines.append(f"  mean: {stats['dist_to_global_mean']:.10f}")
    lines.append(f"  std : {stats['dist_to_global_std']:.10f}")
    lines.append(f"  max : {stats['dist_to_global_max']:.10f}")
    lines.append("")

    lines.append("Distance to global centroid (dimension-normalized):")
    lines.append(f"  mean: {stats['dist_to_global_dimnorm_mean']:.10f}")
    lines.append(f"  std : {stats['dist_to_global_dimnorm_std']:.10f}")
    lines.append(f"  max : {stats['dist_to_global_dimnorm_max']:.10f}")
    lines.append("")

    lines.append("Distance to global centroid (MEB-normalized):")
    lines.append(f"  mean: {stats['dist_to_global_mebnorm_mean']:.10f}")
    lines.append(f"  std : {stats['dist_to_global_mebnorm_std']:.10f}")
    lines.append(f"  max : {stats['dist_to_global_mebnorm_max']:.10f}")
    lines.append("")

    lines.append("Consecutive centroid distances (raw):")
    lines.append(f"  mean: {stats['consecutive_distances_mean']}")
    lines.append(f"  std : {stats['consecutive_distances_std']}")
    lines.append(f"  sum : {stats['consecutive_distances_sum']}")
    lines.append("")

    lines.append("Consecutive centroid distances (dimension-normalized):")
    lines.append(f"  mean: {stats['consecutive_distances_dimnorm_mean']}")
    lines.append(f"  std : {stats['consecutive_distances_dimnorm_std']}")
    lines.append(f"  sum : {stats['consecutive_distances_dimnorm_sum']}")
    lines.append("")

    lines.append("Consecutive centroid distances (MEB-normalized):")
    lines.append(f"  mean: {stats['consecutive_distances_mebnorm_mean']}")
    lines.append(f"  std : {stats['consecutive_distances_mebnorm_std']}")
    lines.append(f"  sum : {stats['consecutive_distances_mebnorm_sum']}")
    lines.append("")

    lines.append("lap_ids:")
    lines.append(np.array2string(stats["lap_ids"], separator=", "))
    lines.append("")
    lines.append("dist_to_global:")
    lines.append(np.array2string(stats["dist_to_global"], precision=6, separator=", "))
    lines.append("")
    lines.append("dist_to_global_dimnorm:")
    lines.append(np.array2string(stats["dist_to_global_dimnorm"], precision=6, separator=", "))
    lines.append("")
    lines.append("dist_to_global_mebnorm:")
    lines.append(np.array2string(stats["dist_to_global_mebnorm"], precision=6, separator=", "))
    lines.append("")
    lines.append("consecutive_distances:")
    lines.append(np.array2string(stats["consecutive_distances"], precision=6, separator=", "))
    lines.append("")
    lines.append("consecutive_distances_dimnorm:")
    lines.append(np.array2string(stats["consecutive_distances_dimnorm"], precision=6, separator=", "))
    lines.append("")
    lines.append("consecutive_distances_mebnorm:")
    lines.append(np.array2string(stats["consecutive_distances_mebnorm"], precision=6, separator=", "))

    return "\n".join(lines)


# =============================================================================
# CSV schemas
# =============================================================================

ANALYSIS_INDEX_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "modality",
    "n_neurons",
    "n_timepoints",
    "n_laps",
    "analysis_pkl_path",
    "stats_txt_path",
    "figure_png_path",
    "figure_svg_path",
    "umap_model_path",
]

SUMMARY_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "modality",
    "n_neurons",
    "n_timepoints",
    "n_laps",
    "high_thr",
    "low_thr",

    "meb_radius_orig",
    "meb_radius_norm_orig",

    "dist_to_global_mean",
    "dist_to_global_std",
    "dist_to_global_max",
    "dist_to_global_dimnorm_mean",
    "dist_to_global_dimnorm_std",
    "dist_to_global_dimnorm_max",
    "dist_to_global_mebnorm_mean",
    "dist_to_global_mebnorm_std",
    "dist_to_global_mebnorm_max",

    "consecutive_centroid_dist_mean",
    "consecutive_centroid_dist_std",
    "consecutive_centroid_dist_sum",
    "consecutive_centroid_dist_dimnorm_mean",
    "consecutive_centroid_dist_dimnorm_std",
    "consecutive_centroid_dist_dimnorm_sum",
    "consecutive_centroid_dist_mebnorm_mean",
    "consecutive_centroid_dist_mebnorm_std",
    "consecutive_centroid_dist_mebnorm_sum",

    "analysis_pkl_path",
    "umap_model_path",
]


def init_csv(csv_path, columns):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_row(csv_path, columns, row_dict):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row_dict)


# =============================================================================
# Main per-row processing
# =============================================================================

def process_row(row_idx, row):
    standardized_file_path = row["standardized_file_path"]

    if not os.path.isfile(standardized_file_path):
        print(f"[WARNING] Missing file for row {row_idx}: {standardized_file_path}")
        return False, None, None

    data = load_pickle_standard(standardized_file_path)

    required_keys = ["signal", "pos", "time"]
    for key in required_keys:
        if key not in data:
            print(f"[WARNING] Row {row_idx}: missing key '{key}' in {standardized_file_path}")
            return False, None, None

    signal = ensure_2d(data["signal"])
    pos_1d = ensure_1d(get_pos_1d(data["pos"]))
    time_1d = ensure_1d(data["time"])

    n_time, n_neurons = signal.shape

    if len(pos_1d) != n_time:
        raise ValueError(f"Row {row_idx}: pos length mismatch")
    if len(time_1d) != n_time:
        raise ValueError(f"Row {row_idx}: time length mismatch")

    modality_label = infer_modality_label(row, standardized_file_path, data)

    signal_proc = signal.copy()

    if APPLY_GAUSSIAN_SMOOTHING:
        signal_proc = smooth_calcium_signals(signal_proc, sigma=SMOOTH_SIGMA)

    if APPLY_OUTLIER_FILTER_FOR_SIGNAL_POINTS:
        _, keep_idx = filter_noisy_outliers(signal_proc)
        signal_proc = signal_proc[keep_idx]
        pos_1d = pos_1d[keep_idx]
        time_1d = time_1d[keep_idx]

    # Always compute full-lap trial ID
    trial_id, high_thr, low_thr = compute_full_lap_trial_id(pos_1d)

    # Original-space centroids
    centroids_orig, valid_trials, trial_counts = compute_centroids_per_trial(signal_proc, trial_id)

    # Fit UMAP and save model
    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        random_state=UMAP_RANDOM_STATE,
    )
    umap_embedding = umap_model.fit_transform(signal_proc)
    umap_centroids = umap_model.transform(centroids_orig) if len(centroids_orig) else np.empty((0, 3))

    out_base = base_name_from_row(row)

    umap_model_path = os.path.join(MODEL_DIR, out_base + "_umap_model.pkl")
    save_pickle_standard(umap_model_path, umap_model)
    print(f"[OK] Saved: {umap_model_path}")

    # Load previous view angles
    elev, azim = DEFAULT_ELEV, DEFAULT_AZIM
    prev_angles_source = None
    if USE_PREVIOUS_VIEW_ANGLES:
        prev_elev, prev_azim, prev_source = load_previous_view_angles(PREV_ANALYSIS_INDEX_CSV, out_base)
        if prev_elev is not None and prev_azim is not None:
            elev, azim = prev_elev, prev_azim
            prev_angles_source = prev_source

    # Dispersion stats in original space only
    stats = compute_dispersion_stats(centroids_orig, valid_trials)

    # Plot
    meb_radius_norm = stats["meb_radius_norm"] if stats.get("n_centroids", 0) > 0 else np.nan
    fig = plot_umap_with_lap_centroids(
        umap_embedding=umap_embedding,
        umap_centroids=umap_centroids,
        elev=elev,
        azim=azim,
        row=row,
        row_idx=row_idx,
        meb_radius_norm=meb_radius_norm
    )
    figure_png_path, figure_svg_path = save_figure_both(fig, out_base + "_umap_lap_centroids")

    # Stats txt
    stats_txt = stats_to_text(
        row=row,
        row_idx=row_idx,
        modality_label=modality_label,
        high_thr=high_thr,
        low_thr=low_thr,
        stats=stats,
        prev_angles_source=prev_angles_source
    )
    stats_txt_path = save_text(stats_txt, out_base + "_umap_lap_centroids_stats")

    # Save data payload
    analysis_pkl_path = os.path.join(PKL_DIR, out_base + "_umap_lap_centroids.pkl")
    payload = {
        "metadata": {
            "dataset_name": row["dataset_name"],
            "mouse_name": row["mouse_name"],
            "session_name": row["session_name"],
            "depth_type": row["depth_type"],
            "modality": modality_label,
            "source_standardized_file_path": standardized_file_path,
            "n_timepoints": int(signal_proc.shape[0]),
            "n_neurons": int(signal_proc.shape[1]),
            "trial_id_source": "recomputed_full_lap_from_position",
            "high_thr": high_thr,
            "low_thr": low_thr,
            "umap_params": {
                "n_neighbors": UMAP_N_NEIGHBORS,
                "min_dist": UMAP_MIN_DIST,
                "n_components": UMAP_N_COMPONENTS,
                "random_state": UMAP_RANDOM_STATE,
            },
            "view_angles": {
                "elev": elev,
                "azim": azim,
            },
            "previous_view_angle_source": prev_angles_source,
            "umap_model_path": umap_model_path,
        },
        "behavior": {
            "pos": pos_1d,
            "time": time_1d,
            "trial_id_full_lap": trial_id,
        },
        "signal_original_or_preprocessed": signal_proc,
        "umap_embedding": umap_embedding,
        "centroids_original_space": centroids_orig,
        "centroids_umap_space": umap_centroids,
        "valid_trials": valid_trials,
        "trial_counts": trial_counts,
        "dispersion_stats_original_space": stats,
    }
    save_pickle_standard(analysis_pkl_path, payload)
    print(f"[OK] Saved: {analysis_pkl_path}")

    # Analysis index row
    analysis_index_row = {
        "dataset_name": row["dataset_name"],
        "mouse_name": row["mouse_name"],
        "session_name": row["session_name"],
        "depth_type": row["depth_type"],
        "modality": modality_label,
        "n_neurons": int(signal_proc.shape[1]),
        "n_timepoints": int(signal_proc.shape[0]),
        "n_laps": int(len(valid_trials)),
        "analysis_pkl_path": analysis_pkl_path,
        "stats_txt_path": stats_txt_path,
        "figure_png_path": figure_png_path,
        "figure_svg_path": figure_svg_path,
        "umap_model_path": umap_model_path,
    }

    # Summary CSV row
    summary_row = {
        "dataset_name": row["dataset_name"],
        "mouse_name": row["mouse_name"],
        "session_name": row["session_name"],
        "depth_type": row["depth_type"],
        "modality": modality_label,
        "n_neurons": int(signal_proc.shape[1]),
        "n_timepoints": int(signal_proc.shape[0]),
        "n_laps": int(len(valid_trials)),
        "high_thr": high_thr,
        "low_thr": low_thr,

        "meb_radius_orig": stats.get("meb_radius", np.nan),
        "meb_radius_norm_orig": stats.get("meb_radius_norm", np.nan),

        "dist_to_global_mean": stats.get("dist_to_global_mean", np.nan),
        "dist_to_global_std": stats.get("dist_to_global_std", np.nan),
        "dist_to_global_max": stats.get("dist_to_global_max", np.nan),
        "dist_to_global_dimnorm_mean": stats.get("dist_to_global_dimnorm_mean", np.nan),
        "dist_to_global_dimnorm_std": stats.get("dist_to_global_dimnorm_std", np.nan),
        "dist_to_global_dimnorm_max": stats.get("dist_to_global_dimnorm_max", np.nan),
        "dist_to_global_mebnorm_mean": stats.get("dist_to_global_mebnorm_mean", np.nan),
        "dist_to_global_mebnorm_std": stats.get("dist_to_global_mebnorm_std", np.nan),
        "dist_to_global_mebnorm_max": stats.get("dist_to_global_mebnorm_max", np.nan),

        "consecutive_centroid_dist_mean": stats.get("consecutive_distances_mean", np.nan),
        "consecutive_centroid_dist_std": stats.get("consecutive_distances_std", np.nan),
        "consecutive_centroid_dist_sum": stats.get("consecutive_distances_sum", np.nan),
        "consecutive_centroid_dist_dimnorm_mean": stats.get("consecutive_distances_dimnorm_mean", np.nan),
        "consecutive_centroid_dist_dimnorm_std": stats.get("consecutive_distances_dimnorm_std", np.nan),
        "consecutive_centroid_dist_dimnorm_sum": stats.get("consecutive_distances_dimnorm_sum", np.nan),
        "consecutive_centroid_dist_mebnorm_mean": stats.get("consecutive_distances_mebnorm_mean", np.nan),
        "consecutive_centroid_dist_mebnorm_std": stats.get("consecutive_distances_mebnorm_std", np.nan),
        "consecutive_centroid_dist_mebnorm_sum": stats.get("consecutive_distances_mebnorm_sum", np.nan),

        "analysis_pkl_path": analysis_pkl_path,
        "umap_model_path": umap_model_path,
    }

    return True, analysis_index_row, summary_row


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)
    ensure_dir(TXT_DIR)
    ensure_dir(PKL_DIR)
    ensure_dir(MODEL_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if DATASET_FILTER is not None:
        df = df[df["dataset_name"] == DATASET_FILTER].copy()

    df = df.reset_index(drop=True)

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    print(f"Loaded CSV with {len(df)} rows")
    print(f"Saving outputs to: {OUTPUT_DIR}")

    init_csv(ANALYSIS_INDEX_CSV, ANALYSIS_INDEX_COLUMNS)
    init_csv(SUMMARY_CSV, SUMMARY_COLUMNS)

    ok_count = 0
    fail_count = 0

    for row_idx, row in df.iterrows():
        try:
            success, analysis_index_row, summary_row = process_row(row_idx, row)
            if success:
                append_csv_row(ANALYSIS_INDEX_CSV, ANALYSIS_INDEX_COLUMNS, analysis_index_row)
                append_csv_row(SUMMARY_CSV, SUMMARY_COLUMNS, summary_row)
                ok_count += 1
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1
            print(f"[ERROR] Row {row_idx}: {e}")

    print("\nDone.")
    print(f"Successful entries: {ok_count}")
    print(f"Failed entries: {fail_count}")
    print(f"Summary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
