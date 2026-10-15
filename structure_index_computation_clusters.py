#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster-specific UMAP + SI + decoder pipeline
=============================================

Changes relative to previous version
------------------------------------
1) Adds an extra group:
      cluster_id = -1
      cluster_label = all_cells
   so you can compare cluster subsets against the full population.

2) Decoder tables now store BOTH:
      - r2
      - mse
   for regression targets.

3) Still runs:
      - learning training dataset first
      - then all transfer subsets

4) Still computes:
      - per-group UMAP
      - SI in UMAP space
      - decoder performance from neural activity

5) Saves:
      - cluster_structure_index.csv
      - cluster_decoder_xgboost_long.csv
      - cluster_si_and_decoder_combined.csv

Decoder targets
---------------
Regression:
- position
- time
- speed
- inner_time
- trial_id

Classification:
- direction
- position_direction
- position_binned
- time_binned
- speed_binned
- inner_time_binned
- trial_id_binned
"""

import os
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import KBinsDiscretizer

import umap
from sklearn.metrics import pairwise_distances
import traceback

try:
    from xgboost import XGBRegressor, XGBClassifier
except Exception as exc:
    raise ImportError("xgboost is required. Install with `pip install xgboost`.") from exc

# The SI implementation is a local module in the current project directory.
# Put that project directory first, even if this script is run from elsewhere.
import sys
SI_PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
if SI_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, SI_PROJECT_ROOT)

from structure_index import compute_structure_index
if not callable(compute_structure_index):
    raise TypeError("Imported compute_structure_index is not callable")
print(f"[SI import] module={compute_structure_index.__module__}; "
      f"callable={callable(compute_structure_index)}")


# =============================================================================
# PATHS
# =============================================================================

OLD_PROJECT_ROOT = "/home/melisamc/Documents/time_project_standarization"
PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"


def resolve_standardized_path(path):
    """Translate legacy master-index paths to the current machine's project root."""
    if pd.isna(path):
        return ""
    path = os.path.expanduser(str(path).strip())
    old_prefix = OLD_PROJECT_ROOT + os.sep
    if path.startswith(old_prefix):
        return os.path.join(PROJECT_ROOT, path[len(old_prefix):])
    return path
DATA_INDEX_CSV = os.path.join(PROJECT_ROOT, "data", "metadata", "master_dataset_index.csv")

TRAINING_LABELED_CSV = os.path.join(
    PROJECT_ROOT,
    "analysis",
    "mir_training_and_transfer",
    "training",
    "data",
    "learning_last_session_mir_labeled_neurons.csv",
)

TRANSFER_ROOT = os.path.join(
    PROJECT_ROOT,
    "analysis",
    "mir_training_and_transfer",
    "transfer",
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "analysis",
    "cluster_si_dual_space_outlier_filtered",
)

EMBED_DIR = os.path.join(OUT_DIR, "embeddings")
FIG_DIR = os.path.join(OUT_DIR, "figures")
STATS_DIR = os.path.join(OUT_DIR, "stats")
TABLE_DIR = os.path.join(OUT_DIR, "tables")


# =============================================================================
# PARAMETERS
# =============================================================================

UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 3
UMAP_RANDOM_STATE = 42

SMOOTH_SIGMA = 6
Z_SCORE_NEURONS = False
MIN_NEURONS_PER_GROUP = 15
MIN_TIMEPOINTS = 100
MIN_POINTS_FOR_ANALYSIS = 20

SI_N_BINS = 10
SI_N_NEIGHBORS = 50
SI_NUM_SHUFFLES = 20
SI_DISCRETE_N_BINS = 3
SI_PERC_NEIGH = 1

N_SPLITS = 5
DECODER_RANDOM_STATE = 42
POSITION_BINS_FOR_CLASSIFICATION = 20
TIME_BINS_FOR_CLASSIFICATION = 20
SPEED_BINS_FOR_CLASSIFICATION = 10
TRIAL_BINS_FOR_CLASSIFICATION = 10
INNER_TIME_BINS_FOR_CLASSIFICATION = 20
POSDIR_N_POS_BINS = 10

N_POS_BINS = 30
MIN_COUNT_PER_BIN = 5
PLANE_TILT_DEG = 30.0
TIME_TILT_DEG = 20.0

XGB_REG_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=DECODER_RANDOM_STATE,
    n_jobs=4,
)

XGB_CLS_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=DECODER_RANDOM_STATE,
    n_jobs=4,
)

UNASSIGNED_LABEL = 10
ALL_CELLS_LABEL = -1

GROUP_ORDER = [ALL_CELLS_LABEL, 2, 0, 1]
GROUP_LABELS = {
    ALL_CELLS_LABEL: "all_cells",
    0: "time",
    1: "direction",
    2: "position",
}
GROUP_COLORS = {
    ALL_CELLS_LABEL: "black",
    0: "tab:green",
    1: "tab:blue",
    2: "tab:purple",
}

FIG_DPI = 200


# =============================================================================
# UTILITIES
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def savefig_both(fig, out_base, dpi=FIG_DPI):
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")


def write_text(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(lines, str):
            f.write(lines)
        else:
            for line in lines:
                f.write(str(line) + "\n")


def zscore_cols(x):
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def maybe_extract_position(pos_array):
    pos_array = np.asarray(pos_array)
    if pos_array.ndim == 2:
        return pos_array[:, 0]
    return pos_array.squeeze()


def normalize_01(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return 0.0
    if np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return 0.0
    return np.corrcoef(a[ok], b[ok])[0, 1]


def discretize_for_classification(y, n_bins):
    y = np.asarray(y).reshape(-1, 1)
    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    yb = est.fit_transform(y).astype(int).ravel()
    return yb


def build_position_direction_labels(position, direction, n_pos_bins=10):
    pos_bin = discretize_for_classification(position, n_pos_bins)
    dir_vals = np.asarray(direction).astype(int)
    unique_dir = np.unique(dir_vals)
    dir_map = {cls: i for i, cls in enumerate(unique_dir)}
    dir_enc = np.array([dir_map[v] for v in dir_vals], dtype=int)
    n_dir = len(unique_dir)
    return pos_bin * n_dir + dir_enc


def get_direction_colors():
    cmap = ListedColormap(["blue", "gray", "lightblue"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    return cmap, norm


def format_session_id(dataset_name, mouse_name, session_name, depth_type, modality, subset_name):
    return f"{subset_name}__{dataset_name}__{mouse_name}__{session_name}__{depth_type}__{modality}"


def style_3d_axis(ax, title):
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("white")
        ax.yaxis.pane.set_edgecolor("white")
        ax.zaxis.pane.set_edgecolor("white")
    except Exception:
        pass
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


# =============================================================================
# LOADING
# =============================================================================

def load_master_index(csv_path):
    df = pd.read_csv(csv_path)
    required = ["dataset_name", "mouse_name", "session_name", "standardized_file_path"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column `{c}` in master dataset index")
    if "depth_type" not in df.columns:
        df["depth_type"] = "unknown"
    if "modality" not in df.columns:
        df["modality"] = "unknown"
    return df


def build_session_file_map(master_df):
    file_map = {}
    for _, row in master_df.iterrows():
        key = (
            str(row["dataset_name"]),
            str(row["mouse_name"]),
            str(row["session_name"]),
            str(row["depth_type"]),
        )
        file_map[key] = {
            "standardized_file_path": resolve_standardized_path(row["standardized_file_path"]),
            "modality": row.get("modality", "unknown"),
        }
    return file_map


def find_transfer_csvs(transfer_root):
    pattern = os.path.join(transfer_root, "*", "data", "*_mir_labeled_neurons.csv")
    return sorted(glob.glob(pattern))


def infer_subset_name_from_csv(csv_path):
    data_dir = os.path.dirname(csv_path)
    subset_dir = os.path.dirname(data_dir)
    return os.path.basename(subset_dir)


def collect_cluster_csvs_learning_first(training_csv, transfer_root):
    items = []
    if os.path.isfile(training_csv):
        items.append(("learning_training", training_csv))
    for csv_path in find_transfer_csvs(transfer_root):
        items.append((infer_subset_name_from_csv(csv_path), csv_path))
    return items


def load_cluster_csv(csv_path):
    df = pd.read_csv(csv_path)
    required = [
        "dataset_name",
        "mouse_name",
        "session_name",
        "depth_type",
        "cluster_id",
        "neuron_uid",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in cluster CSV {csv_path}: {missing}")

    if "modality" not in df.columns:
        df["modality"] = "unknown"

    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df = df[np.isfinite(df["cluster_id"])]
    df["cluster_id"] = df["cluster_id"].astype(int)
    return df


def load_standardized_session(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


# =============================================================================
# STANDARDIZED DATA HELPERS
# =============================================================================

def prepare_signal_matrix(data):
    signal = np.asarray(data["signal"])
    if signal.ndim != 2:
        raise ValueError(f"Signal must be 2D, got shape {signal.shape}")

    n0, n1 = signal.shape
    if n0 < n1 and n1 > MIN_TIMEPOINTS:
        signal = signal.T

    if SMOOTH_SIGMA and SMOOTH_SIGMA > 0:
        signal = gaussian_filter1d(signal, sigma=SMOOTH_SIGMA, axis=0)

    if Z_SCORE_NEURONS:
        signal = zscore_cols(signal)

    return signal


def get_behavior_dict(data):
    beh = {}
    beh["position"] = maybe_extract_position(data["pos"])
    beh["time"] = np.asarray(data["time"]).squeeze()

    if "mov_dir" in data:
        beh["direction"] = np.asarray(data["mov_dir"]).squeeze()
    else:
        beh["direction"] = np.zeros_like(beh["time"])

    if "speed" in data:
        beh["speed"] = np.asarray(data["speed"]).squeeze()
    else:
        pos = beh["position"]
        t = beh["time"]
        dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
        beh["speed"] = np.abs(np.gradient(pos, dt))

    if "inner_time" in data:
        beh["inner_time"] = np.asarray(data["inner_time"]).squeeze()
    else:
        beh["inner_time"] = normalize_01(np.arange(len(beh["time"])))

    if "trial_id" in data:
        beh["trial_id"] = np.asarray(data["trial_id"]).squeeze()
    else:
        beh["trial_id"] = np.zeros_like(beh["time"])

    beh["position_direction"] = np.column_stack([beh["position"], beh["direction"]])
    beh["position_direction_class"] = build_position_direction_labels(
        beh["position"],
        beh["direction"],
        n_pos_bins=POSDIR_N_POS_BINS,
    )
    return beh


def build_expected_neuron_uid_map(dataset_name, mouse_name, session_name, depth_type, n_neurons):
    uid_map = {}
    for neuron_index in range(n_neurons):
        uid = (
            f"{dataset_name}__{mouse_name}__{session_name}__{depth_type}"
            f"__row0__neuron{neuron_index}"
        )
        uid_map[uid] = neuron_index
    return uid_map


def get_neuron_uid_map(data, dataset_name, mouse_name, session_name, depth_type, n_neurons):
    for key in ["neuron_uid", "cell_ids", "unit_ids", "neuron_ids"]:
        if key in data:
            ids = np.asarray(data[key]).astype(str)
            if len(ids) == n_neurons:
                return {ids[i]: i for i in range(n_neurons)}

    return build_expected_neuron_uid_map(
        dataset_name,
        mouse_name,
        session_name,
        depth_type,
        n_neurons,
    )


# =============================================================================
# VIEW SELECTION
# =============================================================================

def compute_position_direction_plane_normal(embedding, pos_1d, mov_dir, n_pos_bins=30, min_count_per_bin=5):
    embedding = np.asarray(embedding)
    pos_1d = np.asarray(pos_1d).reshape(-1)
    mov_dir = np.asarray(mov_dir).reshape(-1)

    valid_mask = np.isfinite(pos_1d) & np.isfinite(mov_dir)
    for d in range(3):
        valid_mask &= np.isfinite(embedding[:, d])

    embedding = embedding[valid_mask]
    pos_1d = pos_1d[valid_mask]
    mov_dir = mov_dir[valid_mask]

    pos_min = np.min(pos_1d)
    pos_max = np.max(pos_1d)
    if pos_max <= pos_min:
        raise ValueError("Position has zero range.")

    bin_edges = np.linspace(pos_min, pos_max, n_pos_bins + 1)

    centroids = []
    for d in [-1, 1]:
        mask_d = mov_dir == d
        if np.sum(mask_d) == 0:
            continue
        pos_d = pos_1d[mask_d]
        emb_d = embedding[mask_d]
        bin_ids = np.digitize(pos_d, bin_edges[1:-1], right=False)

        for b in range(n_pos_bins):
            idx = np.where(bin_ids == b)[0]
            if len(idx) >= min_count_per_bin:
                centroids.append(np.mean(emb_d[idx], axis=0))

    centroids = np.asarray(centroids)
    if centroids.shape[0] < 3:
        raise ValueError("Not enough centroids to estimate plane.")

    center = np.mean(centroids, axis=0)
    centroids_centered = centroids - center

    pca = PCA(n_components=3)
    pca.fit(centroids_centered)

    basis = pca.components_.copy()
    normal = basis[2].copy()
    normal = normal / np.linalg.norm(normal)
    return normal


def normal_to_view_angles(normal, tilt_deg=30.0):
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    x, y, z = normal
    azim = np.degrees(np.arctan2(y, x))
    elev = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
    elev = elev - tilt_deg
    return float(elev), float(azim)


def compute_position_group_view(embedding, position, direction):
    try:
        normal = compute_position_direction_plane_normal(
            embedding,
            position,
            direction,
            n_pos_bins=N_POS_BINS,
            min_count_per_bin=MIN_COUNT_PER_BIN,
        )
        elev, azim = normal_to_view_angles(normal, tilt_deg=PLANE_TILT_DEG)
        return elev, azim
    except Exception:
        return 10.0, -60.0


def compute_time_group_view(embedding, time_vec):
    emb = np.asarray(embedding)
    t = np.asarray(time_vec).reshape(-1)

    ok = np.isfinite(t) & np.isfinite(emb).all(axis=1)
    if ok.sum() < 30:
        return 20.0, -35.0

    emb = emb[ok]
    t = normalize_01(t[ok])

    center = emb.mean(axis=0, keepdims=True)
    emb_c = emb - center

    pca = PCA(n_components=3)
    emb_pca = pca.fit_transform(emb_c)

    corrs = np.array([abs(safe_corr(emb_pca[:, i], t)) for i in range(3)])
    time_axis = int(np.argmax(corrs))

    vec = pca.components_[time_axis]
    vec = vec / np.linalg.norm(vec)

    x, y, z = vec
    azim = np.degrees(np.arctan2(y, x))
    elev = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
    elev = elev - TIME_TILT_DEG
    return float(elev), float(azim)


def get_group_view(group_id, emb, behavior):
    if group_id in [ALL_CELLS_LABEL, 2]:
        return compute_position_group_view(emb, behavior["position"], behavior["direction"])
    if group_id == 0:
        return compute_time_group_view(emb, behavior["time"])
    return 22.0, -50.0


# =============================================================================
# SI
# =============================================================================

def filter_noisy_outliers(data):
    """Exact algorithm from the LT/ROT code; runs once per session on FULL signal.

    Warning: computes an NxN pairwise distance matrix (O(N^2) memory/time).
    """
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D, 1), axis=1) - 1
    noise_idx = np.where(nn_dist < np.percentile(nn_dist, 20))[0]
    signal_idx = np.where(nn_dist >= np.percentile(nn_dist, 20))[0]
    return noise_idx, signal_idx


def compute_si_wrapper(space_data, labels, label_name, session_name=''):
    """Use same timepoint mask for neural and UMAP SI.

    Note: position_direction uses mixed categorical semantics here, unlike the
    legacy LT/ROT code that set discrete_label=False for the joint target.
    """
    x = np.asarray(space_data, dtype=float)
    y = np.asarray(labels)
    good = np.isfinite(x).all(axis=1)
    good &= np.isfinite(y).all(axis=1) if y.ndim == 2 else np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < max(MIN_POINTS_FOR_ANALYSIS, SI_N_NEIGHBORS + 2):
        return np.nan, np.nan, 'insufficient_valid_samples'
    params = dict(n_bins=SI_N_BINS, n_neighbors=SI_N_NEIGHBORS,
                  num_shuffles=SI_NUM_SHUFFLES, verbose=False,
                  continuity_kernel=None, perc_neigh=SI_PERC_NEIGH)
    if label_name == 'direction':
        params.update(n_bins=SI_DISCRETE_N_BINS, discrete_label=True)
    elif label_name == 'position_direction':
        params.update(discrete_label=[False, True])
    else:
        params.update(discrete_label=False)
    if label_name == 'trial_id':
        params.update(min_label=float(np.min(y)), max_label=float(np.max(y)))
    try:
        result = compute_structure_index(x, y, **params)
        si = float(result[0]) if isinstance(result, (tuple, list)) else float(result)
        sh = np.asarray(result[3], dtype=float) if isinstance(result, (tuple, list)) and len(result) > 3 else np.array([])
        return si, (float(np.nanmean(sh)) if sh.size else np.nan), 'ok'
    except Exception as exc:
        return np.nan, np.nan, f'{type(exc).__name__}: {exc}'


# =============================================================================
# DECODERS
# =============================================================================

def decode_regression(x, y, n_splits=5):
    x = np.asarray(x)
    y = np.asarray(y)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if x.shape[0] < max(30, n_splits * 5):
        return np.nan, np.nan

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=DECODER_RANDOM_STATE)

    r2_list = []
    mse_list = []

    for tr, te in kf.split(x):
        model = XGBRegressor(**XGB_REG_PARAMS)
        model.fit(x[tr], y[tr])
        pred = model.predict(x[te])
        r2_list.append(r2_score(y[te], pred))
        mse_list.append(mean_squared_error(y[te], pred))

    return float(np.mean(r2_list)), float(np.mean(mse_list))


def decode_classification(x, y, n_splits=5):
    x = np.asarray(x)
    y = np.asarray(y)

    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if x.shape[0] < max(30, n_splits * 5):
        return np.nan, np.nan

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return np.nan, np.nan

    class_to_int = {cls: i for i, cls in enumerate(unique_classes)}
    y_enc = np.array([class_to_int[v] for v in y], dtype=int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=DECODER_RANDOM_STATE)

    acc_list, bacc_list = [], []
    params = dict(XGB_CLS_PARAMS)
    params["num_class"] = len(unique_classes)

    for tr, te in skf.split(x, y_enc):
        model = XGBClassifier(**params)
        model.fit(x[tr], y_enc[tr])
        pred = model.predict(x[te])
        acc_list.append(accuracy_score(y_enc[te], pred))
        bacc_list.append(balanced_accuracy_score(y_enc[te], pred))

    return float(np.mean(acc_list)), float(np.mean(bacc_list))


def run_decoder_suite(signal_t_by_n, behavior):
    x = np.asarray(signal_t_by_n)
    out = []

    reg_targets = [
        ("position", behavior["position"]),
        ("time", behavior["time"]),
        ("speed", behavior["speed"]),
        ("inner_time", behavior["inner_time"]),
        ("trial_id", behavior["trial_id"]),
    ]

    for name, y in reg_targets:
        try:
            r2, mse = decode_regression(x, y, n_splits=N_SPLITS)
        except Exception:
            r2, mse = np.nan, np.nan

        out.append(dict(
            behavior=name,
            decoder_task="regression",
            metric_name="r2",
            metric_value=r2,
        ))
        out.append(dict(
            behavior=name,
            decoder_task="regression",
            metric_name="mse",
            metric_value=mse,
        ))

    class_targets = [
        ("direction", behavior["direction"]),
        ("position_direction", behavior["position_direction_class"]),
        ("position_binned", discretize_for_classification(behavior["position"], POSITION_BINS_FOR_CLASSIFICATION)),
        ("time_binned", discretize_for_classification(behavior["time"], TIME_BINS_FOR_CLASSIFICATION)),
        ("speed_binned", discretize_for_classification(behavior["speed"], SPEED_BINS_FOR_CLASSIFICATION)),
        ("inner_time_binned", discretize_for_classification(behavior["inner_time"], INNER_TIME_BINS_FOR_CLASSIFICATION)),
        ("trial_id_binned", discretize_for_classification(behavior["trial_id"], TRIAL_BINS_FOR_CLASSIFICATION)),
    ]

    for name, y in class_targets:
        try:
            acc, bacc = decode_classification(x, y, n_splits=N_SPLITS)
        except Exception:
            acc, bacc = np.nan, np.nan

        out.append(dict(
            behavior=name,
            decoder_task="classification",
            metric_name="balanced_accuracy",
            metric_value=bacc,
        ))
        out.append(dict(
            behavior=name,
            decoder_task="classification",
            metric_name="accuracy",
            metric_value=acc,
        ))

    return out


# =============================================================================
# PLOTTING
# =============================================================================

def scatter_umap(ax, emb, values, mode, title, elev, azim, s=4):
    emb = np.asarray(emb)

    if mode == "direction":
        cmap, norm = get_direction_colors()
        sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=values, cmap=cmap, norm=norm, s=s, alpha=0.9)
    else:
        sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=values, cmap=mode, s=s, alpha=0.9)

    style_3d_axis(ax, title)
    ax.view_init(elev=elev, azim=azim)
    return sc


def plot_nine_panel_umaps(session_id, emb_by_group, behavior, out_base):
    fig = plt.figure(figsize=(14, 14))
    plot_specs = [
        ("position", behavior["position"], "magma"),
        ("time", behavior["time"], "YlGn_r"),
        ("direction", behavior["direction"], "direction"),
    ]

    rows_for_plot = [0, 1, 2]
    for i, group_id in enumerate(rows_for_plot):
        emb = emb_by_group.get(group_id, None)
        if emb is None:
            continue

        elev, azim = get_group_view(group_id, emb, behavior)
        row_name = GROUP_LABELS[group_id]

        for j, (label_name, vals, mode) in enumerate(plot_specs):
            ax = fig.add_subplot(3, 3, i * 3 + j + 1, projection="3d")
            sc = scatter_umap(
                ax, emb, vals, mode,
                title=f"{row_name} | {label_name}",
                elev=elev, azim=azim, s=5,
            )
            if mode == "direction":
                cb = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.02, ticks=[-1, 0, 1])
            else:
                cb = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.02)
            cb.ax.tick_params(labelsize=8)

    fig.suptitle(session_id, fontsize=13)
    fig.tight_layout()
    savefig_both(fig, out_base)
    plt.close(fig)


# =============================================================================
# CORE ANALYSIS
# =============================================================================

def analyze_one_session(subset_name, session_meta, standardized_path, cluster_df_session):
    dataset_name = session_meta['dataset_name']
    mouse_name = session_meta['mouse_name']
    session_name = session_meta['session_name']
    depth_type = session_meta['depth_type']
    modality = session_meta['modality']
    session_id = format_session_id(dataset_name, mouse_name, session_name,
                                   depth_type, modality, subset_name)
    data = load_standardized_session(standardized_path)
    # Validate against behavioral length; never transpose based merely on shape.
    raw = np.asarray(data['signal'])
    behavior = get_behavior_dict(data)
    n_time_expected = len(behavior['time'])
    if raw.ndim != 2:
        raise ValueError(f'signal must be 2D: {raw.shape}')
    if raw.shape[0] == n_time_expected:
        pass
    elif raw.shape[1] == n_time_expected:
        raw = raw.T
    else:
        raise ValueError(f'signal {raw.shape} incompatible with behavior length {n_time_expected}')
    signal = gaussian_filter1d(raw, sigma=SMOOTH_SIGMA, axis=0) if SMOOTH_SIGMA > 0 else raw.copy()
    if Z_SCORE_NEURONS:
        signal = zscore_cols(signal)
    n_time, n_neurons = signal.shape
    if not np.isfinite(signal).all():
        # The legacy filter does not handle nonfinite activity, fail visibly.
        raise ValueError('Nonfinite neural signal; review standardized file')
    print(f'[outliers] {session_id}: computing exact full-population distances for {n_time} samples')
    noise_idx, signal_idx = filter_noisy_outliers(signal)
    if len(signal_idx) < max(UMAP_N_COMPONENTS + 2, SI_N_NEIGHBORS + 2):
        raise ValueError(f'Outlier filter retained too few samples: {len(signal_idx)}')
    print(f'[outliers] retained {len(signal_idx)}/{n_time}, removed {len(noise_idx)}')

    # Use the saved neuron_index, which is indexed within the matching metadata row.
    # Ambiguous multi-row groups cannot be safely resolved with neuron_index alone.
    row_values = cluster_df_session['row_index'].dropna().unique() if 'row_index' in cluster_df_session else []
    if len(row_values) > 1:
        raise ValueError(f'Multiple row_index values in session/depth ({row_values}); cannot safely map neurons')
    indices = {ALL_CELLS_LABEL: np.arange(n_neurons, dtype=int)}
    for gid in [0, 1, 2]:
        sub = cluster_df_session[cluster_df_session['cluster_id'] == gid]
        if 'neuron_index' in sub.columns:
            idx = pd.to_numeric(sub['neuron_index'], errors='coerce').dropna().astype(int).to_numpy()
        else:
            idx = np.array([int(s.rsplit('__neuron', 1)[1]) for s in sub['neuron_uid']
                            if '__neuron' in str(s)], dtype=int)
        if len(idx) and (np.min(idx) < 0 or np.max(idx) >= n_neurons):
            raise ValueError(f'Out-of-range neuron index for group {gid}: n_neurons={n_neurons}')
        indices[gid] = np.unique(idx)

    si_rows, dec_rows, embeddings = [], [], {}
    base = dict(subset_name=subset_name, dataset_name=dataset_name,
                mouse_name=mouse_name, session_name=session_name,
                depth_type=depth_type, modality=modality)
    targets = {key: behavior[key] for key in
               ['position', 'position_direction', 'direction', 'speed', 'time', 'inner_time', 'trial_id']}
    for gid in GROUP_ORDER:
        idx = indices[gid]
        print(f'[group] {session_id} {GROUP_LABELS[gid]}: {len(idx)} neurons')
        if len(idx) < MIN_NEURONS_PER_GROUP:
            continue
        group_signal = signal[:, idx]
        try:
            emb = umap.UMAP(n_neighbors=min(UMAP_N_NEIGHBORS, n_time - 1),
                            min_dist=UMAP_MIN_DIST, n_components=UMAP_N_COMPONENTS,
                            random_state=UMAP_RANDOM_STATE).fit_transform(group_signal)
            embeddings[gid] = emb
            with open(os.path.join(EMBED_DIR, f'{session_id}__group_{gid}_umap.pkl'), 'wb') as f:
                pickle.dump(dict(embedding=emb, indices=idx, behavior=behavior,
                                 session=base, group_id=gid, si_valid_idx=signal_idx), f)
        except Exception as exc:
            print(f'[UMAP ERROR] {session_id} {gid}: {exc}')
            emb = None

        for space, features in [('neural', group_signal), ('umap', emb)]:
            for beh_name, label in targets.items():
                if features is None:
                    si, sh, status = np.nan, np.nan, 'umap_not_available'
                else:
                    si, sh, status = compute_si_wrapper(
                        features[signal_idx], np.asarray(label)[signal_idx], beh_name, session_name)
                si_rows.append(dict(**base, group_id=gid, group_label=GROUP_LABELS[gid],
                                    n_neurons=len(idx), n_timepoints=n_time,
                                    n_si_samples=len(signal_idx), si_space=space,
                                    behavior=beh_name, si_value=si,
                                    si_shuffle_mean=sh, si_status=status))
                if status != 'ok':
                    print(f'[SI] {session_id} {gid} {space} {beh_name}: {status}')

        # Decoder unchanged: features = neural activity, not UMAP; run on original timepoints.
        try:
            decoder_results = run_decoder_suite(group_signal, behavior)
        except Exception as exc:
            print(f'[DECODER ERROR] {session_id} group {gid}: {exc}')
            decoder_results = []
        for res in decoder_results:
            dec_rows.append(dict(**base, group_id=gid, group_label=GROUP_LABELS[gid],
                                 n_neurons=len(idx), n_timepoints=n_time,
                                 decoder_name='xgboost', **res))

    if embeddings:
        try:
            plot_nine_panel_umaps(session_id, embeddings, behavior,
                                  os.path.join(FIG_DIR, 'umap_9panel', session_id))
        except Exception as exc:
            print(f'[PLOT ERROR] {session_id}: {exc}')
    return si_rows, dec_rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    for path in [
        OUT_DIR,
        EMBED_DIR,
        FIG_DIR,
        STATS_DIR,
        TABLE_DIR,
        os.path.join(FIG_DIR, "umap_9panel"),
    ]:
        ensure_dir(path)

    master_df = load_master_index(DATA_INDEX_CSV)
    session_file_map = build_session_file_map(master_df)
    cluster_csv_items = collect_cluster_csvs_learning_first(TRAINING_LABELED_CSV, TRANSFER_ROOT)

    if len(cluster_csv_items) == 0:
        raise FileNotFoundError("No cluster CSVs found.")

    all_si_rows = []
    all_dec_rows = []

    print("=" * 70)
    print("Cluster UMAP + SI + decoder from MIR clusters + all_cells")
    print("=" * 70)
    print("Processing order:")
    for subset_name, csv_path in cluster_csv_items:
        print(f"  - {subset_name}: {csv_path}")

    for subset_name, csv_path in cluster_csv_items:
        print(f"\n=== Processing subset: {subset_name} ===")
        df_clusters = load_cluster_csv(csv_path)

        group_cols = ["dataset_name", "mouse_name", "session_name", "depth_type"]
        for session_key, cluster_sub in df_clusters.groupby(group_cols, dropna=False):
            dataset_name, mouse_name, session_name, depth_type = session_key
            map_key = (str(dataset_name), str(mouse_name), str(session_name), str(depth_type))

            if map_key not in session_file_map:
                print(f"[missing standardized file entry] {map_key}")
                continue

            standardized_path = session_file_map[map_key]["standardized_file_path"]
            modality = session_file_map[map_key]["modality"]

            if not os.path.exists(standardized_path):
                print(f"[file not found] {standardized_path}")
                continue

            session_meta = dict(
                dataset_name=str(dataset_name),
                mouse_name=str(mouse_name),
                session_name=str(session_name),
                depth_type=str(depth_type),
                modality=str(modality),
            )

            print(
                "Processing session: "
                f"{session_meta['dataset_name']} | {session_meta['mouse_name']} | "
                f"{session_meta['session_name']} | {session_meta['depth_type']}"
            )

            try:
                si_rows, dec_rows = analyze_one_session(
                    subset_name,
                    session_meta,
                    standardized_path,
                    cluster_sub,
                )
                all_si_rows.extend(si_rows)
                all_dec_rows.extend(dec_rows)
            except Exception as exc:
                print(f"[ERROR] {session_key}: {exc}")

    si_df = pd.DataFrame(all_si_rows)
    dec_df = pd.DataFrame(all_dec_rows)

    si_csv = os.path.join(TABLE_DIR, "cluster_structure_index_dual_space.csv")
    dec_csv = os.path.join(TABLE_DIR, "cluster_decoder_xgboost_long.csv")

    si_df.to_csv(si_csv, index=False)
    dec_df.to_csv(dec_csv, index=False)

    # Convenience wide merge
    if (not si_df.empty) and (not dec_df.empty):
        combo = pd.merge(
            si_df,
            dec_df,
            on=[
                "subset_name", "dataset_name", "mouse_name", "session_name",
                "depth_type", "modality", "group_id", "group_label",
                "n_neurons", "n_timepoints", "behavior"
            ],
            how="outer",
        )
        combo.to_csv(os.path.join(TABLE_DIR, "cluster_si_and_decoder_combined_dual_space.csv"), index=False)

    print("\nDone.")
    print(f"SI table: {si_csv}")
    print(f"Decoder table: {dec_csv}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
