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

try:
    from xgboost import XGBRegressor, XGBClassifier
except Exception as exc:
    raise ImportError("xgboost is required. Install with `pip install xgboost`.") from exc

try:
    from structure_index import compute_structure_index
except Exception:
    compute_structure_index = None
    warnings.warn(
        "Could not import `compute_structure_index` from structure_index. "
        "Update the import to your local SI package if needed."
    )


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = "/home/melisamc/Documents/time_project_standarization"
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
    "cluster_umap_si_decoder_with_all_cells_and_mse",
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
SI_NUM_SHUFFLES = 0
SI_DISCRETE_N_BINS = 2
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

GROUP_ORDER = [ALL_CELLS_LABEL, 0, 1, 2]
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
            "standardized_file_path": row["standardized_file_path"],
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

def compute_si_wrapper(emb, label, label_name):
    if compute_structure_index is None:
        return np.nan

    emb = np.asarray(emb)
    label = np.asarray(label)
    ok = np.isfinite(emb).all(axis=1)

    if label.ndim == 1:
        ok &= np.isfinite(label)
        y = label[ok]
    else:
        ok &= np.isfinite(label).all(axis=1)
        y = label[ok]
    x = emb[ok]

    if x.shape[0] < MIN_POINTS_FOR_ANALYSIS:
        return np.nan

    if label_name == "direction":
        params = dict(
            n_bins=SI_DISCRETE_N_BINS,
            discrete_label=True,
            n_neighbors=SI_N_NEIGHBORS,
            num_shuffles=SI_NUM_SHUFFLES,
        )
    elif label_name == "position_direction":
        params = dict(
            n_bins=SI_N_BINS,
            discrete_label=[False, True],
            n_neighbors=SI_N_NEIGHBORS,
            num_shuffles=SI_NUM_SHUFFLES,
        )
    else:
        params = dict(
            n_bins=SI_N_BINS,
            discrete_label=False,
            continuity_kernel=None,
            perc_neigh=SI_PERC_NEIGH,
            n_neighbors=SI_N_NEIGHBORS,
            num_shuffles=SI_NUM_SHUFFLES,
        )

    try:
        out = compute_structure_index(x, y, **params)
        if isinstance(out, (tuple, list, np.ndarray)):
            return float(np.asarray(out[0]).squeeze())
        return float(np.asarray(out).squeeze())
    except Exception:
        return np.nan


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
    dataset_name = session_meta["dataset_name"]
    mouse_name = session_meta["mouse_name"]
    session_name = session_meta["session_name"]
    depth_type = session_meta["depth_type"]
    modality = session_meta["modality"]
    session_id = format_session_id(dataset_name, mouse_name, session_name, depth_type, modality, subset_name)

    data = load_standardized_session(standardized_path)
    signal = prepare_signal_matrix(data)
    behavior = get_behavior_dict(data)

    n_time, n_neurons = signal.shape
    uid_map = get_neuron_uid_map(data, dataset_name, mouse_name, session_name, depth_type, n_neurons)

    # Match neurons for the explicit MIR clusters
    matched_indices = {}
    matched_uids = {}

    for group_id in [0, 1, 2]:
        sub = cluster_df_session[cluster_df_session["cluster_id"] == group_id].copy()
        keep_idx = []
        keep_uid = []

        for uid in sub["neuron_uid"].astype(str).values:
            if uid in uid_map:
                keep_idx.append(uid_map[uid])
                keep_uid.append(uid)
            elif "__neuron" in uid:
                try:
                    neuron_index = int(uid.split("__neuron")[-1])
                    if 0 <= neuron_index < n_neurons:
                        keep_idx.append(neuron_index)
                        keep_uid.append(uid)
                except Exception:
                    pass

        keep_idx = np.array(sorted(set(keep_idx)), dtype=int)
        matched_indices[group_id] = keep_idx
        matched_uids[group_id] = keep_uid

    # Add all-cells group
    matched_indices[ALL_CELLS_LABEL] = np.arange(n_neurons, dtype=int)
    matched_uids[ALL_CELLS_LABEL] = [f"all_cells__{i}" for i in range(n_neurons)]

    emb_by_group = {}
    si_rows = []
    dec_rows = []

    for group_id in GROUP_ORDER:
        keep_idx = matched_indices[group_id]
        keep_uid = matched_uids[group_id]

        print(f"[info] {session_id} group {GROUP_LABELS[group_id]}: matched neurons = {len(keep_idx)}")

        if len(keep_idx) < MIN_NEURONS_PER_GROUP:
            print(f"[skip] {session_id} group {GROUP_LABELS[group_id]}: only {len(keep_idx)} neurons")
            continue

        group_signal = signal[:, keep_idx]

        reducer = umap.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            n_components=UMAP_N_COMPONENTS,
            random_state=UMAP_RANDOM_STATE,
        )
        emb = reducer.fit_transform(group_signal)
        emb_by_group[group_id] = emb

        with open(os.path.join(EMBED_DIR, f"{session_id}__group_{group_id}_umap.pkl"), "wb") as f:
            pickle.dump(
                dict(
                    subset_name=subset_name,
                    dataset_name=dataset_name,
                    mouse_name=mouse_name,
                    session_name=session_name,
                    depth_type=depth_type,
                    modality=modality,
                    group_id=group_id,
                    group_label=GROUP_LABELS[group_id],
                    neuron_indices=keep_idx,
                    neuron_uids=keep_uid,
                    embedding=emb,
                    behavior=behavior,
                ),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        si_targets = {
            "position": behavior["position"],
            "time": behavior["time"],
            "direction": behavior["direction"],
            "speed": behavior["speed"],
            "inner_time": behavior["inner_time"],
            "trial_id": behavior["trial_id"],
            "position_direction": behavior["position_direction"],
        }

        for beh_name, label in si_targets.items():
            try:
                si_val = compute_si_wrapper(emb, label, beh_name)
            except Exception:
                si_val = np.nan

            si_rows.append(dict(
                subset_name=subset_name,
                dataset_name=dataset_name,
                mouse_name=mouse_name,
                session_name=session_name,
                depth_type=depth_type,
                modality=modality,
                group_id=group_id,
                group_label=GROUP_LABELS[group_id],
                n_neurons=len(keep_idx),
                n_timepoints=n_time,
                behavior=beh_name,
                si_value=si_val,
            ))

        decoder_results = run_decoder_suite(group_signal, behavior)
        for res in decoder_results:
            dec_rows.append(dict(
                subset_name=subset_name,
                dataset_name=dataset_name,
                mouse_name=mouse_name,
                session_name=session_name,
                depth_type=depth_type,
                modality=modality,
                group_id=group_id,
                group_label=GROUP_LABELS[group_id],
                n_neurons=len(keep_idx),
                n_timepoints=n_time,
                decoder_name="xgboost",
                behavior=res["behavior"],
                decoder_task=res["decoder_task"],
                metric_name=res["metric_name"],
                metric_value=res["metric_value"],
            ))

    if len(emb_by_group) > 0:
        plot_nine_panel_umaps(
            session_id,
            emb_by_group,
            behavior,
            os.path.join(FIG_DIR, "umap_9panel", session_id),
        )

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

    si_csv = os.path.join(TABLE_DIR, "cluster_structure_index.csv")
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
        combo.to_csv(os.path.join(TABLE_DIR, "cluster_si_and_decoder_combined.csv"), index=False)

    print("\nDone.")
    print(f"SI table: {si_csv}")
    print(f"Decoder table: {dec_csv}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()

###################### PLOT ##########################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot SI vs decoder performance
==============================

This script reads:
- cluster_structure_index.csv
- cluster_decoder_xgboost_long.csv

from:
    analysis/cluster_umap_si_decoder_with_all_cells_and_mse/tables/

and generates SI-vs-decoder scatter plots for:
1) all datasets combined
2) each dataset separately

For regression targets, it plots both:
- SI vs R2
- SI vs MSE

For classification targets, it plots:
- SI vs balanced_accuracy
- SI vs accuracy

Group colors:
- all_cells: black
- time: green
- direction: blue
- position: purple
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = "/home/melisamc/Documents/time_project_standarization"
ANALYSIS_DIR = os.path.join(
    PROJECT_ROOT,
    "analysis",
    "cluster_umap_si_decoder_with_all_cells_and_mse",
)

TABLE_DIR = os.path.join(ANALYSIS_DIR, "tables")
SI_CSV = os.path.join(TABLE_DIR, "cluster_structure_index.csv")
DECODER_CSV = os.path.join(TABLE_DIR, "cluster_decoder_xgboost_long.csv")

OUT_DIR = os.path.join(ANALYSIS_DIR, "figures_si_vs_decoder")
COMBINED_DIR = os.path.join(OUT_DIR, "all_datasets_combined")
PER_DATASET_DIR = os.path.join(OUT_DIR, "per_dataset")


# =============================================================================
# STYLING
# =============================================================================

GROUP_ORDER = [-1, 0, 1, 2]
GROUP_LABELS = {
    -1: "all_cells",
    0: "time",
    1: "direction",
    2: "position",
}
GROUP_COLORS = {
    -1: "black",
    0: "tab:green",
    1: "tab:blue",
    2: "tab:purple",
}

FIG_DPI = 200


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def savefig_both(fig, out_base, dpi=FIG_DPI):
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def sanitize_for_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def merge_si_decoder(si_df, dec_df):
    merge_cols = [
        "subset_name",
        "dataset_name",
        "mouse_name",
        "session_name",
        "depth_type",
        "modality",
        "group_id",
        "group_label",
        "behavior",
    ]
    return pd.merge(si_df, dec_df, on=merge_cols, how="inner")


def plot_scatter(df, dataset_name, out_dir):
    if df.empty:
        return

    metrics = sorted(df["metric_name"].dropna().unique())
    behaviors = sorted(df["behavior"].dropna().unique())

    for metric_name in metrics:
        sub_metric = df[df["metric_name"] == metric_name].copy()
        if sub_metric.empty:
            continue

        for behavior in behaviors:
            sub = sub_metric[sub_metric["behavior"] == behavior].copy()
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(6, 5))

            for gid in GROUP_ORDER:
                ss = sub[sub["group_id"] == gid]
                if ss.empty:
                    continue
                ax.scatter(
                    ss["si_value"],
                    ss["metric_value"],
                    s=36,
                    alpha=0.8,
                    color=GROUP_COLORS[gid],
                    label=GROUP_LABELS[gid],
                )

            ax.set_xlabel("SI")
            ax.set_ylabel(metric_name)
            if dataset_name is None:
                ax.set_title(f"All datasets | {behavior} | {metric_name}")
                out_name = f"all_datasets__{sanitize_for_filename(behavior)}__{sanitize_for_filename(metric_name)}"
            else:
                ax.set_title(f"{dataset_name} | {behavior} | {metric_name}")
                out_name = f"{sanitize_for_filename(dataset_name)}__{sanitize_for_filename(behavior)}__{sanitize_for_filename(metric_name)}"

            ax.legend(frameon=False)
            fig.tight_layout()
            savefig_both(fig, os.path.join(out_dir, out_name))


# =============================================================================
# MAIN
# =============================================================================

def main():
    ensure_dir(OUT_DIR)
    ensure_dir(COMBINED_DIR)
    ensure_dir(PER_DATASET_DIR)

    if not os.path.isfile(SI_CSV):
        raise FileNotFoundError(f"SI CSV not found: {SI_CSV}")
    if not os.path.isfile(DECODER_CSV):
        raise FileNotFoundError(f"Decoder CSV not found: {DECODER_CSV}")

    si_df = pd.read_csv(SI_CSV)
    dec_df = pd.read_csv(DECODER_CSV)

    merged = merge_si_decoder(si_df, dec_df)

    # all datasets combined
    plot_scatter(merged, dataset_name=None, out_dir=COMBINED_DIR)

    # per dataset
    datasets = sorted(merged["dataset_name"].dropna().astype(str).unique())
    for ds in datasets:
        ds_dir = os.path.join(PER_DATASET_DIR, sanitize_for_filename(ds))
        ensure_dir(ds_dir)
        plot_scatter(merged[merged["dataset_name"] == ds].copy(), dataset_name=ds, out_dir=ds_dir)

    print("Done.")
    print(f"Combined plots: {COMBINED_DIR}")
    print(f"Per-dataset plots: {PER_DATASET_DIR}")


if __name__ == "__main__":
    main()

############################### BOX PLOT ###############################################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Boxplots for SI and decoder performance
======================================

This script uses the outputs from:

    /home/melisamc/Documents/time_project_standarization/analysis/
    cluster_umap_si_decoder_with_all_cells_and_mse/tables/

and generates boxplots for:

1) SI values
2) decoder performance

for:
- all datasets combined
- each dataset individually

It also saves statistics .txt files for every plot.

Expected inputs
---------------
- cluster_structure_index.csv
- cluster_decoder_xgboost_long.csv

Group convention
----------------
-1 -> all_cells
 0 -> time
 1 -> direction
 2 -> position

Notes
-----
- SI is plotted by behavior
- decoder is plotted by behavior AND metric_name
- stats include:
    * descriptive stats per group
    * one-way ANOVA
    * pairwise Welch t-tests
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = "/home/melisamc/Documents/time_project_standarization"
ANALYSIS_DIR = os.path.join(
    PROJECT_ROOT,
    "analysis",
    "cluster_umap_si_decoder_with_all_cells_and_mse",
)

TABLE_DIR = os.path.join(ANALYSIS_DIR, "tables")
SI_CSV = os.path.join(TABLE_DIR, "cluster_structure_index.csv")
DECODER_CSV = os.path.join(TABLE_DIR, "cluster_decoder_xgboost_long.csv")

OUT_DIR = os.path.join(ANALYSIS_DIR, "figures_boxplots")

SI_COMBINED_DIR = os.path.join(OUT_DIR, "si", "all_datasets_combined")
SI_PER_DATASET_DIR = os.path.join(OUT_DIR, "si", "per_dataset")

DEC_COMBINED_DIR = os.path.join(OUT_DIR, "decoder", "all_datasets_combined")
DEC_PER_DATASET_DIR = os.path.join(OUT_DIR, "decoder", "per_dataset")

SI_STATS_COMBINED_DIR = os.path.join(OUT_DIR, "stats_si", "all_datasets_combined")
SI_STATS_PER_DATASET_DIR = os.path.join(OUT_DIR, "stats_si", "per_dataset")

DEC_STATS_COMBINED_DIR = os.path.join(OUT_DIR, "stats_decoder", "all_datasets_combined")
DEC_STATS_PER_DATASET_DIR = os.path.join(OUT_DIR, "stats_decoder", "per_dataset")


# =============================================================================
# STYLE
# =============================================================================

GROUP_ORDER = [-1, 0, 1, 2]
GROUP_LABELS = {
    -1: "all_cells",
    0: "time",
    1: "direction",
    2: "position",
}
GROUP_COLORS = {
    -1: "black",
    0: "tab:green",
    1: "tab:blue",
    2: "tab:purple",
}

FIG_DPI = 200
POINT_JITTER = 0.04
POINT_SIZE = 18


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def savefig_both(fig, out_base, dpi=FIG_DPI):
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def sanitize_for_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def write_text(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(lines, str):
            f.write(lines)
        else:
            for line in lines:
                f.write(str(line) + "\n")


def summarize_array(x):
    x = pd.Series(x).dropna().values
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def one_way_anova(groups_dict):
    arrays = [np.asarray(v, dtype=float) for v in groups_dict.values() if len(v) > 1]
    if len(arrays) < 2:
        return np.nan, np.nan
    try:
        stat, pval = stats.f_oneway(*arrays)
        return float(stat), float(pval)
    except Exception:
        return np.nan, np.nan


def pairwise_welch_tests(groups_dict):
    out = []
    labels = list(groups_dict.keys())
    for g1, g2 in itertools.combinations(labels, 2):
        v1 = pd.Series(groups_dict[g1]).dropna().values
        v2 = pd.Series(groups_dict[g2]).dropna().values
        if len(v1) < 2 or len(v2) < 2:
            out.append(f"{g1} vs {g2}: skipped (insufficient n)")
            continue
        try:
            stat, pval = stats.ttest_ind(v1, v2, equal_var=False, nan_policy="omit")
            out.append(f"{g1} vs {g2}: t={stat:.6f}, p={pval:.6e}")
        except Exception as e:
            out.append(f"{g1} vs {g2}: failed ({e})")
    return out


def build_stats_text(df, value_col, group_col="group_label", title=""):
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

    groups_dict = {}
    for g, sub in df.groupby(group_col):
        vals = sub[value_col].dropna().values
        groups_dict[g] = vals

    lines.append("Descriptive statistics")
    lines.append("----------------------")
    for g in groups_dict:
        s = summarize_array(groups_dict[g])
        lines.append(
            f"{g}: n={s['n']}, mean={s['mean']:.6f}, std={s['std']:.6f}, "
            f"median={s['median']:.6f}, min={s['min']:.6f}, max={s['max']:.6f}"
        )
    lines.append("")

    stat, pval = one_way_anova(groups_dict)
    lines.append("One-way ANOVA")
    lines.append("-------------")
    lines.append(f"F={stat:.6f}, p={pval:.6e}")
    lines.append("")

    lines.append("Pairwise Welch t-tests")
    lines.append("----------------------")
    lines.extend(pairwise_welch_tests(groups_dict))
    return lines


def get_present_group_order(df):
    present = set(df["group_id"].dropna().astype(int).unique().tolist())
    return [gid for gid in GROUP_ORDER if gid in present]


# =============================================================================
# BOXPLOT ROUTINES
# =============================================================================

def plot_boxplot(df, value_col, ylabel, title, out_base):
    group_order = get_present_group_order(df)
    if len(group_order) == 0:
        return False

    fig, ax = plt.subplots(figsize=(6, 5))

    labels = [GROUP_LABELS[g] for g in group_order]
    data = [df.loc[df["group_id"] == g, value_col].dropna().values for g in group_order]

    ax.boxplot(data, labels=labels, showfliers=False)

    rng = np.random.default_rng(42)
    for i, gid in enumerate(group_order, start=1):
        vals = df.loc[df["group_id"] == gid, value_col].dropna().values
        if len(vals) == 0:
            continue
        x = rng.normal(i, POINT_JITTER, size=len(vals))
        ax.scatter(
            x,
            vals,
            s=POINT_SIZE,
            alpha=0.75,
            color=GROUP_COLORS[gid],
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    savefig_both(fig, out_base)
    return True


# =============================================================================
# SI PLOTS
# =============================================================================

def make_si_plots_combined(si_df):
    behaviors = sorted(si_df["behavior"].dropna().astype(str).unique().tolist())

    for behavior in behaviors:
        sub = si_df[si_df["behavior"] == behavior].copy()
        if sub.empty:
            continue

        out_base = os.path.join(
            SI_COMBINED_DIR,
            f"all_datasets__si_boxplot__{sanitize_for_filename(behavior)}"
        )
        ok = plot_boxplot(
            sub,
            value_col="si_value",
            ylabel="SI",
            title=f"All datasets | SI | {behavior}",
            out_base=out_base,
        )
        if ok:
            stats_lines = build_stats_text(
                sub,
                value_col="si_value",
                group_col="group_label",
                title=f"All datasets | SI | {behavior}",
            )
            stats_path = os.path.join(
                SI_STATS_COMBINED_DIR,
                f"all_datasets__si_boxplot__{sanitize_for_filename(behavior)}.txt"
            )
            write_text(stats_path, stats_lines)


def make_si_plots_per_dataset(si_df):
    datasets = sorted(si_df["dataset_name"].dropna().astype(str).unique().tolist())

    for dataset_name in datasets:
        ds_df = si_df[si_df["dataset_name"] == dataset_name].copy()
        if ds_df.empty:
            continue

        ds_fig_dir = os.path.join(SI_PER_DATASET_DIR, sanitize_for_filename(dataset_name))
        ds_stats_dir = os.path.join(SI_STATS_PER_DATASET_DIR, sanitize_for_filename(dataset_name))
        ensure_dir(ds_fig_dir)
        ensure_dir(ds_stats_dir)

        behaviors = sorted(ds_df["behavior"].dropna().astype(str).unique().tolist())

        for behavior in behaviors:
            sub = ds_df[ds_df["behavior"] == behavior].copy()
            if sub.empty:
                continue

            out_base = os.path.join(
                ds_fig_dir,
                f"{sanitize_for_filename(dataset_name)}__si_boxplot__{sanitize_for_filename(behavior)}"
            )
            ok = plot_boxplot(
                sub,
                value_col="si_value",
                ylabel="SI",
                title=f"{dataset_name} | SI | {behavior}",
                out_base=out_base,
            )
            if ok:
                stats_lines = build_stats_text(
                    sub,
                    value_col="si_value",
                    group_col="group_label",
                    title=f"{dataset_name} | SI | {behavior}",
                )
                stats_path = os.path.join(
                    ds_stats_dir,
                    f"{sanitize_for_filename(dataset_name)}__si_boxplot__{sanitize_for_filename(behavior)}.txt"
                )
                write_text(stats_path, stats_lines)


# =============================================================================
# DECODER PLOTS
# =============================================================================

def make_decoder_plots_combined(dec_df):
    metrics = sorted(dec_df["metric_name"].dropna().astype(str).unique().tolist())
    behaviors = sorted(dec_df["behavior"].dropna().astype(str).unique().tolist())

    for metric_name in metrics:
        sub_metric = dec_df[dec_df["metric_name"] == metric_name].copy()
        if sub_metric.empty:
            continue

        for behavior in behaviors:
            sub = sub_metric[sub_metric["behavior"] == behavior].copy()
            if sub.empty:
                continue

            out_base = os.path.join(
                DEC_COMBINED_DIR,
                f"all_datasets__decoder_boxplot__{sanitize_for_filename(behavior)}__{sanitize_for_filename(metric_name)}"
            )
            ok = plot_boxplot(
                sub,
                value_col="metric_value",
                ylabel=metric_name,
                title=f"All datasets | Decoder | {behavior} | {metric_name}",
                out_base=out_base,
            )
            if ok:
                stats_lines = build_stats_text(
                    sub,
                    value_col="metric_value",
                    group_col="group_label",
                    title=f"All datasets | Decoder | {behavior} | {metric_name}",
                )
                stats_path = os.path.join(
                    DEC_STATS_COMBINED_DIR,
                    f"all_datasets__decoder_boxplot__{sanitize_for_filename(behavior)}__{sanitize_for_filename(metric_name)}.txt"
                )
                write_text(stats_path, stats_lines)


def make_decoder_plots_per_dataset(dec_df):
    datasets = sorted(dec_df["dataset_name"].dropna().astype(str).unique().tolist())

    for dataset_name in datasets:
        ds_df = dec_df[dec_df["dataset_name"] == dataset_name].copy()
        if ds_df.empty:
            continue

        ds_fig_dir = os.path.join(DEC_PER_DATASET_DIR, sanitize_for_filename(dataset_name))
        ds_stats_dir = os.path.join(DEC_STATS_PER_DATASET_DIR, sanitize_for_filename(dataset_name))
        ensure_dir(ds_fig_dir)
        ensure_dir(ds_stats_dir)

        metrics = sorted(ds_df["metric_name"].dropna().astype(str).unique().tolist())
        behaviors = sorted(ds_df["behavior"].dropna().astype(str).unique().tolist())

        for metric_name in metrics:
            sub_metric = ds_df[ds_df["metric_name"] == metric_name].copy()
            if sub_metric.empty:
                continue

            for behavior in behaviors:
                sub = sub_metric[sub_metric["behavior"] == behavior].copy()
                if sub.empty:
                    continue

                out_base = os.path.join(
                    ds_fig_dir,
                    f"{sanitize_for_filename(dataset_name)}__decoder_boxplot__{sanitize_for_filename(behavior)}__{sanitize_for_filename(metric_name)}"
                )
                ok = plot_boxplot(
                    sub,
                    value_col="metric_value",
                    ylabel=metric_name,
                    title=f"{dataset_name} | Decoder | {behavior} | {metric_name}",
                    out_base=out_base,
                )
                if ok:
                    stats_lines = build_stats_text(
                        sub,
                        value_col="metric_value",
                        group_col="group_label",
                        title=f"{dataset_name} | Decoder | {behavior} | {metric_name}",
                    )
                    stats_path = os.path.join(
                        ds_stats_dir,
                        f"{sanitize_for_filename(dataset_name)}__decoder_boxplot__{sanitize_for_filename(behavior)}__{sanitize_for_filename(metric_name)}.txt"
                    )
                    write_text(stats_path, stats_lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    for path in [
        OUT_DIR,
        SI_COMBINED_DIR,
        SI_PER_DATASET_DIR,
        DEC_COMBINED_DIR,
        DEC_PER_DATASET_DIR,
        SI_STATS_COMBINED_DIR,
        SI_STATS_PER_DATASET_DIR,
        DEC_STATS_COMBINED_DIR,
        DEC_STATS_PER_DATASET_DIR,
    ]:
        ensure_dir(path)

    if not os.path.isfile(SI_CSV):
        raise FileNotFoundError(f"SI CSV not found: {SI_CSV}")
    if not os.path.isfile(DECODER_CSV):
        raise FileNotFoundError(f"Decoder CSV not found: {DECODER_CSV}")

    si_df = pd.read_csv(SI_CSV)
    dec_df = pd.read_csv(DECODER_CSV)

    # normalize labels if needed
    if "group_label" not in si_df.columns and "group_id" in si_df.columns:
        si_df["group_label"] = si_df["group_id"].map(GROUP_LABELS)

    if "group_label" not in dec_df.columns and "group_id" in dec_df.columns:
        dec_df["group_label"] = dec_df["group_id"].map(GROUP_LABELS)

    print("=" * 70)
    print("Generating SI and decoder boxplots")
    print("=" * 70)

    print("SI combined...")
    make_si_plots_combined(si_df)

    print("SI per dataset...")
    make_si_plots_per_dataset(si_df)

    print("Decoder combined...")
    make_decoder_plots_combined(dec_df)

    print("Decoder per dataset...")
    make_decoder_plots_per_dataset(dec_df)

    print("\nDone.")
    print(f"SI combined figures: {SI_COMBINED_DIR}")
    print(f"SI per-dataset figures: {SI_PER_DATASET_DIR}")
    print(f"Decoder combined figures: {DEC_COMBINED_DIR}")
    print(f"Decoder per-dataset figures: {DEC_PER_DATASET_DIR}")


if __name__ == "__main__":
    main()