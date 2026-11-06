#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure Index (SI) Analysis Pipeline across Complete Neural Space & Functional Clusters
--------------------------------------------------------------------------------------
What this script does:
1) Reads the Master Dataset Index and Mutual Information tables for the target dataset (e.g., 'lt_egrin').
2) Loads standardized session pickles containing full neural activity (T x N), behaviors, and LFP modalities.
3) Performs neuron clustering (Time Cluster, Direction Cluster, Position Cluster) using the established robust semantic mapping.
4) Filters timepoints where mov_dir == 0 and removes noisy outliers via local density thresholding.
5) Computes the Structure Index (SI) for:
   - Behavioral variables: position, speed, time, inner_time, trial_id, direction, position_direction (2D).
   - Multi-band LFP modalities across all defined frequency bands:
     * Waveforms
     * Hilbert Envelopes
     * Phase Angles
     * CWT Power Bands
   - Computes SI independently for BOTH the Complete Population and each individual Functional Cluster.
6) Saves detailed session-level pickle outputs and aggregates results into a Master SI CSV table.
7) Generates comprehensive comparison bar plots and box plots summarizing SI across datasets, clusters, and features.
"""

from __future__ import annotations

import os
import csv
import glob
import pickle
import copy
import timeit
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

# Local Structure Index module
from structure_index import compute_structure_index

# =============================================================================
# Paths & Settings
# =============================================================================

POSSIBLE_ROOTS = [
    Path("/home/hippolab/Documents/melisa/time_project_standarization"),
    Path("/home/melisamc/Documents/time_project_standarization"),
]
PROJECT_ROOT = next((p for p in POSSIBLE_ROOTS if p.is_dir()), POSSIBLE_ROOTS[0])
TARGET_DATASET = "lt_egrin"

MASTER_INDEX_CSV = PROJECT_ROOT / "data" / "metadata" / "master_dataset_index.csv"
MI_CSV = PROJECT_ROOT / "analysis" / "mutual_information_behavior_and_lfp" / "mutual_information_per_neuron.csv"

OUTPUT_DIR = PROJECT_ROOT / "analysis" / f"structure_index_manifold_suite_{TARGET_DATASET}"
SESSION_PKL_DIR = OUTPUT_DIR / "session_pickles"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

SI_SESSION_CSV = OUTPUT_DIR / f"structure_index_manifold_per_session_{TARGET_DATASET}.csv"
SI_SUMMARY_CSV = OUTPUT_DIR / f"structure_index_manifold_summary_{TARGET_DATASET}.csv"

# =============================================================================
# Parameters & Configurations
# =============================================================================

MAX_ROWS = None
SMOOTH_SIGMA = 6  # Gaussian smoothing sigma for neural signals
Z_SCORE_NEURONS = False

# Outlier filtering parameters in neural space
OUTLIER_DIST_TH = 5
OUTLIER_NOISE_TH = 25

# Clustering parameters
N_CLUSTERS = 3
UNASSIGNED_LABEL = 10
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 50

MIR_FEATURE_COLUMNS = [
    "mi_position",
    "mi_position_x_direction",
    "mi_direction",
    "mi_speed",
    "mi_time",
    "mi_inner_time",
    "mi_trial_id",
]

CLUSTER_NAMES = {
    0: "Time Cluster",
    1: "Direction Cluster",
    2: "Position Cluster",
}

# Standardized LFP & CWT Band Definitions (1:1 Matching with clustering suite)
LFP_BANDS_CONFIG = [
    {"key": "slow", "name": "Slow [0.1 - 1 Hz]", "f_low": 0.10, "f_high": 1.00},
    {"key": "delta", "name": "Delta [1 - 3 Hz]", "f_low": 1.00, "f_high": 4.00},
    {"key": "theta", "name": "Theta [8 - 12 Hz]", "f_low": 8.00, "f_high": 12.00},
    {"key": "gamma", "name": "Gamma [40 - 90 Hz]", "f_low": 40.00, "f_high": 90.00},
    {"key": "ripple", "name": "Ripple [100 - 250 Hz]", "f_low": 100.0, "f_high": 250.0},
    {"key": "mua", "name": "MUA [300 - 1000 Hz]", "f_low": 300.0, "f_high": 1000.0},
]

# SI Parameter Presets
SI_PARAMS_CONTINUOUS = {
    "n_bins": 10,
    "discrete_label": False,
    "continuity_kernel": None,
    "perc_neigh": 1,
    "n_neighbors": 50,
    "num_shuffles": 0,
    "verbose": False,
}

SI_PARAMS_DISCRETE = {
    "n_bins": 2,
    "discrete_label": True,
    "continuity_kernel": None,
    "n_neighbors": 50,
    "num_shuffles": 0,
    "verbose": False,
}

SI_PARAMS_MIXED = {
    "n_bins": [10, 2],
    "discrete_label": [False, True],
    "n_neighbors": 50,
    "num_shuffles": 0,
    "verbose": False,
}

SI_PARAMS_PHASE = {
    "n_bins": 10,
    "discrete_label": False,
    "continuity_kernel": "circular",  # Circular kernel support for phase variables if applicable, or continuous
    "perc_neigh": 1,
    "n_neighbors": 50,
    "num_shuffles": 0,
    "verbose": False,
}


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_pkl_path(row_subset: pd.Series) -> str | None:
    for col in row_subset.index:
        val = row_subset[col]
        if pd.notna(val) and isinstance(val, str) and ".pkl" in val:
            corrected = val
            if "/home/melisamc/Documents/time_project_standarization" in corrected:
                corrected = corrected.replace("/home/melisamc/Documents/time_project_standarization", str(PROJECT_ROOT))
            if os.path.isfile(corrected):
                return corrected
            basename = os.path.basename(corrected)
            matches = glob.glob(str(PROJECT_ROOT / "data" / "standardized_sessions" / "**" / basename), recursive=True)
            if matches and os.path.isfile(matches[0]):
                return matches[0]

    mouse_name = str(row_subset.get("mouse_name", "")).strip()
    depth_type = str(row_subset.get("depth_type", "")).strip()
    if mouse_name:
        matches = glob.glob(
            str(PROJECT_ROOT / "data" / "standardized_sessions" / TARGET_DATASET / "**" / f"*{mouse_name}*.pkl"),
            recursive=True)
        for m in matches:
            if depth_type in m and os.path.isfile(m):
                return m
        if matches and os.path.isfile(matches[0]):
            return matches[0]
    return None


def load_pickle_standard(file_path: str | Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle_standard(file_path: str | Path, obj: object) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def ensure_1d(x: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    return x.reshape(-1)


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def sanitize_for_filename(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text))


def base_name_from_row(row_idx: int, row: pd.Series | dict) -> str:
    return (
        f"row_{row_idx:04d}_"
        f"{sanitize_for_filename(row.get('dataset_name', TARGET_DATASET))}_"
        f"{sanitize_for_filename(row.get('mouse_name', 'unknown_mouse'))}_"
        f"{sanitize_for_filename(row.get('session_name', 'unknown_session'))}_"
        f"{sanitize_for_filename(row.get('depth_type', 'unknown_depth'))}"
    )


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj: str) -> None:
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self) -> None:
        for f in self.files:
            f.flush()


def filter_noisy_outliers(data: np.ndarray, D: np.ndarray | None = None, dist_th: float = 5, noise_th: float = 25) -> \
        Tuple[np.ndarray, np.ndarray]:
    if D is None:
        D = pairwise_distances(data)
    D_work = D.copy()
    np.fill_diagonal(D_work, np.nan)
    nn_dist = np.sum(D_work < np.nanpercentile(D_work, dist_th), axis=1)
    noise_idx = np.where(nn_dist < np.percentile(nn_dist, noise_th))[0]
    signal_idx = np.where(nn_dist >= np.percentile(nn_dist, noise_th))[0]
    return noise_idx, signal_idx


def align_array_length(arr, target_len):
    if arr is None:
        return None
    arr = np.asarray(arr).reshape(-1)
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), mode="edge")


def rowwise_zscore(X):
    Xz = zscore(X, axis=1, nan_policy="omit")
    return np.where(np.isfinite(Xz), Xz, 0.0)


def infer_intra_dataset_semantic_mapping(df_dataset, raw_labels):
    """
    Semantic mapping rules (Robust median-based with spatial penalty for time cluster):
    1) Position cluster (ID 2, Pink): Max median (mi_position + mi_position_x_direction).
    2) Time cluster (ID 0, Green): Maximize (Time Score - Position/Direction Score) among remaining.
       Ensures time cluster has high time info and minimum information about position, direction, or directed positions.
    3) Direction cluster (ID 1, Cyan): The remaining cluster.
    """
    tmp = df_dataset.copy()
    tmp["raw_label"] = raw_labels
    cluster_ids = sorted(tmp["raw_label"].unique().tolist())
    if len(cluster_ids) < N_CLUSTERS:
        return {cid: cid for cid in cluster_ids}

    cluster_scores = {}
    for cid in cluster_ids:
        sub = tmp[tmp["raw_label"] == cid]

        # Position and Direction information components
        pos_score = np.median(sub["mi_position"] + sub["mi_position_x_direction"])
        dir_score = np.median(sub["mi_direction"]) if "mi_direction" in sub.columns else 0.0
        pos_dir_total = pos_score + dir_score

        # Time information components
        time_score = np.median(sub["mi_time"] + sub["mi_trial_id"]) if "mi_trial_id" in sub.columns else np.median(
            sub["mi_time"])

        # Net time score: rewards high time info while penalizing position/direction info
        net_time_score = time_score - pos_dir_total

        cluster_scores[cid] = {
            "pos": pos_score,
            "net_time": net_time_score
        }

    # 1. Position Cluster: Highest position score
    pos_cid = max(cluster_ids, key=lambda c: cluster_scores[c]["pos"])
    remaining_after_pos = [c for c in cluster_ids if c != pos_cid]

    if remaining_after_pos:
        # 2. Time Cluster: Highest net time score (maximizing time while minimizing position/direction info)
        time_cid = max(remaining_after_pos, key=lambda c: cluster_scores[c]["net_time"])
    else:
        time_cid = pos_cid

    # 3. Direction Cluster: Whatever remains
    remaining_after_time = [c for c in remaining_after_pos if c != time_cid]
    dir_cid = remaining_after_time[0] if remaining_after_time else (
        remaining_after_pos[0] if remaining_after_pos else pos_cid)

    return {pos_cid: 2, time_cid: 0, dir_cid: 1}


def cluster_session_neurons(df_session):
    X_raw = df_session[MIR_FEATURE_COLUMNS].astype(float).to_numpy()
    valid_mask = np.all(np.isfinite(X_raw), axis=1)
    df_sub = df_session.loc[valid_mask].copy().reset_index(drop=True)
    X_raw = X_raw[valid_mask]

    if len(df_sub) < N_CLUSTERS:
        df_session["cluster_id"] = UNASSIGNED_LABEL
        return df_session

    X_z = rowwise_zscore(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_z)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
    raw_labels = kmeans.fit_predict(X_scaled)

    remap_dict = infer_intra_dataset_semantic_mapping(df_sub, raw_labels)
    semantic_labels = np.array([remap_dict.get(int(l), UNASSIGNED_LABEL) for l in raw_labels], dtype=int)
    df_sub["cluster_id"] = semantic_labels
    return df_sub


def extract_session_features(std_data, target_len):
    pos = std_data.get("pos", std_data.get("position", None))
    if pos is not None:
        pos = np.asarray(pos)
        pos = pos[:, 0] if pos.ndim == 2 else pos.reshape(-1)
    pos = align_array_length(pos, target_len)

    time_vec = ensure_1d(std_data.get("time", None))
    if time_vec is None:
        time_vec = np.arange(target_len, dtype=float)
    else:
        time_vec = align_array_length(time_vec, target_len)

    mov_dir = ensure_1d(std_data.get("mov_dir", std_data.get("direction", None)))
    if mov_dir is None:
        mov_dir = np.zeros(target_len, dtype=float)
    else:
        mov_dir = align_array_length(mov_dir, target_len)

    speed = ensure_1d(std_data.get("speed", None))
    if speed is None and pos is not None and len(pos) > 1:
        dt = np.median(np.diff(time_vec)) if len(time_vec) > 1 else 1.0
        speed = np.abs(np.gradient(pos, dt))
    speed = align_array_length(speed, target_len)

    inner_time = ensure_1d(std_data.get("inner_time", None))
    if inner_time is None:
        inner_time = normalize_01(time_vec)
    else:
        inner_time = align_array_length(inner_time, target_len)

    trial_id = ensure_1d(std_data.get("trial_id", None))
    trial_id = align_array_length(trial_id, target_len) if trial_id is not None else None

    # LFP & CWT extraction
    cwt_matrix = std_data.get("cwt")
    cwt_freqs = std_data.get("cwt_freqs")
    if cwt_matrix is not None:
        cwt_matrix = np.asarray(cwt_matrix)
        if cwt_matrix.shape[0] != target_len:
            cwt_matrix = align_array_length(cwt_matrix, target_len) if cwt_matrix.ndim == 1 else \
                np.pad(cwt_matrix, ((0, max(0, target_len - cwt_matrix.shape[0])), (0, 0)), mode="edge")[:target_len, :]
    if cwt_freqs is not None:
        cwt_freqs = np.asarray(cwt_freqs).reshape(-1)

    features = {
        "position": pos,
        "speed": speed,
        "time": time_vec,
        "inner_time": inner_time,
        "trial_id": trial_id,
        "direction": mov_dir,
    }

    for cfg in LFP_BANDS_CONFIG:
        bkey = cfg["key"]
        features[f"lfp_waveform_{bkey}"] = align_array_length(std_data.get(f"lfp_{bkey}_20hz"), target_len)

        env_arr = std_data.get(f"lfp_{bkey}_hilbert_20hz")
        if env_arr is None and features[f"lfp_waveform_{bkey}"] is not None:
            env_arr = np.abs(features[f"lfp_waveform_{bkey}"])
        features[f"lfp_envelope_{bkey}"] = align_array_length(env_arr, target_len)

        features[f"lfp_phase_{bkey}"] = align_array_length(std_data.get(f"lfp_{bkey}_phase_20hz"), target_len)

        if cwt_matrix is not None and cwt_freqs is not None:
            f_low, f_high = cfg["f_low"], cfg["f_high"]
            idx_range = np.where((cwt_freqs >= f_low) & (cwt_freqs <= f_high))[0]
            if len(idx_range) == 0:
                nearest_idx = np.argmin(np.abs(cwt_freqs - 0.5 * (f_low + f_high)))
                idx_range = np.array([nearest_idx])
            mean_power = np.nanmean(cwt_matrix[:, idx_range], axis=1)
            features[f"cwt_power_{bkey}"] = align_array_length(10.0 * np.log10(np.maximum(mean_power, 1e-12)),
                                                               target_len)
        else:
            features[f"cwt_power_{bkey}"] = None

    return features


def get_si_params_for_feature(feat_name: str, feat_values: np.ndarray) -> dict:
    if feat_name in ["position", "speed", "time", "inner_time",
                     "trial_id"] or "envelope" in feat_name or "cwt_power" in feat_name or "waveform" in feat_name:
        params = copy.deepcopy(SI_PARAMS_CONTINUOUS)
        if feat_name == "trial_id":
            params["min_label"] = float(np.nanmin(feat_values))
            params["max_label"] = float(np.nanmax(feat_values))
        return params

    if feat_name == "direction":
        return copy.deepcopy(SI_PARAMS_DISCRETE)

    if feat_name == "position_direction":
        return copy.deepcopy(SI_PARAMS_MIXED)

    if "phase" in feat_name:
        return copy.deepcopy(SI_PARAMS_PHASE)

    return copy.deepcopy(SI_PARAMS_CONTINUOUS)


# =============================================================================
# CSV Table Handling & Aggregation
# =============================================================================

def load_existing_keys(csv_path: Path) -> Set[Tuple[str, str, str, str]]:
    if not csv_path.is_file():
        return set()
    try:
        df_existing = pd.read_csv(csv_path, dtype=str)
        required = {"dataset_name", "mouse_name", "session_name", "depth_type"}
        if required.issubset(df_existing.columns):
            return set(zip(
                df_existing["dataset_name"].str.strip().str.lower(),
                df_existing["mouse_name"].str.strip(),
                df_existing["session_name"].str.strip(),
                df_existing["depth_type"].str.strip().str.lower()
            ))
    except Exception as exc:
        print(f"[WARN] Could not parse existing CSV: {exc}")
    return set()


def append_csv_row(csv_path: Path, columns: List[str], row: Dict[str, object]) -> None:
    file_exists = csv_path.is_file()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# Core Processing per Session
# =============================================================================

def process_session_row(row_idx: int, row: pd.Series, df_session_clustered: pd.DataFrame) -> Dict[str, Any]:
    session_path = resolve_pkl_path(row)
    if not session_path or not os.path.isfile(session_path):
        raise FileNotFoundError(f"Standardized session file not found for row {row_idx}")

    std_data = load_pickle_standard(session_path)
    signal = std_data.get("signal")
    if signal is None and "clean_traces" in std_data:
        signal = std_data["clean_traces"]
    if signal is None:
        raise KeyError(f"No signal matrix found in {session_path}")

    raw_signal = np.asarray(signal)
    n_timepoints, total_neurons = raw_signal.shape
    features_dict = extract_session_features(std_data, target_len=n_timepoints)

    if raw_signal.shape[0] != n_timepoints and raw_signal.shape[1] == n_timepoints:
        raw_signal = raw_signal.T

    signal = raw_signal.copy()
    if SMOOTH_SIGMA > 0:
        signal = gaussian_filter1d(signal, sigma=SMOOTH_SIGMA, axis=0)

    if Z_SCORE_NEURONS:
        mu = np.nanmean(signal, axis=0, keepdims=True)
        sd = np.nanstd(signal, axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        signal = (signal - mu) / sd

    # 1. Filter mov_dir == 0
    mov_dir = features_dict["direction"]
    valid_idx = (mov_dir != 0) & np.isfinite(mov_dir)
    signal_valid = signal[valid_idx, :]
    features_valid = {k: (None if v is None else v[valid_idx]) for k, v in features_dict.items()}

    if signal_valid.shape[0] < 20:
        raise ValueError(f"Row {row_idx}: too few points after direction filter.")

    # 2. Outlier filtering
    D = pairwise_distances(signal_valid)
    noise_idx, signal_idx = filter_noisy_outliers(signal_valid, D=D, dist_th=OUTLIER_DIST_TH, noise_th=OUTLIER_NOISE_TH)
    del D

    signal_clean = signal_valid[signal_idx, :]
    features_clean = {k: (None if v is None else v[signal_idx]) for k, v in features_valid.items()}

    if signal_clean.shape[0] < 20:
        raise ValueError(f"Row {row_idx}: too few points after outlier filtering.")

    # Construct 2D position_direction
    pos_clean = features_clean["position"].reshape(-1, 1)
    dir_clean = features_clean["direction"].reshape(-1, 1)
    features_clean["position_direction"] = np.concatenate([pos_clean, dir_clean], axis=1)

    # Define population spaces to analyze: Complete + Functional Clusters
    spaces_to_analyze = {"complete_population": np.arange(total_neurons)}
    for cid, cname in CLUSTER_NAMES.items():
        sub_neurons = df_session_clustered[df_session_clustered["cluster_id"] == cid]["neuron_index"].astype(
            int).tolist()
        sub_neurons = [n for n in sub_neurons if 0 <= n < total_neurons]
        if len(sub_neurons) >= 3:
            spaces_to_analyze[cname] = sub_neurons

    session_results = {}
    si_details_all = {}

    for space_name, neuron_indices in spaces_to_analyze.items():
        sub_signal = signal_clean[:, neuron_indices]
        space_results = {}
        space_details = {}

        for feat_name, feat_val in features_clean.items():
            if feat_val is None or not np.isfinite(feat_val).any():
                space_results[feat_name] = np.nan
                space_details[feat_name] = None
                continue

            params = get_si_params_for_feature(feat_name, feat_val)
            try:
                si, process_info, overlap_mat, shuf_si = compute_structure_index(
                    sub_signal,
                    feat_val,
                    **params
                )
                space_results[feat_name] = float(si)
                space_details[feat_name] = {
                    "si": float(si),
                    "process_info": copy.deepcopy(process_info),
                    "overlap_mat": copy.deepcopy(overlap_mat),
                    "shuf_si": copy.deepcopy(shuf_si),
                }
            except Exception as exc:
                space_results[feat_name] = np.nan
                space_details[feat_name] = None

        session_results[space_name] = space_results
        si_details_all[space_name] = space_details

    meta = std_data.get("metadata", {})
    session_base = base_name_from_row(row_idx, row) + "_si_manifold_suite"
    session_pickle_name = session_base + ".pkl"
    session_pickle_path = SESSION_PKL_DIR / session_pickle_name

    session_payload = {
        "metadata": {
            "dataset_name": row.get("dataset_name", TARGET_DATASET),
            "mouse_name": meta.get("mouse_name", row.get("mouse_name", "")),
            "session_name": meta.get("session_name", row.get("session_name", "")),
            "depth_type": meta.get("depth_type", row.get("depth_type", "")),
            "row_index": row_idx,
            "standardized_file_path": session_path,
            "n_neurons_total": total_neurons,
            "spaces_analyzed": list(spaces_to_analyze.keys()),
        },
        "si_results": session_results,
        "si_details": si_details_all,
    }
    save_pickle_standard(session_pickle_path, session_payload)

    return {
        "dataset_name": row.get("dataset_name", TARGET_DATASET),
        "mouse_name": row.get("mouse_name", ""),
        "session_name": row.get("session_name", ""),
        "depth_type": row.get("depth_type", ""),
        "row_index": row_idx,
        "session_pickle_path": str(session_pickle_path),
        "results": session_results,
    }


# =============================================================================
# Plotting Functions for Summary Comparisons
# =============================================================================

def generate_summary_plots(df_summary: pd.DataFrame) -> None:
    ensure_dir(FIGURES_DIR)
    sns.set_theme(style="whitegrid")

    # 1. Bar plot comparing SI across spaces for key behavioral features
    plt.figure(figsize=(14, 6))
    key_behaviors = ["position", "speed", "time", "direction", "position_direction"]
    df_behav = df_summary[df_summary["feature_name"].isin(key_behaviors)]

    if not df_behav.empty:
        ax = sns.barplot(data=df_behav, x="feature_name", y="structure_index", hue="neural_space", palette="Set2")
        plt.title(f"Structure Index Across Behavioral Features ({TARGET_DATASET})", fontsize=14, weight="bold")
        plt.xlabel("Behavioral Feature", fontsize=12)
        plt.ylabel("Structure Index (SI)", fontsize=12)
        plt.legend(title="Neural Space / Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"si_comparison_behavior_{TARGET_DATASET}.png", dpi=300)
        plt.savefig(FIGURES_DIR / f"si_comparison_behavior_{TARGET_DATASET}.svg", format="svg")
        plt.close()

    # 2. Box plots for LFP Band Waveforms & Envelopes across spaces
    plt.figure(figsize=(16, 8))
    df_lfp_env = df_summary[df_summary["feature_name"].str.contains("envelope|cwt_power", na=False)]
    if not df_lfp_env.empty:
        ax = sns.boxplot(data=df_lfp_env, x="feature_name", y="structure_index", hue="neural_space", palette="muted")
        plt.title(f"Structure Index Across LFP Envelopes & CWT Power Bands ({TARGET_DATASET})", fontsize=14,
                  weight="bold")
        plt.xlabel("LFP Feature", fontsize=12)
        plt.ylabel("Structure Index (SI)", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.legend(title="Neural Space / Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"si_comparison_lfp_bands_{TARGET_DATASET}.png", dpi=300)
        plt.savefig(FIGURES_DIR / f"si_comparison_lfp_bands_{TARGET_DATASET}.svg", format="svg")
        plt.close()

    print(f"[OK] Summary plots generated successfully in {FIGURES_DIR}")


# =============================================================================
# Main Execution Pipeline
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SESSION_PKL_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(LOG_DIR)

    log_path = LOG_DIR / "structure_index_manifold_suite_log.txt"
    log_file = open(log_path, "a", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print(f"Structure Index Manifold Suite Pipeline for {TARGET_DATASET}")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")

    if not MASTER_INDEX_CSV.is_file() or not MI_CSV.is_file():
        raise FileNotFoundError("Required Master Index CSV or Mutual Information CSV is missing.")

    df_mi = pd.read_csv(MI_CSV)
    df_master = pd.read_csv(MASTER_INDEX_CSV, engine="python", on_bad_lines="skip")

    join_cols = ["dataset_name", "mouse_name", "session_name", "depth_type"]
    add_cols = [c for c in df_master.columns if c not in join_cols]

    df_merged = df_mi.merge(df_master[join_cols + add_cols].drop_duplicates(), on=join_cols, how="left")
    df_egrin = df_merged[df_merged["dataset_name"].astype(str).str.strip() == TARGET_DATASET].copy()

    if df_egrin.empty:
        raise ValueError(f"No sessions found for dataset {TARGET_DATASET}")

    session_cols = [c for c in ["mouse_name", "session_name", "depth_type"] if c in df_egrin.columns]
    session_groups = df_egrin.groupby(session_cols)

    existing_keys = load_existing_keys(SI_SESSION_CSV)
    summary_rows = []

    ok_count = 0
    skip_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for idx, (group_keys, df_session) in enumerate(session_groups):
        row_sample = df_session.iloc[0]
        dname = str(row_sample.get("dataset_name", TARGET_DATASET)).strip().lower()
        mname = str(row_sample.get("mouse_name", "")).strip()
        sname = str(row_sample.get("session_name", "")).strip()
        dtype = str(row_sample.get("depth_type", "")).strip().lower()
        key = (dname, mname, sname, dtype)

        if key in existing_keys:
            skip_count += 1
            continue

        t0 = timeit.default_timer()
        try:
            df_clustered = cluster_session_neurons(df_session)
            session_output = process_session_row(idx, row_sample, df_clustered)

            # Append summary records for aggregation CSV
            for space_name, space_res in session_output["results"].items():
                for feat_name, si_val in space_res.items():
                    summary_rows.append({
                        "dataset_name": dname,
                        "mouse_name": mname,
                        "session_name": sname,
                        "depth_type": dtype,
                        "neural_space": space_name,
                        "feature_name": feat_name,
                        "structure_index": si_val,
                        "session_pickle_path": session_output["session_pickle_path"],
                    })

            # Append flattened session row to session CSV
            flat_row = {
                "dataset_name": dname,
                "mouse_name": mname,
                "session_name": sname,
                "depth_type": dtype,
                "row_index": idx,
                "session_pickle_path": session_output["session_pickle_path"],
            }
            for space_name, space_res in session_output["results"].items():
                for feat_name, si_val in space_res.items():
                    flat_row[f"{space_name}__{feat_name}"] = si_val

            session_columns = list(flat_row.keys())
            append_csv_row(SI_SESSION_CSV, session_columns, flat_row)
            existing_keys.add(key)
            ok_count += 1

            elapsed = timeit.default_timer() - t0
            print(f"[OK] ({idx + 1}/{len(session_groups)}) Processed {mname} ({dtype}) in {elapsed:.2f} s")
        except Exception as exc:
            fail_count += 1
            print(f"[ERROR] Session {key} failed: {exc}")

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(SI_SUMMARY_CSV, index=False)
        generate_summary_plots(df_summary)

    total_elapsed = timeit.default_timer() - global_time
    print("==============================================================")
    print("Execution Finished")
    print(f"Computed: {ok_count}")
    print(f"Skipped:  {skip_count} (already existing)")
    print(f"Failed:   {fail_count}")
    print(f"Total time: {total_elapsed:.2f} s")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()