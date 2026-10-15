#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure Index analysis pipeline in Complete Neural Space.

What this script does
---------------------
1) Reads the Master Dataset Index CSV (or UMAP analysis index).
2) Loads each standardized session pickle to obtain the full neural signal (T x N).
3) Uses the high-dimensional neural activity matrix as the data space for Structure Index.
4) Filters timepoints where mov_dir == 0.
5) Filters noisy outliers in the full neural space via density thresholding.
6) Computes Structure Index (SI) in full neural space for:
       - position
       - speed
       - time
       - inner_time
       - trial_id
       - direction
       - position_direction  (2D label: [position, direction])
7) Saves:
       - one CSV with one row per session
       - one pickle per session with full SI details
       - one execution log file
"""

import os
import csv
import pickle
import copy
import timeit
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import pairwise_distances

# Local Structure Index module
from structure_index import compute_structure_index

# =============================================================================
# Paths & Settings
# =============================================================================

POSSIBLE_ROOTS = [
    "/home/hippolab/Documents/melisa/time_project_standarization",
    "/home/melisamc/Documents/time_project_standarization",
]
PROJECT_ROOT = next((p for p in POSSIBLE_ROOTS if os.path.isdir(p)), POSSIBLE_ROOTS[0])

# Primary master metadata index
MASTER_INDEX_CSV = os.path.join(PROJECT_ROOT, "data", "metadata", "master_dataset_index.csv")

# Output directory for neural space Structure Index
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "structure_index_neural_space")
SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

SI_SESSION_CSV = os.path.join(OUTPUT_DIR, "structure_index_neural_space_per_session.csv")

# =============================================================================
# Parameters
# =============================================================================

DATASET_FILTER = "None"  # Set to None to run across all datasets, or 'dreadds'
MAX_ROWS = None
SMOOTH_SIGMA = 6  # Set to 0 if signal is already smoothed in your standardized pkl
Z_SCORE_NEURONS = False

# Outlier filtering parameters in neural space
OUTLIER_DIST_TH = 5
OUTLIER_NOISE_TH = 25

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

SI_BEHAVIOR_ORDER = [
    "position",
    "speed",
    "time",
    "inner_time",
    "trial_id",
    "direction",
    "position_direction",
]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_path(file_path):
    if not file_path or pd.isna(file_path):
        return ""
    file_path = str(file_path).strip()

    # Try direct existence
    if os.path.exists(file_path):
        return file_path
    if os.path.exists(file_path + ".pkl"):
        return file_path + ".pkl"

    # Cross-machine prefix swapping
    for src_root in POSSIBLE_ROOTS:
        for dst_root in POSSIBLE_ROOTS:
            if src_root in file_path:
                cand = file_path.replace(src_root, dst_root)
                if os.path.exists(cand):
                    return cand
                if os.path.exists(cand + ".pkl"):
                    return cand + ".pkl"
    return file_path


def load_pickle_standard(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle_standard(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def ensure_1d(x):
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    return x.reshape(-1)


def normalize_01(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def sanitize_for_filename(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text))


def base_name_from_row(row_idx, row):
    return (
        f"row_{row_idx:04d}_"
        f"{sanitize_for_filename(row.get('dataset_name', 'unknown_dataset'))}_"
        f"{sanitize_for_filename(row.get('mouse_name', 'unknown_mouse'))}_"
        f"{sanitize_for_filename(row.get('session_name', 'unknown_session'))}_"
        f"{sanitize_for_filename(row.get('depth_type', 'unknown_depth'))}"
    )


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
# Outlier Filtering
# =============================================================================

def filter_noisy_outliers(data, D=None, dist_th=5, noise_th=25):
    """Density-based outlier rejection in the provided data space."""
    if D is None:
        D = pairwise_distances(data)

    D_work = D.copy()
    np.fill_diagonal(D_work, np.nan)
    nn_dist = np.sum(D_work < np.nanpercentile(D_work, dist_th), axis=1)
    noise_idx = np.where(nn_dist < np.percentile(nn_dist, noise_th))[0]
    signal_idx = np.where(nn_dist >= np.percentile(nn_dist, noise_th))[0]

    return noise_idx, signal_idx


# =============================================================================
# Behavior Parsing
# =============================================================================

def extract_behavior_dict(data, n_timepoints):
    pos = data.get("pos", data.get("position", None))
    if pos is not None:
        pos = np.asarray(pos)
        pos = pos[:, 0] if pos.ndim == 2 else pos.reshape(-1)

    time_vec = ensure_1d(data.get("time", None))
    if time_vec is None:
        time_vec = np.arange(n_timepoints, dtype=float)

    mov_dir = ensure_1d(data.get("mov_dir", data.get("direction", None)))
    if mov_dir is None:
        mov_dir = np.zeros(n_timepoints, dtype=float)

    speed = ensure_1d(data.get("speed", None))
    if speed is None and pos is not None and len(pos) > 1:
        dt = np.median(np.diff(time_vec)) if len(time_vec) > 1 else 1.0
        speed = np.abs(np.gradient(pos, dt))

    inner_time = ensure_1d(data.get("inner_time", None))
    if inner_time is None:
        inner_time = normalize_01(time_vec)

    trial_id = ensure_1d(data.get("trial_id", None))

    return {
        "position": pos,
        "speed": speed,
        "time": time_vec,
        "inner_time": inner_time,
        "trial_id": trial_id,
        "direction": mov_dir,
    }


def get_si_params_for_behavior(behavior_name, behavior_values):
    if behavior_name in ["position", "speed", "time", "inner_time", "trial_id"]:
        params = copy.deepcopy(SI_PARAMS_CONTINUOUS)
        if behavior_name == "trial_id":
            params["min_label"] = float(np.nanmin(behavior_values))
            params["max_label"] = float(np.nanmax(behavior_values))
        return params

    if behavior_name == "direction":
        return copy.deepcopy(SI_PARAMS_DISCRETE)

    if behavior_name == "position_direction":
        return copy.deepcopy(SI_PARAMS_MIXED)

    raise ValueError(f"Unknown behavior: {behavior_name}")


# =============================================================================
# CSV Schema
# =============================================================================

SI_SESSION_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "modality",
    "standardized_file_path",
    "row_index",
    "data_space",
    "n_neurons",
    "n_timepoints_original",
    "n_timepoints_after_dir_filter",
    "n_timepoints_after_outlier_filter",
    "si_position",
    "si_speed",
    "si_time",
    "si_inner_time",
    "si_trial_id",
    "si_direction",
    "si_position_direction",
    "session_pickle_name",
    "session_pickle_path",
]


def init_csv(csv_path, columns):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_row(csv_path, columns, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


# =============================================================================
# Row Processing
# =============================================================================

def process_row(row_idx, row):
    # Support both column names: 'standardized_file_path' or legacy 'analysis_file_path'
    raw_path = row.get("standardized_file_path", row.get("analysis_file_path", ""))

    # Fix corrupted columns where path shifted into 'contains_pos'
    if str(raw_path).isdigit():
        raw_path = row.get("contains_pos", raw_path)

    session_path = resolve_path(raw_path)
    if not os.path.isfile(session_path):
        raise FileNotFoundError(f"Standardized session file not found: {raw_path}")

    payload = load_pickle_standard(session_path)

    # Signal matrix extraction (T x N)
    if "signal" in payload:
        raw_signal = np.asarray(payload["signal"])
    elif "clean_traces" in payload:
        raw_signal = np.asarray(payload["clean_traces"])
    else:
        raise KeyError(f"No signal matrix ('signal' or 'clean_traces') found in {session_path}")

    # Extract behavior
    n_time_candidate = raw_signal.shape[0]
    beh_dict = extract_behavior_dict(payload, n_time_candidate)

    # Transpose check: ensure dimension 0 is time (T) matching behavioral length
    time_len = len(beh_dict["time"])
    if raw_signal.shape[0] != time_len and raw_signal.shape[1] == time_len:
        raw_signal = raw_signal.T

    n_timepoints_original, n_neurons = raw_signal.shape
    if n_timepoints_original != len(beh_dict["time"]):
        raise ValueError(
            f"Signal timepoints ({n_timepoints_original}) do not match behavior ({len(beh_dict['time'])})"
        )

    # Signal conditioning
    signal = raw_signal.copy()
    if SMOOTH_SIGMA > 0:
        signal = gaussian_filter1d(signal, sigma=SMOOTH_SIGMA, axis=0)

    if Z_SCORE_NEURONS:
        mu = np.nanmean(signal, axis=0, keepdims=True)
        sd = np.nanstd(signal, axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        signal = (signal - mu) / sd

    # -------------------------------------------------------------------------
    # 1. Filter mov_dir == 0
    # -------------------------------------------------------------------------
    mov_dir = beh_dict["direction"]
    valid_idx = (mov_dir != 0) & np.isfinite(mov_dir)
    signal_valid = signal[valid_idx, :]

    beh_valid = {}
    for k, v in beh_dict.items():
        beh_valid[k] = None if v is None else v[valid_idx]

    n_timepoints_after_dir_filter = signal_valid.shape[0]
    if n_timepoints_after_dir_filter < 20:
        raise ValueError(
            f"Row {row_idx}: too few points after mov_dir != 0 filtering ({n_timepoints_after_dir_filter})")

    # -------------------------------------------------------------------------
    # 2. Density outlier filtering in Complete Neural Space
    # -------------------------------------------------------------------------
    D = pairwise_distances(signal_valid)
    noise_idx, signal_idx = filter_noisy_outliers(
        signal_valid,
        D=D,
        dist_th=OUTLIER_DIST_TH,
        noise_th=OUTLIER_NOISE_TH,
    )
    del D

    signal_clean = signal_valid[signal_idx, :]
    beh_clean = {}
    for k, v in beh_valid.items():
        beh_clean[k] = None if v is None else v[signal_idx]

    n_timepoints_after_outlier_filter = signal_clean.shape[0]
    if n_timepoints_after_outlier_filter < 20:
        raise ValueError(f"Row {row_idx}: too few points after outlier filtering ({n_timepoints_after_outlier_filter})")

    # Construct joint (position, direction) label
    pos_clean = beh_clean["position"].reshape(-1, 1)
    dir_clean = beh_clean["direction"].reshape(-1, 1)
    position_direction = np.concatenate([pos_clean, dir_clean], axis=1)

    behaviors_for_si = {
        "position": beh_clean["position"],
        "speed": beh_clean["speed"],
        "time": beh_clean["time"],
        "inner_time": beh_clean["inner_time"],
        "trial_id": beh_clean["trial_id"],
        "direction": beh_clean["direction"],
        "position_direction": position_direction,
    }

    # -------------------------------------------------------------------------
    # 3. Compute Structure Index in Complete Neural Space
    # -------------------------------------------------------------------------
    si_results = {}
    si_details = {}

    for beh_name in SI_BEHAVIOR_ORDER:
        beh_val = behaviors_for_si[beh_name]

        if beh_val is None or not np.isfinite(beh_val).any():
            si_results[beh_name] = np.nan
            si_details[beh_name] = None
            continue

        params = get_si_params_for_behavior(beh_name, beh_val)

        try:
            si, process_info, overlap_mat, shuf_si = compute_structure_index(
                signal_clean,
                beh_val,
                **params,
            )
            si_results[beh_name] = float(si)
            si_details[beh_name] = {
                "si": float(si),
                "process_info": copy.deepcopy(process_info),
                "overlap_mat": copy.deepcopy(overlap_mat),
                "shuf_si": copy.deepcopy(shuf_si),
                "params": copy.deepcopy(params),
            }
        except Exception as exc:
            print(f"    [SI WARN] {beh_name} failed: {exc}")
            si_results[beh_name] = np.nan
            si_details[beh_name] = None

    # Save session details
    session_base = base_name_from_row(row_idx, row) + "_si_neural_space"
    session_pickle_name = session_base + ".pkl"
    session_pickle_path = os.path.join(SESSION_PKL_DIR, session_pickle_name)

    session_payload = {
        "metadata": {
            "dataset_name": row.get("dataset_name", ""),
            "mouse_name": row.get("mouse_name", ""),
            "session_name": row.get("session_name", ""),
            "depth_type": row.get("depth_type", ""),
            "modality": row.get("modality", "calcium_imaging"),
            "row_index": row_idx,
            "standardized_file_path": session_path,
            "data_space": "complete_neural_space",
            "n_neurons": n_neurons,
            "n_timepoints_original": n_timepoints_original,
            "n_timepoints_after_dir_filter": n_timepoints_after_dir_filter,
            "n_timepoints_after_outlier_filter": n_timepoints_after_outlier_filter,
            "outlier_dist_th": OUTLIER_DIST_TH,
            "outlier_noise_th": OUTLIER_NOISE_TH,
        },
        "indices": {
            "valid_idx_mov_dir_nonzero": valid_idx,
            "noise_idx_after_dir_filter": noise_idx,
            "signal_idx_after_dir_filter": signal_idx,
        },
        "si": si_details,
    }

    save_pickle_standard(session_pickle_path, session_payload)

    session_row = {
        "dataset_name": row.get("dataset_name", ""),
        "mouse_name": row.get("mouse_name", ""),
        "session_name": row.get("session_name", ""),
        "depth_type": row.get("depth_type", ""),
        "modality": row.get("modality", "calcium_imaging"),
        "standardized_file_path": session_path,
        "row_index": row_idx,
        "data_space": "complete_neural_space",
        "n_neurons": n_neurons,
        "n_timepoints_original": n_timepoints_original,
        "n_timepoints_after_dir_filter": n_timepoints_after_dir_filter,
        "n_timepoints_after_outlier_filter": n_timepoints_after_outlier_filter,
        "si_position": si_results["position"],
        "si_speed": si_results["speed"],
        "si_time": si_results["time"],
        "si_inner_time": si_results["inner_time"],
        "si_trial_id": si_results["trial_id"],
        "si_direction": si_results["direction"],
        "si_position_direction": si_results["position_direction"],
        "session_pickle_name": session_pickle_name,
        "session_pickle_path": session_pickle_path,
    }

    def fmt(v):
        return f"{v:.4f}" if np.isfinite(v) else "NaN"

    print(
        f"Row {row_idx:03d} | {row.get('dataset_name')} | {row.get('mouse_name')} | "
        f"{row.get('session_name')} | {row.get('depth_type')} | N={n_neurons} | "
        f"SI pos={fmt(session_row['si_position'])} | "
        f"dir={fmt(session_row['si_direction'])} | "
        f"time={fmt(session_row['si_time'])} | "
        f"(pos,dir)={fmt(session_row['si_position_direction'])}"
    )

    return session_row


# =============================================================================
# Main Loop
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SESSION_PKL_DIR)
    ensure_dir(LOG_DIR)

    if not os.path.isfile(MASTER_INDEX_CSV):
        raise FileNotFoundError(f"Master metadata CSV not found: {MASTER_INDEX_CSV}")

    df = pd.read_csv(MASTER_INDEX_CSV)

    # Optional filter by dataset
    if DATASET_FILTER is not None:
        # Match 'dreadds' or case variations
        df = df[df["dataset_name"].astype(str).str.lower() == DATASET_FILTER.lower()].copy()

    df = df.reset_index(drop=True)
    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    init_csv(SI_SESSION_CSV, SI_SESSION_COLUMNS)

    log_path = os.path.join(LOG_DIR, "structure_index_neural_space_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Structure Index analysis in COMPLETE NEURAL SPACE")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Metadata CSV: {MASTER_INDEX_CSV}")
    print(f"Dataset filter: {DATASET_FILTER}")
    print(f"Total sessions: {len(df)}")
    print(f"Output table: {SI_SESSION_CSV}")
    print(f"Pickles dir:  {SESSION_PKL_DIR}")
    print("==============================================================\n")

    ok_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        t0 = timeit.default_timer()
        try:
            session_row = process_row(row_idx, row)
            append_csv_row(SI_SESSION_CSV, SI_SESSION_COLUMNS, session_row)
            ok_count += 1
            elapsed = timeit.default_timer() - t0
            print(f"[OK] Completed in {elapsed:.2f} s\n")
        except Exception as e:
            fail_count += 1
            print(f"[ERROR] Row {row_idx}: {e}\n")

    total_elapsed = timeit.default_timer() - global_time
    print("==============================================================")
    print("Execution Finished")
    print(f"End time: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Successful: {ok_count}")
    print(f"Failed:     {fail_count}")
    print(f"Total time: {total_elapsed:.2f} s")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()
