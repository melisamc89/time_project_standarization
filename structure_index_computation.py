#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure Index analysis pipeline in UMAP space.

This version assumes you already have the Structure Index code available
locally and importable from your environment.

What this script does
---------------------
1) Reads the UMAP/PCA analysis index CSV.
2) Loads each saved embedding-behavior pickle.
3) Uses the UMAP embedding (3D) as the data space for Structure Index.
4) Removes time points where mov_dir == 0.
5) Filters noisy outliers in UMAP space.
6) Computes Structure Index (SI) for:
       - position
       - speed
       - time
       - inner_time
       - trial_id
       - direction
       - position_direction  (2D label: [position, direction])
7) Saves:
       - one CSV with one row per session
       - one pickle per session with full SI outputs
       - one log txt file
"""

import os
import csv
import pickle
import copy
import timeit
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
from structure_index import compute_structure_index


# =============================================================================
# Paths
# =============================================================================

ANALYSIS_INDEX_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/umap_pca_per_row/umap_pca_analysis_index.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/structure_index_umap"
SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

SI_SESSION_CSV = os.path.join(OUTPUT_DIR, "structure_index_umap_per_session.csv")


# =============================================================================
# Parameters
# =============================================================================

MAX_ROWS = None
VERBOSE = False

# Outlier filtering parameters
OUTLIER_DIST_TH = 5
OUTLIER_NOISE_TH = 25

# Which embedding to use: "umap" or "pca"
EMBEDDING_KEY = "umap"

SI_PARAMS_CONTINUOUS = {
    "n_bins": 10,
    "discrete_label": False,
    "continuity_kernel": None,
    "perc_neigh": 1,
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
        f"{sanitize_for_filename(row.get('dataset_name', 'unknown_dataset'))}_"
        f"{sanitize_for_filename(row.get('mouse_name', 'unknown_mouse'))}_"
        f"{sanitize_for_filename(row.get('session_name', 'unknown_session'))}_"
        f"{sanitize_for_filename(row.get('depth_type', 'unknown_depth'))}"
    )


def infer_recording_type(row):
    text_to_search = " ".join([
        str(row.get("dataset_name", "")),
        str(row.get("analysis_file_path", "")),
    ]).lower()

    if any(k in text_to_search for k in ["calcium", "caim", "dff", "trace"]):
        return "calcium"
    if any(k in text_to_search for k in ["ephys", "lfp", "spike", "electro"]):
        return "ephys"

    return "unknown"


def validate_lengths(data_space, behavior_dict, row_idx):
    n_time = data_space.shape[0]
    for key, value in behavior_dict.items():
        if value is None:
            continue
        if len(value) != n_time:
            raise ValueError(
                f"Row {row_idx}: length mismatch for '{key}' "
                f"({len(value)} vs {n_time})"
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
# Noisy outlier filtering
# =============================================================================

def filter_noisy_outliers(data, D=None, dist_th=5, noise_th=25):
    """
    Filter outliers based on local density.

    Parameters
    ----------
    data : numpy array [t, n]
        t: timestamps; n: features
    D : numpy array [t, t] or None
        matrix of distances between rows
    dist_th : float
        percentile distance threshold below which a point is considered neighbor
    noise_th : float
        percentile threshold in number of neighbors below which points are outliers

    Returns
    -------
    noise_idx : np.ndarray
    signal_idx : np.ndarray
    """
    if D is None:
        D = pairwise_distances(data)

    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D, dist_th), axis=1)
    noise_idx = np.where(nn_dist < np.percentile(nn_dist, noise_th))[0]
    signal_idx = np.where(nn_dist >= np.percentile(nn_dist, noise_th))[0]

    return noise_idx, signal_idx


# =============================================================================
# Behavior parsing
# =============================================================================

def get_si_params_for_behavior(behavior_name, behavior_values):
    if behavior_name in ["position", "speed", "time", "inner_time", "trial_id"]:
        params = copy.deepcopy(SI_PARAMS_CONTINUOUS)
        params["n_neighbors"] = 50
        if behavior_name == "trial_id":
            params["min_label"] = float(np.min(behavior_values))
            params["max_label"] = float(np.max(behavior_values))
        return params

    if behavior_name == "direction":
        return copy.deepcopy(SI_PARAMS_DISCRETE)

    if behavior_name == "position_direction":
        return copy.deepcopy(SI_PARAMS_MIXED)

    raise ValueError(f"Unknown behavior: {behavior_name}")


# =============================================================================
# CSV schema
# =============================================================================

SI_SESSION_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "recording_type",
    "analysis_file_path",
    "row_index",
    "embedding_type",
    "embedding_dim",
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
# Main row processing
# =============================================================================

def process_row(row_idx, row):
    analysis_file_path = row["analysis_file_path"]

    if not os.path.isfile(analysis_file_path):
        raise FileNotFoundError(f"Missing analysis file: {analysis_file_path}")

    payload = load_pickle_standard(analysis_file_path)

    if EMBEDDING_KEY not in payload:
        raise KeyError(f"Missing embedding key '{EMBEDDING_KEY}' in {analysis_file_path}")
    if "behavior" not in payload:
        raise KeyError(f"Missing 'behavior' in {analysis_file_path}")

    data_space = ensure_2d(payload[EMBEDDING_KEY])
    behavior = payload["behavior"]

    pos = ensure_1d(behavior["pos"])
    mov_dir = ensure_1d(behavior["mov_dir"])
    time_1d = ensure_1d(behavior["time"])

    speed = ensure_1d(behavior["speed"]) if behavior.get("speed", None) is not None else None
    trial_id = ensure_1d(behavior["trial_id"]) if behavior.get("trial_id", None) is not None else None
    inner_time = ensure_1d(behavior["inner_time"]) if behavior.get("inner_time", None) is not None else None

    beh_dict = {
        "position": pos,
        "speed": speed,
        "time": time_1d,
        "inner_time": inner_time,
        "trial_id": trial_id,
        "direction": mov_dir,
    }

    validate_lengths(data_space, beh_dict, row_idx)

    n_timepoints_original = data_space.shape[0]
    embedding_dim = data_space.shape[1]

    # -------------------------------------------------------------------------
    # remove mov_dir == 0
    # -------------------------------------------------------------------------
    valid_idx = mov_dir != 0
    data_valid = data_space[valid_idx, :]

    beh_valid = {}
    for key, val in beh_dict.items():
        beh_valid[key] = None if val is None else val[valid_idx]

    n_timepoints_after_dir_filter = data_valid.shape[0]
    if n_timepoints_after_dir_filter < 10:
        raise ValueError(f"Row {row_idx}: too few timepoints after mov_dir != 0 filtering")

    # -------------------------------------------------------------------------
    # outlier filtering in embedding space
    # -------------------------------------------------------------------------
    D = pairwise_distances(data_valid)
    noise_idx, signal_idx = filter_noisy_outliers(
        data_valid,
        D=D,
        dist_th=OUTLIER_DIST_TH,
        noise_th=OUTLIER_NOISE_TH,
    )

    data_clean = data_valid[signal_idx, :]

    beh_clean = {}
    for key, val in beh_valid.items():
        beh_clean[key] = None if val is None else val[signal_idx]

    n_timepoints_after_outlier_filter = data_clean.shape[0]
    if n_timepoints_after_outlier_filter < 10:
        raise ValueError(f"Row {row_idx}: too few timepoints after outlier filtering")

    position_direction = np.concatenate(
        (
            beh_clean["position"].reshape(-1, 1),
            beh_clean["direction"].reshape(-1, 1),
        ),
        axis=1,
    )

    behaviors_for_si = {
        "position": beh_clean["position"],
        "speed": beh_clean["speed"],
        "time": beh_clean["time"],
        "inner_time": beh_clean["inner_time"],
        "trial_id": beh_clean["trial_id"],
        "direction": beh_clean["direction"],
        "position_direction": position_direction,
    }

    si_results = {}
    si_details = {}

    for beh_name in SI_BEHAVIOR_ORDER:
        beh_val = behaviors_for_si[beh_name]

        if beh_val is None:
            si_results[beh_name] = np.nan
            si_details[beh_name] = None
            continue

        params = get_si_params_for_behavior(beh_name, beh_val)

        si, process_info, overlap_mat, shuf_si = compute_structure_index(
            data_clean,
            beh_val,
            **params,
        )

        si_results[beh_name] = si
        si_details[beh_name] = {
            "si": copy.deepcopy(si),
            "process_info": copy.deepcopy(process_info),
            "overlap_mat": copy.deepcopy(overlap_mat),
            "shuf_si": copy.deepcopy(shuf_si),
            "params": copy.deepcopy(params),
        }

    recording_type = infer_recording_type(row)

    session_base = base_name_from_row(row_idx, row) + f"_si_{EMBEDDING_KEY}"
    session_pickle_name = session_base + ".pkl"
    session_pickle_path = os.path.join(SESSION_PKL_DIR, session_pickle_name)

    session_payload = {
        "metadata": {
            "dataset_name": row["dataset_name"],
            "mouse_name": row["mouse_name"],
            "session_name": row["session_name"],
            "depth_type": row["depth_type"],
            "recording_type": recording_type,
            "row_index": row_idx,
            "analysis_file_path": analysis_file_path,
            "embedding_type": EMBEDDING_KEY,
            "embedding_dim": embedding_dim,
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
        "dataset_name": row["dataset_name"],
        "mouse_name": row["mouse_name"],
        "session_name": row["session_name"],
        "depth_type": row["depth_type"],
        "recording_type": recording_type,
        "analysis_file_path": analysis_file_path,
        "row_index": row_idx,
        "embedding_type": EMBEDDING_KEY,
        "embedding_dim": embedding_dim,
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

    print(
        f"Row {row_idx} | dataset={row['dataset_name']} | mouse={row['mouse_name']} | "
        f"session={row['session_name']} | depth={row['depth_type']} | "
        f"emb={EMBEDDING_KEY} | "
        f"SI pos={session_row['si_position']:.4f} | "
        f"speed={session_row['si_speed']:.4f} | "
        f"time={session_row['si_time']:.4f} | "
        f"inner_time={session_row['si_inner_time']:.4f} | "
        f"trial_id={session_row['si_trial_id']:.4f} | "
        f"dir={session_row['si_direction']:.4f} | "
        f"(pos,dir)={session_row['si_position_direction']:.4f}"
    )

    return session_row


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SESSION_PKL_DIR)
    ensure_dir(LOG_DIR)

    if not os.path.isfile(ANALYSIS_INDEX_CSV):
        raise FileNotFoundError(f"Analysis index CSV not found: {ANALYSIS_INDEX_CSV}")

    df = pd.read_csv(ANALYSIS_INDEX_CSV)
    df = df[df["dataset_name"] == "dreadds"].copy()
    df = df.reset_index(drop=True)
    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    init_csv(SI_SESSION_CSV, SI_SESSION_COLUMNS)

    log_path = os.path.join(LOG_DIR, "structure_index_umap_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Structure Index analysis pipeline in embedding space")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {ANALYSIS_INDEX_CSV}")
    print(f"Embedding used: {EMBEDDING_KEY}")
    print(f"Rows to process: {len(df)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Session CSV: {SI_SESSION_CSV}")
    print(f"Session PKL dir: {SESSION_PKL_DIR}")
    print("")

    ok_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        local_time = timeit.default_timer()
        try:
            session_row = process_row(row_idx, row)
            append_csv_row(SI_SESSION_CSV, SI_SESSION_COLUMNS, session_row)
            ok_count += 1

            elapsed = timeit.default_timer() - local_time
            print(f"[OK] Row {row_idx} finished in {elapsed:.2f} s\n")

        except Exception as e:
            fail_count += 1
            print(f"[ERROR] Row {row_idx}: {e}\n")

    total_elapsed = timeit.default_timer() - global_time
    print("==============================================================")
    print("Done")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Successful entries: {ok_count}")
    print(f"Failed entries: {fail_count}")
    print(f"Total elapsed time: {total_elapsed:.2f} s")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()


########################################################################
######################## PLOTTING RESULTS ###################################
########################################################################

