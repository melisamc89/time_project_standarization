#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mutual Information analysis pipeline for standardized datasets.

What this script does
---------------------
1) Reads the master standardized dataset index CSV.
2) Loads each standardized session pickle.
3) Computes per-neuron mutual information with:
       - position
       - position_x_direction
       - speed
       - time
       - inner_time
       - direction
       - trial_id
4) Saves:
       - one CSV with one row per neuron across all datasets/sessions
       - one CSV with one row per session summarizing mean/std MI
       - one pickle per session with full MI arrays
       - one log txt file

Assumptions
-----------
- signal is shaped: (n_timepoints, n_neurons)
- pos may be:
      * 1D: (n_timepoints,)
      * 2D: (n_timepoints, d)
  In this script, "position" is defined as the first coordinate: pos[:, 0]
- "position_x_direction" = position_1d * mov_dir
- mutual_info_regression is used as in the previous code base
"""

import os
import csv
import pickle
import timeit
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


# =============================================================================
# Paths
# =============================================================================

CSV_PATH = "/home/melisamc/Documents/time_project_standarization/data/metadata/master_dataset_index.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/mutual_information"
SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

NEURON_MI_CSV = os.path.join(OUTPUT_DIR, "mutual_information_per_neuron.csv")
SESSION_MI_CSV = os.path.join(OUTPUT_DIR, "mutual_information_per_session_summary.csv")


# =============================================================================
# Parameters
# =============================================================================

MAX_ROWS = None                # set to integer for testing
MI_N_NEIGHBORS = 50
MI_RANDOM_STATE = 16
VERBOSE = True

BEHAVIOR_LABELS = [
    "position",
    "position_x_direction",
    "speed",
    "time",
    "inner_time",
    "direction",
    "trial_id",
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


def get_pos_1d(pos):
    pos = np.asarray(pos)
    if pos.ndim == 1:
        return pos.reshape(-1)
    if pos.ndim == 2:
        return pos[:, 0].reshape(-1)
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
        f"{sanitize_for_filename(row.get('dataset_name', 'unknown_dataset'))}_"
        f"{sanitize_for_filename(row.get('mouse_name', 'unknown_mouse'))}_"
        f"{sanitize_for_filename(row.get('session_name', 'unknown_session'))}_"
        f"{sanitize_for_filename(row.get('depth_type', 'unknown_depth'))}"
    )


def infer_recording_type(row):
    """
    Returns 'calcium', 'ephys', or 'unknown'.

    Priority:
    1) explicit columns if present
    2) infer from dataset_name / file path
    """
    candidate_columns = [
        "recording_type",
        "modality",
        "signal_type",
        "dataset_type",
        "data_type",
    ]
    for col in candidate_columns:
        if col in row and pd.notna(row[col]):
            value = str(row[col]).strip().lower()
            if "calcium" in value:
                return "calcium"
            if "ephys" in value or "electro" in value:
                return "ephys"

    text_to_search = " ".join([
        str(row.get("dataset_name", "")),
        str(row.get("standardized_file_path", "")),
    ]).lower()

    if any(k in text_to_search for k in ["calcium", "caim", "dff", "trace"]):
        return "calcium"
    if any(k in text_to_search for k in ["ephys", "lfp", "spike", "electro"]):
        return "ephys"

    return "unknown"


def validate_lengths(signal, behavior_dict, row_idx):
    n_time = signal.shape[0]
    for key, value in behavior_dict.items():
        if value is None:
            continue
        if len(value) != n_time:
            raise ValueError(
                f"Row {row_idx}: length mismatch for '{key}' "
                f"({len(value)} vs {n_time})"
            )


def compute_single_feature_mi(feature_1d, signal_2d, n_neighbors=50, random_state=16):
    """
    Compute MI(feature -> neuron signal) for all neurons independently.

    feature_1d : shape (n_time,)
    signal_2d  : shape (n_time, n_neurons)

    Returns
    -------
    mi_values : shape (n_neurons,)
    """
    x = ensure_1d(feature_1d).reshape(-1, 1)
    signal_2d = ensure_2d(signal_2d)

    n_neurons = signal_2d.shape[1]
    mi_values = np.full(n_neurons, np.nan, dtype=float)

    for neuron_idx in range(n_neurons):
        y = signal_2d[:, neuron_idx]
        mi = mutual_info_regression(
            x,
            y,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        mi_values[neuron_idx] = mi[0]

    return mi_values


def build_behavior_features(data):
    """
    Create the behavioral variables used for MI.
    """
    signal = ensure_2d(data["signal"])
    pos_1d = ensure_1d(get_pos_1d(data["pos"]))
    mov_dir = ensure_1d(data["mov_dir"])
    time_1d = ensure_1d(data["time"])

    speed = ensure_1d(data["speed"]) if data.get("speed", None) is not None else None
    inner_time = ensure_1d(data["inner_time"]) if data.get("inner_time", None) is not None else None
    trial_id = ensure_1d(data["trial_id"]) if data.get("trial_id", None) is not None else None

    position_x_direction = pos_1d * mov_dir

    behavior_dict = {
        "position": pos_1d,
        "position_x_direction": position_x_direction,
        "speed": speed,
        "time": time_1d,
        "inner_time": inner_time,
        "direction": mov_dir,
        "trial_id": trial_id,
    }

    validate_lengths(signal, behavior_dict, row_idx="unknown")

    return signal, behavior_dict


def compute_mi_for_session(data):
    """
    Returns
    -------
    mi_dict : dict
        keys are behavior names, values are arrays of shape (n_neurons,)
    """
    signal, behavior_dict = build_behavior_features(data)

    mi_dict = {}
    for label in BEHAVIOR_LABELS:
        feature = behavior_dict[label]
        if feature is None:
            mi_dict[label] = None
        else:
            mi_dict[label] = compute_single_feature_mi(
                feature_1d=feature,
                signal_2d=signal,
                n_neighbors=MI_N_NEIGHBORS,
                random_state=MI_RANDOM_STATE,
            )
    return mi_dict, signal.shape[0], signal.shape[1]


# =============================================================================
# CSV schemas
# =============================================================================

NEURON_MI_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "recording_type",
    "standardized_file_path",
    "row_index",
    "neuron_index",
    "n_timepoints",
    "n_neurons_in_session",
    "mi_position",
    "mi_position_x_direction",
    "mi_speed",
    "mi_time",
    "mi_inner_time",
    "mi_direction",
    "mi_trial_id",
]

SESSION_MI_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "recording_type",
    "standardized_file_path",
    "row_index",
    "n_timepoints",
    "n_neurons",
    "mi_position_mean",
    "mi_position_std",
    "mi_position_x_direction_mean",
    "mi_position_x_direction_std",
    "mi_speed_mean",
    "mi_speed_std",
    "mi_time_mean",
    "mi_time_std",
    "mi_inner_time_mean",
    "mi_inner_time_std",
    "mi_direction_mean",
    "mi_direction_std",
    "mi_trial_id_mean",
    "mi_trial_id_std",
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
# Logging
# =============================================================================

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
# Main processing
# =============================================================================

def process_row(row_idx, row):
    standardized_file_path = row["standardized_file_path"]

    if not os.path.isfile(standardized_file_path):
        raise FileNotFoundError(f"Missing standardized file: {standardized_file_path}")

    data = load_pickle_standard(standardized_file_path)

    required_keys = ["signal", "pos", "mov_dir", "time"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in file: {standardized_file_path}")

    signal = ensure_2d(data["signal"])
    pos_1d = ensure_1d(get_pos_1d(data["pos"]))
    mov_dir = ensure_1d(data["mov_dir"])
    time_1d = ensure_1d(data["time"])

    speed = ensure_1d(data["speed"]) if data.get("speed", None) is not None else None
    inner_time = ensure_1d(data["inner_time"]) if data.get("inner_time", None) is not None else None
    trial_id = ensure_1d(data["trial_id"]) if data.get("trial_id", None) is not None else None

    behavior_dict = {
        "position": pos_1d,
        "position_x_direction": pos_1d * mov_dir,
        "speed": speed,
        "time": time_1d,
        "inner_time": inner_time,
        "direction": mov_dir,
        "trial_id": trial_id,
    }
    validate_lengths(signal, behavior_dict, row_idx=row_idx)

    n_timepoints = signal.shape[0]
    n_neurons = signal.shape[1]

    if n_neurons < 1:
        raise ValueError(f"Row {row_idx}: signal has no neurons")

    if VERBOSE:
        print(
            f"Row {row_idx} | dataset={row['dataset_name']} | mouse={row['mouse_name']} | "
            f"session={row['session_name']} | depth={row['depth_type']} | "
            f"timepoints={n_timepoints} | neurons={n_neurons}"
        )

    mi_dict, _, _ = compute_mi_for_session(data)

    recording_type = infer_recording_type(row)

    # Save session pickle
    session_base = base_name_from_row(row_idx, row) + "_mi"
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
            "standardized_file_path": standardized_file_path,
            "n_timepoints": n_timepoints,
            "n_neurons": n_neurons,
            "mi_method": "mutual_info_regression",
            "mi_n_neighbors": MI_N_NEIGHBORS,
            "mi_random_state": MI_RANDOM_STATE,
            "behavior_labels": BEHAVIOR_LABELS,
        },
        "mi": mi_dict,
    }
    save_pickle_standard(session_pickle_path, session_payload)

    # Build per-neuron rows
    neuron_rows = []
    for neuron_idx in range(n_neurons):
        neuron_rows.append({
            "dataset_name": row["dataset_name"],
            "mouse_name": row["mouse_name"],
            "session_name": row["session_name"],
            "depth_type": row["depth_type"],
            "recording_type": recording_type,
            "standardized_file_path": standardized_file_path,
            "row_index": row_idx,
            "neuron_index": neuron_idx,
            "n_timepoints": n_timepoints,
            "n_neurons_in_session": n_neurons,
            "mi_position": mi_dict["position"][neuron_idx] if mi_dict["position"] is not None else np.nan,
            "mi_position_x_direction": mi_dict["position_x_direction"][neuron_idx] if mi_dict["position_x_direction"] is not None else np.nan,
            "mi_speed": mi_dict["speed"][neuron_idx] if mi_dict["speed"] is not None else np.nan,
            "mi_time": mi_dict["time"][neuron_idx] if mi_dict["time"] is not None else np.nan,
            "mi_inner_time": mi_dict["inner_time"][neuron_idx] if mi_dict["inner_time"] is not None else np.nan,
            "mi_direction": mi_dict["direction"][neuron_idx] if mi_dict["direction"] is not None else np.nan,
            "mi_trial_id": mi_dict["trial_id"][neuron_idx] if mi_dict["trial_id"] is not None else np.nan,
        })

    # Build session summary row
    session_row = {
        "dataset_name": row["dataset_name"],
        "mouse_name": row["mouse_name"],
        "session_name": row["session_name"],
        "depth_type": row["depth_type"],
        "recording_type": recording_type,
        "standardized_file_path": standardized_file_path,
        "row_index": row_idx,
        "n_timepoints": n_timepoints,
        "n_neurons": n_neurons,
        "mi_position_mean": np.nanmean(mi_dict["position"]) if mi_dict["position"] is not None else np.nan,
        "mi_position_std": np.nanstd(mi_dict["position"]) if mi_dict["position"] is not None else np.nan,
        "mi_position_x_direction_mean": np.nanmean(mi_dict["position_x_direction"]) if mi_dict["position_x_direction"] is not None else np.nan,
        "mi_position_x_direction_std": np.nanstd(mi_dict["position_x_direction"]) if mi_dict["position_x_direction"] is not None else np.nan,
        "mi_speed_mean": np.nanmean(mi_dict["speed"]) if mi_dict["speed"] is not None else np.nan,
        "mi_speed_std": np.nanstd(mi_dict["speed"]) if mi_dict["speed"] is not None else np.nan,
        "mi_time_mean": np.nanmean(mi_dict["time"]) if mi_dict["time"] is not None else np.nan,
        "mi_time_std": np.nanstd(mi_dict["time"]) if mi_dict["time"] is not None else np.nan,
        "mi_inner_time_mean": np.nanmean(mi_dict["inner_time"]) if mi_dict["inner_time"] is not None else np.nan,
        "mi_inner_time_std": np.nanstd(mi_dict["inner_time"]) if mi_dict["inner_time"] is not None else np.nan,
        "mi_direction_mean": np.nanmean(mi_dict["direction"]) if mi_dict["direction"] is not None else np.nan,
        "mi_direction_std": np.nanstd(mi_dict["direction"]) if mi_dict["direction"] is not None else np.nan,
        "mi_trial_id_mean": np.nanmean(mi_dict["trial_id"]) if mi_dict["trial_id"] is not None else np.nan,
        "mi_trial_id_std": np.nanstd(mi_dict["trial_id"]) if mi_dict["trial_id"] is not None else np.nan,
        "session_pickle_name": session_pickle_name,
        "session_pickle_path": session_pickle_path,
    }

    if VERBOSE:
        print(
            "  Means | "
            f"pos={session_row['mi_position_mean']:.4f} | "
            f"pos*dir={session_row['mi_position_x_direction_mean']:.4f} | "
            f"speed={session_row['mi_speed_mean']:.4f} | "
            f"time={session_row['mi_time_mean']:.4f} | "
            f"inner_time={session_row['mi_inner_time_mean']:.4f} | "
            f"dir={session_row['mi_direction_mean']:.4f} | "
            f"trial_id={session_row['mi_trial_id_mean']:.4f}"
        )

    return neuron_rows, session_row


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SESSION_PKL_DIR)
    ensure_dir(LOG_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    init_csv(NEURON_MI_CSV, NEURON_MI_COLUMNS)
    init_csv(SESSION_MI_CSV, SESSION_MI_COLUMNS)

    log_path = os.path.join(LOG_DIR, "mutual_information_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Mutual Information analysis pipeline")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Rows to process: {len(df)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Neuron CSV: {NEURON_MI_CSV}")
    print(f"Session CSV: {SESSION_MI_CSV}")
    print(f"Session PKL dir: {SESSION_PKL_DIR}")
    print(f"MI n_neighbors: {MI_N_NEIGHBORS}")
    print(f"MI random_state: {MI_RANDOM_STATE}")
    print("")

    ok_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        local_time = timeit.default_timer()
        try:
            neuron_rows, session_row = process_row(row_idx, row)

            for nr in neuron_rows:
                append_csv_row(NEURON_MI_CSV, NEURON_MI_COLUMNS, nr)

            append_csv_row(SESSION_MI_CSV, SESSION_MI_COLUMNS, session_row)

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