#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mutual Information Analysis Pipeline with Dynamic Path & Schema Correction.

Fixes:
1. Translates old root paths (/home/melisamc/...) to active paths (/home/hippolab/Documents/melisa/...).
2. Recovers malformed/shifted paths (e.g. '1000' in shifted CSV rows).
3. Handles '_pkl' vs '.pkl' filename differences.
4. Computes per-neuron MI against behaviors and full LFP features (envelopes and circular phases).
"""

import os
import csv
import glob
import pickle
import timeit
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# =============================================================================
# Paths & Settings
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
OLD_PROJECT_ROOT = "/home/melisamc/Documents/time_project_standarization"

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "master_dataset_index.csv")
STANDARDIZED_DIR = os.path.join(PROJECT_ROOT, "data", "standardized_sessions")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "mutual_information_behavior_and_lfp")
SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

NEURON_MI_CSV = os.path.join(OUTPUT_DIR, "mutual_information_per_neuron.csv")
SESSION_MI_CSV = os.path.join(OUTPUT_DIR, "mutual_information_per_session_summary.csv")

MAX_ROWS = None
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

LFP_BAND_NAMES = [
    "ultra_ultra_slow",
    "ultra_slow",
    "slow",
    "delta",
    "theta",
    "gamma",
    "ripple",
    "mua",
]

LFP_WAVEFORM_LABELS = [f"lfp_{band}" for band in LFP_BAND_NAMES]
LFP_ENVELOPE_LABELS = [f"lfp_{band}_envelope" for band in LFP_BAND_NAMES]
LFP_PHASE_LABELS = [f"lfp_{band}_phase" for band in LFP_BAND_NAMES]

ALL_FEATURE_LABELS = (
        BEHAVIOR_LABELS
        + LFP_WAVEFORM_LABELS
        + LFP_ENVELOPE_LABELS
        + LFP_PHASE_LABELS
)


# =============================================================================
# Helper Utilities
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
    return "".join([ch if (ch.isalnum() or ch in ["-", "_"]) else "_" for ch in text])


def base_name_from_row(row_idx, row):
    return (
        f"row_{row_idx:04d}_"
        f"{sanitize_for_filename(row.get('dataset_name', 'unknown_dataset'))}_"
        f"{sanitize_for_filename(row.get('mouse_name', 'unknown_mouse'))}_"
        f"{sanitize_for_filename(row.get('session_name', 'unknown_session'))}_"
        f"{sanitize_for_filename(row.get('depth_type', 'unknown_depth'))}"
    )


# =============================================================================
# Dynamic Path Resolver
# =============================================================================

def resolve_actual_file_path(row):
    """
    Resolves the actual standardized .pkl path on disk, correcting:
    1. Base directory shifts (/home/melisamc/... -> /home/hippolab/Documents/melisa/...)
    2. Column-shifted entries (e.g. '1000' in place of path)
    3. Extensions named '_pkl' instead of '.pkl'
    """
    raw_path = str(row.get("standardized_file_path", "")).strip()
    dataset_name = str(row.get("dataset_name", "")).strip()
    mouse_name = str(row.get("mouse_name", "")).strip()
    depth_type = str(row.get("depth_type", "")).strip()

    candidate_paths = []

    # 1. Direct translation of old root to new root
    if raw_path and not raw_path.replace(".", "", 1).isdigit():
        if raw_path.startswith(OLD_PROJECT_ROOT):
            migrated = raw_path.replace(OLD_PROJECT_ROOT, PROJECT_ROOT)
            candidate_paths.append(migrated)
            if migrated.endswith("_pkl"):
                candidate_paths.append(migrated[:-4] + ".pkl")
        candidate_paths.append(raw_path)
        if raw_path.endswith("_pkl"):
            candidate_paths.append(raw_path[:-4] + ".pkl")

    # 2. Check candidate paths
    for p in candidate_paths:
        if os.path.isfile(p):
            return p

    # 3. Reconstruct path from metadata if column was shifted (e.g. '1000')
    if dataset_name and mouse_name:
        dataset_dir = os.path.join(STANDARDIZED_DIR, dataset_name)
        if os.path.isdir(dataset_dir):
            patterns = [
                os.path.join(dataset_dir, "**", f"*{mouse_name}*.pkl"),
                os.path.join(dataset_dir, "**", f"*{mouse_name}*_pkl"),
            ]
            for pat in patterns:
                matches = glob.glob(pat, recursive=True)
                if matches:
                    if depth_type:
                        for m in matches:
                            if depth_type in m:
                                return m
                    return matches[0]

    # 4. Search across all of STANDARDIZED_DIR by mouse_name
    if mouse_name:
        matches = glob.glob(os.path.join(STANDARDIZED_DIR, "**", f"*{mouse_name}*.pkl"), recursive=True)
        if matches:
            if depth_type:
                for m in matches:
                    if depth_type in m:
                        return m
            return matches[0]

    return None


# =============================================================================
# Mutual Information Computation
# =============================================================================

def compute_mi_feature_to_signal(feature_arr, signal_2d, n_neighbors=50, random_state=16):
    x = ensure_2d(feature_arr)
    signal_2d = ensure_2d(signal_2d)
    n_neurons = signal_2d.shape[1]
    mi_values = np.full(n_neurons, np.nan, dtype=float)

    valid_mask = ~np.any(np.isnan(x), axis=1)
    if np.sum(valid_mask) < 20 or np.any(np.std(x[valid_mask, :], axis=0) == 0):
        return mi_values

    x_clean = x[valid_mask, :]

    for neuron_idx in range(n_neurons):
        y = signal_2d[valid_mask, neuron_idx]
        if np.std(y) == 0:
            mi_values[neuron_idx] = 0.0
            continue

        mi = mutual_info_regression(
            x_clean,
            y,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        mi_values[neuron_idx] = mi[0]

    return mi_values


# =============================================================================
# CSV Schemas
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
                    ] + [f"mi_{feat}" for feat in ALL_FEATURE_LABELS]

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
]
for feat in ALL_FEATURE_LABELS:
    SESSION_MI_COLUMNS.extend([f"mi_{feat}_mean", f"mi_{feat}_std"])

SESSION_MI_COLUMNS.extend(["session_pickle_name", "session_pickle_path"])


def init_csv(csv_path, columns):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_row(csv_path, columns, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
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
# Session Processing Routine
# =============================================================================

def process_row(row_idx, row):
    actual_file_path = resolve_actual_file_path(row)

    if actual_file_path is None or not os.path.isfile(actual_file_path):
        raise FileNotFoundError(
            f"Could not resolve file for row {row_idx}: "
            f"dataset={row.get('dataset_name')}, mouse={row.get('mouse_name')}, "
            f"raw_path={row.get('standardized_file_path')}"
        )

    data = load_pickle_standard(actual_file_path)

    required_keys = ["signal", "pos", "mov_dir", "time"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in: {actual_file_path}")

    signal = ensure_2d(data["signal"])
    n_timepoints, n_neurons = signal.shape
    if n_neurons < 1:
        raise ValueError(f"Row {row_idx}: Session has no neurons.")

    # 1. Behavioral Features
    pos_1d = ensure_1d(get_pos_1d(data["pos"]))
    mov_dir = ensure_1d(data["mov_dir"])
    time_1d = ensure_1d(data["time"])
    speed = ensure_1d(data["speed"]) if data.get("speed") is not None else None
    inner_time = ensure_1d(data["inner_time"]) if data.get("inner_time") is not None else None
    trial_id = ensure_1d(data["trial_id"]) if data.get("trial_id") is not None else None

    features_dict = {
        "position": pos_1d,
        "position_x_direction": pos_1d * mov_dir,
        "speed": speed,
        "time": time_1d,
        "inner_time": inner_time,
        "direction": mov_dir,
        "trial_id": trial_id,
    }

    # 2. LFP Waveforms, Envelopes, and Circular Phases
    for band in LFP_BAND_NAMES:
        wav_key = f"lfp_{band}"
        env_key = f"lfp_{band}_envelope"
        phase_key = f"lfp_{band}_phase"

        features_dict[wav_key] = ensure_1d(data[wav_key]) if wav_key in data and data[wav_key] is not None else None
        features_dict[env_key] = ensure_1d(data[env_key]) if env_key in data and data[env_key] is not None else None

        if phase_key in data and data[phase_key] is not None:
            phi = ensure_1d(data[phase_key])
            features_dict[phase_key] = np.column_stack([np.cos(phi), np.sin(phi)])
        else:
            features_dict[phase_key] = None

    # Length alignment validation
    for feat_name, arr in features_dict.items():
        if arr is not None and len(arr) != n_timepoints:
            raise ValueError(f"Row {row_idx}: Length mismatch for '{feat_name}' ({len(arr)} vs {n_timepoints}).")

    if VERBOSE:
        print(
            f"Row {row_idx:03d} | dataset={row.get('dataset_name', '')} | mouse={row.get('mouse_name', '')} | "
            f"session={row.get('session_name', '')} | depth={row.get('depth_type', '')} | "
            f"timepoints={n_timepoints} | neurons={n_neurons}"
        )

    # 3. Compute MI across all features
    mi_dict = {}
    for label in ALL_FEATURE_LABELS:
        feat = features_dict.get(label)
        if feat is None:
            mi_dict[label] = None
        else:
            mi_dict[label] = compute_mi_feature_to_signal(
                feature_arr=feat,
                signal_2d=signal,
                n_neighbors=MI_N_NEIGHBORS,
                random_state=MI_RANDOM_STATE,
            )

    recording_type = "calcium" if "calcium" in str(row.get("modality", "")).lower() else "ephys"

    # 4. Save Session Pickle
    session_base = base_name_from_row(row_idx, row) + "_behavior_and_lfp_mi"
    session_pickle_name = session_base + ".pkl"
    session_pickle_path = os.path.join(SESSION_PKL_DIR, session_pickle_name)

    session_payload = {
        "metadata": {
            "dataset_name": row.get("dataset_name", ""),
            "mouse_name": row.get("mouse_name", ""),
            "session_name": row.get("session_name", ""),
            "depth_type": row.get("depth_type", ""),
            "recording_type": recording_type,
            "row_index": row_idx,
            "standardized_file_path": actual_file_path,
            "n_timepoints": n_timepoints,
            "n_neurons": n_neurons,
            "mi_method": "mutual_info_regression",
            "mi_n_neighbors": MI_N_NEIGHBORS,
            "mi_random_state": MI_RANDOM_STATE,
            "feature_labels": ALL_FEATURE_LABELS,
        },
        "mi": mi_dict,
    }
    save_pickle_standard(session_pickle_path, session_payload)

    # 5. Per-Neuron Rows
    neuron_rows = []
    for neuron_idx in range(n_neurons):
        nrow = {
            "dataset_name": row.get("dataset_name", ""),
            "mouse_name": row.get("mouse_name", ""),
            "session_name": row.get("session_name", ""),
            "depth_type": row.get("depth_type", ""),
            "recording_type": recording_type,
            "standardized_file_path": actual_file_path,
            "row_index": row_idx,
            "neuron_index": neuron_idx,
            "n_timepoints": n_timepoints,
            "n_neurons_in_session": n_neurons,
        }
        for label in ALL_FEATURE_LABELS:
            nrow[f"mi_{label}"] = mi_dict[label][neuron_idx] if mi_dict[label] is not None else np.nan
        neuron_rows.append(nrow)

    # 6. Session Summary Row
    session_row = {
        "dataset_name": row.get("dataset_name", ""),
        "mouse_name": row.get("mouse_name", ""),
        "session_name": row.get("session_name", ""),
        "depth_type": row.get("depth_type", ""),
        "recording_type": recording_type,
        "standardized_file_path": actual_file_path,
        "row_index": row_idx,
        "n_timepoints": n_timepoints,
        "n_neurons": n_neurons,
        "session_pickle_name": session_pickle_name,
        "session_pickle_path": session_pickle_path,
    }
    for label in ALL_FEATURE_LABELS:
        arr = mi_dict[label]
        session_row[f"mi_{label}_mean"] = np.nanmean(arr) if arr is not None else np.nan
        session_row[f"mi_{label}_std"] = np.nanstd(arr) if arr is not None else np.nan

    if VERBOSE:
        has_theta_env = mi_dict.get("lfp_theta_envelope") is not None
        theta_env_str = f"{session_row.get('mi_lfp_theta_envelope_mean', np.nan):.4f}" if has_theta_env else "N/A"
        print(
            f"   pos={session_row['mi_position_mean']:.4f} | "
            f"speed={session_row['mi_speed_mean']:.4f} | "
            f"theta_env={theta_env_str}"
        )

    return neuron_rows, session_row


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SESSION_PKL_DIR)
    ensure_dir(LOG_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"Master CSV index not found: {CSV_PATH}")

    try:
        df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(CSV_PATH)

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    init_csv(NEURON_MI_CSV, NEURON_MI_COLUMNS)
    init_csv(SESSION_MI_CSV, SESSION_MI_COLUMNS)

    log_path = os.path.join(LOG_DIR, "mutual_information_behavior_and_lfp_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Mutual Information Pipeline: Behavior, LFP Envelopes & Phases")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Sessions in index: {len(df)}")
    print(f"Features: {len(ALL_FEATURE_LABELS)} (7 Behavior, 8 Waveform, 8 Envelope, 8 Phase)")
    print(f"Output directory: {OUTPUT_DIR}\n")

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
            print(f"[OK] Row {row_idx} completed in {elapsed:.2f} s\n")

        except Exception as e:
            fail_count += 1
            print(f"[ERROR] Row {row_idx}: {e}\n")

    total_elapsed = timeit.default_timer() - global_time
    print("==============================================================")
    print("Mutual Information Analysis Complete")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Successful sessions: {ok_count} / {len(df)}")
    print(f"Failed sessions    : {fail_count}")
    print(f"Total elapsed time : {total_elapsed:.2f} s")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()
