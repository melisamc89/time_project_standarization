#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mutual Information Analysis Pipeline for Standardized Neural Datasets (Multi-Band LFP, Envelopes, Phases, & CWT).

Features:
- Automatic path redirection for old CSV paths to the new project root.
- Granular verification: Checks per-neuron progress (row_index, mouse_name, session_name, depth_type, neuron_index).
- Skips fully computed sessions/neurons and dynamically computes/appends only missing ones.
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
# Paths & Auto-Correction
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "master_dataset_index.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "mutual_information_behavior_and_lfp")
SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

NEURON_MI_CSV = os.path.join(OUTPUT_DIR, "mutual_information_per_neuron.csv")
SESSION_MI_CSV = os.path.join(OUTPUT_DIR, "mutual_information_per_session_summary.csv")


def correct_path(path_str):
    """Translates legacy master CSV paths to the current project root."""
    if not isinstance(path_str, str):
        return path_str
    if "/home/melisamc/Documents/time_project_standarization" in path_str:
        path_str = path_str.replace("/home/melisamc/Documents/time_project_standarization", PROJECT_ROOT)
    return path_str


# =============================================================================
# Parameters & Band Configuration
# =============================================================================

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
    "slow",
    "delta",
    "theta",
    "gamma",
    "ripple",
    "mua",
]

# Aligned perfectly with ingestion frequency band definitions
LFP_BAND_LIMITS = {
    "slow": (0.1, 1),
    "delta": (1.00, 3.00),
    "theta": (8.00, 12.00),
    "gamma": (40.00, 90.00),
    "ripple": (100.0, 250.0),
    "mua": (300.0, 1000.0),
}

LFP_WAVEFORM_LABELS = [f"lfp_{band}" for band in LFP_BAND_NAMES]
LFP_ENVELOPE_LABELS = [f"lfp_{band}_envelope" for band in LFP_BAND_NAMES]
LFP_PHASE_LABELS = [f"lfp_{band}_phase" for band in LFP_BAND_NAMES]
CWT_POWER_LABELS = [f"cwt_{band}" for band in LFP_BAND_NAMES]

ALL_FEATURE_LABELS = (
    BEHAVIOR_LABELS
    + LFP_WAVEFORM_LABELS
    + LFP_ENVELOPE_LABELS
    + LFP_PHASE_LABELS
    + CWT_POWER_LABELS
)


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
    return "".join([ch if (ch.isalnum() or ch in ["-", "_"]) else "_" for ch in text])


def base_name_from_row(row_idx, row):
    return (
        f"row_{row_idx:04d}_"
        f"{sanitize_for_filename(row.get('dataset_name', 'unknown_dataset'))}_"
        f"{sanitize_for_filename(row.get('mouse_name', 'unknown_mouse'))}_"
        f"{sanitize_for_filename(row.get('session_name', 'unknown_session'))}_"
        f"{sanitize_for_filename(row.get('depth_type', 'unknown_depth'))}"
    )


def infer_recording_type(row):
    for col in ["recording_type", "modality", "signal_type", "dataset_type"]:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip().lower()
            if "calcium" in val:
                return "calcium"
            if "ephys" in val or "electro" in val:
                return "ephys"

    text = " ".join([str(row.get("dataset_name", "")), str(row.get("standardized_file_path", ""))]).lower()
    if any(k in text for k in ["calcium", "caim", "dff", "trace"]):
        return "calcium"
    if any(k in text for k in ["ephys", "lfp", "spike", "electro"]):
        return "ephys"
    return "unknown"


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
# CSV Column Schemas
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
    if "standardized_file_path" in row:
        row["standardized_file_path"] = correct_path(row["standardized_file_path"])
    if "session_pickle_path" in row:
        row["session_pickle_path"] = correct_path(row["session_pickle_path"])

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
# Progress Tracking / Verification Helper
# =============================================================================

def load_computed_neuron_progress(neuron_csv_path):
    computed_set = set()
    if not os.path.isfile(neuron_csv_path):
        return computed_set

    try:
        df_neuron = pd.read_csv(neuron_csv_path, engine="python", on_bad_lines="skip",
                                usecols=["row_index", "mouse_name", "session_name", "depth_type", "neuron_index"])
        for _, r in df_neuron.iterrows():
            key = (
                int(r["row_index"]),
                str(r["mouse_name"]).strip(),
                str(r["session_name"]).strip(),
                str(r["depth_type"]).strip(),
                int(r["neuron_index"])
            )
            computed_set.add(key)
    except Exception as e:
        print(f"[WARNING] Could not parse existing neuron MI CSV for verification: {e}")

    return computed_set


# =============================================================================
# Session Processing Pipeline
# =============================================================================

def process_row(row_idx, row, computed_neuron_set):
    raw_std_path = row.get("standardized_file_path", "")
    standardized_file_path = correct_path(raw_std_path)

    if not os.path.isfile(standardized_file_path):
        raise FileNotFoundError(f"Missing standardized file: {standardized_file_path}")

    data = load_pickle_standard(standardized_file_path)

    required_keys = ["signal", "pos", "mov_dir", "time"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in: {standardized_file_path}")

    signal = ensure_2d(data["signal"])
    n_timepoints, n_neurons = signal.shape
    if n_neurons < 1:
        raise ValueError(f"Row {row_idx}: Session has no neurons.")

    mouse_name = str(row.get("mouse_name", "")).strip()
    session_name = str(row.get("session_name", "")).strip()
    depth_type = str(row.get("depth_type", "")).strip()

    missing_neurons = [
        n_idx for n_idx in range(n_neurons)
        if (int(row_idx), mouse_name, session_name, depth_type, n_idx) not in computed_neuron_set
    ]

    session_pickle_name = base_name_from_row(row_idx, row) + "_behavior_and_lfp_mi.pkl"
    session_pickle_path = os.path.join(SESSION_PKL_DIR, session_pickle_name)

    if len(missing_neurons) == 0 and os.path.isfile(session_pickle_path):
        return None, None, True  # Skipped

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

    cwt_matrix = data.get("cwt", None)
    cwt_freqs = data.get("cwt_freqs", None)
    if cwt_freqs is not None:
        cwt_freqs = np.asarray(cwt_freqs).reshape(-1)

    for band in LFP_BAND_NAMES:
        wav_key = f"lfp_{band}"
        env_key = f"lfp_{band}_envelope"
        phase_key = f"lfp_{band}_phase"
        cwt_key = f"cwt_{band}"

        ingest_wav = f"lfp_{band}_20hz" if f"lfp_{band}_20hz" in data else f"lfp_{band}"
        ingest_env = f"lfp_{band}_hilbert_20hz" if f"lfp_{band}_hilbert_20hz" in data else f"lfp_{band}_envelope"
        ingest_phase = f"lfp_{band}_phase_20hz" if f"lfp_{band}_phase_20hz" in data else f"lfp_{band}_phase"

        features_dict[wav_key] = ensure_1d(data[ingest_wav]) if ingest_wav in data and data[ingest_wav] is not None else None
        features_dict[env_key] = ensure_1d(data[ingest_env]) if ingest_env in data and data[ingest_env] is not None else None

        if ingest_phase in data and data[ingest_phase] is not None:
            phi = ensure_1d(data[ingest_phase])
            features_dict[phase_key] = np.column_stack([np.cos(phi), np.sin(phi)])
        else:
            features_dict[phase_key] = None

        if cwt_matrix is not None and cwt_freqs is not None:
            f_low, f_high = LFP_BAND_LIMITS[band]
            idx_range = np.where((cwt_freqs >= f_low) & (cwt_freqs <= f_high))[0]
            if len(idx_range) == 0:
                nearest_idx = np.argmin(np.abs(cwt_freqs - 0.5 * (f_low + f_high)))
                idx_range = np.array([nearest_idx])

            mean_power = np.nanmean(cwt_matrix[:, idx_range], axis=1)
            db_power = 10.0 * np.log10(np.maximum(mean_power, 1e-12))
            features_dict[cwt_key] = ensure_1d(db_power)
        else:
            features_dict[cwt_key] = None

    for feat_name, arr in features_dict.items():
        if arr is not None and len(arr) != n_timepoints:
            raise ValueError(f"Row {row_idx}: Length mismatch for '{feat_name}' ({len(arr)} vs {n_timepoints}).")

    if VERBOSE:
        print(
            f"Row {row_idx} | dataset={row['dataset_name']} | mouse={mouse_name} | "
            f"session={session_name} | depth={depth_type} | "
            f"timepoints={n_timepoints} | neurons={n_neurons} | missing_neurons={len(missing_neurons)}"
        )

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

    recording_type = infer_recording_type(row)

    session_payload = {
        "metadata": {
            "dataset_name": row["dataset_name"],
            "mouse_name": mouse_name,
            "session_name": session_name,
            "depth_type": depth_type,
            "recording_type": recording_type,
            "row_index": row_idx,
            "standardized_file_path": standardized_file_path,
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

    neuron_rows = []
    for neuron_idx in missing_neurons:
        nrow = {
            "dataset_name": row["dataset_name"],
            "mouse_name": mouse_name,
            "session_name": session_name,
            "depth_type": depth_type,
            "recording_type": recording_type,
            "standardized_file_path": standardized_file_path,
            "row_index": row_idx,
            "neuron_index": neuron_idx,
            "n_timepoints": n_timepoints,
            "n_neurons_in_session": n_neurons,
        }
        for label in ALL_FEATURE_LABELS:
            nrow[f"mi_{label}"] = mi_dict[label][neuron_idx] if mi_dict[label] is not None else np.nan
        neuron_rows.append(nrow)

    session_row = {
        "dataset_name": row["dataset_name"],
        "mouse_name": mouse_name,
        "session_name": session_name,
        "depth_type": depth_type,
        "recording_type": recording_type,
        "standardized_file_path": standardized_file_path,
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
        print(
            f"    pos={session_row['mi_position_mean']:.4f} | "
            f"speed={session_row['mi_speed_mean']:.4f} | "
            f"theta_env={session_row.get('mi_lfp_theta_envelope_mean', np.nan):.4f} | "
            f"theta_phase={session_row.get('mi_lfp_theta_phase_mean', np.nan):.4f} | "
            f"cwt_theta={session_row.get('mi_cwt_theta_mean', np.nan):.4f}"
        )

    return neuron_rows, session_row, False


# =============================================================================
# Main Entry Point
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

    computed_neuron_set = load_computed_neuron_progress(NEURON_MI_CSV)
    print(f"Loaded {len(computed_neuron_set)} previously computed neuron records from disk.")

    if not os.path.isfile(NEURON_MI_CSV):
        init_csv(NEURON_MI_CSV, NEURON_MI_COLUMNS)
    if not os.path.isfile(SESSION_MI_CSV):
        init_csv(SESSION_MI_CSV, SESSION_MI_COLUMNS)

    log_path = os.path.join(LOG_DIR, "mutual_information_behavior_and_lfp_log.txt")
    log_file = open(log_path, "a", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Mutual Information Pipeline: Path Correction & Granular Verification")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Sessions in master index: {len(df)}")
    print(f"Features: {len(ALL_FEATURE_LABELS)} (7 Behavior, 6 Waveform, 6 Envelope, 6 Phase, 6 CWT Power)")
    print(f"Output directory: {OUTPUT_DIR}\n")

    ok_count = 0
    skip_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        local_time = timeit.default_timer()
        try:
            res = process_row(row_idx, row, computed_neuron_set)
            if res is None:
                continue

            neuron_rows, session_row, was_skipped = res

            if was_skipped:
                if VERBOSE:
                    print(
                        f"[SKIP] Row {row_idx} ({row.get('mouse_name')}, {row.get('session_name')}) already fully computed.\n")
                skip_count += 1
                continue

            for nr in neuron_rows:
                append_csv_row(NEURON_MI_CSV, NEURON_MI_COLUMNS, nr)
                computed_neuron_set.add((
                    int(nr["row_index"]),
                    str(nr["mouse_name"]).strip(),
                    str(nr["session_name"]).strip(),
                    str(nr["depth_type"]).strip(),
                    int(nr["neuron_index"])
                ))

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
    print(f"Successfully processed sessions: {ok_count}")
    print(f"Skipped sessions (fully done)  : {skip_count}")
    print(f"Failed sessions                : {fail_count}")
    print(f"Total elapsed time             : {total_elapsed:.2f} s")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()