#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pickle
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# Top-level parameters
# =============================================================================

PROJECT_ROOT = "/home/melisamc/Documents/time_project_standarization"

DATASET_NAME = "lt_egrin"
MODALITY = "calcium_imaging"

# -------------------------------------------------------------------------
# EDIT THESE TO MATCH THE REAL SOURCE FILE
# -------------------------------------------------------------------------
SOURCE_BASE_DIR = "/home/melisamc/Documents/lt_egrin_project"
SOURCE_DATA_FILE = os.path.join(SOURCE_BASE_DIR, "data", "lt_egrin_data.pkl")
# -------------------------------------------------------------------------

SPEED_TH = 3
SIG_FILT = 6
FS = 20.0  # already downsampled

OVERWRITE_STANDARDIZED_FILES = False

CODE_DIR = os.path.join(PROJECT_ROOT, "code")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")

METADATA_DIR = os.path.join(DATA_DIR, "metadata")
STANDARDIZED_DIR = os.path.join(
    DATA_DIR,
    "standardized_sessions",
    DATASET_NAME,
    MODALITY,
)

MASTER_CSV_PATH = os.path.join(METADATA_DIR, "master_dataset_index.csv")


# =============================================================================
# Unified CSV schema
# =============================================================================

MASTER_COLUMNS = [
    "dataset_name",
    "modality",
    "depth_type",
    "mouse_name",
    "session_name",
    "session_name_safe",
    "source_base_dir",
    "source_data_dir",
    "source_mouse_dir",
    "source_file_name",
    "source_file_path",
    "case",
    "cells",
    "signal_name",
    "paradigm",
    "model",
    "standardized_file_name",
    "standardized_file_path",
    "n_timepoints",
    "n_neurons",
    "pos_dim",
    "contains_signal",
    "contains_pos",
    "contains_mov_dir",
    "contains_speed",
    "contains_trial_id",
    "contains_time",
    "contains_inner_time",
    "source_container_type",
    "common_crop_length",
    "created_at",
    "notes",
]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def initialize_master_csv_if_needed(csv_path, columns):
    ensure_dir(os.path.dirname(csv_path))
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_row_to_csv(csv_path, row, columns):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def save_pickle_standard(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_standard(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def sanitize_for_filename(text):
    text = str(text)
    out = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def yes_no(x):
    return "yes" if x else "no"


def ensure_signal_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def ensure_pos_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def ensure_col(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def compute_inner_time(trial_id):
    """
    Reconstruct inner-trial time from trial_id on the already filtered data.
    """
    trial_id = np.asarray(trial_id).reshape(-1)
    inner_time = np.zeros_like(trial_id, dtype=float)

    if len(trial_id) == 0:
        return inner_time.reshape(-1, 1)

    current_trial = trial_id[0]
    start_idx = 0

    for i in range(1, len(trial_id)):
        if trial_id[i] != current_trial:
            current_trial = trial_id[i]
            start_idx = i
        inner_time[i] = (i - start_idx) / FS

    return inner_time.reshape(-1, 1)


def infer_depth_from_mouse_name(mouse_name):
    low = mouse_name.lower()
    if "calb" in low:
        return "superficial"
    if "thy" in low:
        return "deep"
    return "unknown"


# =============================================================================
# Main standardization
# =============================================================================

def build_standardized_mouse_output(mouse_name, mouse_data):
    depth_type = infer_depth_from_mouse_name(mouse_name)

    signal = ensure_signal_2d(mouse_data["clean_traces"])
    speed = np.asarray(mouse_data["speed"]).reshape(-1)
    pos = ensure_pos_2d(mouse_data["position"])
    trial_id = np.asarray(mouse_data["trial_id"]).reshape(-1)
    mov_dir = np.asarray(mouse_data["mov_direction"]).reshape(-1)
    lfp = ensure_signal_2d(mouse_data["lfp"])

    if signal.shape[0] != len(speed):
        raise ValueError(
            f"{mouse_name}: signal and speed length mismatch: "
            f"{signal.shape[0]} vs {len(speed)}"
        )

    if pos.shape[0] != signal.shape[0]:
        raise ValueError(
            f"{mouse_name}: position and signal length mismatch: "
            f"{pos.shape[0]} vs {signal.shape[0]}"
        )

    if len(trial_id) != signal.shape[0]:
        raise ValueError(
            f"{mouse_name}: trial_id and signal length mismatch: "
            f"{len(trial_id)} vs {signal.shape[0]}"
        )

    if len(mov_dir) != signal.shape[0]:
        raise ValueError(
            f"{mouse_name}: mov_dir and signal length mismatch: "
            f"{len(mov_dir)} vs {signal.shape[0]}"
        )

    if lfp.shape[0] != signal.shape[0]:
        raise ValueError(
            f"{mouse_name}: lfp and signal length mismatch: "
            f"{lfp.shape[0]} vs {signal.shape[0]}"
        )

    # -------------------------------------------------------------------------
    # Select valid moving frames
    # -------------------------------------------------------------------------
    valid_index = np.where(speed > SPEED_TH)[0]

    # -------------------------------------------------------------------------
    # Apply valid_index first, then smooth signal with gaussian sigma=6
    # -------------------------------------------------------------------------
    signal = signal[valid_index, :]
    signal = gaussian_filter1d(signal, sigma=SIG_FILT, axis=0)

    speed = speed[valid_index]
    pos = pos[valid_index, :]
    trial_id = trial_id[valid_index]
    mov_dir = mov_dir[valid_index]
    lfp_20hz = lfp[valid_index, :]

    inner_time = compute_inner_time(trial_id)
    time = np.arange(pos.shape[0]).reshape(-1, 1)

    standardized = {
        "metadata": {
            "dataset_name": DATASET_NAME,
            "modality": MODALITY,
            "depth_type": depth_type,
            "mouse_name": mouse_name,
            "session_name": "lt_egrin",
            "case": f"speed_threshold_{SPEED_TH}_gaussian_sigma{SIG_FILT}",
            "cells": "all_cells",
            "signal_name": f"clean_traces_gaussian_sigma{SIG_FILT}",
            "paradigm": "",
            "model": "",
            "common_crop_length": "",
            "fs": FS,
        },
        "signal": signal,
        "pos": pos,
        "mov_dir": ensure_col(mov_dir),
        "speed": ensure_col(speed),
        "trial_id": ensure_col(trial_id),
        "time": time,
        "inner_time": inner_time,
        "lfp_20hz": lfp_20hz,
        "valid_index": valid_index,
        "cwt_freqs": mouse_data.get("cwt_freqs", None),
        "cwt": mouse_data.get("cwt", None),
    }

    return standardized, depth_type


def row_from_standardized(
    mouse_name,
    depth_type,
    standardized,
    source_file_name,
    source_file_path,
    standardized_file_name,
    standardized_file_path,
):
    signal = standardized["signal"]
    pos = standardized["pos"]
    mov_dir = standardized["mov_dir"]
    speed = standardized["speed"]
    trial_id = standardized["trial_id"]
    time = standardized["time"]
    inner_time = standardized["inner_time"]

    return {
        "dataset_name": DATASET_NAME,
        "modality": MODALITY,
        "depth_type": depth_type,
        "mouse_name": mouse_name,
        "session_name": "lt_egrin",
        "session_name_safe": "lt_egrin",
        "source_base_dir": SOURCE_BASE_DIR,
        "source_data_dir": os.path.dirname(SOURCE_DATA_FILE),
        "source_mouse_dir": os.path.dirname(SOURCE_DATA_FILE),
        "source_file_name": source_file_name,
        "source_file_path": source_file_path,
        "case": standardized["metadata"]["case"],
        "cells": standardized["metadata"]["cells"],
        "signal_name": standardized["metadata"]["signal_name"],
        "paradigm": "",
        "model": "",
        "standardized_file_name": standardized_file_name,
        "standardized_file_path": standardized_file_path,
        "n_timepoints": int(signal.shape[0]),
        "n_neurons": int(signal.shape[1]),
        "pos_dim": int(pos.shape[1]) if pos.ndim > 1 else 1,
        "contains_signal": yes_no(signal is not None),
        "contains_pos": yes_no(pos is not None),
        "contains_mov_dir": yes_no(mov_dir is not None),
        "contains_speed": yes_no(speed is not None),
        "contains_trial_id": yes_no(trial_id is not None),
        "contains_time": yes_no(time is not None),
        "contains_inner_time": yes_no(inner_time is not None),
        "source_container_type": "single_mouse_dict_from_global_pickle",
        "common_crop_length": "",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "includes lfp_20hz (16 channels), valid_index, optional cwt and cwt_freqs",
    }


# =============================================================================
# Main
# =============================================================================

def main():
    for path in [CODE_DIR, DATA_DIR, ANALYSIS_DIR, METADATA_DIR, STANDARDIZED_DIR]:
        ensure_dir(path)

    initialize_master_csv_if_needed(MASTER_CSV_PATH, MASTER_COLUMNS)

    if not os.path.isfile(SOURCE_DATA_FILE):
        raise FileNotFoundError(f"Source file not found: {SOURCE_DATA_FILE}")

    data = load_pickle_standard(SOURCE_DATA_FILE)
    if not isinstance(data, dict):
        raise TypeError(f"Expected source data to be a dict, got {type(data)}")

    source_file_name = os.path.basename(SOURCE_DATA_FILE)
    source_file_path = SOURCE_DATA_FILE

    total_rows = 0

    for mouse_name in sorted(data.keys()):
        print(f"\nProcessing mouse: {mouse_name}")

        mouse_data = data[mouse_name]

        standardized, depth_type = build_standardized_mouse_output(
            mouse_name=mouse_name,
            mouse_data=mouse_data,
        )

        mouse_output_dir = os.path.join(STANDARDIZED_DIR, depth_type, mouse_name)
        ensure_dir(mouse_output_dir)

        standardized_file_name = (
            f"{DATASET_NAME}_{mouse_name}_lt_egrin_{MODALITY}_{depth_type}_"
            f"{standardized['metadata']['signal_name']}.pkl"
        )
        standardized_file_name = sanitize_for_filename(standardized_file_name)
        standardized_file_path = os.path.join(mouse_output_dir, standardized_file_name)

        if os.path.exists(standardized_file_path) and not OVERWRITE_STANDARDIZED_FILES:
            print(f"File exists, skipping save: {standardized_file_name}")
        else:
            save_pickle_standard(standardized_file_path, standardized)
            print(f"Saved: {standardized_file_name}")

        row = row_from_standardized(
            mouse_name=mouse_name,
            depth_type=depth_type,
            standardized=standardized,
            source_file_name=source_file_name,
            source_file_path=source_file_path,
            standardized_file_name=standardized_file_name,
            standardized_file_path=standardized_file_path,
        )
        append_row_to_csv(MASTER_CSV_PATH, row, MASTER_COLUMNS)
        total_rows += 1

    print("\nDone.")
    print(f"Appended rows: {total_rows}")
    print(f"Master CSV: {MASTER_CSV_PATH}")


if __name__ == "__main__":
    main()