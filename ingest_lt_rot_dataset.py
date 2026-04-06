#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingest lt_rot dataset into the standardized project structure.

This script:
1) reads source files from deepsup_project
2) extracts lt / rot sessions
3) computes standardized arrays using learning_repo.general_utils
4) crops all sessions of the same mouse to the same minimum length
5) saves one standardized pickle per session
6) appends rows to the shared master CSV
"""

import os
import csv
import copy
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.signal as scs

# ---------------------------------------------------------------------
# Compatibility patch for newer SciPy versions
# ---------------------------------------------------------------------
if not hasattr(scs, "gaussian"):
    from scipy.signal.windows import gaussian as _gaussian
    scs.gaussian = _gaussian

import learning_repo.general_utils as lrgu


# ============================================================================
# Top-level parameters
# ============================================================================

PROJECT_ROOT = "/home/melisamc/Documents/time_project_standarization"

DATASET_NAME = "lt_rot"
MODALITY = "calcium_imaging"

SOURCE_BASE_DIR = "/home/melisamc/Documents/deepsup_project"
SOURCE_DATA_DIR = os.path.join(SOURCE_BASE_DIR, "data")

COLORING = "single_color"
PARADIGMS = ["lt_and_rot", "only_lt"]

GENERAL_MICE_LIST = {
    "lt_and_rot": {
        "Calb":   ["CGrin1", "CZ3", "CZ6", "CZ8", "CZ9"],
        "Chrna7": ["ChZ4"],
        "Thy1":   ["GC2", "GC3", "GC5_nvista", "TGrin1"],
    },
    "only_lt": {
        "Calb":   ["CZ4"],
        "Chrna7": ["ChZ7", "ChZ8"],
        "Thy1":   ["GC7"],
    },
}

MODEL_TO_DEPTH = {
    "Calb": "superficial",
    "Chrna7": "deep",
    "Thy1": "deep",
}

SIGNAL_NAME = "clean_traces"

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


# ============================================================================
# IO utilities
# ============================================================================

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


# ============================================================================
# CSV utilities
# ============================================================================

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
    "contains_trial_id",
    "contains_inner_time",
    "source_container_type",
    "common_crop_length",
    "created_at",
    "notes",
]


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


# ============================================================================
# Session handling
# ============================================================================

def build_session_dataframe_like(session_obj):
    if isinstance(session_obj, pd.DataFrame):
        return session_obj.copy()

    if isinstance(session_obj, dict):
        try:
            return pd.DataFrame(session_obj)
        except Exception as exc:
            raise TypeError(f"Could not convert dict-like session to DataFrame: {exc}")

    raise TypeError(
        f"Unsupported session object type: {type(session_obj)}. "
        f"Expected pandas DataFrame or dict-like object."
    )


def extract_sessions_from_mouse_object(mouse_obj, mouse_name):
    """
    Return a dict: {session_name: session_dataframe}

    Handles:
    1) top-level dict with keys containing lt / rot
    2) one single dataframe-like session -> named mouse_lt
    """
    if isinstance(mouse_obj, dict):
        keys = list(mouse_obj.keys())

        session_keys = [
            k for k in keys
            if isinstance(k, str) and (("lt" in k.lower()) or ("rot" in k.lower()))
        ]
        if len(session_keys) > 0:
            return {k: build_session_dataframe_like(mouse_obj[k]) for k in sorted(session_keys)}

        try:
            df = pd.DataFrame(mouse_obj)
            return {f"{mouse_name}_lt": df}
        except Exception:
            pass

    if isinstance(mouse_obj, pd.DataFrame):
        return {f"{mouse_name}_lt": mouse_obj.copy()}

    raise TypeError(
        f"Could not extract sessions from object of type {type(mouse_obj)} "
        f"for mouse {mouse_name}"
    )


# ============================================================================
# Standardization
# ============================================================================

def standardize_one_session(df, dataset_name, modality, depth_type,
                            mouse_name, session_name, paradigm, model):
    """
    Build the standardized dictionary for one session using the original
    learning_repo utilities for trial-structured dataframes.
    """
    df = df.copy()

    rename_map = {"Fs": "sf", "pos": "position", "vel": "speed"}
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    if "mov_direction" not in df.columns:
        df = lrgu.add_mov_direction_mat_field(df)
    if "inner_trial_time" not in df.columns:
        df = lrgu.add_inner_trial_time_field(df)
    if "trial_id_mat" not in df.columns:
        df = lrgu.add_trial_id_mat_field(df)

    signal = np.asarray(lrgu.get_signal(df, "clean_traces"))
    pos = np.asarray(lrgu.get_signal(df, "position"))
    speed = np.asarray(lrgu.get_signal(df, "speed")).reshape(-1, 1)
    trial_id = np.asarray(lrgu.get_signal(df, "trial_id_mat")).reshape(-1, 1)
    mov_dir = np.asarray(lrgu.get_signal(df, "mov_direction")).reshape(-1, 1)
    inner_time = np.asarray(lrgu.get_signal(df, "inner_trial_time")).reshape(-1, 1)

    if pos.ndim == 1:
        pos = pos.reshape(-1, 1)

    # Clean mov_dir to avoid gaps, following your original logic
    temp_mov = np.diff(pos, axis=0) * 20
    if temp_mov.ndim == 1:
        temp_mov = temp_mov.reshape(-1, 1)
    temp_mov = np.concatenate([temp_mov[0, :].reshape(1, -1), temp_mov], axis=0)[:, 0]
    temp_mov = gaussian_filter1d(temp_mov, sigma=5, axis=0)
    temp_mov[temp_mov < 0] = -1
    temp_mov = np.sign(temp_mov)

    zero_idx = np.where(mov_dir.reshape(-1) == 0)[0]
    if len(zero_idx) > 0:
        mov_dir[zero_idx, 0] = temp_mov[zero_idx]

    n_timepoints = pos.shape[0]
    time = np.arange(n_timepoints).reshape(-1, 1)

    if signal.shape[0] != n_timepoints:
        raise ValueError(
            f"Length mismatch in {mouse_name} / {session_name}: "
            f"signal={signal.shape[0]}, pos={n_timepoints}"
        )

    standardized = {
        "metadata": {
            "dataset_name": dataset_name,
            "modality": modality,
            "depth_type": depth_type,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "paradigm": paradigm,
            "model": model,
            "signal_name": SIGNAL_NAME,
            "case": "same_length_across_mouse_sessions",
            "cells": "all_cells",
        },
        "signal": signal,
        "pos": pos,
        "mov_dir": mov_dir,
        "speed": speed,
        "trial_id": trial_id,
        "time": time,
        "inner_time": inner_time,
    }

    return standardized


def crop_standardized_dict(standardized, crop_length):
    cropped = copy.deepcopy(standardized)
    for key in ["signal", "pos", "mov_dir", "speed", "trial_id", "time", "inner_time"]:
        cropped[key] = cropped[key][:crop_length]
    cropped["metadata"]["common_crop_length"] = int(crop_length)
    return cropped


# ============================================================================
# Per-mouse processing
# ============================================================================

def process_mouse(paradigm, model, mouse_name):
    depth_type = MODEL_TO_DEPTH[model]

    source_mouse_dir = os.path.join(
        SOURCE_DATA_DIR,
        COLORING,
        paradigm,
        "processed_data",
        model,
        mouse_name,
    )
    source_file_name = f"{mouse_name}_df_dict.pkl"
    source_file_path = os.path.join(source_mouse_dir, source_file_name)

    if not os.path.isfile(source_file_path):
        print(f"[WARNING] Missing source file: {source_file_path}")
        return []

    print(f"\nProcessing mouse: {mouse_name} | model={model} | paradigm={paradigm}")

    mouse_obj = load_pickle_standard(source_file_path)
    session_dict = extract_sessions_from_mouse_object(mouse_obj, mouse_name)

    filtered_session_dict = {}
    for session_name, df in session_dict.items():
        s = session_name.lower()
        if ("lt" in s) or ("rot" in s):
            filtered_session_dict[session_name] = df

    if len(filtered_session_dict) == 0:
        print(f"  No lt/rot sessions found for {mouse_name}, skipping.")
        return []

    standardized_sessions = {}
    session_lengths = {}

    for session_name, df in filtered_session_dict.items():
        std = standardize_one_session(
            df=df,
            dataset_name=DATASET_NAME,
            modality=MODALITY,
            depth_type=depth_type,
            mouse_name=mouse_name,
            session_name=session_name,
            paradigm=paradigm,
            model=model,
        )
        standardized_sessions[session_name] = std
        session_lengths[session_name] = std["signal"].shape[0]

    common_crop_length = min(session_lengths.values())

    print(f"  Session lengths: {session_lengths}")
    print(f"  Common crop length: {common_crop_length}")

    mouse_output_dir = os.path.join(STANDARDIZED_DIR, depth_type, mouse_name)
    ensure_dir(mouse_output_dir)

    rows = []

    for session_name, standardized in standardized_sessions.items():
        session_name_safe = sanitize_for_filename(session_name)
        standardized_cropped = crop_standardized_dict(standardized, common_crop_length)

        standardized_file_name = (
            f"{DATASET_NAME}_{mouse_name}_{session_name_safe}_"
            f"{MODALITY}_{depth_type}_{paradigm}_{model}_{SIGNAL_NAME}.pkl"
        )
        standardized_file_path = os.path.join(mouse_output_dir, standardized_file_name)

        if os.path.exists(standardized_file_path) and not OVERWRITE_STANDARDIZED_FILES:
            print(f"  File exists, skipping save: {standardized_file_name}")
        else:
            save_pickle_standard(standardized_file_path, standardized_cropped)
            print(f"  Saved: {standardized_file_name}")

        signal = standardized_cropped["signal"]
        pos = standardized_cropped["pos"]
        trial_id = standardized_cropped["trial_id"]
        inner_time = standardized_cropped["inner_time"]

        row = {
            "dataset_name": DATASET_NAME,
            "modality": MODALITY,
            "depth_type": depth_type,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "session_name_safe": session_name_safe,
            "source_base_dir": SOURCE_BASE_DIR,
            "source_data_dir": SOURCE_DATA_DIR,
            "source_mouse_dir": source_mouse_dir,
            "source_file_name": source_file_name,
            "source_file_path": source_file_path,
            "case": "same_length_across_mouse_sessions",
            "cells": "all_cells",
            "signal_name": SIGNAL_NAME,
            "paradigm": paradigm,
            "model": model,
            "standardized_file_name": standardized_file_name,
            "standardized_file_path": standardized_file_path,
            "n_timepoints": int(signal.shape[0]),
            "n_neurons": int(signal.shape[1]) if signal.ndim > 1 else 1,
            "pos_dim": int(pos.shape[1]) if pos.ndim > 1 else 1,
            "contains_trial_id": trial_id is not None,
            "contains_inner_time": inner_time is not None,
            "source_container_type": "mouse_file_with_lt_rot_sessions",
            "common_crop_length": int(common_crop_length),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "",
        }
        rows.append(row)

    return rows


# ============================================================================
# Main
# ============================================================================

def main():
    for path in [CODE_DIR, DATA_DIR, ANALYSIS_DIR, METADATA_DIR, STANDARDIZED_DIR]:
        ensure_dir(path)

    initialize_master_csv_if_needed(MASTER_CSV_PATH, MASTER_COLUMNS)

    total_rows = 0
    all_rows = []

    for paradigm in PARADIGMS:
        mice_by_model = GENERAL_MICE_LIST[paradigm]
        for model, mice_list in mice_by_model.items():
            for mouse_name in mice_list:
                rows = process_mouse(paradigm, model, mouse_name)
                all_rows.extend(rows)

    for row in all_rows:
        append_row_to_csv(MASTER_CSV_PATH, row, MASTER_COLUMNS)
        total_rows += 1

    print("\nDone.")
    print(f"Appended rows: {total_rows}")
    print(f"Master CSV: {MASTER_CSV_PATH}")


if __name__ == "__main__":
    main()