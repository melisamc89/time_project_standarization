#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import copy
from datetime import datetime

import numpy as np
import learning_repo.general_utils as lrgu


# ============================================================================
# Top-level parameters
# ============================================================================

PROJECT_ROOT = "/home/melisamc/Documents/time_project_standarization"

DATASET_NAME = "learning"
MODALITY = "calcium_imaging"
DEPTH_TYPE = "whole_population"

SOURCE_BASE_DIR = "/home/melisamc/Documents/learning_project"
SOURCE_DATA_DIR = os.path.join(SOURCE_BASE_DIR, "processed_data")

MOUSE_NAMES = ["M2019", "M2023", "M2024", "M2025", "M2026"]

CASE = "mov_same_length"
CELLS = "all_cells"
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
    DEPTH_TYPE,
)

MASTER_CSV_PATH = os.path.join(METADATA_DIR, "master_dataset_index.csv")


# ============================================================================
# Unified CSV schema
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


# ============================================================================
# Helpers
# ============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def yes_no(x):
    return "yes" if x else "no"


def initialize_master_csv(csv_path, columns):
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_row_to_csv(csv_path, row, columns):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


# ============================================================================
# Standardization
# ============================================================================

def build_standardized_session_dict(mouse_dict, mouse_name, session_name):
    session_data = mouse_dict[session_name]

    signal = np.asarray(lrgu.get_signal(session_data, SIGNAL_NAME))
    if CELLS == "same_cells":
        signal = signal[:, session_data["global_id"][0]]

    pos = np.asarray(lrgu.get_signal(session_data, "position"))
    mov_dir = np.asarray(lrgu.get_signal(session_data, "mov_direction")).reshape(-1, 1)
    speed = np.asarray(lrgu.get_signal(session_data, "speed")).reshape(-1, 1)
    trial_id = np.asarray(lrgu.get_signal(session_data, "trial_id_mat")).reshape(-1, 1)
    inner_time = np.asarray(lrgu.get_signal(session_data, "inner_trial_time")).reshape(-1, 1)

    if pos.ndim == 1:
        pos = pos.reshape(-1, 1)

    n_timepoints = signal.shape[0]
    n_neurons = signal.shape[1]
    time = np.arange(n_timepoints).reshape(-1, 1)

    if pos.shape[0] != n_timepoints:
        raise ValueError(
            f"Length mismatch in {mouse_name} / {session_name}: "
            f"signal.shape[0]={signal.shape[0]}, pos.shape[0]={pos.shape[0]}"
        )

    standardized = {
        "metadata": {
            "dataset_name": DATASET_NAME,
            "modality": MODALITY,
            "depth_type": DEPTH_TYPE,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "case": CASE,
            "cells": CELLS,
            "signal_name": SIGNAL_NAME,
            "paradigm": "",
            "model": "",
            "common_crop_length": "",
        },
        "signal": signal,
        "pos": pos,
        "mov_dir": mov_dir,
        "speed": speed,
        "trial_id": trial_id,
        "time": time,
        "inner_time": inner_time,
    }

    return standardized, n_timepoints, n_neurons


def process_mouse(mouse_name):
    source_mouse_dir = os.path.join(SOURCE_DATA_DIR, mouse_name)
    source_file_name = f"{mouse_name}_{CASE}_df_dict.pkl"
    source_file_path = os.path.join(source_mouse_dir, source_file_name)

    if not os.path.isfile(source_file_path):
        print(f"[WARNING] Source file not found: {source_file_path}")
        return []

    print(f"\nProcessing mouse: {mouse_name}")

    mouse_dict = lrgu.load_pickle(source_mouse_dir, source_file_name)
    session_names = sorted(list(mouse_dict.keys()))

    mouse_output_dir = os.path.join(STANDARDIZED_DIR, mouse_name)
    ensure_dir(mouse_output_dir)

    rows = []

    for idx, session_name in enumerate(session_names, start=1):
        print(f"  [{idx}/{len(session_names)}] Session: {session_name}")

        session_name_safe = sanitize_for_filename(session_name)
        standardized_file_name = (
            f"{DATASET_NAME}_{mouse_name}_{session_name_safe}_"
            f"{MODALITY}_{DEPTH_TYPE}_{CASE}_{CELLS}_{SIGNAL_NAME}.pkl"
        )
        standardized_file_path = os.path.join(mouse_output_dir, standardized_file_name)

        if os.path.exists(standardized_file_path) and not OVERWRITE_STANDARDIZED_FILES:
            standardized = lrgu.load_pickle(mouse_output_dir, standardized_file_name)
            print(f"    File exists, skipping save: {standardized_file_name}")
            signal = standardized["signal"]
            pos = standardized["pos"]
            mov_dir = standardized["mov_dir"]
            speed = standardized["speed"]
            trial_id = standardized["trial_id"]
            time = standardized["time"]
            inner_time = standardized["inner_time"]
            n_timepoints = signal.shape[0]
            n_neurons = signal.shape[1]
        else:
            standardized, n_timepoints, n_neurons = build_standardized_session_dict(
                mouse_dict, mouse_name, session_name
            )
            lrgu.save_pickle(mouse_output_dir, standardized_file_name, standardized)
            print(f"    Saved: {standardized_file_name}")

            signal = standardized["signal"]
            pos = standardized["pos"]
            mov_dir = standardized["mov_dir"]
            speed = standardized["speed"]
            trial_id = standardized["trial_id"]
            time = standardized["time"]
            inner_time = standardized["inner_time"]

        row = {
            "dataset_name": DATASET_NAME,
            "modality": MODALITY,
            "depth_type": DEPTH_TYPE,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "session_name_safe": session_name_safe,
            "source_base_dir": SOURCE_BASE_DIR,
            "source_data_dir": SOURCE_DATA_DIR,
            "source_mouse_dir": source_mouse_dir,
            "source_file_name": source_file_name,
            "source_file_path": source_file_path,
            "case": CASE,
            "cells": CELLS,
            "signal_name": SIGNAL_NAME,
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
            "source_container_type": "mouse_file_with_multiple_session_keys",
            "common_crop_length": "",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "",
        }
        rows.append(row)

    return rows


def main():
    for path in [CODE_DIR, DATA_DIR, ANALYSIS_DIR, METADATA_DIR, STANDARDIZED_DIR]:
        ensure_dir(path)

    initialize_master_csv(MASTER_CSV_PATH, MASTER_COLUMNS)

    total_rows = 0
    for mouse_name in MOUSE_NAMES:
        rows = process_mouse(mouse_name)
        for row in rows:
            append_row_to_csv(MASTER_CSV_PATH, row, MASTER_COLUMNS)
            total_rows += 1

    print("\nDone.")
    print(f"Rows written: {total_rows}")
    print(f"Master CSV: {MASTER_CSV_PATH}")


if __name__ == "__main__":
    main()