#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import copy
import pickle
from datetime import datetime

import numpy as np
import pandas as pd


# =============================================================================
# Top-level parameters
# =============================================================================

PROJECT_ROOT = "/home/melisamc/Documents/time_project_standarization"

DATASET_NAME = "dreadds"
MODALITY = "calcium_imaging"

SOURCE_DATASETS = [
    {
        "group": "superficial",
        "name": "Calb_sup",
        "root": "/home/melisamc/Documents/DREADDs_all/Calb",
        "processed_dirname": "processed_data",
        "mice_list": ['CalbCharly2', 'CalbCharly11_concat', 'CalbV23', 'DD2', 'CalbCharly10', 'CalbCharly3'],
    },
    {
        "group": "deep",
        "name": "ChRNA7_deep",
        "root": "/home/melisamc/Documents/DREADDs_all/ChRNA7",
        "processed_dirname": "processed_data",
        "mice_list": ['ChRNA7Charly1', 'ChRNA7Charly2', 'ChRNA7Dreadd1', 'ChRNA7Charly4'],
    },
]

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


def load_pickle_standard(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle_standard(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


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


def get_signal(pd_struct: pd.DataFrame, field_name: str) -> np.ndarray:
    return copy.deepcopy(np.concatenate(pd_struct[field_name].values, axis=0))


def ensure_trial_id_mat_field(df: pd.DataFrame) -> pd.DataFrame:
    if "trial_id_mat" in df.columns:
        return df

    if "trial_id" not in df.columns:
        raise KeyError("Missing 'trial_id' and cannot build 'trial_id_mat'.")

    out = df.copy()
    out["trial_id_mat"] = [
        np.ones((out["position"].iloc[i].shape[0], 1), dtype=int) * int(out["trial_id"].iloc[i])
        for i in range(out.shape[0])
    ]
    return out


def ensure_inner_trial_time_field(df: pd.DataFrame) -> pd.DataFrame:
    if "inner_trial_time" in df.columns:
        return df

    out = df.copy()
    out["inner_trial_time"] = [
        np.arange(out["position"].iloc[i].shape[0]).reshape(-1, 1).astype(int)
        for i in range(out.shape[0])
    ]
    return out


def get_direction_vector(df: pd.DataFrame) -> np.ndarray:
    if "mov_direction" in df.columns:
        return get_signal(df, "mov_direction").reshape(-1)

    if "direction" in df.columns:
        return get_signal(df, "direction").reshape(-1)

    if "dir" in df.columns:
        d = get_signal(df, "dir").reshape(-1)
        if np.issubdtype(np.asarray(d).dtype, np.number):
            return np.asarray(d).reshape(-1)

        mapping = {"L": -1, "R": 1, "N": 0, "l": -1, "r": 1, "n": 0}
        return np.array([mapping.get(str(x), 0) for x in d], dtype=float)

    raise KeyError("No direction field found: expected 'mov_direction' or 'direction' or 'dir'.")


def split_mouse_condition(folder_name):
    if "_" not in folder_name:
        return folder_name, "NA"
    base, cond = folder_name.rsplit("_", 1)
    return base, cond


def normalize_condition(cond):
    c = str(cond).strip().lower()
    if c == "cno":
        return "cno"
    if c in ["veh", "vehicle"]:
        return "veh"
    return c


def find_pre_rot_keys(animal_dict):
    fnames = list(animal_dict.keys())
    fname_pre = [f for f in fnames if isinstance(f, str) and "lt" in f.lower()][0]
    fname_rot = [f for f in fnames if isinstance(f, str) and "rot" in f.lower()][0]
    return fname_pre, fname_rot


def session_label_from_key_and_condition(session_key, cond):
    key = str(session_key).lower()
    cond = normalize_condition(cond)

    if "rot" in key:
        sess = "rot"
    else:
        sess = "lt"

    return f"{sess}_{cond}"


# =============================================================================
# Core extraction
# =============================================================================

def build_standardized_session(df_session, session_name, mouse_name, depth_type, source_info):
    df_session = ensure_trial_id_mat_field(df_session)
    df_session = ensure_inner_trial_time_field(df_session)

    signal = ensure_2d(get_signal(df_session, SIGNAL_NAME))
    pos = ensure_2d(get_signal(df_session, "position"))
    mov_dir = ensure_2d(get_direction_vector(df_session))
    speed_field = "speed" if "speed" in df_session.columns else "vel"
    speed = ensure_2d(get_signal(df_session, speed_field))
    trial_id = ensure_2d(get_signal(df_session, "trial_id_mat"))
    time = np.arange(signal.shape[0]).reshape(-1, 1)
    inner_time = ensure_2d(get_signal(df_session, "inner_trial_time"))

    n_timepoints = signal.shape[0]
    n_neurons = signal.shape[1]

    if pos.shape[0] != n_timepoints:
        raise ValueError(f"{mouse_name} {session_name}: pos length mismatch")
    if mov_dir.shape[0] != n_timepoints:
        raise ValueError(f"{mouse_name} {session_name}: mov_dir length mismatch")
    if speed.shape[0] != n_timepoints:
        raise ValueError(f"{mouse_name} {session_name}: speed length mismatch")
    if trial_id.shape[0] != n_timepoints:
        raise ValueError(f"{mouse_name} {session_name}: trial_id length mismatch")
    if inner_time.shape[0] != n_timepoints:
        raise ValueError(f"{mouse_name} {session_name}: inner_time length mismatch")

    standardized = {
        "metadata": {
            "dataset_name": DATASET_NAME,
            "modality": MODALITY,
            "depth_type": depth_type,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "case": source_info["condition"],
            "cells": "all_cells",
            "signal_name": SIGNAL_NAME,
            "paradigm": "",
            "model": source_info["group_name"],
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

    return standardized


def row_from_standardized(
    standardized,
    source_info,
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
    meta = standardized["metadata"]

    return {
        "dataset_name": DATASET_NAME,
        "modality": MODALITY,
        "depth_type": meta["depth_type"],
        "mouse_name": meta["mouse_name"],
        "session_name": meta["session_name"],
        "session_name_safe": sanitize_for_filename(meta["session_name"]),
        "source_base_dir": source_info["source_base_dir"],
        "source_data_dir": source_info["source_data_dir"],
        "source_mouse_dir": source_info["source_mouse_dir"],
        "source_file_name": source_info["source_file_name"],
        "source_file_path": source_info["source_file_path"],
        "case": meta["case"],
        "cells": meta["cells"],
        "signal_name": meta["signal_name"],
        "paradigm": meta["paradigm"],
        "model": meta["model"],
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
        "source_container_type": "condition_folder_with_df_dict_and_two_sessions",
        "common_crop_length": "",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "DREADDs dataset; sessions named lt_veh, lt_cno, rot_veh, rot_cno depending on folder condition and session key",
    }


# =============================================================================
# Main
# =============================================================================

def main():
    for path in [CODE_DIR, DATA_DIR, ANALYSIS_DIR, METADATA_DIR, STANDARDIZED_DIR]:
        ensure_dir(path)

    initialize_master_csv_if_needed(MASTER_CSV_PATH, MASTER_COLUMNS)

    total_rows = 0

    for ds in SOURCE_DATASETS:
        group_name = ds["name"]
        depth_type = ds["group"]
        source_base_dir = ds["root"]
        source_data_dir = os.path.join(source_base_dir, ds["processed_dirname"])

        if not os.path.isdir(source_data_dir):
            print(f"[WARNING] Missing processed dir: {source_data_dir}")
            continue

        all_folders = [
            d for d in os.listdir(source_data_dir)
            if os.path.isdir(os.path.join(source_data_dir, d))
        ]

        for mouse_name in ds["mice_list"]:
            mouse_folders = [d for d in all_folders if d.startswith(mouse_name + "_")]

            if len(mouse_folders) == 0:
                print(f"[WARNING] No folders found for mouse: {mouse_name}")
                continue

            for folder_name in sorted(mouse_folders):
                base_mouse, cond = split_mouse_condition(folder_name)
                if base_mouse != mouse_name:
                    continue

                source_mouse_dir = os.path.join(source_data_dir, folder_name)
                source_file_name = f"{folder_name}_df_dict.pkl"
                source_file_path = os.path.join(source_mouse_dir, source_file_name)

                if not os.path.isfile(source_file_path):
                    print(f"[WARNING] Missing pickle: {source_file_path}")
                    continue

                print(f"\nProcessing {mouse_name} | folder={folder_name} | depth={depth_type}")

                df_dict = load_pickle_standard(source_file_path)
                fname_pre, fname_rot = find_pre_rot_keys(df_dict)

                session_map = {
                    session_label_from_key_and_condition(fname_pre, cond): df_dict[fname_pre],
                    session_label_from_key_and_condition(fname_rot, cond): df_dict[fname_rot],
                }

                for session_name, df_session in session_map.items():
                    source_info = {
                        "group_name": group_name,
                        "condition": normalize_condition(cond),
                        "source_base_dir": source_base_dir,
                        "source_data_dir": source_data_dir,
                        "source_mouse_dir": source_mouse_dir,
                        "source_file_name": source_file_name,
                        "source_file_path": source_file_path,
                    }

                    standardized = build_standardized_session(
                        df_session=df_session,
                        session_name=session_name,
                        mouse_name=mouse_name,
                        depth_type=depth_type,
                        source_info=source_info,
                    )

                    mouse_output_dir = os.path.join(STANDARDIZED_DIR, depth_type, mouse_name)
                    ensure_dir(mouse_output_dir)

                    standardized_file_name = sanitize_for_filename(
                        f"{DATASET_NAME}_{mouse_name}_{session_name}_{MODALITY}_{depth_type}_{SIGNAL_NAME}.pkl"
                    )
                    standardized_file_path = os.path.join(mouse_output_dir, standardized_file_name)

                    if os.path.exists(standardized_file_path) and not OVERWRITE_STANDARDIZED_FILES:
                        print(f"File exists, skipping save: {standardized_file_name}")
                    else:
                        save_pickle_standard(standardized_file_path, standardized)
                        print(f"Saved: {standardized_file_name}")

                    row = row_from_standardized(
                        standardized=standardized,
                        source_info=source_info,
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