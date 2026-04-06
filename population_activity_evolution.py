#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Population activity evolution over time.

For each standardized session:
1) Load signal with shape (n_timepoints, n_neurons)
2) Compute mean activity across neurons at each timepoint
3) Split the population-mean trace into first half and last half
4) Compute mean and std for each half
5) Save one row per session to CSV
"""

import os
import csv
import pickle
import timeit
from datetime import datetime

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

CSV_PATH = "/home/melisamc/Documents/time_project_standarization/data/metadata/master_dataset_index.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/population_activity_evolution"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "population_activity_first_vs_last_half.csv")


# =============================================================================
# Parameters
# =============================================================================

MAX_ROWS = None
VERBOSE = True


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_pickle_standard(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def infer_recording_type(row):
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
# CSV schema
# =============================================================================

OUTPUT_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "recording_type",
    "standardized_file_path",
    "row_index",
    "n_timepoints",
    "n_neurons",
    "first_half_mean",
    "last_half_mean",
    "first_half_std",
    "last_half_std",
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
# Main processing
# =============================================================================

def process_row(row_idx, row):
    standardized_file_path = row["standardized_file_path"]

    if not os.path.isfile(standardized_file_path):
        raise FileNotFoundError(f"Missing standardized file: {standardized_file_path}")

    data = load_pickle_standard(standardized_file_path)

    if "signal" not in data:
        raise KeyError(f"Missing key 'signal' in file: {standardized_file_path}")

    signal = ensure_2d(data["signal"])

    n_timepoints, n_neurons = signal.shape

    if n_timepoints < 2:
        raise ValueError(f"Row {row_idx}: signal has fewer than 2 timepoints")
    if n_neurons < 1:
        raise ValueError(f"Row {row_idx}: signal has no neurons")

    # mean over neurons at each time point
    population_mean_trace = np.nanmean(signal, axis=1)

    # split into first and last half
    split_idx = n_timepoints // 2
    first_half = population_mean_trace[:split_idx]
    last_half = population_mean_trace[split_idx:]

    if len(first_half) < 1 or len(last_half) < 1:
        raise ValueError(f"Row {row_idx}: invalid split into halves")

    out_row = {
        "dataset_name": row["dataset_name"],
        "mouse_name": row["mouse_name"],
        "session_name": row["session_name"],
        "depth_type": row["depth_type"],
        "recording_type": infer_recording_type(row),
        "standardized_file_path": standardized_file_path,
        "row_index": row_idx,
        "n_timepoints": n_timepoints,
        "n_neurons": n_neurons,
        "first_half_mean": float(np.nanmean(first_half)),
        "last_half_mean": float(np.nanmean(last_half)),
        "first_half_std": float(np.nanstd(first_half)),
        "last_half_std": float(np.nanstd(last_half)),
    }

    if VERBOSE:
        print(
            f"Row {row_idx} | dataset={row['dataset_name']} | mouse={row['mouse_name']} | "
            f"session={row['session_name']} | depth={row['depth_type']} | "
            f"first_mean={out_row['first_half_mean']:.6f} | "
            f"last_mean={out_row['last_half_mean']:.6f} | "
            f"first_std={out_row['first_half_std']:.6f} | "
            f"last_std={out_row['last_half_std']:.6f}"
        )

    return out_row


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(LOG_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    init_csv(OUTPUT_CSV, OUTPUT_COLUMNS)

    log_path = os.path.join(LOG_DIR, "population_activity_evolution_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Population activity evolution analysis")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Rows to process: {len(df)}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("")

    ok_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        local_time = timeit.default_timer()
        try:
            out_row = process_row(row_idx, row)
            append_csv_row(OUTPUT_CSV, OUTPUT_COLUMNS, out_row)
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