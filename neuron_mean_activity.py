#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-neuron mean activity / firing-rate-equivalent analysis.

What this script does
---------------------
1) Reads the master standardized dataset index CSV.
2) Loads each standardized session pickle.
3) For each neuron, computes across time:
       - mean activity
       - std activity
4) Saves:
       - one CSV with one row per neuron across all datasets/sessions
       - one CSV with one row per session summarizing mean/std across neurons
       - one pickle per session with full per-neuron arrays
       - one log txt file

Interpretation
--------------
- For electrophysiology datasets, this is the mean/std of the standardized neural signal
  (typically firing-rate-like if your standardization converted spikes to rates).
- For calcium datasets, this is the mean/std of the standardized calcium signal,
  i.e. a firing-rate-equivalent measure.

Assumptions
-----------
- signal has shape: (n_timepoints, n_neurons)
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

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/neuron_mean_activity"
SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

NEURON_CSV = os.path.join(OUTPUT_DIR, "neuron_mean_activity_per_neuron.csv")
SESSION_CSV = os.path.join(OUTPUT_DIR, "neuron_mean_activity_per_session_summary.csv")


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


def save_pickle_standard(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


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
# CSV schemas
# =============================================================================

NEURON_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "recording_type",
    "standardized_file_path",
    "row_index",
    "neuron_index",
    "neuron_uid",
    "n_timepoints",
    "n_neurons_in_session",
    "mean_activity",
    "std_activity",
]

SESSION_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "recording_type",
    "standardized_file_path",
    "row_index",
    "n_timepoints",
    "n_neurons",
    "mean_activity_across_neurons_mean",
    "mean_activity_across_neurons_std",
    "std_activity_across_neurons_mean",
    "std_activity_across_neurons_std",
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

    if n_timepoints < 1:
        raise ValueError(f"Row {row_idx}: signal has no timepoints")
    if n_neurons < 1:
        raise ValueError(f"Row {row_idx}: signal has no neurons")

    recording_type = infer_recording_type(row)

    # per-neuron mean/std across time
    mean_activity = np.nanmean(signal, axis=0)
    std_activity = np.nanstd(signal, axis=0)

    session_base = base_name_from_row(row_idx, row) + "_mean_activity"
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
        },
        "per_neuron": {
            "mean_activity": mean_activity,
            "std_activity": std_activity,
        },
    }
    save_pickle_standard(session_pickle_path, session_payload)

    neuron_rows = []
    for neuron_idx in range(n_neurons):
        neuron_uid = (
            f"{row['dataset_name']}__{row['mouse_name']}__{row['session_name']}__"
            f"{row['depth_type']}__row{row_idx}__neuron{neuron_idx}"
        )

        neuron_rows.append({
            "dataset_name": row["dataset_name"],
            "mouse_name": row["mouse_name"],
            "session_name": row["session_name"],
            "depth_type": row["depth_type"],
            "recording_type": recording_type,
            "standardized_file_path": standardized_file_path,
            "row_index": row_idx,
            "neuron_index": neuron_idx,
            "neuron_uid": neuron_uid,
            "n_timepoints": n_timepoints,
            "n_neurons_in_session": n_neurons,
            "mean_activity": float(mean_activity[neuron_idx]),
            "std_activity": float(std_activity[neuron_idx]),
        })

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
        "mean_activity_across_neurons_mean": float(np.nanmean(mean_activity)),
        "mean_activity_across_neurons_std": float(np.nanstd(mean_activity)),
        "std_activity_across_neurons_mean": float(np.nanmean(std_activity)),
        "std_activity_across_neurons_std": float(np.nanstd(std_activity)),
        "session_pickle_name": session_pickle_name,
        "session_pickle_path": session_pickle_path,
    }

    if VERBOSE:
        print(
            f"Row {row_idx} | dataset={row['dataset_name']} | mouse={row['mouse_name']} | "
            f"session={row['session_name']} | depth={row['depth_type']} | "
            f"timepoints={n_timepoints} | neurons={n_neurons} | "
            f"mean(mean_activity)={session_row['mean_activity_across_neurons_mean']:.6f} | "
            f"mean(std_activity)={session_row['std_activity_across_neurons_mean']:.6f}"
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

    init_csv(NEURON_CSV, NEURON_COLUMNS)
    init_csv(SESSION_CSV, SESSION_COLUMNS)

    log_path = os.path.join(LOG_DIR, "neuron_mean_activity_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Per-neuron mean activity / firing-rate-equivalent analysis")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Rows to process: {len(df)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Neuron CSV: {NEURON_CSV}")
    print(f"Session CSV: {SESSION_CSV}")
    print(f"Session PKL dir: {SESSION_PKL_DIR}")
    print("")

    ok_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        local_time = timeit.default_timer()
        try:
            neuron_rows, session_row = process_row(row_idx, row)

            for nr in neuron_rows:
                append_csv_row(NEURON_CSV, NEURON_COLUMNS, nr)

            append_csv_row(SESSION_CSV, SESSION_COLUMNS, session_row)

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


####################################################################################################################
##############                  PLOTTING                          ###################################################
####################################################################################################################

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot neuron mean activity histograms across conditions.

Input
-----
CSV produced by the per-neuron mean activity script:
    neuron_mean_activity_per_neuron.csv

What this script does
---------------------
1) Learning dataset:
   - parse session date from session_name
   - order sessions within each mouse
   - assign session number 1,2,3,4,...
   - combine mice by session number
   - plot all histograms together using the same bins

2) Datasets with deep/superficial populations:
   - for each dataset containing both deep and superficial
   - combine mice within each depth
   - plot deep vs superficial histograms together
   - use the usual color coding:
        superficial -> #9900ff
        deep        -> #cc9900

Outputs
-------
- PNG and SVG figures
- CSV summaries with histogram counts
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Paths
# =============================================================================

INPUT_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/neuron_mean_activity/neuron_mean_activity_per_neuron.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/neuron_mean_activity_histograms"
PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

# =============================================================================
# Parameters
# =============================================================================

# exact learning dataset name(s) if needed
LEARNING_DATASET_NAMES = ["learning"]

# histogram parameters
N_BINS = 40
ALPHA = 0.35
LINEWIDTH = 2.0
DENSITY = False  # False -> counts, True -> probability density

# color coding
DEPTH_COLORS = {
    "superficial": "#9900ff",
    "deep": "#cc9900",
    "whole population": "gray",
    "unknown": "gray",
}

# learning session colors
SESSION_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
    6: "#8c564b",
}


# =============================================================================
# Helpers
# =============================================================================

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


def save_figure_both(fig, out_base):
    png_path = os.path.join(PNG_DIR, out_base + ".png")
    svg_path = os.path.join(SVG_DIR, out_base + ".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")


def parse_session_datetime_from_name(session_name):
    s = str(session_name)

    dt = pd.to_datetime(s, format="%Y%m%d_%H%M%S", errors="coerce")
    if pd.notna(dt):
        return dt

    match = re.search(r"(\d{8}_\d{6})", s)
    if match is not None:
        return pd.to_datetime(match.group(1), format="%Y%m%d_%H%M%S", errors="coerce")

    return pd.NaT


def normalize_depth_label(depth_value):
    if pd.isna(depth_value):
        return "unknown"
    depth = str(depth_value).strip().lower()
    if depth in ["sup", "superficial"]:
        return "superficial"
    if depth == "deep":
        return "deep"
    if depth in ["whole", "whole population", "whole_population", "all"]:
        return "whole population"
    return depth


def compute_common_bins(values, n_bins=40):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite values available to compute histogram bins.")

    vmin = np.min(values)
    vmax = np.max(values)

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    bins = np.linspace(vmin, vmax, n_bins + 1)
    return bins


def save_hist_counts_csv(groups_dict, bins, out_name):
    rows = []
    bin_left = bins[:-1]
    bin_right = bins[1:]

    for group_name, values in groups_dict.items():
        hist, _ = np.histogram(values, bins=bins)
        for i in range(len(hist)):
            rows.append({
                "group": group_name,
                "bin_index": i,
                "bin_left": bin_left[i],
                "bin_right": bin_right[i],
                "count": int(hist[i]),
            })

    out_path = os.path.join(CSV_DIR, out_name + ".csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")


# =============================================================================
# Plotting functions
# =============================================================================

def plot_learning_session_histograms(df):
    learning_mask = df["dataset_name"].astype(str).str.lower().isin([x.lower() for x in LEARNING_DATASET_NAMES])
    df_learning = df[learning_mask].copy()

    if len(df_learning) == 0:
        print("[WARNING] No rows found for learning dataset.")
        return

    df_learning["session_datetime"] = df_learning["session_name"].apply(parse_session_datetime_from_name)

    # order sessions within each mouse, then assign session number 1,2,3,4,...
    session_table = (
        df_learning[["mouse_name", "session_name", "session_datetime"]]
        .drop_duplicates()
        .sort_values(["mouse_name", "session_datetime", "session_name"])
        .copy()
    )

    session_table["session_number"] = session_table.groupby("mouse_name").cumcount() + 1

    df_learning = df_learning.merge(
        session_table[["mouse_name", "session_name", "session_number"]],
        on=["mouse_name", "session_name"],
        how="left"
    )

    max_session = int(df_learning["session_number"].max())
    groups = {}
    for sn in range(1, max_session + 1):
        vals = df_learning.loc[df_learning["session_number"] == sn, "mean_activity"].dropna().to_numpy()
        if len(vals) > 0:
            groups[f"session{sn}"] = vals

    if len(groups) == 0:
        print("[WARNING] No valid learning-session groups found.")
        return

    all_values = np.concatenate(list(groups.values()))
    bins = compute_common_bins(all_values, n_bins=N_BINS)

    fig, ax = plt.subplots(figsize=(7, 5))

    for sn in range(1, max_session + 1):
        key = f"session{sn}"
        if key not in groups:
            continue

        color = SESSION_COLORS.get(sn, None)
        ax.hist(
            groups[key],
            bins=bins,
            alpha=ALPHA,
            density=DENSITY,
            color=color,
            label=f"Session {sn}",
            edgecolor=None,
        )
        ax.hist(
            groups[key],
            bins=bins,
            histtype="step",
            linewidth=LINEWIDTH,
            density=DENSITY,
            color=color,
        )

    ax.set_title("Learning dataset: mean activity per neuron across ordered sessions")
    ax.set_xlabel("Mean activity per neuron")
    ax.set_ylabel("Density" if DENSITY else "Count")
    ax.legend(frameon=False)
    ax.grid(False)

    plt.tight_layout()
    save_figure_both(fig, "learning_mean_activity_histograms_by_session")

    save_hist_counts_csv(groups, bins, "learning_mean_activity_histograms_by_session")


def plot_deep_sup_histograms_per_dataset(df):
    df = df.copy()
    df["depth_norm"] = df["depth_type"].apply(normalize_depth_label)

    dataset_names = sorted(df["dataset_name"].dropna().astype(str).unique().tolist())

    for dataset_name in dataset_names:
        df_dataset = df[df["dataset_name"] == dataset_name].copy()

        present_depths = sorted(df_dataset["depth_norm"].dropna().unique().tolist())
        has_deep = "deep" in present_depths
        has_sup = "superficial" in present_depths

        if not (has_deep and has_sup):
            continue

        deep_vals = df_dataset.loc[df_dataset["depth_norm"] == "deep", "mean_activity"].dropna().to_numpy()
        sup_vals = df_dataset.loc[df_dataset["depth_norm"] == "superficial", "mean_activity"].dropna().to_numpy()

        if len(deep_vals) == 0 or len(sup_vals) == 0:
            continue

        all_values = np.concatenate([deep_vals, sup_vals])
        bins = compute_common_bins(all_values, n_bins=N_BINS)

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.hist(
            sup_vals,
            bins=bins,
            alpha=ALPHA,
            density=DENSITY,
            color=DEPTH_COLORS["superficial"],
            label="superficial",
            edgecolor=None,
        )
        ax.hist(
            sup_vals,
            bins=bins,
            histtype="step",
            linewidth=LINEWIDTH,
            density=DENSITY,
            color=DEPTH_COLORS["superficial"],
        )

        ax.hist(
            deep_vals,
            bins=bins,
            alpha=ALPHA,
            density=DENSITY,
            color=DEPTH_COLORS["deep"],
            label="deep",
            edgecolor=None,
        )
        ax.hist(
            deep_vals,
            bins=bins,
            histtype="step",
            linewidth=LINEWIDTH,
            density=DENSITY,
            color=DEPTH_COLORS["deep"],
        )

        ax.set_title(f"{dataset_name}: mean activity per neuron (deep vs superficial)")
        ax.set_xlabel("Mean activity per neuron")
        ax.set_ylabel("Density" if DENSITY else "Count")
        ax.legend(frameon=False)
        ax.grid(False)

        plt.tight_layout()
        out_base = f"{sanitize_for_filename(dataset_name)}_mean_activity_histogram_deep_vs_superficial"
        save_figure_both(fig, out_base)

        save_hist_counts_csv(
            {"superficial": sup_vals, "deep": deep_vals},
            bins,
            f"{sanitize_for_filename(dataset_name)}_mean_activity_histogram_deep_vs_superficial"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)
    ensure_dir(CSV_DIR)

    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = [
        "dataset_name",
        "mouse_name",
        "session_name",
        "depth_type",
        "mean_activity",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {missing_cols}")

    print(f"Loaded {len(df)} neuron rows")
    print("Available dataset_name values:")
    print(sorted(df["dataset_name"].dropna().astype(str).unique().tolist()))

    plot_learning_session_histograms(df)
    plot_deep_sup_histograms_per_dataset(df)

    print("Done.")


if __name__ == "__main__":
    main()
