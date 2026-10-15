#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Energy slope analysis pipeline on standardized datasets.

This version saves:
1) one CSV with one row per session x top_k
2) one CSV with one row per session using the top_k that maximizes r_squared
3) one pickle per session with full outputs
4) per-session plots
5) one log txt file
"""

import os
import csv
import copy
import pickle
import timeit
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy.stats import linregress


# =============================================================================
# Paths
# =============================================================================

CSV_PATH = "/home/melisamc/Documents/time_project_standarization/data/metadata/master_dataset_index.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/energy_filter_standardized"
SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
PNG_DIR = os.path.join(PLOTS_DIR, "png")
SVG_DIR = os.path.join(PLOTS_DIR, "svg")
TXT_DIR = os.path.join(OUTPUT_DIR, "session_txt")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

ENERGY_RESULTS_CSV = os.path.join(OUTPUT_DIR, "energy_slope_per_session_topk.csv")
ENERGY_BEST_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "energy_slope_per_session_best_topk.csv")


# =============================================================================
# Parameters
# =============================================================================

MAX_ROWS = None
VERBOSE = True

PERCENTAGE_TRAIN = 10
TOP_K_VALUES = [
    10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    100, 200, 300
]

# only for the intermediate visualization plot
PLOT_TOPK = 50

SMOOTH_SIGMA = 16
LOCAL_MINIMA_ORDER = 1

MIN_TIMEPOINTS = 30
MIN_EVAL_TIMEPOINTS = 10
MIN_MINIMA_FOR_REGRESSION = 2


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
# Signal processing and energy functions
# =============================================================================

def smooth_signals(signal, sigma=16):
    signal = ensure_2d(signal)
    return gaussian_filter1d(signal, sigma=sigma, axis=0)


def compute_hopfield_energy(X, pattern_indices, eval_indices):
    patterns = X[pattern_indices, :]
    W = patterns.T @ patterns
    np.fill_diagonal(W, 0.0)

    X_eval = X[eval_indices, :]
    energy = -0.5 * np.einsum("ij,ij->i", X_eval @ W, X_eval)
    return energy, W


def get_energy_segments(signal, percentage_train=10):
    X = ensure_2d(signal)
    T = X.shape[0]

    train_len = int(percentage_train / 100.0 * T)
    train_len = max(train_len, 1)

    train_idx = np.arange(0, train_len)
    eval_idx = np.arange(train_len, T)

    if len(eval_idx) < 1:
        raise ValueError("No evaluation timepoints left after training split.")

    energy, W = compute_hopfield_energy(
        X,
        pattern_indices=train_idx,
        eval_indices=eval_idx,
    )

    label = f"First_{int(percentage_train)}"
    return {
        label: {
            "energy": energy,
            "eval_idx": eval_idx,
            "train_idx": train_idx,
            "W": W,
        }
    }


def compute_regression_on_topk_minima(energy, top_k, order=1):
    energy = ensure_1d(energy)
    x_rel = np.arange(len(energy), dtype=float)

    minima_idx_all = argrelextrema(energy, np.less, order=order)[0]
    minima_vals_all = energy[minima_idx_all] if len(minima_idx_all) > 0 else np.array([])

    if len(minima_idx_all) == 0:
        return {
            "n_minima_total": 0,
            "n_minima_used": 0,
            "minima_idx_all": minima_idx_all,
            "minima_vals_all": minima_vals_all,
            "minima_idx_used": np.array([], dtype=int),
            "minima_vals_used": np.array([], dtype=float),
            "slope": np.nan,
            "intercept": np.nan,
            "r_value": np.nan,
            "r_squared": np.nan,
            "p_value": np.nan,
            "std_err": np.nan,
            "x_used": np.array([], dtype=float),
        }

    if len(minima_idx_all) > top_k:
        keep = np.argsort(minima_vals_all)[:top_k]
        minima_idx_used = minima_idx_all[keep]
        minima_vals_used = minima_vals_all[keep]
    else:
        minima_idx_used = minima_idx_all.copy()
        minima_vals_used = minima_vals_all.copy()

    x_used = x_rel[minima_idx_used]

    if len(x_used) >= MIN_MINIMA_FOR_REGRESSION:
        slope, intercept, r_value, p_value, std_err = linregress(x_used, minima_vals_used)
        r_squared = r_value ** 2
    else:
        slope = np.nan
        intercept = np.nan
        r_value = np.nan
        r_squared = np.nan
        p_value = np.nan
        std_err = np.nan

    return {
        "n_minima_total": len(minima_idx_all),
        "n_minima_used": len(minima_idx_used),
        "minima_idx_all": minima_idx_all,
        "minima_vals_all": minima_vals_all,
        "minima_idx_used": minima_idx_used,
        "minima_vals_used": minima_vals_used,
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "r_squared": r_squared,
        "p_value": p_value,
        "std_err": std_err,
        "x_used": x_used,
    }


def choose_best_topk(result_dict):
    """
    Choose top_k maximizing r_squared.
    Tie-breaker: choose smallest top_k.
    Ignore NaN r_squared values.
    """
    candidates = []
    for top_k, res in result_dict.items():
        r2 = res["r_squared"]
        if np.isfinite(r2):
            candidates.append((top_k, r2))

    if len(candidates) == 0:
        return None

    max_r2 = max(x[1] for x in candidates)
    tied = [x[0] for x in candidates if x[1] == max_r2]
    best_topk = min(tied)
    return best_topk


# =============================================================================
# Plotting
# =============================================================================

def plot_energy_trace_with_regression(
    energy,
    topk_result,
    title,
    png_path,
    svg_path,
):
    energy = ensure_1d(energy)
    x = np.arange(len(energy))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, energy, linewidth=1.2, label="Energy")

    minima_idx_all = topk_result["minima_idx_all"]
    minima_vals_all = topk_result["minima_vals_all"]
    minima_idx_used = topk_result["minima_idx_used"]
    minima_vals_used = topk_result["minima_vals_used"]

    if len(minima_idx_all) > 0:
        ax.scatter(minima_idx_all, minima_vals_all, s=18, alpha=0.35, label="All minima")

    if len(minima_idx_used) > 0:
        ax.scatter(minima_idx_used, minima_vals_used, s=35, alpha=0.9, label="Selected minima")

    if len(minima_idx_used) >= MIN_MINIMA_FOR_REGRESSION and np.isfinite(topk_result["slope"]):
        x_line = np.linspace(np.min(minima_idx_used), np.max(minima_idx_used), 200)
        y_line = topk_result["slope"] * x_line + topk_result["intercept"]
        ax.plot(
            x_line,
            y_line,
            "--",
            linewidth=1.5,
            label=f"slope={topk_result['slope']:.4g}, R²={topk_result['r_squared']:.4g}",
        )

    ax.set_xlabel("Relative eval time")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def plot_stability_curve(
    topk_values,
    slope_values,
    r2_values,
    title_prefix,
    png_path,
    svg_path,
):
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

    axs[0].plot(topk_values, slope_values, "-o", linewidth=1.5, markersize=4)
    axs[0].set_ylabel("Slope")
    axs[0].set_title(f"{title_prefix} | slope vs top_k")
    axs[0].grid(False)

    axs[1].plot(topk_values, r2_values, "-o", linewidth=1.5, markersize=4)
    axs[1].set_xlabel("top_k minima")
    axs[1].set_ylabel("R²")
    axs[1].set_title(f"{title_prefix} | R² vs top_k")
    axs[1].grid(False)

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CSV schemas
# =============================================================================

ENERGY_RESULTS_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "recording_type",
    "standardized_file_path",
    "row_index",
    "segment_label",
    "percentage_train",
    "top_k",
    "smoothing_sigma",
    "n_timepoints",
    "n_neurons",
    "n_train",
    "n_eval",
    "n_minima_total",
    "n_minima_used",
    "slope",
    "intercept",
    "r_value",
    "r_squared",
    "p_value",
    "std_err",
    "session_pickle_name",
    "session_pickle_path",
]

ENERGY_BEST_SUMMARY_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "recording_type",
    "standardized_file_path",
    "row_index",
    "segment_label",
    "percentage_train",
    "best_top_k",
    "smoothing_sigma",
    "n_timepoints",
    "n_neurons",
    "n_train",
    "n_eval",
    "n_minima_total",
    "n_minima_used",
    "slope",
    "intercept",
    "r_value",
    "r_squared",
    "p_value",
    "std_err",
    "session_pickle_name",
    "session_pickle_path",
]


# =============================================================================
# Main row processing
# =============================================================================

def process_row(row_idx, row):
    standardized_file_path = row["standardized_file_path"]

    if not os.path.isfile(standardized_file_path):
        raise FileNotFoundError(f"Missing standardized file: {standardized_file_path}")

    data = load_pickle_standard(standardized_file_path)

    if "signal" not in data:
        raise KeyError(f"Missing key 'signal' in file: {standardized_file_path}")

    signal = ensure_2d(data["signal"])
    n_timepoints = signal.shape[0]
    n_neurons = signal.shape[1]

    if n_timepoints < MIN_TIMEPOINTS:
        raise ValueError(f"Row {row_idx}: too few timepoints ({n_timepoints})")
    if n_neurons < 1:
        return None, None

    signal_smooth = smooth_signals(signal, sigma=SMOOTH_SIGMA)
    energies = get_energy_segments(signal_smooth, percentage_train=PERCENTAGE_TRAIN)

    recording_type = infer_recording_type(row)

    session_base = base_name_from_row(row_idx, row) + "_energy"
    session_pickle_name = session_base + ".pkl"
    session_pickle_path = os.path.join(SESSION_PKL_DIR, session_pickle_name)

    all_rows = []
    best_summary_row = None

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
            "percentage_train": PERCENTAGE_TRAIN,
            "smoothing_sigma": SMOOTH_SIGMA,
            "top_k_values": copy.deepcopy(TOP_K_VALUES),
            "plot_topk": PLOT_TOPK,
            "local_minima_order": LOCAL_MINIMA_ORDER,
        },
        "energy_segments": {},
    }

    for segment_label, seg in energies.items():
        energy = ensure_1d(seg["energy"])
        eval_idx = ensure_1d(seg["eval_idx"]).astype(int)
        train_idx = ensure_1d(seg["train_idx"]).astype(int)

        if len(eval_idx) < MIN_EVAL_TIMEPOINTS:
            raise ValueError(
                f"Row {row_idx}: too few evaluation timepoints after split ({len(eval_idx)})"
            )

        session_payload["energy_segments"][segment_label] = {
            "energy": copy.deepcopy(energy),
            "eval_idx": copy.deepcopy(eval_idx),
            "train_idx": copy.deepcopy(train_idx),
            "W": copy.deepcopy(seg["W"]),
            "regressions": {},
        }

        slope_values_for_plot = []
        r2_values_for_plot = []

        for top_k in TOP_K_VALUES:
            result = compute_regression_on_topk_minima(
                energy,
                top_k=top_k,
                order=LOCAL_MINIMA_ORDER,
            )

            session_payload["energy_segments"][segment_label]["regressions"][top_k] = copy.deepcopy(result)

            row_out = {
                "dataset_name": row["dataset_name"],
                "mouse_name": row["mouse_name"],
                "session_name": row["session_name"],
                "depth_type": row["depth_type"],
                "recording_type": recording_type,
                "standardized_file_path": standardized_file_path,
                "row_index": row_idx,
                "segment_label": segment_label,
                "percentage_train": PERCENTAGE_TRAIN,
                "top_k": top_k,
                "smoothing_sigma": SMOOTH_SIGMA,
                "n_timepoints": n_timepoints,
                "n_neurons": n_neurons,
                "n_train": len(train_idx),
                "n_eval": len(eval_idx),
                "n_minima_total": result["n_minima_total"],
                "n_minima_used": result["n_minima_used"],
                "slope": result["slope"],
                "intercept": result["intercept"],
                "r_value": result["r_value"],
                "r_squared": result["r_squared"],
                "p_value": result["p_value"],
                "std_err": result["std_err"],
                "session_pickle_name": session_pickle_name,
                "session_pickle_path": session_pickle_path,
            }
            all_rows.append(row_out)

            slope_values_for_plot.append(result["slope"])
            r2_values_for_plot.append(result["r_squared"])

        best_topk = choose_best_topk(session_payload["energy_segments"][segment_label]["regressions"])
        if best_topk is not None:
            best_result = session_payload["energy_segments"][segment_label]["regressions"][best_topk]
            best_summary_row = {
                "dataset_name": row["dataset_name"],
                "mouse_name": row["mouse_name"],
                "session_name": row["session_name"],
                "depth_type": row["depth_type"],
                "recording_type": recording_type,
                "standardized_file_path": standardized_file_path,
                "row_index": row_idx,
                "segment_label": segment_label,
                "percentage_train": PERCENTAGE_TRAIN,
                "best_top_k": best_topk,
                "smoothing_sigma": SMOOTH_SIGMA,
                "n_timepoints": n_timepoints,
                "n_neurons": n_neurons,
                "n_train": len(train_idx),
                "n_eval": len(eval_idx),
                "n_minima_total": best_result["n_minima_total"],
                "n_minima_used": best_result["n_minima_used"],
                "slope": best_result["slope"],
                "intercept": best_result["intercept"],
                "r_value": best_result["r_value"],
                "r_squared": best_result["r_squared"],
                "p_value": best_result["p_value"],
                "std_err": best_result["std_err"],
                "session_pickle_name": session_pickle_name,
                "session_pickle_path": session_pickle_path,
            }

        plot_result = session_payload["energy_segments"][segment_label]["regressions"].get(PLOT_TOPK, None)
        if plot_result is not None:
            title_prefix = (
                f"{row['dataset_name']} | {row['mouse_name']} | "
                f"{row['session_name']} | {row['depth_type']} | "
                f"{segment_label} | top_k={PLOT_TOPK}"
            )
            plot_energy_trace_with_regression(
                energy=energy,
                topk_result=plot_result,
                title=title_prefix,
                png_path=os.path.join(PNG_DIR, session_base + f"_{segment_label}_energy_trace.png"),
                svg_path=os.path.join(SVG_DIR, session_base + f"_{segment_label}_energy_trace.svg"),
            )

        title_prefix = (
            f"{row['dataset_name']} | {row['mouse_name']} | "
            f"{row['session_name']} | {row['depth_type']} | {segment_label}"
        )
        plot_stability_curve(
            topk_values=TOP_K_VALUES,
            slope_values=slope_values_for_plot,
            r2_values=r2_values_for_plot,
            title_prefix=title_prefix,
            png_path=os.path.join(PNG_DIR, session_base + f"_{segment_label}_stability_curve.png"),
            svg_path=os.path.join(SVG_DIR, session_base + f"_{segment_label}_stability_curve.svg"),
        )

        txt_lines = []
        txt_lines.append("Energy analysis session summary")
        txt_lines.append(f"dataset_name: {row['dataset_name']}")
        txt_lines.append(f"mouse_name: {row['mouse_name']}")
        txt_lines.append(f"session_name: {row['session_name']}")
        txt_lines.append(f"depth_type: {row['depth_type']}")
        txt_lines.append(f"recording_type: {recording_type}")
        txt_lines.append(f"standardized_file_path: {standardized_file_path}")
        txt_lines.append(f"n_timepoints: {n_timepoints}")
        txt_lines.append(f"n_neurons: {n_neurons}")
        txt_lines.append(f"percentage_train: {PERCENTAGE_TRAIN}")
        txt_lines.append(f"smoothing_sigma: {SMOOTH_SIGMA}")
        txt_lines.append(f"segment_label: {segment_label}")
        txt_lines.append(f"n_train: {len(train_idx)}")
        txt_lines.append(f"n_eval: {len(eval_idx)}")
        txt_lines.append("")
        txt_lines.append("Best top_k by max r_squared")
        if best_topk is not None:
            txt_lines.append(f"best_top_k: {best_topk}")
            txt_lines.append(f"n_minima_total: {best_result['n_minima_total']}")
            txt_lines.append(f"n_minima_used: {best_result['n_minima_used']}")
            txt_lines.append(f"slope: {best_result['slope']}")
            txt_lines.append(f"intercept: {best_result['intercept']}")
            txt_lines.append(f"r_value: {best_result['r_value']}")
            txt_lines.append(f"r_squared: {best_result['r_squared']}")
            txt_lines.append(f"p_value: {best_result['p_value']}")
            txt_lines.append(f"std_err: {best_result['std_err']}")

        txt_path = os.path.join(TXT_DIR, session_base + f"_{segment_label}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

    save_pickle_standard(session_pickle_path, session_payload)

    if VERBOSE:
        print(
            f"Row {row_idx} | dataset={row['dataset_name']} | mouse={row['mouse_name']} | "
            f"session={row['session_name']} | depth={row['depth_type']} | "
            f"rec={recording_type} | timepoints={n_timepoints} | neurons={n_neurons}"
        )
        if best_summary_row is not None:
            slope_str = "nan" if pd.isna(best_summary_row["slope"]) else f"{best_summary_row['slope']:.6g}"
            r2_str = "nan" if pd.isna(best_summary_row["r_squared"]) else f"{best_summary_row['r_squared']:.6g}"
            print(
                f"  best_top_k={best_summary_row['best_top_k']} | "
                f"slope={slope_str} | r_squared={r2_str}"
            )

    return all_rows, best_summary_row


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SESSION_PKL_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)
    ensure_dir(TXT_DIR)
    ensure_dir(LOG_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    init_csv(ENERGY_RESULTS_CSV, ENERGY_RESULTS_COLUMNS)
    init_csv(ENERGY_BEST_SUMMARY_CSV, ENERGY_BEST_SUMMARY_COLUMNS)

    log_path = os.path.join(LOG_DIR, "energy_filter_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Energy slope analysis pipeline")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Rows to process: {len(df)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Results CSV: {ENERGY_RESULTS_CSV}")
    print(f"Best summary CSV: {ENERGY_BEST_SUMMARY_CSV}")
    print(f"Session PKL dir: {SESSION_PKL_DIR}")
    print(f"Plots dir: {PLOTS_DIR}")
    print(f"PERCENTAGE_TRAIN: {PERCENTAGE_TRAIN}")
    print(f"SMOOTH_SIGMA: {SMOOTH_SIGMA}")
    print("")

    ok_count = 0
    skip_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        local_time = timeit.default_timer()
        try:
            result = process_row(row_idx, row)

            if result == (None, None):
                skip_count += 1
                print(f"[SKIP] Row {row_idx}: signal has no neurons\n")
                continue

            all_rows, best_summary_row = result

            for out_row in all_rows:
                append_csv_row(ENERGY_RESULTS_CSV, ENERGY_RESULTS_COLUMNS, out_row)

            if best_summary_row is not None:
                append_csv_row(ENERGY_BEST_SUMMARY_CSV, ENERGY_BEST_SUMMARY_COLUMNS, best_summary_row)

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
    print(f"Skipped entries: {skip_count}")
    print(f"Failed entries: {fail_count}")
    print(f"Total elapsed time: {total_elapsed:.2f} s")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()