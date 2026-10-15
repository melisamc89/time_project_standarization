#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated filter analysis across all datasets.

This script computes, for every standardized dataset/session/depth entry:
    1) MEB expansion of trial centroids across smoothing filters
    2) Structure Index (SI) across smoothing filters

Input:
    - master dataset index CSV
    - standardized per-session pickle files

Expected standardized pickle fields:
    signal         : array (T, N)
    pos            : array (T,) or position-like vector
    time           : array (T,)
    mov_dir        : array (T,)
    speed          : array (T,)
    trial_id       : array (T,)   [optional if recomputing]
    inner_time     : array (T,)   [optional]

Outputs:
    - session-level pickle with all results
    - combined CSV with all MEB and SI values
    - summary plots (.png and .svg)
    - stats text files with same basename as corresponding figures

Author: ChatGPT adaptation for Emanuel
"""

import os
import copy
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu

from sklearn.metrics import pairwise_distances

from structure_index import compute_structure_index


# =============================================================================
# PARAMETERS
# =============================================================================

# ---------- Paths ----------
BASE_STD_DIR = "/home/melisamc/Documents/time_project_standarization"
MASTER_INDEX_CSV = os.path.join(BASE_STD_DIR, "data", "metadata", "master_dataset_index.csv")

SAVE_ROOT = os.path.join(BASE_STD_DIR, "analysis", "filter_influence_combined")
os.makedirs(SAVE_ROOT, exist_ok=True)

# ---------- Dataset selection ----------
# Set to None to use all datasets found in master index
DATASETS_TO_INCLUDE = None
# Example:
# DATASETS_TO_INCLUDE = ["learning", "lt_rot", "lt_ephys_mice"]

DEPTHS_TO_INCLUDE = ["deep", "superficial", "whole population"]

# ---------- Signal field ----------
SIGNAL_KEY = "signal"

# ---------- Filtering ----------
SAMPLING_RATE = 20.0  # Hz
FILTER_SIGMAS = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

# ---------- Valid timepoint selection ----------
USE_ONLY_MOVING = False
MOVING_DIR_EXCLUDE_ZERO = False
MOVING_SPEED_THRESHOLD = None   # Example: 3.0

# ---------- Trial definition ----------
# Options:
#   "stored"      -> use trial_id from standardized file
#   "recomputed"  -> compute from position using thresholds below
TRIAL_ID_MODE = "stored"

TRIAL_HIGH_THR = 50
TRIAL_LOW_THR = 10

# ---------- Outlier filtering ----------
USE_SIGNAL_OUTLIER_FILTER = True
OUTLIER_DISTANCE_PERCENTILE = 1
OUTLIER_KEEP_PERCENTILE = 20

# ---------- SI parameters ----------
SI_NEIGH = 50

SI_PARAMS_CONT = {
    'n_bins': 10,
    'discrete_label': False,
    'continuity_kernel': None,
    'perc_neigh': 1,
    'num_shuffles': 0,
    'verbose': False
}

SI_PARAMS_DISC = {
    'n_bins': 3,
    'discrete_label': True,
    'continuity_kernel': None,
    'n_neighbors': 50,
    'num_shuffles': 0,
    'verbose': False
}

SI_BEHAVIORS = ["pos", "time", "speed", "trial_id", "dir", "(pos,dir)"]

# ---------- Optional neuron subsampling ----------
# Keep None to use all neurons
N_NEURONS_SUBSAMPLE = None
SUBSAMPLE_RANDOM_STATE = 42

# ---------- Plotting ----------
AREA_PALETTE = {
    "superficial": "#9900ff",
    "deep": "#cc9900",
    "whole population": "#666666"
}

# ---------- Stats ----------
STATS_FILTER_TIME_SEC = 0.2
STATS_BEHAVIORS = ["pos", "time"]

# =============================================================================
# UTILITIES
# =============================================================================

def load_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle_file(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def smooth_calcium_signals(signal, sigma=4):
    """
    Gaussian smoothing over time for each neuron.
    signal shape: (T, N)
    """
    if sigma == 0:
        return signal.copy()

    smoothed = np.zeros_like(signal, dtype=float)
    for i in range(signal.shape[1]):
        smoothed[:, i] = gaussian_filter1d(signal[:, i], sigma=sigma)
    return smoothed


def filter_noisy_outliers(data,
                          distance_percentile=1,
                          keep_percentile=20):
    """
    Filters noisy timepoints based on local neighborhood density.

    data: (T, D)
    returns:
        noise_idx, signal_idx
    """
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)

    thr = np.nanpercentile(D, distance_percentile)
    nn_count = np.sum(D < thr, axis=1) - 1

    cutoff = np.percentile(nn_count, keep_percentile)
    noise_idx = np.where(nn_count < cutoff)[0]
    signal_idx = np.where(nn_count >= cutoff)[0]
    return noise_idx, signal_idx


def compute_trial_id_from_position(pos, high_thr=50, low_thr=10):
    """
    Compute trial IDs from 1D position.
    """
    pos = np.asarray(pos).ravel()
    trial_id = np.zeros_like(pos, dtype=int)

    in_high = False
    current_id = 0

    for t in range(1, len(pos)):
        if (not in_high) and (pos[t] > high_thr):
            in_high = True
        elif in_high and (pos[t] < low_thr):
            current_id += 1
            in_high = False
        trial_id[t] = current_id

    return trial_id


# =============================================================================
# MEB FUNCTIONS
# =============================================================================

def _ball_from(points):
    P = np.asarray(points, dtype=float)

    if len(P) == 0:
        return np.zeros(1), 0.0
    if len(P) == 1:
        return P[0], 0.0
    if len(P) == 2:
        c = (P[0] + P[1]) / 2.0
        r = np.linalg.norm(P[0] - c)
        return c, r

    A = 2 * (P[1:] - P[0])
    b = np.sum(P[1:]**2 - P[0]**2, axis=1)
    try:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        c = P.mean(axis=0)

    r = np.max(np.linalg.norm(P - c, axis=1))
    return c, r


def _is_in_ball(p, c, r, eps=1e-10):
    return np.linalg.norm(p - c) <= r + eps


def _welzl(P, R, d):
    if len(R) == d + 1 or len(P) == 0:
        c, r = _ball_from(R)
        return c, r

    p = P.pop()
    c, r = _welzl(P, R, d)

    if _is_in_ball(p, c, r):
        P.append(p)
        return c, r

    R.append(p)
    c2, r2 = _welzl(P, R, d)
    R.pop()
    P.append(p)
    return c2, r2


def minimum_enclosing_ball(X):
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if n == 0:
        return np.zeros(d), 0.0

    P = [x for x in X]
    rng = np.random.default_rng(123)
    rng.shuffle(P)

    c, r = _welzl(P, [], d)
    return c, float(r)


def compute_meb_expansion(signal, trial_id):
    """
    Compute normalized MEB radius of trial centroids.
    signal: (T, N)
    trial_id: (T,)
    """
    trial_id = np.asarray(trial_id).astype(int).ravel()
    unique_tids = np.unique(trial_id)

    if unique_tids.size < 2:
        return np.nan, np.empty((0, signal.shape[1]))

    centroids = np.vstack([
        np.nanmean(signal[trial_id == tid], axis=0)
        for tid in unique_tids
    ])

    _, radius = minimum_enclosing_ball(centroids)
    radius_norm = radius / np.sqrt(centroids.shape[1])
    return radius_norm, centroids


# =============================================================================
# STANDARDIZED DATA HANDLING
# =============================================================================

def get_column(df, preferred_names):
    """
    Return first existing column from preferred_names.
    """
    for c in preferred_names:
        if c in df.columns:
            return c
    return None


def build_behavior_dict(data_dict):
    """
    Build behavior dictionary with flexible key handling.
    """
    beh = {}

    pos_key = None
    for k in ["pos", "position"]:
        if k in data_dict:
            pos_key = k
            break
    if pos_key is not None:
        beh["pos"] = np.asarray(data_dict[pos_key]).ravel()

    if "time" in data_dict:
        beh["time"] = np.asarray(data_dict["time"]).ravel()

    if "speed" in data_dict:
        beh["speed"] = np.asarray(data_dict["speed"]).ravel()

    if "mov_dir" in data_dict:
        beh["dir"] = np.asarray(data_dict["mov_dir"]).ravel()

    if "trial_id" in data_dict:
        beh["trial_id"] = np.asarray(data_dict["trial_id"]).ravel()

    if "inner_time" in data_dict:
        beh["inner_time"] = np.asarray(data_dict["inner_time"]).ravel()

    if ("pos" in beh) and ("dir" in beh):
        beh["(pos,dir)"] = np.column_stack([beh["pos"], beh["dir"]])

    return beh


def get_valid_time_mask(beh, n_timepoints):
    mask = np.ones(n_timepoints, dtype=bool)

    if USE_ONLY_MOVING:
        if MOVING_DIR_EXCLUDE_ZERO and "dir" in beh:
            mask &= (beh["dir"] != 0)
        if (MOVING_SPEED_THRESHOLD is not None) and ("speed" in beh):
            mask &= np.abs(beh["speed"]) >= MOVING_SPEED_THRESHOLD

    return mask


def maybe_subsample_neurons(signal, n_neurons=None, random_state=42):
    if n_neurons is None:
        return signal

    if signal.shape[1] <= n_neurons:
        return signal

    rng = np.random.default_rng(random_state)
    idx = rng.choice(signal.shape[1], size=n_neurons, replace=False)
    idx = np.sort(idx)
    return signal[:, idx]


def prepare_trial_id(beh):
    if TRIAL_ID_MODE == "stored":
        if "trial_id" not in beh:
            raise ValueError("TRIAL_ID_MODE='stored' but no trial_id found in file.")
        return beh["trial_id"].astype(int)

    elif TRIAL_ID_MODE == "recomputed":
        if "pos" not in beh:
            raise ValueError("TRIAL_ID_MODE='recomputed' but no position found in file.")
        return compute_trial_id_from_position(
            beh["pos"],
            high_thr=TRIAL_HIGH_THR,
            low_thr=TRIAL_LOW_THR
        )

    else:
        raise ValueError(f"Unknown TRIAL_ID_MODE: {TRIAL_ID_MODE}")


def get_si_params_for_behavior(beh_name, beh_val):
    if beh_name == "dir":
        params = copy.deepcopy(SI_PARAMS_DISC)
        params["n_neighbors"] = SI_NEIGH
    else:
        params = copy.deepcopy(SI_PARAMS_CONT)
        params["n_neighbors"] = SI_NEIGH

    if beh_name == "trial_id":
        params["min_label"] = float(np.nanmin(beh_val))
        params["max_label"] = float(np.nanmax(beh_val))

    return params


# =============================================================================
# PER-SESSION ANALYSIS
# =============================================================================

def analyze_single_session(data_dict,
                           dataset_name,
                           mouse_name,
                           session_name,
                           depth_type):
    """
    Run MEB + SI filter analysis for one standardized session file.
    """

    if SIGNAL_KEY not in data_dict:
        raise KeyError(f"Missing '{SIGNAL_KEY}' in data dictionary.")

    signal = np.asarray(data_dict[SIGNAL_KEY], dtype=float)
    if signal.ndim != 2:
        raise ValueError(f"Signal must be 2D (T, N), got shape {signal.shape}")

    signal = maybe_subsample_neurons(
        signal,
        n_neurons=N_NEURONS_SUBSAMPLE,
        random_state=SUBSAMPLE_RANDOM_STATE
    )

    beh = build_behavior_dict(data_dict)

    n_time = signal.shape[0]
    valid_mask = get_valid_time_mask(beh, n_time)
    valid_idx = np.where(valid_mask)[0]

    signal_valid = signal[valid_idx, :]

    beh_valid = {}
    for k, v in beh.items():
        if v.ndim == 1:
            beh_valid[k] = v[valid_idx]
        else:
            beh_valid[k] = v[valid_idx, :]

    trial_id_valid = prepare_trial_id(beh_valid)

    # Outlier filter in signal space
    if USE_SIGNAL_OUTLIER_FILTER:
        _, signal_idx = filter_noisy_outliers(
            signal_valid,
            distance_percentile=OUTLIER_DISTANCE_PERCENTILE,
            keep_percentile=OUTLIER_KEEP_PERCENTILE
        )
    else:
        signal_idx = np.arange(signal_valid.shape[0])

    results = {
        "metadata": {
            "dataset_name": dataset_name,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "depth_type": depth_type,
            "n_timepoints_raw": int(signal.shape[0]),
            "n_neurons": int(signal.shape[1]),
            "n_timepoints_valid": int(signal_valid.shape[0]),
            "n_timepoints_signal_idx": int(len(signal_idx))
        },
        "meb": {},
        "si": {}
    }

    # ----- MEB across filters -----
    for sigma in FILTER_SIGMAS:
        sig_f = smooth_calcium_signals(signal_valid, sigma=sigma)
        meb_value, _ = compute_meb_expansion(sig_f, trial_id_valid)

        results["meb"][str(sigma)] = {
            "expansion": float(meb_value) if np.isfinite(meb_value) else np.nan
        }

    # ----- SI across filters -----
    for beh_name in SI_BEHAVIORS:
        if beh_name not in beh_valid:
            continue

        beh_val = beh_valid[beh_name]
        results["si"][beh_name] = {}

        params = get_si_params_for_behavior(beh_name, beh_val)

        for sigma in FILTER_SIGMAS:
            sig_f = smooth_calcium_signals(signal_valid, sigma=sigma)

            try:
                si, process_info, overlap_mat, _ = compute_structure_index(
                    sig_f[signal_idx],
                    beh_val[signal_idx],
                    **params
                )
            except Exception as e:
                warnings.warn(
                    f"SI failed for {dataset_name} | {mouse_name} | {session_name} | "
                    f"{depth_type} | {beh_name} | sigma={sigma}: {e}"
                )
                si = np.nan
                process_info = None
                overlap_mat = None

            results["si"][beh_name][str(sigma)] = {
                "si": float(si) if np.isfinite(si) else np.nan,
                "beh_params": copy.deepcopy(params),
                "process_info": process_info,
                "overlap_mat": overlap_mat
            }

    return results


# =============================================================================
# MASTER LOOP
# =============================================================================

def load_master_index():
    df = pd.read_csv(MASTER_INDEX_CSV)

    dataset_col = get_column(df, ["dataset_name"])
    mouse_col = get_column(df, ["mouse_name"])
    session_col = get_column(df, ["session_name"])
    depth_col = get_column(df, ["depth_type"])
    path_col = get_column(df, ["standardized_file_path"])

    required = {
        "dataset_name": dataset_col,
        "mouse_name": mouse_col,
        "session_name": session_col,
        "depth_type": depth_col,
        "standardized_file_path": path_col
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing columns in master index: {missing}")

    df = df.rename(columns={
        dataset_col: "dataset_name",
        mouse_col: "mouse_name",
        session_col: "session_name",
        depth_col: "depth_type",
        path_col: "standardized_file_path"
    })

    return df


def analyze_all_sessions():
    master_df = load_master_index()

    if DATASETS_TO_INCLUDE is not None:
        master_df = master_df[master_df["dataset_name"].isin(DATASETS_TO_INCLUDE)].copy()

    if DEPTHS_TO_INCLUDE is not None:
        master_df = master_df[master_df["depth_type"].isin(DEPTHS_TO_INCLUDE)].copy()

    master_df = master_df.reset_index(drop=True)

    all_rows_meb = []
    all_rows_si = []

    n_total = len(master_df)
    print(f"Found {n_total} entries to analyze.")

    for i, row in master_df.iterrows():
        dataset_name = row["dataset_name"]
        mouse_name = row["mouse_name"]
        session_name = row["session_name"]
        depth_type = row["depth_type"]
        fpath = row["standardized_file_path"]

        print(f"[{i+1}/{n_total}] {dataset_name} | {mouse_name} | {session_name} | {depth_type}")

        try:
            data_dict = load_pickle_file(fpath)
        except Exception as e:
            print(f"  -> Could not load file: {fpath}")
            print(f"  -> Error: {e}")
            continue

        try:
            res = analyze_single_session(
                data_dict=data_dict,
                dataset_name=dataset_name,
                mouse_name=mouse_name,
                session_name=session_name,
                depth_type=depth_type
            )
        except Exception as e:
            print(f"  -> Analysis failed.")
            print(f"  -> Error: {e}")
            continue

        # Save session-level pickle
        safe_dataset = str(dataset_name).replace("/", "_")
        safe_mouse = str(mouse_name).replace("/", "_")
        safe_session = str(session_name).replace("/", "_")
        safe_depth = str(depth_type).replace("/", "_").replace(" ", "_")

        session_save_dir = os.path.join(SAVE_ROOT, "session_pickles", safe_dataset, safe_mouse)
        os.makedirs(session_save_dir, exist_ok=True)

        session_pkl = os.path.join(
            session_save_dir,
            f"{safe_session}__{safe_depth}__filter_analysis.pkl"
        )
        save_pickle_file(session_pkl, res)

        # Flatten MEB
        for sigma_str, val in res["meb"].items():
            sigma = int(sigma_str)
            all_rows_meb.append({
                "dataset_name": dataset_name,
                "mouse_name": mouse_name,
                "session_name": session_name,
                "depth_type": depth_type,
                "filter_sigma": sigma,
                "filter_time_sec": sigma / SAMPLING_RATE,
                "meb_expansion": val["expansion"]
            })

        # Flatten SI
        for beh_name, beh_dict in res["si"].items():
            for sigma_str, val in beh_dict.items():
                sigma = int(sigma_str)
                all_rows_si.append({
                    "dataset_name": dataset_name,
                    "mouse_name": mouse_name,
                    "session_name": session_name,
                    "depth_type": depth_type,
                    "behavioral_label": beh_name,
                    "filter_sigma": sigma,
                    "filter_time_sec": sigma / SAMPLING_RATE,
                    "si": val["si"]
                })

    df_meb = pd.DataFrame(all_rows_meb)
    df_si = pd.DataFrame(all_rows_si)

    df_meb.to_csv(os.path.join(SAVE_ROOT, "filter_analysis_meb_all_sessions.csv"), index=False)
    df_si.to_csv(os.path.join(SAVE_ROOT, "filter_analysis_si_all_sessions.csv"), index=False)

    return df_meb, df_si


# =============================================================================
# PLOTTING
# =============================================================================

def save_stats_txt(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(str(line) + "\n")


def plot_meb_summary(df_meb):
    """
    One row per dataset, lineplot of MEB vs filter time.
    """
    if df_meb.empty:
        return

    datasets = sorted(df_meb["dataset_name"].unique())
    fig, axes = plt.subplots(len(datasets), 1,
                             figsize=(8, max(2.5, 2.5 * len(datasets))),
                             sharex=True)

    if len(datasets) == 1:
        axes = [axes]

    stats_lines = []

    for ax, ds in zip(axes, datasets):
        sub = df_meb[df_meb["dataset_name"] == ds].copy()

        sns.lineplot(
            data=sub,
            x="filter_time_sec",
            y="meb_expansion",
            hue="depth_type",
            palette=AREA_PALETTE,
            errorbar="sd",
            marker="o",
            ax=ax
        )

        ax.set_title(f"MEB expansion vs filter time | {ds}")
        ax.set_ylabel("MEB expansion")
        ax.grid(False)

        # stats summary
        tmp = (
            sub.groupby(["depth_type", "filter_time_sec"])["meb_expansion"]
               .agg(["mean", "std", "count"])
               .reset_index()
        )
        stats_lines.append(f"=== DATASET: {ds} ===")
        stats_lines.append(tmp.to_string(index=False))
        stats_lines.append("")

    axes[-1].set_xlabel("Filter sigma / time bins (s)")
    plt.tight_layout()

    out_png = os.path.join(SAVE_ROOT, "MEB_filter_summary_all_datasets.png")
    out_svg = os.path.join(SAVE_ROOT, "MEB_filter_summary_all_datasets.svg")
    out_txt = os.path.join(SAVE_ROOT, "MEB_filter_summary_all_datasets.txt")

    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_svg, dpi=400, bbox_inches="tight")
    plt.close(fig)

    save_stats_txt(out_txt, stats_lines)


def plot_si_summary(df_si):
    """
    For each dataset, plot SI vs filter time for selected behaviors.
    """
    if df_si.empty:
        return

    datasets = sorted(df_si["dataset_name"].unique())

    for ds in datasets:
        sub_ds = df_si[df_si["dataset_name"] == ds].copy()
        behaviors = [b for b in SI_BEHAVIORS if b in sub_ds["behavioral_label"].unique()]
        if len(behaviors) == 0:
            continue

        fig, axes = plt.subplots(len(behaviors), 1,
                                 figsize=(8, max(2.2, 2.2 * len(behaviors))),
                                 sharex=True)

        if len(behaviors) == 1:
            axes = [axes]

        stats_lines = [f"=== DATASET: {ds} ==="]

        for ax, beh in zip(axes, behaviors):
            sub = sub_ds[sub_ds["behavioral_label"] == beh].copy()

            sns.lineplot(
                data=sub,
                x="filter_time_sec",
                y="si",
                hue="depth_type",
                palette=AREA_PALETTE,
                errorbar="sd",
                marker="o",
                ax=ax
            )

            ax.set_title(f"SI vs filter time | {ds} | {beh}")
            ax.set_ylabel("SI")
            ax.set_ylim(0, 1)
            ax.grid(False)

            tmp = (
                sub.groupby(["depth_type", "filter_time_sec"])["si"]
                   .agg(["mean", "std", "count"])
                   .reset_index()
            )
            stats_lines.append(f"--- behavior: {beh} ---")
            stats_lines.append(tmp.to_string(index=False))
            stats_lines.append("")

        axes[-1].set_xlabel("Filter sigma / time bins (s)")
        plt.tight_layout()

        safe_ds = ds.replace("/", "_")
        out_png = os.path.join(SAVE_ROOT, f"SI_filter_summary_{safe_ds}.png")
        out_svg = os.path.join(SAVE_ROOT, f"SI_filter_summary_{safe_ds}.svg")
        out_txt = os.path.join(SAVE_ROOT, f"SI_filter_summary_{safe_ds}.txt")

        fig.savefig(out_png, dpi=400, bbox_inches="tight")
        fig.savefig(out_svg, dpi=400, bbox_inches="tight")
        plt.close(fig)

        save_stats_txt(out_txt, stats_lines)


def plot_si_bar_at_fixed_filter(df_si,
                                filter_time_sec=0.2,
                                behaviors=("pos", "time")):
    """
    Per dataset bar plot at one chosen filter time.
    Uses Mann-Whitney U (two-sided) between deep and superficial.
    """
    if df_si.empty:
        return

    datasets = sorted(df_si["dataset_name"].unique())

    for ds in datasets:
        sub = df_si[
            (df_si["dataset_name"] == ds) &
            (np.isclose(df_si["filter_time_sec"], filter_time_sec)) &
            (df_si["behavioral_label"].isin(behaviors)) &
            (df_si["depth_type"].isin(["deep", "superficial"]))
        ].copy()

        if sub.empty:
            continue

        fig = plt.figure(figsize=(7, 5))
        ax = sns.barplot(
            data=sub,
            x="behavioral_label",
            y="si",
            hue="depth_type",
            palette=AREA_PALETTE,
            errorbar="sd",
            capsize=0.1
        )

        y_max = np.nanmax(sub["si"].values) if len(sub) else 1.0
        y_step = 0.05
        stats_lines = [f"=== DATASET: {ds} | filter_time_sec={filter_time_sec} ==="]

        for i, beh in enumerate(behaviors):
            sub_b = sub[sub["behavioral_label"] == beh]
            si_deep = sub_b[sub_b["depth_type"] == "deep"]["si"].dropna().values
            si_sup = sub_b[sub_b["depth_type"] == "superficial"]["si"].dropna().values

            if (len(si_deep) < 2) or (len(si_sup) < 2):
                stats_lines.append(f"{beh}: insufficient data")
                continue

            stat, pval = mannwhitneyu(si_deep, si_sup, alternative="two-sided")
            stats_lines.append(
                f"{beh}: Mann-Whitney U={stat:.4f}, p={pval:.6g}, "
                f"deep_mean={np.mean(si_deep):.4f}, sup_mean={np.mean(si_sup):.4f}, "
                f"n_deep={len(si_deep)}, n_sup={len(si_sup)}"
            )

            if pval < 0.05:
                x1, x2 = i - 0.2, i + 0.2
                y = y_max + (i + 1) * y_step
                ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw=1.2, color="black")
                ax.text((x1 + x2) / 2, y + 0.015, "*", ha="center", va="bottom", fontsize=12)

        ax.set_title(f"SI at {filter_time_sec:.2f}s | {ds}")
        ax.set_ylabel("Structure Index")
        ax.set_xlabel("Behavior")
        ax.set_ylim(0, max(1.0, y_max + 0.20))
        sns.despine()
        plt.tight_layout()

        safe_ds = ds.replace("/", "_")
        out_png = os.path.join(SAVE_ROOT, f"SI_bar_{safe_ds}_{filter_time_sec:.2f}s.png")
        out_svg = os.path.join(SAVE_ROOT, f"SI_bar_{safe_ds}_{filter_time_sec:.2f}s.svg")
        out_txt = os.path.join(SAVE_ROOT, f"SI_bar_{safe_ds}_{filter_time_sec:.2f}s.txt")

        fig.savefig(out_png, dpi=400, bbox_inches="tight")
        fig.savefig(out_svg, dpi=400, bbox_inches="tight")
        plt.close(fig)

        save_stats_txt(out_txt, stats_lines)


def plot_joint_pos_time_scatter(df_si, filter_time_sec=0.2):
    """
    Scatter of SI(pos) vs SI(time), per dataset.
    """
    if df_si.empty:
        return

    datasets = sorted(df_si["dataset_name"].unique())

    for ds in datasets:
        sub = df_si[
            (df_si["dataset_name"] == ds) &
            (np.isclose(df_si["filter_time_sec"], filter_time_sec)) &
            (df_si["behavioral_label"].isin(["pos", "time"]))
        ].copy()

        if sub.empty:
            continue

        wide = sub.pivot_table(
            index=["dataset_name", "mouse_name", "session_name", "depth_type"],
            columns="behavioral_label",
            values="si"
        ).reset_index()

        if ("pos" not in wide.columns) or ("time" not in wide.columns):
            continue

        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            data=wide,
            x="pos",
            y="time",
            hue="depth_type",
            palette=AREA_PALETTE,
            s=100,
            edgecolor="black"
        )
        plt.axline((0, 0), slope=1, linestyle="--", color="gray", linewidth=1)
        plt.xlabel("SI(pos)")
        plt.ylabel("SI(time)")
        plt.title(f"SI pos vs time | {ds} | {filter_time_sec:.2f}s")
        plt.tight_layout()

        safe_ds = ds.replace("/", "_")
        out_png = os.path.join(SAVE_ROOT, f"SI_scatter_pos_time_{safe_ds}_{filter_time_sec:.2f}s.png")
        out_svg = os.path.join(SAVE_ROOT, f"SI_scatter_pos_time_{safe_ds}_{filter_time_sec:.2f}s.svg")

        plt.savefig(out_png, dpi=400, bbox_inches="tight")
        plt.savefig(out_svg, dpi=400, bbox_inches="tight")
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def save_run_config():
    cfg = {
        "BASE_STD_DIR": BASE_STD_DIR,
        "MASTER_INDEX_CSV": MASTER_INDEX_CSV,
        "SAVE_ROOT": SAVE_ROOT,
        "DATASETS_TO_INCLUDE": DATASETS_TO_INCLUDE,
        "DEPTHS_TO_INCLUDE": DEPTHS_TO_INCLUDE,
        "SIGNAL_KEY": SIGNAL_KEY,
        "SAMPLING_RATE": SAMPLING_RATE,
        "FILTER_SIGMAS": FILTER_SIGMAS,
        "USE_ONLY_MOVING": USE_ONLY_MOVING,
        "MOVING_DIR_EXCLUDE_ZERO": MOVING_DIR_EXCLUDE_ZERO,
        "MOVING_SPEED_THRESHOLD": MOVING_SPEED_THRESHOLD,
        "TRIAL_ID_MODE": TRIAL_ID_MODE,
        "TRIAL_HIGH_THR": TRIAL_HIGH_THR,
        "TRIAL_LOW_THR": TRIAL_LOW_THR,
        "USE_SIGNAL_OUTLIER_FILTER": USE_SIGNAL_OUTLIER_FILTER,
        "OUTLIER_DISTANCE_PERCENTILE": OUTLIER_DISTANCE_PERCENTILE,
        "OUTLIER_KEEP_PERCENTILE": OUTLIER_KEEP_PERCENTILE,
        "SI_NEIGH": SI_NEIGH,
        "SI_BEHAVIORS": SI_BEHAVIORS,
        "N_NEURONS_SUBSAMPLE": N_NEURONS_SUBSAMPLE,
        "SUBSAMPLE_RANDOM_STATE": SUBSAMPLE_RANDOM_STATE,
        "STATS_FILTER_TIME_SEC": STATS_FILTER_TIME_SEC,
        "STATS_BEHAVIORS": list(STATS_BEHAVIORS)
    }

    with open(os.path.join(SAVE_ROOT, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    save_run_config()

    df_meb, df_si = analyze_all_sessions()

    print("\nSaved combined CSVs:")
    print(os.path.join(SAVE_ROOT, "filter_analysis_meb_all_sessions.csv"))
    print(os.path.join(SAVE_ROOT, "filter_analysis_si_all_sessions.csv"))

    plot_meb_summary(df_meb)
    plot_si_summary(df_si)
    plot_si_bar_at_fixed_filter(
        df_si,
        filter_time_sec=STATS_FILTER_TIME_SEC,
        behaviors=STATS_BEHAVIORS
    )
    plot_joint_pos_time_scatter(
        df_si,
        filter_time_sec=STATS_FILTER_TIME_SEC
    )

    print("\nDone.")


if __name__ == "__main__":
    main()