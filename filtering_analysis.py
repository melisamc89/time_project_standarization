#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated filter analysis across datasets (Learning & LT_Rot).
Computes & Saves:
    1) MEB expansion of trial centroids in neural activity space across filters [4, 8, 16]
    2) Structure Index (SI) in neural activity space across filters [4, 8, 16]
    3) Filtered 3D UMAP manifold embeddings & triplet plots (position, time, direction)
    4) Persistent Homology (H1 lifetime tracking) in UMAP space across filters
    5) Trial centroid MEB trajectories projected into UMAP space colored by Trial ID
    6) Comprehensive CSVs, SVG/PNG figures, and statistical text reports.

Author: Adapted for Melisa Beatriz Maidana Capitán / Emanuel Sánchez Aimar
"""

import os
import copy
import json
import pickle
import warnings
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import umap
from ripser import ripser as tda

from structure_index import compute_structure_index

# =============================================================================
# PARAMETERS & PATHS
# =============================================================================

POSSIBLE_ROOTS = [
    Path("/home/hippolab/Documents/melisa/time_project_standarization"),
    Path("/home/melisamc/Documents/time_project_standarization"),
]
BASE_STD_DIR = next((str(p) for p in POSSIBLE_ROOTS if p.exists()), str(POSSIBLE_ROOTS[0]))

MASTER_INDEX_CSV = os.path.join(BASE_STD_DIR, "data", "metadata", "master_dataset_index.csv")
SAVE_ROOT = os.path.join(BASE_STD_DIR, "analysis", "filter_influence_combined_complete")
os.makedirs(SAVE_ROOT, exist_ok=True)

# ---------- Dataset selection ----------
DATASETS_TO_INCLUDE = ["learning", "lt_rot"]
DEPTHS_TO_INCLUDE = ["deep", "superficial", "whole population"]

# ---------- Signal field ----------
SIGNAL_KEY = "signal"

# ---------- Filtering ----------
SAMPLING_RATE = 20.0  # Hz
FILTER_SIGMAS = [4, 8, 16]

# ---------- UMAP Parameters ----------
UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
N_COMPONENTS = 3
N_POS_BINS = 30
MIN_COUNT_PER_BIN = 5
PLANE_TILT_DEG = 30.0

# ---------- TDA Parameters ----------
N_SUBSAMPLE_RAW = 1200
N_SUBSAMPLE_UMAP = 1500

# ---------- Valid timepoint selection ----------
USE_ONLY_MOVING = False
MOVING_DIR_EXCLUDE_ZERO = False
MOVING_SPEED_THRESHOLD = None

# ---------- Trial definition (Updated to recomputed with dynamic thresholds) ----------
TRIAL_ID_MODE = "recomputed"
TRIAL_HIGH_PERCENTILE = 90.0  # Fraction of track length to consider reaching the edge
TRIAL_LOW_PERCENTILE = 15.0  # Fraction of track length to consider returning to start

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

N_NEURONS_SUBSAMPLE = None
SUBSAMPLE_RANDOM_STATE = 42

AREA_PALETTE = {
    "superficial": "#9900ff",
    "deep": "#cc9900",
    "whole population": "#666666"
}


# =============================================================================
# UTILITIES & SHAPE ALIGNMENT HELPERS
# =============================================================================

def load_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle_file(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_stats_txt(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(str(line) + "\n")


def smooth_calcium_signals(signal, sigma=4):
    if sigma == 0:
        return signal.copy()
    smoothed = np.zeros_like(signal, dtype=float)
    for i in range(signal.shape[1]):
        smoothed[:, i] = gaussian_filter1d(signal[:, i], sigma=sigma)
    return smoothed


def maybe_subsample_neurons(signal, n_neurons=None, random_state=42):
    if n_neurons is None:
        return signal
    if signal.shape[1] <= n_neurons:
        return signal
    rng = np.random.default_rng(random_state)
    idx = rng.choice(signal.shape[1], size=n_neurons, replace=False)
    idx = np.sort(idx)
    return signal[:, idx]


def filter_noisy_outliers(data, distance_percentile=1, keep_percentile=20):
    D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    thr = np.nanpercentile(D, distance_percentile)
    nn_count = np.sum(D < thr, axis=1) - 1
    cutoff = np.percentile(nn_count, keep_percentile)
    noise_idx = np.where(nn_count < cutoff)[0]
    signal_idx = np.where(nn_count >= cutoff)[0]
    return noise_idx, signal_idx


def compute_trial_id_from_position_dynamic(pos):
    """
    Computes full trial IDs (leave home, reach edge, return) using
    session-specific dynamic thresholds derived from position percentiles.
    """
    pos = np.asarray(pos).ravel()
    valid_pos = pos[np.isfinite(pos)]
    if len(valid_pos) == 0:
        return np.zeros_like(pos, dtype=int)

    high_thr = np.percentile(valid_pos, TRIAL_HIGH_PERCENTILE)
    low_thr = np.percentile(valid_pos, TRIAL_LOW_PERCENTILE)

    trial_id = np.zeros_like(pos, dtype=int)
    in_high = False
    current_id = 0

    for t in range(1, len(pos)):
        if (not in_high) and (pos[t] >= high_thr):
            in_high = True
        elif in_high and (pos[t] <= low_thr):
            current_id += 1
            in_high = False
        trial_id[t] = current_id

    return trial_id


def sanitize_for_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def ensure_1d(x, target_len=None):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    elif x.ndim == 2 and x.shape[0] > 1 and x.shape[1] > 1:
        x = x[:, 0]
    res = x.reshape(-1)
    if target_len is not None and len(res) != target_len:
        if len(res) == 2 * target_len:
            res = res[:target_len]
        else:
            res = res[:target_len] if len(res) > target_len else np.pad(res, (0, target_len - len(res)), mode='edge')
    return res


# =============================================================================
# MEB FUNCTIONS (Evaluated in Neural Activity Space)
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
    b = np.sum(P[1:] ** 2 - P[0] ** 2, axis=1)
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
# DATA HANDLING & BEHAVIOR
# =============================================================================

def get_column(df, preferred_names):
    for c in preferred_names:
        if c in df.columns:
            return c
    return None


def build_behavior_dict(data_dict, target_len):
    beh = {}
    pos_key = next((k for k in ["pos", "position"] if k in data_dict), None)
    if pos_key is not None:
        beh["pos"] = ensure_1d(data_dict[pos_key], target_len=target_len)

    if "time" in data_dict:
        beh["time"] = ensure_1d(data_dict["time"], target_len=target_len)

    if "speed" in data_dict:
        beh["speed"] = ensure_1d(data_dict["speed"], target_len=target_len)

    if "mov_dir" in data_dict:
        beh["dir"] = ensure_1d(data_dict["mov_dir"], target_len=target_len)

    if "trial_id" in data_dict:
        beh["trial_id"] = ensure_1d(data_dict["trial_id"], target_len=target_len).astype(int)

    if "inner_time" in data_dict:
        beh["inner_time"] = ensure_1d(data_dict["inner_time"], target_len=target_len)

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


def prepare_trial_id(beh, target_len):
    if TRIAL_ID_MODE == "stored":
        if "trial_id" not in beh:
            raise ValueError("TRIAL_ID_MODE='stored' but no trial_id found.")
        return ensure_1d(beh["trial_id"], target_len=target_len).astype(int)
    elif TRIAL_ID_MODE == "recomputed":
        if "pos" not in beh:
            raise ValueError("TRIAL_ID_MODE='recomputed' but no position found.")
        return compute_trial_id_from_position_dynamic(beh["pos"])
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
# PLOTTING & TDA HELPERS
# =============================================================================

def extract_h1_lifetime_metrics(dgms):
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    if len(h1) == 0:
        return 0.0, 0.0, 0.0

    lifetimes = h1[:, 1] - h1[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if len(lifetimes) == 0:
        return 0.0, 0.0, 0.0

    lifetimes = np.sort(lifetimes)[::-1]
    max_l1 = float(lifetimes[0])
    second_l1 = float(lifetimes[1]) if len(lifetimes) > 1 else 0.0
    ratio = float(max_l1 / second_l1) if second_l1 > 0 else float(max_l1)
    return max_l1, second_l1, ratio


def style_3d_axis(ax, title):
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(False)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("white")
    ax.set_xlabel("dim 1", fontsize=8)
    ax.set_ylabel("dim 2", fontsize=8)
    ax.set_zlabel("dim 3", fontsize=8)


def plot_umap_triplet(umap_emb, pos_1d, mov_dir, time_1d, dataset_name, mouse_name, session_name, depth_type, sigma,
                      save_dir):
    fig = plt.figure(figsize=(18, 6), dpi=300)
    fig.suptitle(
        f"UMAPPED Filtered Data (sigma={sigma}) | {dataset_name} | {mouse_name} | {session_name} | {depth_type}",
        fontsize=12, fontweight="bold")

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    sc1 = ax1.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=pos_1d, cmap="magma", s=3, alpha=0.85)
    style_3d_axis(ax1, "Colored by Position")
    fig.colorbar(sc1, ax=ax1, fraction=0.03, pad=0.04).set_label("position")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    dir_cmap = ListedColormap(["blue", "gray", "lightblue"])
    dir_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], dir_cmap.N)
    sc2 = ax2.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=mov_dir, cmap=dir_cmap, norm=dir_norm, s=3,
                      alpha=0.85)
    style_3d_axis(ax2, "Colored by Direction")
    fig.colorbar(sc2, ax=ax2, fraction=0.03, pad=0.04, ticks=[-1, 0, 1]).set_label("direction")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    sc3 = ax3.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=time_1d, cmap="YlGn_r", s=3, alpha=0.85)
    style_3d_axis(ax3, "Colored by Time")
    fig.colorbar(sc3, ax=ax3, fraction=0.03, pad=0.04).set_label("time")

    plt.tight_layout()
    safe_base = f"{sanitize_for_filename(dataset_name)}_{sanitize_for_filename(mouse_name)}_{sanitize_for_filename(session_name)}_{sanitize_for_filename(depth_type)}_sigma{sigma}_umap_triplet"
    fig.savefig(os.path.join(save_dir, safe_base + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, safe_base + ".svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trial_centroid_meb_umap(umap_emb, trial_id, dataset_name, mouse_name, session_name, depth_type, sigma,
                                 save_dir):
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], color="#d3d3d3", s=2, alpha=0.35, label="Manifold")

    unique_tids = np.unique(trial_id)
    trial_umap_centroids = []
    valid_tids = []
    for tid in unique_tids:
        mask = (trial_id == tid)
        if np.any(mask):
            trial_umap_centroids.append(np.mean(umap_emb[mask], axis=0))
            valid_tids.append(tid)
    trial_umap_centroids = np.asarray(trial_umap_centroids)

    if len(trial_umap_centroids) > 0:
        sc = ax.scatter(
            trial_umap_centroids[:, 0], trial_umap_centroids[:, 1], trial_umap_centroids[:, 2],
            c=valid_tids, cmap="viridis", s=50, edgecolor="k", linewidth=0.8, label="Trial Centroids"
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.05)
        cbar.set_label("Trial ID")

    ax.set_title(f"Trial Centroids on UMAP (sigma={sigma}) | {dataset_name} | {mouse_name}", fontsize=10,
                 fontweight="bold")
    style_3d_axis(ax, "")
    plt.tight_layout()

    safe_name = f"{sanitize_for_filename(dataset_name)}_{sanitize_for_filename(mouse_name)}_{sanitize_for_filename(session_name)}_{sanitize_for_filename(depth_type)}_sigma{sigma}_trial_meb_umap"
    fig.savefig(os.path.join(save_dir, safe_name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, safe_name + ".svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# PER-SESSION ANALYSIS LOOP
# =============================================================================

def analyze_single_session(data_dict, dataset_name, mouse_name, session_name, depth_type):
    if SIGNAL_KEY not in data_dict:
        raise KeyError(f"Missing '{SIGNAL_KEY}' in data dictionary.")

    signal = np.asarray(data_dict[SIGNAL_KEY], dtype=float)
    if signal.ndim != 2:
        raise ValueError(f"Signal must be 2D (T, N), got shape {signal.shape}")

    signal = maybe_subsample_neurons(signal, n_neurons=N_NEURONS_SUBSAMPLE, random_state=SUBSAMPLE_RANDOM_STATE)
    n_time = signal.shape[0]

    beh = build_behavior_dict(data_dict, target_len=n_time)

    # Recompute trial IDs dynamically using position bounds
    trial_id_full = prepare_trial_id(beh, target_len=n_time)
    beh["trial_id"] = trial_id_full

    valid_mask = get_valid_time_mask(beh, n_time)
    valid_idx = np.where(valid_mask)[0]

    signal_valid = signal[valid_idx, :]
    beh_valid = {k: (v[valid_idx] if v.ndim == 1 else v[valid_idx, :]) for k, v in beh.items()}
    trial_id_valid = beh_valid["trial_id"]

    if USE_SIGNAL_OUTLIER_FILTER:
        _, signal_idx = filter_noisy_outliers(signal_valid, distance_percentile=OUTLIER_DISTANCE_PERCENTILE,
                                              keep_percentile=OUTLIER_KEEP_PERCENTILE)
    else:
        signal_idx = np.arange(signal_valid.shape[0])

    session_results = {
        "metadata": {
            "dataset_name": dataset_name, "mouse_name": mouse_name,
            "session_name": session_name, "depth_type": depth_type,
            "n_timepoints_raw": int(n_time), "n_neurons": int(signal.shape[1])
        },
        "meb": {}, "si": {}, "tda": {}
    }

    umap_plots_dir = os.path.join(SAVE_ROOT, "umap_filtered_plots")
    trial_meb_dir = os.path.join(SAVE_ROOT, "trial_meb_plots")
    os.makedirs(umap_plots_dir, exist_ok=True)
    os.makedirs(trial_meb_dir, exist_ok=True)

    for sigma in FILTER_SIGMAS:
        sig_f = smooth_calcium_signals(signal_valid, sigma=sigma)

        meb_val, _ = compute_meb_expansion(sig_f, trial_id_valid)
        session_results["meb"][str(sigma)] = {"expansion": float(meb_val) if np.isfinite(meb_val) else np.nan}

        session_results["si"][str(sigma)] = {}
        for beh_name in SI_BEHAVIORS:
            if beh_name not in beh_valid:
                continue
            beh_val = beh_valid[beh_name]
            params = get_si_params_for_behavior(beh_name, beh_val)
            try:
                si, process_info, overlap_mat, _ = compute_structure_index(
                    sig_f[signal_idx], beh_val[signal_idx], **params
                )
            except Exception as e:
                si = np.nan
                process_info, overlap_mat = None, None

            session_results["si"][str(sigma)][beh_name] = {
                "si": float(si) if np.isfinite(si) else np.nan,
                "process_info": process_info,
                "overlap_mat": overlap_mat
            }

        try:
            umap_model = umap.UMAP(n_components=N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                                   random_state=UMAP_RANDOM_STATE)
            umap_emb = umap_model.fit_transform(sig_f)

            pos_1d = beh_valid.get("pos", np.zeros(len(valid_idx)))
            mov_dir = beh_valid.get("dir", np.zeros(len(valid_idx)))
            time_1d = beh_valid.get("time", np.arange(len(valid_idx)))

            plot_umap_triplet(
                umap_emb=umap_emb, pos_1d=pos_1d, mov_dir=mov_dir, time_1d=time_1d,
                dataset_name=dataset_name, mouse_name=mouse_name, session_name=session_name,
                depth_type=depth_type, sigma=sigma, save_dir=umap_plots_dir
            )

            plot_trial_centroid_meb_umap(
                umap_emb=umap_emb, trial_id=trial_id_valid,
                dataset_name=dataset_name, mouse_name=mouse_name, session_name=session_name,
                depth_type=depth_type, sigma=sigma, save_dir=trial_meb_dir
            )

            D_umap = pairwise_distances(umap_emb)
            noise_u = filter_noisy_outliers(umap_emb, D=D_umap)
            clean_u = umap_emb[~noise_u]
            n_sub_u = min(N_SUBSAMPLE_UMAP, clean_u.shape[0])
            sub_u = clean_u[np.random.permutation(clean_u.shape[0])[:n_sub_u]]

            dgms_umap = tda(sub_u, maxdim=2, thresh=float(np.nanpercentile(pairwise_distances(sub_u), 98)))["dgms"]
            max_l1_u, sec_l1_u, ratio_u = extract_h1_lifetime_metrics(dgms_umap)

            session_results["tda"][str(sigma)] = {
                "umap_h1_max_lifetime": max_l1_u,
                "umap_h1_second_lifetime": sec_l1_u,
                "umap_h1_ratio": ratio_u,
                "diagrams": dgms_umap
            }
        except Exception as e:
            print(f"  -> UMAP/TDA warning for sigma={sigma}: {e}")

        gc.collect()

    return session_results


# =============================================================================
# MASTER EXECUTION
# =============================================================================

def load_master_index():
    df = pd.read_csv(MASTER_INDEX_CSV)
    dataset_col = get_column(df, ["dataset_name"])
    mouse_col = get_column(df, ["mouse_name"])
    session_col = get_column(df, ["session_name"])
    depth_col = get_column(df, ["depth_type"])
    path_col = get_column(df, ["standardized_file_path"])

    df = df.rename(columns={
        dataset_col: "dataset_name", mouse_col: "mouse_name",
        session_col: "session_name", depth_col: "depth_type",
        path_col: "standardized_file_path"
    })
    return df


def main():
    master_df = load_master_index()

    if DATASETS_TO_INCLUDE is not None:
        master_df = master_df[master_df["dataset_name"].isin(DATASETS_TO_INCLUDE)].copy()
    if DEPTHS_TO_INCLUDE is not None:
        master_df = master_df[master_df["depth_type"].isin(DEPTHS_TO_INCLUDE)].copy()

    master_df = master_df.reset_index(drop=True)
    print(f"Found {len(master_df)} sessions to process for datasets: {DATASETS_TO_INCLUDE}")

    all_meb_rows, all_si_rows, all_tda_rows = [], [], []

    for i, row in master_df.iterrows():
        dataset_name, mouse_name, session_name, depth_type, fpath = (
            row["dataset_name"], row["mouse_name"], row["session_name"], row["depth_type"],
            row["standardized_file_path"]
        )

        if not os.path.exists(fpath):
            alt_path = fpath.replace("/home/melisamc/Documents/time_project_standarization/", BASE_STD_DIR + "/")
            if os.path.exists(alt_path):
                fpath = alt_path

        print(f"[{i + 1}/{len(master_df)}] {dataset_name} | {mouse_name} | {session_name} | {depth_type}")

        try:
            data_dict = load_pickle_file(fpath)
            res = analyze_single_session(data_dict, dataset_name, mouse_name, session_name, depth_type)
        except Exception as e:
            print(f"  -> Error processing session: {e}")
            continue

        for sigma_str, val in res["meb"].items():
            sigma = int(sigma_str)
            all_meb_rows.append({
                "dataset_name": dataset_name, "mouse_name": mouse_name, "session_name": session_name,
                "depth_type": depth_type, "filter_sigma": sigma, "filter_time_sec": sigma / SAMPLING_RATE,
                "meb_expansion": val["expansion"]
            })

        for sigma_str, beh_dict in res["si"].items():
            sigma = int(sigma_str)
            for beh_name, val in beh_dict.items():
                all_si_rows.append({
                    "dataset_name": dataset_name, "mouse_name": mouse_name, "session_name": session_name,
                    "depth_type": depth_type, "behavioral_label": beh_name, "filter_sigma": sigma,
                    "filter_time_sec": sigma / SAMPLING_RATE, "si": val["si"]
                })

        for sigma_str, tda_val in res.get("tda", {}).items():
            sigma = int(sigma_str)
            all_tda_rows.append({
                "dataset_name": dataset_name, "mouse_name": mouse_name, "session_name": session_name,
                "depth_type": depth_type, "filter_sigma": sigma, "filter_time_sec": sigma / SAMPLING_RATE,
                "umap_h1_max_lifetime": tda_val["umap_h1_max_lifetime"],
                "umap_h1_second_lifetime": tda_val["umap_h1_second_lifetime"],
                "umap_h1_ratio": tda_val["umap_h1_ratio"]
            })

    pd.DataFrame(all_meb_rows).to_csv(os.path.join(SAVE_ROOT, "filter_analysis_meb_all_sessions.csv"), index=False)
    pd.DataFrame(all_si_rows).to_csv(os.path.join(SAVE_ROOT, "filter_analysis_si_all_sessions.csv"), index=False)
    pd.DataFrame(all_tda_rows).to_csv(os.path.join(SAVE_ROOT, "filter_analysis_tda_all_sessions.csv"), index=False)

    print(f"\n[OK] Analysis complete! All results and plots saved to: {SAVE_ROOT}")


if __name__ == "__main__":
    main()