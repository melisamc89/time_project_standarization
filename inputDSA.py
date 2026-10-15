#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
InputDSA / DSA analysis on standardized neural datasets

Analyses
--------
1) Learning dataset
   - For each mouse, sort sessions chronologically using session_name format:
         YYYYMMDD_HHMMSS
   - Keep the first 4 sessions
   - Crop all 4 sessions to the minimum shared duration within mouse
   - Preprocess:
         z-score neurons across time
         PCA to 99% explained variance
         3D UMAP using project parameters
   - Compute pairwise distances for all session pairs
   - Save per-mouse matrices and pooled summaries

2) lt_rot dataset
   - Run LT and ROT separately
   - Use only depth_type in {'deep', 'superficial'}
   - Crop all items within LT to the common minimum duration
   - Crop all items within ROT to the common minimum duration
   - Preprocess:
         z-score neurons across time
         PCA to 99% explained variance
         3D UMAP using project parameters
   - Compute pairwise distances and group into:
         deep_deep, deep_sup, sup_sup

Important
---------
The functions:
    compute_state_distance(...)
    compute_inputdsa_distances(...)
are still placeholders for now.

They let the pipeline run structurally, but they are NOT the final DSA/InputDSA metric.
Replace them with the exact DSA/InputDSA package calls once you are ready.

Project assumptions
-------------------
Master CSV:
    /home/melisamc/Documents/time_project_standarization/data/metadata/master_dataset_index.csv

Expected master CSV columns:
    dataset_name
    mouse_name
    session_name
    depth_type
    standardized_file_path

Expected contents of each standardized pickle:
    signal      -> array (T, N)
    pos         -> array (T,)
    speed       -> array (T,)
    mov_dir     -> array (T,)
Optional:
    time
    trial_id
    inner_time
"""

import os
import re
import json
import pickle
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import umap


# =============================================================================
# PARAMETERS
# =============================================================================

# Paths
BASE_DIR = "/home/melisamc/Documents/time_project_standarization"
MASTER_CSV = os.path.join(BASE_DIR, "data", "metadata", "master_dataset_index.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "analysis", "inputdsa_umap")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(OUTPUT_DIR, f"inputdsa_umap_{TIMESTAMP}")

# Dataset names
LTROT_DATASET_NAME = "lt_rot"

# UMAP parameters used previously in the project
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

# PCA
PCA_EXPLAINED_VAR = 0.99

# Signal preprocessing
ZSCORE_NEURONS_ACROSS_TIME = True

# Surrogate inputs
USE_INPUT_FEATURES = ["pos", "speed", "mov_dir"]
NORMALIZE_POS_TO_01 = True
ZSCORE_SPEED = True

# Save figures
SAVE_PNG = True
SAVE_SVG = True
DPI = 200

EPS = 1e-12


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_txt(text, path):
    with open(path, "w") as f:
        f.write(text)


def save_figure(fig, out_base):
    if SAVE_PNG:
        fig.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def load_master_csv(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = [
        "dataset_name",
        "mouse_name",
        "session_name",
        "depth_type",
        "standardized_file_path",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"MASTER CSV missing required columns: {missing}")

    return df


def load_standardized_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


# =============================================================================
# DATASET / SESSION PARSING
# =============================================================================

def get_learning_rows(master_df):
    """
    Flexible selection of the learning dataset.
    First exact 'learning', then any dataset_name containing 'learn'.
    """
    df = master_df.copy()
    ds = df["dataset_name"].astype(str)

    exact = df[ds.str.lower() == "learning"].copy()
    if len(exact) > 0:
        print("[LEARNING] Using exact dataset_name='learning'")
        return exact

    fuzzy = df[ds.str.contains("learn", case=False, na=False)].copy()
    if len(fuzzy) > 0:
        print("[LEARNING] Using fuzzy dataset_name match containing 'learn'")
        print(fuzzy["dataset_name"].value_counts(dropna=False))
        return fuzzy

    print("[LEARNING] No learning dataset found.")
    print("[LEARNING] Available dataset_name values:")
    print(df["dataset_name"].value_counts(dropna=False))
    return df.iloc[0:0].copy()


def parse_learning_session_datetime(session_name):
    """
    Parse session_name of the form:
        YYYYMMDD_HHMMSS
    example:
        20150228_174527
    """
    s = str(session_name).strip()
    return datetime.strptime(s, "%Y%m%d_%H%M%S")


def assign_learning_session_order(sub_df):
    """
    For one mouse:
    - parse session_name as datetime
    - sort chronologically
    - assign session_idx = 1..N
    """
    sub = sub_df.copy()
    sub["session_dt"] = sub["session_name"].apply(parse_learning_session_datetime)
    sub = sub.sort_values("session_dt").reset_index(drop=True)
    sub["session_idx"] = np.arange(1, len(sub) + 1)
    return sub


def infer_ltrot_condition(session_name):
    s = str(session_name).lower()
    if "rot" in s:
        return "rot"
    if "lt" in s:
        return "lt"
    return None


# =============================================================================
# STANDARDIZED DATA ACCESS
# =============================================================================

def get_signal_array(data_dict):
    if "signal" not in data_dict:
        raise KeyError("Standardized file missing key 'signal'")
    X = np.asarray(data_dict["signal"])
    if X.ndim != 2:
        raise ValueError(f"'signal' must be 2D, got shape {X.shape}")
    return X


def get_behavior_vector(data_dict, key):
    if key not in data_dict:
        raise KeyError(f"Standardized file missing key '{key}'")

    v = np.asarray(data_dict[key])

    # Common case: already 1D
    if v.ndim == 1:
        return v

    # Common standardized position case: (T, 2) -> use first column
    if v.ndim == 2:
        if v.shape[1] >= 1:
            return v[:, 0]
        else:
            raise ValueError(f"Behavior key '{key}' has invalid 2D shape {v.shape}")

    # Squeezed fallback
    v2 = np.squeeze(v)
    if v2.ndim == 1:
        return v2

    raise ValueError(f"Behavior key '{key}' must be 1D or 2D with time in axis 0, got shape {v.shape}")


def crop_time_series_dict(data_dict, target_len):
    """
    Crop along the first dimension only when the array starts with time.
    This is safe for typical standardized variables:
        signal: (T, N)
        pos, speed, mov_dir, time, trial_id, inner_time: (T,)
    """
    out = {}
    for k, v in data_dict.items():
        try:
            arr = np.asarray(v)
        except Exception:
            out[k] = v
            continue

        if arr.ndim >= 1 and arr.shape[0] >= target_len:
            out[k] = arr[:target_len]
        else:
            out[k] = v
    return out


# =============================================================================
# PREPROCESSING
# =============================================================================

def zscore_neurons_over_time(X):
    """
    X shape: (T, N)
    z-score each neuron across time.
    """
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < EPS, 1.0, sd)
    return (X - mu) / sd


def fit_pca_99(X):
    """
    X shape: (T, N)
    PCA retaining 99% explained variance.
    """
    pca = PCA(n_components=PCA_EXPLAINED_VAR, svd_solver="full")
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def fit_umap_3d(X):
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    X_umap = reducer.fit_transform(X)
    return X_umap, reducer


def normalize_to_01(x):
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if mx - mn < EPS:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def zscore_1d(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd < EPS:
        sd = 1.0
    return (x - mu) / sd


def build_input_matrix(data_dict):
    """
    Surrogate inputs:
        [pos, speed, mov_dir]
    """
    cols = []

    if "pos" in USE_INPUT_FEATURES:
        pos = get_behavior_vector(data_dict, "pos").astype(float)
        if NORMALIZE_POS_TO_01:
            pos = normalize_to_01(pos)
        cols.append(pos[:, None])

    if "speed" in USE_INPUT_FEATURES:
        speed = get_behavior_vector(data_dict, "speed").astype(float)
        if ZSCORE_SPEED:
            speed = zscore_1d(speed)
        cols.append(speed[:, None])

    if "mov_dir" in USE_INPUT_FEATURES:
        mov_dir = get_behavior_vector(data_dict, "mov_dir").astype(float)
        cols.append(mov_dir[:, None])

    if len(cols) == 0:
        raise ValueError("No input features selected for surrogate input matrix.")

    U = np.concatenate(cols, axis=1)
    return U


def preprocess_single_item(data_dict):
    """
    Returns:
        X_raw
        X_proc
        X_pca
        X_umap
        pca_model
        umap_model
        U
    """
    X_raw = get_signal_array(data_dict)

    X_proc = X_raw.copy()
    if ZSCORE_NEURONS_ACROSS_TIME:
        X_proc = zscore_neurons_over_time(X_proc)

    X_pca, pca_model = fit_pca_99(X_proc)
    X_umap, umap_model = fit_umap_3d(X_pca)
    U = build_input_matrix(data_dict)

    if len(X_umap) != len(U):
        raise ValueError(
            f"Length mismatch between state and input: {len(X_umap)} vs {len(U)}"
        )

    return {
        "X_raw": X_raw,
        "X_proc": X_proc,
        "X_pca": X_pca,
        "X_umap": X_umap,
        "pca_model": pca_model,
        "umap_model": umap_model,
        "U": U,
    }


# =============================================================================
# PLACEHOLDER DISTANCE FUNCTIONS
# =============================================================================

def compute_state_distance(X1, X2):
    """
    Placeholder state comparison.

    Replace this with the real DSA state comparison.

    Current placeholder:
    - center each trajectory
    - normalize
    - align with orthogonal Procrustes
    - return residual error
    """
    A = np.asarray(X1, dtype=float)
    B = np.asarray(X2, dtype=float)

    if A.shape != B.shape:
        raise ValueError(f"State shapes must match, got {A.shape} and {B.shape}")

    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)

    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)

    if normA < EPS or normB < EPS:
        return np.nan

    A = A / normA
    B = B / normB

    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    A_aligned = A @ R

    err = np.linalg.norm(A_aligned - B) / np.sqrt(A.shape[0])
    return float(err)


def compute_inputdsa_distances(X1, U1, X2, U2):
    """
    Placeholder InputDSA comparison.

    Replace this with the real InputDSA package calls.

    Returns:
        state_dist
        control_dist
        joint_dist
    """
    state_dist = compute_state_distance(X1, X2)

    C1 = np.cov(U1.T)
    C2 = np.cov(U2.T)

    if np.ndim(C1) == 0:
        C1 = np.array([[C1]])
    if np.ndim(C2) == 0:
        C2 = np.array([[C2]])

    control_dist = np.linalg.norm(C1 - C2)
    joint_dist = state_dist + control_dist

    return {
        "state_dist": float(state_dist),
        "control_dist": float(control_dist),
        "joint_dist": float(joint_dist),
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_distance_matrix(matrix, labels, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_group_distributions(df, group_col, value_col, title):
    groups = list(df[group_col].dropna().unique())
    groups_sorted = sorted(groups)
    data = [df.loc[df[group_col] == g, value_col].dropna().values for g in groups_sorted]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=groups_sorted, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(value_col)
    fig.tight_layout()
    return fig


# =============================================================================
# LEARNING ANALYSIS
# =============================================================================

def run_learning_analysis(master_df, out_dir):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "tables"))
    ensure_dir(os.path.join(out_dir, "figures"))
    ensure_dir(os.path.join(out_dir, "pkl"))

    df = get_learning_rows(master_df)
    if df.empty:
        print("[LEARNING] No learning rows found.")
        return

    print("[LEARNING] Initial rows found:", len(df))
    print("[LEARNING] Dataset names:")
    print(df["dataset_name"].value_counts(dropna=False))

    print("[LEARNING] Example session names:")
    print(df["session_name"].dropna().astype(str).unique()[:10])

    all_rows = []
    per_mouse_payload = {}

    animals = sorted(df["mouse_name"].dropna().unique())
    print(f"[LEARNING] Found {len(animals)} animals")

    for mouse in animals:
        sub = df[df["mouse_name"] == mouse].copy()

        try:
            sub = assign_learning_session_order(sub)
        except Exception as e:
            print(f"[LEARNING] Skipping {mouse}: could not parse session datetimes. Error: {e}")
            continue

        if len(sub) < 4:
            print(f"[LEARNING] Skipping {mouse}: fewer than 4 sessions available.")
            continue

        # Keep earliest 4 sessions only
        sub = sub.iloc[:4].copy()
        sub["session_idx"] = np.arange(1, 5)

        print(f"[LEARNING] Mouse={mouse}, ordered sessions:")
        for _, row in sub.iterrows():
            print(f"    session_idx={row['session_idx']}  session_name={row['session_name']}")

        records = []
        lengths = []

        for _, row in sub.iterrows():
            pkl_path = row["standardized_file_path"]
            data = load_standardized_pickle(pkl_path)
            X = get_signal_array(data)
            lengths.append(X.shape[0])
            records.append((row, data))

        T_min = int(np.min(lengths))
        print(f"[LEARNING] Mouse={mouse}, cropping to T_min={T_min}")

        proc_items = {}
        session_labels = []

        for row, data in records:
            sess = int(row["session_idx"])
            cropped = crop_time_series_dict(data, T_min)
            proc = preprocess_single_item(cropped)
            proc_items[sess] = proc
            session_labels.append(f"S{sess}")

        sessions = [1, 2, 3, 4]
        n = len(sessions)

        state_mat = np.full((n, n), np.nan, dtype=float)
        control_mat = np.full((n, n), np.nan, dtype=float)
        joint_mat = np.full((n, n), np.nan, dtype=float)

        for i, j in combinations(range(n), 2):
            s1 = sessions[i]
            s2 = sessions[j]

            A = proc_items[s1]
            B = proc_items[s2]

            d = compute_inputdsa_distances(
                A["X_umap"], A["U"],
                B["X_umap"], B["U"]
            )

            state_mat[i, j] = state_mat[j, i] = d["state_dist"]
            control_mat[i, j] = control_mat[j, i] = d["control_dist"]
            joint_mat[i, j] = joint_mat[j, i] = d["joint_dist"]

            all_rows.append({
                "mouse_name": mouse,
                "session_a": s1,
                "session_b": s2,
                "session_name_a": sub.iloc[i]["session_name"],
                "session_name_b": sub.iloc[j]["session_name"],
                "lag": abs(s2 - s1),
                "T_min": T_min,
                "state_dist": d["state_dist"],
                "control_dist": d["control_dist"],
                "joint_dist": d["joint_dist"],
            })

        per_mouse_payload[mouse] = {
            "T_min": T_min,
            "session_info": sub[["session_idx", "session_name", "session_dt"]].copy(),
            "proc_items": proc_items,
            "state_mat": state_mat,
            "control_mat": control_mat,
            "joint_mat": joint_mat,
            "session_labels": session_labels,
        }

        fig = plot_distance_matrix(state_mat, session_labels, f"{mouse} - state distance")
        save_figure(fig, os.path.join(out_dir, "figures", f"{mouse}_state_matrix"))

        fig = plot_distance_matrix(control_mat, session_labels, f"{mouse} - control distance")
        save_figure(fig, os.path.join(out_dir, "figures", f"{mouse}_control_matrix"))

        fig = plot_distance_matrix(joint_mat, session_labels, f"{mouse} - joint distance")
        save_figure(fig, os.path.join(out_dir, "figures", f"{mouse}_joint_matrix"))

    if len(all_rows) == 0:
        print("[LEARNING] No valid animals found.")
        return

    dist_df = pd.DataFrame(all_rows)
    dist_df.to_csv(
        os.path.join(out_dir, "tables", "learning_pairwise_distances.csv"),
        index=False
    )

    pair_summary = (
        dist_df
        .assign(pair=lambda x: x["session_a"].astype(str) + "-" + x["session_b"].astype(str))
        .groupby("pair", as_index=False)[["state_dist", "control_dist", "joint_dist"]]
        .mean()
    )
    pair_summary.to_csv(
        os.path.join(out_dir, "tables", "learning_summary_by_pair.csv"),
        index=False
    )

    lag_summary = (
        dist_df
        .groupby("lag", as_index=False)[["state_dist", "control_dist", "joint_dist"]]
        .mean()
    )
    lag_summary.to_csv(
        os.path.join(out_dir, "tables", "learning_summary_by_lag.csv"),
        index=False
    )

    dist_df["pair"] = dist_df["session_a"].astype(str) + "-" + dist_df["session_b"].astype(str)

    fig = plot_group_distributions(
        dist_df, "pair", "state_dist", "Learning - state distance by pair"
    )
    save_figure(fig, os.path.join(out_dir, "figures", "learning_state_by_pair"))

    fig = plot_group_distributions(
        dist_df, "lag", "state_dist", "Learning - state distance by lag"
    )
    save_figure(fig, os.path.join(out_dir, "figures", "learning_state_by_lag"))

    fig = plot_group_distributions(
        dist_df, "pair", "control_dist", "Learning - control distance by pair"
    )
    save_figure(fig, os.path.join(out_dir, "figures", "learning_control_by_pair"))

    fig = plot_group_distributions(
        dist_df, "lag", "control_dist", "Learning - control distance by lag"
    )
    save_figure(fig, os.path.join(out_dir, "figures", "learning_control_by_lag"))

    fig = plot_group_distributions(
        dist_df, "pair", "joint_dist", "Learning - joint distance by pair"
    )
    save_figure(fig, os.path.join(out_dir, "figures", "learning_joint_by_pair"))

    fig = plot_group_distributions(
        dist_df, "lag", "joint_dist", "Learning - joint distance by lag"
    )
    save_figure(fig, os.path.join(out_dir, "figures", "learning_joint_by_lag"))

    with open(os.path.join(out_dir, "pkl", "learning_payload.pkl"), "wb") as f:
        pickle.dump(per_mouse_payload, f)

    print(f"[LEARNING] Saved results to: {out_dir}")


# =============================================================================
# LT / ROT DEEP-SUP ANALYSIS
# =============================================================================

def assign_pair_class(depth_a, depth_b):
    da = str(depth_a).lower()
    db = str(depth_b).lower()

    if da == "deep" and db == "deep":
        return "deep_deep"
    if da == "superficial" and db == "superficial":
        return "sup_sup"
    if set([da, db]) == set(["deep", "superficial"]):
        return "deep_sup"
    return None


def run_ltrot_depth_analysis(master_df, out_dir):
    ensure_dir(out_dir)

    df = master_df.loc[master_df["dataset_name"] == LTROT_DATASET_NAME].copy()
    if df.empty:
        print(f"[LTROT] No rows found for dataset_name='{LTROT_DATASET_NAME}'")
        return

    df["condition"] = df["session_name"].apply(infer_ltrot_condition)
    df = df[df["condition"].isin(["lt", "rot"])].copy()
    df = df[df["depth_type"].isin(["deep", "superficial"])].copy()

    if df.empty:
        print("[LTROT] No LT/ROT deep-superficial rows found.")
        return

    for condition in ["lt", "rot"]:
        cond_dir = os.path.join(out_dir, condition)
        ensure_dir(cond_dir)
        ensure_dir(os.path.join(cond_dir, "tables"))
        ensure_dir(os.path.join(cond_dir, "figures"))
        ensure_dir(os.path.join(cond_dir, "pkl"))

        sub = df[df["condition"] == condition].copy()
        if sub.empty:
            print(f"[LTROT-{condition.upper()}] No rows found.")
            continue

        records = []
        lengths = []

        for _, row in sub.iterrows():
            pkl_path = row["standardized_file_path"]
            data = load_standardized_pickle(pkl_path)
            X = get_signal_array(data)
            lengths.append(X.shape[0])
            records.append((row, data))

        T_min = int(np.min(lengths))
        print(f"[LTROT-{condition.upper()}] Cropping all items to T_min={T_min}")

        items = []
        for row, data in records:
            cropped = crop_time_series_dict(data, T_min)
            proc = preprocess_single_item(cropped)

            items.append({
                "mouse_name": row["mouse_name"],
                "session_name": row["session_name"],
                "depth_type": row["depth_type"],
                "condition": condition,
                "T_min": T_min,
                "proc": proc,
            })

        rows_out = []
        payload = {
            "T_min": T_min,
            "items": items,
        }

        for i, j in combinations(range(len(items)), 2):
            A = items[i]
            B = items[j]

            pair_class = assign_pair_class(A["depth_type"], B["depth_type"])
            if pair_class is None:
                continue

            d = compute_inputdsa_distances(
                A["proc"]["X_umap"], A["proc"]["U"],
                B["proc"]["X_umap"], B["proc"]["U"]
            )

            rows_out.append({
                "condition": condition,
                "mouse_a": A["mouse_name"],
                "mouse_b": B["mouse_name"],
                "session_a": A["session_name"],
                "session_b": B["session_name"],
                "depth_a": A["depth_type"],
                "depth_b": B["depth_type"],
                "same_animal": A["mouse_name"] == B["mouse_name"],
                "pair_class": pair_class,
                "T_min": T_min,
                "state_dist": d["state_dist"],
                "control_dist": d["control_dist"],
                "joint_dist": d["joint_dist"],
            })

        if len(rows_out) == 0:
            print(f"[LTROT-{condition.upper()}] No valid pairwise rows.")
            continue

        dist_df = pd.DataFrame(rows_out)
        dist_df.to_csv(
            os.path.join(cond_dir, "tables", f"{condition}_pairwise_distances.csv"),
            index=False
        )

        summary_df = (
            dist_df
            .groupby("pair_class", as_index=False)[["state_dist", "control_dist", "joint_dist"]]
            .mean()
        )
        summary_df.to_csv(
            os.path.join(cond_dir, "tables", f"{condition}_summary_by_pairclass.csv"),
            index=False
        )

        fig = plot_group_distributions(
            dist_df, "pair_class", "state_dist", f"{condition.upper()} - state distance"
        )
        save_figure(fig, os.path.join(cond_dir, "figures", f"{condition}_state_by_pairclass"))

        fig = plot_group_distributions(
            dist_df, "pair_class", "control_dist", f"{condition.upper()} - control distance"
        )
        save_figure(fig, os.path.join(cond_dir, "figures", f"{condition}_control_by_pairclass"))

        fig = plot_group_distributions(
            dist_df, "pair_class", "joint_dist", f"{condition.upper()} - joint distance"
        )
        save_figure(fig, os.path.join(cond_dir, "figures", f"{condition}_joint_by_pairclass"))

        with open(os.path.join(cond_dir, "pkl", f"{condition}_payload.pkl"), "wb") as f:
            pickle.dump(payload, f)

        print(f"[LTROT-{condition.upper()}] Saved results to: {cond_dir}")


# =============================================================================
# RUN METADATA
# =============================================================================

def save_run_metadata():
    meta = {
        "timestamp": TIMESTAMP,
        "base_dir": BASE_DIR,
        "master_csv": MASTER_CSV,
        "ltrot_dataset_name": LTROT_DATASET_NAME,
        "umap_n_components": UMAP_N_COMPONENTS,
        "umap_n_neighbors": UMAP_N_NEIGHBORS,
        "umap_min_dist": UMAP_MIN_DIST,
        "umap_random_state": UMAP_RANDOM_STATE,
        "pca_explained_var": PCA_EXPLAINED_VAR,
        "zscore_neurons_across_time": ZSCORE_NEURONS_ACROSS_TIME,
        "use_input_features": USE_INPUT_FEATURES,
    }
    ensure_dir(RUN_DIR)
    with open(os.path.join(RUN_DIR, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ensure_dir(RUN_DIR)
    save_run_metadata()

    print(f"RUN_DIR: {RUN_DIR}")

    master_df = load_master_csv(MASTER_CSV)

    # Learning
    learning_out = os.path.join(RUN_DIR, "learning")
    run_learning_analysis(master_df, learning_out)

    # LT/ROT deep-sup
    ltrot_out = os.path.join(RUN_DIR, "lt_rot_deepsup")
    run_ltrot_depth_analysis(master_df, ltrot_out)

    print("Done.")


if __name__ == "__main__":
    main()