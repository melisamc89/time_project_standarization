#!/usr/init/env python3
# -*- coding: utf-8 -*-

"""Neural Manifold Embedding Overlaid with Behavior, Multi-Band LFP Waveform, Envelope, Phase, and CWT Power.

Applies per-neuron 1D Gaussian smoothing across FILTER_SIGMAS, computes 3D UMAP embeddings
for each session's population signal, and generates comprehensive figure suites (PNG and SVG)
for all modalities without clustering.
"""

import glob
import os
import pickle
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import umap

# =============================================================================
# Paths & Settings
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
TARGET_DATASET = "lt_egrin"

MASTER_INDEX_CSV = os.path.join(PROJECT_ROOT, "data", "metadata", "master_dataset_index.csv")
MI_CSV = os.path.join(PROJECT_ROOT, "analysis", "mutual_information_behavior_and_lfp",
                      "mutual_information_per_neuron.csv")

OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "analysis", "neural_manifold_lfp_suite_filtered_sigmas")

FILTER_SIGMAS = [0, 2, 4, 8, 16, 20]  # Adjust smoothing sigmas as needed
SAMPLING_RATE = 20.0  # Hz

UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
N_COMPONENTS = 3

N_POS_BINS = 30
MIN_COUNT_PER_BIN = 5
PLANE_TILT_DEG = 30.0
CMAP_CONTINUOUS = "jet"

# Standardized LFP & CWT Band Definitions (1:1 Matching)
LFP_BANDS_CONFIG = [
    {"key": "slow", "name": "Slow [0.1 - 1 Hz]", "unit": "Waveform (µV)", "f_low": 0.10, "f_high": 1.00},
    {"key": "delta", "name": "Delta [1 - 3 Hz]", "unit": "Waveform (µV)", "f_low": 1.00, "f_high": 4.00},
    {"key": "theta", "name": "Theta [8 - 12 Hz]", "unit": "Waveform (µV)", "f_low": 8.00, "f_high": 12.00},
    {"key": "gamma", "name": "Gamma [40 - 90 Hz]", "unit": "Waveform (µV)", "f_low": 40.00, "f_high": 90.00},
    {"key": "ripple", "name": "Ripple [100 - 250 Hz]", "unit": "Waveform (µV)", "f_low": 100.0, "f_high": 250.0},
    {"key": "mua", "name": "MUA [300 - 1000 Hz]", "unit": "Waveform (µV)", "f_low": 300.0, "f_high": 1000.0},
]


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_figure_both_formats(fig, directory, prefix, suffix):
    ensure_dir(directory)
    base_name = f"{prefix}_{suffix}"
    png_path = os.path.join(directory, f"{base_name}.png")
    svg_path = os.path.join(directory, f"{base_name}.svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")


def sanitize_for_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def correct_path(path_str):
    if not isinstance(path_str, str):
        return path_str
    if "/home/melisamc/Documents/time_project_standarization" in path_str:
        path_str = path_str.replace("/home/melisamc/Documents/time_project_standarization", PROJECT_ROOT)
    return path_str


def resolve_pkl_path(row_subset):
    for col in row_subset.index:
        val = row_subset[col]
        if pd.notna(val) and isinstance(val, str) and ".pkl" in val:
            corrected = correct_path(val)
            if os.path.isfile(corrected):
                return corrected
            basename = os.path.basename(corrected)
            matches = glob.glob(os.path.join(PROJECT_ROOT, "data", "standardized_sessions", "**", basename),
                                recursive=True)
            if matches and os.path.isfile(matches[0]):
                return matches[0]

    mouse_name = str(row_subset.get("mouse_name", "")).strip()
    depth_type = str(row_subset.get("depth_type", "")).strip()
    if mouse_name:
        matches = glob.glob(
            os.path.join(PROJECT_ROOT, "data", "standardized_sessions", TARGET_DATASET, "**", f"*{mouse_name}*.pkl"),
            recursive=True)
        for m in matches:
            if depth_type in m and os.path.isfile(m):
                return m
        if matches and os.path.isfile(matches[0]):
            return matches[0]
    return None


def smooth_calcium_signals(signal, sigma=4):
    """
    Gaussian smoothing over time (axis=0) for each neuron (axis=1).
    signal shape: (T, N)
    """
    if sigma == 0:
        return signal.copy()

    smoothed = np.zeros_like(signal, dtype=float)
    for i in range(signal.shape[1]):
        smoothed[:, i] = gaussian_filter1d(signal[:, i], sigma=sigma)
    return smoothed


def style_3d_axis(ax, title):
    ax.set_title(title, fontsize=9, pad=4)
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor("white")
        except Exception:
            pass
    ax.set_xlabel("dim 1", labelpad=-7, fontsize=7)
    ax.set_ylabel("dim 2", labelpad=-7, fontsize=7)
    ax.set_zlabel("dim 3", labelpad=-7, fontsize=7)
    ax.tick_params(labelsize=6)


def get_direction_colors():
    cmap = ListedColormap(["blue", "gray", "lightblue"])  # -1, 0, 1
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    return cmap, norm


def align_array_length(arr, target_len):
    if arr is None:
        return None
    arr = np.asarray(arr).reshape(-1)
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), mode="edge")


def compute_position_direction_plane_normal(embedding, pos_1d, mov_dir, n_pos_bins=30, min_count_per_bin=5):
    embedding = np.asarray(embedding)
    pos_1d = np.asarray(pos_1d).reshape(-1)
    mov_dir = np.asarray(mov_dir).reshape(-1)

    valid_mask = np.isfinite(pos_1d) & np.isfinite(mov_dir)
    for d in range(3):
        valid_mask &= np.isfinite(embedding[:, d])

    embedding = embedding[valid_mask]
    pos_1d = pos_1d[valid_mask]
    mov_dir = mov_dir[valid_mask]

    if embedding.shape[1] != 3:
        raise ValueError("Expected a 3D embedding.")

    pos_min, pos_max = np.min(pos_1d), np.max(pos_1d)
    if pos_max <= pos_min:
        return 30.0, 45.0

    bin_edges = np.linspace(pos_min, pos_max, n_pos_bins + 1)
    centroids = []
    for d in [-1, 1]:
        mask_d = mov_dir == d
        if np.sum(mask_d) == 0:
            continue
        pos_d, emb_d = pos_1d[mask_d], embedding[mask_d]
        bin_ids = np.digitize(pos_d, bin_edges[1:-1], right=False)
        for b in range(n_pos_bins):
            idx = np.where(bin_ids == b)[0]
            if len(idx) >= min_count_per_bin:
                centroids.append(np.mean(emb_d[idx], axis=0))

    centroids = np.asarray(centroids)
    if centroids.shape[0] < 3:
        return 30.0, 45.0

    center = np.mean(centroids, axis=0)
    pca = PCA(n_components=3)
    pca.fit(centroids - center)
    normal = pca.components_[2]
    normal /= np.linalg.norm(normal)

    x, y, z = normal
    azim = np.degrees(np.arctan2(y, x))
    elev = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2))) - PLANE_TILT_DEG
    return float(elev), float(azim)


# =============================================================================
# Feature Extraction from Standardized Session
# =============================================================================

def extract_lfp_modalities(std_data, target_len):
    waveforms = {}
    envelopes = {}
    phases = {}
    cwt_bands = {}

    cwt_matrix = std_data.get("cwt")
    cwt_freqs = std_data.get("cwt_freqs")
    if cwt_matrix is not None:
        cwt_matrix = np.asarray(cwt_matrix)
        if cwt_matrix.shape[0] != target_len:
            cwt_matrix = align_array_length(cwt_matrix, target_len) if cwt_matrix.ndim == 1 else \
                np.pad(cwt_matrix, ((0, max(0, target_len - cwt_matrix.shape[0])), (0, 0)), mode="edge")[:target_len, :]
    if cwt_freqs is not None:
        cwt_freqs = np.asarray(cwt_freqs).reshape(-1)

    for cfg in LFP_BANDS_CONFIG:
        bkey = cfg["key"]
        waveforms[bkey] = align_array_length(std_data.get(f"lfp_{bkey}_20hz"), target_len)

        env_arr = std_data.get(f"lfp_{bkey}_hilbert_20hz")
        if env_arr is None and waveforms[bkey] is not None:
            env_arr = np.abs(waveforms[bkey])
        envelopes[bkey] = align_array_length(env_arr, target_len)

        phases[bkey] = align_array_length(std_data.get(f"lfp_{bkey}_phase_20hz"), target_len)

        if cwt_matrix is not None and cwt_freqs is not None:
            f_low, f_high = cfg["f_low"], cfg["f_high"]
            idx_range = np.where((cwt_freqs >= f_low) & (cwt_freqs <= f_high))[0]
            if len(idx_range) == 0:
                nearest_idx = np.argmin(np.abs(cwt_freqs - 0.5 * (f_low + f_high)))
                idx_range = np.array([nearest_idx])

            mean_power = np.nanmean(cwt_matrix[:, idx_range], axis=1)
            cwt_bands[bkey] = align_array_length(10.0 * np.log10(np.maximum(mean_power, 1e-12)), target_len)
        else:
            cwt_bands[bkey] = None

    return waveforms, envelopes, phases, cwt_bands


def plot_single_feature_manifold(
        feat_title,
        feat_vals,
        unit_label,
        cmap_arg,
        is_div,
        is_phase,
        manifold_full,
        n_full,
        behavior,
        metadata,
        elev,
        azim,
        sigma,
):
    fig = plt.figure(figsize=(6, 5.0))
    ax0 = fig.add_subplot(111, projection="3d")

    if feat_title == "Direction":
        cmap, norm = cmap_arg
    elif feat_title == "Position":
        norm = Normalize(vmin=np.nanmin(feat_vals), vmax=np.nanmax(feat_vals))
        cmap = cmap_arg
    elif is_div:
        p1, p99 = np.nanpercentile(feat_vals, 1), np.nanpercentile(feat_vals, 99)
        limit = max(abs(p1), abs(p99))
        norm = Normalize(vmin=-limit, vmax=limit)
        cmap = cmap_arg
    else:
        p1, p99 = np.nanpercentile(feat_vals, 1), np.nanpercentile(feat_vals, 99)
        norm = Normalize(vmin=p1, vmax=max(p99, p1 + 1e-3))
        cmap = cmap_arg

    sc0 = ax0.scatter(manifold_full[:, 0], manifold_full[:, 1], manifold_full[:, 2], c=feat_vals, cmap=cmap,
                      norm=norm, s=15, alpha=0.9 if feat_title == "Direction" else 0.85)
    style_3d_axis(ax0, f"Neural Manifold [σ={sigma}] (n = {n_full} neurons)")
    ax0.view_init(elev=elev, azim=azim)
    cb0 = fig.colorbar(sc0, ax=ax0, fraction=0.035, pad=0.04)
    if feat_title == "Direction":
        cb0.set_ticks([-1, 0, 1])
        cb0.ax.set_yticklabels(["left", "stop", "right"], fontsize=6)
    else:
        cb0.set_label(unit_label, fontsize=7)

    fig.suptitle(
        f"{metadata.get('dataset_name', TARGET_DATASET)} | {metadata.get('mouse_name', '')} | {metadata.get('depth_type', '')} | Feature: {feat_title}",
        fontsize=10, y=0.98)
    plt.tight_layout()
    return fig


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    if not os.path.isfile(MASTER_INDEX_CSV) or not os.path.isfile(MI_CSV):
        print(f"[ERROR] Required CSV files missing.")
        return

    df_mi = pd.read_csv(MI_CSV)
    df_master = pd.read_csv(MASTER_INDEX_CSV, engine="python", on_bad_lines="skip")

    join_cols = ["dataset_name", "mouse_name", "session_name", "depth_type"]
    add_cols = [c for c in df_master.columns if c not in join_cols]

    df_merged = df_mi.merge(df_master[join_cols + add_cols].drop_duplicates(), on=join_cols, how="left")
    df_egrin = df_merged[df_merged["dataset_name"].astype(str).str.strip() == TARGET_DATASET].copy()

    if len(df_egrin) == 0:
        print(f"[ERROR] No sessions found for dataset {TARGET_DATASET}.")
        return

    session_cols = [c for c in ["mouse_name", "session_name", "depth_type"] if c in df_egrin.columns]
    session_groups = df_egrin.groupby(session_cols)

    print(
        f"Found {len(session_groups)} sessions for {TARGET_DATASET}. Computing filtered UMAP manifolds across sigmas {FILTER_SIGMAS}...")

    for idx, (group_keys, df_session) in enumerate(session_groups):
        row_sample = df_session.iloc[0]
        std_path = resolve_pkl_path(row_sample)

        if not std_path or not os.path.isfile(std_path):
            print(f"[WARNING] Standardized file not found for session group {group_keys}")
            continue

        std_data = load_pickle(std_path)
        meta = std_data.get("metadata", {})
        mouse_name = meta.get("mouse_name", row_sample.get("mouse_name", "mouse"))
        depth_type = meta.get("depth_type", row_sample.get("depth_type", "depth"))
        session_name = meta.get("session_name", row_sample.get("session_name", "session"))

        print(f"\n[{idx + 1}/{len(session_groups)}] Processing {mouse_name} ({depth_type})...")

        raw_signal = std_data.get("signal")
        if raw_signal is None:
            continue
        raw_signal = np.asarray(raw_signal, dtype=float)
        n_timepoints, total_neurons = raw_signal.shape

        pos = std_data.get("pos")
        pos_1d = pos[:, 0] if pos is not None and pos.ndim > 1 else pos
        pos_1d = align_array_length(pos_1d, n_timepoints)
        mov_dir = align_array_length(std_data.get("mov_dir"), n_timepoints)
        speed_1d = align_array_length(std_data.get("speed"), n_timepoints)
        time_1d = align_array_length(std_data.get("time"), n_timepoints)

        behavior_payload = {
            "pos": pos_1d,
            "mov_dir": mov_dir,
            "speed": speed_1d,
            "time": time_1d,
        }

        waveforms, envelopes, phases, cwt_bands = extract_lfp_modalities(std_data, target_len=n_timepoints)

        # Loop through each sigma, apply Gaussian filtering, compute UMAP, and save
        for sigma in FILTER_SIGMAS:
            print(f"  -> Running UMAP with Gaussian sigma = {sigma}...")

            # 1. Apply Gaussian filter per neuron
            filtered_signal = smooth_calcium_signals(raw_signal, sigma=sigma)

            # 2. Compute UMAP on filtered signal
            reducer_full = umap.UMAP(
                n_components=N_COMPONENTS,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST,
                random_state=UMAP_RANDOM_STATE
            )
            manifold_full = reducer_full.fit_transform(filtered_signal)
            n_full = total_neurons

            elev, azim = compute_position_direction_plane_normal(
                manifold_full, pos_1d, mov_dir, n_pos_bins=N_POS_BINS, min_count_per_bin=MIN_COUNT_PER_BIN
            )

            # Output directories per sigma
            sigma_base = os.path.join(OUTPUT_BASE_DIR, f"sigma_{sigma}")
            wav_dir = os.path.join(sigma_base, "lfp_waveforms")
            env_dir = os.path.join(sigma_base, "lfp_envelopes")
            pha_dir = os.path.join(sigma_base, "lfp_phases")
            cwt_dir = os.path.join(sigma_base, "cwt_power")

            prefix = f"lt_egrin_{mouse_name}_{session_name}_{depth_type}_umap_sigma{sigma}"

            behavior_features = {
                "Position": (pos_1d, "magma", False, "pos (cm)"),
                "Direction": (mov_dir, get_direction_colors(), False, "direction"),
                "Speed": (speed_1d, "viridis", False, "speed (cm/s)"),
                "Time": (time_1d, "YlGn_r", False, "time (s)"),
            }

            lfp_features_to_plot = {}
            for cfg in LFP_BANDS_CONFIG:
                bkey = cfg["key"]
                if waveforms.get(bkey) is not None:
                    lfp_features_to_plot[f"Waveform: {cfg['name']}"] = (waveforms[bkey], "seismic", True, cfg["unit"])
                if envelopes.get(bkey) is not None:
                    lfp_features_to_plot[f"Envelope: {cfg['name']}"] = (envelopes[bkey], CMAP_CONTINUOUS, False,
                                                                        cfg["unit"])
                if phases.get(bkey) is not None:
                    lfp_features_to_plot[f"Phase: {cfg['name']}"] = (phases[bkey], "twilight", False,
                                                                     "Radians (-π to π)")
                if cwt_bands.get(bkey) is not None:
                    lfp_features_to_plot[f"CWT Power: {cfg['name']}"] = (cwt_bands[bkey], CMAP_CONTINUOUS, False,
                                                                         "Power (dB)")

            all_features = {**behavior_features, **lfp_features_to_plot}

            for feat_title, (feat_vals, cmap_arg, is_div, unit_label) in all_features.items():
                if feat_vals is None or len(feat_vals) != n_timepoints:
                    continue

                is_phase = "Phase" in feat_title
                fig = plot_single_feature_manifold(
                    feat_title=feat_title,
                    feat_vals=feat_vals,
                    unit_label=unit_label,
                    cmap_arg=cmap_arg,
                    is_div=is_div,
                    is_phase=is_phase,
                    manifold_full=manifold_full,
                    n_full=n_full,
                    behavior=behavior_payload,
                    metadata=meta,
                    elev=elev,
                    azim=azim,
                    sigma=sigma,
                )

                if "Waveform" in feat_title:
                    save_figure_both_formats(fig, wav_dir, prefix, sanitize_for_filename(feat_title))
                elif "Envelope" in feat_title:
                    save_figure_both_formats(fig, env_dir, prefix, sanitize_for_filename(feat_title))
                elif "Phase" in feat_title:
                    save_figure_both_formats(fig, pha_dir, prefix, sanitize_for_filename(feat_title))
                elif "CWT" in feat_title or feat_title in ["Position", "Direction", "Speed", "Time"]:
                    save_figure_both_formats(fig, cwt_dir, prefix, sanitize_for_filename(feat_title))

    print("\n[SUCCESS] All filtered single-manifold feature suites generated successfully!")


if __name__ == "__main__":
    main()