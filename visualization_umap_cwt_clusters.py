#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Neural Manifold Embedding Overlaid with Behavior, Multi-Band LFP Waveform, Envelope, Phase, and CWT Power.

Loads standardized sessions via the master dataset index, performs neuron clustering and semantic labeling,
computes 3D UMAP embeddings for both the Complete Population and individual Functional Clusters,
and generates comprehensive figure suites (PNG and SVG) for all modalities.
"""

import glob
import os
import pickle
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "analysis", "neural_manifold_lfp_split_cluster_suite_lt_egrin")

WAVEFORM_DIR = os.path.join(OUTPUT_BASE_DIR, "lfp_waveforms")
ENVELOPE_DIR = os.path.join(OUTPUT_BASE_DIR, "lfp_envelopes")
PHASE_DIR = os.path.join(OUTPUT_BASE_DIR, "lfp_phases")
CWT_DIR = os.path.join(OUTPUT_BASE_DIR, "cwt_power")
EMBEDDING_DATA_DIR = os.path.join(OUTPUT_BASE_DIR, "embedding_data")

MIR_FEATURE_COLUMNS = [
    "mi_position",
    "mi_position_x_direction",
    "mi_direction",
    "mi_speed",
    "mi_time",
    "mi_inner_time",
    "mi_trial_id",
]

N_CLUSTERS = 3
UNASSIGNED_LABEL = 10
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 50

UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
N_COMPONENTS = 3

N_POS_BINS = 30
MIN_COUNT_PER_BIN = 5
PLANE_TILT_DEG = 30.0
CMAP_CONTINUOUS = "jet"

CLUSTER_NAMES = {
    0: "Time Cluster",
    1: "Direction Cluster",
    2: "Position Cluster",
}

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


def rowwise_zscore(X):
    Xz = zscore(X, axis=1, nan_policy="omit")
    return np.where(np.isfinite(Xz), Xz, 0.0)


def infer_intra_dataset_semantic_mapping(df_dataset, raw_labels):
    tmp = df_dataset.copy()
    tmp["raw_label"] = raw_labels

    cluster_ids = sorted(tmp["raw_label"].unique().tolist())
    if len(cluster_ids) < N_CLUSTERS:
        return {cid: cid for cid in cluster_ids}

    cluster_means = {}
    for cid in cluster_ids:
        sub = tmp[tmp["raw_label"] == cid]
        pos_score = sub["mi_position"].mean() + sub["mi_position_x_direction"].mean()
        time_score = sub["mi_time"].mean() + sub["mi_trial_id"].mean() if "mi_trial_id" in sub.columns else sub[
            "mi_time"].mean()
        cluster_means[cid] = {"pos": pos_score, "time": time_score}

    pos_cid = max(cluster_ids, key=lambda c: cluster_means[c]["pos"])
    remaining_after_pos = [c for c in cluster_ids if c != pos_cid]

    if remaining_after_pos:
        time_cid = max(remaining_after_pos, key=lambda c: cluster_means[c]["time"])
    else:
        time_cid = pos_cid

    remaining_after_time = [c for c in remaining_after_pos if c != time_cid]
    dir_cid = remaining_after_time[0] if remaining_after_time else (
        remaining_after_pos[0] if remaining_after_pos else pos_cid)

    return {pos_cid: 2, time_cid: 0, dir_cid: 1}


def cluster_session_neurons(df_session):
    X_raw = df_session[MIR_FEATURE_COLUMNS].astype(float).to_numpy()
    valid_mask = np.all(np.isfinite(X_raw), axis=1)
    df_sub = df_session.loc[valid_mask].copy().reset_index(drop=True)
    X_raw = X_raw[valid_mask]

    if len(df_sub) < N_CLUSTERS:
        df_session["cluster_id"] = UNASSIGNED_LABEL
        return df_session

    X_z = rowwise_zscore(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_z)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
    raw_labels = kmeans.fit_predict(X_scaled)

    remap_dict = infer_intra_dataset_semantic_mapping(df_sub, raw_labels)
    semantic_labels = np.array([remap_dict.get(int(l), UNASSIGNED_LABEL) for l in raw_labels], dtype=int)

    df_sub["cluster_id"] = semantic_labels
    return df_sub


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


def plot_single_feature_split_manifold(
        feat_title,
        feat_vals,
        unit_label,
        cmap_arg,
        is_div,
        is_phase,
        manifold_full,
        cluster_manifolds,
        cluster_neuron_counts,
        n_full,
        behavior,
        metadata,
        elev,
        azim,
):
    fig = plt.figure(figsize=(18, 5.0))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.25)

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

    # 1. Complete Manifold
    ax0 = fig.add_subplot(gs[0], projection="3d")
    sc0 = ax0.scatter(manifold_full[:, 0], manifold_full[:, 1], manifold_full[:, 2], c=feat_vals, cmap=cmap,
                      norm=norm, s=15, alpha=0.9 if feat_title == "Direction" else 0.85)
    style_3d_axis(ax0, f"Complete Manifold\n(n = {n_full} neurons)")
    ax0.view_init(elev=elev, azim=azim)
    cb0 = fig.colorbar(sc0, ax=ax0, fraction=0.035, pad=0.04)
    if feat_title == "Direction":
        cb0.set_ticks([-1, 0, 1])
        cb0.ax.set_yticklabels(["left", "stop", "right"], fontsize=6)
    else:
        cb0.set_label(unit_label, fontsize=7)

    # 2-4. Cluster-Specific Manifolds
    for idx, (cid, cname) in enumerate(CLUSTER_NAMES.items(), start=1):
        ax = fig.add_subplot(gs[idx], projection="3d")
        c_manifold = cluster_manifolds.get(cid)
        n_clustered = cluster_neuron_counts.get(cid, 0)

        if c_manifold is not None and len(c_manifold) > 0:
            ax.scatter(c_manifold[:, 0], c_manifold[:, 1], c_manifold[:, 2], c=feat_vals, cmap=cmap, norm=norm,
                       s=18, alpha=0.95 if feat_title == "Direction" else 0.9)
            style_3d_axis(ax, f"{cname}\n(n = {n_clustered} neurons)")
        else:
            ax.text2D(0.5, 0.5, "Insufficient Neurons", ha="center", va="center", color="gray",
                      transform=ax.transAxes)
            style_3d_axis(ax, f"{cname}\n(n = {n_clustered} neurons)")
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(
        f"{metadata.get('dataset_name', TARGET_DATASET)} | {metadata.get('mouse_name', '')} | {metadata.get('depth_type', '')} | Feature: {feat_title} (UMAP 3D | elev={elev:.1f}, azim={azim:.1f})",
        fontsize=11, y=0.98)
    plt.tight_layout()
    return fig


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    for p in [OUTPUT_BASE_DIR, WAVEFORM_DIR, ENVELOPE_DIR, PHASE_DIR, CWT_DIR, EMBEDDING_DATA_DIR]:
        ensure_dir(p)

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

    print(f"Found {len(session_groups)} sessions for {TARGET_DATASET}. Computing complete and cluster manifolds...")

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

        signal = std_data.get("signal")
        if signal is None:
            continue
        signal = np.asarray(signal)
        n_timepoints, total_neurons = signal.shape

        df_clustered = cluster_session_neurons(df_session)

        reducer_full = umap.UMAP(
            n_components=N_COMPONENTS,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            random_state=UMAP_RANDOM_STATE
        )
        manifold_full = reducer_full.fit_transform(signal)
        n_full = total_neurons

        cluster_manifolds = {}
        cluster_neuron_counts = {}
        for cid, cname in CLUSTER_NAMES.items():
            sub_neurons = df_clustered[df_clustered["cluster_id"] == cid]["neuron_index"].astype(int).tolist()
            sub_neurons = [n for n in sub_neurons if 0 <= n < total_neurons]
            cluster_neuron_counts[cid] = len(sub_neurons)

            if len(sub_neurons) < 3:
                cluster_manifolds[cid] = None
                continue
            signal_cluster = signal[:, sub_neurons]
            reducer_cluster = umap.UMAP(
                n_neighbors=min(UMAP_N_NEIGHBORS, max(5, len(sub_neurons) - 1)),
                min_dist=UMAP_MIN_DIST,
                n_components=N_COMPONENTS,
                random_state=UMAP_RANDOM_STATE
            )
            cluster_manifolds[cid] = reducer_cluster.fit_transform(signal_cluster)

        pos = std_data.get("pos")
        pos_1d = pos[:, 0] if pos is not None and pos.ndim > 1 else pos
        pos_1d = align_array_length(pos_1d, n_timepoints)
        mov_dir = align_array_length(std_data.get("mov_dir"), n_timepoints)
        speed_1d = align_array_length(std_data.get("speed"), n_timepoints)
        time_1d = align_array_length(std_data.get("time"), n_timepoints)

        elev, azim = compute_position_direction_plane_normal(
            manifold_full, pos_1d, mov_dir, n_pos_bins=N_POS_BINS, min_count_per_bin=MIN_COUNT_PER_BIN
        )

        behavior_payload = {
            "pos": pos_1d,
            "mov_dir": mov_dir,
            "speed": speed_1d,
            "time": time_1d,
        }

        waveforms, envelopes, phases, cwt_bands = extract_lfp_modalities(std_data, target_len=n_timepoints)
        prefix = f"lt_egrin_{mouse_name}_{session_name}_{depth_type}_umap"

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
                lfp_features_to_plot[f"Phase: {cfg['name']}"] = (phases[bkey], "twilight", False, "Radians (-π to π)")
            if cwt_bands.get(bkey) is not None:
                lfp_features_to_plot[f"CWT Power: {cfg['name']}"] = (cwt_bands[bkey], CMAP_CONTINUOUS, False,
                                                                     "Power (dB)")

        all_features = {**behavior_features, **lfp_features_to_plot}

        for feat_title, (feat_vals, cmap_arg, is_div, unit_label) in all_features.items():
            if feat_vals is None or len(feat_vals) != n_timepoints:
                continue

            is_phase = "Phase" in feat_title
            fig = plot_single_feature_split_manifold(
                feat_title=feat_title,
                feat_vals=feat_vals,
                unit_label=unit_label,
                cmap_arg=cmap_arg,
                is_div=is_div,
                is_phase=is_phase,
                manifold_full=manifold_full,
                cluster_manifolds=cluster_manifolds,
                cluster_neuron_counts=cluster_neuron_counts,
                n_full=n_full,
                behavior=behavior_payload,
                metadata=meta,
                elev=elev,
                azim=azim,
            )

            if "Waveform" in feat_title:
                save_figure_both_formats(fig, WAVEFORM_DIR, prefix, sanitize_for_filename(feat_title))
            elif "Envelope" in feat_title:
                save_figure_both_formats(fig, ENVELOPE_DIR, prefix, sanitize_for_filename(feat_title))
            elif "Phase" in feat_title:
                save_figure_both_formats(fig, PHASE_DIR, prefix, sanitize_for_filename(feat_title))
            elif "CWT" in feat_title or feat_title in ["Position", "Direction", "Speed", "Time"]:
                save_figure_both_formats(fig, CWT_DIR, prefix, sanitize_for_filename(feat_title))

    print("\n[SUCCESS] All split-cluster neural manifold feature suites generated successfully!")


if __name__ == "__main__":
    main()
