#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Neural Manifold Embedding Overlaid with Multi-Band LFP Waveform, Envelope, Phase, and CWT Power.

Generates 4 comprehensive figure suites for each lt_egrin session using identical frequency bands,
saving outputs in both PNG and SVG formats.
"""

import glob
import os
import pickle
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Paths & Settings
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
STANDARDIZED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "standardized_sessions",
    "lt_egrin",
    "calcium_imaging",
)
EMBEDDING_DATA_DIR = os.path.join(
    PROJECT_ROOT, "analysis", "umap_pca_per_row", "embedding_data"
)
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "analysis", "neural_manifold_lfp_suite_lt_egrin")

WAVEFORM_DIR = os.path.join(OUTPUT_BASE_DIR, "lfp_waveforms")
ENVELOPE_DIR = os.path.join(OUTPUT_BASE_DIR, "lfp_envelopes")
PHASE_DIR = os.path.join(OUTPUT_BASE_DIR, "lfp_phases")
CWT_DIR = os.path.join(OUTPUT_BASE_DIR, "cwt_power")

EMBEDDING_METHOD = "umap"  # "umap" or "pca"
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


def save_figure_both_formats(fig, directory, prefix, suffix):
    """Saves the given figure as both PNG and SVG."""
    ensure_dir(directory)
    base_name = f"{prefix}_{suffix}"
    png_path = os.path.join(directory, f"{base_name}.png")
    svg_path = os.path.join(directory, f"{base_name}.svg")

    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")


def style_3d_axis(ax, title):
    ax.set_title(title, fontsize=9, pad=4)
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor("white")
        except Exception:
            pass
    ax.set_xlabel("x", labelpad=-7, fontsize=7)
    ax.set_ylabel("y", labelpad=-7, fontsize=7)
    ax.set_zlabel("z", labelpad=-7, fontsize=7)
    ax.tick_params(labelsize=6)


def get_direction_colors():
    cmap = ListedColormap(["#1f77b4", "#7f7f7f", "#17becf"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    return cmap, norm


def resolve_standardized_path(meta):
    source_pkl = meta.get("source_standardized_file_path", "")
    if source_pkl and os.path.isfile(source_pkl):
        return source_pkl

    mouse_name = str(meta.get("mouse_name", "")).strip()
    depth_type = str(meta.get("depth_type", "")).strip()

    if mouse_name:
        candidates = glob.glob(
            os.path.join(STANDARDIZED_DIR, "**", f"*{mouse_name}*.pkl"),
            recursive=True,
        )
        if candidates:
            if depth_type:
                for c in candidates:
                    if depth_type in c:
                        return c
            return candidates[0]
    return None


def align_array_length(arr, target_len):
    if arr is None:
        return None
    arr = np.asarray(arr).reshape(-1)
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), mode="edge")


# =============================================================================
# Feature Extraction from Ingested Session
# =============================================================================

def extract_lfp_modalities(std_data, target_len):
    """Pulls ingested multi-band LFP waveforms, Hilbert envelopes, phases, and CWT power bands."""
    waveforms = {}
    envelopes = {}
    phases = {}
    cwt_bands = {}

    cwt_matrix = std_data.get("cwt")
    cwt_freqs = std_data.get("cwt_freqs")
    if cwt_matrix is not None:
        cwt_matrix = np.asarray(cwt_matrix)
        if cwt_matrix.shape[0] != target_len:
            if cwt_matrix.shape[0] > target_len:
                cwt_matrix = cwt_matrix[:target_len, :]
            else:
                pad_w = ((0, target_len - cwt_matrix.shape[0]), (0, 0))
                cwt_matrix = np.pad(cwt_matrix, pad_w, mode="edge")
    if cwt_freqs is not None:
        cwt_freqs = np.asarray(cwt_freqs).reshape(-1)

    for cfg in LFP_BANDS_CONFIG:
        bkey = cfg["key"]
        # 1. Waveform
        waveforms[bkey] = align_array_length(std_data.get(f"lfp_{bkey}_20hz"), target_len)

        # 2. Envelope
        env_arr = std_data.get(f"lfp_{bkey}_hilbert_20hz")
        if env_arr is None and waveforms[bkey] is not None:
            env_arr = np.abs(waveforms[bkey])
        envelopes[bkey] = align_array_length(env_arr, target_len)

        # 3. Phase
        phases[bkey] = align_array_length(std_data.get(f"lfp_{bkey}_phase_20hz"), target_len)

        # 4. CWT Power (dB) extracted using the exact same frequency band
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


# =============================================================================
# Generic 2x5 Grid Plotter
# =============================================================================

def plot_manifold_feature_grid(
        embedding,
        behavior,
        features_dict,
        feature_configs,
        title_suffix,
        metadata,
        view_angles=None,
        is_diverging=False,
        is_phase=False,
):
    fig = plt.figure(figsize=(20.0, 7.8))
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.30, wspace=0.28)

    title = (
        f"{metadata.get('dataset_name', 'lt_egrin')} | "
        f"{metadata.get('mouse_name', '')} | {metadata.get('depth_type', '')} | "
        f"{title_suffix}"
    )
    fig.suptitle(title, fontsize=12, y=0.98)

    elev = view_angles.get("elev", 30.0) if view_angles else 30.0
    azim = view_angles.get("azim", 45.0) if view_angles else 45.0

    pos_1d = behavior["pos"]
    mov_dir = behavior["mov_dir"]
    speed_1d = behavior["speed"]
    time_1d = behavior["time"]

    # ROW 1: Position, Direction, Speed, Time, Band 1
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    sc1 = ax1.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=pos_1d, cmap="magma", s=3.5, alpha=0.9)
    style_3d_axis(ax1, "Position")
    ax1.view_init(elev=elev, azim=azim)
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=0.035, pad=0.04)
    cb1.set_label("pos (cm)", fontsize=7)
    cb1.ax.tick_params(labelsize=6)

    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    dir_cmap, dir_norm = get_direction_colors()
    sc2 = ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=mov_dir, cmap=dir_cmap, norm=dir_norm, s=3.5,
                      alpha=0.9)
    style_3d_axis(ax2, "Direction")
    ax2.view_init(elev=elev, azim=azim)
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.035, pad=0.04, ticks=[-1, 0, 1])
    cb2.ax.set_yticklabels(["left", "stop", "right"], fontsize=6)

    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    sc3 = ax3.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=speed_1d, cmap="viridis", s=3.5, alpha=0.9)
    style_3d_axis(ax3, "Speed")
    ax3.view_init(elev=elev, azim=azim)
    cb3 = fig.colorbar(sc3, ax=ax3, fraction=0.035, pad=0.04)
    cb3.set_label("speed (cm/s)", fontsize=7)
    cb3.ax.tick_params(labelsize=6)

    ax4 = fig.add_subplot(gs[0, 3], projection="3d")
    sc4 = ax4.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=time_1d, cmap="YlGn_r", s=3.5, alpha=0.9)
    style_3d_axis(ax4, "Time")
    ax4.view_init(elev=elev, azim=azim)
    cb4 = fig.colorbar(sc4, ax=ax4, fraction=0.035, pad=0.04)
    cb4.set_label("time (s)", fontsize=7)
    cb4.ax.tick_params(labelsize=6)

    # Subplot 5: Band 1
    cfg0 = feature_configs[0]
    val0 = features_dict.get(cfg0["key"])
    ax5 = fig.add_subplot(gs[0, 4], projection="3d")
    if val0 is not None:
        if is_phase:
            norm5 = Normalize(vmin=-np.pi, vmax=np.pi)
            cmap = "twilight"
        elif is_diverging:
            p1, p99 = np.nanpercentile(val0, 1), np.nanpercentile(val0, 99)
            limit = max(abs(p1), abs(p99))
            norm5 = Normalize(vmin=-limit, vmax=limit)
            cmap = "seismic"
        else:
            p1, p99 = np.nanpercentile(val0, 1), np.nanpercentile(val0, 99)
            norm5 = Normalize(vmin=p1, vmax=max(p99, p1 + 1e-3))
            cmap = CMAP_CONTINUOUS

        sc5 = ax5.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=val0, cmap=cmap, norm=norm5, s=3.5,
                          alpha=0.85)
        style_3d_axis(ax5, cfg0["name"])
        ax5.view_init(elev=elev, azim=azim)
        cb5 = fig.colorbar(sc5, ax=ax5, fraction=0.035, pad=0.04)
        if is_phase:
            cb5.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cb5.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
        cb5.set_label(cfg0["unit"], fontsize=7)
        cb5.ax.tick_params(labelsize=6)
    else:
        style_3d_axis(ax5, cfg0["name"])
        ax5.text(0.5, 0.5, 0.5, "N/A", ha="center", va="center", color="gray")

    # ROW 2: Remaining 5 Bands
    for col_idx, cfg in enumerate(feature_configs[1:]):
        val = features_dict.get(cfg["key"])
        ax = fig.add_subplot(gs[1, col_idx], projection="3d")
        if val is not None and len(val) == len(embedding):
            if is_phase:
                norm = Normalize(vmin=-np.pi, vmax=np.pi)
                cmap = "twilight"
            elif is_diverging:
                p1, p99 = np.nanpercentile(val, 1), np.nanpercentile(val, 99)
                limit = max(abs(p1), abs(p99))
                norm = Normalize(vmin=-limit, vmax=limit)
                cmap = "seismic"
            else:
                p1, p99 = np.nanpercentile(val, 1), np.nanpercentile(val, 99)
                norm = Normalize(vmin=p1, vmax=max(p99, p1 + 1e-3))
                cmap = CMAP_CONTINUOUS

            sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=val, cmap=cmap, norm=norm, s=3.5,
                            alpha=0.85)
            style_3d_axis(ax, cfg["name"])
            ax.view_init(elev=elev, azim=azim)
            cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)
            if is_phase:
                cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
                cb.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
            cb.set_label(cfg["unit"], fontsize=7)
            cb.ax.tick_params(labelsize=6)
        else:
            style_3d_axis(ax, cfg["name"])
            ax.text(0.5, 0.5, 0.5, "Band Absent", ha="center", va="center", color="gray", fontsize=7)

    return fig


# =============================================================================
# Main Execution Pipeline
# =============================================================================

def main():
    for p in [OUTPUT_BASE_DIR, WAVEFORM_DIR, ENVELOPE_DIR, PHASE_DIR, CWT_DIR]:
        ensure_dir(p)

    emb_files = sorted(
        glob.glob(os.path.join(EMBEDDING_DATA_DIR, "*_embedding_behavior.pkl"))
    )
    egrin_files = [f for f in emb_files if "egrin" in os.path.basename(f).lower()]
    targets = egrin_files if len(egrin_files) > 0 else emb_files

    print(f"Discovered {len(targets)} session embeddings. Generating 4 figure suites (PNG & SVG)...")

    count = 0
    for idx, emb_path in enumerate(targets):
        emb_data = load_pickle(emb_path)
        meta = emb_data.get("metadata", {})

        std_path = resolve_standardized_path(meta)
        if std_path is None or not os.path.isfile(std_path):
            print(f"[{idx + 1}/{len(targets)}] Skip: standardized file not found for {meta.get('mouse_name')}")
            continue

        std_data = load_pickle(std_path)
        embedding = emb_data.get(EMBEDDING_METHOD.lower())
        if embedding is None:
            continue

        n_points = embedding.shape[0]

        beh = emb_data.get("behavior", {})
        pos_1d = beh.get("pos", None)
        mov_dir = beh.get("mov_dir", None)
        time_1d = beh.get("time", None)
        speed_1d = beh.get("speed", None)

        if pos_1d is None:
            pos_1d = np.asarray(std_data["pos"])
            pos_1d = pos_1d[:, 0] if pos_1d.ndim > 1 else pos_1d
        if mov_dir is None:
            mov_dir = np.asarray(std_data["mov_dir"]).reshape(-1)
        if time_1d is None:
            time_1d = np.asarray(std_data["time"]).reshape(-1)
        if speed_1d is None:
            speed_1d = np.asarray(std_data["speed"]).reshape(-1)

        behavior_payload = {
            "pos": np.asarray(pos_1d).reshape(-1)[:n_points],
            "mov_dir": np.asarray(mov_dir).reshape(-1)[:n_points],
            "speed": np.asarray(speed_1d).reshape(-1)[:n_points],
            "time": np.asarray(time_1d).reshape(-1)[:n_points],
        }

        waveforms, envelopes, phases, cwt_bands = extract_lfp_modalities(std_data, target_len=n_points)

        view_angles = meta.get(f"{EMBEDDING_METHOD.lower()}_view_angles", None)
        mouse_name = meta.get("mouse_name", "mouse")
        depth_type = meta.get("depth_type", "depth")
        session_name = meta.get("session_name", "session")

        prefix = f"lt_egrin_{mouse_name}_{session_name}_{depth_type}_{EMBEDDING_METHOD}"

        # 1. Waveforms
        fig_wav = plot_manifold_feature_grid(
            embedding=embedding, behavior=behavior_payload,
            features_dict=waveforms, feature_configs=LFP_BANDS_CONFIG,
            title_suffix="Filtered LFP Waveforms (µV)", metadata=meta,
            view_angles=view_angles, is_diverging=True, is_phase=False,
        )
        save_figure_both_formats(fig_wav, WAVEFORM_DIR, prefix, "lfp_waveforms")

        # 2. Envelopes
        fig_env = plot_manifold_feature_grid(
            embedding=embedding, behavior=behavior_payload,
            features_dict=envelopes, feature_configs=LFP_BANDS_CONFIG,
            title_suffix="Hilbert Amplitude Envelopes (µV)", metadata=meta,
            view_angles=view_angles, is_diverging=False, is_phase=False,
        )
        save_figure_both_formats(fig_env, ENVELOPE_DIR, prefix, "lfp_envelopes")

        # 3. Phases
        fig_pha = plot_manifold_feature_grid(
            embedding=embedding, behavior=behavior_payload,
            features_dict=phases, feature_configs=LFP_BANDS_CONFIG,
            title_suffix="Instantaneous Phases (radians)", metadata=meta,
            view_angles=view_angles, is_diverging=False, is_phase=True,
        )
        save_figure_both_formats(fig_pha, PHASE_DIR, prefix, "lfp_phases")

        # 4. CWT Power Bands
        cwt_configs = [{**cfg, "unit": f"{cfg['key'].capitalize()} (dB)"} for cfg in LFP_BANDS_CONFIG]
        fig_cwt = plot_manifold_feature_grid(
            embedding=embedding, behavior=behavior_payload,
            features_dict=cwt_bands, feature_configs=cwt_configs,
            title_suffix="Continuous Wavelet Transform Band Power (dB)", metadata=meta,
            view_angles=view_angles, is_diverging=False, is_phase=False,
        )
        save_figure_both_formats(fig_cwt, CWT_DIR, prefix, "cwt_power")

        print(f"[{idx + 1}/{len(targets)}] Saved PNG & SVG suites for {mouse_name}")
        count += 1

    print(f"\n[SUCCESS] Completed figure suite generation (PNG & SVG) across {count} sessions.")


if __name__ == "__main__":
    main()
