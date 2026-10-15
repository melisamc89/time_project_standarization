#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Neural Manifold Embedding Overlaid with Multi-Band LFP Modalities.

Generates 4 comprehensive 2x5 3D figure suites for each session:
1. Figure Suite 1 (Standardized LFP Spectral Power Bands in dB):
   - Row 1: Position, Direction, Speed, Time, Ultraslow [0.01 - 0.1 Hz] (dB)
   - Row 2: Delta [1 - 4 Hz], Theta [6 - 12 Hz], Gamma [60 - 80 Hz], Ripple [100 - 250 Hz], MUA [300 - 1000 Hz] (dB)
2. Figure Suite 2 (LFP Hilbert Amplitude Envelopes in µV):
   - Row 1: Position, Direction, Speed, Time, Ultraslow Envelope
   - Row 2: Delta, Theta, Gamma, Ripple, MUA Envelopes
3. Figure Suite 3 (Filtered LFP Waveforms in µV):
   - Row 1: Position, Direction, Speed, Time, Ultraslow Waveform
   - Row 2: Delta, Theta, Gamma, Ripple, MUA Waveforms
4. Figure Suite 4 (CWT Wavelet Scalogram Power in dB):
   - Row 1: Position, Direction, Speed, Time, Delta Slow CWT Power [0.01 - 0.1 Hz]
   - Row 2: Delta Fast [1 - 3 Hz], Theta [8 - 12 Hz], Gamma [40 - 90 Hz], HFO [100 - 250 Hz], MUA [300 - 1000 Hz] (dB)
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
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "analysis", "neural_manifold_lfp_suite")

POWER_BANDS_DIR = os.path.join(OUTPUT_BASE_DIR, "standardized_spectral_power_bands")
ENVELOPE_DIR = os.path.join(OUTPUT_BASE_DIR, "lfp_envelopes")
WAVEFORM_DIR = os.path.join(OUTPUT_BASE_DIR, "lfp_waveforms")
CWT_DIR = os.path.join(OUTPUT_BASE_DIR, "cwt_power")

EMBEDDING_METHOD = "umap"  # "umap" or "pca"
CMAP_CONTINUOUS = "jet"

# Standardized Spectral Bands Configuration (Matching exact target figure)
STANDARDIZED_LFP_BANDS_CONFIG = [
    {"key": "ultra_slow", "name": "Ultraslow [0.01 - 0.1 Hz]", "unit": "Ultraslow (dB)"},
    {"key": "delta",      "name": "Delta [1 - 4 Hz]",          "unit": "Delta (dB)"},
    {"key": "theta",      "name": "Theta [6 - 12 Hz]",         "unit": "Theta (dB)"},
    {"key": "gamma",      "name": "Gamma [60 - 80 Hz]",        "unit": "Gamma (dB)"},
    {"key": "ripple",     "name": "Ripple [100 - 250 Hz]",     "unit": "Ripple (dB)"},
    {"key": "mua",        "name": "MUA [300 - 1000 Hz]",       "unit": "MUA (dB)"},
]

# Amplitude Envelopes & Waveforms Configuration
LFP_ENVELOPE_CONFIG = [
    {"key": "ultra_slow", "name": "Ultraslow [0.01 - 0.1 Hz]", "unit": "Env (µV)"},
    {"key": "delta",      "name": "Delta [1 - 4 Hz]",          "unit": "Env (µV)"},
    {"key": "theta",      "name": "Theta [6 - 12 Hz]",         "unit": "Env (µV)"},
    {"key": "gamma",      "name": "Gamma [60 - 80 Hz]",        "unit": "Env (µV)"},
    {"key": "ripple",     "name": "Ripple [100 - 250 Hz]",     "unit": "Env (µV)"},
    {"key": "mua",        "name": "MUA [300 - 1000 Hz]",       "unit": "Env (µV)"},
]

# Wavelet CWT Scalogram Configuration
CWT_BANDS_CONFIG = [
    {"key": "delta_slow", "name": "Delta [0.01 - 0.1 Hz]",  "f_low": 0.01,  "f_high": 0.1,    "unit": "Delta (dB)"},
    {"key": "delta_fast", "name": "Delta [1 - 3 Hz]",        "f_low": 1.0,   "f_high": 3.0,    "unit": "Delta (dB)"},
    {"key": "theta",      "name": "Theta [8 - 12 Hz]",       "f_low": 8.0,   "f_high": 12.0,   "unit": "Theta (dB)"},
    {"key": "gamma",      "name": "Gamma [40 - 90 Hz]",      "f_low": 40.0,  "f_high": 90.0,   "unit": "Gamma (dB)"},
    {"key": "hfo",        "name": "HFO [100 - 250 Hz]",      "f_low": 100.0, "f_high": 250.0,  "unit": "HFO (dB)"},
    {"key": "mua",        "name": "MUA [300 - 1000 Hz]",     "f_low": 300.0, "f_high": 1000.0, "unit": "MUA (dB)"},
]


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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
# Signal Extraction & Standardization
# =============================================================================

def extract_all_lfp_modalities(std_data, target_len):
    """Pulls downsampled LFP waveforms, envelopes, spectral power (dB), and CWT."""
    waveforms = {}
    envelopes = {}
    power_bands_db = {}

    for cfg in STANDARDIZED_LFP_BANDS_CONFIG:
        bkey = cfg["key"]

        # 1. Filtered Waveform
        raw_key = f"lfp_{bkey}"
        waveforms[bkey] = align_array_length(std_data.get(raw_key), target_len)

        # 2. Hilbert Amplitude Envelope
        env_key = f"lfp_{bkey}_envelope"
        env_arr = std_data.get(env_key)
        if env_arr is None and waveforms[bkey] is not None:
            env_arr = np.abs(waveforms[bkey])
        env_aligned = align_array_length(env_arr, target_len)
        envelopes[bkey] = env_aligned

        # 3. Standardized Spectral Power (dB)
        # Check if precomputed in std_data
        direct_power_keys = [f"lfp_{bkey}_power", f"lfp_{bkey}_power_db", f"power_{bkey}"]
        p_found = None
        for pk in direct_power_keys:
            if pk in std_data and std_data[pk] is not None:
                p_found = align_array_length(std_data[pk], target_len)
                break

        if p_found is not None:
            power_bands_db[bkey] = p_found
        elif env_aligned is not None:
            # P(t) = Env(t)^2 -> 10 * log10(Env^2) = 20 * log10(Env)
            safe_env = np.maximum(env_aligned, 1e-6)
            power_bands_db[bkey] = 20.0 * np.log10(safe_env)
        else:
            power_bands_db[bkey] = None

    # 4. Continuous Wavelet Transform Power (dB)
    cwt_bands = {}
    cwt_matrix = std_data.get("cwt")
    cwt_freqs = std_data.get("cwt_freqs")

    if cwt_matrix is not None and cwt_freqs is not None:
        cwt_matrix = np.asarray(cwt_matrix)
        cwt_freqs = np.asarray(cwt_freqs).reshape(-1)

        if cwt_matrix.shape[0] != target_len:
            if cwt_matrix.shape[0] > target_len:
                cwt_matrix = cwt_matrix[:target_len, :]
            else:
                pad_w = ((0, target_len - cwt_matrix.shape[0]), (0, 0))
                cwt_matrix = np.pad(cwt_matrix, pad_w, mode="edge")

        for cfg in CWT_BANDS_CONFIG:
            bkey = cfg["key"]
            f_low, f_high = cfg["f_low"], cfg["f_high"]
            idx_range = np.where((cwt_freqs >= f_low) & (cwt_freqs <= f_high))[0]
            if len(idx_range) == 0:
                nearest_idx = np.argmin(np.abs(cwt_freqs - 0.5 * (f_low + f_high)))
                idx_range = np.array([nearest_idx])

            mean_power = np.nanmean(cwt_matrix[:, idx_range], axis=1)
            cwt_bands[bkey] = 10.0 * np.log10(np.maximum(mean_power, 1e-12))
    else:
        for cfg in CWT_BANDS_CONFIG:
            cwt_bands[cfg["key"]] = None

    return power_bands_db, envelopes, waveforms, cwt_bands


# =============================================================================
# Generic 2x5 3D Manifold Grid Plotter
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

    # -------------------------------------------------------------------------
    # ROW 1: Position, Direction, Speed, Time, Band 1 (Ultraslow)
    # -------------------------------------------------------------------------
    # 1. Position
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    sc1 = ax1.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=pos_1d, cmap="magma", s=3.5, alpha=0.9
    )
    style_3d_axis(ax1, "Position")
    ax1.view_init(elev=elev, azim=azim)
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=0.035, pad=0.04)
    cb1.set_label("position (cm)", fontsize=7)
    cb1.ax.tick_params(labelsize=6)

    # 2. Direction
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    dir_cmap, dir_norm = get_direction_colors()
    sc2 = ax2.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=mov_dir, cmap=dir_cmap, norm=dir_norm, s=3.5, alpha=0.9
    )
    style_3d_axis(ax2, "Direction")
    ax2.view_init(elev=elev, azim=azim)
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.035, pad=0.04, ticks=[-1, 0, 1])
    cb2.ax.set_yticklabels(["left", "stop", "right"], fontsize=6)

    # 3. Speed
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    sc3 = ax3.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=speed_1d, cmap="viridis", s=3.5, alpha=0.9
    )
    style_3d_axis(ax3, "Speed")
    ax3.view_init(elev=elev, azim=azim)
    cb3 = fig.colorbar(sc3, ax=ax3, fraction=0.035, pad=0.04)
    cb3.set_label("speed (cm/s)", fontsize=7)
    cb3.ax.tick_params(labelsize=6)

    # 4. Time
    ax4 = fig.add_subplot(gs[0, 3], projection="3d")
    sc4 = ax4.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=time_1d, cmap="YlGn_r", s=3.5, alpha=0.9
    )
    style_3d_axis(ax4, "Time")
    ax4.view_init(elev=elev, azim=azim)
    cb4 = fig.colorbar(sc4, ax=ax4, fraction=0.035, pad=0.04)
    cb4.set_label("time (s)", fontsize=7)
    cb4.ax.tick_params(labelsize=6)

    # 5. Band 1 (Ultraslow)
    cfg0 = feature_configs[0]
    val0 = features_dict.get(cfg0["key"])
    ax5 = fig.add_subplot(gs[0, 4], projection="3d")
    if val0 is not None:
        p1, p99 = np.nanpercentile(val0, 1), np.nanpercentile(val0, 99)
        if is_diverging:
            limit = max(abs(p1), abs(p99))
            norm5 = Normalize(vmin=-limit, vmax=limit)
            cmap = "seismic"
        else:
            norm5 = Normalize(vmin=p1, vmax=max(p99, p1 + 1e-3))
            cmap = CMAP_CONTINUOUS
        sc5 = ax5.scatter(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            c=val0, cmap=cmap, norm=norm5, s=3.5, alpha=0.85
        )
        style_3d_axis(ax5, cfg0["name"])
        ax5.view_init(elev=elev, azim=azim)
        cb5 = fig.colorbar(sc5, ax=ax5, fraction=0.035, pad=0.04)
        cb5.set_label(cfg0["unit"], fontsize=7)
        cb5.ax.tick_params(labelsize=6)
    else:
        style_3d_axis(ax5, cfg0["name"])
        ax5.text(0.5, 0.5, 0.5, "N/A", ha="center", va="center", color="gray")

    # -------------------------------------------------------------------------
    # ROW 2: Remaining 5 Spectral Bands (Delta, Theta, Gamma, Ripple, MUA)
    # -------------------------------------------------------------------------
    for col_idx, cfg in enumerate(feature_configs[1:]):
        val = features_dict.get(cfg["key"])
        ax = fig.add_subplot(gs[1, col_idx], projection="3d")
        if val is not None and len(val) == len(embedding):
            p1, p99 = np.nanpercentile(val, 1), np.nanpercentile(val, 99)
            if is_diverging:
                limit = max(abs(p1), abs(p99))
                norm = Normalize(vmin=-limit, vmax=limit)
                cmap = "seismic"
            else:
                norm = Normalize(vmin=p1, vmax=max(p99, p1 + 1e-3))
                cmap = CMAP_CONTINUOUS
            sc = ax.scatter(
                embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=val, cmap=cmap, norm=norm, s=3.5, alpha=0.85
            )
            style_3d_axis(ax, cfg["name"])
            ax.view_init(elev=elev, azim=azim)
            cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)
            cb.set_label(cfg["unit"], fontsize=7)
            cb.ax.tick_params(labelsize=6)
        else:
            style_3d_axis(ax, cfg["name"])
            ax.text(0.5, 0.5, 0.5, "Band Absent", ha="center", va="center", color="gray", fontsize=7)

    return fig


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    for p in [OUTPUT_BASE_DIR, POWER_BANDS_DIR, ENVELOPE_DIR, WAVEFORM_DIR, CWT_DIR]:
        ensure_dir(p)

    emb_files = sorted(
        glob.glob(os.path.join(EMBEDDING_DATA_DIR, "*_embedding_behavior.pkl"))
    )
    egrin_files = [f for f in emb_files if "egrin" in os.path.basename(f).lower()]
    targets = egrin_files if len(egrin_files) > 0 else emb_files

    print(
        f"Discovered {len(targets)} session embeddings. Generating figures across\n"
        "  1) Standardized Spectral Power Bands (dB)\n"
        "  2) Hilbert Amplitude Envelopes (µV)\n"
        "  3) Filtered Waveforms (µV)\n"
        "  4) CWT Power (dB)..."
    )

    count = 0

    for idx, emb_path in enumerate(targets):
        emb_data = load_pickle(emb_path)
        meta = emb_data.get("metadata", {})

        std_path = resolve_standardized_path(meta)
        if std_path is None or not os.path.isfile(std_path):
            print(f"[{idx+1}/{len(targets)}] Skip: standardized file not found for {meta.get('mouse_name')}")
            continue

        std_data = load_pickle(std_path)
        embedding = emb_data.get(EMBEDDING_METHOD.lower())
        if embedding is None:
            continue

        n_points = embedding.shape[0]

        # Extract aligned behavior
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

        # Extract all modalities
        power_bands_db, envelopes, waveforms, cwt_bands = extract_all_lfp_modalities(
            std_data, target_len=n_points
        )

        view_angles = meta.get(f"{EMBEDDING_METHOD.lower()}_view_angles", None)
        mouse_name = meta.get("mouse_name", "mouse")
        depth_type = meta.get("depth_type", "depth")
        session_name = meta.get("session_name", "session")

        prefix = f"lt_egrin_{mouse_name}_{session_name}_{depth_type}_{EMBEDDING_METHOD}"

        # ---------------------------------------------------------------------
        # 1. Standardized LFP Spectral Power Bands (dB)
        # ---------------------------------------------------------------------
        fig_pow = plot_manifold_feature_grid(
            embedding=embedding,
            behavior=behavior_payload,
            features_dict=power_bands_db,
            feature_configs=STANDARDIZED_LFP_BANDS_CONFIG,
            title_suffix="Standardized LFP Spectral Power Bands (dB)",
            metadata=meta,
            view_angles=view_angles,
            is_diverging=False,
        )
        pow_path = os.path.join(POWER_BANDS_DIR, f"{prefix}_lfp_bands.png")
        fig_pow.savefig(pow_path, dpi=200, bbox_inches="tight")
        plt.close(fig_pow)

        # ---------------------------------------------------------------------
        # 2. Hilbert Amplitude Envelopes (µV)
        # ---------------------------------------------------------------------
        fig_env = plot_manifold_feature_grid(
            embedding=embedding,
            behavior=behavior_payload,
            features_dict=envelopes,
            feature_configs=LFP_ENVELOPE_CONFIG,
            title_suffix="LFP Hilbert Amplitude Envelopes (µV)",
            metadata=meta,
            view_angles=view_angles,
            is_diverging=False,
        )
        env_path = os.path.join(ENVELOPE_DIR, f"{prefix}_lfp_envelopes.png")
        fig_env.savefig(env_path, dpi=200, bbox_inches="tight")
        plt.close(fig_env)

        # ---------------------------------------------------------------------
        # 3. Filtered Waveforms (µV)
        # ---------------------------------------------------------------------
        fig_wav = plot_manifold_feature_grid(
            embedding=embedding,
            behavior=behavior_payload,
            features_dict=waveforms,
            feature_configs=LFP_ENVELOPE_CONFIG,
            title_suffix="Filtered LFP Waveforms (µV)",
            metadata=meta,
            view_angles=view_angles,
            is_diverging=True,
        )
        wav_path = os.path.join(WAVEFORM_DIR, f"{prefix}_lfp_waveforms.png")
        fig_wav.savefig(wav_path, dpi=200, bbox_inches="tight")
        plt.close(fig_wav)

        # ---------------------------------------------------------------------
        # 4. Continuous Wavelet Transform Power (dB)
        # ---------------------------------------------------------------------
        fig_cwt = plot_manifold_feature_grid(
            embedding=embedding,
            behavior=behavior_payload,
            features_dict=cwt_bands,
            feature_configs=CWT_BANDS_CONFIG,
            title_suffix="Continuous Wavelet Transform Power (dB)",
            metadata=meta,
            view_angles=view_angles,
            is_diverging=False,
        )
        cwt_path = os.path.join(CWT_DIR, f"{prefix}_cwt_power.png")
        fig_cwt.savefig(cwt_path, dpi=200, bbox_inches="tight")
        plt.close(fig_cwt)

        print(f"[{idx+1}/{len(targets)}] Saved spectral power, envelope, waveform & CWT suites for {mouse_name}")
        count += 1

    print(f"\n[SUCCESS] Completed figure suite generation across {count} sessions.")
    print(f"Output directory: {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    main()