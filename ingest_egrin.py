#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural Manifold Embedding Overlaid with CWT Spectral Power Bands.

Plots 3D UMAP/PCA neural embeddings colored by:
Row 1: Position, Direction, Speed, Time, and Delta Slow CWT Power [0.01 - 0.1 Hz]
Row 2: Delta Fast [1 - 3 Hz], Theta [8 - 12 Hz], Gamma [40 - 90 Hz],
       HFO/Ripple [100 - 250 Hz], MUA [300 - 1000 Hz]

Integrates directly with the downsampled CWT matrix and frequency grid
from the standardized session files.
"""

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm

# =============================================================================
# Paths & Configuration
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
STANDARDIZED_DIR = os.path.join(
    PROJECT_ROOT, "data", "standardized_sessions", "lt_egrin", "calcium_imaging"
)
EMBEDDING_DATA_DIR = os.path.join(
    PROJECT_ROOT, "analysis", "umap_pca_per_row", "embedding_data"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "cwt_manifold_spectrum")

PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")

# Frequency bands matching the canonical CWT reference figure
CWT_BANDS = [
    {"key": "delta_slow", "name": "Delta [0.01 - 0.1 Hz]",  "f_low": 0.01,  "f_high": 0.1,    "unit": "Delta (dB)"},
    {"key": "delta_fast", "name": "Delta [1 - 3 Hz]",       "f_low": 1.0,   "f_high": 3.0,    "unit": "Delta (dB)"},
    {"key": "theta",      "name": "Theta [8 - 12 Hz]",      "f_low": 8.0,   "f_high": 12.0,   "unit": "Theta (dB)"},
    {"key": "gamma",      "name": "Gamma [40 - 90 Hz]",     "f_low": 40.0,  "f_high": 90.0,   "unit": "Gamma (dB)"},
    {"key": "hfo",        "name": "HFO [100 - 250 Hz]",     "f_low": 100.0, "f_high": 250.0,  "unit": "HFO (dB)"},
    {"key": "mua",        "name": "MUA [300 - 1000 Hz]",    "f_low": 300.0, "f_high": 1000.0, "unit": "MUA (dB)"},
]

EMBEDDING_METHOD = "umap"  # "umap" or "pca"
CMAP_DB_POWER = "jet"


# =============================================================================
# Helpers
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
    cmap = ListedColormap(["#1f77b4", "#7f7f7f", "#17becf"])  # -1 (left), 0 (none), 1 (right)
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
            os.path.join(STANDARDIZED_DIR, "**", f"*{mouse_name}*.pkl"), recursive=True
        )
        if candidates:
            if depth_type:
                for c in candidates:
                    if depth_type in c:
                        return c
            return candidates[0]

    return None


# =============================================================================
# CWT Band Power Extraction
# =============================================================================

def extract_cwt_band_powers(cwt_matrix, cwt_freqs, target_len):
    """
    Integrates spectral power across frequency bands from the 20 Hz CWT matrix
    and converts to logarithmic decibels: 10 * log10(power).
    """
    if cwt_matrix is None or cwt_freqs is None:
        return None

    cwt_matrix = np.asarray(cwt_matrix)
    cwt_freqs = np.asarray(cwt_freqs).reshape(-1)

    # Standardize dimensions (time x freqs)
    if cwt_matrix.ndim == 2:
        if cwt_matrix.shape[0] == len(cwt_freqs) and cwt_matrix.shape[1] != len(cwt_freqs):
            cwt_repr = cwt_matrix.T
        else:
            cwt_repr = cwt_matrix
    else:
        return None

    # Length alignment
    if cwt_repr.shape[0] != target_len:
        if cwt_repr.shape[0] > target_len:
            cwt_repr = cwt_repr[:target_len, :]
        else:
            pad_width = ((0, target_len - cwt_repr.shape[0]), (0, 0))
            cwt_repr = np.pad(cwt_repr, pad_width, mode="edge")

    band_features = {}
    for band in CWT_BANDS:
        bkey = band["key"]
        f_low, f_high = band["f_low"], band["f_high"]

        f_indices = np.where((cwt_freqs >= f_low) & (cwt_freqs <= f_high))[0]
        if len(f_indices) == 0:
            nearest_idx = np.argmin(np.abs(cwt_freqs - 0.5 * (f_low + f_high)))
            f_indices = np.array([nearest_idx])

        band_slice = cwt_repr[:, f_indices]
        # Calculate mean band power and convert to dB
        mean_power = np.nanmean(band_slice, axis=1)
        power_db = 10.0 * np.log10(np.maximum(mean_power, 1e-12))
        band_features[bkey] = power_db

    return band_features


# =============================================================================
# 2x5 Reference Plotter
# =============================================================================

def plot_cwt_manifold_grid(embedding, behavior, cwt_bands, metadata, view_angles=None):
    fig = plt.figure(figsize=(20.0, 7.8))
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.30, wspace=0.28)

    title = (
        f"{metadata.get('dataset_name', 'lt_egrin')} | {metadata.get('mouse_name', '')} | "
        f"{metadata.get('depth_type', '')} | Neural Manifold Overlaid with CWT Power (dB)"
    )
    fig.suptitle(title, fontsize=12, y=0.98)

    elev = view_angles.get("elev", 30.0) if view_angles else 30.0
    azim = view_angles.get("azim", 45.0) if view_angles else 45.0

    pos_1d = behavior["pos"]
    mov_dir = behavior["mov_dir"]
    speed_1d = behavior["speed"]
    time_1d = behavior["time"]

    # -------------------------------------------------------------------------
    # ROW 1: Position, Direction, Speed, Time, Delta Slow [0.01 - 0.1 Hz]
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

    # 5. Delta Slow [0.01 - 0.1 Hz]
    d_slow = cwt_bands["delta_slow"]
    ax5 = fig.add_subplot(gs[0, 4], projection="3d")
    if d_slow is not None:
        p1, p99 = np.nanpercentile(d_slow, 1), np.nanpercentile(d_slow, 99)
        norm5 = Normalize(vmin=p1, vmax=max(p99, p1 + 1e-3))
        sc5 = ax5.scatter(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            c=d_slow, cmap=CMAP_DB_POWER, norm=norm5, s=3.5, alpha=0.85
        )
        style_3d_axis(ax5, CWT_BANDS[0]["name"])
        ax5.view_init(elev=elev, azim=azim)
        cb5 = fig.colorbar(sc5, ax=ax5, fraction=0.035, pad=0.04)
        cb5.set_label(CWT_BANDS[0]["unit"], fontsize=7)
        cb5.ax.tick_params(labelsize=6)
    else:
        style_3d_axis(ax5, CWT_BANDS[0]["name"])
        ax5.text(0.5, 0.5, 0.5, "N/A", ha="center", va="center", color="gray")

    # -------------------------------------------------------------------------
    # ROW 2: Delta Fast, Theta, Gamma, HFO, MUA
    # -------------------------------------------------------------------------
    row2_bands = CWT_BANDS[1:]
    for col_idx, cfg in enumerate(row2_bands):
        bkey = cfg["key"]
        val = cwt_bands.get(bkey, None)
        ax = fig.add_subplot(gs[1, col_idx], projection="3d")

        if val is not None and len(val) == len(embedding):
            p1, p99 = np.nanpercentile(val, 1), np.nanpercentile(val, 99)
            norm = Normalize(vmin=p1, vmax=max(p99, p1 + 1e-3))
            sc = ax.scatter(
                embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=val, cmap=CMAP_DB_POWER, norm=norm, s=3.5, alpha=0.85
            )
            style_3d_axis(ax, cfg["name"])
            ax.view_init(elev=elev, azim=azim)
            cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)
            cb.set_label(cfg["unit"], fontsize=7)
            cb.ax.tick_params(labelsize=6)
        else:
            style_3d_axis(ax, cfg["name"])
            ax.text(
                0.5, 0.5, 0.5, "Band Absent", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=7
            )

    return fig


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)

    emb_files = sorted(glob.glob(os.path.join(EMBEDDING_DATA_DIR, "*_embedding_behavior.pkl")))
    egrin_files = [f for f in emb_files if "egrin" in os.path.basename(f).lower()]
    targets = egrin_files if len(egrin_files) > 0 else emb_files

    print(f"Processing {len(targets)} sessions for CWT manifold visualization...")
    count = 0

    for idx, emb_path in enumerate(targets):
        emb_data = load_pickle(emb_path)
        meta = emb_data.get("metadata", {})

        std_path = resolve_standardized_path(meta)
        if std_path is None or not os.path.isfile(std_path):
            print(f"[{idx+1}/{len(targets)}] Skip: standardized file not found for {meta.get('mouse_name')}")
            continue

        std_data = load_pickle(std_path)
        if "cwt" not in std_data or std_data["cwt"] is None or "cwt_freqs" not in std_data:
            print(f"[{idx+1}/{len(targets)}] Skip: CWT not present in {os.path.basename(std_path)}")
            continue

        embedding = emb_data.get(EMBEDDING_METHOD.lower(), None)
        if embedding is None:
            continue

        n_points = embedding.shape[0]

        # Extract behavior aligned to embedding length
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

        # Extract CWT band power from the ingested 20 Hz matrix
        cwt_bands = extract_cwt_band_powers(
            cwt_matrix=std_data["cwt"],
            cwt_freqs=std_data["cwt_freqs"],
            target_len=n_points,
        )

        view_angles = meta.get(f"{EMBEDDING_METHOD.lower()}_view_angles", None)
        dataset_name = meta.get("dataset_name", "lt_egrin")
        mouse_name = meta.get("mouse_name", "mouse")
        depth_type = meta.get("depth_type", "depth")
        session_name = meta.get("session_name", "session")

        file_prefix = f"{dataset_name}_{mouse_name}_{session_name}_{depth_type}_{EMBEDDING_METHOD}_cwt_manifold"

        fig = plot_cwt_manifold_grid(
            embedding=embedding,
            behavior=behavior_payload,
            cwt_bands=cwt_bands,
            metadata=meta,
            view_angles=view_angles,
        )

        png_out = os.path.join(PNG_DIR, f"{file_prefix}.png")
        svg_out = os.path.join(SVG_DIR, f"{file_prefix}.svg")
        fig.savefig(png_out, dpi=200, bbox_inches="tight")
        fig.savefig(svg_out, bbox_inches="tight")
        plt.close(fig)

        print(f"[{idx+1}/{len(targets)}] Saved figure -> {png_out}")
        count += 1

    print(f"\n[DONE] Successfully exported {count} CWT manifold figures to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()