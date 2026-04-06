#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from sklearn.decomposition import PCA
import umap

# ==============================
# User settings
# ==============================
CSV_PATH = "/home/melisamc/Documents/time_project_standarization/data/metadata/master_dataset_index.csv"
OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/embeddings_test"
OVERWRITE = True
MAX_ROWS = None  # set to an integer for quick debugging
POINT_SIZE = 2
FPS = 20
DPI = 180

# UMAP params requested
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

# Colormaps requested
POS_CMAP = "magma"
TIME_CMAP = "YlGn_r"
DIR_COLORS = {
    -1: "blue",
    0: "gray",
    1: "lightblue",
}

REQUIRED_KEYS = ["signal", "pos", "mov_dir", "speed", "time"]


# ==============================
# Helpers
# ==============================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_pickle_standard(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def sanitize_filename(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(text))


def infer_pos_1d(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos)
    if pos.ndim == 1:
        return pos.reshape(-1)
    return pos[:, 0]


def infer_time_1d(time_arr: np.ndarray) -> np.ndarray:
    time_arr = np.asarray(time_arr)
    if time_arr.ndim == 1:
        return time_arr.reshape(-1)
    return time_arr[:, 0]


def infer_mov_dir_1d(mov_dir: np.ndarray) -> np.ndarray:
    mov_dir = np.asarray(mov_dir)
    if mov_dir.ndim == 1:
        out = mov_dir.reshape(-1)
    else:
        out = mov_dir[:, 0]
    # round to nearest among {-1,0,1} when possible
    out2 = np.zeros_like(out, dtype=int)
    out2[out < -0.5] = -1
    out2[out > 0.5] = 1
    return out2


def validate_data_dict(d: dict, row_id: str):
    for k in REQUIRED_KEYS:
        if k not in d:
            raise KeyError(f"{row_id}: missing key '{k}' in standardized file")
    signal = np.asarray(d["signal"])
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    if signal.ndim != 2:
        raise ValueError(f"{row_id}: signal must be 2D after coercion, got shape {signal.shape}")
    n_time = signal.shape[0]
    for k in ["pos", "mov_dir", "speed", "time"]:
        arr = np.asarray(d[k])
        if arr.shape[0] != n_time:
            raise ValueError(f"{row_id}: key '{k}' length {arr.shape[0]} != signal length {n_time}")
    return signal


def compute_embeddings(signal: np.ndarray):
    # PCA
    pca = PCA(n_components=3)
    emb_pca = pca.fit_transform(signal)

    # UMAP 3D
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    emb_umap = reducer.fit_transform(signal)
    return emb_umap, emb_pca


def direction_to_rgba(direction: np.ndarray):
    colors_arr = []
    for x in direction:
        if x < 0:
            colors_arr.append(DIR_COLORS[-1])
        elif x > 0:
            colors_arr.append(DIR_COLORS[1])
        else:
            colors_arr.append(DIR_COLORS[0])
    return colors_arr


def scatter3d(ax, emb, values, mode="continuous", cmap=None, title=""):
    if mode == "continuous":
        sc = ax.scatter(
            emb[:, 0], emb[:, 1], emb[:, 2],
            c=values, s=POINT_SIZE, cmap=cmap, linewidths=0, alpha=0.9,
        )
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    elif mode == "direction":
        sc = ax.scatter(
            emb[:, 0], emb[:, 1], emb[:, 2],
            c=direction_to_rgba(values), s=POINT_SIZE, linewidths=0, alpha=0.9,
        )
    else:
        raise ValueError(mode)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")


def make_embedding_figure(emb_umap, emb_pca, pos1d, time1d, dir1d, title, save_path):
    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    scatter3d(ax1, emb_umap, pos1d, mode="continuous", cmap=POS_CMAP, title="UMAP · Position")

    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    scatter3d(ax2, emb_umap, time1d, mode="continuous", cmap=TIME_CMAP, title="UMAP · Time")

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    scatter3d(ax3, emb_umap, dir1d, mode="direction", title="UMAP · Direction")

    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    scatter3d(ax4, emb_pca, pos1d, mode="continuous", cmap=POS_CMAP, title="PCA · Position")

    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    scatter3d(ax5, emb_pca, time1d, mode="continuous", cmap=TIME_CMAP, title="PCA · Time")

    ax6 = fig.add_subplot(2, 3, 6, projection="3d")
    scatter3d(ax6, emb_pca, dir1d, mode="direction", title="PCA · Direction")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def make_umap_rotation_movie(emb_umap, pos1d, title, save_base):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        emb_umap[:, 0], emb_umap[:, 1], emb_umap[:, 2],
        c=pos1d, s=POINT_SIZE, cmap=POS_CMAP, linewidths=0, alpha=0.9,
    )
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")

    def update(angle):
        ax.view_init(elev=25, azim=angle)
        return (ax,)

    anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

    mp4_path = save_base + ".mp4"
    gif_path = save_base + ".gif"

    saved = False
    try:
        writer = FFMpegWriter(fps=FPS, bitrate=1800)
        anim.save(mp4_path, writer=writer, dpi=DPI)
        saved = True
    except Exception:
        try:
            writer = PillowWriter(fps=FPS)
            anim.save(gif_path, writer=writer, dpi=DPI)
            saved = True
        except Exception:
            pass

    plt.close(fig)
    return saved


def main():
    ensure_dir(OUTPUT_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Point CSV_PATH to the rebuilt master_dataset_index.csv."
        )

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Could not parse CSV: {CSV_PATH}\n"
            "The uploaded earlier CSV was malformed. Use the rebuilt master CSV.\n"
            f"Original error: {e}"
        )

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    summary = []

    for idx, row in df.iterrows():
        row_id = f"row{idx}_{row.get('mouse_name','unknown')}_{row.get('session_name','unknown')}"
        file_path = row["standardized_file_path"]
        if not isinstance(file_path, str) or not os.path.isfile(file_path):
            summary.append({"row": idx, "status": "missing_file", "path": file_path})
            continue

        try:
            data = load_pickle_standard(file_path)
            signal = validate_data_dict(data, row_id)
            pos1d = infer_pos_1d(data["pos"])
            time1d = infer_time_1d(data["time"])
            dir1d = infer_mov_dir_1d(data["mov_dir"])

            emb_umap, emb_pca = compute_embeddings(signal)

            mouse_name = str(row.get("mouse_name", "unknown_mouse"))
            session_name = str(row.get("session_name", "unknown_session"))
            depth_type = str(row.get("depth_type", "unknown_depth"))
            dataset_name = str(row.get("dataset_name", "unknown_dataset"))

            base_name = sanitize_filename(f"{dataset_name}_{mouse_name}_{session_name}_{depth_type}")
            animal_dir = os.path.join(OUTPUT_DIR, dataset_name, mouse_name)
            ensure_dir(animal_dir)

            fig_path = os.path.join(animal_dir, base_name + "_umap_pca_3d.png")
            movie_base = os.path.join(animal_dir, base_name + "_umap_rotation")

            if OVERWRITE or (not os.path.isfile(fig_path)):
                title = f"{dataset_name} | {mouse_name} | {session_name} | {depth_type}"
                make_embedding_figure(emb_umap, emb_pca, pos1d, time1d, dir1d, title, fig_path)

            movie_saved = make_umap_rotation_movie(
                emb_umap,
                pos1d,
                title=f"UMAP rotation · {dataset_name} | {mouse_name} | {session_name} | {depth_type}",
                save_base=movie_base,
            )

            summary.append({
                "row": idx,
                "status": "ok",
                "mouse_name": mouse_name,
                "session_name": session_name,
                "dataset_name": dataset_name,
                "depth_type": depth_type,
                "n_timepoints": int(signal.shape[0]),
                "n_neurons": int(signal.shape[1]),
                "figure": fig_path,
                "movie_saved": movie_saved,
            })

            print(f"[OK] {row_id}")

        except Exception as e:
            summary.append({"row": idx, "status": "error", "error": str(e), "path": file_path})
            print(f"[ERROR] {row_id}: {e}")

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(OUTPUT_DIR, "embedding_test_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
