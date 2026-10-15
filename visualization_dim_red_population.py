#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.decomposition import PCA
import umap

# =============================================================================
# Paths
# =============================================================================

CSV_PATH = "/home/hippolab/Documents/melisa/time_project_standarization/data/metadata/master_dataset_index.csv"
OUTPUT_DIR = "/home/hippolab/Documents/melisa/time_project_standarization/analysis/umap_pca_per_row"

PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")
EMBEDDING_DATA_DIR = os.path.join(OUTPUT_DIR, "embedding_data")

ANALYSIS_INDEX_CSV = os.path.join(OUTPUT_DIR, "umap_pca_analysis_index.csv")

MAX_ROWS = None

# =============================================================================
# Embedding parameters
# =============================================================================

UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
N_COMPONENTS = 3

N_POS_BINS = 30
MIN_COUNT_PER_BIN = 5
PLANE_TILT_DEG = 30.0


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


def sanitize_for_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


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


def get_pos_1d(pos):
    pos = np.asarray(pos)
    if pos.ndim == 1:
        return pos
    if pos.ndim == 2:
        return pos[:, 0]
    raise ValueError(f"Unexpected pos shape: {pos.shape}")


def base_name_from_row(row):
    return (
        f"{sanitize_for_filename(row['dataset_name'])}_"
        f"{sanitize_for_filename(row['mouse_name'])}_"
        f"{sanitize_for_filename(row['session_name'])}_"
        f"{sanitize_for_filename(row['depth_type'])}"
    )


def save_figure_both(fig, out_base):
    png_path = os.path.join(PNG_DIR, out_base + ".png")
    svg_path = os.path.join(SVG_DIR, out_base + ".svg")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")


def style_3d_axis(ax, title):
    ax.set_title(title)
    ax.grid(False)

    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except Exception:
        pass

    try:
        ax.xaxis.pane.set_edgecolor("white")
        ax.yaxis.pane.set_edgecolor("white")
        ax.zaxis.pane.set_edgecolor("white")
    except Exception:
        pass

    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")


def get_direction_colors():
    cmap = ListedColormap(["blue", "gray", "lightblue"])  # -1, 0, 1
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    return cmap, norm


def compute_position_direction_plane_normal(
    embedding,
    pos_1d,
    mov_dir,
    n_pos_bins=30,
    min_count_per_bin=5
):
    """
    Estimate a plane from position bins computed separately for mov_dir = -1 and +1.
    The returned normal is used only to choose camera angles, not to rotate the data.
    """
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

    pos_min = np.min(pos_1d)
    pos_max = np.max(pos_1d)
    if pos_max <= pos_min:
        raise ValueError("Position has zero range.")

    bin_edges = np.linspace(pos_min, pos_max, n_pos_bins + 1)

    centroids = []
    for d in [-1, 1]:
        mask_d = mov_dir == d
        if np.sum(mask_d) == 0:
            continue

        pos_d = pos_1d[mask_d]
        emb_d = embedding[mask_d]
        bin_ids = np.digitize(pos_d, bin_edges[1:-1], right=False)

        for b in range(n_pos_bins):
            idx = np.where(bin_ids == b)[0]
            if len(idx) >= min_count_per_bin:
                centroids.append(np.mean(emb_d[idx], axis=0))

    centroids = np.asarray(centroids)

    if centroids.shape[0] < 3:
        raise ValueError("Not enough centroids to estimate plane.")

    center = np.mean(centroids, axis=0)
    centroids_centered = centroids - center

    pca = PCA(n_components=3)
    pca.fit(centroids_centered)

    basis = pca.components_.copy()
    normal = basis[2].copy()
    normal = normal / np.linalg.norm(normal)

    return normal, basis, center, centroids


def normal_to_view_angles(normal, tilt_deg=30.0):
    """
    Convert a plane normal into matplotlib view angles.
    We look approximately along the plane normal, then tilt by tilt_deg
    so the 3rd dimension remains visible.
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    x, y, z = normal

    azim = np.degrees(np.arctan2(y, x))
    elev = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))

    elev = elev - tilt_deg

    return float(elev), float(azim)


def plot_embedding_triplet(
    embedding,
    pos_1d,
    mov_dir,
    time_1d,
    method_name,
    row_idx,
    row,
    elev=None,
    azim=None
):
    fig = plt.figure(figsize=(18, 6))

    title = (
        f"{method_name} | row={row_idx} | dataset={row['dataset_name']} | "
        f"mouse={row['mouse_name']} | session={row['session_name']} | "
        f"depth={row['depth_type']}"
    )
    fig.suptitle(title, fontsize=13)

    # position
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    sc1 = ax1.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=pos_1d, cmap="magma", s=4, alpha=0.9
    )
    style_3d_axis(ax1, "Colored by position")
    if elev is not None and azim is not None:
        ax1.view_init(elev=elev, azim=azim)
    cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.03, pad=0.04)
    cbar1.set_label("position")

    # direction
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    dir_cmap, dir_norm = get_direction_colors()
    sc2 = ax2.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=mov_dir, cmap=dir_cmap, norm=dir_norm, s=4, alpha=0.9
    )
    style_3d_axis(ax2, "Colored by direction")
    if elev is not None and azim is not None:
        ax2.view_init(elev=elev, azim=azim)
    cbar2 = fig.colorbar(sc2, ax=ax2, fraction=0.03, pad=0.04, ticks=[-1, 0, 1])
    cbar2.set_label("direction")

    # time
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    sc3 = ax3.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=time_1d, cmap="YlGn_r", s=4, alpha=0.9
    )
    style_3d_axis(ax3, "Colored by time")
    if elev is not None and azim is not None:
        ax3.view_init(elev=elev, azim=azim)
    cbar3 = fig.colorbar(sc3, ax=ax3, fraction=0.03, pad=0.04)
    cbar3.set_label("time")

    return fig


# =============================================================================
# Analysis index CSV schema
# =============================================================================

ANALYSIS_INDEX_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "n_neurons",
    "n_timepoints",
    "analysis_file_name",
    "analysis_file_path",
]


def init_analysis_index_csv(csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_INDEX_COLUMNS)
        writer.writeheader()


def append_analysis_index_row(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_INDEX_COLUMNS)
        writer.writerow(row)


# =============================================================================
# Main per-row processing
# =============================================================================

def process_row(row_idx, row):
    standardized_file_path = row["standardized_file_path"]

    if not os.path.isfile(standardized_file_path):
        print(f"[WARNING] Missing file for row {row_idx}: {standardized_file_path}")
        return False, None

    data = load_pickle_standard(standardized_file_path)

    required_keys = ["signal", "pos", "mov_dir", "time"]
    for key in required_keys:
        if key not in data:
            print(f"[WARNING] Row {row_idx}: missing key '{key}' in {standardized_file_path}")
            return False, None

    signal = ensure_2d(data["signal"])
    pos_1d = ensure_1d(get_pos_1d(data["pos"]))
    mov_dir = ensure_1d(data["mov_dir"])
    time_1d = ensure_1d(data["time"])

    speed = ensure_1d(data["speed"]) if data.get("speed", None) is not None else None
    trial_id = ensure_1d(data["trial_id"]) if data.get("trial_id", None) is not None else None
    inner_time = ensure_1d(data["inner_time"]) if data.get("inner_time", None) is not None else None

    n_time = signal.shape[0]
    n_neurons = signal.shape[1]

    if len(pos_1d) != n_time:
        raise ValueError(f"Row {row_idx}: pos length mismatch ({len(pos_1d)} vs {n_time})")
    if len(mov_dir) != n_time:
        raise ValueError(f"Row {row_idx}: mov_dir length mismatch ({len(mov_dir)} vs {n_time})")
    if len(time_1d) != n_time:
        raise ValueError(f"Row {row_idx}: time length mismatch ({len(time_1d)} vs {n_time})")
    if speed is not None and len(speed) != n_time:
        raise ValueError(f"Row {row_idx}: speed length mismatch ({len(speed)} vs {n_time})")
    if trial_id is not None and len(trial_id) != n_time:
        raise ValueError(f"Row {row_idx}: trial_id length mismatch ({len(trial_id)} vs {n_time})")
    if inner_time is not None and len(inner_time) != n_time:
        raise ValueError(f"Row {row_idx}: inner_time length mismatch ({len(inner_time)} vs {n_time})")

    if n_neurons < 1:
        raise ValueError(f"Row {row_idx}: signal has no neurons")

    # PCA
    pca_model = PCA(n_components=N_COMPONENTS)
    pca_embedding = pca_model.fit_transform(signal)

    # UMAP
    umap_model = umap.UMAP(
        n_components=N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    umap_embedding = umap_model.fit_transform(signal)

    # Compute camera angles only (do not rotate embedding)
    umap_normal, umap_basis, umap_center, umap_centroids = \
        compute_position_direction_plane_normal(
            umap_embedding,
            pos_1d,
            mov_dir,
            n_pos_bins=N_POS_BINS,
            min_count_per_bin=MIN_COUNT_PER_BIN
        )
    umap_elev, umap_azim = normal_to_view_angles(umap_normal, tilt_deg=PLANE_TILT_DEG)

    pca_normal, pca_basis, pca_center, pca_centroids = \
        compute_position_direction_plane_normal(
            pca_embedding,
            pos_1d,
            mov_dir,
            n_pos_bins=N_POS_BINS,
            min_count_per_bin=MIN_COUNT_PER_BIN
        )
    pca_elev, pca_azim = normal_to_view_angles(pca_normal, tilt_deg=PLANE_TILT_DEG)

    out_base = base_name_from_row(row)

    # Save figures
    fig_umap = plot_embedding_triplet(
        umap_embedding, pos_1d, mov_dir, time_1d,
        method_name=f"UMAP | elev={umap_elev:.1f}, azim={umap_azim:.1f}",
        row_idx=row_idx,
        row=row,
        elev=umap_elev,
        azim=umap_azim
    )
    save_figure_both(fig_umap, out_base + "_umap")

    fig_pca = plot_embedding_triplet(
        pca_embedding, pos_1d, mov_dir, time_1d,
        method_name=f"PCA | elev={pca_elev:.1f}, azim={pca_azim:.1f}",
        row_idx=row_idx,
        row=row,
        elev=pca_elev,
        azim=pca_azim
    )
    save_figure_both(fig_pca, out_base + "_pca")

    # Save embedding/behavior data file
    analysis_file_name = out_base + "_embedding_behavior.pkl"
    analysis_file_path = os.path.join(EMBEDDING_DATA_DIR, analysis_file_name)

    analysis_payload = {
        "metadata": {
            "dataset_name": row["dataset_name"],
            "mouse_name": row["mouse_name"],
            "session_name": row["session_name"],
            "depth_type": row["depth_type"],
            "n_neurons": n_neurons,
            "n_timepoints": n_time,
            "source_standardized_file_path": standardized_file_path,
            "umap_params": {
                "n_neighbors": UMAP_N_NEIGHBORS,
                "min_dist": UMAP_MIN_DIST,
                "random_state": UMAP_RANDOM_STATE,
                "n_components": N_COMPONENTS,
            },
            "pca_params": {
                "n_components": N_COMPONENTS,
            },
            "plane_view_params": {
                "n_pos_bins": N_POS_BINS,
                "min_count_per_bin": MIN_COUNT_PER_BIN,
                "tilt_deg": PLANE_TILT_DEG,
            },
            "umap_view_angles": {
                "elev": umap_elev,
                "azim": umap_azim,
            },
            "pca_view_angles": {
                "elev": pca_elev,
                "azim": pca_azim,
            },
        },
        "umap": umap_embedding,
        "pca": pca_embedding,
        "behavior": {
            "pos": pos_1d,
            "time": time_1d,
            "mov_dir": mov_dir,
            "speed": speed,
            "trial_id": trial_id,
            "inner_time": inner_time,
        },
        "plane_fit": {
            "umap_normal": umap_normal,
            "umap_basis": umap_basis,
            "umap_center": umap_center,
            "umap_centroids": umap_centroids,
            "pca_normal": pca_normal,
            "pca_basis": pca_basis,
            "pca_center": pca_center,
            "pca_centroids": pca_centroids,
        },
    }
    save_pickle_standard(analysis_file_path, analysis_payload)
    print(f"[OK] Saved embedding data: {analysis_file_path}")

    analysis_index_row = {
        "dataset_name": row["dataset_name"],
        "mouse_name": row["mouse_name"],
        "session_name": row["session_name"],
        "depth_type": row["depth_type"],
        "n_neurons": n_neurons,
        "n_timepoints": n_time,
        "analysis_file_name": analysis_file_name,
        "analysis_file_path": analysis_file_path,
    }

    return True, analysis_index_row


# =============================================================================
# Main & Filtering Logic
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute UMAP and PCA embeddings with behavior alignment."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Dataset name (e.g. 'intervention', 'learning').",
    )
    parser.add_argument(
        "--animal",
        "--mouse",
        "-m",
        type=str,
        default=None,
        help="Animal/Mouse name (e.g. '032' or 'M032'). Partial matches supported.",
    )
    parser.add_argument(
        "--session",
        "-s",
        type=str,
        default=None,
        help="Session name (e.g. 'basal', '1707', 'S1'). Partial matches supported.",
    )
    return parser.parse_args()


def filter_dataframe(df, dataset=None, animal=None, session=None):
    filtered = df.copy()

    if dataset is not None:
        dataset_str = str(dataset).strip()
        mask = filtered["dataset_name"].astype(str) == dataset_str
        if not mask.any():
            print(f"[ERROR] Dataset '{dataset_str}' not found.")
            print(f"Available datasets: {list(df['dataset_name'].unique())}")
            return pd.DataFrame()
        filtered = filtered[mask]

    if animal is not None:
        animal_str = str(animal).strip()
        mask = filtered["mouse_name"].astype(str).str.contains(animal_str, case=False, na=False)
        if not mask.any():
            print(f"[ERROR] Animal matching '{animal_str}' not found.")
            print(f"Available animals: {list(filtered['mouse_name'].unique())}")
            return pd.DataFrame()
        filtered = filtered[mask]

    if session is not None:
        session_str = str(session).strip()
        mask = filtered["session_name"].astype(str).str.contains(session_str, case=False, na=False)
        if not mask.any():
            print(f"[ERROR] Session matching '{session_str}' not found.")
            print(f"Available sessions: {list(filtered['session_name'].unique())}")
            return pd.DataFrame()
        filtered = filtered[mask]

    return filtered.reset_index(drop=True)


def main(dataset=None, animal=None, session=None):
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)
    ensure_dir(EMBEDDING_DATA_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    is_filtered_run = any(arg is not None for arg in (dataset, animal, session))

    # Apply filters
    if is_filtered_run:
        df = filter_dataframe(df, dataset=dataset, animal=animal, session=session)
        if df.empty:
            print("[ABORT] No matching rows found.")
            return
        print(f"\n[INFO] Running for {len(df)} matching row(s):")
        for idx, r in df[["dataset_name", "mouse_name", "session_name"]].iterrows():
            print(f"  [{idx}] {r['dataset_name']} | {r['mouse_name']} | {r['session_name']}")
    else:
        print(f"Running across complete database ({len(df)} rows)")

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    # Manage analysis index CSV: overwrite if complete run; preserve & append if filtered run
    if not is_filtered_run:
        init_analysis_index_csv(ANALYSIS_INDEX_CSV)
    else:
        if not os.path.isfile(ANALYSIS_INDEX_CSV):
            init_analysis_index_csv(ANALYSIS_INDEX_CSV)

    print(f"\nSaving PNG to: {PNG_DIR}")
    print(f"Saving SVG to: {SVG_DIR}")
    print(f"Saving embedding files to: {EMBEDDING_DATA_DIR}")
    print(f"Saving analysis index CSV to: {ANALYSIS_INDEX_CSV}\n")

    ok_count = 0
    fail_count = 0

    for row_idx, row in df.iterrows():
        try:
            success, analysis_index_row = process_row(row_idx, row)
            if success:
                ok_count += 1
                append_analysis_index_row(ANALYSIS_INDEX_CSV, analysis_index_row)
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1
            print(f"[ERROR] Row {row_idx}: {e}")

    print("\nDone.")
    print(f"Successful entries: {ok_count}")
    print(f"Failed entries: {fail_count}")


if __name__ == "__main__":
    args = parse_args()
    main(dataset=args.dataset, animal=args.animal, session=args.session)