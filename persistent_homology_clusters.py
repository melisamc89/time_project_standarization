#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-Safe Persistent Homology and 3D Manifolds Pipeline.
Optimized to prevent Out-Of-Memory (OOM) crashes in IDEs/PyCharm.
"""

from __future__ import annotations

import copy
import gc
import glob
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from ripser import ripser as tda
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import pairwise_distances
import umap

# =============================================================================
# Paths & Settings
# =============================================================================

POSSIBLE_ROOTS = [
    Path("/home/hippolab/Documents/melisa/time_project_standarization"),
    Path("/home/melisamc/Documents/time_project_standarization"),
]
PROJECT_ROOT = next((p for p in POSSIBLE_ROOTS if p.exists()), POSSIBLE_ROOTS[0])

TRAIN_LABELED_CSV = (
        PROJECT_ROOT
        / "analysis"
        / "mir_training_and_transfer"
        / "training"
        / "data"
        / "learning_last_session_mir_labeled_neurons.csv"
)
STANDARDIZED_BASE_DIR = PROJECT_ROOT / "data" / "standardized_sessions"

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "tda_julio_exact_replication_mem_safe"
BARCODES_ORIG_DIR = OUTPUT_DIR / "barcodes_original_space"
BARCODES_UMAP_DIR = OUTPUT_DIR / "barcodes_umap_space"
MANIFOLDS_DIR = OUTPUT_DIR / "manifolds_3d"
TABLES_DIR = OUTPUT_DIR / "tables"

for p in [OUTPUT_DIR, BARCODES_ORIG_DIR, BARCODES_UMAP_DIR, MANIFOLDS_DIR, TABLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CLUSTER_COLORS = {
    "Time": "#bce784ff",  # Light Green
    "Direction": "#66cef4ff",  # Sky Blue
    "Position": "#ec8ef8ff",  # Violet / Orchid
    "Unassigned": "#bbbcc0ff",  # Gray
    "All": "#111111",  # Deep Charcoal
}

CLUSTER_ID_TO_NAME = {
    0: "Time",
    1: "Direction",
    2: "Position",
    10: "Unassigned",
}

ORDERED_GROUPS = ["Time", "Direction", "Position", "Unassigned", "All"]

# Memory-Safe Subsampling:
# 1200 points keeps the 2-simplex count (~2.8e8) safely within 2-3 GB RAM.
N_SUBSAMPLE_RAW = 1200
N_SUBSAMPLE_UMAP = 1500


# =============================================================================
# Helper Functions
# =============================================================================

def filter_noisy_outliers(data: np.ndarray, D: np.ndarray | None = None) -> np.ndarray:
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    D_work = D.copy()
    np.fill_diagonal(D_work, np.nan)
    nn_dist = np.sum(D_work < np.nanpercentile(D_work, 5), axis=1)
    noise_idx = nn_dist < np.percentile(nn_dist, 20)
    del D_work
    return noise_idx


def plot_betti_numbers(diagrams: List[np.ndarray], title: str, cluster_name: str, save_path: Path):
    c = CLUSTER_COLORS.get(cluster_name, "#222222")
    dgms = copy.deepcopy(diagrams)

    finite_maxes = [np.nanmax(h[np.isfinite(h)]) for h in dgms if len(h) > 0 and np.any(np.isfinite(h))]
    max_dist = np.nanmax(finite_maxes) * 1.05 if len(finite_maxes) > 0 else 3.0

    if len(dgms[0]) > 0:
        dgms[0][~np.isfinite(dgms[0])] = max_dist
    max_len = max([np.nanmax(h) for h in dgms if len(h) > 0] + [1.0])

    to_plot = []
    for curr_h in dgms:
        if len(curr_h) == 0:
            to_plot.append(np.empty((0, 2)))
            continue
        bar_lens = curr_h[:, 1] - curr_h[:, 0]
        plot_h = curr_h[(-bar_lens).argsort()[:30]]
        to_plot.append(plot_h[np.argsort(plot_h[:, 0]), :])

    fig = plt.figure(figsize=(9, 7.5), dpi=300)
    gs = gridspec.GridSpec(len(dgms), 1, hspace=0.35)

    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[curr_betti, 0])
        for i, interval in enumerate(reversed(curr_bar)):
            ax.plot([interval[0], interval[1]], [i, i], color=c, lw=1.8)

        ax.set_ylabel(f"$H_{curr_betti}$", fontsize=11, fontweight="bold")
        ax.set_xlim([-0.05, max_len * 1.05])
        ax.set_ylim([-1, max(len(curr_bar), 1)])
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.3)
        ax.spines["bottom"].set_linewidth(1.3)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    fig.savefig(save_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_multicluster_3d_grid(
        mouse_name: str,
        session_name: str,
        cluster_embeddings: Dict[str, Dict],
        pos: np.ndarray,
        dir_mat: np.ndarray,
        time_vec: np.ndarray,
        save_path: Path
):
    valid_clusters = [g for g in ORDERED_GROUPS if g in cluster_embeddings]
    n_rows = len(valid_clusters)
    n_cols = 3

    fig = plt.figure(figsize=(15, 4.2 * n_rows), dpi=250)
    dir_cmap = ListedColormap(["#0000ff", "#666666", "#add8e6"])

    for row_idx, grp in enumerate(valid_clusters):
        emb = cluster_embeddings[grp]["emb"]
        n_cells = cluster_embeddings[grp]["n_cells"]

        # Position
        ax1 = fig.add_subplot(n_rows, n_cols, row_idx * 3 + 1, projection="3d")
        p_val = pos[:, 0] if pos.ndim > 1 else pos
        sc1 = ax1.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=p_val, cmap="magma", s=1.5, alpha=0.85)
        fig.colorbar(sc1, ax=ax1, fraction=0.03, pad=0.05, shrink=0.7)
        ax1.set_title(f"{grp.lower()} | position (N={n_cells})", fontsize=10, fontweight="bold")
        ax1.view_init(elev=20, azim=-60)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])

        # Time
        ax2 = fig.add_subplot(n_rows, n_cols, row_idx * 3 + 2, projection="3d")
        sc2 = ax2.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=time_vec, cmap="YlGn_r", s=1.5, alpha=0.85)
        fig.colorbar(sc2, ax=ax2, fraction=0.03, pad=0.05, shrink=0.7)
        ax2.set_title(f"{grp.lower()} | time (N={n_cells})", fontsize=10, fontweight="bold")
        ax2.view_init(elev=20, azim=-60)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_zticklabels([])

        # Direction
        ax3 = fig.add_subplot(n_rows, n_cols, row_idx * 3 + 3, projection="3d")
        d_val = dir_mat[:, 0] if dir_mat.ndim > 1 else dir_mat
        sc3 = ax3.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=d_val, cmap=dir_cmap, s=1.5, alpha=0.85)
        fig.colorbar(sc3, ax=ax3, fraction=0.03, pad=0.05, shrink=0.7)
        ax3.set_title(f"{grp.lower()} | direction (N={n_cells})", fontsize=10, fontweight="bold")
        ax3.view_init(elev=20, azim=-60)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_zticklabels([])

    fig.suptitle(f"{mouse_name} | {session_name} | 3D Manifolds per Cluster", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved 3D Grid -> {save_path.name}")


def resolve_training_file(mouse_name: str, session_name: str, depth_type: str) -> Path | None:
    patterns = [
        str(STANDARDIZED_BASE_DIR / "**" / f"*{mouse_name}*{session_name}*.pkl"),
        str(STANDARDIZED_BASE_DIR / "**" / f"*{mouse_name}*.pkl"),
    ]
    for pat in patterns:
        cands = glob.glob(pat, recursive=True)
        if cands:
            learning_cands = [c for c in cands if "learning" in c.lower()]
            cands_to_check = learning_cands if learning_cands else cands
            for c in cands_to_check:
                if str(session_name) in c:
                    return Path(c)
            for c in cands_to_check:
                if str(depth_type).lower() in c.lower():
                    return Path(c)
            return Path(cands_to_check[0])
    return None


# =============================================================================
# Execution Loop
# =============================================================================

def run_memory_safe_pipeline():
    if not TRAIN_LABELED_CSV.exists():
        raise FileNotFoundError(f"Missing training CSV: {TRAIN_LABELED_CSV}")

    df_train = pd.read_csv(TRAIN_LABELED_CSV)
    if "cluster_name" not in df_train.columns:
        df_train["cluster_name"] = df_train["cluster_id"].map(CLUSTER_ID_TO_NAME).fillna("Unassigned")

    session_groups = df_train.groupby(["mouse_name", "session_name", "depth_type"])
    print(f"[INFO] Running memory-safe TDA on {len(session_groups)} sessions.")

    records = []

    for (mouse_name, session_name, depth_type), df_sess in session_groups:
        pkl_path = resolve_training_file(mouse_name, session_name, depth_type)
        if pkl_path is None or not pkl_path.exists():
            continue

        with open(pkl_path, "rb") as f:
            std_data = pickle.load(f)

        signal = std_data.get("signal")
        if signal is None or signal.shape[1] < 10:
            continue

        pos = std_data.get("position", np.zeros((signal.shape[0], 1)))
        dir_mat = std_data.get("direction", np.zeros((signal.shape[0], 1)))
        time_vec = np.arange(signal.shape[0])

        print(f"\n=======================================================")
        print(f"--> Mouse: {mouse_name} | Session: {session_name} | Total Neurons: {signal.shape[1]}")
        print(f"=======================================================")

        cluster_embeddings = {}

        for grp in ORDERED_GROUPS:
            if grp == "All":
                grp_col_indices = np.arange(signal.shape[1])
            else:
                sub_cells = df_sess[df_sess["cluster_name"] == grp]
                if sub_cells.empty:
                    continue
                grp_col_indices = sub_cells["neuron_index"].values.astype(int)
                grp_col_indices = [i for i in grp_col_indices if i < signal.shape[1]]

            n_cells = len(grp_col_indices)
            if n_cells < 5:
                print(f"    [SKIP] {grp}: only {n_cells} neurons.")
                continue

            sub_signal = signal[:, grp_col_indices]

            # Fit 3D UMAP manifold
            model_umap = umap.UMAP(n_neighbors=120, n_components=3, min_dist=0.1, random_state=42)
            emb_umap = model_umap.fit_transform(sub_signal)
            cluster_embeddings[grp] = {"emb": emb_umap, "n_cells": n_cells}

            # -----------------------------------------------------------------
            # Space 1: Original Signal Space (Mem-Safe Subsampling)
            # -----------------------------------------------------------------
            D_orig = pairwise_distances(sub_signal)
            noise_idx_o = filter_noisy_outliers(sub_signal, D=D_orig)
            clean_signal = sub_signal[~noise_idx_o, :]
            del D_orig

            n_sub_o = min(N_SUBSAMPLE_RAW, clean_signal.shape[0])
            np.random.seed(42)
            idx_sub_o = np.random.permutation(clean_signal.shape[0])[:n_sub_o]
            sclean_signal = clean_signal[idx_sub_o, :]

            D_sub_o = pairwise_distances(sclean_signal)
            thresh_o = float(np.nanpercentile(D_sub_o, 98))
            del D_sub_o

            # Run Ripser (maxdim=2 on N=1200 uses <2GB RAM)
            barcodes_orig = tda(sclean_signal, maxdim=2, thresh=thresh_o)["dgms"]

            plot_betti_numbers(
                barcodes_orig,
                title=f"Persistent Homology (Original Space)\n{mouse_name} | {grp} (N={n_cells})",
                cluster_name=grp,
                save_path=BARCODES_ORIG_DIR / f"{mouse_name}_{grp.lower()}_orig_barcodes",
            )

            # -----------------------------------------------------------------
            # Space 2: 3D UMAP Space
            # -----------------------------------------------------------------
            D_umap = pairwise_distances(emb_umap)
            noise_idx_u = filter_noisy_outliers(emb_umap, D=D_umap)
            clean_emb = emb_umap[~noise_idx_u, :]
            del D_umap

            n_sub_u = min(N_SUBSAMPLE_UMAP, clean_emb.shape[0])
            np.random.seed(42)
            idx_sub_u = np.random.permutation(clean_emb.shape[0])[:n_sub_u]
            sclean_emb = clean_emb[idx_sub_u, :]

            D_sub_u = pairwise_distances(sclean_emb)
            thresh_u = float(np.nanpercentile(D_sub_u, 98))
            del D_sub_u

            barcodes_umap = tda(sclean_emb, maxdim=2, thresh=thresh_u)["dgms"]

            plot_betti_numbers(
                barcodes_umap,
                title=f"Persistent Homology (3D UMAP Space)\n{mouse_name} | {grp} (N={n_cells})",
                cluster_name=grp,
                save_path=BARCODES_UMAP_DIR / f"{mouse_name}_{grp.lower()}_umap_barcodes",
            )

            # Exact Prominence Ratio: L1 / L2
            def get_ratio(dgms):
                h1 = dgms[1]
                if len(h1) >= 2:
                    bar_lens = h1[:, 1] - h1[:, 0]
                    bar_lens = bar_lens[(-bar_lens).argsort()]
                    return float(bar_lens[0] / bar_lens[1]), float(bar_lens[0])
                elif len(h1) == 1:
                    l = float(h1[0, 1] - h1[0, 0])
                    return l, l
                return 0.0, 0.0

            ratio_orig, max_orig = get_ratio(barcodes_orig)
            ratio_umap, max_umap = get_ratio(barcodes_umap)

            records.append({
                "mouse": mouse_name,
                "session": session_name,
                "group": grp,
                "n_neurons": n_cells,
                "orig_ratio_L1_L2": ratio_orig,
                "orig_max_L1": max_orig,
                "umap_ratio_L1_L2": ratio_umap,
                "umap_max_L1": max_umap,
            })

            print(f"    {grp:11s} (N={n_cells:3d}) | Orig Ratio: {ratio_orig:5.2f} | UMAP Ratio: {ratio_umap:5.2f}")
            gc.collect()

        # Render 3D Manifold Grid
        grid_out = MANIFOLDS_DIR / f"{mouse_name}_{session_name}_3d_manifolds_grid"
        plot_multicluster_3d_grid(
            mouse_name=mouse_name,
            session_name=session_name,
            cluster_embeddings=cluster_embeddings,
            pos=pos,
            dir_mat=dir_mat,
            time_vec=time_vec,
            save_path=grid_out,
        )

        del std_data, signal
        gc.collect()

    df_summary = pd.DataFrame(records)
    summary_path = TABLES_DIR / "tda_julio_replication_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[OK] Pipeline completed! Exported summary table -> {summary_path}")


if __name__ == "__main__":
    run_memory_safe_pipeline()