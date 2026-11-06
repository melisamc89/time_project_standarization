#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lt_Rot Dataset Manifold Geometry Suite & Comparative Plotting (Sigma = 16)
- Computes Arc Length, Tortuosity, and Dimension-Normalized MEB Radius
  in both UMAP space and Full High-Dimensional Neural Space.
- Runs Mann-Whitney U statistical tests comparing Superficial vs. Deep layers.
- Generates publication-quality violin/box comparative plots with annotated stats.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
import umap

# =============================================================================
# Paths & Settings
# =============================================================================

POSSIBLE_ROOTS = [
    Path("/home/hippolab/Documents/melisa/time_project_standarization"),
    Path("/home/melisamc/Documents/time_project_standarization"),
]
PROJECT_ROOT = next((p for p in POSSIBLE_ROOTS if p.is_dir()), POSSIBLE_ROOTS[0])

MASTER_INDEX_CSV = PROJECT_ROOT / "data" / "metadata" / "master_dataset_index.csv"
OUT_ROOT = PROJECT_ROOT / "analysis" / "nature_2024_layer_geometry_sigma16"

FIXED_FS = 20.0
MIN_SPEED_CM_S = 0.02

LAYER_COLOR_MAP = {
    "superficial": "#9900ff",  # Purple
    "deep": "#cc9900",  # Mustard / Gold
}


# =============================================================================
# Helpers & Unpickler
# =============================================================================

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "numpy.core" in module:
            module = module.replace("numpy.core", "numpy._core")
        return super().find_class(module, name)


def safe_load_pickle(file_path: Path | str) -> dict:
    with open(file_path, "rb") as f:
        try:
            return SafeUnpickler(f).load()
        except Exception:
            f.seek(0)
            return pickle.load(f)


def resolve_path(file_path: str | Path | None) -> Path | None:
    if not file_path or pd.isna(file_path):
        return None
    p = Path(str(file_path).strip())
    if p.exists():
        return p
    if p.with_suffix(".pkl").exists():
        return p.with_suffix(".pkl")
    for r in POSSIBLE_ROOTS:
        for prefix in [
            "/home/melisamc/Documents/time_project_standarization",
            "/home/hippolab/Documents/melisa/time_project_standarization",
        ]:
            if prefix in str(p):
                cand = Path(str(p).replace(prefix, str(r)))
                if cand.exists():
                    return cand
                if cand.with_suffix(".pkl").exists():
                    return cand.with_suffix(".pkl")
    return None


def extract_1d(arr):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def align_vector_to_target_length(vec: np.ndarray | None, target_len: int, is_discrete: bool = False) -> np.ndarray:
    if vec is None:
        return np.zeros(target_len, dtype=int if is_discrete else float)
    vec = extract_1d(vec)
    curr_len = len(vec)
    if curr_len == target_len:
        return vec
    x_orig = np.linspace(0, 1, curr_len)
    x_target = np.linspace(0, 1, target_len)
    if is_discrete:
        idx_nearest = np.clip(np.round(np.linspace(0, curr_len - 1, target_len)).astype(int), 0, curr_len - 1)
        return vec[idx_nearest]
    else:
        return np.interp(x_target, x_orig, vec)


def canonicalize_depth(x: str) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip().lower().replace("-", " ").replace("__", "_")
    if s in ["sup", "superficial", "superficial cells"]:
        return "superficial"
    if s in ["deep", "deep cells"]:
        return "deep"
    return "unknown"


def smooth_calcium_signals(signal_mat: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return signal_mat.copy()
    smoothed = np.zeros_like(signal_mat, dtype=float)
    for i in range(signal_mat.shape[1]):
        smoothed[:, i] = gaussian_filter1d(signal_mat[:, i], sigma=sigma)
    return smoothed


# =============================================================================
# Manifold Geometry Metrics
# =============================================================================

def compute_minimum_enclosing_ball_radius(points: np.ndarray) -> float:
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return float(np.max(distances))


def compute_manifold_geometry_metrics(umap_emb: np.ndarray, raw_signal_sigma16: np.ndarray):
    n_neurons = raw_signal_sigma16.shape[1]
    dim_scale = np.sqrt(n_neurons) if n_neurons > 0 else 1.0

    # UMAP Space
    diffs_umap = np.diff(umap_emb, axis=0)
    arc_lengths_umap = np.linalg.norm(diffs_umap, axis=1)
    total_length_umap = float(np.sum(arc_lengths_umap))
    net_disp_umap = float(np.linalg.norm(umap_emb[-1] - umap_emb[0]))
    tortuosity_umap = float(total_length_umap / net_disp_umap) if net_disp_umap > 1e-6 else np.nan

    # Full Neural Space
    diffs_full = np.diff(raw_signal_sigma16, axis=0)
    arc_lengths_full = np.linalg.norm(diffs_full, axis=1)
    total_length_full = float(np.sum(arc_lengths_full))

    net_disp_full = float(np.linalg.norm(raw_signal_sigma16[-1] - raw_signal_sigma16[0]))
    tortuosity_full = float(total_length_full / net_disp_full) if net_disp_full > 1e-6 else np.nan

    meb_radius_full = compute_minimum_enclosing_ball_radius(raw_signal_sigma16)
    norm_meb_radius_full = float(meb_radius_full / dim_scale)
    norm_arc_length_full = float(total_length_full / dim_scale)

    return {
        "umap_arc_length": total_length_umap,
        "umap_tortuosity": tortuosity_umap,
        "full_space_norm_arc_length": norm_arc_length_full,
        "full_space_tortuosity": tortuosity_full,
        "full_space_norm_meb_radius": norm_meb_radius_full,
    }


# =============================================================================
# Plotting & Statistical Annotation Suite
# =============================================================================

def generate_comparative_plots(df_metrics: pd.DataFrame, plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = [
        ("full_space_norm_arc_length", "Normalized Arc Length (Full Space)", "violin"),
        ("full_space_tortuosity", "Manifold Tortuosity (Full Space)", "violin"),
        ("full_space_norm_meb_radius", "Normalized MEB Radius (Full Space)", "violin"),
        ("umap_arc_length", "Arc Length (UMAP Space)", "violin"),
        ("umap_tortuosity", "Tortuosity (UMAP Space)", "violin")
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 5), dpi=300)
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for idx, (col, title, plot_type) in enumerate(metrics_to_plot):
        ax = axes[idx]
        sub_df = df_metrics.dropna(subset=[col])
        if sub_df.empty:
            continue

        sup_vals = sub_df[sub_df["depth_type"] == "superficial"][col].values
        deep_vals = sub_df[sub_df["depth_type"] == "deep"][col].values

        data_to_plot = [v for v in [sup_vals, deep_vals] if len(v) > 0]
        if len(data_to_plot) == 2:
            colors = [LAYER_COLOR_MAP["superficial"], LAYER_COLOR_MAP["deep"]]
            parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.75)

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Superficial", "Deep"], fontweight="bold")

            # Statistical Mann-Whitney U test & annotation
            try:
                stat, pval = stats.mannwhitneyu(sup_vals, deep_vals, alternative='two-sided')
                sig_str = "***" if pval < 0.001 else (
                    "**" if pval < 0.01 else ("*" if pval < 0.05 else f"p = {pval:.3f}"))
                y_max = max(np.max(sup_vals), np.max(deep_vals))
                y_min = min(np.min(sup_vals), np.min(deep_vals))
                span = y_max - y_min if y_max > y_min else 1.0

                ax.plot([1, 1, 2, 2],
                        [y_max + 0.02 * span, y_max + 0.05 * span, y_max + 0.05 * span, y_max + 0.02 * span],
                        color="black", lw=1.0)
                ax.text(1.5, y_max + 0.06 * span, sig_str, ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.set_ylim(y_min - 0.1 * span, y_max + 0.15 * span)
            except Exception:
                pass

        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.set_ylabel("Metric Value", fontsize=9, fontweight="bold")
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Lt_Rot Dataset: Layer Geometry Comparison ($\sigma = 16$)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in [".png", ".svg"]:
        fig.savefig(plot_dir / f"layer_geometry_comparison_plots{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved comparative distribution plots -> {plot_dir}")


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def main():
    print("=" * 80)
    print("LT_ROT DATASET: SIGMA = 16 MANIFOLD GEOMETRY SUITE & STATS")
    print("=" * 80)

    if not MASTER_INDEX_CSV.exists():
        print(f"[ERROR] Master index CSV not found at: {MASTER_INDEX_CSV}")
        return

    master_df = pd.read_csv(MASTER_INDEX_CSV)

    df_sub = master_df[
        (master_df["dataset_name"].astype(str).str.lower() == "lt_rot") &
        (master_df["session_name"].astype(str).str.lower().str.contains("lt"))
        ].copy()

    print(f"Found {len(df_sub)} candidate sessions in lt_rot dataset.")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    geometry_rows = []

    for _, row in df_sub.iterrows():
        mouse_name = str(row["mouse_name"])
        sess_name = str(row["session_name"])
        depth_type = canonicalize_depth(row.get("depth_type", "unknown"))

        if depth_type not in ["superficial", "deep"]:
            continue

        fpath = resolve_path(row.get("standardized_file_path"))
        if fpath is None or not fpath.exists():
            continue

        payload = safe_load_pickle(fpath)
        signal_mat = np.asarray(payload.get("signal", payload.get("clean_traces")))
        raw_pos = payload.get("pos", payload.get("position"))

        if signal_mat.ndim != 2:
            continue

        t_cand0, t_cand1 = signal_mat.shape
        t_ref = len(raw_pos) if raw_pos is not None else max(t_cand0, t_cand1)
        if t_cand0 != t_ref and t_cand1 == t_ref:
            signal_mat = signal_mat.T

        n_time_target, n_cells = signal_mat.shape

        raw_pos_arr = np.asarray(raw_pos)
        pos_x = raw_pos_arr[:, 0] if (raw_pos_arr.ndim == 2 and raw_pos_arr.shape[1] >= 2) else raw_pos_arr
        pos = align_vector_to_target_length(pos_x, n_time_target, is_discrete=False)

        fs = FIXED_FS
        time_vec = np.arange(n_time_target) / fs
        raw_mov_dir = payload.get("mov_dir", payload.get("direction"))
        mov_dir = align_vector_to_target_length(raw_mov_dir, n_time_target, is_discrete=True)

        speed = np.abs(np.gradient(pos, 1.0 / fs))
        valid_mask = (speed >= MIN_SPEED_CM_S) & (mov_dir != 0) & np.isfinite(pos)

        sig_valid = signal_mat[valid_mask, :]
        if sig_valid.shape[0] < 50:
            continue

        # Apply Sigma = 16 Smoothing
        sig_f16 = smooth_calcium_signals(sig_valid, sigma=16)

        try:
            umap_model = umap.UMAP(n_components=3, n_neighbors=120, min_dist=0.1, random_state=42)
            umap_emb = umap_model.fit_transform(sig_f16)
        except Exception:
            continue

        metrics = compute_manifold_geometry_metrics(umap_emb, sig_f16)
        metrics.update({
            "mouse_name": mouse_name,
            "session_name": sess_name,
            "depth_type": depth_type,
            "n_neurons": n_cells
        })
        geometry_rows.append(metrics)

    if not geometry_rows:
        print("[WARNING] No valid sessions processed successfully.")
        return

    df_metrics = pd.DataFrame(geometry_rows)
    metrics_csv = OUT_ROOT / "lt_rot_sigma16_full_space_geometry_comparison.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\n[OK] Saved metrics table -> {metrics_csv}")

    # Statistical Comparison & Console Reporting
    print("\n" + "=" * 65)
    print("STATISTICAL REPORT: SUPERFICIAL vs. DEEP ($\sigma = 16$)")
    print("=" * 65)

    metric_cols = [
        "full_space_norm_arc_length",
        "full_space_tortuosity",
        "full_space_norm_meb_radius",
        "umap_arc_length",
        "umap_tortuosity"
    ]

    for col in metric_cols:
        sup_vals = df_metrics[df_metrics["depth_type"] == "superficial"][col].dropna()
        deep_vals = df_metrics[df_metrics["depth_type"] == "deep"][col].dropna()

        if len(sup_vals) > 0 and len(deep_vals) > 0:
            stat, pval = stats.mannwhitneyu(sup_vals, deep_vals, alternative="two-sided")
            print(f"Metric: {col}")
            print(f"  Superficial (n={len(sup_vals)}): Mean = {sup_vals.mean():.3f} ± {sup_vals.sem():.3f}")
            print(f"  Deep        (n={len(deep_vals)}): Mean = {deep_vals.mean():.3f} ± {deep_vals.sem():.3f}")
            print(f"  Mann-Whitney U statistic: {stat:.2f} | p-value: {pval:.4f}\n")

    # Generate Comparative Plots
    generate_comparative_plots(df_metrics, OUT_ROOT / "summary_plots")

    print("=" * 80)
    print("ANALYSIS AND PLOTTING SUITE SUCCESSFULLY COMPLETED!")
    print(f"Output Directory Root: {OUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()