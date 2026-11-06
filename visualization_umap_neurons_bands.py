#!/usr/init/env python3
# -*- coding: utf-8 -*-

"""
Unified Ultraslow & Trial Frequency Oscillatory Sequences Analysis Pipeline
Targeted for Specific Cells [74, 5, 44] in Animal GC2 (lt_rot Dataset)
Features:
- ONE single master visualization figure per cell combining all filter states, traces, and dual-view manifolds.
- Raw Activity (3rd UMAP row) is explicitly color-coded in log10 scale.
- Exports both Linear and Log10 color-scaled master figures in PNG and SVG formats.
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
OUT_ROOT = PROJECT_ROOT / "analysis" / "nature_2024_gc2_ultimate_master_sheet_lograw"

FIXED_FS = 20.0
N_POS_BINS = 50
MIN_SPEED_CM_S = 0.02
N_SHUFFLES_PLACE = 30

SELECTED_CELL_INDICES = [19]
FILTER_SIGMAS = [4, 8, 16]
UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
N_COMPONENTS = 3


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


def smooth_calcium_signals(signal_mat: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return signal_mat.copy()
    smoothed = np.zeros_like(signal_mat, dtype=float)
    for i in range(signal_mat.shape[1]):
        smoothed[:, i] = gaussian_filter1d(signal_mat[:, i], sigma=sigma)
    return smoothed


def filter_trace_at_band(trace: np.ndarray, fs: float, low_f: float, high_f: float) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = max(0.001, low_f) / nyq
    high_n = min(nyq - 0.001, high_f) / nyq
    if low_n >= high_n:
        return np.zeros_like(trace)
    sos = signal.butter(2, [low_n, high_n], btype="band", output="sos")
    filtered = signal.sosfiltfilt(sos, trace)
    return filtered - np.min(filtered)


def compute_precise_trial_frequency(mov_dir: np.ndarray, time_vec: np.ndarray) -> float:
    if mov_dir is None or len(mov_dir) < 50:
        return np.nan
    dir_bin = (mov_dir > 0).astype(int)
    transitions = np.where(np.diff(dir_bin) == 1)[0] + 1
    if len(transitions) < 2:
        return np.nan
    cycle_times = time_vec[transitions]
    cycle_durations = np.diff(cycle_times)
    valid_durations = cycle_durations[(cycle_durations > 1.0) & (cycle_durations < 300.0)]
    if len(valid_durations) == 0:
        return np.nan
    mean_lap_time = float(np.mean(valid_durations))
    return 1.0 / mean_lap_time if mean_lap_time > 0 else np.nan


# =============================================================================
# Core Spatial Metric Functions
# =============================================================================

def compute_tuning_curve(pos: np.ndarray, trace: np.ndarray, n_bins: int = 50) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    p_min, p_max = float(np.nanmin(pos)), float(np.nanmax(pos))
    if p_max <= p_min or not np.isfinite(p_min) or not np.isfinite(p_max):
        return np.zeros(n_bins), np.zeros(n_bins), np.ones(n_bins) / n_bins
    bin_edges = np.linspace(p_min, p_max, n_bins + 1)
    bin_idx = np.digitize(pos, bin_edges[1:-1])
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    occupancy = np.bincount(bin_idx, minlength=n_bins).astype(float)
    activity = np.bincount(bin_idx, weights=trace, minlength=n_bins).astype(float)
    tuning = np.zeros(n_bins, dtype=float)
    valid = occupancy > 0
    tuning[valid] = activity[valid] / occupancy[valid]
    tuning = gaussian_filter1d(tuning, sigma=1.5, mode="nearest")
    prob_occ = occupancy / np.sum(occupancy) if np.sum(occupancy) > 0 else np.ones(n_bins) / n_bins
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, tuning, prob_occ


def calculate_spatial_information(tuning: np.ndarray, prob_occ: np.ndarray) -> float:
    mean_rate = np.sum(prob_occ * tuning)
    if mean_rate <= 1e-12:
        return 0.0
    ratio = tuning / mean_rate
    ratio[ratio <= 0] = 1e-12
    return float(np.sum(prob_occ * ratio * np.log2(ratio)))


def compute_placeness_score(pos: np.ndarray, trace: np.ndarray, n_shuffles: int = 30) -> float:
    _, tuning, prob = compute_tuning_curve(pos, trace, n_bins=N_POS_BINS)
    actual_si = calculate_spatial_information(tuning, prob)
    if actual_si <= 0 or np.nanstd(trace) < 1e-6:
        return 0.0
    shuf_si = []
    n_pts = len(trace)
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        shift = rng.integers(int(0.05 * n_pts), int(0.95 * n_pts))
        shuf_trace = np.roll(trace, shift)
        _, shuf_tune, _ = compute_tuning_curve(pos, shuf_trace, n_bins=N_POS_BINS)
        shuf_si.append(calculate_spatial_information(shuf_tune, prob))
    mu_shuf = float(np.nanmean(shuf_si))
    sd_shuf = float(np.nanstd(shuf_si))
    placeness_score = float((actual_si - mu_shuf) / sd_shuf) if sd_shuf > 1e-6 else float(actual_si)
    return max(0.0, placeness_score)


# =============================================================================
# Ultimate Master Figure Plotting Suite (Log10 Raw Activity on Row 3)
# =============================================================================

def plot_ultimate_master_figure(state_embeddings: Dict[str, np.ndarray], raw_trace: np.ndarray,
                                slow_trace: np.ndarray, lap_trace: np.ndarray, sigma_traces: Dict[str, np.ndarray],
                                time_vec: np.ndarray, pos_1d: np.ndarray, time_1d: np.ndarray,
                                mouse_name: str, sess_name: str, c_idx: int, place_score: float, save_dir: Path):
    states = ["unfiltered"] + [f"sigma_{s}" for s in FILTER_SIGMAS]
    metrics_config = [
        ("Position", pos_1d, "magma"),
        ("Time", time_1d, "YlGn_r"),
        ("Raw Activity (Log10)", raw_trace, "plasma"),
        ("Slow Band [0.005-0.02Hz]", slow_trace, "plasma"),
        ("Trial Lap Freq", lap_trace, "plasma"),
    ]

    for scale_type in ["linear", "log10"]:
        scale_dir = save_dir / scale_type
        scale_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(16, 26), dpi=300)
        gs = mpl.gridspec.GridSpec(6, 8, height_ratios=[1.2] + [2] * 5, hspace=0.15, wspace=0.1)

        ax_trace = fig.add_subplot(gs[0, :])
        ax_trace.plot(time_vec, raw_trace, color="#95a5a6", lw=0.9, alpha=0.6, label="Raw Trace (ΔF/F)")
        ax_trace.plot(time_vec, slow_trace, color="#e74c3c", lw=1.3, label="Slow Band [0.005-0.02Hz]")
        ax_trace.plot(time_vec, lap_trace, color="#2980b9", lw=1.3, label="Trial Lap Freq")
        for s_key, s_tr in sigma_traces.items():
            ax_trace.plot(time_vec, s_tr, lw=1.1, linestyle="--", label=f"Filter ({s_key})")

        ax_trace.set_title(
            f"Cell {c_idx} Ultimate Master Sheet | Placeness Score: {place_score:.2f} ({scale_type.upper()})",
            fontsize=11, fontweight="bold")
        ax_trace.set_xlabel("Time (s)", fontsize=8)
        ax_trace.set_ylabel("Amplitude", fontsize=8)
        ax_trace.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper right")
        ax_trace.grid(False)
        ax_trace.spines[["top", "right"]].set_visible(False)

        for m_idx, (m_name, m_vals, m_cmap) in enumerate(metrics_config):
            for s_idx, state_name in enumerate(states):
                if state_name not in state_embeddings:
                    continue
                umap_emb = state_embeddings[state_name]

                if m_name == "Position":
                    c_vals = pos_1d
                elif m_name == "Time":
                    c_vals = time_1d
                elif m_name == "Raw Activity (Log10)":
                    c_vals = raw_trace
                elif m_name == "Slow Band [0.005-0.02Hz]":
                    c_vals = slow_trace
                elif m_name == "Trial Lap Freq":
                    c_vals = lap_trace
                else:
                    c_vals = raw_trace

                # Apply log10 explicitly to Raw Activity (row 2 / index 2) in both linear and log10 master figures,
                # and to other activity rows if scale_type == 'log10'
                if m_name == "Raw Activity (Log10)" or (scale_type == "log10" and m_name not in ["Position", "Time"]):
                    v_min = np.min(c_vals)
                    c_vals = np.log10(c_vals - v_min + 1e-4)
                    c_label = f"Log10 {m_name.replace(' (Log10)', '')}"
                else:
                    c_label = m_name

                col_base = s_idx * 2

                ax_l = fig.add_subplot(gs[m_idx + 1, col_base], projection="3d")
                sc_l = ax_l.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=c_vals, cmap=m_cmap, s=3,
                                    alpha=0.85)
                if m_idx == 0:
                    ax_l.set_title(f"{state_name.upper()}\nRing", fontsize=8, fontweight="bold")
                ax_l.view_init(elev=90, azim=-90)
                ax_l.grid(False)

                ax_r = fig.add_subplot(gs[m_idx + 1, col_base + 1], projection="3d")
                sc_r = ax_r.scatter(umap_emb[:, 0], umap_emb[:, 1], umap_emb[:, 2], c=c_vals, cmap=m_cmap, s=3,
                                    alpha=0.85)
                if m_idx == 0:
                    ax_r.set_title(f"{state_name.upper()}\nOrthog", fontsize=8, fontweight="bold")
                ax_r.view_init(elev=0, azim=0)
                ax_r.grid(False)

        safe_name = f"{mouse_name}_{sess_name}_cell_{c_idx:03d}_ultimate_master_figure"
        for ext in [".png", ".svg"]:
            fig.savefig(scale_dir / f"{safe_name}{ext}", bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# Main Pipeline for Animal GC2 (Selected Cells)
# =============================================================================

def process_animal_gc2_ultimate_suite():
    print("=" * 80)
    print(f"PROCESSING ANIMAL GC2 ULTIMATE MASTER FIGURE SUITE FOR CELLS: {SELECTED_CELL_INDICES}")
    print("=" * 80)

    master_df = pd.read_csv(MASTER_INDEX_CSV)

    df_sub = master_df[
        (master_df["dataset_name"].astype(str).str.lower() == "lt_rot") &
        (master_df["mouse_name"].astype(str).str.contains("GC2", case=False, na=False)) &
        (master_df["session_name"].astype(str).str.lower().str.contains("lt"))
        ].copy()

    for _, row in df_sub.iterrows():
        mouse_name = str(row["mouse_name"])
        sess_name = str(row["session_name"])
        depth = str(row.get("depth_type", "unknown"))

        fpath = resolve_path(row.get("standardized_file_path"))
        if fpath is None or not fpath.exists():
            continue

        print(f"\nAnalyzing Session: {sess_name} | Mouse: {mouse_name}")
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
        if raw_pos_arr.ndim == 2 and raw_pos_arr.shape[1] >= 2:
            raw_pos_x = raw_pos_arr[:, 0]
        else:
            raw_pos_x = raw_pos_arr

        pos = align_vector_to_target_length(raw_pos_x, n_time_target, is_discrete=False)

        fs = FIXED_FS
        time_vec = np.arange(n_time_target) / fs
        raw_mov_dir = payload.get("mov_dir", payload.get("direction"))
        mov_dir = align_vector_to_target_length(raw_mov_dir, n_time_target, is_discrete=True)

        trial_freq = compute_precise_trial_frequency(mov_dir, time_vec)

        speed = np.abs(np.gradient(pos, 1.0 / fs))
        valid_mask = (speed >= MIN_SPEED_CM_S) & (mov_dir != 0) & np.isfinite(pos)

        pos_valid = pos[valid_mask]
        time_valid = time_vec[valid_mask]
        sig_valid = signal_mat[valid_mask, :]

        if sig_valid.shape[0] < 50:
            continue

        states_to_process = [("unfiltered", 0.0)] + [(f"sigma_{s}", s) for s in FILTER_SIGMAS]
        state_embeddings = {}

        for state_name, sigma in states_to_process:
            sig_processed = smooth_calcium_signals(sig_valid, sigma=sigma)
            try:
                umap_model = umap.UMAP(n_components=N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
                                       min_dist=UMAP_MIN_DIST, random_state=UMAP_RANDOM_STATE)
                state_embeddings[state_name] = umap_model.fit_transform(sig_processed)
            except Exception as e:
                print(f"    [WARNING] UMAP failed for state {state_name}: {e}")

        session_out_dir = OUT_ROOT / mouse_name / sess_name
        cells_dir = session_out_dir / "ultimate_master_figures"
        cells_dir.mkdir(parents=True, exist_ok=True)

        for c_idx in SELECTED_CELL_INDICES:
            if c_idx >= n_cells:
                continue

            raw_cell_trace = sig_valid[:, c_idx]
            p_score = compute_placeness_score(pos_valid, raw_cell_trace, n_shuffles=N_SHUFFLES_PLACE)

            slow_trace = filter_trace_at_band(raw_cell_trace, fs, 0.005, 0.02)

            if np.isfinite(trial_freq) and trial_freq > 0:
                lap_trace = filter_trace_at_band(raw_cell_trace, fs, trial_freq - 0.02, trial_freq + 0.02)
            else:
                lap_trace = np.zeros_like(raw_cell_trace)

            sigma_traces = {}
            for s in FILTER_SIGMAS:
                sigma_traces[f"sigma_{s}"] = smooth_calcium_signals(raw_cell_trace[:, np.newaxis], sigma=s)[:, 0]

            plot_ultimate_master_figure(state_embeddings, raw_cell_trace, slow_trace, lap_trace, sigma_traces,
                                        time_valid, pos_valid, time_valid, mouse_name, sess_name,
                                        c_idx, p_score, cells_dir)

    print("=" * 80)
    print("ULTIMATE MASTER FIGURE SUITE SUCCESSFULLY COMPLETED!")
    print(f"Output Directory Root: {OUT_ROOT}")
    print("=" * 80)


def main():
    process_animal_gc2_ultimate_suite()


if __name__ == "__main__":
    main()
