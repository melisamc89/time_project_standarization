#!/usr/init/env python3
# -*- coding: utf-8 -*-

"""
Unified Ultraslow & Trial Frequency Oscillatory Sequences Analysis Pipeline
Features:
- Explicit Slow and Trial Oscillation Scoring & Peak Frequencies.
- Target Exemplar Extraction per Animal (Types 1, 2, and 3) with Trial-Aligned Position Matrices.
- Log-Scale Autocorrelogram PSD Heatmaps Sorted by Slow Peak Frequency & Intensity.
- Multi-Panel Statistical Distribution Summary Plots (Violins/Box plots comparing Superficial vs. Deep).
- Comprehensive Measurement CSV Export.
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

# =============================================================================
# Paths & Settings
# =============================================================================

POSSIBLE_ROOTS = [
    Path("/home/hippolab/Documents/melisa/time_project_standarization"),
    Path("/home/melisamc/Documents/time_project_standarization"),
]
PROJECT_ROOT = next((p for p in POSSIBLE_ROOTS if p.is_dir()), POSSIBLE_ROOTS[0])

MASTER_INDEX_CSV = PROJECT_ROOT / "data" / "metadata" / "master_dataset_index.csv"
OUT_ROOT = PROJECT_ROOT / "analysis" / "nature_2024_unified_oscillatory_suite"

LAYER_COLOR_MAP = {
    "deep": "#cc9900",  # Mustard / Gold
    "superficial": "#9900ff",  # Purple
    "whole": "#2c3e50",  # Dark Slate for whole population
    "unknown": "#888888",
}

FIXED_FS = 20.0
N_POS_BINS = 50
MIN_SPEED_CM_S = 0.02
N_SHUFFLES_PLACE = 30
ACORR_MAX_LAG_S = 500.0
MAX_FREQ_PLOTTED = 1.0  # Hz


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
# Core Spatial & Behavioral Metric Functions
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


def compute_placeness_score(pos: np.ndarray, trace: np.ndarray, n_shuffles: int = 30) -> Tuple[float, np.ndarray]:
    _, tuning, prob = compute_tuning_curve(pos, trace, n_bins=N_POS_BINS)
    actual_si = calculate_spatial_information(tuning, prob)
    if actual_si <= 0 or np.nanstd(trace) < 1e-6:
        return 0.0, tuning
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
    return max(0.0, placeness_score), tuning


def compute_directionality_score(pos: np.ndarray, mov_dir: np.ndarray, trace: np.ndarray) -> Tuple[
    float, np.ndarray, np.ndarray, float]:
    mask_pos = (mov_dir > 0) & np.isfinite(pos) & np.isfinite(trace)
    mask_neg = (mov_dir < 0) & np.isfinite(pos) & np.isfinite(trace)
    _, tune_pos, _ = compute_tuning_curve(pos[mask_pos], trace[mask_pos], n_bins=N_POS_BINS) if np.sum(
        mask_pos) > 10 else (None, np.zeros(N_POS_BINS), None)
    _, tune_neg, _ = compute_tuning_curve(pos[mask_neg], trace[mask_neg], n_bins=N_POS_BINS) if np.sum(
        mask_neg) > 10 else (None, np.zeros(N_POS_BINS), None)

    lr_corr = np.nan
    if tune_pos is not None and tune_neg is not None and len(tune_pos) == len(tune_neg):
        valid_bins = np.isfinite(tune_pos) & np.isfinite(tune_neg)
        if np.sum(valid_bins) > 5 and np.std(tune_pos[valid_bins]) > 1e-6 and np.std(tune_neg[valid_bins]) > 1e-6:
            r_val, _ = stats.pearsonr(tune_pos[valid_bins], tune_neg[valid_bins])
            if np.isfinite(r_val):
                lr_corr = float(r_val)

    dir_score = float(1.0 - lr_corr) if np.isfinite(lr_corr) else 0.0
    return dir_score, tune_pos, tune_neg, lr_corr


def compute_drift_score(time_vec: np.ndarray, trace: np.ndarray) -> float:
    if np.std(trace) < 1e-6 or np.std(time_vec) < 1e-6:
        return 0.0
    rho, _ = stats.spearmanr(time_vec, trace)
    return float(abs(rho)) if np.isfinite(rho) else 0.0


def compute_speedness_score(speed: np.ndarray, trace: np.ndarray, n_bins: int = 20) -> float:
    s_min, s_max = float(np.nanmin(speed)), float(np.nanmax(speed))
    if s_max <= s_min or np.std(trace) < 1e-6:
        return 0.0
    bin_edges = np.linspace(s_min, s_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(speed, bin_edges[1:-1]), 0, n_bins - 1)
    occupancy = np.bincount(bin_idx, minlength=n_bins).astype(float)
    activity = np.bincount(bin_idx, weights=trace, minlength=n_bins).astype(float)
    tuning = np.zeros(n_bins, dtype=float)
    valid = occupancy > 0
    tuning[valid] = activity[valid] / occupancy[valid]
    valid_t = np.isfinite(tuning)
    if np.sum(valid_t) < 5:
        return 0.0
    rho, _ = stats.spearmanr(np.arange(n_bins)[valid_t], tuning[valid_t])
    return float(abs(rho)) if np.isfinite(rho) else 0.0


def compute_spatial_stability(pos: np.ndarray, trace: np.ndarray, trial_ids: np.ndarray, n_bins: int = 50) -> float:
    unique_trials = np.unique(trial_ids)
    valid_trials = [t for t in unique_trials if t >= 0]
    if len(valid_trials) < 3:
        return np.nan

    p_mi, p_ma = float(np.nanmin(pos)), float(np.nanmax(pos))
    bin_edges = np.linspace(p_mi, p_ma, n_bins + 1)

    trial_tunes = {}
    for t_id in valid_trials:
        mask = (trial_ids == t_id) & np.isfinite(pos) & np.isfinite(trace)
        if np.sum(mask) < 5:
            continue
        p_t, tr_t = pos[mask], trace[mask]
        b_id = np.clip(np.digitize(p_t, bin_edges[1:-1]), 0, n_bins - 1)
        occ = np.bincount(b_id, minlength=n_bins).astype(float)
        act = np.bincount(b_id, weights=tr_t, minlength=n_bins).astype(float)
        tune = np.zeros(n_bins, dtype=float)
        val = occ > 0
        tune[val] = act[val] / occ[val]
        trial_tunes[t_id] = gaussian_filter1d(tune, sigma=1.5, mode="nearest")

    valid_keys = list(trial_tunes.keys())
    if len(valid_keys) < 3:
        return np.nan

    corrs = []
    for t_id in valid_keys:
        t_curr = trial_tunes[t_id]
        others = [trial_tunes[k] for k in valid_keys if k != t_id]
        t_mean = np.mean(others, axis=0)

        valid_bins = np.isfinite(t_curr) & np.isfinite(t_mean) & (np.std(t_curr) > 1e-6) & (np.std(t_mean) > 1e-6)
        if np.sum(valid_bins) > 5:
            r_val, _ = stats.pearsonr(t_curr[valid_bins], t_mean[valid_bins])
            if np.isfinite(r_val):
                corrs.append(r_val)

    return float(np.mean(corrs)) if len(corrs) > 0 else 0.0


def compute_normalized_field_size_variability(pos: np.ndarray, trace: np.ndarray, trial_ids: np.ndarray,
                                              n_bins: int = 50) -> float:
    unique_trials = np.unique(trial_ids)
    valid_trials = [t for t in unique_trials if t >= 0]
    p_mi, p_ma = float(np.nanmin(pos)), float(np.nanmax(pos))
    bin_edges = np.linspace(p_mi, p_ma, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_resolution = bin_centers[1] - bin_centers[0]

    trial_sizes = []
    for t_id in valid_trials:
        mask = (trial_ids == t_id) & np.isfinite(pos) & np.isfinite(trace)
        if np.sum(mask) < 5:
            continue
        p_t, tr_t = pos[mask], trace[mask]
        b_id = np.clip(np.digitize(p_t, bin_edges[1:-1]), 0, n_bins - 1)
        occ = np.bincount(b_id, minlength=n_bins).astype(float)
        act = np.bincount(b_id, weights=tr_t, minlength=n_bins).astype(float)
        tune = np.zeros(n_bins, dtype=float)
        val = occ > 0
        tune[val] = act[val] / occ[val]
        tune = gaussian_filter1d(tune, sigma=1.5, mode="nearest")

        if np.max(tune) > 1e-6:
            half_max = 0.5 * np.max(tune)
            above = tune >= half_max
            if np.any(above):
                idx = np.where(above)[0]
                size = (idx[-1] - idx[0] + 1) * bin_resolution
                trial_sizes.append(size)

    trial_sizes = np.array(trial_sizes)
    trial_sizes = trial_sizes[np.isfinite(trial_sizes) & (trial_sizes > 0)]
    if len(trial_sizes) < 3 or np.mean(trial_sizes) <= 1e-12:
        return np.nan
    return float(np.std(trial_sizes) / np.mean(trial_sizes))


def calculate_place_field_size(tuning: np.ndarray) -> float:
    if len(tuning) == 0 or np.max(tuning) <= 1e-6:
        return 0.0
    peak_val = np.max(tuning)
    half_max = 0.5 * peak_val
    above_half = tuning >= half_max
    if not np.any(above_half):
        return 0.0
    return float(np.sum(above_half) / len(tuning))


def compute_remapping_score(pos: np.ndarray, trace: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    half_pt = len(trace) // 2
    _, tune_h1, _ = compute_tuning_curve(pos[:half_pt], trace[:half_pt], n_bins=N_POS_BINS)
    _, tune_h2, _ = compute_tuning_curve(pos[half_pt:], trace[half_pt:], n_bins=N_POS_BINS)
    if np.std(tune_h1) < 1e-12 or np.std(tune_h2) < 1e-12:
        return 1.0, tune_h1, tune_h2
    r, _ = stats.pearsonr(tune_h1, tune_h2)
    return float(np.clip(1.0 - r, 0.0, 2.0)), tune_h1, tune_h2


# =============================================================================
# Spectral & Explicit Oscillatory Scoring Functions
# =============================================================================

def compute_acorr_psd(trace: np.ndarray, fs: float, trial_freq: float):
    x = trace - np.mean(trace)
    var = np.var(trace)
    if var <= 1e-12:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0

    autocorr = signal.correlate(x, x, mode="full")[len(trace) - 1:]
    autocorr = autocorr / (len(trace) * var)
    lags_s = np.arange(len(autocorr)) / fs

    # Detrend slow envelope to reveal periodic ripples clearly in PSD
    slow_envelope = gaussian_filter1d(autocorr, sigma=int(fs * 20))
    acorn_vals = autocorr - slow_envelope

    max_lag_idx = min(len(lags_s), int(ACORR_MAX_LAG_S * fs))
    acorn_vals_sub = acorn_vals[:max_lag_idx]

    nperseg = max(32, len(acorn_vals_sub) // 4)
    nfft = max(4096, 4 * len(acorn_vals_sub))
    freqs, psd = signal.welch(acorn_vals_sub, fs=fs, nperseg=nperseg, nfft=nfft)

    # 1. Slow Oscillation Score [0.005, 0.05 Hz]
    slow_mask = (freqs >= 0.005) & (freqs <= 0.05)
    if np.any(slow_mask):
        sub_f_slow = freqs[slow_mask]
        sub_p_slow = psd[slow_mask]
        slow_idx = np.argmax(sub_p_slow)
        slow_score = float(sub_p_slow[slow_idx])
        slow_peak_freq = float(sub_f_slow[slow_idx])
    else:
        slow_score = 0.0
        slow_peak_freq = 0.0

    # 2. Trial Oscillation Score [trial_freq ± 0.02 Hz]
    if np.isfinite(trial_freq) and trial_freq > 0:
        lap_mask = (freqs >= trial_freq - 0.02) & (freqs <= trial_freq + 0.02)
        if np.any(lap_mask):
            sub_f_lap = freqs[lap_mask]
            sub_p_lap = psd[lap_mask]
            lap_idx = np.argmax(sub_p_lap)
            trial_score = float(sub_p_lap[lap_idx])
            trial_peak_freq = float(sub_f_lap[lap_idx])
        else:
            trial_score = 0.0
            trial_peak_freq = 0.0
    else:
        trial_score = 0.0
        trial_peak_freq = 0.0

    return freqs, psd, lags_s[:max_lag_idx], autocorr, slow_score, slow_peak_freq, trial_score, trial_peak_freq


def filter_trace_at_band(trace: np.ndarray, fs: float, low_f: float, high_f: float) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = max(0.001, low_f) / nyq
    high_n = min(nyq - 0.001, high_f) / nyq
    if low_n >= high_n:
        return np.zeros_like(trace)
    sos = signal.butter(2, [low_n, high_n], btype="band", output="sos")
    filtered = signal.sosfiltfilt(sos, trace)
    return filtered - np.min(filtered)


# =============================================================================
# Full Characterization Suite Plotting per Exemplar Cell (Trial-Aligned Matrix)
# =============================================================================

def export_full_characterization_suite(cell_rec: dict, diag: dict, trial_freq: float, category_name: str,
                                       save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    c_id = cell_rec["cell_id"]
    d_type = cell_rec["depth_type"]
    safe_cid = re.sub(r'[^A-Za-z0-9_-]+', '_', c_id)

    pos_p = diag["pos"]
    trace_p = diag["trace"]
    time_p = diag["time"]
    trial_ids = diag["trial_ids"]
    fs = diag["fs"]

    # 1. Trial-by-Position Matrix
    n_pos_bins = N_POS_BINS
    p_min, p_max = float(np.nanmin(pos_p)), float(np.nanmax(pos_p))
    pos_edges = np.linspace(p_min, p_max, n_pos_bins + 1)
    bin_centers = 0.5 * (pos_edges[:-1] + pos_edges[1:])

    valid_trials = [t for t in np.unique(trial_ids) if t >= 0]
    n_trials = len(valid_trials)

    trial_rate_matrix = np.full((n_trials, n_pos_bins), np.nan)
    for idx, t_id in enumerate(valid_trials):
        t_mask = (trial_ids == t_id) & np.isfinite(pos_p) & np.isfinite(trace_p)
        if np.sum(t_mask) < 3:
            continue
        t_pos = pos_p[t_mask]
        t_trace = trace_p[t_mask]

        b_idx = np.clip(np.digitize(t_pos, pos_edges[1:-1]), 0, n_pos_bins - 1)
        occ = np.bincount(b_idx, minlength=n_pos_bins).astype(float)
        act = np.bincount(b_idx, weights=t_trace, minlength=n_pos_bins).astype(float)

        t_tune = np.zeros(n_pos_bins, dtype=float)
        val = occ > 0
        t_tune[val] = act[val] / occ[val]
        trial_rate_matrix[idx, :] = gaussian_filter1d(t_tune, sigma=1.2, mode="nearest")

    trial_rate_matrix = np.nan_to_num(trial_rate_matrix, nan=0.0)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.5), dpi=300, gridspec_kw={"height_ratios": [2, 2, 1]})
    ax_heat, ax_waterfall, ax_tune = axes

    im = ax_heat.imshow(trial_rate_matrix, aspect="auto", origin="lower",
                        extent=[p_min, p_max, 0, n_trials],
                        cmap="viridis", interpolation="nearest")
    ax_heat.set_ylabel("Trial / Lap Index", fontsize=9, fontweight="bold")
    ax_heat.set_title(
        f"[{category_name.upper()}] Cell ({c_id}) | Placeness Z = {cell_rec['placeness_score']:.2f}\nLayer: {d_type.capitalize()}",
        fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, label="Activity")
    ax_heat.grid(False)
    ax_heat.spines[["top", "right"]].set_visible(False)

    v_shift = np.max(trial_rate_matrix) * 0.4 if np.max(trial_rate_matrix) > 0 else 0.1
    for t_i in range(n_trials):
        ax_waterfall.plot(bin_centers, trial_rate_matrix[t_i, :] + t_i * v_shift,
                          color=LAYER_COLOR_MAP.get(d_type, "#9b59b6"), lw=1.1, alpha=0.85)
    ax_waterfall.set_ylabel("Trials (Shifted)", fontsize=9, fontweight="bold")
    ax_waterfall.grid(False)
    ax_waterfall.spines[["top", "right"]].set_visible(False)

    ax_tune.plot(bin_centers, diag["tune_map"], color=LAYER_COLOR_MAP.get(d_type, "#9b59b6"), lw=2.2)
    ax_tune.set_xlim(p_min, p_max)
    ax_tune.set_xlabel("Linear Track Position", fontsize=9, fontweight="bold")
    ax_tune.set_ylabel("Mean Activity", fontsize=9, fontweight="bold")
    ax_tune.grid(False)
    ax_tune.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / f"placeness_suite_{safe_cid}.png", bbox_inches="tight")
    fig.savefig(save_dir / f"placeness_suite_{safe_cid}.svg", bbox_inches="tight")
    plt.close(fig)

    # 2. PSD & Autocorrelogram Suite (Log Scale PSD)
    fig, (ax_psd, ax_acorr) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5), dpi=300,
                                           gridspec_kw={"width_ratios": [1, 1.4]})
    freqs = diag["freqs"]
    psd = diag["psd"]
    zoom_mask = (freqs >= 0.0) & (freqs <= 0.5)

    ax_psd.plot(freqs[zoom_mask], psd[zoom_mask], color="black", lw=1.5)
    ax_psd.set_yscale('log')

    if np.isfinite(trial_freq):
        ax_psd.axvline(trial_freq, color="blue", linestyle=":", label=f"Lap Freq: {trial_freq:.2f} Hz")
    ax_psd.set_xlim(0, 0.5)
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=9, fontweight="bold")
    ax_psd.set_ylabel("Power (Log Scale)", fontsize=9, fontweight="bold")
    ax_psd.set_title("Autocorrelogram PSD (Log Scale)", fontsize=10, fontweight="bold")
    ax_psd.legend(frameon=False, fontsize=7.5)
    ax_psd.grid(False)
    ax_psd.spines[["top", "right"]].set_visible(False)

    lags = diag["lags"]
    acorr = diag["autocorr"]
    ax_acorr.plot(lags, acorr[:len(lags)], color="black", lw=1.2)
    ax_acorr.set_xlim(0, 300.0)
    ax_acorr.set_xlabel("Lag (s)", fontsize=9, fontweight="bold")
    ax_acorr.set_ylabel("Autocorr", fontsize=9, fontweight="bold")
    ax_acorr.set_title("Autocorrelogram", fontsize=10, fontweight="bold")
    ax_acorr.grid(False)
    ax_acorr.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / f"psd_acorr_{safe_cid}.png", bbox_inches="tight")
    fig.savefig(save_dir / f"psd_acorr_{safe_cid}.svg", bbox_inches="tight")
    plt.close(fig)

    # 3. Filtered Trace Overlays
    fig, ax_env = plt.subplots(figsize=(8, 4.5), dpi=300)
    slow_filt = filter_trace_at_band(trace_p, fs, 0.005, 0.05)
    ax_env.plot(time_p - time_p[0], trace_p, color="gray", lw=0.7, alpha=0.3, label=r"Raw $\Delta F/F$")
    ax_env.plot(time_p - time_p[0], slow_filt, color="#c0392b", lw=1.5, alpha=0.85,
                label="Slow Component [0.005 - 0.05 Hz]")

    if np.isfinite(trial_freq):
        lap_filt = filter_trace_at_band(trace_p, fs, trial_freq - 0.02, trial_freq + 0.02)
        ax_env.plot(time_p - time_p[0], lap_filt, color="#2980b9", lw=1.2, alpha=0.8,
                    label=f"Lap Frequency Component (~{trial_freq:.2f} Hz)")

    ax_env.set_xlabel("Session Time (s)", fontsize=10, fontweight="bold")
    ax_env.set_ylabel("Amplitude", fontsize=10, fontweight="bold")
    ax_env.set_title(f"Filtered Trace Comparison | Cell ({c_id})", fontsize=10.5, fontweight="bold")
    ax_env.legend(frameon=False, fontsize=8)
    ax_env.grid(False)
    ax_env.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / f"trace_filtered_overlay_{safe_cid}.png", bbox_inches="tight")
    fig.savefig(save_dir / f"trace_filtered_overlay_{safe_cid}.svg", bbox_inches="tight")
    plt.close(fig)

    # 4. Directionality Suite
    tune_p = diag.get("tune_pos")
    tune_n = diag.get("tune_neg")
    if tune_p is not None and tune_n is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
        bin_x = np.linspace(p_min, p_max, N_POS_BINS)
        ax.plot(bin_x, tune_p, color="#3498db", lw=2.2, label="Outward (+1)")
        ax.plot(bin_x, tune_n, color="#e74c3c", lw=2.2, label="Inward (-1)")
        ax.set_xlabel("Linear Track Position", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Activity", fontsize=10, fontweight="bold")
        ax.legend(frameon=False)
        ax.set_title(
            f"Directionality Suite | DI = {cell_rec['directionality_score']:.3f}\nLayer: {d_type.capitalize()}",
            fontsize=10.5, fontweight="bold")
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(save_dir / f"directionality_{safe_cid}.png", bbox_inches="tight")
        fig.savefig(save_dir / f"directionality_{safe_cid}.svg", bbox_inches="tight")
        plt.close(fig)

    # 5. Remapping Suite
    tune_h1 = diag.get("tune_h1")
    tune_h2 = diag.get("tune_h2")
    if tune_h1 is not None and tune_h2 is not None and np.isfinite(cell_rec["remapping_score"]):
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
        bin_x = np.linspace(p_min, p_max, N_POS_BINS)
        ax.plot(bin_x, tune_h1, color="#2ecc71", lw=2.2, label="Early (H1)")
        ax.plot(bin_x, tune_h2, color="#e67e22", lw=2.2, label="Late (H2)")
        ax.set_xlabel("Linear Track Position", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Activity", fontsize=10, fontweight="bold")
        ax.legend(frameon=False)
        ax.set_title(f"Remapping Suite | Score = {cell_rec['remapping_score']:.3f}\nLayer: {d_type.capitalize()}",
                     fontsize=10.5, fontweight="bold")
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(save_dir / f"remapping_{safe_cid}.png", bbox_inches="tight")
        fig.savefig(save_dir / f"remapping_{safe_cid}.svg", bbox_inches="tight")
        plt.close(fig)

    # 6. Temporal Drift Suite
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.scatter(time_p, trace_p, s=5, color="#e67e22", alpha=0.5, rasterized=True)
    slope, intercept, _, _, _ = stats.linregress(time_p, trace_p)
    ax.plot(time_p, slope * time_p + intercept, color="black", lw=2.0, label="Monotonic Drift Trend")
    ax.set_xlabel("Session Time (s)", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"$\Delta F/F$", fontsize=10, fontweight="bold")
    ax.set_title(f"Temporal Drift Suite | |rho| = {cell_rec['temporal_drift_score']:.3f}\nLayer: {d_type.capitalize()}",
                 fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / f"drift_{safe_cid}.png", bbox_inches="tight")
    fig.savefig(save_dir / f"drift_{safe_cid}.svg", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Summary Distribution Plots Generation (Superficial vs Deep)
# =============================================================================

def generate_summary_distribution_plots(df_cells: pd.DataFrame, dataset_key: str, dataset_out: Path):
    hist_dir = dataset_out / "summary_plots_depth_comparison"
    hist_dir.mkdir(parents=True, exist_ok=True)

    features = [
        ("placeness_score", "Placeness Score (SI Z-score)", "violin"),
        ("directionality_score", "Directionality Score (1 - r_LR)", "violin"),
        ("temporal_drift_score", "Temporal Drift Score (|Spearman rho|)", "violin"),
        ("speedness_score", "Speedness Score", "violin"),
        ("spatial_stability", "Spatial Stability Score [Place Cells Only]", "violin"),
        ("field_size_cv", "Norm. Field Size Variability [Place Cells Only]", "violin"),
        ("slow_oscillation_score", "Slow Oscillation Score [0.005 - 0.05 Hz] (Power)", "violin"),
        ("slow_peak_frequency_hz", "Slow Peak Frequency (Hz)", "violin"),
        ("trial_oscillation_score", "Trial Oscillation Score (Power)", "violin"),
        ("trial_frequency_hz", "Trial / Lap Frequency (Hz)", "box"),
        ("remapping_score", "Remapping Score [Place Cells Only]", "violin"),
        ("place_field_size", "Place Field Size (FWHM) [Place Cells Only]", "violin"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 13), dpi=300)
    axes = axes.flatten()

    for idx, (col, title, plot_type) in enumerate(features):
        ax = axes[idx]
        sub_df = df_cells.dropna(subset=[col])
        if sub_df.empty:
            continue

        sup_vals = sub_df[sub_df["depth_type"] == "superficial"][col].values
        deep_vals = sub_df[sub_df["depth_type"] == "deep"][col].values

        data_to_plot = [v for v in [sup_vals, deep_vals] if len(v) > 0]
        if len(data_to_plot) == 2:
            colors = [LAYER_COLOR_MAP["superficial"], LAYER_COLOR_MAP["deep"]]
            if plot_type == "box":
                bp = ax.boxplot(data_to_plot, tick_labels=["Superficial", "Deep"], patch_artist=True)
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(colors[i])
                    patch.set_alpha(0.75)
            else:
                parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.75)
                ax.set_xticks([1, 2])
                ax.set_xticklabels(["Superficial", "Deep"])

            try:
                _, pval = stats.mannwhitneyu(sup_vals, deep_vals, alternative='two-sided')
                sig_str = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
                y_max = max(np.max(sup_vals), np.max(deep_vals))
                ax.plot([1, 1, 2, 2], [y_max * 0.9, y_max * 0.95, y_max * 0.95, y_max * 0.9], color="black", lw=1.0)
                ax.text(1.5, y_max * 0.96, sig_str, ha='center', va='bottom', fontsize=11, fontweight='bold')
            except Exception:
                pass

        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_ylabel("Value", fontsize=9, fontweight="bold")
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Summary Plots ({dataset_key.upper()} Dataset)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    fig.savefig(hist_dir / "summary_distribution_plots.png", bbox_inches="tight")
    fig.savefig(hist_dir / "summary_distribution_plots.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved summary distribution plots for {dataset_key.upper()}.")


# =============================================================================
# Session-Level PSD Heatmap Generator (Sorted by Slow Peak Frequency & Intensity)
# =============================================================================

def generate_session_psd_heatmaps(dataset_key: str, dataset_out: Path, session_data_list: List[dict]):
    heatmap_dir = dataset_out / "session_psd_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for sess_info in session_data_list:
        mouse_name = sess_info["mouse_name"]
        sess_name = sess_info["session_name"]
        depth = sess_info["depth_type"]
        cell_psds = sess_info["cell_psds"]
        slow_peak_freqs = sess_info["slow_peak_freqs"]
        slow_scores = sess_info["slow_scores"]
        freqs_ref = sess_info["freqs_ref"]

        if len(cell_psds) == 0:
            continue

        cell_psds = np.array(cell_psds)
        slow_peak_freqs = np.array(slow_peak_freqs)
        slow_scores = np.array(slow_scores)

        sort_idx = np.lexsort((-slow_scores, slow_peak_freqs))
        cell_psds_sorted = cell_psds[sort_idx, :]

        freq_mask = freqs_ref <= MAX_FREQ_PLOTTED
        plot_freqs = freqs_ref[freq_mask]
        plot_matrix = cell_psds_sorted[:, freq_mask]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        valid_vals = plot_matrix[np.isfinite(plot_matrix) & (plot_matrix > 0)]
        vmin = max(1e-6, np.min(valid_vals)) if len(valid_vals) > 0 else 1e-6
        vmax = np.max(valid_vals) if len(valid_vals) > 0 else 1.0

        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(plot_matrix, aspect="auto", origin="lower",
                       extent=[plot_freqs[0], plot_freqs[-1], 0, len(plot_matrix)],
                       cmap="inferno", norm=norm)
        cbar = fig.colorbar(im, ax=ax, label="PSD Power (Log Scale)")

        ax.set_xlim(0, MAX_FREQ_PLOTTED)
        ax.set_xlabel("Autocorrelogram Frequency (Hz)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Cells (Sorted by Slow Freq & Intensity)", fontsize=10, fontweight="bold")
        ax.set_title(f"Session PSD Heatmap | Mouse: {mouse_name} | Sess: {sess_name}\nLayer: {depth.capitalize()}",
                     fontsize=11, fontweight="bold")
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        safe_sess_name = re.sub(r'[^A-Za-z0-9_-]+', '_', f"{mouse_name}_{sess_name}_{depth}")
        out_path = heatmap_dir / f"psd_heatmap_{safe_sess_name}.png"
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(heatmap_dir / f"psd_heatmap_{safe_sess_name}.svg", bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# Main Pipeline per Dataset
# =============================================================================

def process_dataset(dataset_key: str):
    print("=" * 80)
    print(f"PROCESSING DATASET (UNIFIED OSCILLATORY SUITE): {dataset_key.upper()}")
    print("=" * 80)

    master_df = pd.read_csv(MASTER_INDEX_CSV)
    df_sub = master_df[master_df["dataset_name"].astype(str).str.lower() == dataset_key.lower()].copy()

    if dataset_key.lower() == "learning":
        if "session_name" in df_sub.columns and "mouse_name" in df_sub.columns:
            df_sub = df_sub.sort_values(by=["mouse_name", "session_name"])
            df_sub = df_sub.groupby("mouse_name").tail(1).copy()
        print(f"Learning dataset filtered to last session per animal. Total: {len(df_sub)}")
    else:
        df_sub = df_sub[df_sub["session_name"].astype(str).str.lower().str.contains("lt")].copy()
        print(f"lt_rot dataset filtered to LT sessions. Total: {len(df_sub)}")

    dataset_out = OUT_ROOT / dataset_key.lower()
    table_dir = dataset_out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    cell_records = []
    session_data_list = []
    animal_exemplars = {}

    for _, row in df_sub.iterrows():
        mouse_name = str(row["mouse_name"])
        sess_name = str(row["session_name"])
        depth = canonicalize_depth(row.get("depth_type", "unknown"))
        if dataset_key.lower() == "learning":
            depth = "whole"

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
        pos = align_vector_to_target_length(raw_pos, n_time_target, is_discrete=False)

        fs = FIXED_FS
        time_vec = np.arange(n_time_target) / fs
        raw_mov_dir = payload.get("mov_dir", payload.get("direction"))
        mov_dir = align_vector_to_target_length(raw_mov_dir, n_time_target, is_discrete=True)

        trial_freq = compute_precise_trial_frequency(mov_dir, time_vec)

        speed = np.abs(np.gradient(pos, 1.0 / fs))
        valid_mask = (speed >= MIN_SPEED_CM_S) & (mov_dir != 0) & np.isfinite(pos)

        pos_valid = pos[valid_mask]
        time_valid = time_vec[valid_mask]
        dir_valid = mov_dir[valid_mask]
        sig_valid = signal_mat[valid_mask, :]

        if sig_valid.shape[0] < 50:
            continue

        trial_bin = (dir_valid > 0).astype(int)
        trial_ids = np.cumsum(np.where(np.diff(trial_bin) == 1, 1, 0))
        if len(trial_ids) < len(pos_valid):
            trial_ids = np.pad(trial_ids, (0, len(pos_valid) - len(trial_ids)), mode='edge')

        if mouse_name not in animal_exemplars:
            animal_exemplars[mouse_name] = {
                "type_1_lap_freq": (None, None, -1.0, None),
                "type_2_slow_only": (None, None, -1.0, None),
                "type_3_both": (None, None, -1.0, None),
            }

        sess_cell_psds = []
        sess_slow_freqs = []
        sess_slow_scores = []
        freqs_ref = None

        for c_idx in range(n_cells):
            trace = sig_valid[:, c_idx]
            if np.std(trace) <= 1e-12:
                continue

            cell_id = f"{mouse_name}_{sess_name}_{depth}_cell_{c_idx:03d}"
            place_score, tune_map = compute_placeness_score(pos_valid, trace, n_shuffles=N_SHUFFLES_PLACE)
            dir_score, tune_pos, tune_neg, _ = compute_directionality_score(pos_valid, dir_valid, trace)
            drift_score = compute_drift_score(time_valid, trace)
            speedness_score = compute_speedness_score(speed[valid_mask], trace)
            remap_score, tune_h1, tune_h2 = compute_remapping_score(pos_valid, trace)
            field_size = calculate_place_field_size(tune_map)
            spatial_stability = compute_spatial_stability(pos_valid, trace, trial_ids) if place_score >= 1.5 else np.nan
            field_size_cv = compute_normalized_field_size_variability(pos_valid, trace,
                                                                      trial_ids) if place_score >= 1.5 else np.nan

            freqs, psd, lags, autocorr, slow_score, slow_peak_freq, trial_score, trial_peak_freq = compute_acorr_psd(
                trace, fs, trial_freq)
            if len(freqs) == 0:
                continue

            if freqs_ref is None:
                freqs_ref = freqs

            sess_cell_psds.append(psd)
            sess_slow_freqs.append(slow_peak_freq)
            sess_slow_scores.append(slow_score)

            rec = {
                "cell_id": cell_id, "mouse_name": mouse_name, "session_name": sess_name,
                "depth_type": depth, "neuron_idx": c_idx, "placeness_score": place_score,
                "directionality_score": dir_score, "temporal_drift_score": drift_score,
                "speedness_score": speedness_score, "spatial_stability": spatial_stability,
                "field_size_cv": field_size_cv, "remapping_score": remap_score,
                "place_field_size": field_size, "trial_frequency_hz": trial_freq,
                "slow_oscillation_score": slow_score, "slow_peak_frequency_hz": slow_peak_freq,
                "trial_oscillation_score": trial_score, "trial_peak_frequency_hz": trial_peak_freq
            }
            cell_records.append(rec)

            diag = {
                "trace": trace, "pos": pos_valid, "time": time_valid, "trial_ids": trial_ids, "fs": fs,
                "freqs": freqs, "psd": psd, "lags": lags, "autocorr": autocorr,
                "tune_map": tune_map, "tune_pos": tune_pos, "tune_neg": tune_neg,
                "tune_h1": tune_h1, "tune_h2": tune_h2
            }

            # Animal exemplar selection criteria
            if trial_score > 0.001 and trial_score > 2.0 * slow_score:
                if trial_score > animal_exemplars[mouse_name]["type_1_lap_freq"][2]:
                    animal_exemplars[mouse_name]["type_1_lap_freq"] = (rec, diag, trial_score, trial_freq)

            elif slow_score > 0.01 and slow_score > 3.0 * trial_score:
                if slow_score > animal_exemplars[mouse_name]["type_2_slow_only"][2]:
                    animal_exemplars[mouse_name]["type_2_slow_only"] = (rec, diag, slow_score, trial_freq)

            elif slow_score > 0.005 and trial_score > 0.001:
                comb = slow_score + trial_score
                if comb > animal_exemplars[mouse_name]["type_3_both"][2]:
                    animal_exemplars[mouse_name]["type_3_both"] = (rec, diag, comb, trial_freq)

        if len(sess_cell_psds) > 0:
            session_data_list.append({
                "mouse_name": mouse_name, "session_name": sess_name, "depth_type": depth,
                "cell_psds": sess_cell_psds, "slow_peak_freqs": sess_slow_freqs,
                "slow_scores": sess_slow_scores, "freqs_ref": freqs_ref
            })

    # Save master CSV measurements table
    df_cells = pd.DataFrame(cell_records)
    table_path = table_dir / f"{dataset_key.lower()}_complete_oscillatory_measurements.csv"
    df_cells.to_csv(table_path, index=False)
    print(f"[OK] Saved comprehensive measurement table -> {table_path.name} ({len(df_cells)} cells)")

    # Generate summary distribution plots comparing Superficial vs Deep layers
    generate_summary_distribution_plots(df_cells, dataset_key, dataset_out)

    # Generate session heatmaps sorted by slow frequency & intensity
    generate_session_psd_heatmaps(dataset_key, dataset_out, session_data_list)

    # Export animal exemplars
    exemplar_dir = dataset_out / "animal_exemplars"
    for mouse_name, cats in animal_exemplars.items():
        mouse_dir = exemplar_dir / mouse_name
        for cat_key, (cell_rec, diag, score, t_freq) in cats.items():
            if cell_rec is not None:
                export_full_characterization_suite(cell_rec, diag, t_freq, cat_key, mouse_dir / cat_key)
                print(f"[OK] Exported {cat_key} for animal {mouse_name}")

    print(f"[OK] Completed analysis for {dataset_key.upper()} dataset.\n")


def main():
    print("=" * 80)
    print("UNIFIED OSCILLATORY SEQUENCES ANALYSIS PIPELINE")
    print("=" * 80)

    process_dataset("learning")
    process_dataset("lt_rot")

    print("=" * 80)
    print("ALL ANALYSES & REPORTS SUCCESSFULLY COMPLETED!")
    print(f"Output Directory Root: {OUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()