#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hybrid Ingestion Pipeline:
1. Neural Data: Clean traces + speed thresholding (> 3.0) + Gaussian smoothing (sigma = 6) from lt_egrin_data.pkl.
2. LFP & CWT: Raw LFP extraction, best theta channel selection, continuous CWT, and Hilbert envelopes/phases
   loaded directly from individual .mat files.
"""

import csv
from datetime import datetime
import glob
import os
import pickle
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, fftconvolve, hilbert, sosfiltfilt, welch

# =============================================================================
# Directories & Settings
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
SOURCE_BASE_DIR = "/home/hippolab/Documents/melisa/lt_egrin_project"
SOURCE_PICKLE_FILE = os.path.join(SOURCE_BASE_DIR, "data", "lt_egrin_data.pkl")
SOURCE_MAT_DIR = os.path.join(SOURCE_BASE_DIR, "data")  # Where mouse folders containing .mat files reside

DATASET_NAME = "lt_egrin"
MODALITY = "calcium_imaging"

SPEED_TH = 3.0
SIG_FILT = 6.0
FS_IMAGING = 20.0

OVERWRITE_STANDARDIZED_FILES = True

CODE_DIR = os.path.join(PROJECT_ROOT, "code")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")
DIAGNOSTICS_DIR = os.path.join(ANALYSIS_DIR, "cwt_and_lfp_diagnostics")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
STANDARDIZED_DIR = os.path.join(
    DATA_DIR,
    "standardized_sessions",
    DATASET_NAME,
    MODALITY,
)

MASTER_CSV_PATH = os.path.join(METADATA_DIR, "master_dataset_index.csv")

MASTER_COLUMNS = [
    "dataset_name", "modality", "depth_type", "mouse_name", "session_name",
    "session_name_safe", "source_base_dir", "source_data_dir", "source_mouse_dir",
    "source_file_name", "source_file_path", "case", "cells", "signal_name",
    "best_theta_channel", "fs_lfp", "cwt_fmin", "cwt_fmax", "n_cwt_freqs",
    "paradigm", "model", "standardized_file_name", "standardized_file_path",
    "n_timepoints", "n_neurons", "pos_dim", "contains_signal", "contains_pos",
    "contains_mov_dir", "contains_speed", "contains_trial_id", "contains_time",
    "contains_inner_time", "contains_full_rate_lfp", "source_container_type",
    "common_crop_length", "created_at", "notes"
]

LFP_BANDS = {
    "slow":   (0.01,  0.10),
    "delta":  (1.00,  3.00),
    "theta":  (8.00,  12.00),
    "gamma":  (40.00, 90.00),
    "ripple": (100.0, 250.0),
    "mua":    (300.0, 1000.0)
}

CWT_FMIN = 0.001
CWT_FMAX = 1000.0
CWT_NUM_FREQS = 90
CWT_W0 = 6.0


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(text):
    text = str(text)
    return "".join([ch if (ch.isalnum() or ch in ["-", "_"]) else "_" for ch in text])


def ensure_col(x):
    x = np.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def infer_depth_from_mouse_name(mouse_name):
    low = mouse_name.lower()
    if "calb" in low:
        return "superficial"
    if "thy" in low:
        return "deep"
    return "unknown"


def compute_inner_time(trial_id, fs):
    trial_id = np.asarray(trial_id).reshape(-1)
    inner_time = np.zeros_like(trial_id, dtype=float)
    if len(trial_id) == 0:
        return inner_time.reshape(-1, 1)

    current_trial = trial_id[0]
    start_idx = 0
    for i in range(1, len(trial_id)):
        if trial_id[i] != current_trial:
            current_trial = trial_id[i]
            start_idx = i
        inner_time[i] = (i - start_idx) / fs
    return inner_time.reshape(-1, 1)


def compute_movement_direction(position, sf=20.0, speed_th=2.0):
    pos_x = np.asarray(position)[:, 0] if position.ndim > 1 else np.asarray(position)
    x_speed = np.diff(pos_x) / (1.0 / sf)
    x_speed = np.hstack((x_speed[0], x_speed))
    x_speed = uniform_filter1d(x_speed, size=int(sf))

    mov_dir = np.zeros_like(x_speed, dtype=int)
    mov_dir[x_speed > speed_th] = 1
    mov_dir[x_speed < -speed_th] = -1
    return mov_dir


# =============================================================================
# Hilbert & CWT Extraction Functions
# =============================================================================

def extract_lfp_bands_and_hilbert_20hz(lfp_chan, fs_lfp, target_len, fs_target=20.0):
    t_lfp = np.arange(len(lfp_chan)) / fs_lfp
    t_target = np.arange(target_len) / fs_target

    extracted_dict = {}
    smooth_win = max(1, int(round((1.0 / fs_target) * fs_lfp)))
    nyquist = fs_lfp / 2.0

    for label, (low, high) in LFP_BANDS.items():
        high_safe = min(high, nyquist - 1.0)
        low_safe = max(0.0005, low)

        if low_safe >= high_safe:
            w_20hz = np.zeros(target_len, dtype=float)
            extracted_dict[f"lfp_{label}"] = w_20hz
            extracted_dict[f"lfp_{label}_20hz"] = w_20hz
            extracted_dict[f"lfp_{label}_envelope"] = w_20hz
            extracted_dict[f"lfp_{label}_hilbert_20hz"] = w_20hz
            extracted_dict[f"lfp_{label}_phase"] = w_20hz
            extracted_dict[f"lfp_{label}_phase_20hz"] = w_20hz
            continue

        sos = butter(2 if low_safe < 0.1 else 3, [low_safe, high_safe], btype="bandpass", fs=fs_lfp, output="sos")
        filt_full = sosfiltfilt(sos, lfp_chan)

        analytic_sig = hilbert(filt_full)
        env_full = np.abs(analytic_sig)
        phase_full = np.angle(analytic_sig)

        f_sig = interp1d(t_lfp, filt_full, bounds_error=False, fill_value="extrapolate")
        w_20hz = f_sig(t_target)

        env_smooth = uniform_filter1d(env_full, size=smooth_win)
        f_env = interp1d(t_lfp, env_smooth, bounds_error=False, fill_value="extrapolate")
        env_20hz = f_env(t_target)

        phase_unwrapped = np.unwrap(phase_full)
        f_phase = interp1d(t_lfp, phase_unwrapped, bounds_error=False, fill_value="extrapolate")
        phase_20hz_unwrapped = f_phase(t_target)
        phase_20hz = (phase_20hz_unwrapped + np.pi) % (2.0 * np.pi) - np.pi

        extracted_dict[f"lfp_{label}"] = w_20hz
        extracted_dict[f"lfp_{label}_20hz"] = w_20hz
        extracted_dict[f"lfp_{label}_envelope"] = env_20hz
        extracted_dict[f"lfp_{label}_hilbert_20hz"] = env_20hz
        extracted_dict[f"lfp_{label}_phase"] = phase_20hz
        extracted_dict[f"lfp_{label}_phase_20hz"] = phase_20hz

    return extracted_dict


def select_maximal_theta_channel(lfp_raw, fs, speed=None):
    n_channels = lfp_raw.shape[1]
    theta_powers = np.zeros(n_channels, dtype=float)

    eval_matrix = lfp_raw[speed > SPEED_TH, :] if (speed is not None and np.sum(speed > SPEED_TH) > int(5 * fs)) else lfp_raw
    nperseg = int(min(eval_matrix.shape[0], round(2.0 * fs)))

    for ch in range(n_channels):
        freqs, psd = welch(eval_matrix[:, ch], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, scaling="density")
        theta_mask = (freqs >= 8.0) & (freqs <= min(12.0, fs / 2.0 - 0.5))
        if np.sum(theta_mask) == 0:
            theta_mask = (freqs >= 1.0) & (freqs <= min(3.0, fs / 2.0 - 0.5))
        theta_powers[ch] = trapezoid(psd[theta_mask], freqs[theta_mask])

    best_ch = int(np.argmax(theta_powers))
    print(f"  Selected Channel {best_ch + 1} with max theta power: {theta_powers[best_ch]:.2e}")
    return best_ch


def compute_full_cwt(signal_1d, fs):
    n_pts = len(signal_1d)
    total_sec = n_pts / fs
    fmax_safe = min(CWT_FMAX, fs / 2.0 - 0.5)
    freqs = np.geomspace(CWT_FMIN, fmax_safe, CWT_NUM_FREQS)
    cwt_power = np.zeros((n_pts, CWT_NUM_FREQS), dtype=np.float32)

    print(f"  Computing CWT ({CWT_NUM_FREQS} scales from {CWT_FMIN:.3e} Hz to {fmax_safe:.1f} Hz)...")
    for i, f0 in enumerate(freqs):
        s = CWT_W0 / (2.0 * np.pi * f0)
        t_max = min(4.0 * s, max(8.0, total_sec * 0.5))
        t_kern = np.arange(-t_max, t_max, 1.0 / fs)
        wavelet = (
                np.pi ** (-0.25)
                * np.exp(1j * 2.0 * np.pi * f0 * t_kern)
                * np.exp(-0.5 * (t_kern / s) ** 2)
        )
        wavelet /= np.sqrt(s * fs)
        conv = fftconvolve(signal_1d, wavelet, mode="same")
        cwt_power[:, i] = (np.abs(conv) ** 2).astype(np.float32)

    return cwt_power, freqs


# =============================================================================
# Main Processing Pipeline (Hybrid: Pickle Clean Traces + .mat LFP Files)
# =============================================================================

def process_mouse(mouse_name, mouse_pickle_data):
    depth_type = infer_depth_from_mouse_name(mouse_name)

    # 1. Load Neural Activity from Pickle File
    raw_traces = np.asarray(mouse_pickle_data["clean_traces"], dtype=float)
    n_frames = raw_traces.shape[0]

    position = np.asarray(mouse_pickle_data["position"])
    speed = np.asarray(mouse_pickle_data["speed"]).reshape(-1)
    trial_id = np.asarray(mouse_pickle_data["trial_id"]).reshape(-1)
    mov_dir = np.asarray(mouse_pickle_data["mov_direction"]).reshape(-1)
    t_beh = np.arange(n_frames) / FS_IMAGING

    # 2. Locate and Load Raw LFP from Corresponding .mat file
    mouse_dir = os.path.join(SOURCE_MAT_DIR, mouse_name)
    mat_candidates = glob.glob(os.path.join(mouse_dir, "*_tracesLFP.mat"))
    if not mat_candidates:
        mat_candidates = glob.glob(os.path.join(mouse_dir, "*.mat"))

    if not mat_candidates:
        print(f"[WARNING] No .mat file found for {mouse_name} in {mouse_dir}. Skipping LFP extraction.")
        return None

    mat_path = mat_candidates[0]
    mat_file_name = os.path.basename(mat_path)
    print(f"\n{'='*70}\nLoading LFP from .mat for: {mouse_name} ({mat_file_name})\n{'='*70}")

    mat_data = loadmat(mat_path, simplify_cells=True)
    tlfp = mat_data["tracesLFP"]

    raw_lfp = np.asarray(tlfp["LFP"]["LFP"], dtype=float)
    if raw_lfp.shape[0] == 16 and raw_lfp.shape[1] > 16:
        raw_lfp = raw_lfp.T

    lfp_time = tlfp["LFP"].get("time", None)
    if lfp_time is not None and len(lfp_time) > 1:
        t_lfp = np.asarray(lfp_time).reshape(-1)
        fs_lfp = 1.0 / np.nanmedian(np.diff(t_lfp))
    else:
        fs_lfp = float(raw_lfp.shape[0]) / (float(n_frames) / FS_IMAGING)
        t_lfp = np.linspace(0, (n_frames - 1) / FS_IMAGING, raw_lfp.shape[0])

    print(f"  LFP Raw Matrix: {raw_lfp.shape} | fs: {fs_lfp:.1f} Hz")

    speed_on_lfp = interp1d(t_beh, speed, bounds_error=False, fill_value="extrapolate")(t_lfp)
    best_ch_idx = select_maximal_theta_channel(raw_lfp, fs_lfp, speed=speed_on_lfp)
    best_lfp_raw = raw_lfp[:, best_ch_idx]

    cwt_full, cwt_freqs = compute_full_cwt(best_lfp_raw, fs_lfp)

    print(f"  Extracting Hilbert bands, envelopes, and phases -> downsampling to {FS_IMAGING} Hz...")
    cwt_20hz_continuous = interp1d(t_lfp, cwt_full, axis=0, bounds_error=False, fill_value="extrapolate")(t_beh)
    lfp_hilbert_dict = extract_lfp_bands_and_hilbert_20hz(best_lfp_raw, fs_lfp, target_len=n_frames, fs_target=FS_IMAGING)
    lfp_raw_20hz = interp1d(t_lfp, best_lfp_raw, axis=0, bounds_error=False, fill_value="extrapolate")(t_beh).reshape(-1, 1)

    # 3. Apply Speed Thresholding (> 3.0) and Gaussian Filtering (sigma = 6)
    is_running = speed > SPEED_TH
    valid_index = np.where(is_running)[0]

    print(f"  Frames: {n_frames} | Retained running frames: {len(valid_index)} ({100*len(valid_index)/n_frames:.1f}%)")

    inner_time_full = compute_inner_time(trial_id, FS_IMAGING)

    signal_filtered = raw_traces[valid_index, :]
    signal_filtered = gaussian_filter1d(signal_filtered, sigma=SIG_FILT, axis=0)

    pos_filtered = ensure_col(position[valid_index]) if position.ndim == 1 else position[valid_index, :]
    speed_filtered = ensure_col(speed[valid_index])
    mov_dir_filtered = ensure_col(mov_dir[valid_index])
    trial_id_filtered = ensure_col(trial_id[valid_index])
    inner_time_filtered = ensure_col(inner_time_full[valid_index])
    time_filtered = ensure_col(valid_index / FS_IMAGING)

    cwt_20hz_filtered = cwt_20hz_continuous[valid_index, :]
    lfp_20hz_filtered = lfp_raw_20hz[valid_index, :]
    hilbert_features_filtered = {k: ensure_col(v[valid_index]) for k, v in lfp_hilbert_dict.items()}

    standardized = {
        "metadata": {
            "dataset_name": DATASET_NAME,
            "modality": MODALITY,
            "depth_type": depth_type,
            "mouse_name": mouse_name,
            "session_name": "lt_egrin",
            "case": f"speed_threshold_{SPEED_TH}_gaussian_sigma{SIG_FILT}_mat_lfp_suite",
            "cells": "all_cells",
            "signal_name": f"clean_traces_gaussian_sigma{SIG_FILT}",
            "fs": FS_IMAGING,
            "fs_lfp": float(fs_lfp),
            "signal_fs_final": FS_IMAGING,
            "best_theta_channel": int(best_ch_idx + 1),
            "cwt_fmin": float(cwt_freqs[0]),
            "cwt_fmax": float(cwt_freqs[-1]),
            "n_cwt_freqs": len(cwt_freqs),
            "has_cwt": True,
            "has_lfp_bands": True,
            "has_lfp_envelopes": True,
            "has_lfp_phases": True,
            "has_full_rate_lfp": True,
            "total_session_frames": int(n_frames),
            "retained_valid_frames": int(len(valid_index)),
        },
        "signal": signal_filtered,
        "pos": pos_filtered,
        "speed": speed_filtered,
        "mov_dir": mov_dir_filtered,
        "trial_id": trial_id_filtered,
        "time": time_filtered,
        "inner_time": inner_time_filtered,
        "lfp_20hz": lfp_20hz_filtered,
        "cwt": cwt_20hz_filtered,
        "cwt_freqs": np.asarray(cwt_freqs, dtype=float),
        "valid_index": valid_index,
        **hilbert_features_filtered,
        "lfp_full_rate_clean": raw_lfp,
        "t_lfp": t_lfp,
    }

    return standardized, depth_type, mat_file_name, mat_path, int(best_ch_idx + 1), float(fs_lfp), cwt_freqs


def append_to_master_csv(csv_path, standardized, mouse_name, depth_type, src_name, src_path, std_name, std_path, best_theta_ch, fs_lfp, cwt_freqs):
    sig = standardized["signal"]
    pos = standardized["pos"]
    row = {
        "dataset_name": DATASET_NAME, "modality": MODALITY, "depth_type": depth_type,
        "mouse_name": mouse_name, "session_name": "lt_egrin", "session_name_safe": "lt_egrin",
        "source_base_dir": SOURCE_BASE_DIR, "source_data_dir": SOURCE_MAT_DIR,
        "source_mouse_dir": os.path.dirname(src_path), "source_file_name": src_name,
        "source_file_path": src_path, "case": standardized["metadata"]["case"],
        "cells": standardized["metadata"]["cells"], "signal_name": standardized["metadata"]["signal_name"],
        "best_theta_channel": best_theta_ch, "fs_lfp": f"{fs_lfp:.2f}",
        "cwt_fmin": f"{cwt_freqs[0]:.6f}", "cwt_fmax": f"{cwt_freqs[-1]:.2f}",
        "n_cwt_freqs": len(cwt_freqs), "paradigm": "", "model": "",
        "standardized_file_name": std_name, "standardized_file_path": std_path,
        "n_timepoints": int(sig.shape[0]), "n_neurons": int(sig.shape[1]),
        "pos_dim": int(pos.shape[1]) if pos.ndim > 1 else 1,
        "contains_signal": "yes" if sig is not None else "no",
        "contains_pos": "yes" if pos is not None else "no",
        "contains_mov_dir": "yes" if standardized["mov_dir"] is not None else "no",
        "contains_speed": "yes" if standardized["speed"] is not None else "no",
        "contains_trial_id": "yes" if standardized["trial_id"] is not None else "no",
        "contains_time": "yes" if standardized["time"] is not None else "no",
        "contains_inner_time": "yes" if standardized["inner_time"] is not None else "no",
        "contains_full_rate_lfp": "yes", "source_container_type": "hybrid_pickle_and_mat",
        "common_crop_length": "", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": f"Clean traces from pickle + raw LFP from .mat (Ch {best_theta_ch})."
    }

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    for p in [CODE_DIR, DATA_DIR, ANALYSIS_DIR, DIAGNOSTICS_DIR, METADATA_DIR, STANDARDIZED_DIR]:
        ensure_dir(p)

    if not os.path.isfile(MASTER_CSV_PATH):
        with open(MASTER_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MASTER_COLUMNS).writeheader()

    if not os.path.isfile(SOURCE_PICKLE_FILE):
        raise FileNotFoundError(f"Source pickle file not found: {SOURCE_PICKLE_FILE}")

    with open(SOURCE_PICKLE_FILE, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected source data to be a dict, got {type(data)}")

    print(f"Loaded source pickle with {len(data)} mice: {sorted(data.keys())}")

    total_rows = 0
    for mouse_name in sorted(data.keys()):
        mouse_pickle_data = data[mouse_name]

        res = process_mouse(mouse_name, mouse_pickle_data)
        if res is None:
            continue

        standardized, depth_type, src_name, src_path, best_ch, fs_lfp, freqs = res

        mouse_out_dir = os.path.join(STANDARDIZED_DIR, depth_type, mouse_name)
        ensure_dir(mouse_out_dir)

        std_file_name = sanitize_for_filename(
            f"{DATASET_NAME}_{mouse_name}_lt_egrin_{MODALITY}_{depth_type}_{standardized['metadata']['signal_name']}.pkl"
        )
        if not std_file_name.endswith(".pkl"):
            std_file_name += ".pkl"

        std_file_path = os.path.join(mouse_out_dir, std_file_name)

        if os.path.exists(std_file_path) and not OVERWRITE_STANDARDIZED_FILES:
            print(f"File exists, skipping save: {std_file_name}")
        else:
            with open(std_file_path, "wb") as f:
                pickle.dump(standardized, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Standardized dataset written to: {std_file_path}")

        append_to_master_csv(
            csv_path=MASTER_CSV_PATH, standardized=standardized, mouse_name=mouse_name,
            depth_type=depth_type, src_name=src_name, src_path=src_path, std_name=std_file_name,
            std_path=std_file_path, best_theta_ch=best_ch, fs_lfp=fs_lfp, cwt_freqs=freqs
        )
        total_rows += 1

    print(f"\n[SUCCESS] Completed hybrid ingestion successfully across {total_rows} mice.")


if __name__ == "__main__":
    main()