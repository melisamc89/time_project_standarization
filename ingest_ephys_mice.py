#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pickle
from datetime import datetime

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, filtfilt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from elephant.kernels import GaussianKernel
from elephant.statistics import instantaneous_rate
from quantities import ms
import neo

from general_utils import get_speed, get_trial_id


# =============================================================================
# Top-level parameters
# =============================================================================

PROJECT_ROOT = "/home/melisamc/Documents/time_project_standarization"

DATASET_NAME = "lt_ephys_mice"
MODALITY = "ephys"

SOURCE_BASE_DIR = "/home/melisamc/Documents/time_project_mice"
SOURCE_DATA_DIR = os.path.join(SOURCE_BASE_DIR, "data")

MOUSE_NAMES = [
    "Calb01_11_03_16_02_25_time_project",
    "Calb01uLED_10_17_09_05_56_time_project",
    "Calb01uLED_10_18_11_23_15_time_project",
    "Calb01uLED_10_19_11_25_09_time_project",
    "Calb01uLED_10_21_12_48_35_time_project",
    "Thy01uLED_10_19_09_07_25_time_project",
    "Thy01uLED_10_10_09_10_26_time_project",
]

TARGET_FS = 20
RATE_KERNEL_WIDTH = 10

OVERWRITE_STANDARDIZED_FILES = False

CODE_DIR = os.path.join(PROJECT_ROOT, "code")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")

METADATA_DIR = os.path.join(DATA_DIR, "metadata")
STANDARDIZED_DIR = os.path.join(
    DATA_DIR,
    "standardized_sessions",
    DATASET_NAME,
    MODALITY,
)

MASTER_CSV_PATH = os.path.join(METADATA_DIR, "master_dataset_index.csv")


# =============================================================================
# Per-mouse tuned parameters
# Using the values from the code you just shared
# =============================================================================

MOUSE_CONFIG = {
    "Calb01_11_03_16_02_25_time_project": {
        "dir_limit": [20, 100],
        "speed_lim1": 8,
    },
    "Calb01uLED_10_17_09_05_56_time_project": {
        "dir_limit": [20, 100],
        "speed_lim1": 8,
    },
    "Calb01uLED_10_18_11_23_15_time_project": {
        "dir_limit": [30, 120],
        "speed_lim1": 10,
    },
    "Calb01uLED_10_19_11_25_09_time_project": {
        "dir_limit": [15, 75],
        "speed_lim1": 10,
    },
    "Calb01uLED_10_21_12_48_35_time_project": {
        "dir_limit": [15, 75],
        "speed_lim1": 5,
    },
    "Thy01uLED_10_19_09_07_25_time_project": {
        "dir_limit": [15, 75],
        "speed_lim1": 2,
    },
    "Thy01uLED_10_10_09_10_26_time_project": {
        "dir_limit": [15, 75],
        "speed_lim1": 2,
    },
}


# =============================================================================
# Unified CSV schema
# =============================================================================

MASTER_COLUMNS = [
    "dataset_name",
    "modality",
    "depth_type",
    "mouse_name",
    "session_name",
    "session_name_safe",
    "source_base_dir",
    "source_data_dir",
    "source_mouse_dir",
    "source_file_name",
    "source_file_path",
    "case",
    "cells",
    "signal_name",
    "paradigm",
    "model",
    "standardized_file_name",
    "standardized_file_path",
    "n_timepoints",
    "n_neurons",
    "pos_dim",
    "contains_signal",
    "contains_pos",
    "contains_mov_dir",
    "contains_speed",
    "contains_trial_id",
    "contains_time",
    "contains_inner_time",
    "source_container_type",
    "common_crop_length",
    "created_at",
    "notes",
]


# =============================================================================
# Generic helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def initialize_master_csv_if_needed(csv_path, columns):
    ensure_dir(os.path.dirname(csv_path))
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_row_to_csv(csv_path, row, columns):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def save_pickle_standard(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def sanitize_for_filename(text):
    text = str(text)
    out = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def yes_no(x):
    return "yes" if x else "no"


def ensure_signal_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def ensure_pos_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def ensure_col(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


# =============================================================================
# Smooth helper to mimic learning_repo.general_utils.data_process_utils.smooth_data
# =============================================================================

def smooth_data_like_learning_repo(x, sf, std=0.5):
    x = np.asarray(x, dtype=float)
    sigma_samples = std * sf
    return gaussian_filter1d(x, sigma=sigma_samples, mode="nearest")


# =============================================================================
# LFP / ripple helpers
# =============================================================================

def bandpass_lfp(LFP, fs, low=100.0, high=250.0, order=4):
    LFP = np.asarray(LFP)
    nyq = fs * 0.5
    if high >= nyq:
        raise ValueError(f"High cutoff ({high} Hz) must be < Nyquist ({nyq:.2f} Hz).")

    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")

    def _interp_nans_1d(x):
        n = len(x)
        idx = np.arange(n)
        m = np.isfinite(x)
        if m.sum() < 2:
            return x
        y = x.copy()
        y[~m] = np.interp(idx[~m], idx[m], x[m])
        return y

    if LFP.ndim == 1:
        x = _interp_nans_1d(LFP)
        y = sosfiltfilt(sos, x)
        y[~np.isfinite(LFP)] = np.nan
        return y

    if LFP.ndim == 2:
        out = np.empty_like(LFP, dtype=float)
        for c in range(LFP.shape[1]):
            x = _interp_nans_1d(LFP[:, c])
            y = sosfiltfilt(sos, x)
            y[~np.isfinite(LFP[:, c])] = np.nan
            out[:, c] = y
        return out

    raise ValueError("LFP must be 1D or 2D.")


def split_lfp_into_shanks(LFP, n_cols=32, shank_size=8):
    LFP = np.asarray(LFP)
    assert LFP.ndim == 2, "LFP must be 2D (time x channels)"
    assert n_cols % shank_size == 0, "n_cols must be divisible by shank_size"

    LFP = LFP[:, :n_cols]
    n_shanks = n_cols // shank_size

    shank_dict = {}
    for i in range(n_shanks):
        start = i * shank_size
        end = (i + 1) * shank_size
        shank_dict[f"shank{i+1}"] = LFP[:, start:end]
    return shank_dict


def ripple_power_by_shank(lfp_shanks, ripple_time, ripple_shank, fs=None, units="samples", normalize=True):
    ripple_time = np.asarray(ripple_time)
    ripple_shank = np.asarray(ripple_shank).astype(int).ravel()

    if units == "seconds":
        if fs is None:
            raise ValueError("fs is required when units='seconds'")
        ripple_time = (ripple_time * fs).astype(int)

    ripple_power = {shank: np.zeros(8) for shank in lfp_shanks.keys()}

    for i, (start, end) in enumerate(ripple_time):
        shank_id = f"shank{ripple_shank[i]}"
        if shank_id not in lfp_shanks:
            continue

        lfp = lfp_shanks[shank_id]
        start = max(0, int(start))
        end = min(lfp.shape[0], int(end))
        if end <= start:
            continue

        segment = lfp[start:end, :]
        ripple_power[shank_id] += np.sum(segment ** 2, axis=0)

    if normalize:
        for shank, power in ripple_power.items():
            max_val = np.max(power)
            ripple_power[shank] = power / max_val if max_val > 0 else np.zeros_like(power)

    return ripple_power


def classify_neurons_depth(neuron_chNear, best_channels):
    neuron_depth = []

    for ch in neuron_chNear:
        shank_num = (int(ch) - 1) // 8 + 1
        local_ch = (int(ch) - 1) % 8
        shank_key = f"shank{shank_num}"

        best_ch = best_channels.get(shank_key, None)
        if best_ch is None:
            neuron_depth.append("unknown")
        elif local_ch <= best_ch:
            neuron_depth.append("deep")
        else:
            neuron_depth.append("superficial")

    return neuron_depth


def simplify_cell_types(cell_type_list):
    return ["pyr" if "pyr" in str(ct).lower() else "int" for ct in cell_type_list]


# =============================================================================
# Behavior / spike helpers
# =============================================================================

def compute_internal_trial_time(trial_ids, sampling_rate):
    """
    Recompute inner time AFTER filtering.
    Every time trial_id changes, inner_time resets to 0.
    """
    trial_ids = np.asarray(trial_ids).reshape(-1)

    if len(trial_ids) == 0:
        return np.array([], dtype=float)

    inner_time = np.zeros(len(trial_ids), dtype=float)
    dt = 1.0 / sampling_rate

    start_idx = 0
    for i in range(1, len(trial_ids)):
        if trial_ids[i] != trial_ids[i - 1]:
            start_idx = i
        inner_time[i] = (i - start_idx) * dt

    return inner_time


def spikes_to_rates(spikes, kernel_width=10):
    """
    spikes: neurons x time
    returns: time x neurons
    """
    gk = GaussianKernel(kernel_width * ms)
    rates = []

    for sp in spikes:
        sp_times = np.where(sp > 0)[0]
        st = neo.SpikeTrain(sp_times, units="ms", t_stop=len(sp))
        r = instantaneous_rate(
            st,
            kernel=gk,
            sampling_period=1.0 * ms
        ).magnitude.squeeze()
        rates.append(r)

    rates = np.vstack(rates)
    return rates.T


def resample_lfp_to_behavior_grid(lfp, lfp_fs, behavior_times):
    lfp = np.asarray(lfp)
    t_lfp = np.arange(lfp.shape[0]) / float(lfp_fs)
    behavior_times = np.asarray(behavior_times).reshape(-1)

    if lfp.ndim == 1:
        f = interp1d(t_lfp, lfp, kind="linear", bounds_error=False, fill_value="extrapolate")
        return f(behavior_times)

    out = np.zeros((len(behavior_times), lfp.shape[1]), dtype=float)
    for i in range(lfp.shape[1]):
        f = interp1d(t_lfp, lfp[:, i], kind="linear", bounds_error=False, fill_value="extrapolate")
        out[:, i] = f(behavior_times)
    return out


def compute_movement_direction_like_shared_code(position, speed=None, sf=20, speed_th=2):
    """
    Keep this direction computation exactly as in the newer standardized code.
    """
    position = ensure_pos_2d(position)

    if speed is None:
        speed = np.sqrt(np.sum((np.diff(position, axis=0) * sf) ** 2, axis=1))
        speed = np.hstack((speed[0], speed))
        speed = smooth_data_like_learning_repo(speed, sf=sf, std=0.5)
    else:
        speed = np.asarray(speed).reshape(-1)

    mov_direction = np.zeros((speed.shape[0],)) * np.nan
    mov_direction[speed < speed_th] = 0

    x_speed = np.diff(position[:, 0]) / (1 / sf)
    x_speed = np.hstack((x_speed[0], x_speed))
    x_speed = smooth_data_like_learning_repo(x_speed, sf=sf, std=0.5)

    right_moving = np.logical_and(speed > speed_th, x_speed > 0)
    mov_direction[right_moving] = 1

    left_moving = np.logical_and(speed > speed_th, x_speed < 0)
    mov_direction[left_moving] = -1

    mov_direction = np.round(
        smooth_data_like_learning_repo(mov_direction, sf=sf, std=0.5), 0
    ).astype(int).copy()

    return mov_direction


# =============================================================================
# Core per-mouse builder
# =============================================================================

def build_standardized_mouse_outputs(mouse_name):
    if mouse_name not in MOUSE_CONFIG:
        raise KeyError(f"Missing config for mouse: {mouse_name}")

    cfg = MOUSE_CONFIG[mouse_name]
    dir_limit = cfg["dir_limit"]
    speed_lim1 = cfg["speed_lim1"]

    source_file_name = f"{mouse_name}.mat"
    source_file_path = os.path.join(SOURCE_DATA_DIR, source_file_name)

    if not os.path.isfile(source_file_path):
        print(f"[WARNING] Missing source file: {source_file_path}")
        return []

    print(f"\nProcessing mouse: {mouse_name}")

    mat = loadmat(source_file_path, simplify_cells=True)
    time_project = mat["time_project"]

    pos_raw = ensure_pos_2d(time_project["beh"]["pos"])
    time_stamps_raw = np.asarray(time_project["beh"]["timeStamps"]).reshape(-1)
    beh_fs = float(time_project["beh"]["fs"])

    LFP = ensure_signal_2d(time_project["LFP"]["LFP"])
    lfp_fs = float(time_project["LFP"]["fs"])

    spk_times = np.asarray(time_project["spks"]["spkTimes"]).reshape(-1)
    spk_clus = np.asarray(time_project["spks"]["spkClus"]).reshape(-1)

    neuron_chNear = np.asarray(time_project["spks"]["chNear"]).reshape(-1)
    neuron_cellType = list(np.asarray(time_project["unit"]["cellType"]).reshape(-1))
    neuron_cID = np.asarray(time_project["unit"]["cIds"]).reshape(-1)

    ripple_time = np.asarray(time_project["ripple"]["time"])
    ripple_shank = np.asarray(time_project["ripple"]["shank"]).reshape(-1)

    # -------------------------------------------------------------------------
    # Deep/superficial split from ripple-band LFP
    # -------------------------------------------------------------------------
    LFP_bp = bandpass_lfp(LFP, fs=lfp_fs, low=100.0, high=250.0, order=4)
    lfp_shanks = split_lfp_into_shanks(LFP_bp, n_cols=32, shank_size=8)
    ripple_power = ripple_power_by_shank(
        lfp_shanks,
        ripple_time,
        ripple_shank,
        fs=lfp_fs,
        units="samples",
        normalize=True,
    )
    best_channels = {shank: int(np.argmax(power)) for shank, power in ripple_power.items()}

    neuron_depth_labels = np.array(classify_neurons_depth(neuron_chNear, best_channels))
    neuron_cellType_simple = np.array(simplify_cell_types(neuron_cellType))

    # -------------------------------------------------------------------------
    # Behavior-aligned signal construction
    # -------------------------------------------------------------------------
    non_nan_idx = ~np.isnan(pos_raw[:, 0])
    position = pos_raw[non_nan_idx, :]
    time_stamps = time_stamps_raw[non_nan_idx]

    position_x = position[:, 0]

    if beh_fs == 20:
        nyquist = beh_fs / 2.0
        cutoff = nyquist - 5
        normalized_cutoff = cutoff / (beh_fs / 2.0)
        b, a = butter(N=4, Wn=normalized_cutoff, btype="low")

        filtered_positions = filtfilt(b, a, position_x)

        position_ds = filtered_positions
        time_stamps_ds = time_stamps
        signal_fs = 20.0

    elif beh_fs == 2500:
        downsampling_factor = int(beh_fs / TARGET_FS)

        nyquist = TARGET_FS / 2.0
        cutoff = nyquist - 5
        normalized_cutoff = cutoff / (beh_fs / 2.0)
        b, a = butter(N=4, Wn=normalized_cutoff, btype="low")

        filtered_positions = filtfilt(b, a, position_x)

        position_ds = filtered_positions[::downsampling_factor]
        time_stamps_ds = time_stamps[::downsampling_factor]
        signal_fs = 20.0

    else:
        raise ValueError(f"Unsupported behavior fs for {mouse_name}: {beh_fs}")

    speed = np.abs(get_speed(position_ds, time_stamps_ds))

    # Keep newer direction computation
    direction = compute_movement_direction_like_shared_code(
        position=ensure_pos_2d(position_ds),
        speed=speed,
        sf=signal_fs,
        speed_th=speed_lim1,
    )

    # Keep trial_id computed with dir_limit as before
    trial_id = get_trial_id(position_ds, dir_limit[1], dir_limit[0])

    # Build spike count matrix on the behavior bins
    n_neurons_total = len(neuron_cID)
    spike_matrix = np.zeros((len(time_stamps_ds), n_neurons_total), dtype=float)

    for idx, neuron in enumerate(neuron_cID):
        neuron_spikes_index = np.where(spk_clus == neuron)[0]
        neuron_spikes = spk_times[neuron_spikes_index]

        indices = np.searchsorted(time_stamps_ds, neuron_spikes, side="right") - 1
        valid_indices = indices[(indices >= 0) & (indices < len(time_stamps_ds))]
        counts = np.bincount(valid_indices, minlength=len(time_stamps_ds))
        spike_matrix[:, idx] = counts

    # -------------------------------------------------------------------------
    # Filtering rule aligned with the older code logic:
    # remove only low-speed samples
    # DO NOT remove samples outside dir_limit
    # DO NOT reindex time from zero
    # -------------------------------------------------------------------------
    keep_idx = np.where(np.isfinite(speed) & (speed > speed_lim1))[0]

    print(f"Original n_timepoints: {len(speed)}")
    print(f"Kept n_timepoints: {len(keep_idx)}")

    position_final = position_ds[keep_idx]
    speed_final = speed[keep_idx]
    direction_final = direction[keep_idx]
    time_final = time_stamps_ds[keep_idx]   # keep original timestamps, no reindex
    trial_id_final = trial_id[keep_idx]
    spike_counts_final = spike_matrix[keep_idx, :]

    lfp_original = LFP
    lfp_20hz_full = resample_lfp_to_behavior_grid(LFP, lfp_fs, time_stamps_ds)
    lfp_20hz_final = lfp_20hz_full[keep_idx, :]

    # Recompute inner time AFTER filtering
    inner_time_final = compute_internal_trial_time(trial_id_final, signal_fs)

    # -------------------------------------------------------------------------
    # Apply Gaussian kernel AFTER filtering
    # -------------------------------------------------------------------------
    data_rates = spikes_to_rates(spike_counts_final.T, kernel_width=RATE_KERNEL_WIDTH)

    pyr_index = np.where(neuron_cellType_simple == "pyr")[0]
    deep_index = np.where((neuron_cellType_simple == "pyr") & (neuron_depth_labels == "deep"))[0]
    sup_index = np.where((neuron_cellType_simple == "pyr") & (neuron_depth_labels == "superficial"))[0]

    whole_signal = ensure_signal_2d(data_rates[:, pyr_index])
    deep_signal = ensure_signal_2d(data_rates[:, deep_index])
    sup_signal = ensure_signal_2d(data_rates[:, sup_index])

    pos_matrix = ensure_pos_2d(position_final)
    mov_dir_col = ensure_col(direction_final)
    speed_col = ensure_col(speed_final)
    trial_id_col = ensure_col(trial_id_final)
    time_col = ensure_col(time_final)
    inner_time_col = ensure_col(inner_time_final)

    subset_map = {
        "whole_population": whole_signal,
        "deep": deep_signal,
        "superficial": sup_signal,
    }

    outputs = []
    for depth_type, signal in subset_map.items():
        standardized = {
            "metadata": {
                "dataset_name": DATASET_NAME,
                "modality": MODALITY,
                "depth_type": depth_type,
                "mouse_name": mouse_name,
                "session_name": "lt_ephys",
                "case": (
                    f"speed_filtered_keep_original_time_"
                    f"new_direction_old_mouse_params_"
                    f"speedth_{speed_lim1}_"
                    f"dir_{dir_limit[0]}_{dir_limit[1]}_"
                    f"rates_kernel{RATE_KERNEL_WIDTH}"
                ),
                "cells": "pyramidal_only",
                "signal_name": f"spikes_to_rates_kernel{RATE_KERNEL_WIDTH}",
                "paradigm": "",
                "model": "",
                "common_crop_length": "",
                "beh_fs_original": beh_fs,
                "signal_fs_final": signal_fs,
                "lfp_fs_original": lfp_fs,
                "dir_limit": dir_limit,
                "speed_lim1": speed_lim1,
                "n_timepoints_before_filter": int(len(time_stamps_ds)),
                "n_timepoints_after_filter": int(len(time_final)),
            },
            "signal": signal,
            "pos": pos_matrix,
            "mov_dir": mov_dir_col,
            "speed": speed_col,
            "trial_id": trial_id_col,
            "time": time_col,
            "inner_time": inner_time_col,
            "lfp_original_fs": lfp_original,
            "lfp_20hz": lfp_20hz_final,
        }
        outputs.append((depth_type, standardized, source_file_name, source_file_path))

    return outputs


# =============================================================================
# CSV row builder
# =============================================================================

def row_from_standardized(
    mouse_name,
    depth_type,
    standardized,
    source_file_name,
    source_file_path,
    standardized_file_name,
    standardized_file_path,
):
    signal = standardized["signal"]
    pos = standardized["pos"]
    mov_dir = standardized["mov_dir"]
    speed = standardized["speed"]
    trial_id = standardized["trial_id"]
    time = standardized["time"]
    inner_time = standardized["inner_time"]

    return {
        "dataset_name": DATASET_NAME,
        "modality": MODALITY,
        "depth_type": depth_type,
        "mouse_name": mouse_name,
        "session_name": "lt_ephys",
        "session_name_safe": "lt_ephys",
        "source_base_dir": SOURCE_BASE_DIR,
        "source_data_dir": SOURCE_DATA_DIR,
        "source_mouse_dir": SOURCE_DATA_DIR,
        "source_file_name": source_file_name,
        "source_file_path": source_file_path,
        "case": standardized["metadata"]["case"],
        "cells": standardized["metadata"]["cells"],
        "signal_name": standardized["metadata"]["signal_name"],
        "paradigm": "",
        "model": "",
        "standardized_file_name": standardized_file_name,
        "standardized_file_path": standardized_file_path,
        "n_timepoints": int(signal.shape[0]),
        "n_neurons": int(signal.shape[1]),
        "pos_dim": int(pos.shape[1]) if pos.ndim > 1 else 1,
        "contains_signal": yes_no(signal is not None),
        "contains_pos": yes_no(pos is not None),
        "contains_mov_dir": yes_no(mov_dir is not None),
        "contains_speed": yes_no(speed is not None),
        "contains_trial_id": yes_no(trial_id is not None),
        "contains_time": yes_no(time is not None),
        "contains_inner_time": yes_no(inner_time is not None),
        "source_container_type": "single_mat_file_split_into_whole_deep_superficial",
        "common_crop_length": "",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": (
            "uses ripple/LFP-based deep-superficial labeling; "
            "uses current direction computation; "
            "uses old per-mouse speed_lim1 and dir_limit values; "
            "time keeps original timestamps after filtering; "
            "inner_time recomputed after filtering; "
            "kernel width = 10"
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    for path in [CODE_DIR, DATA_DIR, ANALYSIS_DIR, METADATA_DIR, STANDARDIZED_DIR]:
        ensure_dir(path)

    initialize_master_csv_if_needed(MASTER_CSV_PATH, MASTER_COLUMNS)

    total_rows = 0

    for mouse_name in MOUSE_NAMES:
        outputs = build_standardized_mouse_outputs(mouse_name)

        for depth_type, standardized, source_file_name, source_file_path in outputs:
            mouse_output_dir = os.path.join(STANDARDIZED_DIR, depth_type, mouse_name)
            ensure_dir(mouse_output_dir)

            standardized_file_name = (
                f"{DATASET_NAME}_{mouse_name}_lt_ephys_{MODALITY}_{depth_type}_"
                f"{sanitize_for_filename(standardized['metadata']['signal_name'])}.pkl"
            )
            standardized_file_path = os.path.join(mouse_output_dir, standardized_file_name)

            if os.path.exists(standardized_file_path) and not OVERWRITE_STANDARDIZED_FILES:
                print(f"File exists, skipping save: {standardized_file_name}")
            else:
                save_pickle_standard(standardized_file_path, standardized)
                print(f"Saved: {standardized_file_name}")

            row = row_from_standardized(
                mouse_name=mouse_name,
                depth_type=depth_type,
                standardized=standardized,
                source_file_name=source_file_name,
                source_file_path=source_file_path,
                standardized_file_name=standardized_file_name,
                standardized_file_path=standardized_file_path,
            )
            append_row_to_csv(MASTER_CSV_PATH, row, MASTER_COLUMNS)
            total_rows += 1

    print("\nDone.")
    print(f"Appended rows: {total_rows}")
    print(f"Master CSV: {MASTER_CSV_PATH}")


if __name__ == "__main__":
    main()