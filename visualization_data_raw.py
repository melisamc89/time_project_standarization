#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

CSV_PATH = "/home/melisamc/Documents/time_project_standarization/data/metadata/master_dataset_index.csv"
OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/basic_timeseries_test_split"

PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")

# Limit number of neurons in shifted trace plot for readability
MAX_NEURONS_TO_PLOT = 120

# Neuron ordering mode:
# "peak_time" -> sort by time of maximum peak
# "none"      -> keep original order
NEURON_ORDER_MODE = "peak_time"

# Plot one vertical line every N trial transitions
TRIAL_LINE_EVERY = 4


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_pickle_standard(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


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


def normalize_0_1_per_neuron(signal):
    signal = np.asarray(signal, dtype=float)
    mins = np.nanmin(signal, axis=0, keepdims=True)
    maxs = np.nanmax(signal, axis=0, keepdims=True)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    return (signal - mins) / denom


def base_name_from_row(row_idx, row):
    return (
        f"row_{row_idx:04d}_"
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


def compute_neuron_order(signal, mode="peak_time"):
    signal = ensure_2d(signal)

    if mode == "none":
        return np.arange(signal.shape[1])

    if mode == "peak_time":
        peak_idx = np.argmax(signal, axis=0)
        return np.argsort(peak_idx)

    raise ValueError(f"Unknown NEURON_ORDER_MODE: {mode}")


def reorder_signal(signal, neuron_order):
    signal = ensure_2d(signal)
    neuron_order = np.asarray(neuron_order, dtype=int)
    return signal[:, neuron_order]


def build_shifted_signal(signal, neuron_order=None, max_neurons=120, trace_scale=0.9):
    signal = ensure_2d(signal)

    if neuron_order is not None:
        signal = reorder_signal(signal, neuron_order)

    n_time, n_neurons = signal.shape
    n_plot = min(n_neurons, max_neurons)
    signal_plot = signal[:, :n_plot]

    signal_norm = normalize_0_1_per_neuron(signal_plot)
    signal_scaled = signal_norm * trace_scale
    offsets = np.arange(n_plot).reshape(1, -1)
    signal_shifted = signal_scaled + offsets
    return signal_shifted, n_plot


def get_trial_change_positions(trial_id, every=1):
    """
    Return indices where trial_id changes, subsampled every `every` transitions.
    """
    if trial_id is None:
        return np.array([], dtype=int)

    trial_id = ensure_1d(trial_id)
    change_idx = np.where(np.diff(trial_id) != 0)[0] + 1

    if len(change_idx) == 0:
        return change_idx

    return change_idx[::every]


# =============================================================================
# Plotting
# =============================================================================

def plot_behavior_figure(row_idx, row, time, pos_1d, mov_dir, speed, trial_id, inner_time):
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    fig.subplots_adjust(hspace=0.25)

    title = (
        f"Behavior | row={row_idx} | dataset={row['dataset_name']} | "
        f"mouse={row['mouse_name']} | session={row['session_name']} | "
        f"depth={row['depth_type']}"
    )
    fig.suptitle(title, fontsize=13)

    axes[0].plot(time, pos_1d, linewidth=1)
    axes[0].set_ylabel("pos")
    axes[0].set_title("Position vs Time")

    axes[1].plot(time, mov_dir, linewidth=1)
    axes[1].set_ylabel("mov_dir")
    axes[1].set_title("Movement Direction vs Time")

    axes[2].plot(time, speed, linewidth=1)
    axes[2].set_ylabel("speed")
    axes[2].set_title("Speed vs Time")

    if trial_id is not None:
        axes[3].plot(time, trial_id, linewidth=1)
        axes[3].set_ylabel("trial_id")
        axes[3].set_title("Trial ID vs Time")
    else:
        axes[3].text(0.5, 0.5, "trial_id not available", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_ylabel("trial_id")
        axes[3].set_title("Trial ID vs Time")

    if inner_time is not None:
        axes[4].plot(time, inner_time, linewidth=1)
        axes[4].set_ylabel("inner_time")
        axes[4].set_title("Inner Time vs Time")
    else:
        axes[4].text(0.5, 0.5, "inner_time not available", ha="center", va="center", transform=axes[4].transAxes)
        axes[4].set_ylabel("inner_time")
        axes[4].set_title("Inner Time vs Time")

    axes[4].set_xlabel("time")
    return fig


def plot_traces_figure(row_idx, row, time, signal, trial_change_positions, neuron_order):
    signal_shifted, n_plot = build_shifted_signal(
        signal,
        neuron_order=neuron_order,
        max_neurons=MAX_NEURONS_TO_PLOT,
        trace_scale=0.95,
    )

    fig, ax = plt.subplots(1, 1, figsize=(18, 14))

    title = (
        f"Neural Traces | row={row_idx} | dataset={row['dataset_name']} | "
        f"mouse={row['mouse_name']} | session={row['session_name']} | "
        f"depth={row['depth_type']} | order={NEURON_ORDER_MODE}"
    )
    fig.suptitle(title, fontsize=13)

    for neuron_idx in range(n_plot):
        ax.plot(time, signal_shifted[:, neuron_idx], linewidth=0.7)

    for idx in trial_change_positions:
        if 0 <= idx < len(time):
            ax.axvline(time[idx], color="black", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("time")
    ax.set_ylabel("neuron index")
    ax.set_title(f"Signal vs Time (normalized 0-1, shifted +1 per neuron, first {n_plot} neurons)")
    ax.set_ylim(-0.5, n_plot + 0.5)

    return fig


def plot_raster_figure(row_idx, row, signal, trial_change_positions, neuron_order):
    signal = ensure_2d(signal)

    if neuron_order is not None:
        signal = reorder_signal(signal, neuron_order)

    signal_norm = normalize_0_1_per_neuron(signal).T  # neurons x time

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))

    title = (
        f"Neural Raster/Heatmap | row={row_idx} | dataset={row['dataset_name']} | "
        f"mouse={row['mouse_name']} | session={row['session_name']} | "
        f"depth={row['depth_type']} | order={NEURON_ORDER_MODE}"
    )
    fig.suptitle(title, fontsize=13)

    im = ax.imshow(
        signal_norm,
        aspect="auto",
        cmap="gray_r",
        interpolation="nearest",
        origin="lower",
    )

    for idx in trial_change_positions:
        ax.axvline(idx, color="red", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("time index")
    ax.set_ylabel("neuron index")
    ax.set_title("Normalized neural activity heatmap")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("normalized activity")

    return fig


def plot_one_entry(row_idx, row):
    standardized_file_path = row["standardized_file_path"]

    if not os.path.isfile(standardized_file_path):
        print(f"[WARNING] Missing file for row {row_idx}: {standardized_file_path}")
        return False

    data = load_pickle_standard(standardized_file_path)

    required_keys = ["signal", "pos", "mov_dir", "speed", "time"]
    for key in required_keys:
        if key not in data:
            print(f"[WARNING] Row {row_idx}: missing key '{key}' in {standardized_file_path}")
            return False

    signal = ensure_2d(data["signal"])
    pos_1d = ensure_1d(get_pos_1d(data["pos"]))
    mov_dir = ensure_1d(data["mov_dir"])
    speed = ensure_1d(data["speed"])
    time = ensure_1d(data["time"])

    trial_id = ensure_1d(data["trial_id"]) if data.get("trial_id", None) is not None else None
    inner_time = ensure_1d(data["inner_time"]) if data.get("inner_time", None) is not None else None

    n_time = signal.shape[0]

    def match_len(x, name):
        if x is None:
            return None
        if len(x) != n_time:
            raise ValueError(
                f"Row {row_idx}: length mismatch for {name}. "
                f"Expected {n_time}, got {len(x)}"
            )
        return x

    pos_1d = match_len(pos_1d, "pos")
    mov_dir = match_len(mov_dir, "mov_dir")
    speed = match_len(speed, "speed")
    time = match_len(time, "time")
    trial_id = match_len(trial_id, "trial_id") if trial_id is not None else None
    inner_time = match_len(inner_time, "inner_time") if inner_time is not None else None

    neuron_order = compute_neuron_order(signal, mode=NEURON_ORDER_MODE)
    trial_change_positions = get_trial_change_positions(trial_id, every=TRIAL_LINE_EVERY)

    out_base = base_name_from_row(row_idx, row)

    fig_beh = plot_behavior_figure(row_idx, row, time, pos_1d, mov_dir, speed, trial_id, inner_time)
    save_figure_both(fig_beh, out_base + "_behavior")

    fig_traces = plot_traces_figure(row_idx, row, time, signal, trial_change_positions, neuron_order)
    save_figure_both(fig_traces, out_base + "_traces")

    fig_raster = plot_raster_figure(row_idx, row, signal, trial_change_positions, neuron_order)
    save_figure_both(fig_raster, out_base + "_raster")

    return True


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Keep only lt_ephys dataset
    df = df[df["dataset_name"] == "lt_ephys_mice"].copy()
    df = df.reset_index(drop=True)

    print(f"Loaded CSV with {len(df)} lt_ephys_mice rows")
    print(f"Saving PNG to: {PNG_DIR}")
    print(f"Saving SVG to: {SVG_DIR}")
    print(f"Neuron order mode: {NEURON_ORDER_MODE}")
    print(f"Trial line every: {TRIAL_LINE_EVERY}")

    ok_count = 0
    fail_count = 0

    for row_idx, row in df.iterrows():
        try:
            success = plot_one_entry(row_idx, row)
            if success:
                ok_count += 1
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1
            print(f"[ERROR] Row {row_idx}: {e}")

    print("\nDone.")
    print(f"Successful entries: {ok_count}")
    print(f"Failed entries: {fail_count}")


if __name__ == "__main__":
    main()