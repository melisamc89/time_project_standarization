#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

# =============================================================================
# Paths & Settings
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
SOURCE_BASE_DIR = "/home/hippolab/Documents/melisa/time_project_grin_lens"
SOURCE_DATA_DIR = os.path.join(SOURCE_BASE_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "all_neurons_complete_traces_cwt")

PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")

SPEED_TH = 3.0
FS = 20.0
CMAP_CWT = "jet"
CMAP_NEURONS = "magma"  # Heatmap colormap for all neurons

# =============================================================================
# Trace Preprocessing Pipeline
# =============================================================================

def preprocess_traces(traces, sig_filt=5, sig_up=4, sig_down=12, peak_th=0.1):
    traces = np.asarray(traces, dtype=float)
    lp_traces = uniform_filter1d(traces, size=4000, axis=0)
    clean_traces = gaussian_filter1d(traces, sigma=sig_filt, axis=0)

    for cell in range(clean_traces.shape[1]):
        hist_counts, bin_edges = np.histogram(traces[:, cell], bins=100)
        baseline = bin_edges[np.argmax(hist_counts)] + lp_traces[:, cell] - np.min(lp_traces[:, cell])
        clean_traces[:, cell] = clean_traces[:, cell] - baseline
        max_val = np.max(clean_traces[:, cell], axis=0)
        if max_val > 0:
            clean_traces[:, cell] = clean_traces[:, cell] / max_val

    clean_traces[clean_traces < 0] = 0.0
    conv_traces = np.zeros_like(clean_traces)

    gaus = lambda x, sig, amp, vo: amp * np.exp(-((x ** 2) / (2 * sig ** 2))) + vo
    x = np.arange(-5 * sig_down, 5 * sig_down, 1)
    gaus_kernel = gaus(x, sig_down, 1, 0) + gaus(x, sig_up, 1, 0)
    gaus_kernel[5 * sig_down + 1:] = gaus(x, sig_down, 1, 0)[5 * sig_down + 1:]

    for cell in range(clean_traces.shape[1]):
        peak_idx, _ = find_peaks(clean_traces[:, cell], height=peak_th)
        conv_traces[peak_idx, cell] = clean_traces[peak_idx, cell]
        conv_traces[:, cell] = np.convolve(conv_traces[:, cell], gaus_kernel, mode="same")

    return conv_traces


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_cwt_db(cwt_raw, freqs):
    cwt_mat = np.asarray(cwt_raw)
    freqs = np.asarray(freqs).reshape(-1)

    if cwt_mat.ndim == 3:
        cwt_repr = cwt_mat[:, 0, :] if cwt_mat.shape[2] == len(freqs) else cwt_mat[:, :, 0]
    elif cwt_mat.ndim == 2:
        cwt_repr = cwt_mat.T if (cwt_mat.shape[0] == len(freqs) and cwt_mat.shape[1] != len(freqs)) else cwt_mat
    else:
        return None

    if np.iscomplexobj(cwt_repr):
        power = np.abs(cwt_repr) ** 2
    else:
        power = cwt_repr ** 2 if np.all(cwt_repr >= 0) else np.abs(cwt_repr)

    return 10.0 * np.log10(np.maximum(power, 1e-12))


# =============================================================================
# Plotting All Neurons Across Complete Duration
# =============================================================================

def render_full_population_and_cwt(t_axis, traces, cwt_db, freqs, pos, speed, title_text, out_path, is_moving_only=False, valid_mask=None):
    n_neurons = traces.shape[1]

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.2, 1.3, 0.8], hspace=0.22)
    fig.suptitle(f"{title_text} | Population: {n_neurons} Neurons | Duration: {t_axis[-1]:.1f} s", fontsize=12, y=0.96)

    # -------------------------------------------------------------------------
    # Panel 1: Complete Population Heatmap (ALL Neurons)
    # -------------------------------------------------------------------------
    ax_pop = fig.add_subplot(gs[0])

    # Sort neurons by peak activity time across the session for clear visual structure
    peak_times = np.argmax(traces, axis=0)
    sort_order = np.argsort(peak_times)
    traces_sorted = traces[:, sort_order]

    # Normalize each neuron 0 to 1
    norm_traces = (traces_sorted - np.min(traces_sorted, axis=0)) / (np.ptp(traces_sorted, axis=0) + 1e-6)

    im = ax_pop.imshow(
        norm_traces.T,
        aspect="auto",
        cmap=CMAP_NEURONS,
        origin="lower",
        extent=[t_axis[0], t_axis[-1], 0, n_neurons],
        interpolation="nearest"
    )

    # In continuous mode, shade non-moving intervals in gray
    if not is_moving_only and valid_mask is not None:
        ax_pop.fill_between(t_axis, 0, n_neurons, where=~valid_mask, color="gray", alpha=0.35, label=f"Stationary (≤ {SPEED_TH} cm/s)")
        ax_pop.legend(loc="upper right", fontsize=8)

    ax_pop.set_xlim(t_axis[0], t_axis[-1])
    ax_pop.set_ylim(0, n_neurons)
    ax_pop.set_ylabel(f"All Neurons (1 to {n_neurons})", fontsize=10)
    ax_pop.set_title("Neural Population Activity (All Neurons Normalized & Sorted by Peak Time)", fontsize=10, loc="left")
    ax_pop.set_xticklabels([])

    cb_pop = fig.colorbar(im, ax=ax_pop, orientation="vertical", pad=0.012, fraction=0.015)
    cb_pop.set_label("Activity", fontsize=8)

    # -------------------------------------------------------------------------
    # Panel 2: Continuous Wavelet Transform (CWT) Spectrogram
    # -------------------------------------------------------------------------
    ax_cwt = fig.add_subplot(gs[1])
    f_order = np.argsort(freqs)
    freqs_sorted = freqs[f_order]
    cwt_sorted = cwt_db[:, f_order]

    vmin, vmax = np.nanpercentile(cwt_sorted, 5), np.nanpercentile(cwt_sorted, 98)
    mesh = ax_cwt.pcolormesh(
        t_axis,
        freqs_sorted,
        cwt_sorted.T,
        shading="auto",
        cmap=CMAP_CWT,
        norm=Normalize(vmin=vmin, vmax=vmax)
    )
    ax_cwt.set_yscale("log")
    ax_cwt.set_xlim(t_axis[0], t_axis[-1])
    ax_cwt.set_ylabel("Frequency (Hz)", fontsize=10)
    ax_cwt.set_title("LFP Continuous Wavelet Transform (Power in dB)", fontsize=10, loc="left")
    ax_cwt.set_xticklabels([])

    cb_cwt = fig.colorbar(mesh, ax=ax_cwt, orientation="vertical", pad=0.012, fraction=0.015)
    cb_cwt.set_label("dB", fontsize=8)

    # -------------------------------------------------------------------------
    # Panel 3: Position & Speed Behavior
    # -------------------------------------------------------------------------
    ax_beh = fig.add_subplot(gs[2])
    ax_beh.plot(t_axis, pos, color="black", lw=1.2, label="Position")
    ax_beh.set_ylabel("Position (cm)", fontsize=9)
    ax_beh.set_xlim(t_axis[0], t_axis[-1])
    xlabel_text = "Time (seconds)" if not is_moving_only else "Cumulative Moving Time (seconds)"
    ax_beh.set_xlabel(xlabel_text, fontsize=10)
    ax_beh.grid(True, linestyle=":", alpha=0.5)

    ax_spd = ax_beh.twinx()
    ax_spd.plot(t_axis, speed, color="#ff7f0e", lw=0.9, alpha=0.8, linestyle="--", label="Speed")
    if not is_moving_only:
        ax_spd.axhline(SPEED_TH, color="red", linestyle=":", lw=0.9, label=f"Threshold ({SPEED_TH} cm/s)")
    ax_spd.set_ylabel("Speed (cm/s)", color="#ff7f0e", fontsize=9)
    ax_spd.tick_params(axis="y", labelcolor="#ff7f0e", labelsize=8)

    fig.savefig(out_path + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out_path}.png")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)

    mouse_folders = [
        d for d in os.listdir(SOURCE_DATA_DIR)
        if os.path.isdir(os.path.join(SOURCE_DATA_DIR, d)) and not d.startswith(".")
    ]

    print("\n" + "=" * 70)
    print("SESSION POPULATION SUMMARY")
    print("=" * 70)

    records = []
    for mouse_name in sorted(mouse_folders):
        mouse_dir = os.path.join(SOURCE_DATA_DIR, mouse_name)
        mat_candidates = glob.glob(os.path.join(mouse_dir, "*_tracesLFP.mat"))
        if not mat_candidates:
            mat_candidates = glob.glob(os.path.join(mouse_dir, "*.mat"))
        if not mat_candidates:
            continue

        mat_path = mat_candidates[0]
        mat_data = loadmat(mat_path, simplify_cells=True)
        if "tracesLFP" not in mat_data:
            continue

        tlfp = mat_data["tracesLFP"]
        raw_traces = tlfp["rawTraces"]
        n_time, n_cells = raw_traces.shape
        speed = np.asarray(tlfp["speed"]).reshape(-1)
        n_mov = int(np.sum(speed > SPEED_TH))

        print(f"  Mouse: {mouse_name:<18} | Total Neurons: {n_cells:<4} | Total Time: {n_time/FS:6.1f}s | Moving Time: {n_mov/FS:6.1f}s")
        records.append((mouse_name, tlfp, n_cells))

    print("=" * 70 + "\n")

    for mouse_name, tlfp, n_neurons in records:
        print(f"Processing plots for: {mouse_name} (Showing ALL {n_neurons} Neurons, Full Time)")

        raw_traces = tlfp["rawTraces"]
        conv_traces = preprocess_traces(raw_traces)

        pos = np.asarray(tlfp["position"])
        pos_x = pos[:, 0] if pos.ndim > 1 else pos.reshape(-1)
        speed = np.asarray(tlfp["speed"]).reshape(-1)
        n_frames = conv_traces.shape[0]

        # Align CWT to imaging frame count
        lfp_struct = tlfp.get("LFP", {})
        cwt_raw = lfp_struct.get("CWT", None)
        cwt_freqs = lfp_struct.get("CWTfreqs", None)
        lfp_time = lfp_struct.get("time", None)

        if cwt_raw is not None and cwt_raw.shape[0] != n_frames:
            t_target = np.arange(n_frames) / FS
            t_src = np.asarray(lfp_time).reshape(-1) if lfp_time is not None else np.linspace(0, (n_frames - 1) / FS, cwt_raw.shape[0])
            cwt_full = interp1d(t_src, cwt_raw, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")(t_target)
        else:
            cwt_full = cwt_raw

        cwt_db_full = compute_cwt_db(cwt_full, cwt_freqs)
        time_full = np.arange(n_frames) / FS
        valid_mask = speed > SPEED_TH

        # ---------------------------------------------------------------------
        # Plot 1: Complete Continuous Session (All Periods, Full Time)
        # ---------------------------------------------------------------------
        out_cont = os.path.join(PNG_DIR, f"{mouse_name}_all_neurons_continuous_full_time")
        render_full_population_and_cwt(
            t_axis=time_full,
            traces=conv_traces,
            cwt_db=cwt_db_full,
            freqs=cwt_freqs,
            pos=pos_x,
            speed=speed,
            title_text=f"{mouse_name} [Continuous Full Session]",
            out_path=out_cont,
            is_moving_only=False,
            valid_mask=valid_mask,
        )

        # ---------------------------------------------------------------------
        # Plot 2: Moving Periods Only (Shortened True Moving Signal)
        # ---------------------------------------------------------------------
        moving_idx = np.where(valid_mask)[0]
        n_mov_frames = len(moving_idx)
        time_moving = np.arange(n_mov_frames) / FS  # Continuous contracted moving timeline

        traces_moving = conv_traces[moving_idx, :]
        cwt_moving = cwt_db_full[moving_idx, :]
        pos_moving = pos_x[moving_idx]
        speed_moving = speed[moving_idx]

        out_mov = os.path.join(PNG_DIR, f"{mouse_name}_all_neurons_moving_periods_shortened")
        render_full_population_and_cwt(
            t_axis=time_moving,
            traces=traces_moving,
            cwt_db=cwt_moving,
            freqs=cwt_freqs,
            pos=pos_moving,
            speed=speed_moving,
            title_text=f"{mouse_name} [Moving Periods Only: > {SPEED_TH} cm/s]",
            out_path=out_mov,
            is_moving_only=True,
        )

    print("\n[DONE] All plots generated successfully.")


if __name__ == "__main__":
    main()