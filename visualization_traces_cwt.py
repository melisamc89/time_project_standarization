#!/usr/init/env python3
# -*- coding: utf-8 -*-

"""
Full-Rate LFP Processing & Trial Zoom Visualization Pipeline:
- Loads neural traces and full-rate LFP directly from ingestion sources.
- Filters all LFP frequency bands (Slow, Delta, Theta, Gamma, Ripple, MUA) at the
  native full sampling rate (fs_lfp) to accurately capture high-frequency oscillations.
- Plots neural heatmap in black-to-white (gray_r, black = max intensity).
- Generates complete session overview and automated single-trial movement zoom-ins.
"""

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.io import loadmat
from scipy.signal import find_peaks, butter, sosfiltfilt, welch, fftconvolve
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.integrate import trapezoid

# =============================================================================
# Paths & Settings
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
SOURCE_BASE_DIR = "/home/hippolab/Documents/melisa/time_project_grin_lens"
SOURCE_PICKLE_FILE = os.path.join(SOURCE_BASE_DIR, "data", "lt_egrin_data.pkl")
SOURCE_MAT_DIR = os.path.join(SOURCE_BASE_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "full_rate_lfp_visualizations")

PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")

SPEED_TH = 3.0
FS_IMAGING = 20.0
SIG_FILT = 6.0
CMAP_CWT = "jet"
CMAP_NEURONS = "gray_r"  # Black = max intensity

LFP_BANDS = {
    "slow": (0.01, 0.10),
    "delta": (1.00, 3.00),
    "theta": (8.00, 12.00),
    "gamma": (40.00, 90.00),
    "ripple": (100.0, 250.0),
    "mua": (300.0, 1000.0)
}

CWT_FMIN = 0.1
CWT_FMAX = 1000.0
CWT_NUM_FREQS = 90
CWT_W0 = 6.0


# =============================================================================
# Helper & Processing Functions
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def preprocess_traces(traces, sig_filt=6.0, peak_th=0.1):
    traces = np.asarray(traces, dtype=float)
    clean_traces = gaussian_filter1d(traces, sigma=sig_filt, axis=0)
    for cell in range(clean_traces.shape[1]):
        max_val = np.max(clean_traces[:, cell], axis=0)
        if max_val > 0:
            clean_traces[:, cell] /= max_val
    clean_traces[clean_traces < 0] = 0.0
    return clean_traces


def select_maximal_theta_channel(lfp_raw, fs, speed=None, speed_th=3.0):
    n_channels = lfp_raw.shape[1]
    theta_powers = np.zeros(n_channels, dtype=float)
    eval_matrix = lfp_raw[speed > speed_th, :] if (
                speed is not None and np.sum(speed > speed_th) > int(5 * fs)) else lfp_raw
    nperseg = int(min(eval_matrix.shape[0], round(2.0 * fs)))

    for ch in range(n_channels):
        freqs, psd = welch(eval_matrix[:, ch], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, scaling="density")
        theta_mask = (freqs >= 8.0) & (freqs <= min(12.0, fs / 2.0 - 0.5))
        if np.sum(theta_mask) == 0:
            theta_mask = (freqs >= 1.0) & (freqs <= min(3.0, fs / 2.0 - 0.5))
        theta_powers[ch] = trapezoid(psd[theta_mask], freqs[theta_mask])

    return int(np.argmax(theta_powers))


def compute_full_cwt_full_rate(signal_1d, fs):
    n_pts = len(signal_1d)
    total_sec = n_pts / fs
    fmax_safe = min(CWT_FMAX, fs / 2.0 - 1.0)
    freqs = np.geomspace(CWT_FMIN, fmax_safe, CWT_NUM_FREQS)
    cwt_power = np.zeros((n_pts, CWT_NUM_FREQS), dtype=np.float32)

    for i, f0 in enumerate(freqs):
        s = CWT_W0 / (2.0 * np.pi * f0)
        t_max = min(4.0 * s, max(4.0, total_sec * 0.2))
        t_kern = np.arange(-t_max, t_max, 1.0 / fs)
        if len(t_kern) == 0:
            continue
        wavelet = (
                np.pi ** (-0.25)
                * np.exp(1j * 2.0 * np.pi * f0 * t_kern)
                * np.exp(-0.5 * (t_kern / s) ** 2)
        )
        wavelet /= np.sqrt(s * fs)
        conv = fftconvolve(signal_1d, wavelet, mode="same")
        cwt_power[:, i] = (np.abs(conv) ** 2).astype(np.float32)

    return cwt_power, freqs


def extract_bands_at_full_rate(lfp_chan, fs_lfp):
    nyquist = fs_lfp / 2.0
    bands_full = {}
    for label, (low, high) in LFP_BANDS.items():
        high_safe = min(high, nyquist - 1.0)
        low_safe = max(0.0005, low)
        if low_safe >= high_safe:
            bands_full[label] = np.zeros_like(lfp_chan)
            continue
        order = 2 if low_safe < 0.1 else 3
        sos = butter(order, [low_safe, high_safe], btype="bandpass", fs=fs_lfp, output="sos")
        bands_full[label] = sosfiltfilt(sos, lfp_chan)
    return bands_full


def compute_cwt_db(cwt_raw, freqs):
    cwt_mat = np.asarray(cwt_raw)
    freqs = np.asarray(freqs).reshape(-1)
    if cwt_mat.ndim == 3:
        cwt_repr = cwt_mat[:, 0, :] if cwt_mat.shape[2] == len(freqs) else cwt_mat[:, :, 0]
    elif cwt_mat.ndim == 2:
        cwt_repr = cwt_mat.T if (cwt_mat.shape[0] == len(freqs) and cwt_mat.shape[1] != len(freqs)) else cwt_mat
    else:
        return None
    power = np.abs(cwt_repr) ** 2 if np.iscomplexobj(cwt_repr) else cwt_repr ** 2
    return 10.0 * np.log10(np.maximum(power, 1e-12))


# =============================================================================
# Plotting Function
# =============================================================================

def render_multi_panel_plot(t_imaging, traces, cwt_db_img, freqs, raw_lfp_img, band_traces_img, pos, speed, title_text,
                            out_path, zoom_range=None, valid_mask=None):
    n_neurons = traces.shape[1]
    band_keys = list(LFP_BANDS.keys())

    n_panels = 2 + len(band_keys) + 1 + 1
    height_ratios = [2.2, 1.0] + [0.9] * len(band_keys) + [1.5, 1.0]

    fig = plt.figure(figsize=(16, 2.2 + 1.2 * n_panels))
    gs = gridspec.GridSpec(n_panels, 1, height_ratios=height_ratios, hspace=0.15)
    fig.suptitle(f"{title_text} | Population: {n_neurons} Neurons", fontsize=13, y=0.96, fontweight="bold")

    panel_idx = 0

    # 1. Neural Population Heatmap (Black = Max Intensity)
    ax_pop = fig.add_subplot(gs[panel_idx])
    peak_times = np.argmax(traces, axis=0)
    sort_order = np.argsort(peak_times)
    traces_sorted = traces[:, sort_order]
    norm_traces = (traces_sorted - np.min(traces_sorted, axis=0)) / (np.ptp(traces_sorted, axis=0) + 1e-6)

    ax_pop.imshow(
        norm_traces.T,
        aspect="auto",
        cmap=CMAP_NEURONS,
        origin="lower",
        extent=[t_imaging[0], t_imaging[-1], 0, n_neurons],
        interpolation="nearest"
    )
    if zoom_range is None and valid_mask is not None:
        ax_pop.fill_between(t_imaging, 0, n_neurons, where=~valid_mask, color="gray", alpha=0.3,
                            label=f"Stationary (≤ {SPEED_TH} cm/s)")
        ax_pop.legend(loc="upper right", fontsize=8)

    ax_pop.set_ylabel(f"Neurons\n(1-{n_neurons})", fontsize=8, fontweight="bold")
    ax_pop.set_title("Neural Population Activity (Black = Max Intensity)", fontsize=9, loc="left", fontweight="bold")
    ax_pop.tick_params(labelbottom=False)
    panel_idx += 1

    # 2. Raw LFP Trace
    ax_lfp = fig.add_subplot(gs[panel_idx], sharex=ax_pop)
    ax_lfp.plot(t_imaging, raw_lfp_img, color="black", lw=0.8)
    ax_lfp.set_ylabel("Raw LFP", fontsize=8, fontweight="bold")
    ax_lfp.grid(True, linestyle=":", alpha=0.5)
    ax_lfp.tick_params(labelbottom=False)
    panel_idx += 1

    # 3. Frequency Bands (Slow, Delta, Theta, Gamma, Ripple, MUA)
    for b_name in band_keys:
        ax_b = fig.add_subplot(gs[panel_idx], sharex=ax_pop)
        ax_b.plot(t_imaging, band_traces_img[b_name], color="#2b5c8f", lw=0.8)
        ax_b.set_ylabel(f"{b_name.capitalize()}", fontsize=8, fontweight="bold")
        ax_b.grid(True, linestyle=":", alpha=0.5)
        ax_b.tick_params(labelbottom=False)
        panel_idx += 1

    # 4. CWT Spectrogram
    ax_cwt = fig.add_subplot(gs[panel_idx], sharex=ax_pop)
    f_order = np.argsort(freqs)
    freqs_sorted = freqs[f_order]
    cwt_sorted = cwt_db_img[:, f_order]

    vmin, vmax = np.nanpercentile(cwt_sorted, 5), np.nanpercentile(cwt_sorted, 98)
    ax_cwt.pcolormesh(
        t_imaging,
        freqs_sorted,
        cwt_sorted.T,
        shading="auto",
        cmap=CMAP_CWT,
        norm=Normalize(vmin=vmin, vmax=vmax)
    )
    ax_cwt.set_yscale("log")
    ax_cwt.set_ylim(max(0.1, freqs_sorted[0]), min(1000.0, freqs_sorted[-1]))
    ax_cwt.set_ylabel("CWT Freq\n(Hz, Log)", fontsize=8, fontweight="bold")
    ax_cwt.tick_params(labelbottom=False)
    panel_idx += 1

    # 5. Position & Speed Behavior
    ax_beh = fig.add_subplot(gs[panel_idx], sharex=ax_pop)
    ax_beh.plot(t_imaging, pos, color="black", lw=1.1, label="Position")
    ax_beh.set_ylabel("Position\n(cm)", fontsize=8, fontweight="bold")
    ax_beh.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    ax_beh.grid(True, linestyle=":", alpha=0.5)

    ax_spd = ax_beh.twinx()
    ax_spd.plot(t_imaging, speed, color="#ff7f0e", lw=0.8, alpha=0.85, linestyle="--", label="Speed")
    if zoom_range is None:
        ax_spd.axhline(SPEED_TH, color="red", linestyle=":", lw=0.9, label=f"Threshold ({SPEED_TH})")
    ax_spd.set_ylabel("Speed", color="#ff7f0e", fontsize=8, fontweight="bold")
    ax_spd.tick_params(axis="y", labelcolor="#ff7f0e", labelsize=8)

    if zoom_range is not None:
        ax_pop.set_xlim(zoom_range[0], zoom_range[1])

    fig.savefig(out_path + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out_path}.png")


# =============================================================================
# Main Execution Pipeline
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)

    if not os.path.isfile(SOURCE_PICKLE_FILE):
        print(f"[ERROR] Source pickle file not found: {SOURCE_PICKLE_FILE}")
        return

    with open(SOURCE_PICKLE_FILE, "rb") as f:
        data = pickle.load(f)

    print(f"\nLoaded source pickle with {len(data)} mice: {sorted(data.keys())}")

    for mouse_name in sorted(data.keys()):
        mouse_pickle_data = data[mouse_name]
        raw_traces = np.asarray(mouse_pickle_data["clean_traces"], dtype=float)
        n_frames = raw_traces.shape[0]
        speed = np.asarray(mouse_pickle_data["speed"]).reshape(-1)
        position = np.asarray(mouse_pickle_data["position"])
        pos_x = position[:, 0] if position.ndim > 1 else position.reshape(-1)
        t_imaging = np.arange(n_frames) / FS_IMAGING

        # Locate .mat file
        mouse_dir = os.path.join(SOURCE_MAT_DIR, mouse_name)
        mat_candidates = glob.glob(os.path.join(mouse_dir, "*_tracesLFP.mat"))
        if not mat_candidates:
            mat_candidates = glob.glob(os.path.join(mouse_dir, "*.mat"))
        if not mat_candidates:
            continue

        mat_data = loadmat(mat_candidates[0], simplify_cells=True)
        tlfp = mat_data["tracesLFP"]
        raw_lfp_matrix = np.asarray(tlfp["LFP"]["LFP"], dtype=float)
        if raw_lfp_matrix.shape[0] == 16 and raw_lfp_matrix.shape[1] > 16:
            raw_lfp_matrix = raw_lfp_matrix.T

        lfp_time = tlfp["LFP"].get("time", None)
        if lfp_time is not None and len(lfp_time) > 1:
            t_lfp = np.asarray(lfp_time).reshape(-1)
            fs_lfp = 1.0 / np.nanmedian(np.diff(t_lfp))
        else:
            fs_lfp = float(raw_lfp_matrix.shape[0]) / (float(n_frames) / FS_IMAGING)
            t_lfp = np.linspace(0, (n_frames - 1) / FS_IMAGING, raw_lfp_matrix.shape[0])

        # Select best theta channel using behavior speed interpolated to LFP timeline
        speed_on_lfp = interp1d(t_imaging, speed, bounds_error=False, fill_value="extrapolate")(t_lfp)
        best_ch = select_maximal_theta_channel(raw_lfp_matrix, fs_lfp, speed=speed_on_lfp)
        best_lfp_raw = raw_lfp_matrix[:, best_ch]

        print(f"Processing {mouse_name} | Best Theta Ch: {best_ch + 1} | Full-rate LFP fs: {fs_lfp:.1f} Hz")

        # 1. Compute CWT and Frequency Bands at FULL NATIVE RATE
        cwt_full_rate, cwt_freqs = compute_full_cwt_full_rate(best_lfp_raw, fs_lfp)
        bands_full_rate = extract_bands_at_full_rate(best_lfp_raw, fs_lfp)

        # 2. Downsample CWT, Bands, and Raw LFP to Imaging Frame Grid (20Hz) for Aligned Plotting
        cwt_db_full_rate = compute_cwt_db(cwt_full_rate, cwt_freqs)

        cwt_db_img = interp1d(t_lfp, cwt_db_full_rate, axis=0, bounds_error=False, fill_value="extrapolate")(t_imaging)
        raw_lfp_img = interp1d(t_lfp, best_lfp_raw, bounds_error=False, fill_value="extrapolate")(t_imaging)

        band_traces_img = {}
        for b_name, b_sig in bands_full_rate.items():
            band_traces_img[b_name] = interp1d(t_lfp, b_sig, bounds_error=False, fill_value="extrapolate")(t_imaging)

        conv_traces = preprocess_traces(raw_traces, sig_filt=SIG_FILT)
        valid_mask = speed > SPEED_TH

        # 3. Complete Session Overview Plot
        out_full = os.path.join(PNG_DIR, f"{mouse_name}_complete_full_rate_lfp")
        render_multi_panel_plot(
            t_imaging=t_imaging,
            traces=conv_traces,
            cwt_db_img=cwt_db_img,
            freqs=cwt_freqs,
            raw_lfp_img=raw_lfp_img,
            band_traces_img=band_traces_img,
            pos=pos_x,
            speed=speed,
            title_text=f"{mouse_name} [Full-Rate LFP & Bands Overview]",
            out_path=out_full,
            valid_mask=valid_mask
        )

        # 4. Automated Single-Trial Zoom-In Plot (Isolating a prominent running trial)
        moving_indices = np.where(speed > SPEED_TH)[0]
        if len(moving_indices) > 50:
            diffs = np.diff(moving_indices)
            split_points = np.where(diffs > 1)[0]
            blocks = np.split(moving_indices, split_points + 1)
            longest_block = max(blocks, key=len)

            t_start = max(0, t_imaging[longest_block[0]] - 1.5)
            t_end = min(t_imaging[-1], t_imaging[longest_block[-1]] + 1.5)

            out_zoom = os.path.join(PNG_DIR, f"{mouse_name}_zoom_single_trial")
            render_multi_panel_plot(
                t_imaging=t_imaging,
                traces=conv_traces,
                cwt_db_img=cwt_db_img,
                freqs=cwt_freqs,
                raw_lfp_img=raw_lfp_img,
                band_traces_img=band_traces_img,
                pos=pos_x,
                speed=speed,
                title_text=f"{mouse_name} [Zoom-In: Complete Movement Trial]",
                out_path=out_zoom,
                zoom_range=(t_start, t_end)
            )

    print("\n[SUCCESS] All full-rate LFP visualizations and trial zoom-ins completed successfully.")


if __name__ == "__main__":
    main()