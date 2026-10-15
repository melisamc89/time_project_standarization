#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-Stage LFP Artifact Cleaning with Full-Session High-Pass Extraction
and Quick-Return Recombination (eliminates the long artificial zero-amplitude ramp).
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import butter, sosfiltfilt, find_peaks, hilbert, fftconvolve

# =============================================================================
# Paths & Settings
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
SOURCE_BASE_DIR = "/home/hippolab/Documents/melisa/time_project_grin_lens"
SOURCE_DATA_DIR = os.path.join(SOURCE_BASE_DIR, "data")
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "analysis", "two_stage_full_hf_cleaning")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

BEH_FS = 20.0
REP_CHANNEL_IDX = 7  # Channel 8 (0-indexed)

# Stage 1: Noise filter settings
SPIKE_SLEW_Z = 12.0
CORE_BLANK_MS = 0.040

# Stage 2: Slow component filter settings
STD_THRESH = 2.0  # Trigger on >= 2 STD deviation
SLOW_BAND_HZ = (0.2, 2.5)  # Capacitive sag band
FAST_SPLIT_HZ = 4.0  # Preserves genuine neural rhythms (>4 Hz)
MIN_EVENT_SEP_SEC = 2.5  # Separation between distinct knock events

# Window durations
EVENT_PLOT_PRE_SEC = 2.0
EVENT_PLOT_POST_SEC = 5.0
MAX_SETTLE_TRACE_SEC = 4.5
CONTEXT_SEC = 0.30


# =============================================================================
# 1. Fast CWT Spectrogram Helper
# =============================================================================

def compute_fast_cwt(signal_1d, fs_orig, f_min=0.08, f_max=100.0, num_freqs=50, w0=6.0):
    ds = 5
    sig_ds = signal_1d[::ds]
    fs_ds = fs_orig / ds
    freqs = np.geomspace(f_min, f_max, num_freqs)
    power_db = np.zeros((len(sig_ds), num_freqs), dtype=float)

    for i, f0 in enumerate(freqs):
        s = w0 / (2.0 * np.pi * f0)
        t_max = min(4.0 * s, 8.0)
        t_kern = np.arange(-t_max, t_max, 1.0 / fs_ds)
        wavelet = np.pi ** (-0.25) * np.exp(1j * 2.0 * np.pi * f0 * t_kern) * np.exp(-0.5 * (t_kern / s) ** 2)
        wavelet /= np.sqrt(s * fs_ds)
        conv = fftconvolve(sig_ds, wavelet, mode="same")
        power_db[:, i] = 10.0 * np.log10(np.maximum(np.abs(conv) ** 2, 1e-12))

    t_ds = np.linspace(0, len(signal_1d) / fs_orig, len(sig_ds))
    return power_db, freqs, t_ds


# =============================================================================
# 2. Stage 1: Acute Noise / Slew-Rate Spike Filter
# =============================================================================

def stage1_noise_filter(raw_lfp, fs):
    n_samples, n_channels = raw_lfp.shape
    lfp_stage1 = raw_lfp.copy()

    ch_med = np.median(raw_lfp, axis=0)
    centered = raw_lfp - ch_med
    dLFP = np.diff(centered, axis=0, prepend=centered[:1, :])
    d_mad = np.median(np.abs(dLFP), axis=0) * 1.4826
    d_mad[d_mad == 0] = 1e-6
    z_slew = np.abs(dLFP) / d_mad

    score_slew = np.percentile(z_slew, 80, axis=1)
    min_spike_sep = int(round(0.05 * fs))
    spike_peaks, _ = find_peaks(score_slew, distance=min_spike_sep, height=SPIKE_SLEW_Z)

    bridge_half = int(round(CORE_BLANK_MS * fs * 0.5))
    for sp in spike_peaks:
        c0 = max(0, sp - bridge_half)
        c1 = min(n_samples, sp + bridge_half)
        if c0 > 1 and c1 < n_samples - 1:
            t_core = np.arange(c0, c1)
            t_ref = np.array([c0 - 1, c1])
            for ch in range(n_channels):
                f_lin = interp1d(t_ref, lfp_stage1[t_ref, ch], fill_value="extrapolate")
                lfp_stage1[t_core, ch] = f_lin(t_core)

    return lfp_stage1


# =============================================================================
# 3. Stage 2: Clean Baseline via PCHIP + Full-Strength >4 Hz Oscillations
# =============================================================================

def stage2_slow_component_filter(lfp_stage1, fs):
    n_samples, n_channels = lfp_stage1.shape
    lfp_stage2 = lfp_stage1.copy()

    # 1. Multi-channel deviation from session mean
    session_mean = np.mean(lfp_stage1, axis=0)
    session_std = np.std(lfp_stage1, axis=0)
    session_std[session_std == 0] = 1e-6
    z_dev = np.abs(lfp_stage1 - session_mean) / session_std
    max_ch_dev = np.percentile(z_dev, 80, axis=1)

    # 2. Slow-band power envelope
    sos_slow = butter(2, SLOW_BAND_HZ, btype="bandpass", fs=fs, output="sos")
    lfp_slow = sosfiltfilt(sos_slow, lfp_stage1, axis=0)
    envelope_slow = np.abs(hilbert(lfp_slow, axis=0))
    cross_ch_slow_power = np.mean(envelope_slow, axis=1)
    median_slow_power = np.median(cross_ch_slow_power)

    # Find prominent peaks
    min_sep_pts = int(round(MIN_EVENT_SEP_SEC * fs))
    candidate_peaks, _ = find_peaks(max_ch_dev, distance=min_sep_pts, height=STD_THRESH)

    validated_peaks = []
    check_win = int(round(0.8 * fs))
    for p in candidate_peaks:
        w0 = max(0, p - check_win)
        w1 = min(n_samples, p + check_win)
        if np.mean(cross_ch_slow_power[w0:w1]) > median_slow_power:
            validated_peaks.append(p)

    # 3. Pre-extract >4 Hz oscillations across the ENTIRE continuous session (zero filter transients)
    sos_fast = butter(3, FAST_SPLIT_HZ, btype="highpass", fs=fs, output="sos")
    hf_matrix = sosfiltfilt(sos_fast, lfp_stage1, axis=0)

    cleaned_intervals = []
    event_metadata = []

    pre_pts = int(round(EVENT_PLOT_PRE_SEC * fs))
    post_pts = int(round(EVENT_PLOT_POST_SEC * fs))
    ctx = int(round(CONTEXT_SEC * fs))
    max_settle_pts = int(round(MAX_SETTLE_TRACE_SEC * fs))
    acute_spike_pts = int(round(0.040 * fs))  # 40 ms acute clipping core
    taper_pts = int(round(0.020 * fs))  # 20 ms quick transition

    for peak_idx in validated_peaks:
        win_start = max(0, peak_idx - pre_pts)
        win_end = min(n_samples, peak_idx + post_pts)
        rel_peak = peak_idx - win_start

        raw_seg = lfp_stage1[win_start:win_end, :].copy()
        cleaned_seg = raw_seg.copy()
        n_seg = len(raw_seg)

        # True local deflection peak
        search_r = int(round(0.40 * fs))
        local_start = max(0, rel_peak - search_r)
        local_end = min(n_seg, rel_peak + search_r)
        ch_base = np.median(raw_seg, axis=0)
        cross_mean = np.mean(raw_seg[local_start:local_end, :], axis=1)
        dev = np.abs(cross_mean - np.mean(ch_base))
        true_peak_rel = local_start + np.argmax(dev)

        # Baseline noise scale derived from pre-event clean period
        pre_event_clean = raw_seg[:max(10, true_peak_rel - int(0.15 * fs)), :]
        if len(pre_event_clean) > 10:
            target_baseline = np.median(pre_event_clean, axis=0)
            baseline_noise = np.median(np.abs(pre_event_clean - target_baseline), axis=0) * 1.4826
        else:
            target_baseline = ch_base
            baseline_noise = np.median(np.abs(raw_seg - target_baseline), axis=0) * 1.4826
        baseline_noise[baseline_noise == 0] = 1e-6

        mean_dev_to_base = np.mean(np.abs(raw_seg - target_baseline) / baseline_noise, axis=1)

        # Backward boundary
        t0 = true_peak_rel
        b_limit = max(0, true_peak_rel - int(round(0.35 * fs)))
        while t0 > b_limit and mean_dev_to_base[t0] > 1.5:
            t0 -= 1
        t0 = max(0, t0 - int(round(0.04 * fs)))

        # Forward boundary (traces the long 3-5s recovery tail)
        min_forward_pts = int(round(2.0 * fs))
        t1 = true_peak_rel + min_forward_pts
        f_limit = min(n_seg, true_peak_rel + max_settle_pts)
        while t1 < f_limit and mean_dev_to_base[t1] > 1.4:
            t1 += 1
        t1 = min(n_seg, t1 + int(round(0.08 * fs)))

        # Acute core where the amplifier clipped
        c_left = max(0, true_peak_rel - int(acute_spike_pts * 0.3))
        c_right = min(n_seg, true_peak_rel + int(acute_spike_pts * 0.7))

        # Fit shape-preserving PCHIP for the slow baseline across [t0, t1]
        idx_pre = np.arange(max(0, t0 - ctx), t0)
        idx_post = np.arange(t1, min(n_seg, t1 + ctx))

        if len(idx_pre) >= 3 and len(idx_post) >= 3:
            fit_idx = np.concatenate([idx_pre, idx_post])
            target_idx = np.arange(t0, t1)

            # Fast, smooth weight: ONLY blanks the 40ms spike core, then is 1.0 immediately!
            weight = np.ones(len(target_idx), dtype=float)

            # Map core indices relative to target_idx
            mask_core = (target_idx >= c_left) & (target_idx <= c_right)
            weight[mask_core] = 0.0

            # Quick 20 ms transition on the edges of the core to avoid step clicks
            mask_pre_taper = (target_idx >= (c_left - taper_pts)) & (target_idx < c_left)
            if np.any(mask_pre_taper):
                weight[mask_pre_taper] = np.linspace(1.0, 0.0, np.sum(mask_pre_taper))

            mask_post_taper = (target_idx > c_right) & (target_idx <= (c_right + taper_pts))
            if np.any(mask_post_taper):
                weight[mask_post_taper] = np.linspace(0.0, 1.0, np.sum(mask_post_taper))

            for c in range(n_channels):
                pchip = PchipInterpolator(fit_idx, raw_seg[fit_idx, c])
                clean_slow_baseline = pchip(target_idx)
                # True physiological oscillations running at full amplitude across the entire recovery!
                cleaned_seg[target_idx, c] = clean_slow_baseline + (hf_matrix[win_start + target_idx, c] * weight)

        lfp_stage2[win_start:win_end, :] = cleaned_seg
        cleaned_intervals.append((win_start + t0, win_start + t1))

        event_metadata.append({
            "win_start": win_start,
            "win_end": win_end,
            "raw_seg": raw_seg,
            "cleaned_seg": cleaned_seg,
            "t0_rel": t0,
            "t1_rel": t1,
            "true_peak_rel": true_peak_rel,
        })

    return lfp_stage2, cleaned_intervals, event_metadata


# =============================================================================
# 4. Diagnostic Event Plotter
# =============================================================================

def plot_single_event(t_seg, raw_seg, cleaned_seg, t0_rel, t1_rel, true_peak_rel, fs, ev_num, out_dir, mouse_name):
    t_true_peak = t_seg[true_peak_rel]
    cwt_raw, freqs, t_ds = compute_fast_cwt(raw_seg[:, REP_CHANNEL_IDX], fs)
    cwt_clean, _, _ = compute_fast_cwt(cleaned_seg[:, REP_CHANNEL_IDX], fs)
    t_ds_plot = t_seg[0] + t_ds

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.8, 1.8, 1.0, 1.0], hspace=0.32)

    # Panel 1: Stage 1 Raw LFP
    ax1 = fig.add_subplot(gs[0])
    for c in range(16):
        ax1.plot(t_seg, raw_seg[:, c] - np.median(raw_seg[:, c]) + c * 400, lw=0.7)
    ax1.axvline(t_true_peak, color="red", linestyle="--", alpha=0.7, label=f"Artifact Peak (t={t_true_peak:.3f} s)")
    ax1.axvspan(t_seg[t0_rel], t_seg[t1_rel], color="#d62728", alpha=0.15, label="Extended PCHIP Bridge Zone")
    ax1.set_title(f"Event #{ev_num + 1:02d} at t = {t_true_peak:.3f} s | Stage 1 LFP ({mouse_name})", fontsize=11)
    ax1.set_ylabel("Ch 1 - 16", fontsize=9)
    ax1.set_xlim(t_seg[0], t_seg[-1])
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Panel 2: Stage 2 Cleaned LFP
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for c in range(16):
        ax2.plot(t_seg, cleaned_seg[:, c] - np.median(cleaned_seg[:, c]) + c * 400, lw=0.7)
    ax2.set_title("Stage 2 Cleaned LFP (Sag Subtracted | Full Neural Oscillations Restored Immediately)", fontsize=11)
    ax2.set_ylabel("Ch 1 - 16", fontsize=9)
    ax2.set_xlim(t_seg[0], t_seg[-1])
    ax2.grid(True, linestyle=":", alpha=0.4)

    # Panel 3: Stage 1 CWT Spectrogram
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    vmin = min(np.nanpercentile(cwt_raw, 5), np.nanpercentile(cwt_clean, 5))
    vmax = max(np.nanpercentile(cwt_raw, 98), np.nanpercentile(cwt_clean, 98))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    ax3.pcolormesh(t_ds_plot, freqs, cwt_raw.T, cmap="jet", norm=norm, shading="auto")
    ax3.set_yscale("log")
    ax3.set_ylabel("Freq (Hz)", fontsize=9)
    ax3.set_title(f"Stage 1 CWT Spectrogram (Ch {REP_CHANNEL_IDX + 1}) - Low-Frequency Flare", fontsize=10)

    # Panel 4: Stage 2 Cleaned CWT Spectrogram
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.pcolormesh(t_ds_plot, freqs, cwt_clean.T, cmap="jet", norm=norm, shading="auto")
    ax4.set_yscale("log")
    ax4.set_ylabel("Freq (Hz)", fontsize=9)
    ax4.set_xlabel("Time (seconds)", fontsize=10)
    ax4.set_title(
        f"Stage 2 Cleaned CWT Spectrogram (Ch {REP_CHANNEL_IDX + 1}) - Flare Extinguished, Fast Rhythms Intact",
        fontsize=10)

    png_out = os.path.join(out_dir, "png", f"event_{ev_num + 1:02d}_t_{t_true_peak:.2f}s.png")
    svg_out = os.path.join(out_dir, "svg", f"event_{ev_num + 1:02d}_t_{t_true_peak:.2f}s.svg")
    fig.savefig(png_out, dpi=180, bbox_inches="tight")
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> Saved Event #{ev_num + 1:02d} diagnostic: {png_out}")


# =============================================================================
# 5. Full-Session Comparative Plotter
# =============================================================================

def plot_full_session(mouse_name, lfp_stage1, lfp_stage2, t_lfp, position, speed, intervals, out_dir):
    n_samples, n_channels = lfp_stage1.shape
    t_beh = np.arange(len(position)) / BEH_FS
    pos_x = position[:, 0] if position.ndim > 1 else position.reshape(-1)
    speed_1d = speed.reshape(-1)

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 2.5, 1.0], hspace=0.18)

    total_sec = t_lfp[-1]
    pct = (sum(e - s for s, e in intervals) / n_samples * 100.0) if intervals else 0.0
    fig.suptitle(
        f"{mouse_name} | Full Session (Full HF Recombination) | Repaired: {len(intervals)} ({pct:.2f}%) | Duration: {total_sec:.1f} s",
        fontsize=12, y=0.96
    )

    ch_iqr = np.nanpercentile(lfp_stage1, 75, axis=0) - np.nanpercentile(lfp_stage1, 25, axis=0)
    spacing = np.nanmedian(ch_iqr[ch_iqr > 0]) * 5.5 if np.any(ch_iqr > 0) else np.nanmedian(
        np.ptp(lfp_stage1, axis=0)) * 0.8
    offsets = np.arange(n_channels) * spacing
    colors = plt.cm.tab20(np.linspace(0, 1, n_channels))

    # Panel 1: Stage 1 LFP
    ax1 = fig.add_subplot(gs[0])
    for ch in range(n_channels):
        sig = lfp_stage1[:, ch] - np.nanmedian(lfp_stage1[:, ch])
        ax1.plot(t_lfp, sig + offsets[ch], color=colors[ch], lw=0.5, alpha=0.9)
    for s, e in intervals:
        ax1.axvspan(t_lfp[s], t_lfp[min(e, n_samples - 1)], color="#d62728", alpha=0.30)
    ax1.set_xlim(t_lfp[0], t_lfp[-1])
    ax1.set_ylim(-spacing * 0.8, offsets[-1] + spacing * 0.8)
    ax1.set_yticks(offsets)
    ax1.set_yticklabels([f"Ch {ch + 1}" for ch in range(n_channels)], fontsize=8)
    ax1.set_ylabel("Channels (1 to 16)", fontsize=9)
    ax1.set_title(f"1. Stage 1 LFP ({len(intervals)} Detected Regions in Red)", fontsize=10, loc="left")
    ax1.set_xticklabels([])
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Panel 2: Stage 2 Cleaned LFP
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for ch in range(n_channels):
        sig_clean = lfp_stage2[:, ch] - np.nanmedian(lfp_stage2[:, ch])
        ax2.plot(t_lfp, sig_clean + offsets[ch], color=colors[ch], lw=0.5, alpha=0.9)
    ax2.set_xlim(t_lfp[0], t_lfp[-1])
    ax2.set_ylim(-spacing * 0.8, offsets[-1] + spacing * 0.8)
    ax2.set_yticks(offsets)
    ax2.set_yticklabels([f"Ch {ch + 1}" for ch in range(n_channels)], fontsize=8)
    ax2.set_ylabel("Channels (1 to 16)", fontsize=9)
    ax2.set_title("2. Cleaned LFP (Sag Subtracted | Full-Strength Neural Oscillations Re-Injected)", fontsize=10,
                  loc="left")
    ax2.set_xticklabels([])
    ax2.grid(True, linestyle=":", alpha=0.4)

    # Panel 3: Behavior Tracking
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t_beh, pos_x, color="black", lw=1.1, label="Position")
    ax3.set_ylabel("Position (cm)", fontsize=9)
    ax3.set_xlabel("Time (seconds)", fontsize=10)
    ax3.set_ylim(np.nanmin(pos_x) - 2, np.nanmax(pos_x) + 5)
    ax3.grid(True, linestyle=":", alpha=0.5)

    ax_spd = ax3.twinx()
    ax_spd.plot(t_beh, speed_1d, color="#ff7f0e", lw=0.8, alpha=0.75, linestyle="--", label="Speed")
    ax_spd.set_ylabel("Speed (cm/s)", color="#ff7f0e", fontsize=9)
    ax_spd.tick_params(axis="y", labelcolor="#ff7f0e", labelsize=8)
    ax_spd.set_ylim(0, max(np.nanmax(speed_1d) * 1.1, 15))

    png_out = os.path.join(out_dir, f"{mouse_name}_full_session_comparison.png")
    svg_out = os.path.join(out_dir, f"{mouse_name}_full_session_comparison.svg")
    fig.savefig(png_out, dpi=200, bbox_inches="tight")
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved full-session figure: {png_out}")


# =============================================================================
# 6. Master Session Processing Loop
# =============================================================================

def process_animal_session(mouse_name, mat_path):
    print(f"\n{'=' * 80}\nProcessing: {mouse_name} ({os.path.basename(mat_path)})")
    mat_data = loadmat(mat_path, simplify_cells=True)
    if "tracesLFP" not in mat_data:
        return

    tlfp = mat_data["tracesLFP"]
    position = np.asarray(tlfp["position"])
    speed = np.asarray(tlfp["speed"]).reshape(-1)

    lfp_struct = tlfp.get("LFP", {})
    raw_lfp = lfp_struct.get("LFP", None)
    lfp_time = lfp_struct.get("time", None)
    if raw_lfp is None:
        return

    raw_lfp = np.asarray(raw_lfp, dtype=float)
    if raw_lfp.shape[0] == 16 and raw_lfp.shape[1] > 16:
        raw_lfp = raw_lfp.T

    if lfp_time is not None and len(lfp_time) > 1:
        t_lfp = np.asarray(lfp_time).reshape(-1)
        fs = 1.0 / np.nanmedian(np.diff(t_lfp))
    else:
        fs = float(raw_lfp.shape[0]) / (float(len(position)) / BEH_FS)
        t_lfp = np.linspace(0, len(position) / BEH_FS, raw_lfp.shape[0])

    mouse_out_dir = os.path.join(OUTPUT_BASE_DIR, mouse_name)
    events_dir = os.path.join(mouse_out_dir, "events")
    os.makedirs(os.path.join(events_dir, "png"), exist_ok=True)
    os.makedirs(os.path.join(events_dir, "svg"), exist_ok=True)

    # Stage 1: Noise filter
    print("  Running Stage 1 (Noise/Spike blanking)...")
    lfp_stage1 = stage1_noise_filter(raw_lfp, fs)

    # Stage 2: Deviation & Extended Slow Sag Filter
    print("  Running Stage 2 (Fast Recombination + Baseline Cleaning)...")
    lfp_stage2, intervals, metadata = stage2_slow_component_filter(lfp_stage1, fs)
    print(f"  Repaired {len(intervals)} artifact regions.")

    # Export individual event diagnostics
    for ev_num, meta in enumerate(metadata):
        t_seg = t_lfp[meta["win_start"]: meta["win_end"]]
        plot_single_event(
            t_seg=t_seg,
            raw_seg=meta["raw_seg"],
            cleaned_seg=meta["cleaned_seg"],
            t0_rel=meta["t0_rel"],
            t1_rel=meta["t1_rel"],
            true_peak_rel=meta["true_peak_rel"],
            fs=fs,
            ev_num=ev_num,
            out_dir=events_dir,
            mouse_name=mouse_name,
        )

    # Export full-session comparative plot
    plot_full_session(
        mouse_name=mouse_name,
        lfp_stage1=lfp_stage1,
        lfp_stage2=lfp_stage2,
        t_lfp=t_lfp,
        position=position,
        speed=speed,
        intervals=intervals,
        out_dir=mouse_out_dir,
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    target_mice = [
        "CalbEphys1GRIN1",
        "CalbEphys1GRIN2",
        "Thy1Ephys1GRIN1",
        "Thy1Ephys1GRIN2",
    ]

    for mouse_name in target_mice:
        mouse_dir = os.path.join(SOURCE_DATA_DIR, mouse_name)
        if not os.path.isdir(mouse_dir):
            continue

        mat_candidates = glob.glob(os.path.join(mouse_dir, "*_tracesLFP.mat"))
        if not mat_candidates:
            mat_candidates = glob.glob(os.path.join(mouse_dir, "*.mat"))
        if not mat_candidates:
            continue

        process_animal_session(mouse_name, mat_candidates[0])

    print(f"\n{'=' * 80}")
    print(f"All processing complete. Diagnostic figures saved in:\n  {OUTPUT_BASE_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()