#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""State-Dependent Phase-Amplitude Coupling (PAC) Comodulogram Suite

1. Filters continuous cleaned LFP across the full session (0.01-20 Hz phase, 2-1000 Hz amplitude)
   to eliminate chunk-boundary edge effects and filter step discontinuities.
2. Interpolates behavioral speed from the tracking timeline onto the full-rate LFP timeline.
3. Separates continuous phase/amplitude samples into two behavioral regimes:
   - Moving / Active Run : Speed > 3.0 cm/s (clean artifact-free samples)
   - Quiet / Immobile    : Speed <= 3.0 cm/s (clean artifact-free samples)
4. Computes Tort's Modulation Index independently across Moving and Quiet concatenated epochs.
5. Generates publication-ready comparative side-by-side comodulograms on a shared
   PowerNorm color scale with difference maps (Moving - Quiet).
"""

import glob
import os
import pickle
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

# =============================================================================
# Directories & Hyperparameters
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
STANDARDIZED_DIR = os.path.join(
    PROJECT_ROOT, "data", "standardized_sessions", "lt_egrin", "calcium_imaging"
)
SOURCE_DATA_DIR = "/home/hippolab/Documents/melisa/time_project_grin_lens/data"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "lfp_comodulograms_state_dependent")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPEED_TH = 3.0          # Speed threshold: Run > 3.0 cm/s, Quiet <= 3.0 cm/s
N_PHASE_BINS = 18       # Standard phase binning [0, 2pi)

# Phase frequency continuum: 0.01 Hz (1e-2) to 20.0 Hz
F_PHASE_CENTERS = np.geomspace(0.01, 20.0, 40)

# Amplitude frequency continuum: 2.0 Hz to 1000.0 Hz
F_AMP_CENTERS = np.geomspace(2.0, 1000.0, 48)


# =============================================================================
# 1. Modulation Index (Tort et al., 2010)
# =============================================================================

def compute_tort_modulation_index(phase_rad, amp_envelope, n_bins=N_PHASE_BINS):
    """Computes Tort's Modulation Index (MI) via Shannon entropy of phase-amplitude distribution."""
    if len(phase_rad) < n_bins * 20:
        return 0.0

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_indices = np.digitize(phase_rad, bin_edges) - 1
    bin_indices[bin_indices == n_bins] = n_bins - 1

    mean_amps = np.zeros(n_bins, dtype=float)
    for b in range(n_bins):
        mask = (bin_indices == b)
        if np.any(mask):
            mean_amps[b] = np.mean(amp_envelope[mask])
        else:
            mean_amps[b] = 1e-12

    sum_amp = np.sum(mean_amps)
    if sum_amp <= 0 or np.isnan(sum_amp):
        return 0.0

    p = mean_amps / sum_amp
    p[p <= 0] = 1e-12

    h_p = -np.sum(p * np.log(p))
    h_max = np.log(n_bins)
    mi = (h_max - h_p) / h_max
    return max(0.0, float(mi))


# =============================================================================
# 2. State-Partitioned Comodulogram Engine
# =============================================================================

def compute_state_dependent_comodulograms(
    lfp_clean_1d, fs, f_phase_list, f_amp_list, moving_mask, quiet_mask
):
    """Filters signals across the full uninterrupted recording, then computes MI

    separately over concatenated Moving and Quiet sample subsets.
    """
    n_phases = len(f_phase_list)
    n_amps = len(f_amp_list)

    mi_moving = np.zeros((n_amps, n_phases), dtype=np.float32)
    mi_quiet = np.zeros((n_amps, n_phases), dtype=np.float32)

    print(f"  Filtering {n_phases} phase bands and {n_amps} amplitude bands...")

    # Precompute phase sequences
    phases_mov = []
    phases_quiet = []

    for fp in f_phase_list:
        bw = max(0.005, fp * 0.35)
        low = max(0.002, fp - bw / 2.0)
        high = min(fs / 2.0 - 0.5, fp + bw / 2.0)
        sos_p = butter(2, [low, high], btype="bandpass", fs=fs, output="sos")
        filt_p = sosfiltfilt(sos_p, lfp_clean_1d)
        phase_sig = np.angle(hilbert(filt_p))

        phases_mov.append(phase_sig[moving_mask])
        phases_quiet.append(phase_sig[quiet_mask])

    # Precompute amplitude sequences and evaluate MI
    for ai, fa in enumerate(f_amp_list):
        bw = max(1.5, fa * 0.30)
        low = max(0.8, fa - bw / 2.0)
        high = min(fs / 2.0 - 1.0, fa + bw / 2.0)

        if high <= low:
            continue

        sos_a = butter(3, [low, high], btype="bandpass", fs=fs, output="sos")
        filt_a = sosfiltfilt(sos_a, lfp_clean_1d)
        amp_env = np.abs(hilbert(filt_a))

        amp_mov = amp_env[moving_mask]
        amp_quiet = amp_env[quiet_mask]

        for pi, fp in enumerate(f_phase_list):
            if fa >= (fp + 1.0) and fa >= (fp * 1.20):
                mi_moving[ai, pi] = compute_tort_modulation_index(phases_mov[pi], amp_mov)
                mi_quiet[ai, pi] = compute_tort_modulation_index(phases_quiet[pi], amp_quiet)
            else:
                mi_moving[ai, pi] = 0.0
                mi_quiet[ai, pi] = 0.0

    return mi_moving, mi_quiet


# =============================================================================
# 3. Smooth Comparative Visualization
# =============================================================================

def smooth_comodulogram_matrix(mi_matrix, f_phases, f_amps, dense_n=300):
    """Interpolates discrete MI points onto a dense log-frequency grid."""
    mi_filt = gaussian_filter(mi_matrix, sigma=0.75)
    log_fp_orig = np.log10(f_phases)
    log_fa_orig = np.log10(f_amps)

    interp_func = RegularGridInterpolator(
        (log_fa_orig, log_fp_orig),
        mi_filt,
        method="cubic",
        bounds_error=False,
        fill_value=0.0,
    )

    log_fp_dense = np.linspace(log_fp_orig[0], log_fp_orig[-1], dense_n)
    log_fa_dense = np.linspace(log_fa_orig[0], log_fa_orig[-1], dense_n)
    mesh_fa, mesh_fp = np.meshgrid(log_fa_dense, log_fp_dense, indexing="ij")

    dense_mi = interp_func((mesh_fa, mesh_fp))
    dense_mi = np.maximum(dense_mi, 0.0)

    # Mask unphysiological carrier space
    dense_fa = 10**mesh_fa
    dense_fp = 10**mesh_fp
    invalid_mask = (dense_fa < (dense_fp + 0.8)) | (dense_fa < (dense_fp * 1.15))
    dense_mi[invalid_mask] = 0.0

    return dense_mi, log_fp_dense, log_fa_dense, invalid_mask


def plot_comparative_state_comodulogram(
    mi_mov, mi_quiet, f_phases, f_amps, mouse_name, best_ch, pct_mov, pct_quiet, save_path
):
    dense_mov, log_fp, log_fa, invalid_mask = smooth_comodulogram_matrix(mi_mov, f_phases, f_amps)
    dense_quiet, _, _, _ = smooth_comodulogram_matrix(mi_quiet, f_phases, f_amps)

    # Shared dynamic contrast limits for direct comparison
    combined_valid = np.concatenate([dense_mov[~invalid_mask], dense_quiet[~invalid_mask]])
    vmax = np.percentile(combined_valid[combined_valid > 0], 99.2) if np.any(combined_valid > 0) else 0.010
    norm = mcolors.PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)

    diff_matrix = dense_mov - dense_quiet
    diff_vlim = np.percentile(np.abs(diff_matrix[~invalid_mask]), 98.5) if np.any(diff_matrix != 0) else 0.005

    fig = plt.figure(figsize=(24, 7.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.22)

    cmap_pac = plt.cm.turbo.copy()
    cmap_pac.set_under("#0b001a")

    phase_ticks = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0]
    amp_ticks = [2.0, 4.0, 8.0, 12.0, 30.0, 80.0, 150.0, 300.0, 600.0, 1000.0]

    def _setup_axes(ax, title):
        ax.set_xticks(np.log10(phase_ticks))
        ax.set_xticklabels(["0.01", "0.05", "0.1", "0.5", "1", "2", "4", "8", "12", "20"], fontsize=9)
        ax.set_yticks(np.log10(amp_ticks))
        ax.set_yticklabels(["2", "4", "8", "12", "30", "80", "150", "300", "600", "$10^3$"], fontsize=9)
        ax.set_xlim(log_fp[0], log_fp[-1])
        ax.set_ylim(log_fa[0], log_fa[-1])

        # Guides
        ax.axvline(np.log10(6.0), color="white", linestyle="--", alpha=0.7, lw=0.9, label="Theta (6-12 Hz)")
        ax.axvline(np.log10(12.0), color="white", linestyle="--", alpha=0.7, lw=0.9)
        ax.axvline(np.log10(1.0), color="cyan", linestyle=":", alpha=0.7, lw=0.9, label="Delta (1-4 Hz)")
        ax.axvline(np.log10(4.0), color="cyan", linestyle=":", alpha=0.7, lw=0.9)
        ax.axhline(np.log10(8.0), color="yellow", linestyle="-.", alpha=0.7, lw=0.9, label="Theta Amp (8 Hz)")
        ax.axhline(np.log10(80.0), color="white", linestyle=":", alpha=0.6, lw=0.8, label="Gamma (30-80 Hz)")
        ax.axhline(np.log10(150.0), color="magenta", linestyle=":", alpha=0.7, lw=0.9, label="Ripples (150-250 Hz)")

        ax.set_xlabel(r"Phase Frequency $f_\phi$ (Hz)", fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(True, linestyle=":", alpha=0.25)

    # Panel 1: Moving Periods
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(
        dense_mov,
        extent=[log_fp[0], log_fp[-1], log_fa[0], log_fa[-1]],
        origin="lower",
        cmap=cmap_pac,
        norm=norm,
        aspect="auto",
        interpolation="bicubic",
    )
    _setup_axes(ax1, f"MOVING (Speed > {SPEED_TH} cm/s) | {pct_mov:.1f}% Time")
    ax1.set_ylabel(r"Amplitude Frequency $f_A$ (Hz)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.65)
    cb1 = plt.colorbar(im1, ax=ax1, pad=0.02, aspect=25, extend="max")
    cb1.set_label("Modulation Index (MI)", fontsize=10)

    # Panel 2: Quiet Periods
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(
        dense_quiet,
        extent=[log_fp[0], log_fp[-1], log_fa[0], log_fa[-1]],
        origin="lower",
        cmap=cmap_pac,
        norm=norm,
        aspect="auto",
        interpolation="bicubic",
    )
    _setup_axes(ax2, f"QUIET (Speed $\\leq$ {SPEED_TH} cm/s) | {pct_quiet:.1f}% Time")
    cb2 = plt.colorbar(im2, ax=ax2, pad=0.02, aspect=25, extend="max")
    cb2.set_label("Modulation Index (MI)", fontsize=10)

    # Panel 3: Difference Map (Moving - Quiet)
    ax3 = fig.add_subplot(gs[2])
    diff_display = dense_mov - dense_quiet
    diff_display[invalid_mask] = 0.0

    im3 = ax3.imshow(
        diff_display,
        extent=[log_fp[0], log_fp[-1], log_fa[0], log_fa[-1]],
        origin="lower",
        cmap="bwr",
        vmin=-diff_vlim,
        vmax=diff_vlim,
        aspect="auto",
        interpolation="bicubic",
    )
    _setup_axes(ax3, "DIFFERENCE (Moving - Quiet)")
    cb3 = plt.colorbar(im3, ax=ax3, pad=0.02, aspect=25)
    cb3.set_label(r"$\Delta$ Modulation Index", fontsize=10)

    fig.suptitle(
        f"{mouse_name} | Channel {best_ch} State-Dependent Comodulograms (Moving vs. Quiet)",
        fontsize=14,
        y=0.99,
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved Comparative State Comodulogram: {save_path}")


# =============================================================================
# 4. Master Session Ingestion & Execution
# =============================================================================

def process_session(pkl_path):
    with open(pkl_path, "rb") as f:
        std_data = pickle.load(f)

    meta = std_data["metadata"]
    mouse_name = meta["mouse_name"]
    best_ch = meta.get("best_theta_channel", 1)

    print(f"\n{'='*80}")
    print(f"State-Dependent PAC Analysis: {mouse_name} (Best Channel {best_ch})")
    print(f"{'='*80}")

    # Load raw MATLAB session
    mouse_dir = os.path.join(SOURCE_DATA_DIR, mouse_name)
    mat_candidates = glob.glob(os.path.join(mouse_dir, "*_tracesLFP.mat"))
    if not mat_candidates:
        mat_candidates = glob.glob(os.path.join(mouse_dir, "*.mat"))

    mat_data = loadmat(mat_candidates[0], simplify_cells=True)
    tlfp = mat_data["tracesLFP"]

    # Load LFP matrix and sample clock
    lfp_struct = tlfp.get("LFP", {})
    raw_lfp = np.asarray(lfp_struct.get("LFP"), dtype=float)
    if raw_lfp.shape[0] == 16 and raw_lfp.shape[1] > 16:
        raw_lfp = raw_lfp.T

    lfp_time = lfp_struct.get("time", None)
    if lfp_time is not None and len(lfp_time) > 1:
        t_lfp = np.asarray(lfp_time).reshape(-1)
        dt_lfp = np.nanmedian(np.diff(t_lfp))
        fs_lfp = 1.0 / dt_lfp
    else:
        fs_lfp = 2500.0
        dt_lfp = 1.0 / fs_lfp
        t_lfp = np.arange(raw_lfp.shape[0]) * dt_lfp

    # Load tracking coordinates and velocity timeline
    speed = np.asarray(tlfp["speed"]).reshape(-1)
    position = np.asarray(tlfp["position"])
    n_frames = len(speed)
    total_sec = t_lfp[-1] - t_lfp[0]
    fs_beh = float(n_frames) / total_sec
    t_beh = np.arange(n_frames) / fs_beh

    print(f"  Sample Rates: LFP = {fs_lfp:.1f} Hz | Behavior = {fs_beh:.2f} Hz | Duration = {total_sec:.1f} s")

    # In-place clean for capacitive knocks and large RC sags
    sos_slow = butter(2, (0.2, 2.5), btype="bandpass", fs=fs_lfp, output="sos")
    envelope_slow = np.abs(hilbert(sosfiltfilt(sos_slow, raw_lfp, axis=0), axis=0))
    cross_slow = np.mean(envelope_slow, axis=1)
    med_p = np.median(cross_slow)
    mad_p = np.median(np.abs(cross_slow - med_p)) * 1.4826
    z_slow = (cross_slow - med_p) / max(mad_p, 1e-6)

    knock_peaks, _ = find_peaks(
        z_slow, distance=int(round(2.0 * fs_lfp)), height=6.0, prominence=3.0
    )

    high_cutoff = min(500.0, (fs_lfp / 2.0) - 10.0)
    sos_clean = butter(3, [4.0, high_cutoff], btype="bandpass", fs=fs_lfp, output="sos")
    hf = sosfiltfilt(sos_clean, raw_lfp, axis=0)

    lfp_clean = raw_lfp[:, best_ch - 1].copy()
    ch_base = np.median(raw_lfp[:, best_ch - 1])
    artifact_mask = np.zeros(len(lfp_clean), dtype=bool)

    for p in knock_peaks:
        t0 = max(0, p - int(round(0.40 * fs_lfp)))
        t1 = min(len(lfp_clean), p + int(round(2.5 * fs_lfp)))
        lfp_clean[t0:t1] = hf[t0:t1, best_ch - 1] + ch_base
        artifact_mask[t0:t1] = True

    # Interpolate behavior velocity onto the full-rate LFP timeline
    interp_speed = interp1d(t_beh, speed, bounds_error=False, fill_value="extrapolate")
    speed_on_lfp = interp_speed(t_lfp)

    # Establish clean behavioral partition masks
    clean_mask = ~artifact_mask
    moving_mask = clean_mask & (speed_on_lfp > SPEED_TH)
    quiet_mask = clean_mask & (speed_on_lfp <= SPEED_TH)

    n_total = len(lfp_clean)
    pct_mov = 100.0 * np.sum(moving_mask) / n_total
    pct_quiet = 100.0 * np.sum(quiet_mask) / n_total
    pct_art = 100.0 * np.sum(artifact_mask) / n_total

    print(f"  Epoch Composition:")
    print(f"    * Moving (Speed > {SPEED_TH} cm/s) : {np.sum(moving_mask):,} samples ({pct_mov:.2f}%)")
    print(f"    * Quiet  (Speed <= {SPEED_TH} cm/s): {np.sum(quiet_mask):,} samples ({pct_quiet:.2f}%)")
    print(f"    * Artifacts Excised               : {np.sum(artifact_mask):,} samples ({pct_art:.2f}%)")

    # Compute Moving and Quiet PAC matrices
    mi_mov, mi_quiet = compute_state_dependent_comodulograms(
        lfp_clean_1d=lfp_clean,
        fs=fs_lfp,
        f_phase_list=F_PHASE_CENTERS,
        f_amp_list=F_AMP_CENTERS,
        moving_mask=moving_mask,
        quiet_mask=quiet_mask,
    )

    # Render Comparative Comodulograms
    save_fig_path = os.path.join(OUTPUT_DIR, f"{mouse_name}_moving_vs_quiet_comodulogram.png")
    plot_comparative_state_comodulogram(
        mi_mov=mi_mov,
        mi_quiet=mi_quiet,
        f_phases=F_PHASE_CENTERS,
        f_amps=F_AMP_CENTERS,
        mouse_name=mouse_name,
        best_ch=best_ch,
        pct_mov=pct_mov,
        pct_quiet=pct_quiet,
        save_path=save_fig_path,
    )

    # Serialize matrices
    save_npz_path = os.path.join(OUTPUT_DIR, f"{mouse_name}_state_comodulograms.npz")
    np.savez(
        save_npz_path,
        mi_moving=mi_mov,
        mi_quiet=mi_quiet,
        f_phases=F_PHASE_CENTERS,
        f_amps=F_AMP_CENTERS,
        best_ch=best_ch,
        fs_lfp=fs_lfp,
        fs_beh=fs_beh,
        pct_moving=pct_mov,
        pct_quiet=pct_quiet,
    )
    print(f"  -> Saved state matrices: {save_npz_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    pkl_files = glob.glob(os.path.join(STANDARDIZED_DIR, "**", "*.pkl"), recursive=True)
    if not pkl_files:
        print(f"No standardized pickle files located in {STANDARDIZED_DIR}")
        return

    print(f"Found {len(pkl_files)} standardized sessions.")
    for pkl_path in sorted(pkl_files):
        process_session(pkl_path)

    print(f"\n{'='*80}\nAll state-dependent comodulograms saved to:\n  {OUTPUT_DIR}\n{'='*80}")


if __name__ == "__main__":
    main()