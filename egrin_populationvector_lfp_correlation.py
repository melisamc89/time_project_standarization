#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Time-Lagged Cross-Correlation with Significance Testing & Filtered Summaries.

Workflow:
1. Computes observed cross-correlations across [-15s, +15s] lags between population
   traces and Gaussian-filtered slow LFP envelopes (variance = 6).
2. Generates an autocorrelation-preserving null distribution via N_SHUFFLES circular shifts
   (random offset >= ±30s).
3. Evaluates significance per mouse, cluster, and frequency band (p < 0.05 against the 95th
   percentile null threshold).
4. Generates:
   - Panel Overlay Suite: Real cross-correlation curve overlaid with the shaded 95%
     chance null envelope per band.
   - Significant-Only Summary Bars: Displays optimal lead-lag delays and peak correlation
     magnitudes ONLY for entries meeting significance, plotting empty/faded markers for
     non-significant observations.
"""

import glob
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, correlation_lags

# =============================================================================
# Directories & Settings
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
STANDARDIZED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "standardized_sessions",
    "lt_egrin",
    "calcium_imaging",
)
TRANSFER_CSV = os.path.join(
    PROJECT_ROOT,
    "analysis",
    "opentsne_universal_transfer_and_lfp",
    "transfer",
    "egrin_all",
    "egrin_all_labeled_neurons.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "analysis", "time_lagged_xcorr_with_significance"
)
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

MAX_LAG_SECONDS = 15.0
N_SHUFFLES = 200  # Number of circular shifts per condition
MIN_SHIFT_SECONDS = 30.0

GAUSS_VARIANCE = 6.0
GAUSS_SIGMA = np.sqrt(GAUSS_VARIANCE)

SLOW_BANDS = [
    ("lfp_ultra_ultra_slow_envelope", "U-U-Slow [0.001 - 0.01 Hz]"),
    ("lfp_ultra_slow_envelope", "U-Slow [0.01 - 0.10 Hz]"),
    ("lfp_slow_envelope", "Slow [0.10 - 1.00 Hz]"),
]

CLUSTER_COLORS = {
    "All": "#111111",
    "Time": "#bce784ff",
    "Direction": "#66cef4ff",
    "Position": "#ec8ef8ff",
    "Unassigned": "#bbbcc0ff",
}

CLUSTER_ORDER = ["All", "Time", "Direction", "Position", "Unassigned"]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_standardized_path(mouse_name, depth_type=None):
    candidates = glob.glob(
        os.path.join(STANDARDIZED_DIR, "**", f"*{mouse_name}*.pkl"), recursive=True
    )
    if candidates:
        if depth_type:
            for c in candidates:
                if depth_type in c:
                    return c
        return candidates[0]
    return None


def compute_xcorr_vector(x, y, max_lag_pts):
    x_c = x - np.nanmean(x)
    y_c = y - np.nanmean(y)
    norm = np.sqrt(np.sum(x_c ** 2) * np.sum(y_c ** 2))
    if norm == 0 or np.isnan(norm):
        return np.zeros(2 * max_lag_pts + 1)
    raw = correlate(x_c, y_c, mode="full")
    center = len(raw) // 2
    return (raw / norm)[center - max_lag_pts: center + max_lag_pts + 1]


# =============================================================================
# Cross-Correlation & Circular Shift Null Testing
# =============================================================================

def run_significance_xcorr_analysis(df_egrin):
    unique_mice = sorted(df_egrin["mouse_name"].unique())
    curve_records = []
    summary_records = []

    print(f"Running cross-correlations with {N_SHUFFLES} circular shifts across {len(unique_mice)} mice...")

    for mouse_name in unique_mice:
        df_mouse = df_egrin[df_egrin["mouse_name"] == mouse_name]
        depth_type = df_mouse["depth_type"].iloc[0] if "depth_type" in df_mouse.columns else None

        pkl_path = resolve_standardized_path(mouse_name, depth_type)
        if pkl_path is None or not os.path.isfile(pkl_path):
            continue

        std_data = load_pickle(pkl_path)
        signal_matrix = std_data["signal"]
        fs = float(std_data["metadata"].get("fs", 20.0))
        n_pts = signal_matrix.shape[0]

        max_lag_pts = int(round(MAX_LAG_SECONDS * fs))
        min_shift_pts = int(round(MIN_SHIFT_SECONDS * fs))

        if n_pts < 2 * min_shift_pts:
            continue

        lags_pts = correlation_lags(n_pts, n_pts, mode="full")
        center = len(lags_pts) // 2
        lags_sec = lags_pts[center - max_lag_pts: center + max_lag_pts + 1] / fs

        # Filtered slow LFP envelopes
        lfp_envelopes = {}
        for env_key, _ in SLOW_BANDS:
            raw_env = std_data.get(env_key, np.zeros(n_pts)).reshape(-1)
            lfp_envelopes[env_key] = gaussian_filter1d(raw_env, sigma=GAUSS_SIGMA)

        for grp in CLUSTER_ORDER:
            if grp == "All":
                idx = np.arange(signal_matrix.shape[1])
            else:
                idx = df_mouse.loc[df_mouse["cluster_label"] == grp, "neuron_index"].values.astype(int)

            valid_idx = [i for i in idx if i < signal_matrix.shape[1]]
            if len(valid_idx) == 0:
                continue

            pop_mean = np.nanmean(signal_matrix[:, valid_idx], axis=1)

            for env_key, env_label in SLOW_BANDS:
                # 1. Observed Cross-Correlation
                obs_xcorr = compute_xcorr_vector(pop_mean, lfp_envelopes[env_key], max_lag_pts)

                # 2. Circular Shift Null Distribution
                null_matrix = np.zeros((N_SHUFFLES, len(obs_xcorr)))
                for s in range(N_SHUFFLES):
                    shift = np.random.randint(min_shift_pts, n_pts - min_shift_pts)
                    shuffled_lfp = np.roll(lfp_envelopes[env_key], shift)
                    null_matrix[s, :] = compute_xcorr_vector(pop_mean, shuffled_lfp, max_lag_pts)

                null_mean = np.nanmean(null_matrix, axis=0)
                null_p95 = np.nanpercentile(null_matrix, 95, axis=0)

                # 3. Peak Identification & Statistical Test
                peak_idx = np.nanargmax(obs_xcorr)
                peak_r = obs_xcorr[peak_idx]
                optimal_lag = lags_sec[peak_idx]
                thresh_95 = null_p95[peak_idx]

                is_significant = bool(peak_r > thresh_95)

                summary_records.append({
                    "mouse_name": mouse_name,
                    "cluster": grp,
                    "band_key": env_key,
                    "band_label": env_label,
                    "optimal_lag_sec": optimal_lag,
                    "peak_xcorr": peak_r,
                    "null_95_thresh": thresh_95,
                    "is_significant": is_significant,
                    "n_neurons": len(valid_idx),
                })

                for lag_t, r_obs, r_null_mean, r_null_95 in zip(lags_sec, obs_xcorr, null_mean, null_p95):
                    curve_records.append({
                        "mouse_name": mouse_name,
                        "cluster": grp,
                        "band_key": env_key,
                        "band_label": env_label,
                        "lag_sec": lag_t,
                        "obs_xcorr": r_obs,
                        "null_mean": r_null_mean,
                        "null_95": r_null_95,
                    })

    return pd.DataFrame(curve_records), pd.DataFrame(summary_records)


# =============================================================================
# Plotting Suites
# =============================================================================

def plot_curves_with_null_envelope(curve_df, save_path):
    """Plots cross-correlation curves per band overlaid with the empirical 95% chance envelope."""
    fig, axes = plt.subplots(1, len(SLOW_BANDS), figsize=(18, 5.5), sharey=True, dpi=250)

    for c_idx, (env_key, env_label) in enumerate(SLOW_BANDS):
        ax = axes[c_idx]

        # Plot chance floor (average across all sessions)
        band_sub = curve_df[curve_df["band_key"] == env_key]
        lags_x = band_sub["lag_sec"].unique()
        lags_x.sort()

        null_p95_mean = band_sub.groupby("lag_sec")["null_95"].mean().loc[lags_x].values
        null_mean_mean = band_sub.groupby("lag_sec")["null_mean"].mean().loc[lags_x].values

        # Chance envelope shaded in gray
        ax.fill_between(
            lags_x,
            null_mean_mean,
            null_p95_mean,
            color="lightgray",
            alpha=0.6,
            label="95% Chance Level (Circular Shift)",
            zorder=1,
        )
        ax.plot(lags_x, null_p95_mean, color="darkgray", linestyle=":", lw=1.0, zorder=2)

        for grp in CLUSTER_ORDER:
            c_color = CLUSTER_COLORS[grp]
            c_sub = band_sub[band_sub["cluster"] == grp]
            mice = c_sub["mouse_name"].unique()
            mouse_curves = [c_sub[c_sub["mouse_name"] == m].sort_values("lag_sec")["obs_xcorr"].values for m in mice]

            if mouse_curves:
                arr = np.array(mouse_curves)
                mean_curve = np.nanmean(arr, axis=0)
                sem_curve = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(len(mouse_curves)) if len(
                    mouse_curves) > 1 else np.zeros_like(mean_curve)

                ax.plot(lags_x, mean_curve, color=c_color, lw=2.2, label=grp, zorder=4)
                ax.fill_between(lags_x, mean_curve - sem_curve, mean_curve + sem_curve, color=c_color, alpha=0.15,
                                zorder=3)

        ax.axvline(0, color="black", linestyle="--", lw=0.8, alpha=0.6)
        ax.axhline(0, color="gray", linestyle="-", lw=0.5, alpha=0.5)
        ax.set_title(env_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time Lag τ (seconds)\n[< 0: Ca²⁺ leads | > 0: LFP leads]", fontsize=10)
        ax.set_ylim(-0.25, 0.65)
        ax.grid(True, linestyle=":", alpha=0.4)

        if c_idx == 0:
            ax.set_ylabel("Cross-Correlation (r)", fontsize=11)
            ax.legend(title="Cluster / Model", frameon=True, fontsize=8, loc="upper right")

    fig.suptitle("lt_egrin | Cross-Correlation Lead-Lag Dynamics vs. Autocorrelation-Preserving Chance Envelopes",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Curves with Null Envelopes -> {save_path}")


def plot_significant_only_summary_bars(summary_df, save_path):
    """Grouped bars and scatters showing ONLY statistically significant observations.

    Non-significant mice are rendered as faint open grey circles; bars represent the
    mean ± SEM of significant observations only (annotating N_sig / N_total).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=250)
    metrics = [("optimal_lag_sec", "Optimal Lag Delay τ (s)", axes[0]),
               ("peak_xcorr", "Maximum Cross-Correlation (r)", axes[1])]

    n_bands = len(SLOW_BANDS)
    n_clusters = len(CLUSTER_ORDER)
    x_indices = np.arange(n_bands)
    bar_width = 0.80 / n_clusters

    for val_col, y_lbl, ax in metrics:
        for c_idx, grp in enumerate(CLUSTER_ORDER):
            c_color = CLUSTER_COLORS[grp]
            offset = (c_idx - (n_clusters - 1) / 2.0) * bar_width
            cluster_x = x_indices + offset

            sig_means, sig_sems = [], []

            for b_idx, (env_key, _) in enumerate(SLOW_BANDS):
                sub = summary_df[(summary_df["cluster"] == grp) & (summary_df["band_key"] == env_key)]

                sig_pts = sub[sub["is_significant"] == True][val_col].values
                non_sig_pts = sub[sub["is_significant"] == False][val_col].values

                # Mean and SEM calculated ONLY across significant points
                if len(sig_pts) > 0:
                    sig_means.append(np.mean(sig_pts))
                    sig_sems.append(np.std(sig_pts, ddof=1) / np.sqrt(len(sig_pts)) if len(sig_pts) > 1 else 0.0)
                else:
                    sig_means.append(np.nan)
                    sig_sems.append(0.0)

                # 1. Overlay Significant Points: Solid colored dots
                if len(sig_pts) > 0:
                    jitter = np.random.normal(0, 0.04 * bar_width, size=len(sig_pts))
                    ax.scatter(
                        cluster_x[b_idx] + jitter,
                        sig_pts,
                        color=c_color,
                        edgecolors="#222222",
                        s=45,
                        zorder=4,
                        label=f"{grp} (Sig)" if b_idx == 0 and c_idx == 0 else "",
                    )

                # 2. Overlay Non-Significant Points: Faded open markers
                if len(non_sig_pts) > 0:
                    jitter_ns = np.random.normal(0, 0.04 * bar_width, size=len(non_sig_pts))
                    ax.scatter(
                        cluster_x[b_idx] + jitter_ns,
                        non_sig_pts,
                        facecolors="none",
                        edgecolors="gray",
                        linestyle=":",
                        s=35,
                        alpha=0.55,
                        zorder=3,
                    )

            # Draw bar for significant mean
            ax.bar(
                cluster_x,
                np.nan_to_num(sig_means, nan=0.0),
                yerr=sig_sems,
                width=bar_width,
                color=c_color,
                edgecolor="#222222",
                alpha=0.85,
                label=grp if val_col == "peak_xcorr" else "",
                zorder=2,
                capsize=3,
            )

        ax.axhline(0, color="gray", linestyle="--", lw=0.8, alpha=0.7)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([b[1] for b in SLOW_BANDS], fontsize=10, fontweight="bold")
        ax.set_ylabel(y_lbl, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

    axes[0].set_title("Optimal Lead-Lag Delay (Significant Only)", fontsize=12, fontweight="bold")
    axes[1].set_title("Peak Correlation Magnitude (Significant Only)", fontsize=12, fontweight="bold")
    axes[1].legend(title="Cluster", frameon=True, fontsize=9, loc="upper right")

    fig.suptitle(
        "lt_egrin | Optimal Lags & Peak Correlation Restricted to Significant Observations (p < 0.05 vs. Circular Shift Null)\n"
        "Solid Bars & Dots: Significant | Faint Open Circles: Non-Significant",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Significant-Only Summary Bars -> {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(TABLES_DIR)

    if not os.path.isfile(TRANSFER_CSV):
        candidates = glob.glob(os.path.join(PROJECT_ROOT, "analysis", "**", "*egrin*labeled*.csv"), recursive=True)
        if candidates:
            csv_path = candidates[0]
        else:
            raise FileNotFoundError(f"Missing labeled CSV: {TRANSFER_CSV}")
    else:
        csv_path = TRANSFER_CSV

    print(f"Loading transferred neurons from: {csv_path}")
    df_egrin = pd.read_csv(csv_path)

    # 1. Compute Cross-Correlations and Empirical Significance
    curve_df, summary_df = run_significance_xcorr_analysis(df_egrin)

    # 2. Export Tables
    summary_path = os.path.join(TABLES_DIR, "lt_egrin_significance_xcorr_peaks.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[OK] Saved Significance Table -> {summary_path}")

    # 3. Generate Diagnostic Plots
    plot_curves_with_null_envelope(
        curve_df,
        os.path.join(PLOTS_DIR, "lt_egrin_lagged_curves_with_chance_envelope.png")
    )
    plot_significant_only_summary_bars(
        summary_df,
        os.path.join(PLOTS_DIR, "lt_egrin_significant_only_summary_bars.png")
    )

    # 4. Print Significance Summary
    print("\n" + "=" * 80)
    print("PERCENTAGE OF MICE EXHIBITING SIGNIFICANT CROSS-CORRELATION (p < 0.05):")
    print("=" * 80)
    agg = summary_df.groupby(["band_label", "cluster"]).agg(
        Total_Mice=("mouse_name", "count"),
        Sig_Mice=("is_significant", "sum"),
        Mean_Obs_Peak_r=("peak_xcorr", "mean"),
        Mean_95_Threshold=("null_95_thresh", "mean"),
    ).reset_index()
    agg["Percent_Significant"] = (100.0 * agg["Sig_Mice"] / agg["Total_Mice"]).map("{:.1f}%".format)
    print(agg.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()