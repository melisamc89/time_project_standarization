#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fully Controlled Structure Index Pipeline: Learning Dataset (Last Session per Mouse).

Controls Implemented:
  1. Neuron Count: Matched across clusters to N_min per animal.
  2. Activity Distribution across Clusters: Histogram-matched subsampling so
     position, time, direction, and all_cells have identical firing rate distributions.
  3. Early vs. Late Activity Drift: Chunk-wise activity tracking and per-chunk
     trace normalization (standardizing baseline and variance across chunks).
  4. Temporal Chunks: 5 sequential quintiles (Q1-Q5).
  5. Statistical Control: Two-Way ANOVA + ANCOVA (controlling for Activity as a covariate).

Outputs:
  - Tables: CSV with bootstrap iterations, matched neuron IDs, and activity values.
  - Plots: Cluster quintile curves, activity drift profiles, and regression diagnostics.
  - Stats: Complete ANOVA and ANCOVA tables with Partial eta^2 effect sizes.
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import pairwise_distances
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import umap

# Local Structure Index module
try:
    from structure_index import compute_structure_index
except ImportError:
    from src.structure_index import compute_structure_index

# =============================================================================
# 1. PATHS & SETTINGS
# =============================================================================

POSSIBLE_ROOTS = [
    Path("/home/hippolab/Documents/melisa/time_project_standarization"),
    Path("/home/melisamc/Documents/time_project_standarization"),
]
PROJECT_ROOT = next((p for p in POSSIBLE_ROOTS if p.is_dir()), POSSIBLE_ROOTS[0])

MASTER_INDEX_CSV = PROJECT_ROOT / "data" / "metadata" / "master_dataset_index.csv"
TRAINING_LABELED_CSV = (
        PROJECT_ROOT
        / "analysis"
        / "mir_training_and_transfer"
        / "training"
        / "data"
        / "learning_last_session_mir_labeled_neurons.csv"
)

OUT_DIR = PROJECT_ROOT / "analysis" / "learning_last_session_activity_controlled_si"
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"
STATS_DIR = OUT_DIR / "stats"

for p in [OUT_DIR, TABLE_DIR, PLOT_DIR, STATS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Experimental Parameters
N_CHUNKS = 5
N_BOOTSTRAP = 30
N_ACTIVITY_BINS = 6  # Bins for rate-matching
SMOOTH_SIGMA = 6

UMAP_N_NEIGHBORS = 60
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 3

CLUSTER_MAP = {
    -1: "all_cells",
    0: "time",
    1: "direction",
    2: "position",
}

CLUSTER_COLORS = {
    "position": "#ec8ef8ff",
    "time": "#bce784ff",
    "direction": "#66cef4ff",
    "all_cells": "#888888",
}

TARGET_BEHAVIORS = ["position", "time", "direction"]


# =============================================================================
# 2. ACTIVITY MATCHING & HELPER ROUTINES
# =============================================================================

def parse_session_datetime(session_name: str) -> pd.Timestamp:
    m = re.search(r"(\d{8})_(\d{6})", str(session_name))
    return pd.to_datetime(m.group(1) + "_" + m.group(2), format="%Y%m%d_%H%M%S") if m else pd.NaT


def resolve_path(path_str: str) -> Path | None:
    if not path_str or pd.isna(path_str):
        return None
    p = Path(str(path_str).strip())
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


def select_last_sessions(master_df: pd.DataFrame) -> pd.DataFrame:
    df_learn = master_df[master_df["dataset_name"].astype(str).str.lower() == "learning"].copy()
    df_learn["dt"] = df_learn["session_name"].apply(parse_session_datetime)
    last_idx = df_learn.groupby("mouse_name")["dt"].idxmax()
    return df_learn.loc[last_idx].reset_index(drop=True)


def equalize_activity_across_clusters(
        cluster_indices: Dict[str, np.ndarray],
        mean_activities: np.ndarray,
        n_bins: int = 6,
        random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Finds a subset of neurons from each cluster with matched firing rate distributions.
    Bins mean activities across all neurons and draws an equal count per bin from each cluster.
    """
    rng = np.random.default_rng(random_state)
    target_clusters = ["position", "time", "direction"]

    # Activity range spanning the overlap across clusters
    mins = [np.min(mean_activities[cluster_indices[c]]) for c in target_clusters if len(cluster_indices[c]) > 0]
    maxs = [np.max(mean_activities[cluster_indices[c]]) for c in target_clusters if len(cluster_indices[c]) > 0]

    low_bound = max(mins)
    high_bound = min(maxs)

    if low_bound >= high_bound:
        # Fallback to overall percentiles if strict bounds do not overlap
        low_bound = np.percentile(mean_activities, 5)
        high_bound = np.percentile(mean_activities, 95)

    bin_edges = np.linspace(low_bound, high_bound, n_bins + 1)

    # Determine maximum number of neurons we can take from each bin
    bin_neuron_counts = []
    for b in range(n_bins):
        b_low, b_high = bin_edges[b], bin_edges[b + 1]
        counts_per_cluster = []
        for c in target_clusters:
            idx = cluster_indices[c]
            acts = mean_activities[idx]
            in_bin = np.where((acts >= b_low) & (acts <= b_high if b == n_bins - 1 else acts < b_high))[0]
            counts_per_cluster.append(len(in_bin))
        bin_neuron_counts.append(min(counts_per_cluster))

    # Sample equalized subsets
    matched_indices: Dict[str, List[int]] = {c: [] for c in cluster_indices.keys()}

    for b in range(n_bins):
        n_take = bin_neuron_counts[b]
        if n_take == 0:
            continue
        b_low, b_high = bin_edges[b], bin_edges[b + 1]

        for c in list(target_clusters) + ["all_cells"]:
            idx = cluster_indices[c]
            acts = mean_activities[idx]
            in_bin_local = np.where((acts >= b_low) & (acts <= b_high if b == n_bins - 1 else acts < b_high))[0]
            if len(in_bin_local) >= n_take:
                chosen_local = rng.choice(in_bin_local, size=n_take, replace=False)
                matched_indices[c].extend(idx[chosen_local])
            else:
                # If all_cells or cluster has fewer in that exact bin, sample available
                matched_indices[c].extend(idx[in_bin_local])

    result = {c: np.unique(np.array(arr, dtype=int)) for c, arr in matched_indices.items()}

    # Ensure all target clusters have identical sample size
    min_final = min(len(result[c]) for c in target_clusters)
    if min_final >= 5:
        for c in result.keys():
            if len(result[c]) > min_final:
                result[c] = rng.choice(result[c], size=min_final, replace=False)
    return result


def compute_si(space_data: np.ndarray, labels: np.ndarray, beh_name: str) -> float:
    params = dict(n_bins=10, n_neighbors=30, num_shuffles=0, verbose=False, perc_neigh=1)
    if beh_name == "direction":
        params.update(n_bins=2, discrete_label=True)
    else:
        params.update(discrete_label=False)
    try:
        res = compute_structure_index(space_data, labels, **params)
        return float(res[0]) if isinstance(res, (tuple, list)) else float(res)
    except Exception:
        return np.nan


# =============================================================================
# 3. COMPUTATION LOOP
# =============================================================================

def run_controlled_analysis(force_recompute: bool = False):
    csv_out = TABLE_DIR / "controlled_learning_activity_matched_si.csv"

    if csv_out.is_file() and not force_recompute:
        print(f"[INFO] Found precomputed table:\n  {csv_out}\nProceeding directly to statistics & plotting.")
        df_res = pd.read_csv(csv_out)
    else:
        master_df = pd.read_csv(MASTER_INDEX_CSV)
        cluster_df = pd.read_csv(TRAINING_LABELED_CSV)

        last_sessions = select_last_sessions(master_df)
        print(f"[INFO] Running on {len(last_sessions)} last learning sessions:")
        for _, r in last_sessions.iterrows():
            print(f"  - Mouse: {r['mouse_name']} | Session: {r['session_name']}")

        records = []

        for _, row in last_sessions.iterrows():
            mouse_name = str(row["mouse_name"])
            sess_name = str(row["session_name"])
            fpath = resolve_path(row["standardized_file_path"])

            if fpath is None or not fpath.exists():
                print(f"[WARN] File not found for {mouse_name} | {sess_name}")
                continue

            with open(fpath, "rb") as f:
                data = pickle.load(f)

            raw_signal = np.asarray(data.get("signal", data.get("clean_traces")))
            if raw_signal.ndim != 2:
                continue

            pos = np.asarray(data["pos"])
            pos = pos[:, 0] if pos.ndim == 2 else pos.reshape(-1)
            time_vec = np.asarray(data["time"]).reshape(-1)
            mov_dir = np.asarray(data.get("mov_dir", np.zeros_like(time_vec))).reshape(-1)

            # Dimension orientation
            if raw_signal.shape[0] != len(time_vec) and raw_signal.shape[1] == len(time_vec):
                raw_signal = raw_signal.T

            n_time, n_neurons = raw_signal.shape
            signal = gaussian_filter1d(raw_signal, sigma=SMOOTH_SIGMA,
                                       axis=0) if SMOOTH_SIGMA > 0 else raw_signal.copy()

            # Global mean activity per neuron across entire session
            neuron_overall_activity = np.nanmean(signal, axis=0)

            # Cluster neuron membership
            sub_clusters = cluster_df[
                (cluster_df["mouse_name"] == mouse_name) & (cluster_df["session_name"] == sess_name)]
            cluster_indices = {}
            for gid in [0, 1, 2]:
                c_name = CLUSTER_MAP[gid]
                neurons = sub_clusters.loc[sub_clusters["cluster_id"] == gid, "neuron_index"].dropna().astype(
                    int).values
                neurons = [n for n in neurons if n < n_neurons]
                cluster_indices[c_name] = np.array(neurons)
            cluster_indices["all_cells"] = np.arange(n_neurons)

            # Check validity
            if any(len(cluster_indices[c]) < 5 for c in ["time", "direction", "position"]):
                print(f"[SKIP] Cluster size < 5 in {mouse_name}. Skipping.")
                continue

            # 1. CONTROLLING ACTIVITY ACROSS CLUSTERS: Equalize activity distributions
            matched_indices = equalize_activity_across_clusters(
                cluster_indices, neuron_overall_activity, n_bins=N_ACTIVITY_BINS, random_state=42
            )
            n_matched = len(matched_indices["position"])
            print(f"\n--> {mouse_name} | N_matched={n_matched} neurons per cluster (activity-equalized)")

            # Verify that activity distributions match (Kruskal-Wallis test across clusters)
            acts_to_test = [neuron_overall_activity[matched_indices[c]] for c in ["position", "time", "direction"]]
            kw_stat, kw_p = stats.kruskal(*acts_to_test)
            print(
                f"    Activity matching check: Kruskal-Wallis p = {kw_p:.4f} (p > 0.05 indicates matched distributions)")

            # 2. CONTROLLING DRIFT ACROSS TIME: 5 sequential quintiles
            chunk_edges = np.linspace(0, n_time, N_CHUNKS + 1, dtype=int)

            for chunk_idx in range(N_CHUNKS):
                t_start, t_end = chunk_edges[chunk_idx], chunk_edges[chunk_idx + 1]
                c_time = time_vec[t_start:t_end]
                c_pos = pos[t_start:t_end]
                c_dir = mov_dir[t_start:t_end]
                c_sig = signal[t_start:t_end, :]

                # Direction movement filter within chunk
                valid_t = (c_dir != 0) & np.isfinite(c_dir)
                if np.sum(valid_t) < 40:
                    continue

                c_pos = c_pos[valid_t]
                c_time = c_time[valid_t]
                c_dir = c_dir[valid_t]
                c_sig = c_sig[valid_t, :]

                # Chunk-standardized signal (removes early-to-late amplitude drift)
                mu_chunk = np.nanmean(c_sig, axis=0, keepdims=True)
                sd_chunk = np.nanstd(c_sig, axis=0, keepdims=True)
                sd_chunk[sd_chunk == 0] = 1.0
                c_sig_norm = (c_sig - mu_chunk) / sd_chunk

                beh_targets = {"position": c_pos, "time": c_time, "direction": c_dir}

                for c_name, neuron_pool in matched_indices.items():
                    # Mean chunk activity (unnormalized, to track drift)
                    chunk_activity = float(np.mean(c_sig[:, neuron_pool]))

                    # Bootstrap over matched pool
                    si_neural_boot = {b: [] for b in TARGET_BEHAVIORS}
                    si_umap_boot = {b: [] for b in TARGET_BEHAVIORS}
                    si_neural_norm_boot = {b: [] for b in TARGET_BEHAVIORS}

                    n_draws = N_BOOTSTRAP
                    rng = np.random.default_rng(42 + chunk_idx)

                    for _ in range(n_draws):
                        # Resample with replacement if pool equals n_matched, or draw random subset
                        sampled = rng.choice(neuron_pool, size=n_matched, replace=(len(neuron_pool) == n_matched))
                        sub_sig = c_sig[:, sampled]
                        sub_sig_norm = c_sig_norm[:, sampled]

                        # 3D UMAP fit for this chunk
                        try:
                            emb_umap = umap.UMAP(
                                n_neighbors=min(UMAP_N_NEIGHBORS, len(sub_sig) - 1),
                                n_components=UMAP_N_COMPONENTS,
                                min_dist=UMAP_MIN_DIST,
                                random_state=42,
                            ).fit_transform(sub_sig)
                        except Exception:
                            emb_umap = None

                        for b_name in TARGET_BEHAVIORS:
                            b_val = beh_targets[b_name]
                            si_neural_boot[b_name].append(compute_si(sub_sig, b_val, b_name))
                            si_neural_norm_boot[b_name].append(compute_si(sub_sig_norm, b_val, b_name))
                            if emb_umap is not None:
                                si_umap_boot[b_name].append(compute_si(emb_umap, b_val, b_name))

                    for b_name in TARGET_BEHAVIORS:
                        records.append({
                            "mouse_name": mouse_name,
                            "session_name": sess_name,
                            "chunk": f"Q{chunk_idx + 1}",
                            "chunk_idx": chunk_idx + 1,
                            "epoch": "early" if chunk_idx < 2 else ("late" if chunk_idx >= 3 else "mid"),
                            "cluster": c_name,
                            "n_matched": n_matched,
                            "chunk_mean_activity": chunk_activity,
                            "behavior": b_name,
                            "si_neural": float(np.nanmedian(si_neural_boot[b_name])),
                            "si_neural_drift_controlled": float(np.nanmedian(si_neural_norm_boot[b_name])),
                            "si_umap": float(np.nanmedian(si_umap_boot[b_name])) if len(
                                si_umap_boot[b_name]) else np.nan,
                        })

        df_res = pd.DataFrame(records)
        df_res.to_csv(csv_out, index=False)
        print(f"\n[OK] Saved results table -> {csv_out}")

    # Run statistical reports and figure generation
    run_controlled_statistics(df_res)
    plot_controlled_profiles(df_res)


# =============================================================================
# 4. STATISTICAL EVALUATION (ANCOVA & DRIFT MODELS)
# =============================================================================

def run_controlled_statistics(df: pd.DataFrame):
    report_lines = [
        "FULLY CONTROLLED STRUCTURE INDEX: LEARNING (LAST SESSION)",
        "=" * 85,
        "Controls: 1) Equalized neuron count (N_min)  2) Rate-matched activity distributions",
        "          3) Chunk-level z-scoring (drift removal)  4) ANCOVA covarying for activity\n",
    ]

    spaces_to_test = [
        ("si_neural", "COMPLETE NEURAL SPACE (Rate-Matched Neurons)"),
        ("si_neural_drift_controlled", "COMPLETE NEURAL SPACE (Chunk-Standardized / Drift-Removed)"),
        ("si_umap", "3D UMAP SPACE (Rate-Matched Neurons)"),
    ]

    for col_space, space_title in spaces_to_test:
        report_lines.extend(["=" * 85, f"DATA SPACE: {space_title}", "=" * 85])

        for beh in TARGET_BEHAVIORS:
            sub = df[df["behavior"] == beh].dropna(subset=[col_space, "chunk_mean_activity"]).copy()
            report_lines.append(f"\n>>> TARGET BEHAVIOR: {beh.upper()} <<<")

            # 1. Two-Way ANOVA: Cluster x Chunk
            try:
                model_aov = smf.ols(f"{col_space} ~ C(cluster) + C(chunk) + C(cluster):C(chunk)", data=sub).fit()
                aov_tbl = sm.stats.anova_lm(model_aov, typ=2)
                se = float(aov_tbl.loc["Residual", "sum_sq"])
                de = float(aov_tbl.loc["Residual", "df"])

                report_lines.append("A) Two-Way ANOVA: Cluster x Chunk (Type II SS):")
                for term in aov_tbl.index:
                    if term == "Residual":
                        continue
                    ss = float(aov_tbl.loc[term, "sum_sq"])
                    df_num = float(aov_tbl.loc[term, "df"])
                    f_val = float(aov_tbl.loc[term, "F"])
                    p_val = float(aov_tbl.loc[term, "PR(>F)"])
                    eta_p2 = ss / (ss + se) if (ss + se) > 0 else np.nan
                    report_lines.append(
                        f"   {term:25s} | F({int(df_num)}, {int(de)})={f_val:7.3f} | p={p_val:.4e} | eta_p^2={eta_p2:.4f}"
                    )
            except Exception as exc:
                report_lines.append(f"   ANOVA Failed: {exc}")

            # 2. ANCOVA: Controlling for continuous chunk activity level
            try:
                model_ancova = smf.ols(f"{col_space} ~ C(cluster) + C(chunk) + chunk_mean_activity", data=sub).fit()
                ancova_tbl = sm.stats.anova_lm(model_ancova, typ=2)
                se_anc = float(ancova_tbl.loc["Residual", "sum_sq"])
                de_anc = float(ancova_tbl.loc["Residual", "df"])

                report_lines.append("\nB) ANCOVA: Cluster x Chunk with Activity Level as Covariate:")
                for term in ancova_tbl.index:
                    if term == "Residual":
                        continue
                    ss = float(ancova_tbl.loc[term, "sum_sq"])
                    df_num = float(ancova_tbl.loc[term, "df"])
                    f_val = float(ancova_tbl.loc[term, "F"])
                    p_val = float(ancova_tbl.loc[term, "PR(>F)"])
                    eta_p2 = ss / (ss + se_anc) if (ss + se_anc) > 0 else np.nan
                    report_lines.append(
                        f"   {term:25s} | F({int(df_num)}, {int(de_anc)})={f_val:7.3f} | p={p_val:.4e} | eta_p^2={eta_p2:.4f}"
                    )
            except Exception as exc:
                report_lines.append(f"   ANCOVA Failed: {exc}")

            # 3. Early vs Late Epoch Test (Stability Contrast)
            sub_el = sub[sub["epoch"].isin(["early", "late"])].copy()
            v_early = sub_el.loc[sub_el["epoch"] == "early", col_space].values
            v_late = sub_el.loc[sub_el["epoch"] == "late", col_space].values
            if len(v_early) >= 2 and len(v_late) >= 2:
                t_stat, p_el = stats.ttest_ind(v_early, v_late, equal_var=False)
                act_early = sub_el.loc[sub_el["epoch"] == "early", "chunk_mean_activity"].values
                act_late = sub_el.loc[sub_el["epoch"] == "late", "chunk_mean_activity"].values
                act_diff = np.mean(act_early) - np.mean(act_late)

                report_lines.append(
                    f"\nC) Drift Stability (Early vs Late):\n"
                    f"   SI Difference:        Diff={np.mean(v_early) - np.mean(v_late):+.4f} | t={t_stat:.3f} | p={p_el:.4e}\n"
                    f"   Activity Difference:  Delta_Activity={act_diff:+.4f}"
                )

    report_path = STATS_DIR / "controlled_si_complete_ancova_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[OK] Statistical ANCOVA report saved -> {report_path.name}")


# =============================================================================
# 5. DIAGNOSTIC & SUMMARY PLOTTING (NO HORIZONTAL GUIDELINES)
# =============================================================================

def plot_controlled_profiles(df: pd.DataFrame):
    clusters = ["position", "time", "direction", "all_cells"]
    quintile_labels = ["Q1\n(0-20%)", "Q2\n(20-40%)", "Q3\n(40-60%)", "Q4\n(60-80%)", "Q5\n(80-100%)"]

    # -------------------------------------------------------------------------
    # Plot 1: Activity Drift across Chunks (Verifying Early vs. Late Decay)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    for c in clusters:
        c_df = df[df["cluster"] == c].copy()
        if c_df.empty:
            continue
        stats_act = c_df.groupby("chunk_idx")["chunk_mean_activity"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(
            stats_act["chunk_idx"], stats_act["mean"], yerr=stats_act["sem"],
            label=c, color=CLUSTER_COLORS.get(c, "black"), lw=2, marker="o", capsize=3
        )

    ax.set_xlabel("Session Quintile Chunk (Early -> Late)", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Mean Fluorescence ($\Delta F / F$)", fontsize=10, fontweight="bold")
    ax.set_title("Activity Drift across Chunks (Early vs. Late)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(quintile_labels)
    ax.legend(frameon=False, loc="best")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_drift = PLOT_DIR / "diagnostic_activity_drift_across_chunks"
    fig.savefig(out_drift.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_drift.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved drift diagnostic -> {out_drift.name}.png")

    # -------------------------------------------------------------------------
    # Plot 2: Controlled Structure Index Curves across Quintiles
    # -------------------------------------------------------------------------
    spaces_to_plot = [
        ("si_neural", "Neural Space (Rate-Matched)"),
        ("si_neural_drift_controlled", "Neural Space (Chunk-Normalized / Drift-Removed)"),
        ("si_umap", "3D UMAP Space (Rate-Matched)"),
    ]

    for space_col, space_name in spaces_to_plot:
        for beh in TARGET_BEHAVIORS:
            sub = df[df["behavior"] == beh].copy()
            fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)

            for c in clusters:
                c_df = sub[sub["cluster"] == c]
                if c_df.empty:
                    continue
                stats_chunk = c_df.groupby("chunk_idx")[space_col].agg(["mean", "sem"]).reset_index()
                ax.errorbar(
                    stats_chunk["chunk_idx"], stats_chunk["mean"], yerr=stats_chunk["sem"],
                    label=c, color=CLUSTER_COLORS.get(c, "black"), lw=2, marker="o", capsize=3
                )

            ax.set_xlabel("Session Quintile Chunk (Early -> Late)", fontsize=10, fontweight="bold")
            ax.set_ylabel(f"Structure Index ({space_name})", fontsize=10, fontweight="bold")
            ax.set_title(f"Controlled SI | {beh.upper()} ({space_name})", fontsize=11, fontweight="bold", pad=10)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xticklabels(quintile_labels)
            ax.legend(frameon=False, loc="best")

            # Strictly NO horizontal lines or gridlines
            ax.grid(False)
            ax.spines[["top", "right"]].set_visible(False)

            fig.tight_layout()
            out_base = PLOT_DIR / f"controlled_si_{space_col}_{beh}_quintiles"
            fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
            fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Saved controlled SI plot -> {out_base.name}.png")

    # -------------------------------------------------------------------------
    # Plot 3: SI vs. Activity Level Regression (ANCOVA Diagnostic)
    # -------------------------------------------------------------------------
    for beh in TARGET_BEHAVIORS:
        sub = df[df["behavior"] == beh].copy()
        fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=300)

        for c in clusters:
            c_df = sub[sub["cluster"] == c].dropna(subset=["si_neural", "chunk_mean_activity"])
            if len(c_df) < 5:
                continue

            x = c_df["chunk_mean_activity"].values
            y = c_df["si_neural"].values

            # Scatter points
            ax.scatter(x, y, color=CLUSTER_COLORS.get(c, "black"), alpha=0.6, s=35, label=c)

            # Linear trend
            slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
            x_line = np.linspace(np.min(x), np.max(x), 50)
            ax.plot(x_line, slope * x_line + intercept, color=CLUSTER_COLORS.get(c, "black"), lw=1.6)

        ax.set_xlabel(r"Chunk Mean Activity ($\Delta F / F$)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Neural Structure Index (SI)", fontsize=10, fontweight="bold")
        ax.set_title(f"SI vs. Activity Covariate Diagnostic | {beh.upper()}", fontsize=11, fontweight="bold", pad=10)
        ax.legend(frameon=False, loc="best")
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        out_reg = PLOT_DIR / f"diagnostic_si_vs_activity_regression_{beh}"
        fig.savefig(out_reg.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(out_reg.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved ANCOVA covariate diagnostic -> {out_reg.name}.png")


# =============================================================================
# 6. EXECUTION ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fully controlled SI analysis (neuron counts, activity matching, drift, ANCOVA).")
    parser.add_argument("--force", action="store_true", help="Force recomputation of the bootstrap matching table.")
    args, _ = parser.parse_known_args()

    run_controlled_analysis(force_recompute=args.force)


if __name__ == "__main__":
    main()