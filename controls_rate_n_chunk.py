#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Controlled Structure Index Analysis: Learning Dataset (Last Session per Mouse).

Controls:
  1. Equalizes neuron count across clusters (subsampling to N_min per animal).
  2. Evaluates non-stationarity / drift across 5 sequential chunks (Early vs. Late).
  3. Quantifies mean firing rate / activity levels across chunks and clusters.
  4. Runs in both Complete Neural Space and 3D UMAP Space.
  5. Statistical testing: Two-Way ANOVA (Cluster x Chunk) + Post-Hocs.
"""

from __future__ import annotations

import copy
import glob
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import pairwise_distances
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import umap

# Local Structure Index module
from structure_index import compute_structure_index

# =============================================================================
# Paths & Settings
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

OUT_DIR = PROJECT_ROOT / "analysis" / "learning_last_session_controlled_si"
TABLE_DIR = OUT_DIR / "tables"
PLOT_DIR = OUT_DIR / "plots"
STATS_DIR = OUT_DIR / "stats"

for p in [OUT_DIR, TABLE_DIR, PLOT_DIR, STATS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Experimental Parameters
N_CHUNKS = 5
N_BOOTSTRAP = 25
SMOOTH_SIGMA = 6
Z_SCORE_NEURONS = False

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
# Helpers
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
        for prefix in ["/home/melisamc/Documents/time_project_standarization",
                       "/home/hippolab/Documents/melisa/time_project_standarization"]:
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
# Main Pipeline
# =============================================================================

def run_controlled_analysis():
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
        if SMOOTH_SIGMA > 0:
            signal = gaussian_filter1d(raw_signal, sigma=SMOOTH_SIGMA, axis=0)
        else:
            signal = raw_signal.copy()

        # Map neurons to clusters
        sub_clusters = cluster_df[
            (cluster_df["mouse_name"] == mouse_name) & (cluster_df["session_name"] == sess_name)
            ]

        cluster_indices = {}
        for gid in [0, 1, 2]:
            c_name = CLUSTER_MAP[gid]
            neurons = sub_clusters.loc[sub_clusters["cluster_id"] == gid, "neuron_index"].dropna().astype(int).values
            neurons = [n for n in neurons if n < n_neurons]
            cluster_indices[c_name] = np.array(neurons)

        cluster_indices["all_cells"] = np.arange(n_neurons)

        # 1. CONTROL FOR CELL COUNT: Find minimum N across functional clusters
        cluster_sizes = [len(cluster_indices[c]) for c in ["time", "direction", "position"]]
        n_min = min(cluster_sizes)
        print(
            f"\n--> {mouse_name} | {sess_name} | Sizes: { {k: len(v) for k, v in cluster_indices.items()} } | Subsample target N_min = {n_min}")

        if n_min < 5:
            print(f"[SKIP] N_min ({n_min}) too small for {mouse_name}.")
            continue

        # 2. CONTROL FOR TASK DRIFT: Split session into 5 sequential chunks
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

            beh_targets = {"position": c_pos, "time": c_time, "direction": c_dir}

            for c_name, neuron_pool in cluster_indices.items():
                # 3. CONTROL FOR ACTIVITY: Compute mean chunk activity (Delta F / F)
                mean_activity = float(np.mean(c_sig[:, neuron_pool]))

                # 4. BOOTSTRAP SUBSAMPLING to N_min
                si_neural_boot = {b: [] for b in TARGET_BEHAVIORS}
                si_umap_boot = {b: [] for b in TARGET_BEHAVIORS}

                n_draws = 1 if len(neuron_pool) == n_min else N_BOOTSTRAP
                rng = np.random.default_rng(42)

                for _ in range(n_draws):
                    sampled_neurons = rng.choice(neuron_pool, size=n_min, replace=False) if len(
                        neuron_pool) > n_min else neuron_pool
                    sub_sig = c_sig[:, sampled_neurons]

                    # 3D UMAP fit for this chunk
                    try:
                        emb_umap = umap.UMAP(
                            n_neighbors=min(UMAP_N_NEIGHBORS, len(sub_sig) - 1),
                            n_components=UMAP_N_COMPONENTS,
                            min_dist=UMAP_MIN_DIST,
                            random_state=42
                        ).fit_transform(sub_sig)
                    except Exception:
                        emb_umap = None

                    for b_name in TARGET_BEHAVIORS:
                        b_val = beh_targets[b_name]
                        si_n = compute_si(sub_sig, b_val, b_name)
                        si_neural_boot[b_name].append(si_n)
                        if emb_umap is not None:
                            si_u = compute_si(emb_umap, b_val, b_name)
                            si_umap_boot[b_name].append(si_u)

                for b_name in TARGET_BEHAVIORS:
                    records.append({
                        "mouse_name": mouse_name,
                        "session_name": sess_name,
                        "chunk": f"Q{chunk_idx + 1}",
                        "chunk_idx": chunk_idx + 1,
                        "epoch": "early" if chunk_idx < 2 else ("late" if chunk_idx >= 3 else "mid"),
                        "cluster": c_name,
                        "n_sampled": n_min,
                        "mean_activity": mean_activity,
                        "behavior": b_name,
                        "si_neural": float(np.nanmedian(si_neural_boot[b_name])),
                        "si_umap": float(np.nanmedian(si_umap_boot[b_name])) if len(si_umap_boot[b_name]) else np.nan,
                    })

    df_res = pd.DataFrame(records)
    csv_out = TABLE_DIR / "controlled_learning_last_session_si.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"\n[OK] Saved results -> {csv_out}")

    # =========================================================================
    # 5. STATISTICAL EVALUATION & REPORTING
    # =========================================================================
    run_controlled_statistics(df_res)


def run_controlled_statistics(df: pd.DataFrame):
    report_lines = [
        "CONTROLLED STRUCTURE INDEX: LEARNING (LAST SESSION)",
        "=" * 80,
        "Controls: 1) Equalized neuron counts (N_min)  2) Quintile temporal chunks (Q1-Q5)",
        "          3) Activity quantification (Delta F/F)  4) Early vs. Late epoch contrasts\n",
    ]

    for space in ["si_neural", "si_umap"]:
        space_name = "COMPLETE NEURAL SPACE" if space == "si_neural" else "3D UMAP SPACE"
        report_lines.extend(["=" * 80, f"DATA SPACE: {space_name}", "=" * 80])

        for beh in TARGET_BEHAVIORS:
            sub = df[df["behavior"] == beh].dropna(subset=[space]).copy()
            report_lines.append(f"\n>>> TARGET BEHAVIOR: {beh.upper()} <<<")

            # 1. Two-Way ANOVA: Cluster x Chunk
            try:
                model = smf.ols(f"{space} ~ C(cluster) + C(chunk) + C(cluster):C(chunk)", data=sub).fit()
                aov = sm.stats.anova_lm(model, typ=2)
                se = float(aov.loc["Residual", "sum_sq"])
                de = float(aov.loc["Residual", "df"])

                report_lines.append("Two-Way ANOVA: Cluster x Chunk (Type II SS):")
                for term in aov.index:
                    if term == "Residual":
                        continue
                    ss = float(aov.loc[term, "sum_sq"])
                    df_num = float(aov.loc[term, "df"])
                    f_val = float(aov.loc[term, "F"])
                    p_val = float(aov.loc[term, "PR(>F)"])
                    eta_p2 = ss / (ss + se) if (ss + se) > 0 else np.nan
                    report_lines.append(
                        f"  {term:25s} | F({int(df_num)}, {int(de)})={f_val:7.3f} | p={p_val:.4e} | eta_p^2={eta_p2:.4f}")
            except Exception as exc:
                report_lines.append(f"ANOVA Failed: {exc}")

            # 2. Early vs Late Epoch Test (Main Effect of Drift)
            sub_el = sub[sub["epoch"].isin(["early", "late"])].copy()
            v_early = sub_el.loc[sub_el["epoch"] == "early", space].values
            v_late = sub_el.loc[sub_el["epoch"] == "late", space].values
            if len(v_early) >= 2 and len(v_late) >= 2:
                t_stat, p_el = stats.ttest_ind(v_early, v_late, equal_var=False)
                report_lines.append(
                    f"Early vs. Late Epoch Stability: Diff={np.mean(v_early) - np.mean(v_late):+.4f} | t={t_stat:.3f} | p={p_el:.4e}")

    report_path = STATS_DIR / "controlled_si_statistical_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[OK] Saved statistical report -> {report_path}")

    # Plot Cluster x Chunk Trajectories
    plot_chunk_profiles(df)


def plot_chunk_profiles(df: pd.DataFrame):
    clusters = ["position", "time", "direction", "all_cells"]
    for space in ["si_neural", "si_umap"]:
        space_name = "Neural Space" if space == "si_neural" else "3D UMAP Space"
        for beh in TARGET_BEHAVIORS:
            sub = df[df["behavior"] == beh].copy()
            fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)

            for c in clusters:
                c_df = sub[sub["cluster"] == c]
                if c_df.empty:
                    continue
                stats_chunk = c_df.groupby("chunk_idx")[space].agg(["mean", "sem"]).reset_index()
                ax.errorbar(
                    stats_chunk["chunk_idx"], stats_chunk["mean"], yerr=stats_chunk["sem"],
                    label=c, color=CLUSTER_COLORS.get(c, "black"), lw=2, marker="o", capsize=3
                )

            ax.set_xlabel("Session Quintile Chunk (Early -> Late)", fontsize=10, fontweight="bold")
            ax.set_ylabel(f"Structure Index ({space_name})", fontsize=10, fontweight="bold")
            ax.set_title(f"Controlled SI ({space_name}) | {beh.upper()} | Last Session", fontsize=11, fontweight="bold")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xticklabels(["Q1\n(0-20%)", "Q2\n(20-40%)", "Q3\n(40-60%)", "Q4\n(60-80%)", "Q5\n(80-100%)"])
            ax.legend(frameon=False, loc="best")
            ax.grid(False)
            ax.spines[["top", "right"]].set_visible(False)

            fig.tight_layout()
            out_base = PLOT_DIR / f"controlled_si_{space}_{beh}_quintiles"
            fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
            fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Saved quintile profile figure -> {out_base.name}")


if __name__ == "__main__":
    run_controlled_analysis()