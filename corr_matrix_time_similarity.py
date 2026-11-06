#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Multi-Method Neural Drift Analysis Pipeline:
1. Correlation-Matrix Drift
2. Element-by-Element Timepoint PV Correlation Drift
3. Representational Similarity Analysis (RSA) Drift
4. Procrustes Distance / Manifold Shape Matching Drift

Includes:
- Automated path remapping & strict session-name condition parsing (fixing lt vs rot separation).
- Intervention parameter evaluation (excluding 'combined', grouping by int_i/int_f values).
- Separated Lag Bar Plots (with individual animal points) and Regression Line Plots.
- Parameter Bar Comparisons with 2-way ANOVA, Tukey HSD Post-Hoc, and safe Significance Brackets (*, **, ***).
"""

import os
import csv
import re
import pickle
import timeit
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, ttest_ind, mannwhitneyu
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# PATHS & CONFIGURATION
# =============================================================================

CSV_PATH = (
    "/home/hippolab/Documents/melisa/time_project_standarization/"
    "data/metadata/master_dataset_index.csv"
)

OUTPUT_DIR = (
    "/home/hippolab/Documents/melisa/time_project_standarization/analysis/"
    "Comprehensive_Drift_Analysis_Output"
)

SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
STATS_CSV_DIR = os.path.join(CSV_DIR, "statistics")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
ROW_HEATMAP_DIR = os.path.join(FIG_DIR, "row_heatmaps")

# =============================================================================
# PARAMETERS
# =============================================================================

MAX_ROWS = None
DATASET_FILTER = None

N_CHUNKS = 5
MIN_TIMEPOINTS_PER_CHUNK = 10
MIN_NEURONS = 5
ZSCORE_CELLS = True
REMOVE_MOV_DIR_ZERO = True
EXCLUDE_ZERO_LAG_IN_REGRESSION = True
EXCLUDE_ZERO_LAG_IN_PLOTS = True

HEATMAP_VMIN = -1.0
HEATMAP_VMAX = 1.0

DEEP_COLOR = "#D4A928"
SUPERFICIAL_COLOR = "#A020F0"
WHOLE_POP_COLOR = "#666666"


def get_learning_gray_palette(n_sessions):
    if n_sessions <= 1:
        return ["#666666"]
    vals = np.linspace(0.25, 0.80, n_sessions)
    return [str(v) for v in vals]


# =============================================================================
# HELPERS & DIRECTORIES
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ensure_all_dirs():
    for path in [
        OUTPUT_DIR, SESSION_PKL_DIR, CSV_DIR, STATS_CSV_DIR,
        LOG_DIR, FIG_DIR, ROW_HEATMAP_DIR,
    ]:
        ensure_dir(path)


def init_csv(csv_path, columns):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_row(csv_path, columns, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def append_many_csv_rows(csv_path, columns, rows):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        for row in rows:
            writer.writerow(row)


def load_pickle_standard(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle_standard(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def savefig_png_svg(fig, out_base):
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def sanitize_filename(text):
    text = str(text)
    return "".join([ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in text])


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


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# =============================================================================
# METADATA & STRICT CONDITION PARSING
# =============================================================================

def normalize_depth(depth):
    depth = str(depth).lower()
    if depth in ["sup", "superficial"] or "super" in depth:
        return "superficial"
    if "deep" in depth:
        return "deep"
    if "whole" in depth:
        return "whole_population"
    return depth


def infer_condition(row):
    dataset = str(row.get("dataset_name", "")).lower()
    session = str(row.get("session_name", "")).lower()

    # Strict separation for lt_rot dataset based solely on session_name patterns
    if "lt_rot" in dataset:
        if "_rot_" in session or session.endswith("_rot") or "rot_" in session:
            return "lt_rot_rot"
        if "_lt_" in session or session.endswith("_lt") or "lt_" in session:
            return "lt_rot_lt"

    if "dreadds" in dataset or "dreads" in dataset:
        path = str(row.get("standardized_file_path", "")).lower()
        text = f"{session} {path}"
        if "cno" in text: return "cno"
        if "saline" in text: return "saline"
        if "veh" in text or "vehicle" in text: return "vehicle"
        return "dreadds"
    if "egrin" in dataset:
        path = str(row.get("standardized_file_path", "")).lower()
        text = f"{session} {path}"
        if "pre" in text: return "pre"
        if "post" in text: return "post"
        if "stim" in text: return "stim"
        return "egrin"

    return session


def extract_learning_datetime_from_session(session_name):
    text = str(session_name)
    match = re.search(r"(\d{8})[_\-]?(\d{6})", text)
    if match is not None:
        try:
            return pd.to_datetime(match.group(1) + match.group(2), format="%Y%m%d%H%M%S")
        except Exception:
            pass
    match = re.search(r"(\d{8})", text)
    if match is not None:
        try:
            return pd.to_datetime(match.group(1), format="%Y%m%d")
        except Exception:
            pass
    try:
        return pd.to_datetime(text)
    except Exception:
        return pd.NaT


def build_learning_session_mapping_by_mouse(matrix_df):
    session_df = matrix_df[["mouse_name", "session_name"]].drop_duplicates().copy()
    session_df["session_datetime"] = session_df["session_name"].apply(extract_learning_datetime_from_session)
    session_df = session_df.sort_values(["mouse_name", "session_datetime", "session_name"],
                                        na_position="last").reset_index(drop=True)
    session_df["session_order"] = session_df.groupby("mouse_name").cumcount() + 1
    session_df["learning_session"] = "session_" + session_df["session_order"].astype(str)
    return session_df


# =============================================================================
# DATA LOADING & DRIFT COMPUTATION METHODS
# =============================================================================

def load_standardized_row(row_idx, row):
    standardized_file_path = row["standardized_file_path"]
    if isinstance(standardized_file_path, str):
        if "/home/melisamc/Documents/time_project_standarization" in standardized_file_path:
            standardized_file_path = standardized_file_path.replace(
                "/home/melisamc/Documents/time_project_standarization",
                "/home/hippolab/Documents/melisa/time_project_standarization"
            )

    if not os.path.isfile(standardized_file_path):
        raise FileNotFoundError(f"Missing standardized file for row {row_idx}: {standardized_file_path}")

    data = load_pickle_standard(standardized_file_path)
    required_keys = ["signal", "pos", "mov_dir", "speed", "time"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Row {row_idx}: missing key '{key}'")

    signal = ensure_2d(data["signal"])
    pos = ensure_1d(get_pos_1d(data["pos"]))
    mov_dir = ensure_1d(data["mov_dir"])
    speed = ensure_1d(data["speed"])
    time = ensure_1d(data["time"])

    n_time = signal.shape[0]

    def check_len(x, name):
        if len(x) != n_time:
            raise ValueError(f"Row {row_idx}: length mismatch for {name}")
        return x

    behavior = {
        "pos": check_len(pos, "pos"),
        "mov_dir": check_len(mov_dir, "mov_dir"),
        "speed": check_len(speed, "speed"),
        "time": check_len(time, "time"),
    }
    return signal.astype(float), behavior


def zscore_activity_cells(signal):
    mu = np.nanmean(signal, axis=0, keepdims=True)
    sd = np.nanstd(signal, axis=0, keepdims=True)
    sd[sd == 0] = np.nan
    z = (signal - mu) / sd
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def split_indices_equal(n_time, n_chunks):
    return [x for x in np.array_split(np.arange(n_time), n_chunks)]


def compute_corr_matrix_drift(signal):
    if ZSCORE_CELLS:
        signal = zscore_activity_cells(signal)
    chunk_indices = split_indices_equal(signal.shape[0], N_CHUNKS)
    corr_vecs = []
    for idx in chunk_indices:
        signal_chunk = signal[idx, :]
        if signal_chunk.shape[0] < MIN_TIMEPOINTS_PER_CHUNK or signal_chunk.shape[1] < MIN_NEURONS:
            corr_vec = np.array([])
        else:
            corr_mat = np.corrcoef(signal_chunk, rowvar=False)
            corr_vec = corr_mat[np.triu_indices(corr_mat.shape[0], k=1)]
        corr_vecs.append(corr_vec)

    similarity_matrix = np.full((N_CHUNKS, N_CHUNKS), np.nan)
    for i in range(N_CHUNKS):
        for j in range(N_CHUNKS):
            va, vb = corr_vecs[i], corr_vecs[j]
            valid = np.isfinite(va) & np.isfinite(vb)
            if np.sum(valid) >= 3 and np.nanstd(va[valid]) > 0 and np.nanstd(vb[valid]) > 0:
                similarity_matrix[i, j] = float(np.corrcoef(va[valid], vb[valid])[0, 1])
    return similarity_matrix, chunk_indices


def compute_elementwise_pv_drift(signal):
    if ZSCORE_CELLS:
        signal = zscore_activity_cells(signal)
    chunk_indices = split_indices_equal(signal.shape[0], N_CHUNKS)
    similarity_matrix = np.full((N_CHUNKS, N_CHUNKS), np.nan)
    for i in range(N_CHUNKS):
        idx_i = chunk_indices[i]
        if len(idx_i) < 2: continue
        chunk_i_sig = signal[idx_i, :]
        for j in range(N_CHUNKS):
            idx_j = chunk_indices[j]
            if len(idx_j) < 2: continue
            chunk_j_sig = signal[idx_j, :]
            corr_full = np.corrcoef(chunk_i_sig, chunk_j_sig, rowvar=True)
            T_i = chunk_i_sig.shape[0]
            sub_corr = corr_full[:T_i, T_i:]
            valid_vals = sub_corr[np.isfinite(sub_corr)]
            if len(valid_vals) > 0:
                similarity_matrix[i, j] = float(np.nanmean(valid_vals))
    return similarity_matrix, chunk_indices


def compute_rsa_drift(signal):
    if ZSCORE_CELLS:
        signal = zscore_activity_cells(signal)
    chunk_indices = split_indices_equal(signal.shape[0], N_CHUNKS)
    rdms = []
    T_fixed = 20
    for idx in chunk_indices:
        chunk_sig = signal[idx, :]
        if chunk_sig.shape[0] < MIN_TIMEPOINTS_PER_CHUNK:
            rdms.append(None)
            continue
        indices = np.linspace(0, chunk_sig.shape[0] - 1, T_fixed).astype(int)
        resampled = chunk_sig[indices, :]
        rdm = np.corrcoef(resampled, rowvar=True)
        rdm_vec = rdm[np.triu_indices(T_fixed, k=1)]
        rdms.append(rdm_vec)

    similarity_matrix = np.full((N_CHUNKS, N_CHUNKS), np.nan)
    for i in range(N_CHUNKS):
        for j in range(N_CHUNKS):
            ra, rb = rdms[i], rdms[j]
            if ra is not None and rb is not None:
                valid = np.isfinite(ra) & np.isfinite(rb)
                if np.sum(valid) >= 3 and np.nanstd(ra[valid]) > 0 and np.nanstd(rb[valid]) > 0:
                    similarity_matrix[i, j] = float(np.corrcoef(ra[valid], rb[valid])[0, 1])
    return similarity_matrix, chunk_indices


def compute_procrustes_drift(signal):
    if ZSCORE_CELLS:
        signal = zscore_activity_cells(signal)
    pca = PCA(n_components=min(3, signal.shape[1]))
    embedding = pca.fit_transform(signal)
    chunk_indices = split_indices_equal(signal.shape[0], N_CHUNKS)
    chunk_trajectories = []
    T_fixed = 20
    for idx in chunk_indices:
        traj = embedding[idx, :]
        if traj.shape[0] < MIN_TIMEPOINTS_PER_CHUNK:
            chunk_trajectories.append(None)
            continue
        indices = np.linspace(0, traj.shape[0] - 1, T_fixed).astype(int)
        resampled_traj = traj[indices, :]
        chunk_trajectories.append(resampled_traj)

    similarity_matrix = np.full((N_CHUNKS, N_CHUNKS), np.nan)
    for i in range(N_CHUNKS):
        for j in range(N_CHUNKS):
            ta, tb = chunk_trajectories[i], chunk_trajectories[j]
            if ta is not None and tb is not None:
                try:
                    _, _, disparity = procrustes(ta, tb)
                    similarity_matrix[i, j] = float(max(0.0, 1.0 - disparity))
                except Exception:
                    pass
    return similarity_matrix, chunk_indices


def matrix_to_lag_summary(mat):
    out = {}
    n = mat.shape[0]
    for lag in range(n):
        vals = [mat[i, j] for i in range(n) for j in range(n) if abs(i - j) == lag]
        out[lag] = float(np.nanmean(vals)) if vals else np.nan
    return out


def fit_linear_regression(lag_summary):
    lags = sorted(
        [k for k, v in lag_summary.items() if not (EXCLUDE_ZERO_LAG_IN_REGRESSION and k == 0) and np.isfinite(v)])
    vals = [lag_summary[k] for k in lags]
    if len(lags) < 2:
        return np.nan, np.nan, np.nan
    slope, intercept, _, _, _ = linregress(lags, vals)
    return slope, intercept, 0.0


def mean_offdiagonal(mat):
    vals = [mat[i, j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i != j]
    return float(np.nanmean(vals)) if vals else np.nan


# =============================================================================
# STATISTICAL TESTS, ANOVA & POSTHOC
# =============================================================================

def run_2way_anova_with_posthoc(lag_df, group_col, value_col, out_stat_dir, prefix):
    clean_df = lag_df[[group_col, "lag", value_col]].dropna()
    if len(clean_df) < 5 or clean_df[group_col].nunique() < 2:
        return None
    try:
        model = ols(f"{value_col} ~ C({group_col}) + C(lag) + C({group_col}):C(lag)", data=clean_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_table.to_csv(os.path.join(out_stat_dir, f"{prefix}_2way_anova.csv"))

        tukey = pairwise_tukeyhsd(endog=clean_df[value_col], groups=clean_df[group_col], alpha=0.05)
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tukey_df.to_csv(os.path.join(out_stat_dir, f"{prefix}_tukey_hsd.csv"), index=False)
        return anova_table
    except Exception:
        return None


def run_group_statistical_comparison(df, group_col, value_col, out_csv_path):
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return
    g1, g2 = groups[0], groups[1]
    vals1 = df[df[group_col] == g1][value_col].dropna().values
    vals2 = df[df[group_col] == g2][value_col].dropna().values
    if len(vals1) < 2 or len(vals2) < 2:
        return

    t_stat, t_pval = ttest_ind(vals1, vals2, equal_var=False)
    mwu_stat, mwu_pval = mannwhitneyu(vals1, vals2, alternative="two-sided")

    stat_results = [{
        "comparison_variable": value_col,
        "group_1": str(g1), "mean_1": float(np.mean(vals1)), "std_1": float(np.std(vals1)), "n_1": len(vals1),
        "group_2": str(g2), "mean_2": float(np.mean(vals2)), "std_2": float(np.std(vals2)), "n_2": len(vals2),
        "welch_t_stat": float(t_stat), "welch_t_pval": float(t_pval),
        "mann_whitney_u": float(mwu_stat), "mann_whitney_pval": float(mwu_pval)
    }]
    init_csv(out_csv_path, stat_results[0].keys())
    append_many_csv_rows(out_csv_path, stat_results[0].keys(), stat_results)


# =============================================================================
# PLOTTING HELPERS WITH SAFE BOUNDS & SIGNIFICANCE BARS
# =============================================================================

def plot_similarity_matrix(mat, title, out_base):
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(np.clip(mat, HEATMAP_VMIN, HEATMAP_VMAX), vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, origin="upper",
                   aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Global chunk")
    ax.set_ylabel("Global chunk")
    ax.set_xticks(np.arange(N_CHUNKS))
    ax.set_yticks(np.arange(N_CHUNKS))
    ax.set_xticklabels([str(i + 1) for i in range(N_CHUNKS)], fontsize=6)
    ax.set_yticklabels([str(i + 1) for i in range(N_CHUNKS)], fontsize=6)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Similarity / Correlation")
    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def add_significance_bracket(ax, x1, x2, y, text, h=0.02):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
    ax.text((x1 + x2) * 0.5, y + h, text, ha="center", va="bottom", color="black", fontsize=10)


def plot_lag_barplot_with_points(lag_df, group_col, title, out_base, palette_dict=None, is_gray_scale=False):
    groups = lag_df[group_col].dropna().unique()
    if len(groups) == 0:
        return

    lags = sorted(lag_df["lag"].dropna().unique())
    if EXCLUDE_ZERO_LAG_IN_PLOTS:
        lags = [l for l in lags if l > 0]

    stats = lag_df.groupby([group_col, "lag"])["similarity"].agg(["mean", "sem"]).reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = get_learning_gray_palette(len(groups)) if is_gray_scale else [palette_dict.get(str(g), "#1f77b4") for g in
                                                                           groups]

    x = np.arange(len(lags))
    width = 0.8 / len(groups)

    for idx, g in enumerate(groups):
        sub_stats = stats[stats[group_col] == g]
        means = [sub_stats[sub_stats["lag"] == l]["mean"].values[0] if l in sub_stats["lag"].values else np.nan for l in
                 lags]
        sems = [sub_stats[sub_stats["lag"] == l]["sem"].values[0] if l in sub_stats["lag"].values else 0.0 for l in
                lags]

        offset = (idx - (len(groups) - 1) / 2) * width
        color = colors[idx]

        ax.bar(x + offset, means, width=width, yerr=sems, capsize=3, color=color, edgecolor="black", linewidth=0.7,
               label=str(g), alpha=0.85)

        for lag_idx, l in enumerate(lags):
            indiv_vals = lag_df[(lag_df[group_col] == g) & (lag_df["lag"] == l)]["similarity"].dropna().values
            x_pts = np.random.normal(x[lag_idx] + offset, 0.02, size=len(indiv_vals))
            ax.scatter(x_pts, indiv_vals, color="black", s=12, alpha=0.5, zorder=5)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("|Δ chunk| (Lag)", fontsize=10)
    ax.set_ylabel("Similarity / Correlation", fontsize=10)
    ax.set_ylim(-0.1, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in lags])
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_regression_lines(session_df, lag_df, group_col, title, out_base, palette_dict=None, is_gray_scale=False):
    groups = session_df[group_col].dropna().unique()
    if len(groups) == 0:
        return

    lags = sorted(lag_df["lag"].dropna().unique())
    if EXCLUDE_ZERO_LAG_IN_REGRESSION:
        lags = [l for l in lags if l > 0]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    colors = get_learning_gray_palette(len(groups)) if is_gray_scale else [palette_dict.get(str(g), "#1f77b4") for g in
                                                                           groups]

    for idx, g in enumerate(groups):
        color = colors[idx]
        group_sessions = session_df[session_df[group_col] == g]

        for _, s_row in group_sessions.iterrows():
            slope = s_row["regression_slope"]
            intercept = s_row["regression_intercept"]
            if np.isfinite(slope) and np.isfinite(intercept):
                y_fit = [slope * l + intercept for l in lags]
                ax.plot(lags, y_fit, color=color, alpha=0.2, linewidth=1.0)

        mean_slope = group_sessions["regression_slope"].mean()
        mean_intercept = group_sessions["regression_intercept"].mean()
        if np.isfinite(mean_slope) and np.isfinite(mean_intercept):
            y_mean_fit = [mean_slope * l + mean_intercept for l in lags]
            ax.plot(lags, y_mean_fit, color=color, linewidth=2.8,
                    label=f"{g} (Mean Fit: y={mean_slope:.3f}x+{mean_intercept:.3f})")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("|Δ chunk| (Lag)", fontsize=10)
    ax.set_ylabel("Similarity / Correlation (Regression Fit)", fontsize=10)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticks(lags)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    savefig_png_svg(fig, out_base)


def plot_parameter_bar_comparison(session_df, group_col, param_col, title, out_base, p_val=None, palette_dict=None,
                                  is_gray_scale=False):
    groups = session_df[group_col].dropna().unique()
    if len(groups) == 0:
        return

    means = session_df.groupby(group_col)[param_col].mean()
    sems = session_df.groupby(group_col)[param_col].sem()

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    x = np.arange(len(groups))
    if is_gray_scale:
        colors = get_learning_gray_palette(len(groups))
    else:
        colors = [palette_dict.get(str(g), "#1f77b4") for g in
                  groups] if palette_dict is not None else ["#1f77b4"] * len(groups)

    ax.bar(x, means, yerr=sems, capsize=4, color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    for idx, g in enumerate(groups):
        pts = session_df[session_df[group_col] == g][param_col].dropna().values
        x_pts = np.random.normal(x[idx], 0.03, size=len(pts))
        ax.scatter(x_pts, pts, color="black", s=15, alpha=0.6, zorder=5)

    if len(groups) == 2 and p_val is not None and np.isfinite(p_val):
        max_val = np.nanmax(means + sems)
        if not np.isfinite(max_val) or max_val <= 0:
            max_val = 1.0
        y_max = max_val * 1.15
        sig_text = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        add_significance_bracket(ax, 0, 1, y_max, f"p={p_val:.3f} ({sig_text})")
        ax.set_ylim(min(0, min(means) * 1.2), y_max * 1.25)

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups], rotation=15)
    ax.set_ylabel(f"Mean {param_col.replace('_', ' ').title()}", fontsize=10)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    savefig_png_svg(fig, out_base)


# =============================================================================
# ROW PROCESSING
# =============================================================================

def process_row(row_idx, row, analysis_type="cormat"):
    signal, behavior = load_standardized_row(row_idx, row)
    n_timepoints_original = signal.shape[0]
    n_neurons = signal.shape[1]

    if n_neurons < MIN_NEURONS:
        raise ValueError(f"Too few neurons: {n_neurons}")

    valid_idx = behavior["mov_dir"] != 0 if REMOVE_MOV_DIR_ZERO else np.ones(n_timepoints_original, dtype=bool)
    signal_valid = signal[valid_idx, :]

    if signal_valid.shape[0] < N_CHUNKS * MIN_TIMEPOINTS_PER_CHUNK:
        raise ValueError(f"Too few timepoints after filtering")

    if analysis_type == "cormat":
        similarity_matrix, _ = compute_corr_matrix_drift(signal_valid)
    elif analysis_type == "pv_elementwise":
        similarity_matrix, _ = compute_elementwise_pv_drift(signal_valid)
    elif analysis_type == "rsa":
        similarity_matrix, _ = compute_rsa_drift(signal_valid)
    elif analysis_type == "procrustes":
        similarity_matrix, _ = compute_procrustes_drift(signal_valid)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    lag_summary = matrix_to_lag_summary(similarity_matrix)
    slope, intercept, pval = fit_linear_regression(lag_summary)

    dataset_name = row["dataset_name"]
    mouse_name = row["mouse_name"]
    session_name = row["session_name"]
    depth_type = normalize_depth(row["depth_type"])
    condition = infer_condition(row)

    int_i_val = row.get("int_i", np.nan)
    int_f_val = row.get("int_f", np.nan)

    session_row = {
        "dataset_name": dataset_name, "mouse_name": mouse_name, "session_name": session_name,
        "depth_type": depth_type, "condition": condition,
        "int_i": int_i_val, "int_f": int_f_val,
        "standardized_file_path": row["standardized_file_path"],
        "row_index": row_idx, "n_chunks": N_CHUNKS, "n_neurons": n_neurons,
        "regression_slope": slope, "regression_intercept": intercept, "regression_pval": pval,
    }

    lag_rows = [{
        "dataset_name": dataset_name, "mouse_name": mouse_name, "session_name": session_name,
        "depth_type": depth_type, "condition": condition,
        "int_i": int_i_val, "int_f": int_f_val, "row_index": row_idx,
        "lag": lag, "similarity": val
    } for lag, val in lag_summary.items()]

    return session_row, lag_rows


# =============================================================================
# AGGREGATIONS & STATISTICAL COMPARISONS
# =============================================================================

def run_dataset_aggregations(session_df, lag_df, metric_prefix):
    prefix_dir = os.path.join(FIG_DIR, metric_prefix)
    ensure_dir(prefix_dir)

    def evaluate_and_plot_param(sub_df, group_col, param_col, dataset_title_tag, out_sub, palette, is_gray=False):
        groups = sub_df[group_col].dropna().unique()
        p_val = None
        if len(groups) == 2:
            g1, g2 = groups[0], groups[1]
            v1 = sub_df[sub_df[group_col] == g1][param_col].dropna().values
            v2 = sub_df[sub_df[group_col] == g2][param_col].dropna().values
            if len(v1) >= 2 and len(v2) >= 2:
                _, p_val = ttest_ind(v1, v2, equal_var=False)

        plot_parameter_bar_comparison(
            sub_df, group_col, param_col,
            f"{dataset_title_tag}: {param_col.replace('_', ' ').title()}",
            os.path.join(out_sub, f"{sanitize_filename(dataset_title_tag)}_{param_col}_barplot"),
            p_val=p_val, palette_dict=palette, is_gray_scale=is_gray
        )
        run_group_statistical_comparison(
            sub_df, group_col, param_col,
            os.path.join(STATS_CSV_DIR, f"{metric_prefix}_{sanitize_filename(dataset_title_tag)}_{param_col}_stats.csv")
        )

    # 1. Learning
    learning_sub = session_df[session_df["dataset_name"].str.lower().str.contains("learning")].copy()
    if len(learning_sub) > 0:
        mapping_df = build_learning_session_mapping_by_mouse(learning_sub)
        learning_sub = learning_sub.merge(mapping_df[["mouse_name", "session_name", "learning_session"]],
                                          on=["mouse_name", "session_name"], how="left")

        lag_learning = lag_df[lag_df["row_index"].isin(learning_sub["row_index"].unique())].copy()
        lag_learning = lag_learning.merge(learning_sub[["row_index", "learning_session"]], on="row_index", how="inner")

        out_sub = os.path.join(prefix_dir, "learning")
        ensure_dir(out_sub)
        plot_lag_barplot_with_points(lag_learning, "learning_session", "Learning: Lag Bar Plot with Animal Points",
                                     os.path.join(out_sub, "learning_lag_barplot"), is_gray_scale=True)
        plot_regression_lines(learning_sub, lag_learning, "learning_session", "Learning: Animal & Mean Regressions",
                              os.path.join(out_sub, "learning_regression_lines"), is_gray_scale=True)

        for param in ["regression_slope", "regression_intercept"]:
            evaluate_and_plot_param(learning_sub[learning_sub["learning_session"].isin(["session_1", "session_4"])],
                                    "learning_session", param, "Learning_S1_vs_S4", out_sub, None, is_gray=True)

        run_2way_anova_with_posthoc(lag_learning, "learning_session", "similarity", STATS_CSV_DIR,
                                    f"{metric_prefix}_learning")

    # 2. lt_rot (Explicitly processing BOTH lt_rot_lt and lt_rot_rot independently, restricting to deep & superficial)
    for cond in ["lt_rot_lt", "lt_rot_rot"]:
        cond_data = session_df[
            (session_df["condition"] == cond) & (session_df["depth_type"].isin(["deep", "superficial"]))].copy()
        if len(cond_data) > 0:
            lag_cond = lag_df[lag_df["row_index"].isin(cond_data["row_index"].unique())].copy()

            out_sub = os.path.join(prefix_dir, cond)
            ensure_dir(out_sub)
            palette = {"deep": DEEP_COLOR, "superficial": SUPERFICIAL_COLOR}

            plot_lag_barplot_with_points(lag_cond[lag_cond["row_index"].isin(cond_data["row_index"])], "depth_type",
                                         f"{cond.upper()}: Lag Bar Plot", os.path.join(out_sub, f"{cond}_lag_barplot"),
                                         palette_dict=palette)
            plot_regression_lines(cond_data, lag_cond, "depth_type", f"{cond.upper()}: Regressions",
                                  os.path.join(out_sub, f"{cond}_regression_lines"), palette_dict=palette)

            for param in ["regression_slope", "regression_intercept"]:
                evaluate_and_plot_param(cond_data, "depth_type", param, f"{cond}_deep_vs_sup", out_sub, palette)

            run_2way_anova_with_posthoc(lag_cond, "depth_type", "similarity", STATS_CSV_DIR, f"{metric_prefix}_{cond}")

    # 3. Intervention (Excluding sessions containing 'combined', evaluating specific int_i and int_f numerical parameter values)
    interv_sub = session_df[session_df["dataset_name"].str.lower().str.contains("intervention")].copy()
    if len(interv_sub) > 0:
        if "session_name" in interv_sub.columns:
            interv_sub = interv_sub[~interv_sub["session_name"].astype(str).str.lower().str.contains("combined")].copy()

        out_sub = os.path.join(prefix_dir, "intervention")
        ensure_dir(out_sub)

        if "int_i" in interv_sub.columns and interv_sub["int_i"].nunique() > 0:
            interv_i_sub = interv_sub.copy()
            interv_i_sub["int_i_str"] = interv_i_sub["int_i"].astype(str)
            lag_interv_i = lag_df[lag_df["row_index"].isin(interv_i_sub["row_index"].unique())].copy()
            lag_interv_i = lag_interv_i.merge(interv_i_sub[["row_index", "int_i_str"]], on="row_index", how="inner")

            unique_i_vals = sorted(interv_i_sub["int_i_str"].dropna().unique())
            palette_i = {val: plt.cm.tab10(i % 10) for i, val in enumerate(unique_i_vals)}

            plot_lag_barplot_with_points(lag_interv_i, "int_i_str",
                                         "Intervention (int_i parameter values): Lag Bar Plot",
                                         os.path.join(out_sub, "intervention_int_i_lag_barplot"),
                                         palette_dict=palette_i)
            plot_regression_lines(interv_i_sub, lag_interv_i, "int_i_str",
                                  "Intervention (int_i parameter values): Regressions",
                                  os.path.join(out_sub, "intervention_int_i_regression_lines"), palette_dict=palette_i)
            run_2way_anova_with_posthoc(lag_interv_i, "int_i_str", "similarity", STATS_CSV_DIR,
                                        f"{metric_prefix}_intervention_int_i")

        if "int_f" in interv_sub.columns and interv_sub["int_f"].nunique() > 0:
            interv_f_sub = interv_sub.copy()
            interv_f_sub["int_f_str"] = interv_f_sub["int_f"].astype(str)
            lag_interv_f = lag_df[lag_df["row_index"].isin(interv_f_sub["row_index"].unique())].copy()
            lag_interv_f = lag_interv_f.merge(interv_f_sub[["row_index", "int_f_str"]], on="row_index", how="inner")

            unique_f_vals = sorted(interv_f_sub["int_f_str"].dropna().unique())
            palette_f = {val: plt.cm.Set2(i % 8) for i, val in enumerate(unique_f_vals)}

            plot_lag_barplot_with_points(lag_interv_f, "int_f_str",
                                         "Intervention (int_f parameter values): Lag Bar Plot",
                                         os.path.join(out_sub, "intervention_int_f_lag_barplot"),
                                         palette_dict=palette_f)
            plot_regression_lines(interv_f_sub, lag_interv_f, "int_f_str",
                                  "Intervention (int_f parameter values): Regressions",
                                  os.path.join(out_sub, "intervention_int_f_regression_lines"), palette_dict=palette_f)
            run_2way_anova_with_posthoc(lag_interv_f, "int_f_str", "similarity", STATS_CSV_DIR,
                                        f"{metric_prefix}_intervention_int_f")

    # 4. dreadds, dreads, egrin, lt_ephys_mice
    for ds_kw in ["dreadds", "dreads", "egrin", "lt_ephys_mice"]:
        ds_sub = session_df[session_df["dataset_name"].str.lower().str.contains(ds_kw)].copy()
        if len(ds_sub) > 0:
            lag_ds = lag_df[lag_df["row_index"].isin(ds_sub["row_index"].unique())].copy()

            out_sub = os.path.join(prefix_dir, ds_kw)
            ensure_dir(out_sub)
            palette = {"deep": DEEP_COLOR, "superficial": SUPERFICIAL_COLOR, "whole_population": WHOLE_POP_COLOR}

            plot_lag_barplot_with_points(lag_ds, "depth_type", f"{ds_kw}: Lag Bar Plot",
                                         os.path.join(out_sub, f"{ds_kw}_lag_barplot"), palette_dict=palette)
            plot_regression_lines(ds_sub, lag_ds, "depth_type", f"{ds_kw}: Regressions",
                                  os.path.join(out_sub, f"{ds_kw}_regression_lines"), palette_dict=palette)

            sub_ds_depths = ds_sub[ds_sub["depth_type"].isin(["deep", "superficial"])]
            for param in ["regression_slope", "regression_intercept"]:
                evaluate_and_plot_param(sub_ds_depths, "depth_type", param, f"{ds_kw}_deep_vs_sup", out_sub, palette)

            run_2way_anova_with_posthoc(lag_ds, "depth_type", "similarity", STATS_CSV_DIR, f"{metric_prefix}_{ds_kw}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(analysis_type="cormat"):
    prefixes = {
        "cormat": "CorrMatrix_Drift",
        "pv_elementwise": "PV_Elementwise_Drift",
        "rsa": "RSA_Drift",
        "procrustes": "Procrustes_Drift"
    }
    prefix = prefixes[analysis_type]
    print(f"\n==============================================================")
    print(f"Running Analysis: {prefix}")
    print(f"==============================================================")

    df = pd.read_csv(CSV_PATH)
    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    session_rows, lag_rows = [], []

    for row_idx, row in df.iterrows():
        try:
            s_row, l_rows = process_row(row_idx, row, analysis_type=analysis_type)
            session_rows.append(s_row)
            lag_rows.extend(l_rows)
        except Exception:
            continue

    session_df = pd.DataFrame(session_rows)
    lag_df = pd.DataFrame(lag_rows)

    if len(session_df) > 0:
        run_dataset_aggregations(session_df, lag_df, metric_prefix=prefix)
        session_df.to_csv(os.path.join(CSV_DIR, f"{prefix}_session_summary.csv"), index=False)
        lag_df.to_csv(os.path.join(CSV_DIR, f"{prefix}_lag_summary.csv"), index=False)


def main():
    ensure_all_dirs()
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    log_file = open(os.path.join(LOG_DIR, "comprehensive_drift_analysis_log.txt"), "w", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    # Run all 4 methods: C-Matrix Drift, Elementwise PV, RSA, and Procrustes Manifold Shape Matching
    run_pipeline(analysis_type="cormat")
    run_pipeline(analysis_type="pv_elementwise")
    run_pipeline(analysis_type="rsa")
    run_pipeline(analysis_type="procrustes")

    print("==============================================================")
    print(f"All analyses completed successfully. Results saved in: {OUTPUT_DIR}")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()