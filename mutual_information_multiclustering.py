#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run MIR-based clustering and openTSNE inside each dataset independently.

Compared with the previous train-transfer script, this version:
1) Does NOT train on learning and transfer to others
2) For each dataset subset, fits its own:
   - row-wise z-scored MIR features
   - StandardScaler
   - KMeans
   - openTSNE
3) Keeps the same cluster semantic mapping:
   0 -> Time
   1 -> Direction
   2 -> Position
   10 -> Unassigned
4) Keeps the same colors and plotting style
5) Saves one labeled neuron CSV + plots + stats per subset

This is the "clustering_within_dataset" version.
"""

import os
import re
import json
import pickle
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore, ttest_ind, chi2_contingency, chisquare
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from openTSNE import TSNE

import statsmodels.api as sm
from statsmodels.formula.api import ols


# =============================================================================
# Paths
# =============================================================================

MI_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/mutual_information/mutual_information_per_neuron.csv"
MASTER_INDEX_CSV = "/home/melisamc/Documents/time_project_standarization/data/metadata/master_dataset_index.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/clustering_within_dataset"


# =============================================================================
# Parameters
# =============================================================================

MIR_FEATURE_COLUMNS = [
    "mi_position",
    "mi_position_x_direction",
    "mi_direction",
    "mi_speed",
    "mi_time",
    "mi_inner_time",
    "mi_trial_id",
]

MIR_FEATURE_TITLES = [
    "PositionMIR",
    "DirPositionMIR",
    "MovDirMIR",
    "SpeedMIR",
    "TimeMIR",
    "InnerTimeMIR",
    "TrialIDMIR",
]

N_CLUSTERS = 3
UNASSIGNED_LABEL = 10
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 50
KEEP_PERCENTILE = 75

TSNE_PERPLEXITY = 50
TSNE_RANDOM_STATE = 42
TSNE_N_JOBS = 8
TSNE_INITIALIZATION = "pca"
TSNE_METRIC = "euclidean"

SCATTER_SIZE = 40
FEATURE_CMAP = "coolwarm"
MOUSE_CMAP = "rainbow"

MIR_VMIN_RAW = 0.0
MIR_VMAX_RAW = 0.35
MIR_VMIN_Z = -1.0
MIR_VMAX_Z = 3.5

CLUSTER_COLOR_MAP = {
    2: "#ec8ef8ff",    # Position
    1: "#66cef4ff",    # Direction
    0: "#bce784ff",    # Time
    10: "#bbbcc0ff",   # Unassigned
}

CLUSTER_COLOR_MAP_STR = {
    "2": "#ec8ef8ff",
    "1": "#66cef4ff",
    "0": "#bce784ff",
    "10": "#bbbcc0ff",
}

CLUSTER_NAME_MAP = {
    0: "Time",
    1: "Direction",
    2: "Position",
    10: "Unassigned",
}

CLUSTER_NAME_MAP_STR = {
    "0": "Time",
    "1": "Direction",
    "2": "Position",
    "10": "Unassigned",
}

DEPTH_COLOR_MAP = {
    "superficial": "#9900ff",
    "deep": "#cc9900",
    "whole population": "gray",
    "whole_population": "gray",
    "whole": "gray",
    "all": "gray",
    "unknown": "gray",
}

# Same subset logic as before, but each subset is clustered internally
WITHIN_DATASET_CONFIGS = [
    {
        "name": "learning_latest_per_mouse",
        "dataset_name_contains": ["learning"],
        "subset_mode": "all",
        "depth_mode": "all",
        "latest_session_per_mouse": True,
    },
    {
        "name": "lt_rot_lt",
        "dataset_name_contains": ["lt_rot"],
        "subset_mode": "lt_only",
        "depth_mode": "all",
        "latest_session_per_mouse": False,
    },
    {
        "name": "lt_rot_rot",
        "dataset_name_contains": ["lt_rot"],
        "subset_mode": "rot_only",
        "depth_mode": "all",
        "latest_session_per_mouse": False,
    },
    {
        "name": "ephys_whole",
        "dataset_name_contains": ["ephys"],
        "subset_mode": "all",
        "depth_mode": "whole_only",
        "latest_session_per_mouse": False,
    },
    {
        "name": "ephys_deep_sup",
        "dataset_name_contains": ["ephys"],
        "subset_mode": "all",
        "depth_mode": "deep_sup_only",
        "latest_session_per_mouse": False,
    },
    {
        "name": "egrin_all",
        "dataset_name_contains": ["egrin"],
        "subset_mode": "all",
        "depth_mode": "all",
        "latest_session_per_mouse": False,
    },
    {
        "name": "dreadds_all",
        "dataset_name_contains": ["dreadds"],
        "subset_mode": "all",
        "depth_mode": "all",
        "latest_session_per_mouse": False,
    },
]


# =============================================================================
# Basic helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_pickle(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def sanitize_for_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def save_figure_both(fig, png_dir, svg_dir, out_base):
    ensure_dir(png_dir)
    ensure_dir(svg_dir)
    png_path = os.path.join(png_dir, out_base + ".png")
    svg_path = os.path.join(svg_dir, out_base + ".svg")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(svg_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")


def save_text_file(stats_dir, out_base, text):
    ensure_dir(stats_dir)
    out_path = os.path.join(stats_dir, out_base + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] Saved: {out_path}")


def normalize_depth_label(depth_value):
    if pd.isna(depth_value):
        return "unknown"
    depth = str(depth_value).strip().lower()
    if depth in ["sup", "superficial"]:
        return "superficial"
    if depth in ["deep"]:
        return "deep"
    if depth in ["whole", "whole population", "whole_population", "all", "all_cells"]:
        return "whole population"
    return depth


def parse_session_datetime_from_name(session_name):
    s = str(session_name)
    dt = pd.to_datetime(s, format="%Y%m%d_%H%M%S", errors="coerce")
    if pd.notna(dt):
        return dt
    match = re.search(r"(\d{8}_\d{6})", s)
    if match is not None:
        return pd.to_datetime(match.group(1), format="%Y%m%d_%H%M%S", errors="coerce")
    return pd.NaT


def attach_session_datetime(df):
    out = df.copy()
    out["session_datetime"] = out["session_name"].apply(parse_session_datetime_from_name)
    return out


def add_neuron_uid(df):
    out = df.copy()
    out["neuron_uid"] = (
        out["dataset_name"].astype(str) + "__" +
        out["mouse_name"].astype(str) + "__" +
        out["session_name"].astype(str) + "__" +
        out["depth_type"].astype(str) + "__row" +
        out["row_index"].astype(str) + "__neuron" +
        out["neuron_index"].astype(str)
    )
    return out


def rowwise_zscore(X):
    Xz = zscore(X, axis=1, nan_policy="omit")
    Xz = np.where(np.isfinite(Xz), Xz, 0.0)
    return Xz


def build_mouse_index(df):
    mice_sorted = sorted(df["mouse_name"].dropna().astype(str).unique().tolist())
    mouse_to_idx = {m: i for i, m in enumerate(mice_sorted)}
    out = df.copy()
    out["mouse_index"] = out["mouse_name"].astype(str).map(mouse_to_idx)
    return out, mice_sorted


def print_available_values(df_mi, df_master):
    print("\nAvailable dataset_name values in MI CSV:")
    print(sorted(df_mi["dataset_name"].dropna().astype(str).unique().tolist()))
    if "paradigm" in df_master.columns:
        print("\nAvailable paradigm values in master index:")
        print(sorted(df_master["paradigm"].dropna().astype(str).unique().tolist()))
    if "model" in df_master.columns:
        print("\nAvailable model values in master index:")
        print(sorted(df_master["model"].dropna().astype(str).unique().tolist()))


# =============================================================================
# Statistics helpers
# =============================================================================

def pval_to_star(pval):
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def summarize_series(x, name):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return f"{name}: n=0\n"
    return (
        f"{name}: n={len(x)}, mean={x.mean():.6f}, std={x.std(ddof=1) if len(x) > 1 else 0.0:.6f}, "
        f"median={x.median():.6f}, min={x.min():.6f}, max={x.max():.6f}\n"
    )


def run_cluster_count_gof_test(counts_df, cluster_col="cluster_str", count_col="count"):
    txt = []
    txt.append("Chi-square goodness-of-fit test for cluster counts\n")
    txt.append("Null hypothesis: equal counts across observed clusters.\n\n")

    if cluster_col not in counts_df.columns or count_col not in counts_df.columns:
        txt.append("Test not computed: missing required columns.\n")
        return "".join(txt)

    observed = counts_df[count_col].to_numpy(dtype=float)
    labels = counts_df[cluster_col].astype(str).tolist()

    if len(observed) < 2:
        txt.append("Test not computed: fewer than 2 clusters present.\n")
        return "".join(txt)

    total = observed.sum()
    if total <= 0:
        txt.append("Test not computed: total count is zero.\n")
        return "".join(txt)

    expected = np.ones_like(observed) * (total / len(observed))

    try:
        chi2, pval = chisquare(f_obs=observed, f_exp=expected)
        txt.append(f"Observed clusters: {labels}\n")
        txt.append(f"Observed counts: {observed.tolist()}\n")
        txt.append(f"Expected equal counts: {expected.tolist()}\n")
        txt.append(f"chi2={chi2:.6f}, p={pval:.6e}, dof={len(observed)-1}\n")
    except Exception as e:
        txt.append(f"Test failed: {e}\n")

    return "".join(txt)


def run_chi_square_counts(counts_df, row_var, col_var, count_var="count"):
    txt = []
    txt.append("Chi-square test of independence\n\n")

    if row_var not in counts_df.columns or col_var not in counts_df.columns or count_var not in counts_df.columns:
        txt.append("Test not computed: missing required columns.\n")
        return "".join(txt)

    pivot = counts_df.pivot_table(index=row_var, columns=col_var, values=count_var, fill_value=0)

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        txt.append("Test not computed: contingency table must be at least 2x2.\n")
        return "".join(txt)

    try:
        chi2, pval, dof, expected = chi2_contingency(pivot.values)
        txt.append(f"Rows: {row_var}\nColumns: {col_var}\n")
        txt.append(f"chi2={chi2:.6f}, p={pval:.6e}, dof={dof}\n\n")
        txt.append("Observed contingency table:\n")
        txt.append(pivot.to_string())
        txt.append("\n\nExpected contingency table:\n")
        txt.append(pd.DataFrame(expected, index=pivot.index, columns=pivot.columns).to_string())
        txt.append("\n")
    except Exception as e:
        txt.append(f"Test failed: {e}\n")

    return "".join(txt)


def run_oneway_ttests(df, group_col, value_col, title):
    txt = []
    txt.append(f"{title}\n")
    txt.append("=" * len(title) + "\n\n")

    groups = sorted(df[group_col].dropna().astype(str).unique().tolist())
    for g in groups:
        vals = df.loc[df[group_col].astype(str) == g, value_col]
        txt.append(summarize_series(vals, f"group={g}"))
    txt.append("\n")

    if len(groups) >= 2:
        try:
            model = ols(f"{value_col} ~ C({group_col})", data=df).fit()
            if model.df_resid > 0:
                anova_table = sm.stats.anova_lm(model, typ=2)
                txt.append("One-way ANOVA\n")
                txt.append(anova_table.to_string())
                txt.append("\n\n")
            else:
                txt.append("One-way ANOVA skipped: non-positive residual degrees of freedom.\n\n")
        except Exception as e:
            txt.append(f"One-way ANOVA failed: {e}\n\n")

        txt.append("Pairwise Welch t-tests\n")
        for g1, g2 in itertools.combinations(groups, 2):
            v1 = df.loc[df[group_col].astype(str) == g1, value_col].dropna()
            v2 = df.loc[df[group_col].astype(str) == g2, value_col].dropna()
            if len(v1) > 1 and len(v2) > 1:
                stat, pval = ttest_ind(v1, v2, equal_var=False)
                txt.append(
                    f"{g1} vs {g2}: t={stat:.6f}, p={pval:.6e}, star={pval_to_star(pval)}, "
                    f"n1={len(v1)}, n2={len(v2)}\n"
                )
            else:
                txt.append(f"{g1} vs {g2}: skipped (insufficient n)\n")
    return "".join(txt)


def run_two_way_anova_and_pairs(df, factor_a, factor_b, value_col, title):
    txt = []
    txt.append(f"{title}\n")
    txt.append("=" * len(title) + "\n\n")

    txt.append("Descriptive statistics by group\n")
    grouped = (
        df.groupby([factor_a, factor_b])[value_col]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    txt.append(grouped.to_string(index=False))
    txt.append("\n\n")

    levels_a = sorted(df[factor_a].dropna().astype(str).unique().tolist())
    levels_b = sorted(df[factor_b].dropna().astype(str).unique().tolist())

    enough_for_anova = (
        len(levels_a) >= 2 and
        len(levels_b) >= 2 and
        len(df) > (len(levels_a) * len(levels_b))
    )

    if enough_for_anova:
        try:
            model = ols(f"{value_col} ~ C({factor_a}) * C({factor_b})", data=df).fit()
            if model.df_resid > 0:
                anova_table = sm.stats.anova_lm(model, typ=2)
                txt.append("Two-way ANOVA\n")
                txt.append(anova_table.to_string())
                txt.append("\n\n")
            else:
                txt.append("Two-way ANOVA skipped: non-positive residual degrees of freedom.\n\n")
        except Exception as e:
            txt.append(f"Two-way ANOVA failed: {e}\n\n")
    else:
        txt.append("Two-way ANOVA skipped: insufficient group structure.\n\n")

    txt.append(f"Pairwise Welch t-tests within each {factor_b}\n")
    for b in levels_b:
        sub = df[df[factor_b].astype(str) == b].copy()
        for a1, a2 in itertools.combinations(levels_a, 2):
            v1 = sub.loc[sub[factor_a].astype(str) == a1, value_col].dropna()
            v2 = sub.loc[sub[factor_a].astype(str) == a2, value_col].dropna()
            if len(v1) > 1 and len(v2) > 1:
                stat, pval = ttest_ind(v1, v2, equal_var=False)
                txt.append(
                    f"{factor_b}={b} | {a1} vs {a2}: t={stat:.6f}, p={pval:.6e}, "
                    f"star={pval_to_star(pval)}, n1={len(v1)}, n2={len(v2)}\n"
                )
            else:
                txt.append(f"{factor_b}={b} | {a1} vs {a2}: skipped (insufficient n)\n")
    txt.append("\n")

    txt.append(f"Pairwise Welch t-tests within each {factor_a}\n")
    for a in levels_a:
        sub = df[df[factor_a].astype(str) == a].copy()
        for b1, b2 in itertools.combinations(levels_b, 2):
            v1 = sub.loc[sub[factor_b].astype(str) == b1, value_col].dropna()
            v2 = sub.loc[sub[factor_b].astype(str) == b2, value_col].dropna()
            if len(v1) > 1 and len(v2) > 1:
                stat, pval = ttest_ind(v1, v2, equal_var=False)
                txt.append(
                    f"{factor_a}={a} | {b1} vs {b2}: t={stat:.6f}, p={pval:.6e}, "
                    f"star={pval_to_star(pval)}, n1={len(v1)}, n2={len(v2)}\n"
                )
            else:
                txt.append(f"{factor_a}={a} | {b1} vs {b2}: skipped (insufficient n)\n")

    return "".join(txt)


# =============================================================================
# Metadata merge / filtering
# =============================================================================

def merge_master_metadata(df_mi, df_master):
    join_cols = ["dataset_name", "mouse_name", "session_name", "depth_type"]
    add_cols = [c for c in ["modality", "paradigm", "model", "created_at", "notes"] if c in df_master.columns]
    master_sub = df_master[join_cols + add_cols].drop_duplicates().copy()
    out = df_mi.merge(master_sub, on=join_cols, how="left")
    return out


def dataset_name_mask(df, substrings):
    mask = pd.Series(False, index=df.index)
    name_lower = df["dataset_name"].astype(str).str.lower()
    for ss in substrings:
        mask |= name_lower.str.contains(str(ss).lower(), na=False)
    return mask


def subset_lt_rot(df, mode):
    if mode == "all":
        return df.copy()

    sess = df["session_name"].astype(str).str.lower()

    if mode == "lt_only":
        return df[sess.str.contains("lt", na=False)].copy()

    if mode == "rot_only":
        return df[sess.str.contains("rot", na=False)].copy()

    return df.copy()


def subset_by_depth_mode(df, depth_mode):
    depth_norm = df["depth_type"].apply(normalize_depth_label)

    if depth_mode == "all":
        return df.copy()

    if depth_mode == "whole_only":
        return df[depth_norm == "whole population"].copy()

    if depth_mode == "deep_sup_only":
        return df[depth_norm.isin(["deep", "superficial"])].copy()

    return df.copy()


def select_latest_session_per_mouse(df):
    session_table = (
        df[["mouse_name", "session_name", "session_datetime"]]
        .drop_duplicates()
        .sort_values(["mouse_name", "session_datetime", "session_name"])
    )

    latest_per_mouse = (
        session_table.groupby("mouse_name", as_index=False)["session_datetime"]
        .max()
        .rename(columns={"session_datetime": "latest_datetime"})
    )

    session_table = session_table.merge(latest_per_mouse, on="mouse_name", how="left")

    keep_sessions = session_table[
        session_table["session_datetime"] == session_table["latest_datetime"]
    ][["mouse_name", "session_name"]].drop_duplicates()

    out = df.merge(keep_sessions, on=["mouse_name", "session_name"], how="inner")
    return out


# =============================================================================
# Within-dataset clustering models
# =============================================================================

def infer_cluster_semantic_mapping(df_subset, raw_keep_labels):
    tmp = df_subset.copy()
    tmp["raw_keep_label"] = raw_keep_labels
    tmp = tmp[tmp["raw_keep_label"] != UNASSIGNED_LABEL].copy()

    if len(tmp) == 0:
        raise ValueError("No kept cells after thresholding; cannot infer semantic cluster mapping.")

    z_col_map = {
        "mi_time": "z_mi_time",
        "mi_direction": "z_mi_direction",
        "mi_position": "z_mi_position",
    }

    cluster_ids = sorted(tmp["raw_keep_label"].unique().tolist())
    scores = pd.DataFrame(index=cluster_ids, columns=[0, 1, 2], dtype=float)

    for raw_cid in cluster_ids:
        sub = tmp[tmp["raw_keep_label"] == raw_cid]
        scores.loc[raw_cid, 0] = sub[z_col_map["mi_time"]].mean()
        scores.loc[raw_cid, 1] = sub[z_col_map["mi_direction"]].mean()
        scores.loc[raw_cid, 2] = sub[z_col_map["mi_position"]].mean()

    remap = {}
    used_raw = set()

    for semantic_label in [0, 1, 2]:
        col_scores = scores[semantic_label].copy()
        for raw_cid in used_raw:
            col_scores.loc[raw_cid] = -np.inf
        best_raw = int(col_scores.idxmax())
        remap[best_raw] = semantic_label
        used_raw.add(best_raw)

    for raw_cid in range(N_CLUSTERS):
        if raw_cid not in remap:
            remaining = [x for x in [0, 1, 2] if x not in remap.values()]
            remap[raw_cid] = remaining[0] if len(remaining) > 0 else raw_cid

    remap[UNASSIGNED_LABEL] = UNASSIGNED_LABEL
    return remap


def fit_within_dataset_models(df_subset):
    X_raw = df_subset[MIR_FEATURE_COLUMNS].astype(float).to_numpy()

    valid_mask = np.all(np.isfinite(X_raw), axis=1)
    df_subset = df_subset.loc[valid_mask].copy().reset_index(drop=True)
    X_raw = X_raw[valid_mask]

    if len(df_subset) == 0:
        raise ValueError("All rows were removed due to invalid MIR values.")

    X_z = rowwise_zscore(X_raw)

    z_cols = [f"z_{c}" for c in MIR_FEATURE_COLUMNS]
    for j, col in enumerate(z_cols):
        df_subset[col] = X_z[:, j]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_z)

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
    )
    initial_labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    keep_labels = np.full(len(X_scaled), UNASSIGNED_LABEL, dtype=int)
    thresholds = {}

    for cid in range(N_CLUSTERS):
        cluster_indices = np.where(initial_labels == cid)[0]
        if len(cluster_indices) == 0:
            thresholds[cid] = np.nan
            continue

        cluster_points = X_scaled[cluster_indices]
        centroid = centroids[cid]
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        threshold = np.percentile(dists, KEEP_PERCENTILE)
        thresholds[cid] = float(threshold)

        keep_mask = dists <= threshold
        keep_indices = cluster_indices[keep_mask]
        keep_labels[keep_indices] = cid

    remap_dict = infer_cluster_semantic_mapping(df_subset, keep_labels)
    semantic_labels = np.array([remap_dict.get(int(l), UNASSIGNED_LABEL) for l in keep_labels], dtype=int)
    raw_pred_semantic = np.array([remap_dict.get(int(l), UNASSIGNED_LABEL) for l in initial_labels], dtype=int)

    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        initialization=TSNE_INITIALIZATION,
        metric=TSNE_METRIC,
        n_jobs=TSNE_N_JOBS,
        random_state=TSNE_RANDOM_STATE,
        verbose=True,
    )
    tsne_embedding = tsne.fit(X_scaled)
    emb_np = np.asarray(tsne_embedding)

    df_subset["cluster_initial"] = initial_labels
    df_subset["cluster_id"] = semantic_labels
    df_subset["cluster_pred_semantic"] = raw_pred_semantic
    df_subset["tsne_1"] = emb_np[:, 0]
    df_subset["tsne_2"] = emb_np[:, 1]

    model_package = {
        "scaler": scaler,
        "kmeans": kmeans,
        "tsne_embedding_model": tsne_embedding,
        "thresholds_raw_cluster": thresholds,
        "cluster_remap_raw_to_semantic": remap_dict,
        "feature_columns": MIR_FEATURE_COLUMNS,
        "z_feature_columns": z_cols,
        "n_clusters": N_CLUSTERS,
        "keep_percentile": KEEP_PERCENTILE,
        "unassigned_label": UNASSIGNED_LABEL,
        "centroids_raw": centroids,
    }

    return df_subset, X_raw, X_z, X_scaled, emb_np, model_package


# =============================================================================
# Plotting + stats
# =============================================================================

def plot_embedding_grid(df_plot, out_png_dir, out_svg_dir, out_base, use_z=True, add_depth=False):
    feature_cols = [f"z_{c}" for c in MIR_FEATURE_COLUMNS] if use_z else MIR_FEATURE_COLUMNS
    vmin = MIR_VMIN_Z if use_z else MIR_VMIN_RAW
    vmax = MIR_VMAX_Z if use_z else MIR_VMAX_RAW

    df_plot, _ = build_mouse_index(df_plot)

    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    axs = axs.ravel()

    sns.scatterplot(
        data=df_plot,
        x="tsne_1",
        y="tsne_2",
        hue="cluster_id",
        palette=CLUSTER_COLOR_MAP,
        ax=axs[0],
        legend=False,
        s=SCATTER_SIZE,
    )
    axs[0].set_title("tSNE by Cluster ID")

    sns.scatterplot(
        data=df_plot,
        x="tsne_1",
        y="tsne_2",
        hue="mouse_index",
        palette=MOUSE_CMAP,
        ax=axs[1],
        legend=False,
        s=SCATTER_SIZE,
    )
    axs[1].set_title("tSNE by Mouse ID")

    for i in range(7):
        sc = axs[i + 2].scatter(
            df_plot["tsne_1"],
            df_plot["tsne_2"],
            c=df_plot[feature_cols[i]],
            cmap=FEATURE_CMAP,
            s=SCATTER_SIZE,
            vmin=vmin,
            vmax=vmax,
        )
        axs[i + 2].set_title(f"tSNE by {MIR_FEATURE_TITLES[i]}")
        plt.colorbar(sc, ax=axs[i + 2])

    for ax in axs:
        ax.grid(False)

    plt.tight_layout()
    save_figure_both(fig, out_png_dir, out_svg_dir, out_base)

    if add_depth:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        plot_df = df_plot.copy()
        plot_df["depth_norm"] = plot_df["depth_type"].apply(normalize_depth_label)

        sns.scatterplot(
            data=plot_df,
            x="tsne_1",
            y="tsne_2",
            hue="depth_norm",
            palette=DEPTH_COLOR_MAP,
            s=SCATTER_SIZE,
            ax=ax,
        )
        ax.set_title("tSNE by Depth")
        ax.grid(False)
        plt.tight_layout()
        save_figure_both(fig, out_png_dir, out_svg_dir, out_base + "_by_depth")


def plot_cluster_counts(df_plot, out_png_dir, out_svg_dir, stats_dir, out_base):
    counts = (
        df_plot.groupby("cluster_id")
        .size()
        .reset_index(name="count")
        .copy()
    )
    counts["cluster_str"] = counts["cluster_id"].astype(int).astype(str)

    order = [c for c in ["0", "1", "2", "10"] if c in counts["cluster_str"].tolist()]

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(
        data=counts,
        x="cluster_str",
        y="count",
        hue="cluster_str",
        palette=CLUSTER_COLOR_MAP_STR,
        order=order,
        hue_order=order,
        dodge=False,
        legend=False,
        ax=ax,
    )

    cluster_names = [CLUSTER_NAME_MAP_STR.get(c, c) for c in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(cluster_names, fontsize=10, rotation=30)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of Neurons", fontsize=12)
    plt.tight_layout()
    save_figure_both(fig, out_png_dir, out_svg_dir, out_base)

    counts_csv = os.path.join(os.path.dirname(out_png_dir), out_base + ".csv")
    counts.to_csv(counts_csv, index=False)
    print(f"[OK] Saved: {counts_csv}")

    txt = []
    txt.append("Cluster counts summary\n")
    txt.append("======================\n\n")
    txt.append(counts.to_string(index=False))
    txt.append("\n\n")
    txt.append(run_cluster_count_gof_test(counts, cluster_col="cluster_str", count_col="count"))
    save_text_file(stats_dir, out_base, "".join(txt))


def plot_cluster_counts_per_mouse(df_plot, out_png_dir, out_svg_dir, stats_dir, out_base):
    df_plot, _ = build_mouse_index(df_plot)

    mouse_cluster_counts = (
        df_plot
        .groupby(["mouse_name", "cluster_id"])
        .size()
        .unstack(fill_value=0)
    )

    df_long = mouse_cluster_counts.reset_index().melt(
        id_vars="mouse_name",
        var_name="cluster_id",
        value_name="count"
    )

    df_long["cluster_str"] = df_long["cluster_id"].astype(int).astype(str)
    df_long["cluster_name"] = df_long["cluster_str"].map(CLUSTER_NAME_MAP_STR)

    mean_std_df = (
        df_long
        .groupby(["cluster_str", "cluster_name"])["count"]
        .agg(["mean", "std"])
        .reset_index()
    )

    order = [c for c in ["0", "1", "2", "10"] if c in mean_std_df["cluster_str"].tolist()]
    cluster_name_order = [CLUSTER_NAME_MAP_STR[c] for c in order]

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(
        data=mean_std_df,
        x="cluster_name",
        y="mean",
        hue="cluster_str",
        palette=CLUSTER_COLOR_MAP_STR,
        hue_order=order,
        order=cluster_name_order,
        dodge=False,
        legend=False,
        ax=ax,
    )

    for _, group in df_long.groupby("mouse_name"):
        sns.lineplot(
            data=group,
            x="cluster_name",
            y="count",
            marker="o",
            color="dimgray",
            linewidth=1.5,
            markersize=5,
            alpha=0.5,
            legend=False,
            ax=ax,
        )

    ax.set_xticks(range(len(cluster_name_order)))
    ax.set_xticklabels(cluster_name_order, fontsize=10, rotation=20)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of Neurons", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    save_figure_both(fig, out_png_dir, out_svg_dir, out_base)

    out_csv = os.path.join(os.path.dirname(out_png_dir), out_base + ".csv")
    mouse_cluster_counts.to_csv(out_csv)
    print(f"[OK] Saved: {out_csv}")

    txt = run_oneway_ttests(
        df_long,
        group_col="cluster_name",
        value_col="count",
        title="Cluster counts per mouse"
    )
    save_text_file(stats_dir, out_base, txt)


def plot_cluster_counts_by_depth(df_plot, out_png_dir, out_svg_dir, stats_dir, out_base):
    plot_df = df_plot.copy()
    plot_df["depth_norm"] = plot_df["depth_type"].apply(normalize_depth_label)

    depth_cluster_counts = (
        plot_df
        .groupby(["depth_norm", "cluster_id"])
        .size()
        .reset_index(name="count")
        .copy()
    )

    depth_cluster_counts["cluster_str"] = depth_cluster_counts["cluster_id"].astype(int).astype(str)
    order = [c for c in ["0", "1", "2", "10"] if c in depth_cluster_counts["cluster_str"].tolist()]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(
        data=depth_cluster_counts,
        x="cluster_str",
        y="count",
        hue="depth_norm",
        palette=DEPTH_COLOR_MAP,
        order=order,
        ax=ax,
    )

    cluster_names = [CLUSTER_NAME_MAP_STR.get(c, c) for c in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(cluster_names, rotation=25)

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Neurons")
    ax.set_title("Cluster counts by depth")
    plt.tight_layout()
    save_figure_both(fig, out_png_dir, out_svg_dir, out_base)

    out_csv = os.path.join(os.path.dirname(out_png_dir), out_base + ".csv")
    depth_cluster_counts.to_csv(out_csv, index=False)
    print(f"[OK] Saved: {out_csv}")

    txt = ""
    txt += run_two_way_anova_and_pairs(
        depth_cluster_counts,
        factor_a="cluster_str",
        factor_b="depth_norm",
        value_col="count",
        title="Cluster counts by depth",
    )
    txt += "\n\n"
    txt += run_chi_square_counts(
        depth_cluster_counts,
        row_var="depth_norm",
        col_var="cluster_str",
        count_var="count",
    )
    save_text_file(stats_dir, out_base, txt)


def plot_cluster_information_barplot(df_plot, out_png_dir, out_svg_dir, stats_dir, out_base, use_z=False, small=False):
    plot_df = df_plot[df_plot["cluster_id"] != UNASSIGNED_LABEL].copy()

    rows = []
    for _, row in plot_df.iterrows():
        for beh, feat in zip(MIR_FEATURE_TITLES, MIR_FEATURE_COLUMNS):
            rows.append({
                "behavioral_label": beh,
                "cluster": row["cluster_id"],
                "cluster_str": str(int(row["cluster_id"])),
                "MIR_value": row[f"z_{feat}"] if use_z else row[feat],
            })
    df_long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(3, 3) if small else (8, 3))
    sns.barplot(
        data=df_long,
        y="MIR_value",
        x="behavioral_label",
        hue="cluster_str",
        palette=CLUSTER_COLOR_MAP_STR,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=30)
    if not small:
        ax.set_title("MIR Values by Behavioral Label and Cluster")
        ax.set_ylabel("MIR Value", fontsize=12)
    else:
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title("")
    plt.tight_layout()
    save_figure_both(fig, out_png_dir, out_svg_dir, out_base)

    txt = run_two_way_anova_and_pairs(
        df_long,
        factor_a="behavioral_label",
        factor_b="cluster_str",
        value_col="MIR_value",
        title="MIR values by behavioral label and cluster",
    )
    save_text_file(stats_dir, out_base, txt)


# =============================================================================
# CSV saving
# =============================================================================

def save_labeled_neuron_csv(df_labeled, out_dir, file_name):
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, file_name)
    df_labeled.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)

    if not os.path.isfile(MI_CSV):
        raise FileNotFoundError(f"MI CSV not found: {MI_CSV}")
    if not os.path.isfile(MASTER_INDEX_CSV):
        raise FileNotFoundError(f"Master index CSV not found: {MASTER_INDEX_CSV}")

    df_mi = pd.read_csv(MI_CSV)
    df_master = pd.read_csv(MASTER_INDEX_CSV)

    required_cols = [
        "dataset_name", "mouse_name", "session_name", "depth_type",
        "recording_type", "row_index", "neuron_index"
    ] + MIR_FEATURE_COLUMNS
    missing = [c for c in required_cols if c not in df_mi.columns]
    if missing:
        raise ValueError(f"Missing required columns in MI CSV: {missing}")

    print_available_values(df_mi, df_master)

    df_all = merge_master_metadata(df_mi, df_master)
    df_all = attach_session_datetime(df_all)
    df_all = add_neuron_uid(df_all)

    for cfg in WITHIN_DATASET_CONFIGS:
        name = cfg["name"]
        print(f"\n=== Within-dataset clustering: {name} ===")

        subset_out_dir = os.path.join(OUTPUT_DIR, sanitize_for_filename(name))
        subset_data_dir = os.path.join(subset_out_dir, "data")
        subset_png_dir = os.path.join(subset_out_dir, "png")
        subset_svg_dir = os.path.join(subset_out_dir, "svg")
        subset_stats_dir = os.path.join(subset_out_dir, "stats")
        subset_model_dir = os.path.join(subset_out_dir, "models")

        ensure_dir(subset_data_dir)
        ensure_dir(subset_png_dir)
        ensure_dir(subset_svg_dir)
        ensure_dir(subset_stats_dir)
        ensure_dir(subset_model_dir)

        mask = dataset_name_mask(df_all, cfg["dataset_name_contains"])
        df_subset = df_all[mask].copy()

        if len(df_subset) == 0:
            print(f"[WARNING] No rows matched dataset selector for {name}.")
            continue

        df_subset = subset_lt_rot(df_subset, cfg["subset_mode"])
        df_subset = subset_by_depth_mode(df_subset, cfg["depth_mode"])

        if cfg.get("latest_session_per_mouse", False):
            df_subset = select_latest_session_per_mouse(df_subset)

        if len(df_subset) == 0:
            print(f"[WARNING] No rows remained after filtering for {name}.")
            continue

        print(f"Rows in subset: {len(df_subset)}")
        print("Dataset names:", sorted(df_subset["dataset_name"].dropna().astype(str).unique().tolist()))
        print("Mice:", sorted(df_subset["mouse_name"].dropna().astype(str).unique().tolist()))
        print("Sessions:", sorted(df_subset["session_name"].dropna().astype(str).unique().tolist()))

        try:
            df_labeled, X_raw, X_z, X_scaled, emb_np, model_package = fit_within_dataset_models(df_subset)
        except Exception as e:
            print(f"[WARNING] Failed on subset {name}: {e}")
            continue

        save_pickle(os.path.join(subset_model_dir, "model_package.pkl"), model_package)
        np.save(os.path.join(subset_model_dir, "X_raw.npy"), X_raw)
        np.save(os.path.join(subset_model_dir, "X_z.npy"), X_z)
        np.save(os.path.join(subset_model_dir, "X_scaled.npy"), X_scaled)
        np.save(os.path.join(subset_model_dir, "embedding.npy"), emb_np)

        subset_meta = {
            "name": name,
            "rows": int(len(df_labeled)),
            "dataset_names": sorted(df_labeled["dataset_name"].dropna().astype(str).unique().tolist()),
            "mice": sorted(df_labeled["mouse_name"].dropna().astype(str).unique().tolist()),
            "sessions": sorted(df_labeled["session_name"].dropna().astype(str).unique().tolist()),
            "depths": sorted(df_labeled["depth_type"].dropna().astype(str).unique().tolist()),
            "cluster_remap_raw_to_semantic": model_package["cluster_remap_raw_to_semantic"],
            "thresholds_raw_cluster": model_package["thresholds_raw_cluster"],
        }
        with open(os.path.join(subset_out_dir, "subset_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(subset_meta, f, indent=2)

        save_labeled_neuron_csv(
            df_labeled,
            subset_data_dir,
            f"{sanitize_for_filename(name)}_mir_labeled_neurons.csv"
        )

        add_depth = df_labeled["depth_type"].apply(normalize_depth_label).isin(["deep", "superficial"]).any()

        plot_embedding_grid(
            df_labeled,
            subset_png_dir,
            subset_svg_dir,
            f"{sanitize_for_filename(name)}_clusters_grid",
            use_z=True,
            add_depth=add_depth
        )
        plot_cluster_counts(
            df_labeled,
            subset_png_dir,
            subset_svg_dir,
            subset_stats_dir,
            f"{sanitize_for_filename(name)}_cluster_counts"
        )
        plot_cluster_counts_per_mouse(
            df_labeled,
            subset_png_dir,
            subset_svg_dir,
            subset_stats_dir,
            f"{sanitize_for_filename(name)}_cluster_counts_per_mouse"
        )

        if add_depth:
            plot_cluster_counts_by_depth(
                df_labeled,
                subset_png_dir,
                subset_svg_dir,
                subset_stats_dir,
                f"{sanitize_for_filename(name)}_cluster_counts_by_depth"
            )

        plot_cluster_information_barplot(
            df_labeled,
            subset_png_dir,
            subset_svg_dir,
            subset_stats_dir,
            f"{sanitize_for_filename(name)}_cluster_information_barplot",
            use_z=False,
            small=False
        )
        plot_cluster_information_barplot(
            df_labeled,
            subset_png_dir,
            subset_svg_dir,
            subset_stats_dir,
            f"{sanitize_for_filename(name)}_cluster_information_barplot_small",
            use_z=False,
            small=True
        )

        print(f"[OK] Finished subset: {name}")

    print("\nDone.")


if __name__ == "__main__":
    main()