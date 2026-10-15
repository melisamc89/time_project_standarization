#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lt_rot rate-matched subsampling + SI analysis

Goal
----
For the lt_rot dataset only:

1) Use the per-neuron mean activity database to pool neurons by condition:
       - LT deep vs LT superficial
       - ROT deep vs ROT superficial
   where condition is inferred from session_name containing 'lt' or 'rot'.

2) For each condition, match the deep and superficial mean-activity
   distributions by bin-wise subsampling using the minimum count per bin.

3) Repeat the subsampling 10 times.

4) Save intermediate files describing the selected neurons.

5) Go back to the original standardized files, subset the selected neurons,
   compute UMAP on the subsetted signal, then compute Structure Index in UMAP
   space, and save SI values to a table with:
       - mouse
       - depth
       - session
       - condition
       - repetition ID
       - SI values
"""

import os
import csv
import copy
import pickle
import timeit
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
import umap

from structure_index import compute_structure_index


# =============================================================================
# Paths
# =============================================================================

MEAN_ACTIVITY_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/neuron_mean_activity/neuron_mean_activity_per_neuron.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/lt_rot_rate_matched_SI"
INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, "intermediate")
SELECTED_NEURON_DIR = os.path.join(INTERMEDIATE_DIR, "selected_neurons")
BIN_SUMMARY_DIR = os.path.join(INTERMEDIATE_DIR, "bin_summaries")
SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

SELECTED_NEURONS_ALL_CSV = os.path.join(INTERMEDIATE_DIR, "all_selected_neurons.csv")
SI_RESULTS_CSV = os.path.join(OUTPUT_DIR, "lt_rot_rate_matched_SI_results.csv")


# =============================================================================
# Parameters
# =============================================================================

DATASET_NAME = "lt_rot"

N_REPETITIONS = 10
N_BINS_MATCH = 10
RANDOM_SEED_BASE = 1234

# UMAP for SI
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

# outlier filtering in embedding space
OUTLIER_DIST_TH = 5
OUTLIER_NOISE_TH = 25

# SI params
SI_PARAMS_CONTINUOUS = {
    "n_bins": 10,
    "discrete_label": False,
    "continuity_kernel": None,
    "perc_neigh": 1,
    "num_shuffles": 0,
    "verbose": False,
}

SI_PARAMS_DISCRETE = {
    "n_bins": 2,
    "discrete_label": True,
    "continuity_kernel": None,
    "n_neighbors": 50,
    "num_shuffles": 0,
    "verbose": False,
}

SI_PARAMS_MIXED = {
    "n_bins": [10, 2],
    "discrete_label": [False, True],
    "n_neighbors": 50,
    "num_shuffles": 0,
    "verbose": False,
}

SI_BEHAVIOR_ORDER = [
    "position",
    "speed",
    "time",
    "inner_time",
    "trial_id",
    "direction",
    "position_direction",
]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_pickle_standard(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle_standard(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


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
        return pos.reshape(-1)
    if pos.ndim == 2:
        return pos[:, 0].reshape(-1)
    raise ValueError(f"Unexpected pos shape: {pos.shape}")


def sanitize_for_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def base_name_from_metadata(row):
    return (
        f"{sanitize_for_filename(row['dataset_name'])}_"
        f"{sanitize_for_filename(row['mouse_name'])}_"
        f"{sanitize_for_filename(row['session_name'])}_"
        f"{sanitize_for_filename(row['depth_type'])}"
    )


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
# Condition / depth parsing
# =============================================================================

def normalize_depth_label(depth_value):
    if pd.isna(depth_value):
        return "unknown"
    depth = str(depth_value).strip().lower()
    if depth in ["sup", "superficial"]:
        return "superficial"
    if depth in ["deep"]:
        return "deep"
    if depth in ["whole", "whole population", "whole_population", "all"]:
        return "whole population"
    return depth


def infer_lt_rot_condition(session_name):
    s = str(session_name).lower()

    has_lt = "lt" in s
    has_rot = "rot" in s

    if has_rot and not has_lt:
        return "rot"
    if has_lt and not has_rot:
        return "lt"
    if has_rot:
        return "rot"
    if has_lt:
        return "lt"

    return "unknown"


# =============================================================================
# Outlier filtering
# =============================================================================

def filter_noisy_outliers(data, D=None, dist_th=5, noise_th=25):
    if D is None:
        D = pairwise_distances(data)

    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D, dist_th), axis=1)
    noise_idx = np.where(nn_dist < np.percentile(nn_dist, noise_th))[0]
    signal_idx = np.where(nn_dist >= np.percentile(nn_dist, noise_th))[0]

    return noise_idx, signal_idx


# =============================================================================
# Matching
# =============================================================================

def match_distributions_by_bin(values_deep, values_sup, n_bins=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    values_deep = np.asarray(values_deep).reshape(-1)
    values_sup = np.asarray(values_sup).reshape(-1)

    pooled = np.concatenate([values_deep, values_sup])

    vmin = np.min(pooled)
    vmax = np.max(pooled)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    bin_edges = np.linspace(vmin, vmax, n_bins + 1)

    deep_bins = np.digitize(values_deep, bin_edges[1:-1], right=False)
    sup_bins = np.digitize(values_sup, bin_edges[1:-1], right=False)

    deep_idx_selected = []
    sup_idx_selected = []

    deep_counts = np.zeros(n_bins, dtype=int)
    sup_counts = np.zeros(n_bins, dtype=int)
    matched_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        deep_idx_bin = np.where(deep_bins == b)[0]
        sup_idx_bin = np.where(sup_bins == b)[0]

        deep_counts[b] = len(deep_idx_bin)
        sup_counts[b] = len(sup_idx_bin)

        n_match = min(len(deep_idx_bin), len(sup_idx_bin))
        matched_counts[b] = n_match

        if n_match == 0:
            continue

        deep_sel = rng.choice(deep_idx_bin, size=n_match, replace=False)
        sup_sel = rng.choice(sup_idx_bin, size=n_match, replace=False)

        deep_idx_selected.append(deep_sel)
        sup_idx_selected.append(sup_sel)

    if len(deep_idx_selected) == 0 or len(sup_idx_selected) == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            bin_edges,
            deep_counts,
            sup_counts,
            matched_counts,
            deep_bins,
            sup_bins,
        )

    deep_idx_selected = np.sort(np.concatenate(deep_idx_selected))
    sup_idx_selected = np.sort(np.concatenate(sup_idx_selected))

    return (
        deep_idx_selected,
        sup_idx_selected,
        bin_edges,
        deep_counts,
        sup_counts,
        matched_counts,
        deep_bins,
        sup_bins,
    )


# =============================================================================
# SI helpers
# =============================================================================

def get_si_params_for_behavior(behavior_name, behavior_values):
    if behavior_name in ["position", "speed", "time", "inner_time", "trial_id"]:
        params = copy.deepcopy(SI_PARAMS_CONTINUOUS)
        params["n_neighbors"] = 50
        if behavior_name == "trial_id":
            params["min_label"] = float(np.min(behavior_values))
            params["max_label"] = float(np.max(behavior_values))
        return params

    if behavior_name == "direction":
        return copy.deepcopy(SI_PARAMS_DISCRETE)

    if behavior_name == "position_direction":
        return copy.deepcopy(SI_PARAMS_MIXED)

    raise ValueError(f"Unknown behavior: {behavior_name}")


def compute_si_from_selected_neurons(metadata_row, selected_neuron_indices):
    standardized_file_path = metadata_row["standardized_file_path"]
    data = load_pickle_standard(standardized_file_path)

    signal = ensure_2d(data["signal"])
    pos = ensure_1d(get_pos_1d(data["pos"]))
    mov_dir = ensure_1d(data["mov_dir"])
    time_1d = ensure_1d(data["time"])

    speed = ensure_1d(data["speed"]) if data.get("speed", None) is not None else None
    inner_time = ensure_1d(data["inner_time"]) if data.get("inner_time", None) is not None else None
    trial_id = ensure_1d(data["trial_id"]) if data.get("trial_id", None) is not None else None

    selected_neuron_indices = np.asarray(selected_neuron_indices, dtype=int)

    if selected_neuron_indices.size == 0:
        raise ValueError("No selected neurons for this session/depth.")

    signal = signal[:, selected_neuron_indices]

    n_timepoints_original = signal.shape[0]
    n_selected_neurons = signal.shape[1]

    valid_idx = mov_dir != 0
    signal_valid = signal[valid_idx, :]

    pos_valid = pos[valid_idx]
    mov_dir_valid = mov_dir[valid_idx]
    time_valid = time_1d[valid_idx]
    speed_valid = None if speed is None else speed[valid_idx]
    inner_time_valid = None if inner_time is None else inner_time[valid_idx]
    trial_id_valid = None if trial_id is None else trial_id[valid_idx]

    n_timepoints_after_dir_filter = signal_valid.shape[0]
    if n_timepoints_after_dir_filter < 10:
        raise ValueError("Too few timepoints after mov_dir != 0 filtering.")

    n_neighbors = min(UMAP_N_NEIGHBORS, max(5, n_timepoints_after_dir_filter - 1))
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    emb = reducer.fit_transform(signal_valid)

    D = pairwise_distances(emb)
    _, signal_idx = filter_noisy_outliers(
        emb,
        D=D,
        dist_th=OUTLIER_DIST_TH,
        noise_th=OUTLIER_NOISE_TH,
    )

    emb_clean = emb[signal_idx, :]
    pos_clean = pos_valid[signal_idx]
    mov_dir_clean = mov_dir_valid[signal_idx]
    time_clean = time_valid[signal_idx]
    speed_clean = None if speed_valid is None else speed_valid[signal_idx]
    inner_time_clean = None if inner_time_valid is None else inner_time_valid[signal_idx]
    trial_id_clean = None if trial_id_valid is None else trial_id_valid[signal_idx]

    n_timepoints_after_outlier_filter = emb_clean.shape[0]
    if n_timepoints_after_outlier_filter < 10:
        raise ValueError("Too few timepoints after outlier filtering.")

    position_direction = np.concatenate(
        (
            pos_clean.reshape(-1, 1),
            mov_dir_clean.reshape(-1, 1),
        ),
        axis=1,
    )

    behaviors = {
        "position": pos_clean,
        "speed": speed_clean,
        "time": time_clean,
        "inner_time": inner_time_clean,
        "trial_id": trial_id_clean,
        "direction": mov_dir_clean,
        "position_direction": position_direction,
    }

    si_results = {}
    si_details = {}

    for beh_name in SI_BEHAVIOR_ORDER:
        beh_val = behaviors[beh_name]

        if beh_val is None:
            si_results[beh_name] = np.nan
            si_details[beh_name] = None
            continue

        params = get_si_params_for_behavior(beh_name, beh_val)

        si, process_info, overlap_mat, shuf_si = compute_structure_index(
            emb_clean,
            beh_val,
            **params,
        )

        si_results[beh_name] = float(si)
        si_details[beh_name] = {
            "si": si,
            "process_info": process_info,
            "overlap_mat": overlap_mat,
            "shuf_si": shuf_si,
            "params": params,
        }

    session_payload = {
        "metadata": {
            "dataset_name": metadata_row["dataset_name"],
            "mouse_name": metadata_row["mouse_name"],
            "session_name": metadata_row["session_name"],
            "depth_type": metadata_row["depth_type"],
            "condition": metadata_row["condition"],
            "standardized_file_path": standardized_file_path,
            "n_selected_neurons": n_selected_neurons,
            "n_timepoints_original": n_timepoints_original,
            "n_timepoints_after_dir_filter": n_timepoints_after_dir_filter,
            "n_timepoints_after_outlier_filter": n_timepoints_after_outlier_filter,
        },
        "selected_neuron_indices": selected_neuron_indices,
        "si": si_details,
    }

    return si_results, session_payload, {
        "n_selected_neurons": n_selected_neurons,
        "n_timepoints_original": n_timepoints_original,
        "n_timepoints_after_dir_filter": n_timepoints_after_dir_filter,
        "n_timepoints_after_outlier_filter": n_timepoints_after_outlier_filter,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(INTERMEDIATE_DIR)
    ensure_dir(SELECTED_NEURON_DIR)
    ensure_dir(BIN_SUMMARY_DIR)
    ensure_dir(SESSION_PKL_DIR)
    ensure_dir(LOG_DIR)

    if not os.path.isfile(MEAN_ACTIVITY_CSV):
        raise FileNotFoundError(f"Mean activity CSV not found: {MEAN_ACTIVITY_CSV}")

    mean_df = pd.read_csv(MEAN_ACTIVITY_CSV)

    required_cols_mean = [
        "dataset_name",
        "mouse_name",
        "session_name",
        "depth_type",
        "recording_type",
        "row_index",
        "neuron_index",
        "standardized_file_path",
        "mean_activity",
    ]
    missing_mean_cols = [c for c in required_cols_mean if c not in mean_df.columns]
    if missing_mean_cols:
        raise ValueError(
            f"Mean activity CSV is missing required columns: {missing_mean_cols}"
        )

    if mean_df["standardized_file_path"].isna().any():
        missing = mean_df[mean_df["standardized_file_path"].isna()][
            ["dataset_name", "mouse_name", "session_name", "depth_type", "row_index"]
        ].drop_duplicates()
        raise ValueError(
            "Some rows in mean activity CSV have missing standardized_file_path.\n"
            f"{missing.to_string(index=False)}"
        )

    log_path = os.path.join(LOG_DIR, "lt_rot_rate_matched_SI_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("lt_rot rate-matched subsampling + SI analysis")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Mean activity CSV: {MEAN_ACTIVITY_CSV}")
    print(f"Repetitions: {N_REPETITIONS}")
    print(f"Matching bins: {N_BINS_MATCH}")
    print("")

    mean_df = mean_df[mean_df["dataset_name"] == DATASET_NAME].copy()

    if len(mean_df) == 0:
        raise ValueError("No rows found for lt_rot in mean activity CSV.")

    mean_df["depth_norm"] = mean_df["depth_type"].apply(normalize_depth_label)
    mean_df["condition"] = mean_df["session_name"].apply(infer_lt_rot_condition)

    all_selected_rows = []
    all_si_rows = []

    global_time = timeit.default_timer()

    for condition in ["lt", "rot"]:
        cond_df = mean_df[mean_df["condition"] == condition].copy()

        if len(cond_df) == 0:
            print(f"[WARNING] No neurons found for condition={condition}")
            continue

        deep_df = cond_df[cond_df["depth_norm"] == "deep"].copy().reset_index(drop=True)
        sup_df = cond_df[cond_df["depth_norm"] == "superficial"].copy().reset_index(drop=True)

        if len(deep_df) == 0 or len(sup_df) == 0:
            print(f"[WARNING] Missing deep or superficial neurons for condition={condition}")
            continue

        deep_vals = deep_df["mean_activity"].to_numpy(dtype=float)
        sup_vals = sup_df["mean_activity"].to_numpy(dtype=float)

        print("--------------------------------------------------------------")
        print(f"Condition: {condition}")
        print(f"Deep neurons pooled: {len(deep_df)}")
        print(f"Superficial neurons pooled: {len(sup_df)}")
        print("--------------------------------------------------------------")

        for rep in range(N_REPETITIONS):
            local_time = timeit.default_timer()
            rng = np.random.default_rng(RANDOM_SEED_BASE + rep + (0 if condition == "lt" else 1000))

            (
                deep_sel_idx,
                sup_sel_idx,
                bin_edges,
                deep_counts,
                sup_counts,
                matched_counts,
                deep_bin_ids,
                sup_bin_ids,
            ) = match_distributions_by_bin(
                deep_vals,
                sup_vals,
                n_bins=N_BINS_MATCH,
                rng=rng,
            )

            if len(deep_sel_idx) == 0 or len(sup_sel_idx) == 0:
                print(f"[WARNING] rep={rep:02d} condition={condition}: no matched neurons")
                continue

            deep_selected = deep_df.iloc[deep_sel_idx].copy()
            sup_selected = sup_df.iloc[sup_sel_idx].copy()

            deep_selected["repetition_id"] = rep
            sup_selected["repetition_id"] = rep

            deep_selected["selected_group"] = "deep"
            sup_selected["selected_group"] = "superficial"

            deep_selected["match_bin"] = deep_bin_ids[deep_sel_idx]
            sup_selected["match_bin"] = sup_bin_ids[sup_sel_idx]

            selected_df = pd.concat([deep_selected, sup_selected], axis=0, ignore_index=True)

            selected_out_name = f"selected_neurons_{condition}_rep{rep:02d}.csv"
            selected_out_path = os.path.join(SELECTED_NEURON_DIR, selected_out_name)
            selected_df.to_csv(selected_out_path, index=False)
            print(f"[OK] Saved: {selected_out_path}")

            bin_rows = []
            for b in range(N_BINS_MATCH):
                bin_rows.append({
                    "condition": condition,
                    "repetition_id": rep,
                    "bin_id": b,
                    "bin_left": bin_edges[b],
                    "bin_right": bin_edges[b + 1],
                    "deep_count": int(deep_counts[b]),
                    "superficial_count": int(sup_counts[b]),
                    "matched_count": int(matched_counts[b]),
                })
            bin_df = pd.DataFrame(bin_rows)
            bin_out_name = f"bin_summary_{condition}_rep{rep:02d}.csv"
            bin_out_path = os.path.join(BIN_SUMMARY_DIR, bin_out_name)
            bin_df.to_csv(bin_out_path, index=False)
            print(f"[OK] Saved: {bin_out_path}")

            all_selected_rows.append(selected_df)

            group_cols = [
                "dataset_name",
                "mouse_name",
                "session_name",
                "depth_type",
                "recording_type",
                "row_index",
                "standardized_file_path",
                "condition",
            ]

            for group_key, group_df in selected_df.groupby(group_cols):
                meta = dict(zip(group_cols, group_key))
                selected_neuron_indices = np.sort(group_df["neuron_index"].astype(int).unique())

                try:
                    si_results, session_payload, session_info = compute_si_from_selected_neurons(
                        meta,
                        selected_neuron_indices
                    )

                    session_base = (
                        f"{sanitize_for_filename(condition)}_rep{rep:02d}_"
                        f"{base_name_from_metadata(meta)}_si.pkl"
                    )
                    session_pkl_path = os.path.join(SESSION_PKL_DIR, session_base)
                    save_pickle_standard(session_pkl_path, session_payload)

                    all_si_rows.append({
                        "dataset_name": meta["dataset_name"],
                        "mouse_name": meta["mouse_name"],
                        "session_name": meta["session_name"],
                        "depth_type": meta["depth_type"],
                        "recording_type": meta["recording_type"],
                        "condition": condition,
                        "repetition_id": rep,
                        "row_index": meta["row_index"],
                        "standardized_file_path": meta["standardized_file_path"],
                        "n_selected_neurons": session_info["n_selected_neurons"],
                        "n_timepoints_original": session_info["n_timepoints_original"],
                        "n_timepoints_after_dir_filter": session_info["n_timepoints_after_dir_filter"],
                        "n_timepoints_after_outlier_filter": session_info["n_timepoints_after_outlier_filter"],
                        "si_position": si_results["position"],
                        "si_speed": si_results["speed"],
                        "si_time": si_results["time"],
                        "si_inner_time": si_results["inner_time"],
                        "si_trial_id": si_results["trial_id"],
                        "si_direction": si_results["direction"],
                        "si_position_direction": si_results["position_direction"],
                        "session_pickle_path": session_pkl_path,
                    })

                except Exception as e:
                    print(
                        f"[WARNING] SI failed | cond={condition} rep={rep:02d} | "
                        f"mouse={meta['mouse_name']} | session={meta['session_name']} | "
                        f"depth={meta['depth_type']} | error={e}"
                    )

            elapsed = timeit.default_timer() - local_time
            print(
                f"[OK] Finished condition={condition} rep={rep:02d} | "
                f"matched deep={len(deep_selected)} | matched sup={len(sup_selected)} | "
                f"time={elapsed:.2f}s"
            )

    if len(all_selected_rows) > 0:
        selected_all_df = pd.concat(all_selected_rows, axis=0, ignore_index=True)
        selected_all_df.to_csv(SELECTED_NEURONS_ALL_CSV, index=False)
        print(f"[OK] Saved: {SELECTED_NEURONS_ALL_CSV}")
    else:
        print("[WARNING] No selected neurons to save.")

    if len(all_si_rows) > 0:
        si_df = pd.DataFrame(all_si_rows)
        si_df.to_csv(SI_RESULTS_CSV, index=False)
        print(f"[OK] Saved: {SI_RESULTS_CSV}")
    else:
        print("[WARNING] No SI results to save.")

    total_elapsed = timeit.default_timer() - global_time
    print("==============================================================")
    print("Done")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Total elapsed time: {total_elapsed:.2f} s")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()

#######################################################################################################################
##################################### PLOTTING ########################################################################
#######################################################################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze SI results from lt_rot rate-matched repetitions.

What this script does
---------------------
1) Loads the SI results CSV produced by the rate-matched lt_rot SI pipeline.
2) Creates a long-format SI table.
3) Computes mean SI per animal per feature per repetition.
4) Creates deep vs superficial boxplots for each feature.
5) Does this separately for LT and ROT conditions.
6) Saves:
   - long-format SI table
   - per-animal mean SI table
   - boxplot figures
   - statistics .txt files

Interpretation
--------------
The comparison is made on:
    mean SI per animal, per feature, per repetition

So each point is one mouse-average SI value for one repetition.
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols


# =============================================================================
# Paths
# =============================================================================

INPUT_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/lt_rot_rate_matched_SI/lt_rot_rate_matched_SI_results.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/lt_rot_rate_matched_SI_summary"
PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")


# =============================================================================
# Parameters
# =============================================================================

SI_COLUMNS = [
    "si_position",
    "si_speed",
    "si_time",
    "si_inner_time",
    "si_trial_id",
    "si_direction",
    "si_position_direction",
]

FEATURE_NAME_MAP = {
    "si_position": "position",
    "si_speed": "speed",
    "si_time": "time",
    "si_inner_time": "inner_time",
    "si_trial_id": "trial_id",
    "si_direction": "direction",
    "si_position_direction": "position_direction",
}

DEPTH_COLORS = {
    "superficial": "#9900ff",
    "deep": "#cc9900",
}

BOX_ORDER = ["superficial", "deep"]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_figure_both(fig, out_base):
    png_path = os.path.join(PNG_DIR, out_base + ".png")
    svg_path = os.path.join(SVG_DIR, out_base + ".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")


def save_text_file(out_base, text):
    out_path = os.path.join(STATS_DIR, out_base + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] Saved: {out_path}")


def pval_to_star(pval):
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def normalize_depth_label(depth_value):
    if pd.isna(depth_value):
        return "unknown"
    depth = str(depth_value).strip().lower()
    if depth in ["sup", "superficial"]:
        return "superficial"
    if depth == "deep":
        return "deep"
    return depth


# =============================================================================
# Statistics
# =============================================================================

def run_depth_stats(df, condition_name):
    """
    df should contain:
        mouse_name, depth_norm, repetition_id, feature, si_value
    at animal-mean level
    """
    txt = []
    txt.append(f"Deep vs superficial SI comparison\n")
    txt.append(f"Condition: {condition_name}\n")
    txt.append("=" * 60 + "\n\n")

    features = [f for f in df["feature"].dropna().unique().tolist()]

    for feature in features:
        sub = df[df["feature"] == feature].copy()
        txt.append(f"Feature: {feature}\n")
        txt.append("-" * 40 + "\n")

        grouped = (
            sub.groupby("depth_norm")["si_value"]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .reset_index()
        )
        txt.append("Descriptive statistics\n")
        txt.append(grouped.to_string(index=False))
        txt.append("\n\n")

        depths_present = sorted(sub["depth_norm"].dropna().unique().tolist())
        if len(depths_present) >= 2:
            try:
                model = ols("si_value ~ C(depth_norm)", data=sub).fit()
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
            for d1, d2 in itertools.combinations(depths_present, 2):
                v1 = sub.loc[sub["depth_norm"] == d1, "si_value"].dropna()
                v2 = sub.loc[sub["depth_norm"] == d2, "si_value"].dropna()

                if len(v1) > 1 and len(v2) > 1:
                    stat, pval = ttest_ind(v1, v2, equal_var=False)
                    txt.append(
                        f"{d1} vs {d2}: "
                        f"t={stat:.6f}, p={pval:.6e}, star={pval_to_star(pval)}, "
                        f"n1={len(v1)}, n2={len(v2)}\n"
                    )
                else:
                    txt.append(f"{d1} vs {d2}: skipped (insufficient n)\n")
        else:
            txt.append("Only one depth group present. No between-group statistics.\n")

        txt.append("\n\n")

    return "".join(txt)


# =============================================================================
# Main transformations
# =============================================================================

def make_long_si_table(df):
    id_cols = [
        "dataset_name",
        "mouse_name",
        "session_name",
        "depth_type",
        "recording_type",
        "condition",
        "repetition_id",
        "row_index",
        "n_selected_neurons",
    ]

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=SI_COLUMNS,
        var_name="feature_raw",
        value_name="si_value",
    ).copy()

    long_df["feature"] = long_df["feature_raw"].map(FEATURE_NAME_MAP)
    long_df["depth_norm"] = long_df["depth_type"].apply(normalize_depth_label)

    return long_df


def make_mean_si_per_animal(long_df):
    """
    Mean SI per animal per feature per repetition.
    """
    mean_df = (
        long_df
        .groupby(
            ["dataset_name", "condition", "repetition_id", "mouse_name", "depth_norm", "feature"],
            as_index=False
        )["si_value"]
        .mean()
    )
    mean_df = mean_df.rename(columns={"si_value": "mean_si_per_animal"})
    return mean_df


# =============================================================================
# Plotting
# =============================================================================

def plot_boxplots_per_condition(mean_df, condition_name):
    sub = mean_df[mean_df["condition"] == condition_name].copy()
    if len(sub) == 0:
        print(f"[WARNING] No rows found for condition={condition_name}")
        return

    features_present = [f for f in FEATURE_NAME_MAP.values() if f in sub["feature"].unique().tolist()]
    if len(features_present) == 0:
        print(f"[WARNING] No SI features found for condition={condition_name}")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.boxplot(
        data=sub,
        x="feature",
        y="mean_si_per_animal",
        hue="depth_norm",
        order=features_present,
        hue_order=BOX_ORDER,
        palette=DEPTH_COLORS,
        ax=ax,
        showfliers=False,
    )

    sns.stripplot(
        data=sub,
        x="feature",
        y="mean_si_per_animal",
        hue="depth_norm",
        order=features_present,
        hue_order=BOX_ORDER,
        dodge=True,
        palette=DEPTH_COLORS,
        ax=ax,
        alpha=0.5,
        size=4,
        linewidth=0,
    )

    # clean duplicate legends from boxplot + stripplot
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 0:
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        ax.legend(unique.values(), unique.keys(), frameon=False, title="depth")
    else:
        ax.legend().remove()

    ax.set_title(f"{condition_name}: mean SI per animal per repetition")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean SI")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(False)

    plt.tight_layout()
    save_figure_both(fig, f"{condition_name}_mean_si_per_animal_boxplots")

    stats_text = run_depth_stats(
        sub.rename(columns={"mean_si_per_animal": "si_value"}),
        condition_name=condition_name
    )
    save_text_file(f"{condition_name}_mean_si_per_animal_boxplots_stats", stats_text)


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)
    ensure_dir(CSV_DIR)
    ensure_dir(STATS_DIR)

    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = [
        "dataset_name",
        "mouse_name",
        "session_name",
        "depth_type",
        "condition",
        "repetition_id",
    ] + SI_COLUMNS

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in SI CSV: {missing_cols}")

    print(f"Loaded SI rows: {len(df)}")
    print("Conditions found:")
    print(sorted(df["condition"].dropna().astype(str).unique().tolist()))
    print("Depths found:")
    print(sorted(df["depth_type"].dropna().astype(str).unique().tolist()))

    # -------------------------------------------------------------------------
    # 1) Long-format SI table
    # -------------------------------------------------------------------------
    long_df = make_long_si_table(df)
    long_csv_path = os.path.join(CSV_DIR, "lt_rot_rate_matched_SI_long_table.csv")
    long_df.to_csv(long_csv_path, index=False)
    print(f"[OK] Saved: {long_csv_path}")

    # -------------------------------------------------------------------------
    # 2) Mean SI per animal per feature per repetition
    # -------------------------------------------------------------------------
    mean_df = make_mean_si_per_animal(long_df)
    mean_csv_path = os.path.join(CSV_DIR, "lt_rot_rate_matched_SI_mean_per_animal_feature_repetition.csv")
    mean_df.to_csv(mean_csv_path, index=False)
    print(f"[OK] Saved: {mean_csv_path}")

    # -------------------------------------------------------------------------
    # 3) Plot deep vs superficial boxplots separately for lt and rot
    # -------------------------------------------------------------------------
    for condition_name in ["lt", "rot"]:
        plot_boxplots_per_condition(mean_df, condition_name)

    print("Done.")


if __name__ == "__main__":
    main()

########################################################################################################################
################################# PLOTTING COUNTING ####################################################################
########################################################################################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize how many neurons per mouse/session are used in each downsampled repetition.

Inputs
------
1) Original per-neuron mean activity table:
   neuron_mean_activity_per_neuron.csv
2) Intermediate selected-neuron table from the matching script:
   all_selected_neurons.csv

What this script computes
-------------------------
For each:
    - condition
    - repetition
    - mouse
    - session
    - depth

it computes:
    - n_total_neurons
    - n_selected_neurons
    - selected_fraction = n_selected_neurons / n_total_neurons

Outputs
-------
1) CSV tables
2) Boxplots + stripplots for:
    - selected neuron count
    - selected fraction
3) Stats .txt files

Plots are made:
- pooled across sessions, grouped by depth
- pooled across mice, grouped by session
- separately for LT and ROT
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols


# =============================================================================
# Paths
# =============================================================================

MEAN_ACTIVITY_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/neuron_mean_activity/neuron_mean_activity_per_neuron.csv"
SELECTED_NEURONS_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/lt_rot_rate_matched_SI/intermediate/all_selected_neurons.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/lt_rot_rate_matched_selection_summary"
PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")


# =============================================================================
# Style / colors
# =============================================================================

DEPTH_COLORS = {
    "superficial": "#9900ff",
    "deep": "#cc9900",
}

SESSION_PALETTE = "tab10"
DEPTH_ORDER = ["superficial", "deep"]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_figure_both(fig, out_base):
    png_path = os.path.join(PNG_DIR, out_base + ".png")
    svg_path = os.path.join(SVG_DIR, out_base + ".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")


def save_text_file(out_base, text):
    out_path = os.path.join(STATS_DIR, out_base + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] Saved: {out_path}")


def normalize_depth_label(depth_value):
    if pd.isna(depth_value):
        return "unknown"
    depth = str(depth_value).strip().lower()
    if depth in ["sup", "superficial"]:
        return "superficial"
    if depth == "deep":
        return "deep"
    return depth


def pval_to_star(pval):
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


# =============================================================================
# Stats helpers
# =============================================================================

def run_depth_stats(df, value_col, title):
    """
    Compare deep vs superficial on a scalar metric.
    """
    txt = []
    txt.append(f"{title}\n")
    txt.append("=" * len(title) + "\n\n")

    grouped = (
        df.groupby("depth_norm")[value_col]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    txt.append("Descriptive statistics\n")
    txt.append(grouped.to_string(index=False))
    txt.append("\n\n")

    depths_present = sorted(df["depth_norm"].dropna().unique().tolist())
    if len(depths_present) >= 2:
        try:
            model = ols(f"{value_col} ~ C(depth_norm)", data=df).fit()
            if model.df_resid > 0:
                anova_table = sm.stats.anova_lm(model, typ=2)
                txt.append("One-way ANOVA\n")
                txt.append(anova_table.to_string())
                txt.append("\n\n")
            else:
                txt.append("One-way ANOVA skipped: non-positive residual degrees of freedom.\n\n")
        except Exception as e:
            txt.append(f"One-way ANOVA failed: {e}\n\n")

        txt.append("Welch t-test\n")
        d1, d2 = depths_present[:2]
        v1 = df.loc[df["depth_norm"] == d1, value_col].dropna()
        v2 = df.loc[df["depth_norm"] == d2, value_col].dropna()
        if len(v1) > 1 and len(v2) > 1:
            stat, pval = ttest_ind(v1, v2, equal_var=False)
            txt.append(
                f"{d1} vs {d2}: "
                f"t={stat:.6f}, p={pval:.6e}, star={pval_to_star(pval)}, "
                f"n1={len(v1)}, n2={len(v2)}\n"
            )
        else:
            txt.append("Welch t-test skipped: insufficient n.\n")

    return "".join(txt)


def run_session_stats(df, value_col, title):
    """
    Compare across sessions.
    """
    txt = []
    txt.append(f"{title}\n")
    txt.append("=" * len(title) + "\n\n")

    grouped = (
        df.groupby("session_name")[value_col]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    txt.append("Descriptive statistics\n")
    txt.append(grouped.to_string(index=False))
    txt.append("\n\n")

    sessions = sorted(df["session_name"].dropna().astype(str).unique().tolist())
    if len(sessions) >= 2:
        try:
            model = ols(f"{value_col} ~ C(session_name)", data=df).fit()
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
        for s1, s2 in itertools.combinations(sessions, 2):
            v1 = df.loc[df["session_name"] == s1, value_col].dropna()
            v2 = df.loc[df["session_name"] == s2, value_col].dropna()
            if len(v1) > 1 and len(v2) > 1:
                stat, pval = ttest_ind(v1, v2, equal_var=False)
                txt.append(
                    f"{s1} vs {s2}: t={stat:.6f}, p={pval:.6e}, "
                    f"star={pval_to_star(pval)}, n1={len(v1)}, n2={len(v2)}\n"
                )
            else:
                txt.append(f"{s1} vs {s2}: skipped (insufficient n)\n")

    return "".join(txt)


# =============================================================================
# Build summary table
# =============================================================================

def build_selection_summary(mean_df, selected_df):
    mean_df = mean_df.copy()
    selected_df = selected_df.copy()

    mean_df["depth_norm"] = mean_df["depth_type"].apply(normalize_depth_label)
    selected_df["depth_norm"] = selected_df["depth_type"].apply(normalize_depth_label)

    total_counts = (
        mean_df.groupby(
            ["dataset_name", "mouse_name", "session_name", "depth_type", "depth_norm", "row_index"],
            as_index=False
        )
        .size()
        .rename(columns={"size": "n_total_neurons"})
    )

    selected_counts = (
        selected_df.groupby(
            [
                "dataset_name", "mouse_name", "session_name", "depth_type", "depth_norm",
                "row_index", "condition", "repetition_id"
            ],
            as_index=False
        )
        .size()
        .rename(columns={"size": "n_selected_neurons"})
    )

    summary = selected_counts.merge(
        total_counts,
        on=["dataset_name", "mouse_name", "session_name", "depth_type", "depth_norm", "row_index"],
        how="left"
    )

    summary["selected_fraction"] = summary["n_selected_neurons"] / summary["n_total_neurons"]

    return summary


# =============================================================================
# Plotting
# =============================================================================

def plot_depth_comparison(summary_df, condition_name, value_col, ylabel, out_base):
    sub = summary_df[summary_df["condition"] == condition_name].copy()
    if len(sub) == 0:
        print(f"[WARNING] No rows for condition={condition_name}")
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.boxplot(
        data=sub,
        x="depth_norm",
        y=value_col,
        order=DEPTH_ORDER,
        palette=DEPTH_COLORS,
        ax=ax,
        showfliers=False,
    )

    sns.stripplot(
        data=sub,
        x="depth_norm",
        y=value_col,
        order=DEPTH_ORDER,
        palette=DEPTH_COLORS,
        ax=ax,
        alpha=0.5,
        size=4,
        jitter=0.2,
    )

    ax.set_title(f"{condition_name}: {ylabel} by depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel(ylabel)
    ax.grid(False)

    plt.tight_layout()
    save_figure_both(fig, out_base)

    stats_text = run_depth_stats(sub, value_col=value_col, title=f"{condition_name}: {ylabel} by depth")
    save_text_file(out_base + "_stats", stats_text)


def plot_session_comparison(summary_df, condition_name, value_col, ylabel, out_base):
    sub = summary_df[summary_df["condition"] == condition_name].copy()
    if len(sub) == 0:
        print(f"[WARNING] No rows for condition={condition_name}")
        return

    session_order = sorted(sub["session_name"].dropna().astype(str).unique().tolist())

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.boxplot(
        data=sub,
        x="session_name",
        y=value_col,
        order=session_order,
        palette=SESSION_PALETTE,
        ax=ax,
        showfliers=False,
    )

    sns.stripplot(
        data=sub,
        x="session_name",
        y=value_col,
        order=session_order,
        palette=SESSION_PALETTE,
        ax=ax,
        alpha=0.5,
        size=4,
        jitter=0.2,
    )

    ax.set_title(f"{condition_name}: {ylabel} by session")
    ax.set_xlabel("Session")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(False)

    plt.tight_layout()
    save_figure_both(fig, out_base)

    stats_text = run_session_stats(sub, value_col=value_col, title=f"{condition_name}: {ylabel} by session")
    save_text_file(out_base + "_stats", stats_text)


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)
    ensure_dir(CSV_DIR)
    ensure_dir(STATS_DIR)

    if not os.path.isfile(MEAN_ACTIVITY_CSV):
        raise FileNotFoundError(f"Mean activity CSV not found: {MEAN_ACTIVITY_CSV}")
    if not os.path.isfile(SELECTED_NEURONS_CSV):
        raise FileNotFoundError(f"Selected neurons CSV not found: {SELECTED_NEURONS_CSV}")

    mean_df = pd.read_csv(MEAN_ACTIVITY_CSV)
    selected_df = pd.read_csv(SELECTED_NEURONS_CSV)

    mean_df = mean_df[mean_df["dataset_name"] == "lt_rot"].copy()
    selected_df = selected_df[selected_df["dataset_name"] == "lt_rot"].copy()

    if len(mean_df) == 0:
        raise ValueError("No lt_rot rows found in mean activity CSV.")
    if len(selected_df) == 0:
        raise ValueError("No lt_rot rows found in selected neurons CSV.")

    summary_df = build_selection_summary(mean_df, selected_df)

    summary_csv = os.path.join(CSV_DIR, "lt_rot_selection_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Saved: {summary_csv}")

    # also save a smaller per-session aggregate table if useful
    agg_df = (
        summary_df.groupby(
            ["condition", "repetition_id", "mouse_name", "session_name", "depth_norm"],
            as_index=False
        )[["n_selected_neurons", "n_total_neurons", "selected_fraction"]]
        .mean()
    )
    agg_csv = os.path.join(CSV_DIR, "lt_rot_selection_summary_aggregated.csv")
    agg_df.to_csv(agg_csv, index=False)
    print(f"[OK] Saved: {agg_csv}")

    for condition_name in sorted(summary_df["condition"].dropna().astype(str).unique().tolist()):
        plot_depth_comparison(
            summary_df,
            condition_name=condition_name,
            value_col="n_selected_neurons",
            ylabel="Selected neurons",
            out_base=f"{condition_name}_selected_neurons_by_depth"
        )

        plot_depth_comparison(
            summary_df,
            condition_name=condition_name,
            value_col="selected_fraction",
            ylabel="Selected fraction",
            out_base=f"{condition_name}_selected_fraction_by_depth"
        )

        plot_session_comparison(
            summary_df,
            condition_name=condition_name,
            value_col="n_selected_neurons",
            ylabel="Selected neurons",
            out_base=f"{condition_name}_selected_neurons_by_session"
        )

        plot_session_comparison(
            summary_df,
            condition_name=condition_name,
            value_col="selected_fraction",
            ylabel="Selected fraction",
            out_base=f"{condition_name}_selected_fraction_by_session"
        )

    print("Done.")


if __name__ == "__main__":
    main()


########################################################################################################################
################################# PLOTTING NEW DISTRIBUTIONS ###########################################################
########################################################################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify activity distributions of selected neurons after lt_rot matching.

What this script does
---------------------
Using:
1) the original per-neuron mean activity table
2) the selected-neurons table from the matching pipeline

it creates two kinds of plots:

A) One figure per animal:
   - overlays the distributions of all repetitions for that animal
   - color is determined by the animal's depth group
   - deep animals use deep color
   - superficial animals use superficial color

B) One pooled figure per condition (lt / rot):
   - aggregates all selected neurons from all repetitions
   - compares deep vs superficial

Outputs
-------
- PNG and SVG figures
- CSV histogram summaries
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

MEAN_ACTIVITY_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/neuron_mean_activity/neuron_mean_activity_per_neuron.csv"
SELECTED_NEURONS_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/lt_rot_rate_matched_SI/intermediate/all_selected_neurons.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/lt_rot_rate_matched_distribution_check"
PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")


# =============================================================================
# Parameters
# =============================================================================

DATASET_NAME = "lt_rot"

N_BINS = 40
ALPHA_ANIMAL = 0.18
ALPHA_POOLED = 0.30
LINEWIDTH = 1.8
DENSITY = False

DEPTH_COLORS = {
    "superficial": "#9900ff",
    "deep": "#cc9900",
}


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def normalize_depth_label(depth_value):
    if pd.isna(depth_value):
        return "unknown"
    depth = str(depth_value).strip().lower()
    if depth in ["sup", "superficial"]:
        return "superficial"
    if depth == "deep":
        return "deep"
    return depth


def save_figure_both(fig, out_base):
    png_path = os.path.join(PNG_DIR, out_base + ".png")
    svg_path = os.path.join(SVG_DIR, out_base + ".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")


def compute_common_bins(values, n_bins=40):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError("No finite values available to compute histogram bins.")

    vmin = np.min(values)
    vmax = np.max(values)

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    return np.linspace(vmin, vmax, n_bins + 1)


def save_hist_counts_csv(groups_dict, bins, out_name):
    rows = []
    for group_name, values in groups_dict.items():
        hist, _ = np.histogram(values, bins=bins)
        for i in range(len(hist)):
            rows.append({
                "group": group_name,
                "bin_index": i,
                "bin_left": bins[i],
                "bin_right": bins[i + 1],
                "count": int(hist[i]),
            })

    out_path = os.path.join(CSV_DIR, out_name + ".csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")


# =============================================================================
# Main plotting
# =============================================================================

def plot_per_animal_repetition_distributions(merged_df):
    """
    One plot per animal.
    Overlays all repetition distributions for that animal.
    """
    animals = sorted(merged_df["mouse_name"].dropna().astype(str).unique().tolist())

    for mouse in animals:
        sub_mouse = merged_df[merged_df["mouse_name"].astype(str) == mouse].copy()
        if len(sub_mouse) == 0:
            continue

        depth_values = sub_mouse["depth_norm"].dropna().unique().tolist()
        if len(depth_values) == 0:
            continue
        depth_norm = depth_values[0]
        color = DEPTH_COLORS.get(depth_norm, "gray")

        repetitions = sorted(sub_mouse["repetition_id"].dropna().astype(int).unique().tolist())
        if len(repetitions) == 0:
            continue

        all_vals = sub_mouse["mean_activity"].dropna().to_numpy()
        if len(all_vals) == 0:
            continue

        bins = compute_common_bins(all_vals, n_bins=N_BINS)

        fig, ax = plt.subplots(figsize=(7, 5))

        hist_groups = {}

        for rep in repetitions:
            vals = sub_mouse.loc[sub_mouse["repetition_id"] == rep, "mean_activity"].dropna().to_numpy()
            if len(vals) == 0:
                continue

            label = f"rep{rep:02d}"
            hist_groups[label] = vals

            ax.hist(
                vals,
                bins=bins,
                alpha=ALPHA_ANIMAL,
                density=DENSITY,
                color=color,
                edgecolor=None,
            )
            ax.hist(
                vals,
                bins=bins,
                histtype="step",
                linewidth=LINEWIDTH,
                density=DENSITY,
                color=color,
                label=label,
            )

        ax.set_title(f"{mouse} ({depth_norm}) - selected neuron activity distributions across repetitions")
        ax.set_xlabel("Mean activity")
        ax.set_ylabel("Density" if DENSITY else "Count")
        ax.legend(frameon=False, ncol=2, fontsize=8)
        ax.grid(False)

        plt.tight_layout()
        out_base = f"{sanitize_for_filename(mouse)}_{sanitize_for_filename(depth_norm)}_repetition_distributions"
        save_figure_both(fig, out_base)
        save_hist_counts_csv(hist_groups, bins, out_base)


def plot_pooled_deep_vs_sup(merged_df):
    """
    One pooled plot per condition, comparing all selected subsets together:
    deep vs superficial.
    """
    conditions = sorted(merged_df["condition"].dropna().astype(str).unique().tolist())

    for condition in conditions:
        sub = merged_df[merged_df["condition"].astype(str) == condition].copy()
        if len(sub) == 0:
            continue

        deep_vals = sub.loc[sub["depth_norm"] == "deep", "mean_activity"].dropna().to_numpy()
        sup_vals = sub.loc[sub["depth_norm"] == "superficial", "mean_activity"].dropna().to_numpy()

        if len(deep_vals) == 0 or len(sup_vals) == 0:
            print(f"[WARNING] Missing deep or superficial selected neurons for condition={condition}")
            continue

        bins = compute_common_bins(np.concatenate([deep_vals, sup_vals]), n_bins=N_BINS)

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.hist(
            sup_vals,
            bins=bins,
            alpha=ALPHA_POOLED,
            density=DENSITY,
            color=DEPTH_COLORS["superficial"],
            edgecolor=None,
            label="superficial",
        )
        ax.hist(
            sup_vals,
            bins=bins,
            histtype="step",
            linewidth=LINEWIDTH,
            density=DENSITY,
            color=DEPTH_COLORS["superficial"],
        )

        ax.hist(
            deep_vals,
            bins=bins,
            alpha=ALPHA_POOLED,
            density=DENSITY,
            color=DEPTH_COLORS["deep"],
            edgecolor=None,
            label="deep",
        )
        ax.hist(
            deep_vals,
            bins=bins,
            histtype="step",
            linewidth=LINEWIDTH,
            density=DENSITY,
            color=DEPTH_COLORS["deep"],
        )

        ax.set_title(f"{condition}: pooled selected neuron activity distributions")
        ax.set_xlabel("Mean activity")
        ax.set_ylabel("Density" if DENSITY else "Count")
        ax.legend(frameon=False)
        ax.grid(False)

        plt.tight_layout()
        out_base = f"{sanitize_for_filename(condition)}_pooled_selected_distributions_deep_vs_superficial"
        save_figure_both(fig, out_base)
        save_hist_counts_csv(
            {"superficial": sup_vals, "deep": deep_vals},
            bins,
            out_base
        )


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)
    ensure_dir(CSV_DIR)

    if not os.path.isfile(MEAN_ACTIVITY_CSV):
        raise FileNotFoundError(f"Mean activity CSV not found: {MEAN_ACTIVITY_CSV}")
    if not os.path.isfile(SELECTED_NEURONS_CSV):
        raise FileNotFoundError(f"Selected neurons CSV not found: {SELECTED_NEURONS_CSV}")

    mean_df = pd.read_csv(MEAN_ACTIVITY_CSV)
    selected_df = pd.read_csv(SELECTED_NEURONS_CSV)

    mean_df = mean_df[mean_df["dataset_name"] == DATASET_NAME].copy()
    selected_df = selected_df[selected_df["dataset_name"] == DATASET_NAME].copy()

    if len(mean_df) == 0:
        raise ValueError("No lt_rot rows found in mean activity CSV.")
    if len(selected_df) == 0:
        raise ValueError("No lt_rot rows found in selected neurons CSV.")

    mean_df["depth_norm"] = mean_df["depth_type"].apply(normalize_depth_label)
    selected_df["depth_norm"] = selected_df["depth_type"].apply(normalize_depth_label)

    # Merge back the original mean activity values for the selected neurons
    join_cols = [
        "dataset_name",
        "mouse_name",
        "session_name",
        "depth_type",
        "row_index",
        "neuron_index",
    ]

    mean_sub = mean_df[join_cols + ["mean_activity"]].copy()

    merged_df = selected_df.merge(
        mean_sub,
        on=join_cols,
        how="left",
        suffixes=("", "_orig")
    )

    if merged_df["mean_activity"].isna().any():
        missing = merged_df[merged_df["mean_activity"].isna()][join_cols].drop_duplicates()
        raise ValueError(
            "Some selected neurons could not be matched back to mean activity values.\n"
            f"{missing.to_string(index=False)}"
        )

    merged_csv = os.path.join(CSV_DIR, "lt_rot_selected_neurons_with_mean_activity.csv")
    merged_df.to_csv(merged_csv, index=False)
    print(f"[OK] Saved: {merged_csv}")

    plot_per_animal_repetition_distributions(merged_df)
    plot_pooled_deep_vs_sup(merged_df)

    print("Done.")


if __name__ == "__main__":
    main()

########################################################################################################################
############################### PLOTTING ORIGINAL DISTRIBUTIONS ########################################################
########################################################################################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot ORIGINAL mean-activity distributions for lt_rot, separately for LT and ROT,
using all neurons and coloring by deep vs superficial.

Input
-----
Per-neuron mean activity table:
    neuron_mean_activity_per_neuron.csv

What this script does
---------------------
1) Loads the per-neuron mean activity table
2) Keeps only dataset == 'lt_rot'
3) Infers condition from session_name:
       - contains 'lt'  -> LT
       - contains 'rot' -> ROT
4) Normalizes depth labels to:
       - deep
       - superficial
5) Plots histograms for:
       - LT: deep vs superficial
       - ROT: deep vs superficial
   using common bins within each condition

Outputs
-------
- PNG and SVG figures
- CSV files with histogram counts
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

INPUT_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/neuron_mean_activity/neuron_mean_activity_per_neuron.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/lt_rot_original_activity_distributions"
PNG_DIR = os.path.join(OUTPUT_DIR, "png")
SVG_DIR = os.path.join(OUTPUT_DIR, "svg")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")


# =============================================================================
# Parameters
# =============================================================================

DATASET_NAME = "lt_rot"

N_BINS = 40
ALPHA = 0.30
LINEWIDTH = 2.0
DENSITY = False

DEPTH_COLORS = {
    "superficial": "#9900ff",
    "deep": "#cc9900",
}


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(text):
    text = str(text)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def save_figure_both(fig, out_base):
    png_path = os.path.join(PNG_DIR, out_base + ".png")
    svg_path = os.path.join(SVG_DIR, out_base + ".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")


def normalize_depth_label(depth_value):
    if pd.isna(depth_value):
        return "unknown"
    depth = str(depth_value).strip().lower()
    if depth in ["sup", "superficial"]:
        return "superficial"
    if depth == "deep":
        return "deep"
    if depth in ["whole", "whole population", "whole_population", "all"]:
        return "whole population"
    return depth


def infer_lt_rot_condition(session_name):
    s = str(session_name).lower()

    has_lt = "lt" in s
    has_rot = "rot" in s

    if has_rot and not has_lt:
        return "rot"
    if has_lt and not has_rot:
        return "lt"
    if has_rot:
        return "rot"
    if has_lt:
        return "lt"

    return "unknown"


def compute_common_bins(values, n_bins=40):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError("No finite values available to compute histogram bins.")

    vmin = np.min(values)
    vmax = np.max(values)

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    return np.linspace(vmin, vmax, n_bins + 1)


def save_hist_counts_csv(groups_dict, bins, out_name):
    rows = []
    for group_name, values in groups_dict.items():
        hist, _ = np.histogram(values, bins=bins)
        for i in range(len(hist)):
            rows.append({
                "group": group_name,
                "bin_index": i,
                "bin_left": bins[i],
                "bin_right": bins[i + 1],
                "count": int(hist[i]),
            })

    out_path = os.path.join(CSV_DIR, out_name + ".csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")


# =============================================================================
# Plotting
# =============================================================================

def plot_condition_histogram(df, condition_name):
    sub = df[df["condition"] == condition_name].copy()
    if len(sub) == 0:
        print(f"[WARNING] No rows found for condition={condition_name}")
        return

    deep_vals = sub.loc[sub["depth_norm"] == "deep", "mean_activity"].dropna().to_numpy()
    sup_vals = sub.loc[sub["depth_norm"] == "superficial", "mean_activity"].dropna().to_numpy()

    if len(deep_vals) == 0 or len(sup_vals) == 0:
        print(f"[WARNING] Missing deep or superficial neurons for condition={condition_name}")
        return

    bins = compute_common_bins(np.concatenate([deep_vals, sup_vals]), n_bins=N_BINS)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        sup_vals,
        bins=bins,
        alpha=ALPHA,
        density=DENSITY,
        color=DEPTH_COLORS["superficial"],
        edgecolor=None,
        label="superficial",
    )
    ax.hist(
        sup_vals,
        bins=bins,
        histtype="step",
        linewidth=LINEWIDTH,
        density=DENSITY,
        color=DEPTH_COLORS["superficial"],
    )

    ax.hist(
        deep_vals,
        bins=bins,
        alpha=ALPHA,
        density=DENSITY,
        color=DEPTH_COLORS["deep"],
        edgecolor=None,
        label="deep",
    )
    ax.hist(
        deep_vals,
        bins=bins,
        histtype="step",
        linewidth=LINEWIDTH,
        density=DENSITY,
        color=DEPTH_COLORS["deep"],
    )

    ax.set_title(f"{condition_name}: original mean-activity distributions")
    ax.set_xlabel("Mean activity per neuron")
    ax.set_ylabel("Density" if DENSITY else "Count")
    ax.legend(frameon=False)
    ax.grid(False)

    plt.tight_layout()

    out_base = f"{sanitize_for_filename(condition_name)}_original_mean_activity_deep_vs_superficial"
    save_figure_both(fig, out_base)
    save_hist_counts_csv(
        {"superficial": sup_vals, "deep": deep_vals},
        bins,
        out_base
    )


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PNG_DIR)
    ensure_dir(SVG_DIR)
    ensure_dir(CSV_DIR)

    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = [
        "dataset_name",
        "session_name",
        "depth_type",
        "mean_activity",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {missing_cols}")

    df = df[df["dataset_name"] == DATASET_NAME].copy()
    if len(df) == 0:
        raise ValueError(f"No rows found for dataset_name == '{DATASET_NAME}'")

    df["condition"] = df["session_name"].apply(infer_lt_rot_condition)
    df["depth_norm"] = df["depth_type"].apply(normalize_depth_label)

    print(f"Loaded {len(df)} neuron rows for dataset={DATASET_NAME}")
    print("Conditions found:", sorted(df["condition"].dropna().astype(str).unique().tolist()))
    print("Depths found:", sorted(df["depth_norm"].dropna().astype(str).unique().tolist()))

    for condition_name in ["lt", "rot"]:
        plot_condition_histogram(df, condition_name)

    print("Done.")


if __name__ == "__main__":
    main()
