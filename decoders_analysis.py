#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reduced decoder pipeline across all standardized datasets.

Runs only 3 decoders:
    - position
    - position_direction
    - time

Supports regression models:
    - rf    : Random Forest regression
    - xgb   : XGBoost regression
    - wc    : Wiener cascade regression
    - svr   : Support vector regression

Current setting
---------------
Only Random Forest is run:
    DECODER_TYPES = ["rf"]

Output organization
-------------------
Results are saved under:

    OUTPUT_ROOT/
        rf/
            decoder_per_session.csv
            session_pickles/
            logs/
            pred_vs_true_plots/

For each decoder type, this script saves:
- one CSV with one row per session
- one pickle per session with detailed results
- predicted vs ground-truth plots
- per-point out-of-fold error vectors in the pickle
"""

import os
import csv
import copy
import pickle
import timeit
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgboost
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =============================================================================
# Paths
# =============================================================================

CSV_PATH = "/home/melisamc/Documents/time_project_standarization/data/metadata/master_dataset_index.csv"
OUTPUT_ROOT = "/home/melisamc/Documents/time_project_standarization/analysis/decoders"


# =============================================================================
# Parameters
# =============================================================================

MAX_ROWS = None
VERBOSE = True

# Run only Random Forest now
DECODER_TYPES = ["rf","xgb","wc","svr"]

N_SPLITS = 10
N_SPLITS_BASE = 2
RANDOM_STATE = 42
MIN_TIMEPOINTS = 30

USE_TRIAL_ID_SPLIT = True

# Random Forest parameters
RF_N_ESTIMATORS = 200
RF_RANDOM_STATE = 42
RF_N_JOBS = -1

# XGBoost parameters
XGB_MAX_DEPTH = 3
XGB_NUM_ROUND = 300
XGB_ETA = 0.3
XGB_GPU = -1
XGB_EARLY_STOPPING_ROUNDS = 20
XGB_SEED = 2925

# Wiener cascade parameters
WC_DEGREE = 3

# SVR parameters
SVR_C = 3.0
SVR_MAX_ITER = -1

FOLD_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
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


def base_name_from_row(row_idx, row):
    return (
        f"row_{row_idx:04d}_"
        f"{sanitize_for_filename(row.get('dataset_name', 'unknown_dataset'))}_"
        f"{sanitize_for_filename(row.get('mouse_name', 'unknown_mouse'))}_"
        f"{sanitize_for_filename(row.get('session_name', 'unknown_session'))}_"
        f"{sanitize_for_filename(row.get('depth_type', 'unknown_depth'))}"
    )


def infer_recording_type(row):
    candidate_columns = [
        "recording_type",
        "modality",
        "signal_type",
        "dataset_type",
        "data_type",
    ]
    for col in candidate_columns:
        if col in row and pd.notna(row[col]):
            value = str(row[col]).strip().lower()
            if "calcium" in value:
                return "calcium"
            if "ephys" in value or "electro" in value:
                return "ephys"

    text_to_search = " ".join([
        str(row.get("dataset_name", "")),
        str(row.get("standardized_file_path", "")),
    ]).lower()

    if any(k in text_to_search for k in ["calcium", "caim", "dff", "trace"]):
        return "calcium"
    if any(k in text_to_search for k in ["ephys", "lfp", "spike", "electro"]):
        return "ephys"

    return "unknown"


def validate_lengths(signal, behavior_dict, row_idx):
    n_time = signal.shape[0]
    for key, value in behavior_dict.items():
        if value is None:
            continue
        if len(value) != n_time:
            raise ValueError(
                f"Row {row_idx}: length mismatch for '{key}' "
                f"({len(value)} vs {n_time})"
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


def init_csv_if_needed(csv_path, columns):
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_csv_row(csv_path, columns, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def load_already_processed_paths(csv_path):
    if not os.path.isfile(csv_path):
        return set()
    try:
        df_old = pd.read_csv(csv_path, usecols=["standardized_file_path"])
        return set(df_old["standardized_file_path"].dropna().astype(str).tolist())
    except Exception:
        return set()


def get_output_paths_for_decoder(decoder_type):
    output_dir = os.path.join(OUTPUT_ROOT, decoder_type)
    session_pkl_dir = os.path.join(output_dir, "session_pickles")
    log_dir = os.path.join(output_dir, "logs")
    plots_dir = os.path.join(output_dir, "pred_vs_true_plots")
    png_dir = os.path.join(plots_dir, "png")
    svg_dir = os.path.join(plots_dir, "svg")
    decoder_session_csv = os.path.join(output_dir, "decoder_per_session.csv")

    return {
        "output_dir": output_dir,
        "session_pkl_dir": session_pkl_dir,
        "log_dir": log_dir,
        "plots_dir": plots_dir,
        "png_dir": png_dir,
        "svg_dir": svg_dir,
        "decoder_session_csv": decoder_session_csv,
    }


# =============================================================================
# Decoder model classes
# =============================================================================

class RandomForestRegression:
    def __init__(self, n_estimators=200, random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X_flat_train, y_train):
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        num_outputs = y_train.shape[1]
        models = []

        for y_idx in range(num_outputs):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            model.fit(X_flat_train, y_train[:, y_idx])
            models.append(model)

        self.model = models

    def predict(self, X_flat_test):
        num_outputs = len(self.model)
        y_test_predicted = np.empty((X_flat_test.shape[0], num_outputs), dtype=float)

        for y_idx in range(num_outputs):
            model = self.model[y_idx]
            y_test_predicted[:, y_idx] = model.predict(X_flat_test)

        return y_test_predicted


class XGBoostRegression:
    def __init__(self, max_depth=3, num_round=300, eta=0.3, gpu=-1):
        self.max_depth = max_depth
        self.num_round = num_round
        self.eta = eta
        self.gpu = gpu

    def fit(self, X_flat_train, y_train):
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        num_outputs = y_train.shape[1]

        param = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": self.max_depth,
            "eta": self.eta,
            "seed": XGB_SEED,
        }

        if self.gpu < 0:
            param["nthread"] = -1
            param["tree_method"] = "hist"
        else:
            param["gpu_id"] = self.gpu
            param["tree_method"] = "gpu_hist"

        models = []

        for y_idx in range(num_outputs):
            n_train = X_flat_train.shape[0]
            val_div = int(0.9 * n_train)

            if val_div < 5 or (n_train - val_div) < 5:
                dtrain = xgboost.DMatrix(X_flat_train, label=y_train[:, y_idx])
                bst = xgboost.train(param, dtrain, self.num_round, verbose_eval=False)
            else:
                dtrain = xgboost.DMatrix(
                    X_flat_train[:val_div, :],
                    label=y_train[:val_div, y_idx]
                )
                deval = xgboost.DMatrix(
                    X_flat_train[val_div:, :],
                    label=y_train[val_div:, y_idx]
                )
                bst = xgboost.train(
                    param,
                    dtrain,
                    self.num_round,
                    evals=[(deval, "eval")],
                    early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                    verbose_eval=False,
                )

            models.append(bst)

        self.model = models

    def predict(self, X_flat_test):
        dtest = xgboost.DMatrix(X_flat_test)
        num_outputs = len(self.model)
        y_test_predicted = np.empty((X_flat_test.shape[0], num_outputs), dtype=float)

        for y_idx in range(num_outputs):
            bst = self.model[y_idx]
            y_test_predicted[:, y_idx] = bst.predict(dtest)

        return y_test_predicted


class WienerCascadeRegression:
    def __init__(self, degree=3):
        self.degree = degree

    def fit(self, X_flat_train, y_train):
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        num_outputs = y_train.shape[1]
        models = []

        for i in range(num_outputs):
            regr = linear_model.LinearRegression()
            regr.fit(X_flat_train, y_train[:, i])
            y_train_predicted_linear = regr.predict(X_flat_train)
            p = np.polyfit(y_train_predicted_linear, y_train[:, i], self.degree)
            models.append([regr, p])

        self.model = models

    def predict(self, X_flat_test):
        num_outputs = len(self.model)
        y_test_predicted = np.empty((X_flat_test.shape[0], num_outputs), dtype=float)

        for i in range(num_outputs):
            regr, p = self.model[i]
            y_test_predicted_linear = regr.predict(X_flat_test)
            y_test_predicted[:, i] = np.polyval(p, y_test_predicted_linear)

        return y_test_predicted


class SVRegression:
    def __init__(self, max_iter=-1, C=3.0):
        self.max_iter = max_iter
        self.C = C

    def fit(self, X_flat_train, y_train):
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        num_outputs = y_train.shape[1]
        models = []

        for y_idx in range(num_outputs):
            model = SVR(C=self.C, max_iter=self.max_iter)
            model.fit(X_flat_train, y_train[:, y_idx])
            models.append(model)

        self.model = models

    def predict(self, X_flat_test):
        num_outputs = len(self.model)
        y_test_predicted = np.empty((X_flat_test.shape[0], num_outputs), dtype=float)

        for y_idx in range(num_outputs):
            model = self.model[y_idx]
            y_test_predicted[:, y_idx] = model.predict(X_flat_test)

        return y_test_predicted


def build_regressor(decoder_type):
    if decoder_type == "rf":
        return RandomForestRegression(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RF_RANDOM_STATE,
            n_jobs=RF_N_JOBS,
        )
    if decoder_type == "xgb":
        return XGBoostRegression(
            max_depth=XGB_MAX_DEPTH,
            num_round=XGB_NUM_ROUND,
            eta=XGB_ETA,
            gpu=XGB_GPU,
        )
    if decoder_type == "wc":
        return WienerCascadeRegression(degree=WC_DEGREE)
    if decoder_type == "svr":
        return SVRegression(max_iter=SVR_MAX_ITER, C=SVR_C)

    raise ValueError(f"Unknown decoder_type: {decoder_type}")


# =============================================================================
# CV splitting
# =============================================================================

def build_repeated_twofold_splits(n_samples, trial_signal=None, n_splits=10, random_state=42):
    n_repeats = int(np.ceil(n_splits / 2))

    if isinstance(trial_signal, np.ndarray):
        trial_signal = ensure_1d(trial_signal)
        trial_list = np.unique(trial_signal)

        if len(trial_list) >= 2:
            rkf = RepeatedKFold(
                n_splits=2,
                n_repeats=n_repeats,
                random_state=random_state
            )

            total_index = np.arange(n_samples, dtype=int)
            splits = []

            for train_trial_idx, test_trial_idx in rkf.split(trial_list, trial_list):
                train_trials = trial_list[train_trial_idx]
                test_trials = trial_list[test_trial_idx]

                train_mask = np.isin(trial_signal, train_trials)
                test_mask = np.isin(trial_signal, test_trials)

                train_index = total_index[train_mask]
                test_index = total_index[test_mask]

                if len(train_index) > 0 and len(test_index) > 0:
                    splits.append((train_index, test_index))

                if len(splits) >= n_splits:
                    break

            if len(splits) > 0:
                return splits[:n_splits], "repeated_2fold_trial_level"

        return [], "trial_split_unavailable"

    if n_samples < 2:
        return [], "insufficient_samples"

    rkf = RepeatedKFold(
        n_splits=2,
        n_repeats=n_repeats,
        random_state=random_state
    )

    dummy = np.arange(n_samples)
    splits = []

    for train_index, test_index in rkf.split(dummy, dummy):
        if len(train_index) > 0 and len(test_index) > 0:
            splits.append((train_index, test_index))

        if len(splits) >= n_splits:
            break

    return splits[:n_splits], "repeated_2fold_sample_level"


# =============================================================================
# Plotting
# =============================================================================

def plot_predicted_vs_true(result, decoder_name, title_prefix, base_filename, png_dir, svg_dir):
    preds = result.get("predictions_per_fold", [])
    if len(preds) == 0:
        return

    fig, ax = plt.subplots(figsize=(5.8, 5.2))

    all_true = []
    all_pred = []

    for fold_idx, fold_data in enumerate(preds):
        y_true = np.asarray(fold_data["y_true"], dtype=float)
        y_pred = np.asarray(fold_data["y_pred"], dtype=float)

        if len(y_true) == 0:
            continue

        all_true.append(y_true)
        all_pred.append(y_pred)

        ax.scatter(
            y_true,
            y_pred,
            s=18,
            alpha=0.75,
            color=FOLD_COLORS[fold_idx % len(FOLD_COLORS)],
            edgecolors="none",
            label=f"fold {fold_idx + 1}",
        )

    if len(all_true) == 0:
        plt.close(fig)
        return

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    finite_mask = np.isfinite(all_true) & np.isfinite(all_pred)
    all_true = all_true[finite_mask]
    all_pred = all_pred[finite_mask]

    if len(all_true) == 0:
        plt.close(fig)
        return

    vmin = min(np.min(all_true), np.min(all_pred))
    vmax = max(np.max(all_true), np.max(all_pred))

    ax.plot([vmin, vmax], [vmin, vmax], "--", color="black", linewidth=1.0)

    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title_prefix}\n{decoder_name}: predicted vs ground truth")

    if len(preds) <= 10:
        ax.legend(frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(png_dir, base_filename + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(svg_dir, base_filename + ".svg"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Decoding
# =============================================================================

def build_full_oof_vectors(signal_length, oof_test_index, oof_y_true, oof_y_pred):
    true_full = np.full(signal_length, np.nan, dtype=float)
    pred_full = np.full(signal_length, np.nan, dtype=float)
    err_full = np.full(signal_length, np.nan, dtype=float)

    if len(oof_test_index) > 0:
        true_full[oof_test_index] = oof_y_true
        pred_full[oof_test_index] = oof_y_pred
        err_full[oof_test_index] = np.abs(oof_y_pred - oof_y_true)

    return true_full, pred_full, err_full


def run_regressor_oof(signal, y, splits, decoder_type):
    signal = ensure_2d(signal)
    y = ensure_1d(y)

    fold_mse = []
    all_test_indices = []
    all_y_true = []
    all_y_pred = []
    predictions_per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = signal[train_idx, :]
        X_test = signal[test_idx, :]
        y_train = y[train_idx].reshape(-1, 1)
        y_test = y[test_idx].reshape(-1, 1)

        model = build_regressor(decoder_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[:, 0]

        mse = mean_squared_error(y_test[:, 0], y_pred)
        fold_mse.append(mse)

        all_test_indices.append(test_idx)
        all_y_true.append(y_test[:, 0])
        all_y_pred.append(y_pred)

        predictions_per_fold.append({
            "fold_index": fold_idx,
            "train_index": train_idx.copy(),
            "test_index": test_idx.copy(),
            "y_true": y_test[:, 0].copy(),
            "y_pred": y_pred.copy(),
            "mse": mse,
        })

    if len(all_test_indices) == 0:
        return {
            "fold_mse": np.array([np.nan]),
            "mean_mse": np.nan,
            "oof_test_index": np.array([], dtype=int),
            "oof_y_true": np.array([]),
            "oof_y_pred": np.array([]),
            "oof_true_full": np.array([]),
            "oof_pred_full": np.array([]),
            "oof_error_full": np.array([]),
            "predictions_per_fold": predictions_per_fold,
        }

    oof_test_index = np.concatenate(all_test_indices, axis=0)
    oof_y_true = np.concatenate(all_y_true, axis=0)
    oof_y_pred = np.concatenate(all_y_pred, axis=0)

    true_full, pred_full, err_full = build_full_oof_vectors(
        signal_length=signal.shape[0],
        oof_test_index=oof_test_index,
        oof_y_true=oof_y_true,
        oof_y_pred=oof_y_pred,
    )

    return {
        "fold_mse": np.asarray(fold_mse, dtype=float),
        "mean_mse": float(np.nanmean(fold_mse)),
        "oof_test_index": oof_test_index,
        "oof_y_true": oof_y_true,
        "oof_y_pred": oof_y_pred,
        "oof_true_full": true_full,
        "oof_pred_full": pred_full,
        "oof_error_full": err_full,
        "predictions_per_fold": predictions_per_fold,
    }


def run_time_decoder_normalized(signal, y, splits, signal_length, decoder_type):
    signal = ensure_2d(signal)
    y = ensure_1d(y)

    fold_mse = []
    fold_mae = []
    all_test_indices = []
    all_y_true = []
    all_y_pred = []
    predictions_per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = signal[train_idx, :]
        X_test = signal[test_idx, :]
        y_train = y[train_idx].reshape(-1, 1)
        y_test = y[test_idx].reshape(-1, 1)

        model = build_regressor(decoder_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[:, 0]

        mse = mean_squared_error(y_test[:, 0], y_pred)
        mae = mean_absolute_error(y_test[:, 0], y_pred)

        fold_mse.append(mse)
        fold_mae.append(mae)

        all_test_indices.append(test_idx)
        all_y_true.append(y_test[:, 0])
        all_y_pred.append(y_pred)

        predictions_per_fold.append({
            "fold_index": fold_idx,
            "train_index": train_idx.copy(),
            "test_index": test_idx.copy(),
            "y_true": y_test[:, 0].copy(),
            "y_pred": y_pred.copy(),
            "mse": mse,
            "mae": mae,
        })

    if len(all_test_indices) == 0:
        return {
            "fold_mse": np.array([np.nan]),
            "fold_mae": np.array([np.nan]),
            "mean_mse": np.nan,
            "mean_mae": np.nan,
            "normalized_error": np.nan,
            "normalization_factor": signal_length,
            "oof_test_index": np.array([], dtype=int),
            "oof_y_true": np.array([]),
            "oof_y_pred": np.array([]),
            "oof_true_full": np.array([]),
            "oof_pred_full": np.array([]),
            "oof_error_full": np.array([]),
            "predictions_per_fold": predictions_per_fold,
        }

    oof_test_index = np.concatenate(all_test_indices, axis=0)
    oof_y_true = np.concatenate(all_y_true, axis=0)
    oof_y_pred = np.concatenate(all_y_pred, axis=0)
    oof_mae = mean_absolute_error(oof_y_true, oof_y_pred)
    normalized_error = oof_mae / float(signal_length)

    true_full, pred_full, err_full = build_full_oof_vectors(
        signal_length=signal.shape[0],
        oof_test_index=oof_test_index,
        oof_y_true=oof_y_true,
        oof_y_pred=oof_y_pred,
    )

    return {
        "fold_mse": np.asarray(fold_mse, dtype=float),
        "fold_mae": np.asarray(fold_mae, dtype=float),
        "mean_mse": float(np.nanmean(fold_mse)),
        "mean_mae": float(np.nanmean(fold_mae)),
        "normalized_error": float(normalized_error),
        "normalization_factor": signal_length,
        "oof_test_index": oof_test_index,
        "oof_y_true": oof_y_true,
        "oof_y_pred": oof_y_pred,
        "oof_true_full": true_full,
        "oof_pred_full": pred_full,
        "oof_error_full": err_full,
        "predictions_per_fold": predictions_per_fold,
    }


def make_nan_decode_position(signal_length):
    return {
        "fold_mse": np.array([np.nan]),
        "mean_mse": np.nan,
        "oof_test_index": np.array([], dtype=int),
        "oof_y_true": np.array([]),
        "oof_y_pred": np.array([]),
        "oof_true_full": np.full(signal_length, np.nan, dtype=float),
        "oof_pred_full": np.full(signal_length, np.nan, dtype=float),
        "oof_error_full": np.full(signal_length, np.nan, dtype=float),
        "predictions_per_fold": [],
    }


def make_nan_decode_time(signal_length):
    return {
        "fold_mse": np.array([np.nan]),
        "fold_mae": np.array([np.nan]),
        "mean_mse": np.nan,
        "mean_mae": np.nan,
        "normalized_error": np.nan,
        "normalization_factor": signal_length,
        "oof_test_index": np.array([], dtype=int),
        "oof_y_true": np.array([]),
        "oof_y_pred": np.array([]),
        "oof_true_full": np.full(signal_length, np.nan, dtype=float),
        "oof_pred_full": np.full(signal_length, np.nan, dtype=float),
        "oof_error_full": np.full(signal_length, np.nan, dtype=float),
        "predictions_per_fold": [],
    }


# =============================================================================
# CSV schema
# =============================================================================

DECODER_SESSION_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "recording_type",
    "standardized_file_path",
    "row_index",
    "signal_dim",
    "n_timepoints_original",
    "n_timepoints_after_dir_filter",
    "n_splits",
    "split_mode",
    "decoder_type",
    "decoder_position",
    "decoder_position_metric",
    "decoder_position_direction",
    "decoder_position_direction_metric",
    "decoder_time",
    "decoder_time_metric",
    "time_normalization_factor",
    "session_pickle_name",
    "session_pickle_path",
]


# =============================================================================
# Main row processing
# =============================================================================

def process_row(row_idx, row, decoder_type, paths):
    standardized_file_path = row["standardized_file_path"]

    if not os.path.isfile(standardized_file_path):
        raise FileNotFoundError(f"Missing standardized file: {standardized_file_path}")

    data = load_pickle_standard(standardized_file_path)

    required_keys = ["signal", "pos", "mov_dir", "time"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in file: {standardized_file_path}")

    signal = ensure_2d(data["signal"])
    pos_1d = ensure_1d(get_pos_1d(data["pos"]))
    mov_dir = ensure_1d(data["mov_dir"])
    time_1d = ensure_1d(data["time"])
    trial_id = ensure_1d(data["trial_id"]) if data.get("trial_id", None) is not None else None

    position_direction = pos_1d * mov_dir

    beh_dict = {
        "position": pos_1d,
        "position_direction": position_direction,
        "time": time_1d,
        "direction": mov_dir,
        "trial_id": trial_id,
    }

    validate_lengths(signal, beh_dict, row_idx)

    n_timepoints_original = signal.shape[0]
    signal_dim = signal.shape[1]

    valid_idx = mov_dir != 0
    signal_valid = signal[valid_idx, :]

    beh_valid = {}
    for key, val in beh_dict.items():
        beh_valid[key] = None if val is None else val[valid_idx]

    n_timepoints_after_dir_filter = signal_valid.shape[0]
    if n_timepoints_after_dir_filter < MIN_TIMEPOINTS:
        raise ValueError(
            f"Row {row_idx}: too few timepoints after mov_dir != 0 filtering "
            f"({n_timepoints_after_dir_filter})"
        )

    if USE_TRIAL_ID_SPLIT:
        splits, split_mode = build_repeated_twofold_splits(
            n_samples=n_timepoints_after_dir_filter,
            trial_signal=beh_valid["trial_id"],
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
        )
    else:
        splits, split_mode = build_repeated_twofold_splits(
            n_samples=n_timepoints_after_dir_filter,
            trial_signal=None,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
        )

    if len(splits) < 1:
        decode_position = make_nan_decode_position(n_timepoints_after_dir_filter)
        decode_position_direction = make_nan_decode_position(n_timepoints_after_dir_filter)
        decode_time = make_nan_decode_time(n_timepoints_after_dir_filter)
    else:
        decode_position = run_regressor_oof(
            signal_valid,
            beh_valid["position"],
            splits=splits,
            decoder_type=decoder_type,
        )

        decode_position_direction = run_regressor_oof(
            signal_valid,
            beh_valid["position_direction"],
            splits=splits,
            decoder_type=decoder_type,
        )

        decode_time = run_time_decoder_normalized(
            signal_valid,
            beh_valid["time"],
            splits=splits,
            signal_length=n_timepoints_after_dir_filter,
            decoder_type=decoder_type,
        )

    recording_type = infer_recording_type(row)

    session_base = base_name_from_row(row_idx, row) + f"_decoder_{decoder_type}"
    session_pickle_name = session_base + ".pkl"
    session_pickle_path = os.path.join(paths["session_pkl_dir"], session_pickle_name)

    session_payload = {
        "metadata": {
            "dataset_name": row["dataset_name"],
            "mouse_name": row["mouse_name"],
            "session_name": row["session_name"],
            "depth_type": row["depth_type"],
            "recording_type": recording_type,
            "row_index": row_idx,
            "standardized_file_path": standardized_file_path,
            "signal_dim": signal_dim,
            "n_timepoints_original": n_timepoints_original,
            "n_timepoints_after_dir_filter": n_timepoints_after_dir_filter,
            "n_splits_requested": N_SPLITS,
            "n_splits_obtained": len(splits),
            "split_mode": split_mode,
            "random_state": RANDOM_STATE,
            "use_trial_id_split": USE_TRIAL_ID_SPLIT,
            "decoder_type": decoder_type,
            "rf_n_estimators": RF_N_ESTIMATORS if decoder_type == "rf" else None,
            "rf_random_state": RF_RANDOM_STATE if decoder_type == "rf" else None,
            "rf_n_jobs": RF_N_JOBS if decoder_type == "rf" else None,
            "xgb_max_depth": XGB_MAX_DEPTH if decoder_type == "xgb" else None,
            "xgb_num_round": XGB_NUM_ROUND if decoder_type == "xgb" else None,
            "xgb_eta": XGB_ETA if decoder_type == "xgb" else None,
            "xgb_gpu": XGB_GPU if decoder_type == "xgb" else None,
            "xgb_early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS if decoder_type == "xgb" else None,
            "wc_degree": WC_DEGREE if decoder_type == "wc" else None,
            "svr_c": SVR_C if decoder_type == "svr" else None,
            "svr_max_iter": SVR_MAX_ITER if decoder_type == "svr" else None,
        },
        "indices": {
            "valid_idx_mov_dir_nonzero": valid_idx,
        },
        "behavior": {
            "position": copy.deepcopy(beh_valid["position"]),
            "position_direction": copy.deepcopy(beh_valid["position_direction"]),
            "time": copy.deepcopy(beh_valid["time"]),
            "direction": copy.deepcopy(beh_valid["direction"]),
            "trial_id": copy.deepcopy(beh_valid["trial_id"]),
        },
        "splits": [
            {
                "train_index": train_idx.copy(),
                "test_index": test_idx.copy(),
            }
            for train_idx, test_idx in splits
        ],
        "decoders": {
            "position": copy.deepcopy(decode_position),
            "position_direction": copy.deepcopy(decode_position_direction),
            "time": copy.deepcopy(decode_time),
        },
    }

    save_pickle_standard(session_pickle_path, session_payload)

    title_prefix = (
        f"{row['dataset_name']} | {row['mouse_name']} | "
        f"{row['session_name']} | {row['depth_type']} | {decoder_type}"
    )
    plot_base = base_name_from_row(row_idx, row) + f"_{decoder_type}"

    plot_predicted_vs_true(
        decode_position,
        "position",
        title_prefix,
        f"{plot_base}_position_pred_vs_true",
        paths["png_dir"],
        paths["svg_dir"],
    )
    plot_predicted_vs_true(
        decode_position_direction,
        "position_direction",
        title_prefix,
        f"{plot_base}_position_direction_pred_vs_true",
        paths["png_dir"],
        paths["svg_dir"],
    )
    plot_predicted_vs_true(
        decode_time,
        "time",
        title_prefix,
        f"{plot_base}_time_pred_vs_true",
        paths["png_dir"],
        paths["svg_dir"],
    )

    session_row = {
        "dataset_name": row["dataset_name"],
        "mouse_name": row["mouse_name"],
        "session_name": row["session_name"],
        "depth_type": row["depth_type"],
        "recording_type": recording_type,
        "standardized_file_path": standardized_file_path,
        "row_index": row_idx,
        "signal_dim": signal_dim,
        "n_timepoints_original": n_timepoints_original,
        "n_timepoints_after_dir_filter": n_timepoints_after_dir_filter,
        "n_splits": len(splits),
        "split_mode": split_mode,
        "decoder_type": decoder_type,
        "decoder_position": decode_position["mean_mse"],
        "decoder_position_metric": "mean_squared_error",
        "decoder_position_direction": decode_position_direction["mean_mse"],
        "decoder_position_direction_metric": "mean_squared_error",
        "decoder_time": decode_time["normalized_error"],
        "decoder_time_metric": "normalized_mae_by_signal_length_oof",
        "time_normalization_factor": decode_time["normalization_factor"],
        "session_pickle_name": session_pickle_name,
        "session_pickle_path": session_pickle_path,
    }

    if VERBOSE:
        print(
            f"Row {row_idx} | dec={decoder_type} | dataset={row['dataset_name']} | "
            f"mouse={row['mouse_name']} | session={row['session_name']} | "
            f"depth={row['depth_type']} | pos_mse={session_row['decoder_position']:.6f} | "
            f"posdir_mse={session_row['decoder_position_direction']:.6f} | "
            f"time_norm_err={session_row['decoder_time']:.6f} | "
            f"splits={len(splits)} | mode={split_mode}"
        )

    return session_row


# =============================================================================
# Main
# =============================================================================

def run_for_decoder_type(decoder_type):
    paths = get_output_paths_for_decoder(decoder_type)

    ensure_dir(paths["output_dir"])
    ensure_dir(paths["session_pkl_dir"])
    ensure_dir(paths["log_dir"])
    ensure_dir(paths["plots_dir"])
    ensure_dir(paths["png_dir"])
    ensure_dir(paths["svg_dir"])

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    init_csv_if_needed(paths["decoder_session_csv"], DECODER_SESSION_COLUMNS)
    already_done = load_already_processed_paths(paths["decoder_session_csv"])

    log_path = os.path.join(paths["log_dir"], f"decoder_{decoder_type}_log.txt")
    log_file = open(log_path, "a", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print(f"Reduced decoder analysis pipeline ({decoder_type})")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Rows to process: {len(df)}")
    print(f"Already processed sessions in CSV: {len(already_done)}")
    print(f"Output dir: {paths['output_dir']}")
    print(f"Session CSV: {paths['decoder_session_csv']}")
    print(f"Session PKL dir: {paths['session_pkl_dir']}")
    print(f"Pred-vs-true plots: {paths['plots_dir']}")
    print(f"N_SPLITS requested: {N_SPLITS}")
    print(f"USE_TRIAL_ID_SPLIT: {USE_TRIAL_ID_SPLIT}")
    print("")

    ok_count = 0
    skip_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        standardized_file_path = str(row.get("standardized_file_path", ""))

        if standardized_file_path in already_done:
            skip_count += 1
            print(f"[SKIP] Row {row_idx}: already present in CSV\n")
            continue

        local_time = timeit.default_timer()
        try:
            session_row = process_row(row_idx, row, decoder_type, paths)
            append_csv_row(paths["decoder_session_csv"], DECODER_SESSION_COLUMNS, session_row)
            ok_count += 1

            elapsed = timeit.default_timer() - local_time
            print(f"[OK] Row {row_idx} finished in {elapsed:.2f} s\n")

        except Exception as e:
            fail_count += 1
            print(f"[ERROR] Row {row_idx}: {e}\n")

    total_elapsed = timeit.default_timer() - global_time
    print("==============================================================")
    print("Done")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Successful entries: {ok_count}")
    print(f"Skipped entries: {skip_count}")
    print(f"Failed entries: {fail_count}")
    print(f"Total elapsed time: {total_elapsed:.2f} s")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


def main():
    for decoder_type in DECODER_TYPES:
        run_for_decoder_type(decoder_type)


if __name__ == "__main__":
    main()