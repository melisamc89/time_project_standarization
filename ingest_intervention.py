#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_intervention.py  (versión todo-en-uno)

Hace TODO el proceso de los datos "intervention" en un solo script:

  PASO 1  (antes create_data_base.py)
      Localiza los archivos de cada sesión, divide las trazas de calcio en
      basal / sinusoidal, junta el tracking y crea manifest_database.csv.

  PASO 2  (antes preprocess_data.py)
      Preprocesa trazas de calcio y tracking (Kalman, posición lineal,
      velocidad, dirección...), genera los plots de trazas y guarda un pickle
      por sesión + processed_database.csv.

  PASO 3  (el ingest)
      Convierte cada sesión al formato estandarizado y añade sus filas a
      master_dataset_index.csv (con int_i e int_f en columnas propias).

USO normal:        python ingest_intervention.py

Opciones (todas opcionales):
    --force            rehace todo, ignorando lo que ya estaba calculado
    --no-plots         no genera los plots de trazas (más rápido)
    --mouse 032 521    solo esos ratones
    --session 1 2      solo esas sesiones
    --date 1707        solo esas fechas

Los pasos 1 y 2 guardan lo que ya han calculado, así que si vuelves a
ejecutar el script solo se calcula lo que falte (por ejemplo, una sesión nueva).
"""

from __future__ import annotations

import argparse
import pickle
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # guarda las figuras sin intentar abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA


# =============================================================================
# RUTAS Y PARÁMETROS  (aquí es donde tienes que tocar cosas si cambian las rutas)
# =============================================================================

# --- Datos originales y resultados intermedios (pasos 1 y 2) ---
BASE_DIR = Path("/home/hippolab/Documents/melisa/time_project_intervention")
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "databases"
SPLIT_DATA_DIR = BASE_DIR / "split_data"
PROCESSED_PICKLE_DIR = BASE_DIR / "processed_pickles"
PLOTTING_TRACES_DIR = BASE_DIR / "plotting_traces"

MANIFEST_CSV = DATABASE_DIR / "manifest_database.csv"
PROCESSED_CSV = DATABASE_DIR / "processed_database.csv"

# --- Proyecto de estandarización (paso 3; el mismo que usa ingest_learning) ---
PROJECT_ROOT = Path("/home/hippolab/Documents/melisa/time_project_standarization")

DATASET_NAME = "intervention"
MODALITY = "calcium_imaging"
DEPTH_TYPE = "whole_population"
CASE = "intervention"
CELLS = "all_cells"
SIGNAL_NAME = "retained_traces"

# Qué condiciones se convierten al formato estándar.
# "combined" es basal + sinusoidal pegadas. Borra "combined" si no la quieres.
CONDITIONS = ["basal", "sinusoidal", "combined"]

# False = si el archivo estandarizado ya existe, no lo vuelve a crear
# (con --force sí se rehace)
OVERWRITE_STANDARDIZED_FILES = False

DATA_ROOT = PROJECT_ROOT / "data"
METADATA_DIR = DATA_ROOT / "metadata"
STANDARDIZED_DIR = DATA_ROOT / "standardized_sessions" / DATASET_NAME / MODALITY / DEPTH_TYPE
MASTER_CSV_PATH = METADATA_DIR / "master_dataset_index.csv"

# Mismas columnas que ingest_learning_dataset.py + int_i e int_f
MASTER_COLUMNS = [
    "dataset_name", "modality", "depth_type", "mouse_name", "session_name",
    "session_name_safe", "source_base_dir", "source_data_dir", "source_mouse_dir",
    "source_file_name", "source_file_path", "case", "cells", "signal_name",
    "paradigm", "model", "int_i", "int_f",
    "standardized_file_name", "standardized_file_path",
    "n_timepoints", "n_neurons", "pos_dim", "contains_signal", "contains_pos",
    "contains_mov_dir", "contains_speed", "contains_trial_id", "contains_time",
    "contains_inner_time", "source_container_type", "common_crop_length",
    "created_at", "notes",
]

# --- Parámetros del preprocesado (paso 2) ---
PARAMS = {
    "body_parts": ("neck", "back", "tail"),
    "likelihood_threshold": 0.95,
    "arena_spatial_extent_meters": 1.0,
    "track_edge_exclusion_fraction": 0.075,
    "accepted_cell_label": "accepted",
    "peak_to_noise_threshold": 3.0,
    "kalman_process_acceleration_std": 2.0,
    "kalman_measurement_position_std": 0.01,
    "speed_threshold_m_s": 0.02,
    "direction_velocity_epsilon": 0.002,
    # Deconvolución
    "sig_filt": 5.0,
    "sig_up": 4.0,
    "sig_down": 12.0,
    "peak_th": 0.1,
    "debleach_uniform_filter_size": 4000,
    "debleach_bins": 100,
    # Suavizado después de seleccionar los frames
    "post_selection_gaussian_sigma_frames": 6.0,
    "post_selection_gaussian_mode": "nearest",
}

# int_i / int_f por defecto según la fecha de grabación (formato DDMM)
DATE_PARAMS_MAP = {
    "1707": {"int_i": 3.0, "int_f": 0.17},
    "2309": {"int_i": 1.5, "int_f": 0.17},
    "2509": {"int_i": 1.5, "int_f": 0.17},
    "0110": {"int_i": 1.5, "int_f": 1},
    "0910": {"int_i": 1.5, "int_f": 1},
}

# =============================================================================
# INVENTARIO COMPLETO DE ARCHIVOS (si hay sesiones nuevas, se añaden aquí)
# =============================================================================
SESSION_REGISTRY = [
    # Mouse 032 - 1707 (Session 1)
    {
        "mouse_id": "032",
        "date": "1707",
        "traces": "BS_032_1707_traces.csv",
        "basal_avi": "032Basal-temp-07172026124813-0000.avi",
        "basal_tracking": "032Basal-temp-07172026124813-0000DLC_resnet50_TimeJul22shuffle1_500000.csv",
        "sinus_avi": "032Sinus-temp-07172026141811-0000.avi",
        "sinus_tracking": "032Sinus-temp-07172026141811-0000DLC_resnet50_TimeJul22shuffle1_500000.csv",
    },
    # Mouse 032 - 2309 (Session 2)
    {
        "mouse_id": "032",
        "date": "2309",
        "traces": "BS_032_2309_traces.csv",
        "basal_avi": "032Basal-temp-09232026162619-0000.avi",
        "basal_tracking": "032Basal-temp-09232026162619-0000DLC_Resnet50_linear_interventionOct1shuffle1_snapshot_best-110.csv",
        "sinus_avi": "032Post-temp-09232026165714-0000.avi",  # Sinus/Post video available for this date
        "sinus_tracking": "032Sinus-temp-09232026164148-0000DLC_Resnet50_linear_interventionOct1shuffle1_snapshot_best-110.csv",
    },
    # Mouse 032 - 0110 (Session 3)
    {
        "mouse_id": "032",
        "date": "0110",
        "traces": "BS_032_0110_traces.csv",
        "basal_avi": "032Basal-temp-10012026154059-0000.avi",
        "basal_tracking": "032Basal-temp-10012026154059-0000DLC_Resnet50_linear_interventionOct1shuffle1_snapshot_best-110.csv",
        "sinus_avi": "032Sinus-temp-10012026160003-0000.avi",  # Sinus/Post video available for this date
        "sinus_tracking": "032Sinus-temp-10012026160003-0000DLC_Resnet50_linear_interventionOct1shuffle1_snapshot_best-110.csv",
    },
    # Mouse 520 - 1707 (Session 1)
    {
        "mouse_id": "520",
        "date": "1707",
        "traces": "BS_520_1707_traces.csv",
        "basal_avi": "520Basal-temp-07172026113024-0000.avi",
        "basal_tracking": "520Basal-temp-07172026113024-0000DLC_resnet50_TimeJul22shuffle1_500000.csv",
        "sinus_avi": "520Sinus-temp-07172026114508-0000.avi",
        "sinus_tracking": "520Sinus-temp-07172026114508-0000DLC_resnet50_TimeJul22shuffle1_500000.csv",
    },
    # Mouse 520 - 0910 (Session 2)
    {
        "mouse_id": "520",
        "date": "0910",
        "traces": "BS_520_0910_traces.csv",
        "basal_avi": "520Basal-temp-10092026160403-0000.avi",
        "basal_tracking": "520Basal-temp-10092026160403-0000DLC_Resnet50_linear_interventionOct1shuffle1_snapshot_best-110.csv",
        "sinus_avi": "520Sinus-temp-10092026162107-0000.avi",
        "sinus_tracking": "520Sinus-temp-10092026162107-0000DLC_Resnet50_linear_interventionOct1shuffle1_snapshot_best-110.csv",
    },
    # Mouse 521 - 1707 (Session 1)
    {
        "mouse_id": "521",
        "date": "1707",
        "traces": "BS_521_1707_traces.csv",
        "basal_avi": "521Basal-temp-07172026120154-0000.avi",
        "basal_tracking": "521Basal-temp-07172026120154-0000DLC_resnet50_TimeJul22shuffle1_500000.csv",
        "sinus_avi": "521Sinus-temp-07172026123237-0000.avi",
        "sinus_tracking": "521Sinus-temp-07172026123237-0000DLC_resnet50_TimeJul22shuffle1_500000.csv",
    },
    # Mouse 521 - 2509 (Session 2)
    {
        "mouse_id": "521",
        "date": "2509",
        "traces": "BS_521_2509_traces.csv",
        "basal_avi": "521Basal-temp-09252026163404-0000.avi",
        "basal_tracking": "521Basal-temp-09252026163404-0000DLC_Resnet50_linear_interventionOct1shuffle1_snapshot_best-110.csv",
        "sinus_avi": "521Sinus-temp-09252026165000-0000.avi",
        "sinus_tracking": "521Sinus-temp-09252026165000-0000DLC_Resnet50_linear_interventionOct1shuffle1_snapshot_best-110.csv",
    },
    # Mouse 521 - 0110 (Session 3)
    {
        "mouse_id": "521",
        "date": "0110",
        "traces": "BS_521_0110_traces.csv",
        "basal_avi": "521Basal-temp-10012026163220-0000.avi",
        "basal_tracking": "521Basal-temp-10012026163220-0000DLC_Resnet50_linear_interventionOct1shuffle1_snapshot_best-110.csv",
        "sinus_avi": "521Sinus-temp-10012026171354-0000.avi",
        "sinus_tracking": "521Sinus-temp-10012026171354-0000DLC_Resnet50_linear_interventionOct1shuffle1_snapshot_best-110.csv",
    },
]


# =============================================================================
# FUNCIONES AUXILIARES GENERALES
# =============================================================================

def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_for_filename(text) -> str:
    """Cambia cualquier carácter raro por '_' para usarlo en nombres de archivo."""
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(text))


def yes_no(x) -> str:
    return "yes" if x else "no"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def banner(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


# =============================================================================
# PASO 1: MANIFEST (antes create_data_base.py)
# =============================================================================

def resolve_file(directory: Path, filename: Optional[str]) -> Optional[Path]:
    """Devuelve la ruta del archivo, o None si no existe."""
    if not filename:
        return None
    path = directory / filename
    if path.exists():
        return path
    matches = list(directory.glob(filename))
    if matches:
        return sorted(matches)[0]
    return None


def parse_date_to_sort_key(date_str: str) -> int:
    """Convierte DDMM (p. ej. 1707 -> día 17, mes 07) en un número MMDD para ordenar."""
    day = int(date_str[:2])
    month = int(date_str[2:4])
    return month * 100 + day


def build_manifest_database(force: bool = False) -> pd.DataFrame:
    # Ordena las sesiones cronológicamente por ratón y las numera 1, 2, 3...
    reg_df = pd.DataFrame(SESSION_REGISTRY)
    reg_df["sort_date"] = reg_df["date"].apply(parse_date_to_sort_key)
    reg_df = reg_df.sort_values(by=["mouse_id", "sort_date"]).reset_index(drop=True)
    reg_df["session"] = reg_df.groupby("mouse_id").cumcount() + 1

    records = []

    for _, entry in reg_df.iterrows():
        mouse_id = str(entry["mouse_id"])
        date_str = str(entry["date"])
        session_idx = int(entry["session"])

        print(f"\n--- Manifest: ratón {mouse_id} | fecha {date_str} (sesión {session_idx}) ---")
        out_session_dir = SPLIT_DATA_DIR / f"mouse_{mouse_id}_session_{session_idx}_{date_str}"
        out_session_dir.mkdir(parents=True, exist_ok=True)

        traces_path = resolve_file(DATA_DIR, entry["traces"])
        basal_avi_path = resolve_file(DATA_DIR, entry["basal_avi"])
        basal_track_path = resolve_file(DATA_DIR, entry["basal_tracking"])
        sinus_avi_path = resolve_file(DATA_DIR, entry["sinus_avi"])
        sinus_track_path = resolve_file(DATA_DIR, entry["sinus_tracking"])

        for p, label, fname in [
            (traces_path, "trazas de calcio", entry["traces"]),
            (basal_track_path, "tracking basal", entry["basal_tracking"]),
            (sinus_track_path, "tracking sinusoidal", entry["sinus_tracking"]),
        ]:
            if p is None:
                raise FileNotFoundError(
                    f"No encuentro el archivo de {label}:\n  {fname}\n  (buscado en {DATA_DIR})"
                )

        params = DATE_PARAMS_MAP.get(date_str, {"int_i": 0.0, "int_f": 0.0})
        int_i_val = params["int_i"]
        int_f_val = params["int_f"]

        basal_calcium_file = out_session_dir / f"BS_{mouse_id}_{date_str}_traces_basal.csv"
        sinus_calcium_file = out_session_dir / f"BS_{mouse_id}_{date_str}_traces_sinusoidal.csv"
        combined_track_file = out_session_dir / f"{mouse_id}_{date_str}_combined_tracking.csv"

        already_split = (
            basal_calcium_file.exists() and sinus_calcium_file.exists() and combined_track_file.exists()
        )

        if already_split and not force:
            print("  Archivos divididos ya existen, no los vuelvo a crear")
        else:
            print(f"  Leyendo tracking basal: {basal_track_path.name}")
            basal_track_df = pd.read_csv(basal_track_path, header=[0, 1, 2], index_col=0, low_memory=False)
            basal_len = len(basal_track_df)

            print(f"  Leyendo tracking sinusoidal: {sinus_track_path.name}")
            sinus_track_df = pd.read_csv(sinus_track_path, header=[0, 1, 2], index_col=0, low_memory=False)

            print(f"  Leyendo trazas de calcio: {traces_path.name}")
            calcium_df = pd.read_csv(traces_path, header=[0, 1], skipinitialspace=True, low_memory=False)

            if basal_len >= len(calcium_df):
                raise ValueError(
                    f"El tracking basal ({basal_len} frames) es igual o más largo que las trazas "
                    f"de calcio completas ({len(calcium_df)} frames) en ratón {mouse_id}, {date_str}."
                )

            # Divide las trazas en basal / sinusoidal
            basal_calcium_df = calcium_df.iloc[:basal_len].copy()
            sinus_calcium_df = calcium_df.iloc[basal_len:].copy()
            basal_calcium_df.to_csv(basal_calcium_file, index=False)
            sinus_calcium_df.to_csv(sinus_calcium_file, index=False)
            print(f"  -> trazas basal: {basal_calcium_file.name} ({len(basal_calcium_df)} frames)")
            print(f"  -> trazas sinusoidal: {sinus_calcium_file.name} ({len(sinus_calcium_df)} frames)")

            # Tracking combinado
            combined_track_df = pd.concat([basal_track_df, sinus_track_df], axis=0)
            combined_track_df.to_csv(combined_track_file)
            print(f"  -> tracking combinado: {combined_track_file.name} ({len(combined_track_df)} frames)")

        records.append({
            "mouse_id": mouse_id, "session": session_idx, "date": date_str,
            "condition": "basal", "int_i": 0.0, "int_f": 0.0,
            "path_avi": str(basal_avi_path) if basal_avi_path else None,
            "path_traces": str(basal_calcium_file),
            "path_tracking": str(basal_track_path),
        })
        records.append({
            "mouse_id": mouse_id, "session": session_idx, "date": date_str,
            "condition": "sinusoidal", "int_i": float(int_i_val), "int_f": float(int_f_val),
            "path_avi": str(sinus_avi_path) if sinus_avi_path else None,
            "path_traces": str(sinus_calcium_file),
            "path_tracking": str(sinus_track_path),
        })
        records.append({
            "mouse_id": mouse_id, "session": session_idx, "date": date_str,
            "condition": "combined", "int_i": float(int_i_val), "int_f": float(int_f_val),
            "path_avi": None,
            "path_traces": str(traces_path),
            "path_tracking": str(combined_track_file),
        })

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(MANIFEST_CSV, index=False)

    banner(f"Manifest guardado en:\n  {MANIFEST_CSV}")
    print(manifest_df[["mouse_id", "session", "date", "condition", "int_i", "int_f"]].to_string())
    return manifest_df


# =============================================================================
# PASO 2: PREPROCESADO (antes preprocess_data.py)
# =============================================================================

def preprocess_traces(
    traces: np.ndarray,
    sig_filt: float = 5.0,
    sig_up: float = 4.0,
    sig_down: float = 12.0,
    peak_th: float = 0.1,
) -> np.ndarray:
    """
    Debleaching, umbral y convolución con kernel gaussiano asimétrico:
    1. Filtro uniforme (boxcar) para estimar el photobleaching.
    2. Suavizado gaussiano (sigma = sig_filt).
    3. Alineación de la línea base por la moda, normalización y recorte (< 0 -> 0).
    4. Detección de picos (umbral peak_th) y convolución con el kernel asimétrico.
    """
    traces = np.asarray(traces, dtype=float).copy()
    lp_traces = uniform_filter1d(traces, size=int(PARAMS["debleach_uniform_filter_size"]), axis=0)
    clean_traces = gaussian_filter1d(traces, sigma=float(sig_filt), axis=0)

    for cell in range(clean_traces.shape[1]):
        hist_counts, hist_edges = np.histogram(traces[:, cell], int(PARAMS["debleach_bins"]))
        bleaching = hist_edges[np.argmax(hist_counts)]
        bleaching = bleaching + lp_traces[:, cell] - np.min(lp_traces[:, cell])

        clean_traces[:, cell] = clean_traces[:, cell] - bleaching
        max_val = np.max(clean_traces[:, cell], axis=0)
        if max_val > 0:
            clean_traces[:, cell] = clean_traces[:, cell] / max_val

    clean_traces[clean_traces < 0] = 0.0
    conv_traces = np.zeros_like(clean_traces)

    # Kernel gaussiano asimétrico
    gaus = lambda x, sig, amp, vo: amp * np.exp(-((x ** 2) / (2.0 * sig ** 2))) + vo
    half_w = int(round(5.0 * sig_down))
    x = np.arange(-half_w, half_w, 1, dtype=float)

    left_gauss = gaus(x, sig_up, 1.0, 0.0)
    left_gauss[half_w + 1:] = 0.0

    right_gauss = gaus(x, sig_down, 1.0, 0.0)
    right_gauss[:half_w + 1] = 0.0

    gaus_kernel = right_gauss + left_gauss

    for cell in range(clean_traces.shape[1]):
        peak_idx, _ = find_peaks(clean_traces[:, cell], height=float(peak_th))
        conv_traces[peak_idx, cell] = clean_traces[peak_idx, cell]
        conv_traces[:, cell] = np.convolve(conv_traces[:, cell], gaus_kernel, mode="same")

    return conv_traces


def interpolate_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("No se puede interpolar un vector completamente vacío.")
    if finite.all():
        return values.copy()
    x = np.arange(len(values), dtype=float)
    return np.interp(x, x[finite], values[finite])


def calculate_peak_to_noise_ratio(traces: np.ndarray) -> np.ndarray:
    trace_median = np.median(traces, axis=0)
    peak_amplitude = np.max(traces, axis=0) - trace_median
    robust_noise = 1.4826 * np.median(np.abs(traces - trace_median[None, :]), axis=0)
    ratio = np.full(traces.shape[1], np.nan, dtype=float)
    pos = robust_noise > np.finfo(float).eps
    ratio[pos] = peak_amplitude[pos] / robust_noise[pos]
    ratio[~pos & (peak_amplitude > 0)] = np.inf
    ratio[~pos & (peak_amplitude <= 0)] = 0.0
    return ratio


def read_calcium(file_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(file_path, header=[0, 1], skipinitialspace=True, low_memory=False)
    time_seconds = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)

    cell_cols = list(df.columns[1:])
    cell_names = [str(c[0]).strip() for c in cell_cols]
    cell_status = [str(c[1]).strip().lower() for c in cell_cols]

    traces = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).copy()

    usable_cells = ~np.isnan(traces).all(axis=0)
    if not usable_cells.all():
        traces = traces[:, usable_cells]
        cell_names = [n for n, keep in zip(cell_names, usable_cells) if keep]
        cell_status = [s for s, keep in zip(cell_status, usable_cells) if keep]

    for idx in range(traces.shape[1]):
        if np.isnan(traces[:, idx]).any():
            traces[:, idx] = interpolate_vector(traces[:, idx])

    accepted_mask = np.array([s == PARAMS["accepted_cell_label"] for s in cell_status])
    pnr = calculate_peak_to_noise_ratio(traces)
    pnr_mask = (np.isfinite(pnr) & (pnr > PARAMS["peak_to_noise_threshold"])) | np.isposinf(pnr)
    keep_mask = accepted_mask & pnr_mask

    selected_traces = traces[:, keep_mask].copy()
    selected_names = [name for name, k in zip(cell_names, keep_mask) if k]
    return time_seconds, selected_traces, selected_names


def read_dlc(file_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(file_path, header=[0, 1, 2], index_col=0, skipinitialspace=True, low_memory=False)
    frame_idx = pd.to_numeric(df.index, errors="coerce").to_numpy(dtype=float)
    return df, frame_idx


def extract_dlc_coord(df: pd.DataFrame, bp: str, coord: str) -> np.ndarray:
    matches = [c for c in df.columns if str(c[1]).strip() == bp and str(c[2]).strip().lower() == coord.lower()]
    return pd.to_numeric(df[matches[0]], errors="coerce").to_numpy(dtype=float)


def kalman_smooth(time_s: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(time_s)
    obs = np.column_stack([x_m, y_m])
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    R = (PARAMS["kalman_measurement_position_std"] ** 2) * np.eye(2)
    I = np.eye(4)

    filtered_states = np.zeros((n, 4))
    filtered_covs = np.zeros((n, 4, 4))
    predicted_states = np.zeros((n, 4))
    predicted_covs = np.zeros((n, 4, 4))
    transition_mats = np.zeros((n, 4, 4))

    dt0 = time_s[1] - time_s[0]
    curr_state = np.array([x_m[0], y_m[0], (x_m[1] - x_m[0]) / dt0, (y_m[1] - y_m[0]) / dt0])
    curr_cov = np.diag([PARAMS["kalman_measurement_position_std"] ** 2] * 2 + [1.0, 1.0])

    filtered_states[0], filtered_covs[0] = curr_state, curr_cov
    predicted_states[0], predicted_covs[0] = curr_state, curr_cov
    transition_mats[0] = I

    for i in range(1, n):
        dt = time_s[i] - time_s[i - 1]
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q_var = PARAMS["kalman_process_acceleration_std"] ** 2
        Q = q_var * np.array([
            [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
            [dt ** 3 / 2, 0, dt ** 2, 0],
            [0, dt ** 3 / 2, 0, dt ** 2],
        ])
        transition_mats[i] = F
        p_state = F @ curr_state
        p_cov = F @ curr_cov @ F.T + Q
        predicted_states[i], predicted_covs[i] = p_state, p_cov

        innov = obs[i] - H @ p_state
        s_cov = H @ p_cov @ H.T + R
        K = p_cov @ H.T @ np.linalg.pinv(s_cov)
        curr_state = p_state + K @ innov
        curr_cov = (I - K @ H) @ p_cov @ (I - K @ H).T + K @ R @ K.T
        filtered_states[i], filtered_covs[i] = curr_state, curr_cov

    smoothed_states = filtered_states.copy()
    for i in range(n - 2, -1, -1):
        F = transition_mats[i + 1]
        C = filtered_covs[i] @ F.T @ np.linalg.pinv(predicted_covs[i + 1])
        smoothed_states[i] += C @ (smoothed_states[i + 1] - predicted_states[i + 1])

    return smoothed_states[:, 0], smoothed_states[:, 1], smoothed_states[:, 2], smoothed_states[:, 3]


def plot_raw_vs_processed_waterfall(
    time_s: np.ndarray,
    raw_traces: np.ndarray,
    conv_traces: np.ndarray,
    cell_names: List[str],
    mouse_id: str,
    session_idx: int,
    date_str: str,
    condition: str,
    max_cells_to_plot: Optional[int] = 50,
    zoom_window_s: Optional[float] = 60.0,
):
    n_cells_total = raw_traces.shape[1]
    n_plot = min(n_cells_total, max_cells_to_plot) if max_cells_to_plot is not None else n_cells_total
    indices = np.arange(n_plot)

    color_processed = "#1f77b4" if condition == "basal" else "#ff7f0e"
    color_raw = "#a0a0a0"

    norm_raw = np.zeros((len(time_s), n_plot))
    norm_conv = np.zeros((len(time_s), n_plot))

    for i, c_idx in enumerate(indices):
        r = raw_traces[:, c_idx]
        ptp_r = np.ptp(r)
        norm_raw[:, i] = (r - np.nanmin(r)) / (ptp_r if ptp_r > 1e-12 else 1.0)

        p = conv_traces[:, c_idx]
        ptp_p = np.ptp(p)
        norm_conv[:, i] = (p - np.nanmin(p)) / (ptp_p if ptp_p > 1e-12 else 1.0)

    fig_h = max(8.0, 0.35 * n_plot)
    fig, axes = plt.subplots(1, 2 if zoom_window_s else 1, figsize=(22 if zoom_window_s else 14, fig_h), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    ax_full = axes[0]
    for i in range(n_plot):
        offset = i
        ax_full.plot(time_s, norm_raw[:, i] + offset, color=color_raw, linewidth=0.8, alpha=0.65,
                     label="Raw Traces" if i == 0 else "")
        ax_full.plot(time_s, norm_conv[:, i] + offset, color=color_processed, linewidth=1.1, alpha=0.9,
                     label="Processed Traces (Debleached & Convolved)" if i == 0 else "")

    ax_full.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax_full.set_ylabel("Neuron Index (Normalized [0, 1] + Offset)", fontsize=11, fontweight="bold")
    ax_full.set_yticks(np.arange(n_plot))
    step = max(1, n_plot // 25)
    ytick_labels = [cell_names[i] if i % step == 0 else "" for i in range(n_plot)]
    ax_full.set_yticklabels(ytick_labels, fontsize=8)
    ax_full.set_title(f"Full Session ({time_s[0]:.1f}s – {time_s[-1]:.1f}s)", fontsize=12, fontweight="bold")
    ax_full.set_ylim(-0.5, n_plot + 0.5)
    ax_full.set_xlim(time_s[0], time_s[-1])
    ax_full.grid(axis="y", linestyle=":", alpha=0.5)
    ax_full.legend(loc="upper right", frameon=True, fontsize=10)

    if zoom_window_s and len(axes) > 1:
        ax_zoom = axes[1]
        t_end_zoom = min(time_s[0] + zoom_window_s, time_s[-1])
        zoom_mask = (time_s >= time_s[0]) & (time_s <= t_end_zoom)

        for i in range(n_plot):
            offset = i
            ax_zoom.plot(time_s[zoom_mask], norm_raw[zoom_mask, i] + offset, color=color_raw, linewidth=1.0, alpha=0.7)
            ax_zoom.plot(time_s[zoom_mask], norm_conv[zoom_mask, i] + offset, color=color_processed, linewidth=1.4,
                         alpha=0.95)

        ax_zoom.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
        ax_zoom.set_xlim(time_s[0], t_end_zoom)
        ax_zoom.set_title(f"Zoomed Inspection Window ({time_s[0]:.1f}s – {t_end_zoom:.1f}s)", fontsize=12,
                          fontweight="bold")
        ax_zoom.grid(axis="y", linestyle=":", alpha=0.5)

    title_str = (
        f"Raw vs. Processed Calcium Traces (Before Speed Cut)\n"
        f"Mouse {mouse_id} | Session {session_idx} ({date_str}) | Condition: {condition.capitalize()} "
        f"({n_plot} / {n_cells_total} neurons shown)"
    )
    fig.suptitle(title_str, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

    out_file = PLOTTING_TRACES_DIR / f"traces_raw_vs_processed_M{mouse_id}_S{session_idx}_{date_str}_{condition}.png"
    fig.savefig(out_file, dpi=250)
    plt.close(fig)
    print(f"  [PLOT GUARDADO] {out_file.name}")


def process_single_condition(traces: np.ndarray, time_s: np.ndarray, track_df: pd.DataFrame,
                             frame_idx: np.ndarray) -> Dict:
    n_sync = min(len(time_s), len(track_df))
    time_s = time_s[:n_sync]
    traces = traces[:n_sync]
    track_df = track_df.iloc[:n_sync]
    frame_idx = frame_idx[:n_sync]

    conv_traces = preprocess_traces(
        traces,
        sig_filt=PARAMS["sig_filt"],
        sig_up=PARAMS["sig_up"],
        sig_down=PARAMS["sig_down"],
        peak_th=PARAMS["peak_th"],
    )

    n_frames = len(track_df)
    accepted_x = np.full((n_frames, len(PARAMS["body_parts"])), np.nan, dtype=float)
    accepted_y = np.full_like(accepted_x, np.nan)

    for bp_idx, bp in enumerate(PARAMS["body_parts"]):
        x = extract_dlc_coord(track_df, bp, "x")
        y = extract_dlc_coord(track_df, bp, "y")
        lh = extract_dlc_coord(track_df, bp, "likelihood")
        v = (lh > PARAMS["likelihood_threshold"]) & np.isfinite(x) & np.isfinite(y)
        accepted_x[v, bp_idx] = x[v]
        accepted_y[v, bp_idx] = y[v]

    valid_counts = np.sum(np.isfinite(accepted_x), axis=1)
    has_valid = valid_counts > 0
    x_mean = np.full(n_frames, np.nan, dtype=float)
    y_mean = np.full(n_frames, np.nan, dtype=float)
    x_mean[has_valid] = np.nansum(accepted_x, axis=1)[has_valid] / valid_counts[has_valid]
    y_mean[has_valid] = np.nansum(accepted_y, axis=1)[has_valid] / valid_counts[has_valid]

    x_mean = interpolate_vector(x_mean)
    y_mean = interpolate_vector(y_mean)

    extent_px = np.hypot(np.nanmax(x_mean) - np.nanmin(x_mean), np.nanmax(y_mean) - np.nanmin(y_mean))
    ppm = extent_px / PARAMS["arena_spatial_extent_meters"]
    x_m = (x_mean - np.nanmin(x_mean)) / ppm
    y_m = (y_mean - np.nanmin(y_mean)) / ppm

    xs, ys, vxs, vys = kalman_smooth(time_s, x_m, y_m)
    speed = np.hypot(vxs, vys)

    proj = PCA(n_components=1).fit_transform(np.column_stack([xs, ys])).ravel()
    lin_pos = (proj - np.min(proj)) / (np.max(proj) - np.min(proj))

    vel = np.gradient(lin_pos, time_s)
    direction = np.zeros(len(lin_pos), dtype=np.int8)
    direction[vel > PARAMS["direction_velocity_epsilon"]] = 1
    direction[vel < -PARAMS["direction_velocity_epsilon"]] = -1

    speed_mask = speed > PARAMS["speed_threshold_m_s"]
    border_mask = (lin_pos > PARAMS["track_edge_exclusion_fraction"]) & (
            lin_pos < 1.0 - PARAMS["track_edge_exclusion_fraction"]
    )
    keep_mask = speed_mask & border_mask
    ret_idx = np.flatnonzero(keep_mask)

    ret_traces_raw = conv_traces[ret_idx]
    ret_traces_filtered = gaussian_filter1d(
        ret_traces_raw,
        sigma=PARAMS["post_selection_gaussian_sigma_frames"],
        axis=0,
        mode=PARAMS["post_selection_gaussian_mode"],
    )

    return {
        "sync_time": time_s,
        "sync_indices": ret_idx,
        "original_frames": frame_idx[ret_idx],
        "all_traces_conv": conv_traces,
        "retained_traces": ret_traces_filtered,
        "position_1d": lin_pos[ret_idx],
        "speed": speed[ret_idx],
        "direction": direction[ret_idx],
        "keep_mask": keep_mask,
    }


def preprocess_one_session(s_df: pd.DataFrame, mouse_id: str, session_idx: int, date_str: str,
                           plot_traces: bool) -> Dict:
    """Preprocesa basal + sinusoidal de una sesión y devuelve el paquete (payload) a guardar."""
    s_indexed = s_df.set_index("condition")
    int_i_val = float(s_indexed.loc["sinusoidal", "int_i"])
    int_f_val = float(s_indexed.loc["sinusoidal", "int_f"])

    # 1. Basal
    t_basal, c_basal, cell_names = read_calcium(Path(s_indexed.loc["basal", "path_traces"]))
    df_track_b, f_idx_b = read_dlc(Path(s_indexed.loc["basal", "path_tracking"]))
    res_basal = process_single_condition(c_basal, t_basal, df_track_b, f_idx_b)

    if plot_traces:
        print("  Plot de trazas basal...")
        n_sync_b = len(res_basal["all_traces_conv"])
        plot_raw_vs_processed_waterfall(
            time_s=res_basal["sync_time"][:n_sync_b],
            raw_traces=c_basal[:n_sync_b],
            conv_traces=res_basal["all_traces_conv"],
            cell_names=cell_names,
            mouse_id=mouse_id, session_idx=session_idx, date_str=date_str,
            condition="basal",
        )

    # 2. Sinusoidal
    t_sinus, c_sinus, _ = read_calcium(Path(s_indexed.loc["sinusoidal", "path_traces"]))
    df_track_s, f_idx_s = read_dlc(Path(s_indexed.loc["sinusoidal", "path_tracking"]))
    res_sinus = process_single_condition(c_sinus, t_sinus, df_track_s, f_idx_s)

    if res_basal["retained_traces"].shape[1] != res_sinus["retained_traces"].shape[1]:
        raise ValueError(
            "Basal y sinusoidal tienen distinto número de neuronas tras el filtrado "
            f"({res_basal['retained_traces'].shape[1]} vs {res_sinus['retained_traces'].shape[1]}), "
            "así que no se pueden pegar en 'combined'."
        )

    if plot_traces:
        print("  Plot de trazas sinusoidal...")
        n_sync_s = len(res_sinus["all_traces_conv"])
        plot_raw_vs_processed_waterfall(
            time_s=res_sinus["sync_time"][:n_sync_s],
            raw_traces=c_sinus[:n_sync_s],
            conv_traces=res_sinus["all_traces_conv"],
            cell_names=cell_names,
            mouse_id=mouse_id, session_idx=session_idx, date_str=date_str,
            condition="sinusoidal",
        )

    # 3. Combined (basal + sinusoidal pegadas)
    comb_traces = np.vstack([res_basal["retained_traces"], res_sinus["retained_traces"]])
    comb_pos = np.concatenate([res_basal["position_1d"], res_sinus["position_1d"]])
    comb_speed = np.concatenate([res_basal["speed"], res_sinus["speed"]])
    comb_dir = np.concatenate([res_basal["direction"], res_sinus["direction"]])

    dt_b = float(np.median(np.diff(res_basal["sync_time"]))) if len(res_basal["sync_time"]) > 1 else 0.05
    t_b_zero = res_basal["sync_time"][res_basal["sync_indices"]] - res_basal["sync_time"][0]
    offset = t_b_zero[-1] + dt_b if len(t_b_zero) > 0 else 0.0
    t_s_zero = res_sinus["sync_time"][res_sinus["sync_indices"]] - res_sinus["sync_time"][0] + offset
    comb_time = np.concatenate([t_b_zero, t_s_zero])

    res_combined = {
        "retained_traces": comb_traces,
        "position_1d": comb_pos,
        "speed": comb_speed,
        "direction": comb_dir,
        "time_seconds": comb_time,
        "condition_labels": np.array(
            ["basal"] * len(res_basal["position_1d"]) + ["sinusoidal"] * len(res_sinus["position_1d"]),
            dtype=object),
    }

    return {
        "mouse_id": mouse_id,
        "session": session_idx,
        "date": date_str,
        "int_i": int_i_val,
        "int_f": int_f_val,
        "cell_names": cell_names,
        "parameters": PARAMS,
        "basal": res_basal,
        "sinusoidal": res_sinus,
        "combined": res_combined,
    }


def preprocess_sessions(
    manifest_df: pd.DataFrame,
    filter_mice: Optional[List[str]] = None,
    filter_sessions: Optional[List[int]] = None,
    filter_dates: Optional[List[str]] = None,
    force_recompute: bool = False,
    plot_traces: bool = True,
) -> Tuple[List[Path], List[Tuple[str, str]]]:
    """Preprocesa las sesiones seleccionadas. Devuelve (lista de pickles, lista de fallos)."""
    record_map = {}
    if PROCESSED_CSV.exists():
        proc_df = pd.read_csv(PROCESSED_CSV, dtype={"mouse_id": str, "date": str})
        if {"mouse_id", "session", "date"}.issubset(set(proc_df.columns)):
            for r in proc_df.to_dict(orient="records"):
                record_map[(str(r["mouse_id"]), int(r["session"]), str(r["date"]))] = r

    selected_pickles: List[Path] = []
    failures: List[Tuple[str, str]] = []

    for (mouse_id, session_idx, date_str), s_df in manifest_df.groupby(["mouse_id", "session", "date"]):
        mouse_id = str(mouse_id)
        session_idx = int(session_idx)
        date_str = str(date_str)
        session_key = (mouse_id, session_idx, date_str)
        label = f"ratón {mouse_id} | sesión {session_idx} ({date_str})"

        if filter_mice and mouse_id not in filter_mice:
            continue
        if filter_sessions and session_idx not in filter_sessions:
            continue
        if filter_dates and date_str not in filter_dates:
            continue

        target_pkl = PROCESSED_PICKLE_DIR / f"mouse_{mouse_id}_session_{session_idx}_{date_str}_preprocessed.pkl"

        basal_plot = PLOTTING_TRACES_DIR / f"traces_raw_vs_processed_M{mouse_id}_S{session_idx}_{date_str}_basal.png"
        sinus_plot = PLOTTING_TRACES_DIR / f"traces_raw_vs_processed_M{mouse_id}_S{session_idx}_{date_str}_sinusoidal.png"
        plots_already_exist = basal_plot.exists() and sinus_plot.exists()

        if (not force_recompute and session_key in record_map and target_pkl.exists()
                and (plots_already_exist or not plot_traces)):
            print(f"[YA HECHO] {label}: preprocesado y plots ya existen")
            selected_pickles.append(target_pkl)
            continue

        print(f"\n--- Preprocesando {label} ---")
        try:
            payload = preprocess_one_session(s_df, mouse_id, session_idx, date_str, plot_traces)
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            traceback.print_exc()
            failures.append((label, str(e)))
            continue

        save_pickle(target_pkl, payload)
        print(f"  -> pickle guardado: {target_pkl.name}")
        selected_pickles.append(target_pkl)

        record_map[session_key] = {
            "mouse_id": mouse_id,
            "session": session_idx,
            "date": date_str,
            "int_i": payload["int_i"],
            "int_f": payload["int_f"],
            "pickle_path": str(target_pkl),
            "retained_basal_frames": len(payload["basal"]["position_1d"]),
            "retained_sinus_frames": len(payload["sinusoidal"]["position_1d"]),
            "speed_threshold_m_s": PARAMS["speed_threshold_m_s"],
            "edge_exclusion_fraction": PARAMS["track_edge_exclusion_fraction"],
            "sig_filt": PARAMS["sig_filt"],
            "peak_th": PARAMS["peak_th"],
            "post_gaussian_sigma": PARAMS["post_selection_gaussian_sigma_frames"],
        }

    if record_map:
        updated_df = pd.DataFrame(list(record_map.values()))
        updated_df = updated_df.sort_values(by=["mouse_id", "session"]).reset_index(drop=True)
        updated_df.to_csv(PROCESSED_CSV, index=False)
        banner(f"Processed database guardada en:\n  {PROCESSED_CSV}\nPlots de trazas en:\n  {PLOTTING_TRACES_DIR}")
        print(updated_df[["mouse_id", "session", "date", "int_i", "int_f", "retained_basal_frames",
                          "retained_sinus_frames"]].to_string())

    return selected_pickles, failures


# =============================================================================
# PASO 3: INGEST AL FORMATO ESTANDARIZADO
# =============================================================================

def intervention_params(payload: Dict, condition: str) -> Tuple[float, float]:
    """basal = sin intervención (0, 0); sinusoidal y combined llevan los valores de la sesión."""
    if condition == "basal":
        return 0.0, 0.0
    return float(payload["int_i"]), float(payload["int_f"])


def build_standardized_dict(payload: Dict, condition: str, mouse_name: str, session_name: str) -> Dict:
    res = payload[condition]
    int_i, int_f = intervention_params(payload, condition)

    signal = np.asarray(res["retained_traces"], dtype=float)  # tiempo x neuronas
    pos = np.asarray(res["position_1d"], dtype=float).reshape(-1, 1)
    mov_dir = np.asarray(res["direction"]).reshape(-1, 1)
    speed = np.asarray(res["speed"], dtype=float).reshape(-1, 1)

    # Vector de tiempo (segundos, empezando en 0)
    if condition == "combined":
        time = np.asarray(res["time_seconds"], dtype=float)
    else:
        sync_time = np.asarray(res["sync_time"], dtype=float)
        time = sync_time[res["sync_indices"]] - sync_time[0]
    time = time.reshape(-1, 1)

    n_timepoints = signal.shape[0]
    for name, arr in [("pos", pos), ("mov_dir", mov_dir), ("speed", speed), ("time", time)]:
        if arr.shape[0] != n_timepoints:
            raise ValueError(
                f"Longitudes distintas en {mouse_name}/{session_name}: "
                f"signal={n_timepoints}, {name}={arr.shape[0]}"
            )

    return {
        "metadata": {
            "dataset_name": DATASET_NAME,
            "modality": MODALITY,
            "depth_type": DEPTH_TYPE,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "case": CASE,
            "cells": CELLS,
            "signal_name": SIGNAL_NAME,
            "paradigm": condition,
            "model": "",
            "common_crop_length": "",
            "int_i": int_i,
            "int_f": int_f,
        },
        "signal": signal,
        "pos": pos,
        "mov_dir": mov_dir,
        "speed": speed,
        "trial_id": None,  # intervention no tiene trials definidos
        "time": time,
        "inner_time": None,  # tampoco tiene inner time
    }


def ingest_session_pickle(source_file_path: Path, overwrite: bool) -> List[Dict]:
    payload = load_pickle(source_file_path)
    mouse_name = f"M{payload['mouse_id']}"
    session_idx = payload["session"]
    date_str = payload["date"]

    mouse_output_dir = STANDARDIZED_DIR / mouse_name
    ensure_dir(mouse_output_dir)

    rows = []
    for condition in CONDITIONS:
        session_name = f"S{session_idx}_{date_str}_{condition}"
        session_name_safe = sanitize_for_filename(session_name)
        print(f"  {mouse_name} | {session_name}")

        standardized_file_name = (
            f"{DATASET_NAME}_{mouse_name}_{session_name_safe}_"
            f"{MODALITY}_{DEPTH_TYPE}_{CASE}_{CELLS}_{SIGNAL_NAME}.pkl"
        )
        standardized_file_path = mouse_output_dir / standardized_file_name

        if standardized_file_path.exists() and not overwrite:
            standardized = load_pickle(standardized_file_path)
            print("    Ya existe, no lo vuelvo a guardar")
        else:
            standardized = build_standardized_dict(payload, condition, mouse_name, session_name)
            save_pickle(standardized_file_path, standardized)
            print(f"    Guardado: {standardized_file_name}")

        signal = standardized["signal"]
        pos = standardized["pos"]
        int_i, int_f = intervention_params(payload, condition)

        rows.append({
            "dataset_name": DATASET_NAME,
            "modality": MODALITY,
            "depth_type": DEPTH_TYPE,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "session_name_safe": session_name_safe,
            "source_base_dir": str(BASE_DIR),
            "source_data_dir": str(PROCESSED_PICKLE_DIR),
            "source_mouse_dir": str(PROCESSED_PICKLE_DIR),
            "source_file_name": source_file_path.name,
            "source_file_path": str(source_file_path),
            "case": CASE,
            "cells": CELLS,
            "signal_name": SIGNAL_NAME,
            "paradigm": condition,
            "model": "",
            "int_i": int_i,
            "int_f": int_f,
            "standardized_file_name": standardized_file_name,
            "standardized_file_path": str(standardized_file_path),
            "n_timepoints": int(signal.shape[0]),
            "n_neurons": int(signal.shape[1]),
            "pos_dim": int(pos.shape[1]) if pos.ndim > 1 else 1,
            "contains_signal": yes_no(standardized.get("signal") is not None),
            "contains_pos": yes_no(standardized.get("pos") is not None),
            "contains_mov_dir": yes_no(standardized.get("mov_dir") is not None),
            "contains_speed": yes_no(standardized.get("speed") is not None),
            "contains_trial_id": yes_no(standardized.get("trial_id") is not None),
            "contains_time": yes_no(standardized.get("time") is not None),
            "contains_inner_time": yes_no(standardized.get("inner_time") is not None),
            "source_container_type": "session_pickle_with_multiple_conditions",
            "common_crop_length": "",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "",
        })
    return rows


def update_master_csv(new_rows: List[Dict], replace_whole_dataset: bool) -> None:
    """
    Añade las filas nuevas al CSV maestro SIN borrar las de otros datasets (p. ej. learning).
    - replace_whole_dataset=True: borra todas las filas antiguas de 'intervention' y escribe las nuevas.
    - replace_whole_dataset=False (cuando usas --mouse/--session/--date): solo reemplaza
      las sesiones que acabas de procesar y deja el resto de intervention como estaba.
    """
    ensure_dir(METADATA_DIR)
    new_df = pd.DataFrame(new_rows, columns=MASTER_COLUMNS)

    if MASTER_CSV_PATH.is_file():
        old_df = pd.read_csv(MASTER_CSV_PATH, dtype=str, keep_default_na=False)
        old_df = old_df.reindex(columns=MASTER_COLUMNS, fill_value="")
        is_this_dataset = old_df["dataset_name"] == DATASET_NAME
        if replace_whole_dataset:
            old_df = old_df[~is_this_dataset]
        else:
            new_keys = set(zip(new_df["mouse_name"], new_df["session_name"]))
            same_key = pd.Series(
                [(m, s) in new_keys for m, s in zip(old_df["mouse_name"], old_df["session_name"])],
                index=old_df.index,
            )
            old_df = old_df[~(is_this_dataset & same_key)]
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    out_df.to_csv(MASTER_CSV_PATH, index=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Intervention: manifest + preprocesado + ingest estandarizado")
    parser.add_argument("--mouse", nargs="+", help="Solo estos ratones (p. ej. --mouse 032 521)")
    parser.add_argument("--session", nargs="+", type=int, help="Solo estas sesiones (p. ej. --session 1 2)")
    parser.add_argument("--date", nargs="+", help="Solo estas fechas (p. ej. --date 1707 2309)")
    parser.add_argument("--force", action="store_true", help="Rehace todo (divisiones, preprocesado, plots, estandarizado)")
    parser.add_argument("--no-plots", action="store_true", help="No genera los plots de trazas")
    args, _ = parser.parse_known_args()

    for d in [DATABASE_DIR, SPLIT_DATA_DIR, PROCESSED_PICKLE_DIR, PLOTTING_TRACES_DIR,
              METADATA_DIR, STANDARDIZED_DIR]:
        ensure_dir(d)

    # PASO 1
    banner("PASO 1/3: manifest (dividir trazas, juntar tracking)")
    manifest_df = build_manifest_database(force=args.force)

    # PASO 2
    banner("PASO 2/3: preprocesado de trazas y tracking")
    pickles, failures = preprocess_sessions(
        manifest_df,
        filter_mice=args.mouse,
        filter_sessions=args.session,
        filter_dates=args.date,
        force_recompute=args.force,
        plot_traces=not args.no_plots,
    )

    # PASO 3
    banner("PASO 3/3: conversión al formato estandarizado")
    overwrite = args.force or OVERWRITE_STANDARDIZED_FILES
    all_rows: List[Dict] = []
    for pkl in pickles:
        print(f"\nProcesando: {pkl.name}")
        try:
            all_rows.extend(ingest_session_pickle(pkl, overwrite))
        except Exception as e:
            print(f"[ERROR] {pkl.name}: {e}")
            traceback.print_exc()
            failures.append((pkl.name, str(e)))

    if all_rows:
        has_filters = bool(args.mouse or args.session or args.date)
        update_master_csv(all_rows, replace_whole_dataset=not has_filters)

    banner("LISTO")
    print(f"Filas escritas para '{DATASET_NAME}': {len(all_rows)}")
    print(f"CSV maestro: {MASTER_CSV_PATH}")
    if failures:
        print(f"\nOJO: {len(failures)} sesión(es) fallaron:")
        for lab, msg in failures:
            print(f"  - {lab}: {msg}")


if __name__ == "__main__":
    main()