#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event rate (Hz) por neurona, con la misma estructura que mutual_information.py.

Qué hace
--------
1) Lee el índice maestro (master_dataset_index.csv).
2) Resuelve la ruta real de cada pickle estandarizado (mismo resolver que
   mutual_information.py: cambio de raíz, rutas desplazadas, '_pkl' vs '.pkl').
3) Calcula, para cada neurona, nº de eventos / duración de la sesión (Hz).
4) Guarda:
       - un CSV con una fila por neurona (todas las sesiones/datasets)
       - un CSV con una fila por sesión (media/std/mediana del rate)
       - un pickle por sesión con los arrays completos
       - un log txt

Cómo se cuentan los eventos (EVENT_MODE)
----------------------------------------
Se usa data[clave] con la primera clave encontrada en EVENT_KEYS
(si no hay ninguna, se usa data["signal"]).

  "sum"       : eventos = suma de la señal (señal = conteos de spikes/eventos)
  "nonzero"   : eventos = nº de timepoints con valor > 0
  "onsets"    : eventos = nº de transiciones 0 -> >0 (eventos deconvolucionados
                continuos: un evento = un episodio de actividad)
  "threshold" : eventos = nº de cruces ascendentes de mean + K*std
                (para trazas dF/F sin deconvolucionar)
  "auto"      : elige según la señal:
                  enteros >= 0                  -> "sum"
                  >= 0 y mayoría de ceros       -> "onsets"
                  resto                         -> "threshold"

IMPORTANTE: revisa en el log qué modo eligió "auto" y el dt estimado,
y fija EVENT_MODE a mano si no es lo que esperas.

Duración
--------
duration_s = n_timepoints * dt, con dt = mediana(diff(time)) * TIME_TO_SECONDS.
Si el pickle trae "fs" (Hz) se usa en su lugar.
"""

import os
import csv
import glob
import pickle
import timeit
from datetime import datetime

import numpy as np
import pandas as pd

# =============================================================================
# Paths & Settings
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
OLD_PROJECT_ROOT = "/home/melisamc/Documents/time_project_standarization"

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "master_dataset_index.csv")
STANDARDIZED_DIR = os.path.join(PROJECT_ROOT, "data", "standardized_sessions")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "event_rate")
SESSION_PKL_DIR = os.path.join(OUTPUT_DIR, "session_pickles")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

NEURON_RATE_CSV = os.path.join(OUTPUT_DIR, "event_rate_per_neuron.csv")
SESSION_RATE_CSV = os.path.join(OUTPUT_DIR, "event_rate_per_session_summary.csv")

MAX_ROWS = None            # entero para pruebas
VERBOSE = True

EVENT_KEYS = ["events", "spikes", "spike_train", "deconvolved", "S"]
EVENT_MODE = "auto"        # "auto", "sum", "nonzero", "onsets", "threshold"
THRESHOLD_K = 2.0          # solo para "threshold"
THRESHOLD_K = 2.0          # solo para "threshold"
TIME_TO_SECONDS = 1.0      # pon 1e-3 si "time" está en ms


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


def sanitize_for_filename(text):
    text = str(text)
    return "".join([ch if (ch.isalnum() or ch in ["-", "_"]) else "_" for ch in text])


def base_name_from_row(row_idx, row):
    return (
        f"row_{row_idx:04d}_"
        f"{sanitize_for_filename(row.get('dataset_name', 'unknown_dataset'))}_"
        f"{sanitize_for_filename(row.get('mouse_name', 'unknown_mouse'))}_"
        f"{sanitize_for_filename(row.get('session_name', 'unknown_session'))}_"
        f"{sanitize_for_filename(row.get('depth_type', 'unknown_depth'))}"
    )


# =============================================================================
# Dynamic Path Resolver (idéntico al de mutual_information.py)
# =============================================================================

def resolve_actual_file_path(row):
    raw_path = str(row.get("standardized_file_path", "")).strip()
    dataset_name = str(row.get("dataset_name", "")).strip()
    mouse_name = str(row.get("mouse_name", "")).strip()
    depth_type = str(row.get("depth_type", "")).strip()

    candidate_paths = []

    if raw_path and not raw_path.replace(".", "", 1).isdigit():
        if raw_path.startswith(OLD_PROJECT_ROOT):
            migrated = raw_path.replace(OLD_PROJECT_ROOT, PROJECT_ROOT)
            candidate_paths.append(migrated)
            if migrated.endswith("_pkl"):
                candidate_paths.append(migrated[:-4] + ".pkl")
        candidate_paths.append(raw_path)
        if raw_path.endswith("_pkl"):
            candidate_paths.append(raw_path[:-4] + ".pkl")

    for p in candidate_paths:
        if os.path.isfile(p):
            return p

    if dataset_name and mouse_name:
        dataset_dir = os.path.join(STANDARDIZED_DIR, dataset_name)
        if os.path.isdir(dataset_dir):
            patterns = [
                os.path.join(dataset_dir, "**", f"*{mouse_name}*.pkl"),
                os.path.join(dataset_dir, "**", f"*{mouse_name}*_pkl"),
            ]
            for pat in patterns:
                matches = glob.glob(pat, recursive=True)
                if matches:
                    if depth_type:
                        for m in matches:
                            if depth_type in m:
                                return m
                    return matches[0]

    if mouse_name:
        matches = glob.glob(os.path.join(STANDARDIZED_DIR, "**", f"*{mouse_name}*.pkl"), recursive=True)
        if matches:
            if depth_type:
                for m in matches:
                    if depth_type in m:
                        return m
            return matches[0]

    return None


# =============================================================================
# Event rate computation
# =============================================================================

def pick_event_matrix(data):
    """Devuelve (matriz (n_time, n_neurons), nombre de la clave usada)."""
    for key in EVENT_KEYS:
        if key in data and data[key] is not None:
            return ensure_2d(data[key]).astype(float), key
    return ensure_2d(data["signal"]).astype(float), "signal"


def resolve_mode(mat, mode):
    if mode != "auto":
        return mode
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return "threshold"
    if np.all(finite >= 0):
        if np.allclose(finite, np.round(finite)):
            return "sum"
        if np.mean(finite == 0) > 0.5:
            return "onsets"
    return "threshold"


def count_events(mat, mode, k=2.0):
    """Nº de eventos por neurona. mat: (n_time, n_neurons)."""
    mat = np.where(np.isfinite(mat), mat, 0.0)
    if mode == "sum":
        return mat.sum(axis=0)
    if mode == "nonzero":
        return (mat > 0).sum(axis=0).astype(float)
    if mode == "onsets":
        active = mat > 0
        onsets = active[1:] & ~active[:-1]
        return (onsets.sum(axis=0) + active[0].astype(int)).astype(float)
    if mode == "threshold":
        thr = mat.mean(axis=0) + k * mat.std(axis=0)
        above = mat > thr
        onsets = above[1:] & ~above[:-1]
        return (onsets.sum(axis=0) + above[0].astype(int)).astype(float)
    raise ValueError(f"EVENT_MODE desconocido: {mode}")


def get_dt_seconds(data, n_timepoints):
    if data.get("fs") is not None:
        fs = float(np.ravel(data["fs"])[0])
        if fs > 0:
            return 1.0 / fs
    t = ensure_1d(data["time"]).astype(float)
    if len(t) < 2:
        raise ValueError("No se puede estimar dt: 'time' tiene <2 puntos")
    dt = np.nanmedian(np.diff(t)) * TIME_TO_SECONDS
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt inválido estimado de 'time': {dt}")
    return dt


# =============================================================================
# CSV Schemas
# =============================================================================

NEURON_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "recording_type",
    "standardized_file_path",
    "row_index",
    "neuron_index",
    "n_timepoints",
    "n_neurons_in_session",
    "duration_s",
    "dt_s",
    "event_source",
    "event_mode",
    "n_events",
    "event_rate_hz",
]

SESSION_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "recording_type",
    "standardized_file_path",
    "row_index",
    "n_timepoints",
    "n_neurons",
    "duration_s",
    "dt_s",
    "event_source",
    "event_mode",
    "event_rate_hz_mean",
    "event_rate_hz_std",
    "event_rate_hz_median",
    "session_pickle_name",
    "session_pickle_path",
]


def init_csv(csv_path, columns):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_rows(csv_path, columns, rows):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerows(rows)


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
# Session Processing Routine
# =============================================================================

def process_row(row_idx, row):
    actual_file_path = resolve_actual_file_path(row)

    if actual_file_path is None or not os.path.isfile(actual_file_path):
        raise FileNotFoundError(
            f"Could not resolve file for row {row_idx}: "
            f"dataset={row.get('dataset_name')}, mouse={row.get('mouse_name')}, "
            f"raw_path={row.get('standardized_file_path')}"
        )

    data = load_pickle_standard(actual_file_path)

    for key in ["signal", "time"]:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in: {actual_file_path}")

    mat, event_source = pick_event_matrix(data)
    n_timepoints, n_neurons = mat.shape
    if n_neurons < 1:
        raise ValueError(f"Row {row_idx}: Session has no neurons.")

    time_1d = ensure_1d(data["time"])
    if len(time_1d) != n_timepoints:
        raise ValueError(
            f"Row {row_idx}: Length mismatch for 'time' ({len(time_1d)} vs {n_timepoints})."
        )

    dt_s = get_dt_seconds(data, n_timepoints)
    duration_s = n_timepoints * dt_s

    mode = resolve_mode(mat, EVENT_MODE)
    n_events = count_events(mat, mode, k=THRESHOLD_K)
    rate_hz = n_events / duration_s

    recording_type = "calcium" if "calcium" in str(row.get("modality", "")).lower() else "ephys"

    if VERBOSE:
        print(
            f"Row {row_idx:03d} | dataset={row.get('dataset_name', '')} | mouse={row.get('mouse_name', '')} | "
            f"session={row.get('session_name', '')} | depth={row.get('depth_type', '')} | "
            f"timepoints={n_timepoints} | neurons={n_neurons}"
        )
        print(
            f"   source={event_source} | mode={mode} | dt={dt_s:.4f} s | duration={duration_s:.1f} s | "
            f"rate mean={np.nanmean(rate_hz):.4f} Hz | median={np.nanmedian(rate_hz):.4f} Hz"
        )

    # Pickle de sesión
    session_pickle_name = base_name_from_row(row_idx, row) + "_event_rate.pkl"
    session_pickle_path = os.path.join(SESSION_PKL_DIR, session_pickle_name)
    save_pickle_standard(session_pickle_path, {
        "metadata": {
            "dataset_name": row.get("dataset_name", ""),
            "mouse_name": row.get("mouse_name", ""),
            "session_name": row.get("session_name", ""),
            "depth_type": row.get("depth_type", ""),
            "recording_type": recording_type,
            "row_index": row_idx,
            "standardized_file_path": actual_file_path,
            "n_timepoints": n_timepoints,
            "n_neurons": n_neurons,
            "duration_s": duration_s,
            "dt_s": dt_s,
            "event_source": event_source,
            "event_mode": mode,
            "threshold_k": THRESHOLD_K,
        },
        "n_events": n_events,
        "event_rate_hz": rate_hz,
    })

    base = {
        "dataset_name": row.get("dataset_name", ""),
        "mouse_name": row.get("mouse_name", ""),
        "session_name": row.get("session_name", ""),
        "depth_type": row.get("depth_type", ""),
        "recording_type": recording_type,
        "standardized_file_path": actual_file_path,
        "row_index": row_idx,
        "n_timepoints": n_timepoints,
        "duration_s": duration_s,
        "dt_s": dt_s,
        "event_source": event_source,
        "event_mode": mode,
    }

    neuron_rows = []
    for i in range(n_neurons):
        r = dict(base)
        r.update({
            "neuron_index": i,
            "n_neurons_in_session": n_neurons,
            "n_events": n_events[i],
            "event_rate_hz": rate_hz[i],
        })
        neuron_rows.append(r)

    session_row = dict(base)
    session_row.update({
        "n_neurons": n_neurons,
        "event_rate_hz_mean": np.nanmean(rate_hz),
        "event_rate_hz_std": np.nanstd(rate_hz),
        "event_rate_hz_median": np.nanmedian(rate_hz),
        "session_pickle_name": session_pickle_name,
        "session_pickle_path": session_pickle_path,
    })

    return neuron_rows, session_row


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SESSION_PKL_DIR)
    ensure_dir(LOG_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"Master CSV index not found: {CSV_PATH}")

    try:
        df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(CSV_PATH)

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    init_csv(NEURON_RATE_CSV, NEURON_COLUMNS)
    init_csv(SESSION_RATE_CSV, SESSION_COLUMNS)

    log_path = os.path.join(LOG_DIR, "event_rate_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = os.sys.stdout
    os.sys.stdout = Tee(original_stdout, log_file)

    print("==============================================================")
    print("Event rate pipeline")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M}")
    print("==============================================================")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Sessions in index: {len(df)}")
    print(f"EVENT_MODE: {EVENT_MODE} | EVENT_KEYS: {EVENT_KEYS} | TIME_TO_SECONDS: {TIME_TO_SECONDS}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    ok_count = 0
    fail_count = 0
    global_time = timeit.default_timer()

    for row_idx, row in df.iterrows():
        local_time = timeit.default_timer()
        try:
            neuron_rows, session_row = process_row(row_idx, row)
            append_csv_rows(NEURON_RATE_CSV, NEURON_COLUMNS, neuron_rows)
            append_csv_rows(SESSION_RATE_CSV, SESSION_COLUMNS, [session_row])
            ok_count += 1
            print(f"[OK] Row {row_idx} completed in {timeit.default_timer() - local_time:.2f} s\n")
        except Exception as e:
            fail_count += 1
            print(f"[ERROR] Row {row_idx}: {e}\n")

    total_elapsed = timeit.default_timer() - global_time
    print("==============================================================")
    print("Event rate analysis complete")
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Successful sessions: {ok_count} / {len(df)}")
    print(f"Failed sessions    : {fail_count}")
    print(f"Total elapsed time : {total_elapsed:.2f} s")
    print("==============================================================")

    os.sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()