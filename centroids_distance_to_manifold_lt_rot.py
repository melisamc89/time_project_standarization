#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LT-only centroid-to-nearest-observed-state distances in neural and UMAP spaces.

Execute the WHOLE file in PyCharm's IPython console or use `%run /path/to/file.py`.
The final run call is deliberately unguarded.

For each sigma, compute independent trial centroids and nearest sampled points in:
  (a) filtered full neural space (T x neurons), and
  (b) a newly fitted 3D UMAP of that same filtered signal.

Uses scipy.spatial.cKDTree, never an NxN distance matrix. The reference cloud
contains all timepoints, including those in the trial itself. Distances are to
observed samples, NOT to interpolated paths or a mathematically continuous manifold.
"""
import pickle
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

ROOTS = [Path('/home/hippolab/Documents/melisa/time_project_standarization'),
         Path('/home/melisamc/Documents/time_project_standarization')]
ROOT = next((p for p in ROOTS if p.exists()), ROOTS[0])
MASTER_INDEX_CSV = ROOT / 'data/metadata/master_dataset_index.csv'
OUTPUT_DIR = ROOT / 'analysis/centroid_manifold_lt_both_spaces'
FILTER_SIGMAS = [0, 2, 4, 8, 16, 32]
SAMPLING_RATE_HZ = 20.0
TRIAL_ID_MODE = 'recomputed'      # or 'stored'
TRIAL_HIGH_PERCENTILE = 90.0
TRIAL_LOW_PERCENTILE = 15.0
USE_ONLY_MOVING = False
MOVING_DIR_EXCLUDE_ZERO = False
MOVING_SPEED_THRESHOLD = None
N_NEURONS_SUBSAMPLE = None       # None = full neural space, all neurons
MIN_TRIAL_POINTS = 1
UMAP_N_NEIGHBORS = 120
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
UMAP_N_COMPONENTS = 3
# KDTree can be costly for very high dimensional neural data; the option below
# limits reference points if needed. Default None uses EVERY observed timepoint.
MAX_NEURAL_REFERENCE_POINTS = None
DEPTH_COLOR_MAP = {
    'superficial': '#9900ff',
    'deep': '#cc9900',
    'whole population': 'gray',
}
SPACES = ('neural', 'umap')


def safe(s):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(s))


def one_d(x, n):
    a = np.asarray(x)
    if a.ndim > 1:
        a = a[:, 0]
    a = a.ravel()
    if len(a) < n:
        if not len(a):
            raise ValueError('Empty behavioral field')
        a = np.pad(a, (0, n - len(a)), mode='edge')
    return a[:n]


def trials_from_pos(pos):
    pos = np.asarray(pos, dtype=float)
    finite = pos[np.isfinite(pos)]
    if len(finite) == 0:
        return np.zeros(len(pos), dtype=int)
    high, low = np.percentile(finite, [TRIAL_HIGH_PERCENTILE, TRIAL_LOW_PERCENTILE])
    tids = np.zeros(len(pos), dtype=int)
    in_high, tid = False, 0
    for i in range(1, len(pos)):
        if not in_high and pos[i] >= high:
            in_high = True
        elif in_high and pos[i] <= low:
            tid += 1
            in_high = False
        tids[i] = tid
    return tids


def load_session(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    signal = np.asarray(data['signal'], dtype=np.float32)
    if signal.ndim != 2 or not np.isfinite(signal).all():
        raise ValueError('Expected finite signal array [time, neuron]')
    if N_NEURONS_SUBSAMPLE is not None and signal.shape[1] > N_NEURONS_SUBSAMPLE:
        ix = np.sort(np.random.default_rng(42).choice(signal.shape[1], N_NEURONS_SUBSAMPLE, replace=False))
        signal = signal[:, ix]
    n = len(signal)
    if TRIAL_ID_MODE == 'stored':
        tids = one_d(data['trial_id'], n).astype(int)
    else:
        pos_key = 'pos' if 'pos' in data else 'position'
        tids = trials_from_pos(one_d(data[pos_key], n))
    mask = np.ones(n, dtype=bool)
    if USE_ONLY_MOVING:
        if MOVING_DIR_EXCLUDE_ZERO and 'mov_dir' in data:
            mask &= one_d(data['mov_dir'], n) != 0
        if MOVING_SPEED_THRESHOLD is not None and 'speed' in data:
            mask &= np.abs(one_d(data['speed'], n)) >= MOVING_SPEED_THRESHOLD
    return signal[mask], tids[mask]


def is_lt_session(name):
    """Ignore the dataset prefix 'lt_rot'; require final explicit marker LT."""
    markers = [token for token in re.split(r'[_\W]+', str(name).lower()) if token in ('lt', 'rot')]
    return bool(markers) and markers[-1] == 'lt'


def resolve_file(path):
    p = Path(str(path))
    if p.exists():
        return p
    for old in ROOTS:
        try:
            rel = p.relative_to(old)
        except ValueError:
            continue
        for new in ROOTS:
            candidate = new / rel
            if candidate.exists():
                return candidate
    return p


def trial_centroid_distances(cloud, trial_ids, labels=None, reference_limit=None):
    """Return centroids, exact point-cloud NN distances, and RMS-normalized values.

    Using a KD-tree avoids an NxN matrix. For high dimensional data, KDTree
    searches can still be costly; the reference_limit option is approximate.
    """
    cloud = np.asarray(cloud, dtype=np.float32)
    if labels is None:
        labels = [tid for tid in np.unique(trial_ids)
                  if np.count_nonzero(trial_ids == tid) >= MIN_TRIAL_POINTS]
    if not labels:
        raise ValueError('No valid trials')
    centroids = np.stack([cloud[trial_ids == tid].mean(axis=0) for tid in labels])
    refs = cloud
    if reference_limit is not None and len(cloud) > reference_limit:
        ix = np.random.default_rng(42).choice(len(cloud), reference_limit, replace=False)
        refs = cloud[np.sort(ix)]
    distances, _ = cKDTree(refs).query(centroids, k=1, workers=-1)
    center = cloud.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum((cloud - center) ** 2, axis=1, dtype=np.float64))))
    normalized = distances / radius if radius > 0 else np.full_like(distances, np.nan)
    return labels, distances, normalized, radius


def saveplot(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=190, bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def xlabel(ax):
    ax.set_xticks(FILTER_SIGMAS)
    ax.set_xlabel('Gaussian filter σ (samples)')
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)


def ylabel(ax, space, normalized):
    name = 'Neural space' if space == 'neural' else '3D UMAP'
    ax.set_ylabel(f'{name}: mean nearest-point distance' +
                  (' / RMS radius' if normalized else ''))


def curve(ax, x, y, sem=None, color=None, label=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ax.plot(x, y, '-o', lw=2.3, markersize=5.5, color=color, label=label)
    if sem is not None:
        sem = np.nan_to_num(np.asarray(sem, float), nan=0)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=.18, lw=0)


def plot_mouse(session_df, mouse_df, out):
    dest = out / 'per_mouse_plots'
    dest.mkdir(exist_ok=True)
    for (mouse, depth), frame in mouse_df.groupby(['mouse_name', 'depth_type']):
        color = DEPTH_COLOR_MAP[depth]
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=True)
        for row, space in enumerate(SPACES):
            selected = session_df[(session_df.mouse_name == mouse) &
                                  (session_df.depth_type == depth) &
                                  (session_df.space == space)]
            m = frame[frame.space == space].sort_values('sigma')
            for col, (ycol, semcol, norm) in enumerate([
                ('mean_distance', 'sem_raw', False),
                ('mean_normalized', 'sem_norm', True),
            ]):
                ax = axes[row, col]
                for _, s in selected.groupby('session_name'):
                    s = s.sort_values('sigma')
                    ax.plot(s.sigma, s[ycol], '-', lw=1, alpha=.23, color=color)
                curve(ax, m.sigma, m[ycol], m[semcol], color, 'Mouse mean')
                xlabel(ax)
                ylabel(ax, space, norm)
                ax.set_title(f'{space.upper()} — ' + ('normalized' if norm else 'raw'))
        fig.suptitle(f'{mouse} | {depth} | {selected.session_name.nunique()} LT session(s)',
                     color=color, fontweight='bold')
        fig.tight_layout()
        saveplot(fig, dest / f'{safe(mouse)}_{safe(depth)}_both_spaces')


def aggregate_and_plot(session_df, out):
    keys = ['mouse_name', 'depth_type', 'space', 'sigma']
    mice = session_df.groupby(keys, as_index=False).agg(
        mean_distance=('mean_distance', 'mean'),
        mean_normalized=('mean_normalized', 'mean'),
        n_sessions=('session_name', 'nunique'),
        sd_raw=('mean_distance', 'std'), sd_norm=('mean_normalized', 'std'))
    mice['sem_raw'] = mice.sd_raw / np.sqrt(mice.n_sessions)
    mice['sem_norm'] = mice.sd_norm / np.sqrt(mice.n_sessions)
    mice.to_csv(out / 'centroid_manifold_per_mouse.csv', index=False)
    plot_mouse(session_df, mice, out)

    grouped = mice.groupby(['depth_type', 'space', 'sigma'], as_index=False).agg(
        n_mice=('mouse_name', 'nunique'),
        mean_raw=('mean_distance', 'mean'), sd_raw=('mean_distance', 'std'),
        mean_norm=('mean_normalized', 'mean'), sd_norm=('mean_normalized', 'std'))
    grouped['sem_raw'] = grouped.sd_raw / np.sqrt(grouped.n_mice)
    grouped['sem_norm'] = grouped.sd_norm / np.sqrt(grouped.n_mice)
    grouped.to_csv(out / 'summary_deep_vs_superficial.csv', index=False)

    for space in SPACES:
        for norm, mouse_y, group_y, group_sem, suffix in [
            (False, 'mean_distance', 'mean_raw', 'sem_raw', 'raw'),
            (True, 'mean_normalized', 'mean_norm', 'sem_norm', 'normalized'),
        ]:
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            for depth in ('deep', 'superficial'):
                color = DEPTH_COLOR_MAP[depth]
                ms = mice[(mice.space == space) & (mice.depth_type == depth)]
                for _, mg in ms.groupby('mouse_name'):
                    mg = mg.sort_values('sigma')
                    ax.plot(mg.sigma, mg[mouse_y], '-', lw=1.1, alpha=.2, color=color)
                group = grouped[(grouped.space == space) &
                                (grouped.depth_type == depth)].sort_values('sigma')
                if len(group):
                    curve(ax, group.sigma, group[group_y], group[group_sem], color,
                          f'{depth.capitalize()} (n={group.n_mice.max()} mice)')
            xlabel(ax)
            ylabel(ax, space, norm)
            ax.legend(frameon=False)
            ax.set_title(f'LT sessions: {space.upper()} | deep vs superficial (mouse mean ± SEM)')
            fig.tight_layout()
            saveplot(fig, out / f'lt_deep_vs_superficial_{space}_{suffix}')

    # The main comparison: separate columns for neural and UMAP spaces,
    # separate rows for raw and RMS-normalized measurements.
    fig, axes = plt.subplots(2, 2, figsize=(12.7, 9.2))
    for row, (norm, group_y, group_sem) in enumerate([
        (False, 'mean_raw', 'sem_raw'), (True, 'mean_norm', 'sem_norm')
    ]):
        for col, space in enumerate(SPACES):
            ax = axes[row, col]
            for depth in ('deep', 'superficial'):
                group = grouped[(grouped.space == space) &
                                (grouped.depth_type == depth)].sort_values('sigma')
                if len(group):
                    curve(ax, group.sigma, group[group_y], group[group_sem],
                          DEPTH_COLOR_MAP[depth], depth.capitalize())
            xlabel(ax)
            ylabel(ax, space, norm)
            ax.set_title(f'{space.upper()} | ' + ('normalized' if norm else 'raw'))
            ax.legend(frameon=False)
    fig.suptitle('LT sessions: centroid-to-observed-manifold distances | deep vs superficial',
                 fontweight='bold')
    fig.tight_layout()
    saveplot(fig, out / 'lt_deep_vs_superficial_both_spaces_4panel')
    print('[PLOTS] Per-mouse and combined depth plots saved to', out, flush=True)


def run():
    print('[START] LT-only centroid distances in NEURAL and UMAP spaces', flush=True)
    print('[INDEX]', MASTER_INDEX_CSV, flush=True)
    print('[OUTPUT]', OUTPUT_DIR, flush=True)
    if not MASTER_INDEX_CSV.exists():
        raise FileNotFoundError(f'Master index missing: {MASTER_INDEX_CSV}')
    print('[INFO] Importing UMAP...', flush=True)
    import umap
    print('[INFO] UMAP imported.', flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(MASTER_INDEX_CSV)
    index['depth_type'] = index.depth_type.astype(str).str.strip().str.lower()
    index = index[(index.dataset_name.astype(str).str.strip().str.lower() == 'lt_rot') &
                  index.depth_type.isin(['deep', 'superficial']) &
                  index.session_name.apply(is_lt_session)].reset_index(drop=True)
    print('[INFO] Selected sessions:', len(index), flush=True)
    if len(index):
        print(index.groupby(['depth_type', 'mouse_name']).size().to_string(), flush=True)
    if index.empty:
        raise ValueError('No lt_rot LT sessions in master index.')

    trial_rows, session_rows, errors = [], [], []
    for k, row in index.iterrows():
        meta = dict(dataset_name=str(row.dataset_name), mouse_name=str(row.mouse_name),
                    session_name=str(row.session_name), depth_type=str(row.depth_type))
        print(f'[{k+1}/{len(index)}] {meta}', flush=True)
        try:
            signal, trial_ids = load_session(resolve_file(row.standardized_file_path))
            if len(signal) < 3:
                raise ValueError('Not enough timepoints')
        except Exception as exc:
            print('  [ERROR loading]', exc, flush=True)
            errors.append(dict(**meta, sigma=np.nan, space='', error=str(exc)))
            continue
        labels = [tid for tid in np.unique(trial_ids)
                  if np.count_nonzero(trial_ids == tid) >= MIN_TRIAL_POINTS]
        print(f'  [DATA] {signal.shape[0]} timepoints, {signal.shape[1]} neurons, '
              f'{len(labels)} trials', flush=True)
        for sigma in FILTER_SIGMAS:
            started = time.perf_counter()
            filtered = (signal if sigma == 0 else
                        gaussian_filter1d(signal, sigma=sigma, axis=0))
            # Fast neural-space distance, BEFORE any UMAP.
            results = {}
            try:
                results['neural'] = trial_centroid_distances(
                    filtered, trial_ids, labels, MAX_NEURAL_REFERENCE_POINTS)
                print(f'  [NEURAL] sigma={sigma} complete', flush=True)
            except Exception as exc:
                errors.append(dict(**meta, sigma=sigma, space='neural', error=str(exc)))
                print(f'  [NEURAL ERROR] sigma={sigma}: {exc}', flush=True)
            try:
                print(f'  [UMAP] sigma={sigma} fitting...', flush=True)
                embedding = umap.UMAP(
                    n_components=UMAP_N_COMPONENTS,
                    n_neighbors=min(UMAP_N_NEIGHBORS, len(filtered)-1),
                    min_dist=UMAP_MIN_DIST, random_state=UMAP_RANDOM_STATE,
                    low_memory=True).fit_transform(filtered)
                results['umap'] = trial_centroid_distances(embedding, trial_ids, labels)
            except Exception as exc:
                errors.append(dict(**meta, sigma=sigma, space='umap', error=str(exc)))
                print(f'  [UMAP ERROR] sigma={sigma}: {exc}', flush=True)
            for space, (tids, distances, normalized, radius) in results.items():
                for tid, distance, norm in zip(tids, distances, normalized):
                    trial_rows.append(dict(**meta, sigma=sigma,
                        sigma_seconds=sigma/SAMPLING_RATE_HZ, space=space,
                        trial_id=int(tid), n_trial_points=int(np.count_nonzero(trial_ids == tid)),
                        distance=float(distance), normalized_distance=float(norm)))
                session_rows.append(dict(**meta, sigma=sigma,
                    sigma_seconds=sigma/SAMPLING_RATE_HZ, space=space,
                    n_timepoints=len(signal), n_neurons=signal.shape[1], n_trials=len(tids),
                    rms_radius=radius, mean_distance=float(np.mean(distances)),
                    std_distance=float(np.std(distances, ddof=1)) if len(tids) > 1 else 0.,
                    mean_normalized=float(np.mean(normalized)),
                    std_normalized=float(np.std(normalized, ddof=1)) if len(tids) > 1 else 0.))
                print(f'  [{space.upper()}] sigma={sigma:2d} '
                      f'mean={np.mean(distances):.5f} normalized={np.mean(normalized):.5f}',
                      flush=True)
            print(f'  [DONE sigma={sigma}] {time.perf_counter()-started:.1f} s', flush=True)
        # Incremental checkpoint, so results survive interruptions.
        pd.DataFrame(trial_rows).to_csv(OUTPUT_DIR / 'centroid_manifold_per_trial.csv', index=False)
        pd.DataFrame(session_rows).to_csv(OUTPUT_DIR / 'centroid_manifold_per_session.csv', index=False)
        if errors:
            pd.DataFrame(errors).to_csv(OUTPUT_DIR / 'errors.csv', index=False)
        # Update plots after every mouse session for partial results.
        if session_rows:
            aggregate_and_plot(pd.DataFrame(session_rows), OUTPUT_DIR)
    print('[FINISHED] Results in', OUTPUT_DIR, flush=True)


# Deliberately NOT guarded by `if __name__ == '__main__'` for PyCharm IPython.
run()