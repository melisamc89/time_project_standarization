#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flow analysis from saved UMAP embeddings for all datasets.

This script uses the saved embedding payloads from:
    /home/melisamc/Documents/time_project_standarization/analysis/umap_pca_per_row/umap_pca_analysis_index.csv

For each session:
1) load saved UMAP embedding + behavior
2) clean movement direction from position if needed
3) align ring in 3D:
   - plane normal -> z axis
   - position gradient -> x axis
   - direction consistency -> flip y if needed
4) fit an approximate ellipse in XY
5) compute flow vectors
6) decompose flow into:
   - radial
   - tangential
   - residual
7) compute in-ring / off-ring metrics
8) compute flow-vs-angular-position deviation
9) save:
   - per-session pickle
   - summary CSV
   - in/off split CSV

Notes
-----
- This reproduces the original analysis conceptually, but not byte-for-byte,
  because the original code depended on custom lrgeo utilities.
- The ellipse here is an approximate covariance/PCA ellipse in XY.
- UMAP is NOT recomputed. We use the already-saved UMAP embedding.
"""

import os
import csv
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# Paths
# =============================================================================

ANALYSIS_INDEX_CSV = "/home/melisamc/Documents/time_project_standarization/analysis/umap_pca_per_row/umap_pca_analysis_index.csv"

OUTPUT_DIR = "/home/melisamc/Documents/time_project_standarization/analysis/umap_flow_dynamics"
PKL_DIR = os.path.join(OUTPUT_DIR, "data_pkl")
TXT_DIR = os.path.join(OUTPUT_DIR, "stats_txt")

SUMMARY_CSV = os.path.join(OUTPUT_DIR, "umap_flow_summary.csv")
RING_SPLIT_CSV = os.path.join(OUTPUT_DIR, "umap_flow_ring_split_summary.csv")

MAX_ROWS = None

# Optional filter
DATASET_FILTER = None
# DATASET_FILTER = "learning"


# =============================================================================
# Parameters
# =============================================================================

N_CENTROIDS = 80
FLOW_DIFF = 2
ELLIPSE_NUM_POINTS = 1000
ELLIPSE_NEIGHBORS = 300
OFF_RING_PERCENTILE = 90


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


def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


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
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def base_name_from_metadata(meta):
    return (
        f"{sanitize_for_filename(meta['dataset_name'])}_"
        f"{sanitize_for_filename(meta['mouse_name'])}_"
        f"{sanitize_for_filename(meta['session_name'])}_"
        f"{sanitize_for_filename(meta['depth_type'])}"
    )


def mode_binary_sign(x):
    """
    Robust mode for movement direction values in {-1, 0, +1}.
    """
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0

    vals, counts = np.unique(x.astype(int), return_counts=True)
    return float(vals[np.argmax(counts)])


# =============================================================================
# Geometry helpers
# =============================================================================

def normalize_vector(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def rotation_matrix_from_axis_angle(axis, angle):
    axis = normalize_vector(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ],
    ])
    return R


def rotation_matrix_align_vectors(v_from, v_to):
    """
    Returns rotation matrix that rotates v_from onto v_to.
    """
    a = normalize_vector(v_from)
    b = normalize_vector(v_to)

    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return np.eye(3)

    cross = np.cross(a, b)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    norm_cross = np.linalg.norm(cross)

    if norm_cross < 1e-12:
        if dot > 0:
            return np.eye(3)
        # 180-degree rotation around any orthogonal axis
        trial = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(a, trial)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = normalize_vector(np.cross(a, trial))
        return rotation_matrix_from_axis_angle(axis, np.pi)

    axis = cross / norm_cross
    angle = np.arccos(dot)
    return rotation_matrix_from_axis_angle(axis, angle)


def apply_rotation_to_cloud(points, R, center=None):
    points = np.asarray(points, dtype=float)
    if center is None:
        center = np.zeros(points.shape[1], dtype=float)
    center = np.asarray(center, dtype=float)
    return (points - center) @ R.T + center


def project_onto_vector(v, direction):
    direction = np.asarray(direction, dtype=float)
    v = np.asarray(v, dtype=float)

    dn = np.linalg.norm(direction)
    if dn < 1e-12:
        return np.zeros_like(v)

    u = direction / dn
    return np.dot(v, u) * u


def find_perpendicular_vector_on_xy_plane(radial_direction):
    """
    Given a radial direction, return an XY tangential direction.
    """
    radial_xy = np.array([radial_direction[0], radial_direction[1], 0.0], dtype=float)
    rnorm = np.linalg.norm(radial_xy)
    if rnorm < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    radial_xy /= rnorm
    tangential = np.array([-radial_xy[1], radial_xy[0], 0.0], dtype=float)
    return tangential


# =============================================================================
# Behavior helpers
# =============================================================================

def clean_mov_dir_from_position(pos_1d, mov_dir=None, fs=20.0, sigma=5):
    """
    Reproduce the idea from the original code:
    use smoothed position derivative to fill or replace gaps.
    """
    pos_1d = ensure_1d(pos_1d).astype(float)

    temp_mov = np.diff(pos_1d) * fs
    if len(temp_mov) == 0:
        temp_mov = np.zeros_like(pos_1d)
    else:
        temp_mov = np.concatenate([[temp_mov[0]], temp_mov])

    temp_mov = gaussian_filter1d(temp_mov, sigma=sigma, axis=0)
    temp_sign = np.sign(temp_mov)
    temp_sign[temp_sign == 0] = 1

    if mov_dir is None:
        return temp_sign.astype(float)

    mov_dir = ensure_1d(mov_dir).astype(float).copy()
    bad = ~np.isfinite(mov_dir) | (mov_dir == 0)
    mov_dir[bad] = temp_sign[bad]
    mov_dir = np.sign(mov_dir)
    mov_dir[mov_dir == 0] = 1
    return mov_dir.astype(float)


# =============================================================================
# Centroids / alignment
# =============================================================================

def get_centroids_by_pos_and_dir(embedding, pos_1d, mov_dir, n_centroids=80):
    """
    Compute centroids separately for each direction across position bins.
    Returns:
        centroids: (K, 3)
        centroids_pos_dir: (K, 2) columns = [position_bin_center, dir]
    """
    embedding = np.asarray(embedding, dtype=float)
    pos_1d = ensure_1d(pos_1d).astype(float)
    mov_dir = ensure_1d(mov_dir).astype(float)

    pos_min = np.nanmin(pos_1d)
    pos_max = np.nanmax(pos_1d)
    if not np.isfinite(pos_min) or not np.isfinite(pos_max) or pos_max <= pos_min:
        return np.empty((0, 3)), np.empty((0, 2))

    edges = np.linspace(pos_min, pos_max, n_centroids + 1)

    centroids = []
    centroids_pos_dir = []

    for d in [-1.0, 1.0]:
        mask_d = np.isfinite(mov_dir) & (mov_dir == d)
        if np.sum(mask_d) == 0:
            continue

        pos_d = pos_1d[mask_d]
        emb_d = embedding[mask_d]
        bin_ids = np.digitize(pos_d, edges[1:-1], right=False)

        for b in range(n_centroids):
            idx = np.where(bin_ids == b)[0]
            if len(idx) < 3:
                continue
            c = np.nanmean(emb_d[idx], axis=0)
            p_center = 0.5 * (edges[b] + edges[b + 1])
            centroids.append(c)
            centroids_pos_dir.append([p_center, d])

    if len(centroids) == 0:
        return np.empty((0, 3)), np.empty((0, 2))

    return np.vstack(centroids), np.vstack(centroids_pos_dir)


def parametrize_plane(points):
    """
    PCA plane parametrization.
    Returns:
        normal_vector, cloud_center
    """
    points = np.asarray(points, dtype=float)
    center = np.nanmean(points, axis=0)
    X = points - center
    pca = PCA(n_components=3)
    pca.fit(X)
    normal = pca.components_[2]
    return normalize_vector(normal), center


def align_ring_embedding(umap_emb, pos_1d, mov_dir):
    """
    Reproduce the original alignment idea:
    1) plane normal -> z axis
    2) position gradient -> x axis
    3) direction consistency -> y sign
    """
    umap_emb = np.asarray(umap_emb, dtype=float).copy()
    pos_1d = ensure_1d(pos_1d).astype(float)
    mov_dir = ensure_1d(mov_dir).astype(float)

    # mean-center
    umap_emb -= np.nanmean(umap_emb, axis=0)

    # centroids on original cloud
    centroids, centroids_pos_dir = get_centroids_by_pos_and_dir(
        umap_emb, pos_1d, mov_dir, n_centroids=N_CENTROIDS
    )

    # plane normal -> z axis
    normal, cloud_center = parametrize_plane(umap_emb)
    R_plane = rotation_matrix_align_vectors(normal, np.array([0.0, 0.0, 1.0]))
    horizontal_umap_emb = apply_rotation_to_cloud(umap_emb, R_plane, center=cloud_center)
    horizontal_umap_emb -= np.nanmean(horizontal_umap_emb, axis=0)

    horizontal_centroids, centroids_pos_dir_h = get_centroids_by_pos_and_dir(
        horizontal_umap_emb, pos_1d, mov_dir, n_centroids=N_CENTROIDS
    )

    # collapse directions to get mean centroid per position
    if len(horizontal_centroids) == 0:
        return {
            "aligned_umap_emb": horizontal_umap_emb,
            "aligned_centroids": horizontal_centroids,
            "centroids_pos_dir": centroids_pos_dir_h,
            "normal_vector": normal,
            "cloud_center": cloud_center,
            "R_plane": R_plane,
            "R_pos": np.eye(3),
            "pos_gradient_axis": np.array([1.0, 0.0, 0.0]),
        }

    unique_pos = np.unique(centroids_pos_dir_h[:, 0])
    centroids2D = np.array([
        np.nanmean(horizontal_centroids[centroids_pos_dir_h[:, 0] == val, :], axis=0)
        for val in unique_pos
    ])

    pos_gradient_pca = PCA(n_components=3)
    pos_gradient_pca.fit(centroids2D)
    pos_gradient_axis = pos_gradient_pca.components_[0].copy()

    aligned_centroids2D = pos_gradient_pca.transform(centroids2D)
    low_mask = unique_pos < np.percentile(pos_1d, 30)
    high_mask = unique_pos > np.percentile(pos_1d, 70)

    if np.any(low_mask) and np.any(high_mask):
        low = aligned_centroids2D[low_mask]
        high = aligned_centroids2D[high_mask]
        if np.nanmean(low[:, 0]) > np.nanmean(high[:, 0]):
            pos_gradient_axis *= -1.0

    # rotate pos gradient to x-axis
    R_pos = rotation_matrix_align_vectors(pos_gradient_axis, np.array([-1.0, 0.0, 0.0]))
    aligned_umap_emb = apply_rotation_to_cloud(
        horizontal_umap_emb, R_pos, center=np.nanmean(horizontal_umap_emb, axis=0)
    )
    aligned_centroids = apply_rotation_to_cloud(
        horizontal_centroids, R_pos, center=np.nanmean(horizontal_centroids, axis=0)
    )

    # direction consistency on y
    left = aligned_umap_emb[mov_dir == -1.0, 1]
    right = aligned_umap_emb[mov_dir == +1.0, 1]
    if len(left) > 0 and len(right) > 0:
        if np.nanmean(left) < np.nanmean(right):
            aligned_umap_emb[:, 1] *= -1.0
            if len(aligned_centroids) > 0:
                aligned_centroids[:, 1] *= -1.0

    return {
        "aligned_umap_emb": aligned_umap_emb,
        "aligned_centroids": aligned_centroids,
        "centroids_pos_dir": centroids_pos_dir_h,
        "normal_vector": normal,
        "cloud_center": cloud_center,
        "R_plane": R_plane,
        "R_pos": R_pos,
        "pos_gradient_axis": pos_gradient_axis,
    }


# =============================================================================
# Ellipse fit
# =============================================================================

def fit_ellipse_to_point_cloud_xy(points_xy, num_points=1000, robust_percentile=90):
    """
    Approximate ellipse in XY using covariance/PCA.
    Returns:
        ellipse_points_3d, ellipse_coeff
    ellipse_coeff contains center, axes, rotation matrix.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    center = np.nanmean(points_xy, axis=0)

    X = points_xy - center
    pca = PCA(n_components=2)
    pca.fit(X)

    rot = pca.components_
    proj = pca.transform(X)

    a = np.nanpercentile(np.abs(proj[:, 0]), robust_percentile)
    b = np.nanpercentile(np.abs(proj[:, 1]), robust_percentile)
    a = max(a, 1e-6)
    b = max(b, 1e-6)

    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    ellipse_local = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
    ellipse_xy = ellipse_local @ rot + center

    ellipse_points = np.column_stack([ellipse_xy, np.zeros(len(ellipse_xy))])

    ellipse_coeff = {
        "center_xy": center,
        "axes": np.array([a, b]),
        "rotation_matrix_2d": rot,
    }
    return ellipse_points, ellipse_coeff


def annotate_ellipse_points(ellipse_points, aligned_umap_emb, pos_1d, mov_dir, k=300):
    """
    For each ellipse point, assign average position and majority direction
    from nearest neighbors in aligned embedding.
    """
    ellipse_features = np.zeros((ellipse_points.shape[0], 2), dtype=float)

    XY = aligned_umap_emb[:, :2]
    for i in range(ellipse_points.shape[0]):
        point = ellipse_points[i, :2]
        d = np.sum((XY - point) ** 2, axis=1)
        neigh_idx = np.argsort(d)[:min(k, len(d))]
        ellipse_features[i, 0] = np.nanmean(pos_1d[neigh_idx])
        ellipse_features[i, 1] = mode_binary_sign(mov_dir[neigh_idx])

    return ellipse_features


# =============================================================================
# Flow computation
# =============================================================================

def compute_flow_metrics(aligned_umap_emb, ellipse_points, pos_1d, mov_dir, trial_id, speed):
    """
    Compute flow and derived metrics.
    """
    aligned_umap_emb = np.asarray(aligned_umap_emb, dtype=float)
    n = aligned_umap_emb.shape[0]

    if n <= FLOW_DIFF:
        raise ValueError("Not enough points to compute flow.")

    # angular position on aligned ring
    angular_pos = np.degrees(np.arctan2(aligned_umap_emb[:, 1], aligned_umap_emb[:, 0]))

    # distance to ellipse and in/off ring
    D = pairwise_distances(aligned_umap_emb, ellipse_points)
    dist_to_ellipse = np.min(D, axis=1)
    closest_ellipse_idxs = np.argmin(D, axis=1)

    off_ring_points = dist_to_ellipse > np.percentile(dist_to_ellipse, OFF_RING_PERCENTILE)
    in_ring_points = ~off_ring_points

    # flow
    flow = (aligned_umap_emb[FLOW_DIFF:] - aligned_umap_emb[:-FLOW_DIFF]) / 2.0
    flow_modulus = np.linalg.norm(flow, axis=1)

    flow_xy_angle = np.degrees(np.arctan2(flow[:, 1], flow[:, 0]))
    flow_yz_angle = np.degrees(np.arctan2(flow[:, 2], np.linalg.norm(flow[:, :2], axis=1)))

    tangential_flow = np.zeros_like(flow)
    radial_flow = np.zeros_like(flow)
    residual_flow = np.zeros_like(flow)

    radial_modulus = np.zeros(flow.shape[0], dtype=float)
    tangential_modulus = np.zeros(flow.shape[0], dtype=float)
    radial_modulus_signed = np.zeros(flow.shape[0], dtype=float)

    for i in range(flow.shape[0]):
        emb_point = aligned_umap_emb[i, :]
        closest_ellipse_point = ellipse_points[closest_ellipse_idxs[i], :]
        radial_direction = emb_point - closest_ellipse_point
        tangential_direction = find_perpendicular_vector_on_xy_plane(radial_direction)

        radial_flow[i, :] = project_onto_vector(flow[i, :], radial_direction)
        tangential_flow[i, :] = project_onto_vector(flow[i, :], tangential_direction)
        residual_flow[i, :] = flow[i, :] - radial_flow[i, :] - tangential_flow[i, :]

        radial_modulus[i] = np.linalg.norm(radial_flow[i, :])
        radial_sign = np.sign(np.dot(radial_flow[i, :], radial_direction))
        radial_modulus_signed[i] = radial_sign * radial_modulus[i]
        tangential_modulus[i] = np.linalg.norm(tangential_flow[i, :])

    residual_modulus = np.linalg.norm(residual_flow, axis=1)

    # Match masks/behavior to flow indexing
    in_ring_flow = in_ring_points[:-FLOW_DIFF]
    off_ring_flow = off_ring_points[:-FLOW_DIFF]
    angular_pos_flow = angular_pos[:-FLOW_DIFF]
    pos_flow = pos_1d[:-FLOW_DIFF]
    mov_dir_flow = mov_dir[:-FLOW_DIFF]
    trial_id_flow = trial_id[:-FLOW_DIFF] if trial_id is not None else None
    speed_flow = speed[:-FLOW_DIFF] if speed is not None else None

    # flow vs ideal angular progression
    line_1 = np.column_stack((np.linspace(-180, 90, 1000), np.linspace(-90, 180, 1000)))
    line_2 = np.column_stack((np.linspace(90, 180, 1000), np.linspace(-180, -90, 1000)))
    ideal_dist = np.concatenate((line_1, line_2), axis=0)

    if np.any(in_ring_flow):
        real_dist = np.column_stack((angular_pos_flow[in_ring_flow], flow_xy_angle[in_ring_flow]))
        diff_dist = np.min(pairwise_distances(real_dist, ideal_dist), axis=1)
        flow_pos_diff = float(np.nanmean(diff_dist))
    else:
        diff_dist = np.array([], dtype=float)
        flow_pos_diff = np.nan

    def safe_mean(x, mask=None):
        x = np.asarray(x, dtype=float)
        if mask is not None:
            x = x[mask]
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.nan
        return float(np.mean(x))

    mean_in_rad = safe_mean(radial_modulus, in_ring_flow)
    mean_in_tan = safe_mean(tangential_modulus, in_ring_flow)
    mean_in_res = safe_mean(residual_modulus, in_ring_flow)
    mean_in_flow = safe_mean(flow_modulus, in_ring_flow)
    mean_in_signed_rad = safe_mean(radial_modulus_signed, in_ring_flow)

    mean_off_rad = safe_mean(radial_modulus, off_ring_flow)
    mean_off_tan = safe_mean(tangential_modulus, off_ring_flow)
    mean_off_res = safe_mean(residual_modulus, off_ring_flow)
    mean_off_flow = safe_mean(flow_modulus, off_ring_flow)
    mean_off_signed_rad = safe_mean(radial_modulus_signed, off_ring_flow)

    in_rad_tan_ratio = mean_in_rad / mean_in_tan if np.isfinite(mean_in_rad) and np.isfinite(mean_in_tan) and mean_in_tan != 0 else np.nan
    off_rad_tan_ratio = mean_off_rad / mean_off_tan if np.isfinite(mean_off_rad) and np.isfinite(mean_off_tan) and mean_off_tan != 0 else np.nan

    metrics = {
        "angular_pos": angular_pos,
        "dist_to_ellipse": dist_to_ellipse,
        "closest_ellipse_idxs": closest_ellipse_idxs,
        "in_ring_points": in_ring_points,
        "off_ring_points": off_ring_points,

        "flow": flow,
        "flow_modulus": flow_modulus,
        "flow_xy_angle": flow_xy_angle,
        "flow_yz_angle": flow_yz_angle,

        "tangential_flow": tangential_flow,
        "radial_flow": radial_flow,
        "residual_flow": residual_flow,

        "radial_modulus": radial_modulus,
        "radial_modulus_signed": radial_modulus_signed,
        "tangential_modulus": tangential_modulus,
        "residual_modulus": residual_modulus,

        "in_ring_flow_mask": in_ring_flow,
        "off_ring_flow_mask": off_ring_flow,
        "flow_pos_diff_vector": diff_dist,
        "flow_pos_diff_mean": flow_pos_diff,

        "mean_in_radial_modulus": mean_in_rad,
        "mean_in_tangential_modulus": mean_in_tan,
        "mean_in_residual_modulus": mean_in_res,
        "mean_in_flow_modulus": mean_in_flow,
        "mean_in_radial_modulus_signed": mean_in_signed_rad,
        "mean_in_rad_tan_ratio": in_rad_tan_ratio,

        "mean_off_radial_modulus": mean_off_rad,
        "mean_off_tangential_modulus": mean_off_tan,
        "mean_off_residual_modulus": mean_off_res,
        "mean_off_flow_modulus": mean_off_flow,
        "mean_off_radial_modulus_signed": mean_off_signed_rad,
        "mean_off_rad_tan_ratio": off_rad_tan_ratio,

        "n_in_ring_points": int(np.sum(in_ring_points)),
        "n_off_ring_points": int(np.sum(off_ring_points)),
        "n_in_ring_flow_points": int(np.sum(in_ring_flow)),
        "n_off_ring_flow_points": int(np.sum(off_ring_flow)),
    }

    return metrics


# =============================================================================
# CSV schemas
# =============================================================================

SUMMARY_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "n_timepoints",
    "n_neurons",

    "flow_pos_diff_mean",

    "mean_in_radial_modulus",
    "mean_in_tangential_modulus",
    "mean_in_residual_modulus",
    "mean_in_flow_modulus",
    "mean_in_radial_modulus_signed",
    "mean_in_rad_tan_ratio",

    "mean_off_radial_modulus",
    "mean_off_tangential_modulus",
    "mean_off_residual_modulus",
    "mean_off_flow_modulus",
    "mean_off_radial_modulus_signed",
    "mean_off_rad_tan_ratio",

    "n_in_ring_points",
    "n_off_ring_points",
    "n_in_ring_flow_points",
    "n_off_ring_flow_points",

    "analysis_file_path",
    "flow_pkl_path",
]

RING_SPLIT_COLUMNS = [
    "dataset_name",
    "mouse_name",
    "session_name",
    "depth_type",
    "ring_location",
    "radial_modulus",
    "tangential_modulus",
    "residual_modulus",
    "flow_modulus",
    "radial_modulus_signed",
    "rad_tan_ratio",
    "flow_pos_diff_mean",
]


def init_csv(csv_path, columns):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_row(csv_path, columns, row_dict):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row_dict)


# =============================================================================
# Per-session processing
# =============================================================================

def process_analysis_file(analysis_file_path):
    payload = load_pickle_standard(analysis_file_path)
    meta = payload["metadata"]

    dataset_name = meta["dataset_name"]
    mouse_name = meta["mouse_name"]
    session_name = meta["session_name"]
    depth_type = meta["depth_type"]

    umap_emb = np.asarray(payload["umap"], dtype=float)
    beh = payload["behavior"]

    pos_1d = ensure_1d(beh["pos"]).astype(float)
    mov_dir = ensure_1d(beh["mov_dir"]).astype(float) if beh.get("mov_dir", None) is not None else None
    time_1d = ensure_1d(beh["time"]).astype(float) if beh.get("time", None) is not None else None
    speed = ensure_1d(beh["speed"]).astype(float) if beh.get("speed", None) is not None else None
    trial_id = ensure_1d(beh["trial_id"]).astype(float) if beh.get("trial_id", None) is not None else None

    n_timepoints = int(umap_emb.shape[0])
    n_neurons = int(meta["n_neurons"])

    if len(pos_1d) != n_timepoints:
        raise ValueError("Position length does not match UMAP length.")

    mov_dir_clean = clean_mov_dir_from_position(pos_1d, mov_dir=mov_dir, fs=20.0, sigma=5)

    # align embedding
    align_dict = align_ring_embedding(umap_emb, pos_1d, mov_dir_clean)
    aligned_umap_emb = align_dict["aligned_umap_emb"]

    # fit ellipse in XY
    ellipse_points, ellipse_coeff = fit_ellipse_to_point_cloud_xy(
        aligned_umap_emb[:, :2],
        num_points=ELLIPSE_NUM_POINTS,
        robust_percentile=90
    )
    ellipse_features = annotate_ellipse_points(
        ellipse_points, aligned_umap_emb, pos_1d, mov_dir_clean, k=ELLIPSE_NEIGHBORS
    )

    # flow metrics
    flow_dict = compute_flow_metrics(
        aligned_umap_emb=aligned_umap_emb,
        ellipse_points=ellipse_points,
        pos_1d=pos_1d,
        mov_dir=mov_dir_clean,
        trial_id=trial_id,
        speed=speed
    )

    # save per-session pkl
    out_base = base_name_from_metadata(meta)
    flow_pkl_path = os.path.join(PKL_DIR, out_base + "_flow_dynamics.pkl")

    save_payload = {
        "metadata": {
            "dataset_name": dataset_name,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "depth_type": depth_type,
            "n_timepoints": n_timepoints,
            "n_neurons": n_neurons,
            "source_analysis_file_path": analysis_file_path,
            "flow_diff": FLOW_DIFF,
            "n_centroids": N_CENTROIDS,
            "ellipse_num_points": ELLIPSE_NUM_POINTS,
            "ellipse_neighbors": ELLIPSE_NEIGHBORS,
            "off_ring_percentile": OFF_RING_PERCENTILE,
        },
        "behavior": {
            "pos": pos_1d,
            "mov_dir_clean": mov_dir_clean,
            "time": time_1d,
            "speed": speed,
            "trial_id": trial_id,
        },
        "umap_emb": umap_emb,
        "aligned_umap_emb": aligned_umap_emb,
        "alignment": align_dict,
        "ellipse_points": ellipse_points,
        "ellipse_features": ellipse_features,
        "ellipse_coeff": ellipse_coeff,
        "flow_metrics": flow_dict,
    }
    save_pickle_standard(flow_pkl_path, save_payload)

    # summary row
    summary_row = {
        "dataset_name": dataset_name,
        "mouse_name": mouse_name,
        "session_name": session_name,
        "depth_type": depth_type,
        "n_timepoints": n_timepoints,
        "n_neurons": n_neurons,

        "flow_pos_diff_mean": flow_dict["flow_pos_diff_mean"],

        "mean_in_radial_modulus": flow_dict["mean_in_radial_modulus"],
        "mean_in_tangential_modulus": flow_dict["mean_in_tangential_modulus"],
        "mean_in_residual_modulus": flow_dict["mean_in_residual_modulus"],
        "mean_in_flow_modulus": flow_dict["mean_in_flow_modulus"],
        "mean_in_radial_modulus_signed": flow_dict["mean_in_radial_modulus_signed"],
        "mean_in_rad_tan_ratio": flow_dict["mean_in_rad_tan_ratio"],

        "mean_off_radial_modulus": flow_dict["mean_off_radial_modulus"],
        "mean_off_tangential_modulus": flow_dict["mean_off_tangential_modulus"],
        "mean_off_residual_modulus": flow_dict["mean_off_residual_modulus"],
        "mean_off_flow_modulus": flow_dict["mean_off_flow_modulus"],
        "mean_off_radial_modulus_signed": flow_dict["mean_off_radial_modulus_signed"],
        "mean_off_rad_tan_ratio": flow_dict["mean_off_rad_tan_ratio"],

        "n_in_ring_points": flow_dict["n_in_ring_points"],
        "n_off_ring_points": flow_dict["n_off_ring_points"],
        "n_in_ring_flow_points": flow_dict["n_in_ring_flow_points"],
        "n_off_ring_flow_points": flow_dict["n_off_ring_flow_points"],

        "analysis_file_path": analysis_file_path,
        "flow_pkl_path": flow_pkl_path,
    }

    # split rows
    ring_rows = [
        {
            "dataset_name": dataset_name,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "depth_type": depth_type,
            "ring_location": "in",
            "radial_modulus": flow_dict["mean_in_radial_modulus"],
            "tangential_modulus": flow_dict["mean_in_tangential_modulus"],
            "residual_modulus": flow_dict["mean_in_residual_modulus"],
            "flow_modulus": flow_dict["mean_in_flow_modulus"],
            "radial_modulus_signed": flow_dict["mean_in_radial_modulus_signed"],
            "rad_tan_ratio": flow_dict["mean_in_rad_tan_ratio"],
            "flow_pos_diff_mean": flow_dict["flow_pos_diff_mean"],
        },
        {
            "dataset_name": dataset_name,
            "mouse_name": mouse_name,
            "session_name": session_name,
            "depth_type": depth_type,
            "ring_location": "off",
            "radial_modulus": flow_dict["mean_off_radial_modulus"],
            "tangential_modulus": flow_dict["mean_off_tangential_modulus"],
            "residual_modulus": flow_dict["mean_off_residual_modulus"],
            "flow_modulus": flow_dict["mean_off_flow_modulus"],
            "radial_modulus_signed": flow_dict["mean_off_radial_modulus_signed"],
            "rad_tan_ratio": flow_dict["mean_off_rad_tan_ratio"],
            "flow_pos_diff_mean": flow_dict["flow_pos_diff_mean"],
        },
    ]

    # text summary
    txt_lines = [
        "UMAP flow dynamics summary",
        "=" * 60,
        f"dataset_name: {dataset_name}",
        f"mouse_name: {mouse_name}",
        f"session_name: {session_name}",
        f"depth_type: {depth_type}",
        f"n_timepoints: {n_timepoints}",
        f"n_neurons: {n_neurons}",
        "",
        f"flow_pos_diff_mean: {flow_dict['flow_pos_diff_mean']}",
        "",
        f"mean_in_radial_modulus: {flow_dict['mean_in_radial_modulus']}",
        f"mean_in_tangential_modulus: {flow_dict['mean_in_tangential_modulus']}",
        f"mean_in_residual_modulus: {flow_dict['mean_in_residual_modulus']}",
        f"mean_in_flow_modulus: {flow_dict['mean_in_flow_modulus']}",
        f"mean_in_radial_modulus_signed: {flow_dict['mean_in_radial_modulus_signed']}",
        f"mean_in_rad_tan_ratio: {flow_dict['mean_in_rad_tan_ratio']}",
        "",
        f"mean_off_radial_modulus: {flow_dict['mean_off_radial_modulus']}",
        f"mean_off_tangential_modulus: {flow_dict['mean_off_tangential_modulus']}",
        f"mean_off_residual_modulus: {flow_dict['mean_off_residual_modulus']}",
        f"mean_off_flow_modulus: {flow_dict['mean_off_flow_modulus']}",
        f"mean_off_radial_modulus_signed: {flow_dict['mean_off_radial_modulus_signed']}",
        f"mean_off_rad_tan_ratio: {flow_dict['mean_off_rad_tan_ratio']}",
        "",
        f"n_in_ring_points: {flow_dict['n_in_ring_points']}",
        f"n_off_ring_points: {flow_dict['n_off_ring_points']}",
        f"n_in_ring_flow_points: {flow_dict['n_in_ring_flow_points']}",
        f"n_off_ring_flow_points: {flow_dict['n_off_ring_flow_points']}",
    ]
    txt_path = os.path.join(TXT_DIR, out_base + "_flow_dynamics.txt")
    save_text("\n".join(txt_lines), txt_path)

    return summary_row, ring_rows


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PKL_DIR)
    ensure_dir(TXT_DIR)

    if not os.path.isfile(ANALYSIS_INDEX_CSV):
        raise FileNotFoundError(f"Missing analysis index CSV: {ANALYSIS_INDEX_CSV}")

    df = pd.read_csv(ANALYSIS_INDEX_CSV)

    if DATASET_FILTER is not None:
        df = df[df["dataset_name"] == DATASET_FILTER].copy()

    df = df.reset_index(drop=True)

    if MAX_ROWS is not None:
        df = df.iloc[:MAX_ROWS].copy()

    if "analysis_file_path" not in df.columns:
        raise ValueError("Expected column 'analysis_file_path' in analysis index CSV.")

    init_csv(SUMMARY_CSV, SUMMARY_COLUMNS)
    init_csv(RING_SPLIT_CSV, RING_SPLIT_COLUMNS)

    ok_count = 0
    fail_count = 0

    print("==============================================================")
    print("UMAP flow dynamics")
    print("==============================================================")
    print(f"Rows to process: {len(df)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("")

    for row_idx, row in df.iterrows():
        analysis_file_path = row["analysis_file_path"]
        try:
            summary_row, ring_rows = process_analysis_file(analysis_file_path)
            append_csv_row(SUMMARY_CSV, SUMMARY_COLUMNS, summary_row)
            for rr in ring_rows:
                append_csv_row(RING_SPLIT_CSV, RING_SPLIT_COLUMNS, rr)
            ok_count += 1
            print(f"[OK] Row {row_idx}: {analysis_file_path}")
        except Exception as e:
            fail_count += 1
            print(f"[ERROR] Row {row_idx}: {analysis_file_path} -> {e}")

    print("")
    print("Done.")
    print(f"Successful entries: {ok_count}")
    print(f"Failed entries: {fail_count}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Ring split CSV: {RING_SPLIT_CSV}")


if __name__ == "__main__":
    main()

