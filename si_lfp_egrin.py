#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Structure Index (SI) Analysis on the lt_egrin Dataset (NetworkX optional fallback)."""

import copy
import glob
import os
import pickle
import warnings
from decorator import decorator
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

# Optional imports
try:
  import networkx as nx

  HAS_NETWORKX = True
except ImportError:
  HAS_NETWORKX = False

try:
  import faiss

  use_fast = True
except ImportError:
  use_fast = False


# =============================================================================
# Structure Index Core Algorithm
# =============================================================================

distance_options = ["euclidean", "geodesic"]


def validate_args_types(**decls):

  @decorator
  def wrapper(func, *args, **kwargs):
    code = func.__code__
    fname = func.__name__
    names = code.co_varnames[: code.co_argcount]
    for argname, argtype in decls.items():
      arg_provided = True
      if argname in names:
        argval = args[names.index(argname)]
      elif argname in kwargs:
        argval = kwargs.get(argname)
      else:
        arg_provided = False
      if arg_provided:
        if not isinstance(argval, argtype):
          raise TypeError(
              f"{fname}(...): arg '{argname}': type is {type(argval)}, must be"
              f" {argtype}"
          )
    return func(*args, **kwargs)

  return wrapper


def filter_noisy_outliers(data):
  D = pairwise_distances(data)
  np.fill_diagonal(D, np.nan)
  nn_dist = np.sum(D < np.nanpercentile(D, 1), axis=1) - 1
  noiseIdx = np.where(nn_dist < np.percentile(nn_dist, 20))[0]
  return noiseIdx


def meshgrid2(arrs):
  lens = list(map(len, arrs))
  dim = len(arrs)
  sz = 1
  for s in lens:
    sz *= s
  ans = []
  for i, arr in enumerate(arrs):
    slc = [1] * dim
    slc[i] = lens[i]
    arr2 = np.asarray(arr).reshape(slc)
    for j, sz in enumerate(lens):
      if j != i:
        arr2 = arr2.repeat(sz, axis=j)
    ans.append(arr2)
  return tuple(ans)


def create_ndim_grid(label, n_bins, min_label, max_label, discrete_label):
  ndims = label.shape[1]
  grid_edges = list()
  for nd in range(ndims):
    if discrete_label[nd]:
      grid_edges.append(
          np.tile(np.unique(label[:, nd]).reshape(-1, 1), (1, 2))
      )
    else:
      edges = np.linspace(
          min_label[nd], max_label[nd], n_bins[nd] + 1
      ).reshape(-1, 1)
      grid_edges.append(
          np.concatenate((edges[:-1], edges[1:]), axis=1)
      )

  grid = np.empty([e.shape[0] for e in grid_edges], object)
  mesh = meshgrid2(tuple([np.arange(s) for s in grid.shape]))
  meshIdx = np.vstack([col.ravel() for col in mesh]).T
  coords = np.zeros(meshIdx.shape + (3,))
  grid = grid.ravel()

  for elem, idx in enumerate(meshIdx):
    logic = np.zeros(label.shape[0])
    for dim in range(len(idx)):
      min_edge = grid_edges[dim][idx[dim], 0]
      max_edge = grid_edges[dim][idx[dim], 1]
      logic = logic + 1 * np.logical_and(
          label[:, dim] >= min_edge, label[:, dim] <= max_edge
      )
      coords[elem, dim, 0] = min_edge
      coords[elem, dim, 1] = 0.5 * (min_edge + max_edge)
      coords[elem, dim, 2] = max_edge
    grid[elem] = list(np.where(logic == meshIdx.shape[1])[0])

  return grid, coords


def cloud_overlap_radius(cloud1, cloud2, r, distance_metric):
  cloud_all = np.vstack((cloud1, cloud2)).astype("float32")
  idx_sep = cloud1.shape[0]

  if distance_metric == "euclidean":
    D = distance_matrix(cloud_all, cloud_all, p=2)
  elif distance_metric == "geodesic":
    model_iso = Isomap(n_components=1)
    model_iso.fit_transform(cloud_all)
    D = model_iso.dist_matrix_

  I = np.argsort(D, axis=1)
  for row in range(I.shape[0]):
    D[row, :] = D[row, I[row, :]]
  I = I[:, 1:].astype("float32")
  D = D[:, 1:]
  I[D > r] = np.nan
  num_neigh = I.shape[0] - np.sum(np.isnan(I), axis=1).astype("float32") - 1

  overlap_1_2 = np.sum(I[:idx_sep, :] >= idx_sep) / np.sum(
      num_neigh[:idx_sep]
  )
  overlap_2_1 = np.sum(I[idx_sep:, :] < idx_sep) / np.sum(
      num_neigh[idx_sep:]
  )
  return overlap_1_2, overlap_2_1


def cloud_overlap_neighbors(cloud1, cloud2, k, distance_metric):
  cloud_all = np.vstack((cloud1, cloud2)).astype("float32")
  idx_sep = cloud1.shape[0]

  if distance_metric == "euclidean":
    if use_fast:
      index = faiss.IndexFlatL2(cloud_all.shape[1])
      index.add(cloud_all)
      _, I = index.search(cloud_all, k + 1)
      I = I[:, 1:]
    else:
      knn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2).fit(
          cloud_all
      )
      I = knn.kneighbors(return_distance=False)
  elif distance_metric == "geodesic":
    model_iso = Isomap(n_components=1)
    model_iso.fit_transform(cloud_all)
    dist_mat = model_iso.dist_matrix_
    knn = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(dist_mat)
    I = knn.kneighbors(return_distance=False)

  overlap_1_2 = np.sum(I[:idx_sep, :] >= idx_sep) / (cloud1.shape[0] * k)
  overlap_2_1 = np.sum(I[idx_sep:, :] < idx_sep) / (cloud2.shape[0] * k)
  return overlap_1_2, overlap_2_1


@validate_args_types(
    data=np.ndarray,
    label=np.ndarray,
    n_bins=(int, np.integer, list),
    dims=(type(None), list),
    distance_metric=str,
    n_neighbors=(int, np.integer),
    num_shuffles=(int, np.integer),
    discrete_label=(list, bool),
    verbose=bool,
)
def compute_structure_index(data, label, n_bins=10, dims=None, **kwargs):
  assert data.ndim == 2, (
      "Input 'data' must be a 2D numpy ndarray with shape of samples and m"
      " dimensions."
  )
  if label.ndim == 1:
    label = label.reshape(-1, 1)
  assert label.ndim == 2, "label must be a 1D or 2D array."

  if isinstance(n_bins, (int, np.integer)):
    assert n_bins > 1, (
        "Input 'n_bins' must be an int or list of int larger than 1."
    )
    n_bins = [int(n_bins) for _ in range(label.shape[1])]
  elif isinstance(n_bins, list):
    assert np.all([nb > 1 for nb in n_bins]), (
        "Input 'n_bins' must be an int or list of int larger than 1."
    )

  if dims is None:
    dims = list(range(data.shape[1]))

  distance_metric = kwargs.get("distance_metric", "euclidean")
  assert distance_metric in distance_options, (
      f"Invalid distance_metric. Valid: {distance_options}"
  )

  if ("n_neighbors" in kwargs) and ("radius" in kwargs):
    raise ValueError(
        "Both n_neighbors and radius provided. Please specify only one."
    )

  if "radius" in kwargs:
    neighborhood_size = kwargs["radius"]
    assert neighborhood_size > 0, "Input 'radius' must be larger than 0."
    cloud_overlap = cloud_overlap_radius
  else:
    neighborhood_size = kwargs.get("n_neighbors", 15)
    assert neighborhood_size > 2, "n_neighbors must be larger than 2."
    cloud_overlap = cloud_overlap_neighbors

  if "discrete_label" in kwargs:
    discrete_label = kwargs["discrete_label"]
    if isinstance(discrete_label, bool):
      discrete_label = [discrete_label for _ in range(label.shape[1])]
    else:
      assert np.all([isinstance(idx, bool) for idx in discrete_label]), (
          "Input 'discrete_label' must be boolean or list of booleans."
      )
  else:
    discrete_label = [False for _ in range(label.shape[1])]

  num_shuffles = kwargs.get("num_shuffles", 100)
  assert num_shuffles >= 0, (
      "num_shuffles must fall within the interval [0, np.inf)"
  )
  verbose = kwargs.get("verbose", False)

  # 1. Preprocess
  data = data[:, dims]
  if data.ndim == 1:
    data = data.reshape(-1, 1)

  data_nans = np.any(np.isnan(data), axis=1)
  label_nans = np.any(np.isnan(label), axis=1)
  delete_nans = np.where(data_nans | label_nans)[0]

  data = np.delete(data, delete_nans, axis=0)
  label = np.delete(label, delete_nans, axis=0)

  for dim in range(label.shape[1]):
    num_unique_label = len(np.unique(label[:, dim]))
    if discrete_label[dim]:
      n_bins[dim] = num_unique_label
    elif n_bins[dim] >= num_unique_label:
      warnings.warn(
          f"Along column {dim}, label has less unique values"
          f" ({num_unique_label}) than n_bins ({n_bins[dim]})."
      )
      n_bins[dim] = num_unique_label
      discrete_label[dim] = True

  if "min_label" in kwargs:
    min_label = kwargs["min_label"]
    if not isinstance(min_label, list):
      min_label = [min_label for _ in range(label.shape[1])]
  else:
    min_label = np.percentile(label, 5, axis=0)
    if any(discrete_label):
      min_label[discrete_label] = np.min(label[:, discrete_label])

  if "max_label" in kwargs:
    max_label = kwargs["max_label"]
    if not isinstance(max_label, list):
      max_label = [max_label for _ in range(label.shape[1])]
  else:
    max_label = np.percentile(label, 95, axis=0)
    if any(discrete_label):
      max_label[discrete_label] = np.max(label[:, discrete_label])

  for ld in range(label.shape[1]):
    label[np.where(label[:, ld] < min_label[ld])[0], ld] = (
        min_label[ld] + 0.00001
    )
    label[np.where(label[:, ld] > max_label[ld])[0], ld] = (
        max_label[ld] - 0.00001
    )

  grid, coords = create_ndim_grid(
      label, n_bins, min_label, max_label, discrete_label
  )
  bin_label = np.zeros(label.shape[0]) * np.nan
  for b in range(len(grid)):
    bin_label[grid[b]] = b

  if kwargs.get("filter_noise", False):
    for l in range(len(grid)):
      noise_idx = filter_noisy_outliers(data[bin_label == l, :])
      noise_idx = np.where(bin_label == l)[0][noise_idx]
      bin_label[noise_idx] = 0

  unique_bin_label = np.unique(bin_label[~np.isnan(bin_label)])
  n_points = np.array([np.sum(bin_label == val) for val in unique_bin_label])
  min_points_per_bin = 0.1 * data.shape[0] / np.prod(n_bins)
  del_labels = np.where(n_points < min_points_per_bin)[0]

  for del_idx in del_labels:
    bin_label[bin_label == unique_bin_label[del_idx]] = np.nan

  unique_bin_label = np.unique(bin_label[~np.isnan(bin_label)])
  if len(unique_bin_label) <= 1:
    return np.nan, (np.nan, np.nan), np.nan, np.nan

  # 2. Overlap Matrix & SI
  num_bins = len(unique_bin_label)
  overlap_mat = np.zeros((num_bins, num_bins)) * np.nan

  for a in range(num_bins):
    A = data[bin_label == unique_bin_label[a]]
    for b in range(a + 1, num_bins):
      B = data[bin_label == unique_bin_label[b]]
      overlap_a_b, overlap_b_a = cloud_overlap(
          A, B, neighborhood_size, distance_metric
      )
      overlap_mat[a, b] = overlap_a_b
      overlap_mat[b, a] = overlap_b_a

  degree_nodes = np.nansum(overlap_mat, axis=1)
  SI = 1 - np.mean(degree_nodes) / (num_bins - 1)
  SI = 2 * (SI - 0.5)
  SI = np.max([SI, 0])

  # 3. Shuffling
  shuf_SI = np.zeros((num_shuffles,)) * np.nan
  shuf_overlap_mat = np.zeros(overlap_mat.shape)

  for s_idx in range(num_shuffles):
    shuf_bin_label = copy.deepcopy(bin_label)
    np.random.shuffle(shuf_bin_label)
    shuf_overlap_mat *= np.nan
    for a in range(shuf_overlap_mat.shape[0]):
      A = data[shuf_bin_label == unique_bin_label[a]]
      for b in range(a + 1, shuf_overlap_mat.shape[1]):
        B = data[shuf_bin_label == unique_bin_label[b]]
        overlap_a_b, overlap_b_a = cloud_overlap(
            A, B, neighborhood_size, distance_metric
        )
        shuf_overlap_mat[a, b] = overlap_a_b
        shuf_overlap_mat[b, a] = overlap_b_a

    deg = np.nansum(shuf_overlap_mat, axis=1)
    s_si = 1 - np.mean(deg) / (num_bins - 1)
    s_si = 2 * (s_si - 0.5)
    shuf_SI[s_idx] = np.max([s_si, 0])

  return SI, (bin_label, coords), overlap_mat, shuf_SI


def draw_graph_or_heatmap(
    overlap_mat,
    ax,
    node_cmap=plt.cm.tab10,
    edge_cmap=plt.cm.Greys,
    **kwargs,
):
  """Draws NetworkX circular graph if available, otherwise plots an overlap matrix heatmap."""
  if HAS_NETWORKX:
    if int(nx.__version__[0]) < 3:
      g = nx.from_numpy_matrix(overlap_mat, create_using=nx.DiGraph)
    else:
      g = nx.from_numpy_array(overlap_mat, create_using=nx.DiGraph)

    number_nodes = g.number_of_nodes()
    node_size = kwargs.get("node_size", 800)
    scale_edges = kwargs.get("scale_edges", 5)
    edge_vmin = kwargs.get("edge_vmin", 0)
    edge_vmax = kwargs.get("edge_vmax", 0.5)
    arrow_size = kwargs.get("arrow_size", 20)
    layout_type = kwargs.get("layout_type", nx.circular_layout)

    node_names = kwargs.get("node_names", range(number_nodes))
    names_dict = {
        val[0]: node_names[i] for i, val in enumerate(list(g.nodes(data=True)))
    }
    with_labels = kwargs.get("with_labels", False)

    node_color = kwargs.get("node_color", False)
    if not node_color:
      norm_cmap = matplotlib.colors.Normalize(
          vmin=np.min(node_names), vmax=np.max(node_names)
      )
      node_color = [
          np.array(node_cmap(norm_cmap(node_names[ii]), bytes=True)) / 255
          for ii in range(number_nodes)
      ]

    widths = nx.get_edge_attributes(g, "weight")
    return nx.draw_networkx(
        g,
        pos=layout_type(g),
        node_size=node_size,
        node_color=node_color,
        width=np.array(list(widths.values())) * scale_edges,
        edge_color=np.array(list(widths.values())),
        edge_cmap=edge_cmap,
        arrowsize=arrow_size,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        labels=names_dict,
        arrows=True,
        connectionstyle="arc3,rad=0.15",
        with_labels=with_labels,
        ax=ax,
    )
  else:
    # Matplotlib heatmap fallback if networkx is missing
    im = ax.imshow(overlap_mat, cmap="viridis", vmin=0, vmax=0.5)
    ax.set_title("Overlap Matrix", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# =============================================================================
# Directories & Setup
# =============================================================================

PROJECT_ROOT = "/home/hippolab/Documents/melisa/time_project_standarization"
STANDARDIZED_DIR = os.path.join(
    PROJECT_ROOT, "data", "standardized_sessions", "lt_egrin"
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "analysis", "structure_index_behavior_and_lfp"
)
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
GRAPHS_DIR = os.path.join(PLOTS_DIR, "overlap_graphs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

NUM_PCA_COMPONENTS = 10
NUM_SHUFFLES = 100

BANDS = [
    "ultra_ultra_slow",
    "ultra_slow",
    "slow",
    "delta",
    "theta",
    "gamma",
    "ripple",
    "mua",
]

BAND_DISPLAY = [
    "U-U-Slow",
    "U-Slow",
    "Slow",
    "Delta",
    "Theta",
    "Gamma",
    "Ripple",
    "MUA",
]


# =============================================================================
# Feature Extraction Routine
# =============================================================================

def extract_features(std_data):
  feats = {}

  # 1. Behavior
  pos = (
      std_data["pos"][:, 0]
      if std_data["pos"].ndim > 1
      else std_data["pos"].reshape(-1)
  )
  mov_dir = std_data["mov_dir"].reshape(-1)
  time_arr = std_data["time"].reshape(-1)

  feats["Time"] = (time_arr.reshape(-1, 1), 10, False)
  feats["Position"] = (pos.reshape(-1, 1), 10, False)
  feats["DirPosition"] = ((pos * mov_dir).reshape(-1, 1), 10, False)

  # 2. LFP Envelopes
  for b, b_name in zip(BANDS, BAND_DISPLAY):
    key = f"lfp_{b}_envelope"
    if key in std_data and std_data[key] is not None:
      feats[f"Env_{b_name}"] = (std_data[key].reshape(-1, 1), 10, False)

  # 3. LFP Phases (2D Circular representation)
  for b, b_name in zip(BANDS, BAND_DISPLAY):
    key = f"lfp_{b}_phase"
    if key in std_data and std_data[key] is not None:
      phi = std_data[key].reshape(-1)
      phi_2d = np.column_stack([np.cos(phi), np.sin(phi)])
      feats[f"Phase_{b_name}"] = (phi_2d, [6, 6], [False, False])

  # 4. CWT Power
  if "cwt" in std_data and std_data["cwt"] is not None:
    cwt_obj = std_data["cwt"]
    if isinstance(cwt_obj, dict):
      for b, b_name in zip(BANDS, BAND_DISPLAY):
        if b in cwt_obj and cwt_obj[b] is not None:
          cwt_val = np.abs(cwt_obj[b])
          if cwt_val.ndim == 2:
            cwt_val = np.mean(cwt_val, axis=1)
          feats[f"CWT_{b_name}"] = (cwt_val.reshape(-1, 1), 10, False)
    elif isinstance(cwt_obj, np.ndarray):
      if cwt_obj.ndim == 2 and cwt_obj.shape[1] == len(BANDS):
        for b_idx, b_name in enumerate(BAND_DISPLAY):
          feats[f"CWT_{b_name}"] = (
              np.abs(cwt_obj[:, b_idx]).reshape(-1, 1),
              10,
              False,
          )

  for b, b_name in zip(BANDS, BAND_DISPLAY):
    key = f"cwt_{b}"
    if key in std_data and std_data[key] is not None:
      cwt_val = np.abs(std_data[key])
      if cwt_val.ndim == 2:
        cwt_val = np.mean(cwt_val, axis=1)
      feats[f"CWT_{b_name}"] = (cwt_val.reshape(-1, 1), 10, False)

  return feats


# =============================================================================
# Pipeline Execution
# =============================================================================

def run_si_pipeline():
  files = glob.glob(
      os.path.join(STANDARDIZED_DIR, "**", "*.pkl"), recursive=True
  )
  files += glob.glob(
      os.path.join(STANDARDIZED_DIR, "**", "*_pkl"), recursive=True
  )
  files = sorted(list(set(files)))

  if not files:
    raise FileNotFoundError(
        f"No standardized session files located under {STANDARDIZED_DIR}"
    )

  all_results = []
  print(f"Executing Structure Index on {len(files)} files...")

  for fpath in files:
    fname = os.path.basename(fpath)
    mouse_name = fname.split("_")[0]
    print(f"\n--- Processing: {fname} ---")

    with open(fpath, "rb") as f:
      std_data = pickle.load(f)

    signal = std_data.get("signal")
    if signal is None or signal.shape[1] < 5:
      print("  [WARN] Insufficient neural activity matrix. Skipping.")
      continue

    n_components = min(NUM_PCA_COMPONENTS, signal.shape[1], signal.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    neural_data = pca.fit_transform(signal)

    features_dict = extract_features(std_data)

    for feat_name, (label_mat, n_bins, discrete_flag) in features_dict.items():
      try:
        si, (bin_lbl, coords), overlap_mat, shuf_si = compute_structure_index(
            data=neural_data,
            label=label_mat,
            n_bins=n_bins,
            discrete_label=discrete_flag,
            num_shuffles=NUM_SHUFFLES,
            distance_metric="euclidean",
            n_neighbors=15,
            verbose=False,
        )

        p99 = np.percentile(shuf_si, 99) if len(shuf_si) > 0 else np.nan
        pval = (
            np.mean(shuf_si >= si)
            if (len(shuf_si) > 0 and not np.isnan(si))
            else np.nan
        )

        all_results.append({
            "file_name": fname,
            "mouse_name": mouse_name,
            "feature": feat_name,
            "category": feat_name.split("_")[0],
            "structure_index": si,
            "shuffle_99th": p99,
            "p_value": pval,
            "is_significant": bool(si > p99) if not np.isnan(si) else False,
        })

        if not np.isnan(si) and si > p99 and overlap_mat.shape[0] >= 3:
          fig_g, ax_g = plt.subplots(figsize=(4.5, 4.5), dpi=150)
          draw_graph_or_heatmap(
              overlap_mat,
              ax=ax_g,
              scale_edges=4,
              node_size=500,
              with_labels=False,
          )
          ax_g.set_title(
              f"{mouse_name} | {feat_name}\nSI = {si:.3f} (p < 0.01)",
              fontsize=9,
              fontweight="bold",
          )
          g_out = os.path.join(
              GRAPHS_DIR, f"{mouse_name}_{feat_name}_overlap_graph.png"
          )
          fig_g.savefig(g_out, bbox_inches="tight")
          plt.close(fig_g)

      except Exception as e:
        print(f"  [FAIL] {feat_name}: {e}")

  df_res = pd.DataFrame(all_results)
  table_out = os.path.join(
      TABLES_DIR, "lt_egrin_structure_index_summary_table.csv"
  )
  df_res.to_csv(table_out, index=False)
  print(f"\n[OK] Saved Structure Index Table -> {table_out}")

  plot_si_comparisons(df_res)


def plot_si_comparisons(df_res):
  if df_res.empty:
    return

  fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=250, sharey=True)
  fig.subplots_adjust(hspace=0.35, wspace=0.15)

  panels = [
      ("Behavior", ["Time", "Position", "DirPosition"], axes[0, 0]),
      ("LFP Envelopes", [f"Env_{b}" for b in BAND_DISPLAY], axes[0, 1]),
      (
          "LFP Instantaneous Phases",
          [f"Phase_{b}" for b in BAND_DISPLAY],
          axes[1, 0],
      ),
      (
          "LFP Continuous Wavelet Transform (CWT)",
          [f"CWT_{b}" for b in BAND_DISPLAY],
          axes[1, 1],
      ),
  ]

  for p_title, feat_list, ax in panels:
    sub = df_res[df_res["feature"].isin(feat_list)]
    if sub.empty:
      ax.text(
          0.5,
          0.5,
          "No data found",
          ha="center",
          va="center",
          transform=ax.transAxes,
      )
      continue

    means = sub.groupby("feature")["structure_index"].mean().reindex(feat_list)
    sems = (
        sub.groupby("feature")["structure_index"]
        .sem()
        .reindex(feat_list)
        .fillna(0)
    )
    p99_means = (
        sub.groupby("feature")["shuffle_99th"].mean().reindex(feat_list)
    )

    x_pos = np.arange(len(feat_list))

    ax.bar(
        x_pos,
        means,
        yerr=sems,
        width=0.62,
        color="#3b528b",
        alpha=0.85,
        edgecolor="#222222",
        capsize=3.5,
        zorder=3,
        label="Observed SI",
    )

    ax.scatter(
        x_pos,
        p99_means,
        color="crimson",
        marker="_",
        s=120,
        linewidths=2.5,
        label="Shuffle 99th %",
        zorder=5,
    )

    for idx_f, f_name in enumerate(feat_list):
      f_vals = sub[sub["feature"] == f_name]["structure_index"].values
      jitter = np.random.normal(0, 0.04, size=len(f_vals))
      ax.scatter(
          x_pos[idx_f] + jitter,
          f_vals,
          color="#fde725",
          edgecolors="#222222",
          s=38,
          alpha=0.9,
          zorder=4,
      )

    ax.set_xticks(x_pos)
    labels_clean = [
        f.replace("Env_", "").replace("Phase_", "").replace("CWT_", "")
        for f in feat_list
    ]
    ax.set_xticklabels(labels_clean, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Structure Index (SI)", fontsize=10)
    ax.set_title(p_title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.set_ylim(-0.02, 1.0)

    for spine in ["top", "right"]:
      ax.spines[spine].set_visible(False)

    if ax == axes[0, 0]:
      ax.legend(frameon=True, fontsize=8, loc="upper right")

  fig.suptitle(
      "lt_egrin | Neural Manifold Structure Index: Behavior vs. LFP"
      " Modalities",
      fontsize=14,
      fontweight="bold",
      y=0.98,
  )
  plot_out = os.path.join(
      PLOTS_DIR, "lt_egrin_structure_index_comparison_summary.png"
  )
  fig.savefig(plot_out, bbox_inches="tight")
  plt.close(fig)
  print(f"[OK] Saved Summary Bar Plot -> {plot_out}")


if __name__ == "__main__":
  run_si_pipeline()