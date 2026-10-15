#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rnn_osc_interference_2d_sweep.py

This script performs a 2D parameter sweep: external frequency (f) x external coupling strength (g_ext)
for TWO distinct comparison modes:
  1) slow_component: Direct infraslow/slow oscillation input fed along with fast theta (8 Hz).
  2) slow_beat: Oscillator interference beating where a second fast theta (8.0 + f) Hz is fed along with fast theta (8 Hz).

For each condition and (f, g_ext) pair:
  1) Trains a leaky RNN on repeated 15-lap triangular linear-track navigation trajectories.
  2) Computes Structure Index (SI) on raw hidden states for:
       - position
       - trial time (tau_trial)
       - lap time (tau_lap)
  3) Generates heatmaps of SI, critical frequency curves, and phase (dominance) diagrams.

Outputs are saved in:
  ./sim_freq_x_gext_SI/sweep_slow_component_<timestamp>/
  ./sim_freq_x_gext_SI/sweep_slow_beat_<timestamp>/
"""

import os
import math
import time
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from structure_index import compute_structure_index


# ============================================================
# 0) Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 1) Config
# ============================================================

@dataclass
class Config:
    # ---- Sampling / trial ----
    Fs: float = 20.0
    dt: float = 1.0 / 20.0
    T: int = 1250
    track_len: float = 1.0

    # Laps per trial
    n_go_back: int = 15
    lap_jitter: float = 0.25

    # ---- Inputs ----
    n_landmarks: int = 8
    landmark_width: float = 0.08
    sensory_noise: float = 0.05
    vel_noise: float = 0.02
    osc_amp: float = 1.0
    osc_noise: float = 0.05

    # Reward pulse channel
    use_pulse: bool = True
    pulse_mode: str = "lap_end"
    pulse_amp: float = 1.0
    pulse_width_s: float = 0.25

    # ---- Fast theta base ----
    freq_osc1: float = 8.0

    # ---- Model ----
    n_in: int = 0  # set dynamically: speed (1) + landmarks (8) + osc (4) + pulse (1) = 14
    n_hidden: int = 128
    n_out: int = 8  # [pos, tau_t, sin_t, cos_t, tau_l, sin_l, cos_l, dir_logit]

    tau_rnn: float = 0.1
    w_rec_scale: float = 0.9
    w_in_scale: float = 0.5

    # External coupling parameter (sweep axis)
    g_ext: float = 1.0

    # Regularization
    l2_weight: float = 1e-5
    rate_l2: float = 1e-4

    # Training
    device: str = "cpu"
    batch_size: int = 64
    eval_batch_size: int = 64
    n_epochs: int = 150  # 150 epochs for faster execution during extensive sweeps
    lr: float = 1e-3
    grad_clip: float = 1.0

    # ---- SI params ----
    si_n_bins: int = 10
    si_n_neighbors: int = 50
    si_num_shuffles: int = 0
    si_distance_metric: str = "euclidean"
    si_verbose: bool = False
    si_discrete_label: bool = False


# ============================================================
# 2) Data Simulation & Processing
# ============================================================

def make_landmarks(n_landmarks: int, track_len: float):
    return np.linspace(0.05, track_len - 0.05, n_landmarks)


def sensory_features_from_position(x, centers, width):
    x = np.asarray(x)[..., None]
    centers = centers[None, :]
    return np.exp(-0.5 * ((x - centers) / width) ** 2)


def variable_duration_triangle(cfg: Config, rng: np.random.Generator):
    """
    Build x(t) with EXACTLY cfg.n_go_back laps, variable length, sum == T.
    """
    T = cfg.T
    n = cfg.n_go_back

    w = rng.lognormal(mean=0.0, sigma=cfg.lap_jitter, size=n).astype(np.float64)
    L = np.floor(T * (w / w.sum())).astype(int)

    diff = T - L.sum()
    if diff > 0:
        L[:diff] += 1
    elif diff < 0:
        idx = np.argsort(-L)
        for k in range(-diff):
            L[idx[k % n]] -= 1

    min_len = 8
    if np.any(L < min_len):
        idx_small = np.where(L < min_len)[0]
        idx_big = np.argsort(-L)
        for i in idx_small:
            need = min_len - L[i]
            for _ in range(need):
                j = idx_big[0]
                if L[j] <= min_len:
                    break
                L[j] -= 1
                L[i] += 1

    assert L.sum() == T, f"L.sum()={L.sum()} != T={T}"
    assert np.all(L >= 4)

    x_list, dir_list, lap_list, tauL_list = [], [], [], []

    for lap_i, Li in enumerate(L):
        tauL = (np.arange(Li, dtype=np.float32) / float(Li)).astype(np.float32)
        frac = np.linspace(0.0, 1.0, Li, endpoint=False).astype(np.float32)
        p = 1.0 - np.abs(2.0 * frac - 1.0)
        x_seg = (p * cfg.track_len).astype(np.float32)
        dir_seg = np.where(frac < 0.5, 1.0, -1.0).astype(np.float32)
        lap_seg = np.ones(Li, dtype=np.int32) * lap_i

        x_list.append(x_seg)
        dir_list.append(dir_seg)
        lap_list.append(lap_seg)
        tauL_list.append(tauL)

    x = np.concatenate(x_list, axis=0)
    dir_t = np.concatenate(dir_list, axis=0)
    lap_id = np.concatenate(lap_list, axis=0)
    tau_lap = np.concatenate(tauL_list, axis=0)

    return x, dir_t, lap_id, L, tau_lap


def build_reward_pulse(cfg: Config, L: np.ndarray):
    T = cfg.T
    p = np.zeros(T, dtype=np.float32)
    if not cfg.use_pulse:
        return p

    width = max(1, int(round(cfg.pulse_width_s / cfg.dt)))

    if cfg.pulse_mode == "trial_end":
        p[max(0, T - width):T] = cfg.pulse_amp
        return p

    if cfg.pulse_mode != "lap_end":
        raise ValueError(f"Unknown pulse_mode: {cfg.pulse_mode}")

    start = 0
    for Li in L:
        end = start + int(Li)
        p[max(start, end - width):end] = cfg.pulse_amp
        start = end
    return p


def generate_trial(cfg: Config, landmark_centers, f: float, mode: str, rng: np.random.Generator):
    T = cfg.T
    dt = cfg.dt

    # 1. Trajectory
    x, dir_t, lap_id, L, tau_lap = variable_duration_triangle(cfg, rng=rng)

    # 2. Speed input
    dx = np.diff(x, prepend=x[0])
    v_true = np.abs(dx / dt).astype(np.float32)
    v_in = (v_true - np.mean(v_true)) / (np.std(v_true) + 1e-6)
    v_in = v_in + cfg.vel_noise * rng.standard_normal(T).astype(np.float32)
    v_in = v_in[:, None]

    # 3. Landmarks
    s = sensory_features_from_position(x / cfg.track_len, landmark_centers, cfg.landmark_width).astype(np.float32)
    s = s + cfg.sensory_noise * rng.standard_normal(s.shape).astype(np.float32)

    # 4. Oscillators
    t = np.arange(T, dtype=np.float32) * dt
    
    # Osc 1 is fast base theta (8.0 Hz)
    phase1 = 2.0 * np.pi * cfg.freq_osc1 * t
    osc1_sin = cfg.osc_amp * np.sin(phase1)
    osc1_cos = cfg.osc_amp * np.cos(phase1)

    # Osc 2 is either direct slow component (mode: slow_component) or slow beat (mode: slow_beat)
    if mode == "slow_component":
        f_osc2 = f
    elif mode == "slow_beat":
        f_osc2 = cfg.freq_osc1 + f
    else:
        raise ValueError(f"Unknown mode: {mode}")

    phase2 = 2.0 * np.pi * f_osc2 * t + (np.pi / 4.0)
    osc2_sin = cfg.osc_amp * np.sin(phase2)
    osc2_cos = cfg.osc_amp * np.cos(phase2)

    if cfg.osc_noise > 0:
        osc1_sin += cfg.osc_noise * rng.standard_normal(T)
        osc1_cos += cfg.osc_noise * rng.standard_normal(T)
        osc2_sin += cfg.osc_noise * rng.standard_normal(T)
        osc2_cos += cfg.osc_noise * rng.standard_normal(T)

    osc_in = np.stack([osc1_sin, osc1_cos, osc2_sin, osc2_cos], axis=1).astype(np.float32)

    # 5. Reward pulse
    pulse = build_reward_pulse(cfg, L=L)

    # Inputs: speed (1) + landmarks (8) + osc (4) + pulse (1) = 14 channels
    X_in = np.concatenate([v_in, s, osc_in, pulse[:, None]], axis=1).astype(np.float32)

    # Targets
    y_pos = (x / cfg.track_len).astype(np.float32)[:, None]

    tauT = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
    phaseT = 2.0 * np.pi * tauT
    sinT = np.sin(phaseT).astype(np.float32)
    cosT = np.cos(phaseT).astype(np.float32)

    tauL = tau_lap.astype(np.float32)[:, None]
    phaseL = 2.0 * np.pi * tauL
    sinL = np.sin(phaseL).astype(np.float32)
    cosL = np.cos(phaseL).astype(np.float32)

    dir01 = ((dir_t + 1.0) / 2.0).astype(np.float32)[:, None]

    Y_out = np.concatenate([y_pos, tauT, sinT, cosT, tauL, sinL, cosL, dir01], axis=1).astype(np.float32)

    aux = {"lap_id": lap_id.astype(np.int32)}
    return X_in, Y_out, aux


def make_batch(cfg: Config, landmark_centers, batch_size: int, f: float, mode: str):
    X_list, Y_list = [], []
    seeds = np.random.randint(0, 2**31 - 1, size=batch_size, dtype=np.int64)
    for i in range(batch_size):
        rng = np.random.default_rng(int(seeds[i]))
        X_in, Y_out, _aux = generate_trial(cfg, landmark_centers, f, mode, rng=rng)
        X_list.append(X_in)
        Y_list.append(Y_out)

    X = torch.tensor(np.stack(X_list, axis=0), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y_list, axis=0), dtype=torch.float32)
    return X, Y


# ============================================================
# 3) Leaky RNN with Dynamic External Coupling
# ============================================================

class LeakyRNN(nn.Module):
    """
    Leaky integrator with a scaling factor g_ext applied to osc2 channels
    (indices 11 and 12 in the 14-channel input representation).
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n_in = cfg.n_in
        nH = cfg.n_hidden

        self.W_in = nn.Parameter(torch.randn(n_in, nH) * cfg.w_in_scale / math.sqrt(max(1, n_in)))
        self.W_rec = nn.Parameter(torch.randn(nH, nH) * cfg.w_rec_scale / math.sqrt(nH))
        self.b = nn.Parameter(torch.zeros(nH))

        self.W_out = nn.Linear(nH, cfg.n_out)
        self.alpha = cfg.dt / cfg.tau_rnn

    def forward(self, X):
        B, T, _ = X.shape
        device = X.device
        h = torch.zeros(B, self.cfg.n_hidden, device=device)

        Y_hat, H = [], []

        for t in range(T):
            x_t = X[:, t, :].clone()
            # Indices 11 and 12 contain osc2_sin and osc2_cos
            x_t[:, 11:13] = x_t[:, 11:13] * self.cfg.g_ext

            pre = x_t @ self.W_in + h @ self.W_rec + self.b
            h_tilde = torch.tanh(pre)
            h = (1.0 - self.alpha) * h + self.alpha * h_tilde

            y_t = self.W_out(h)
            Y_hat.append(y_t)
            H.append(h)

        return torch.stack(Y_hat, dim=1), torch.stack(H, dim=1)


# ============================================================
# 4) Loss Function
# ============================================================

def loss_fn(Y_hat, Y_true):
    pos_loss = F.mse_loss(Y_hat[:, :, 0], Y_true[:, :, 0])  # position
    time_t_loss = F.mse_loss(Y_hat[:, :, 1:4], Y_true[:, :, 1:4])  # trial time (tau, sin, cos)
    time_l_loss = F.mse_loss(Y_hat[:, :, 4:7], Y_true[:, :, 4:7])  # lap time (tau, sin, cos)
    dir_loss = F.binary_cross_entropy_with_logits(Y_hat[:, :, 7], Y_true[:, :, 7])  # direction

    return pos_loss + time_t_loss + time_l_loss + 0.5 * dir_loss


# ============================================================
# 5) Plotting Helpers
# ============================================================

def plot_heatmap(M: np.ndarray, xvals: List[float], yvals: List[float],
                 title: str, out_no_ext: str, xlabel: str, ylabel: str):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("SI")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(len(xvals)))
    ax.set_xticklabels([f"{f:g}" for f in xvals], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(yvals)))
    ax.set_yticklabels([f"{g:g}" for g in yvals])

    fig.savefig(out_no_ext + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_no_ext + ".svg", bbox_inches="tight")
    plt.close(fig)


def compute_critical_frequency_curve(SI_mat: np.ndarray,
                                    freqs: List[float],
                                    couplings: List[float],
                                    threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    freqs = np.array(freqs, dtype=float)
    fcrit = np.full((len(couplings),), np.nan, dtype=np.float32)
    for i, _g in enumerate(couplings):
        row = SI_mat[i, :]
        ok = np.where(row >= threshold)[0]
        if ok.size > 0:
            fcrit[i] = float(freqs[np.max(ok)])
    return np.array(couplings, dtype=float), fcrit


def plot_critical_freq_multi_thresholds(SI_mat: np.ndarray,
                                        freqs: List[float],
                                        couplings: List[float],
                                        thresholds: List[float],
                                        out_no_ext: str,
                                        title_prefix: str):
    thr_sorted = sorted(thresholds)
    n = len(thr_sorted)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for k, thr in enumerate(thr_sorted):
        g_vals, fcrit = compute_critical_frequency_curve(SI_mat, freqs, couplings, thr)
        gray = 0.15 + 0.75 * (k / max(1, n-1))
        ax.plot(g_vals, fcrit, marker="o", linewidth=2, color=str(gray), label=f"thr={thr:g}")

    ax.set_xlabel("external coupling g_ext")
    ax.set_ylabel("critical frequency (Hz)")
    ax.set_title(f"{title_prefix}: max f with SI >= threshold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    
    fig.savefig(out_no_ext + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_no_ext + ".svg", bbox_inches="tight")
    plt.close(fig)


def dominance_map(SI_pos: np.ndarray, SI_tt: np.ndarray, SI_tl: np.ndarray):
    stack = np.stack([SI_pos, SI_tt, SI_tl], axis=0)
    dom = np.nanargmax(stack, axis=0)
    return dom.astype(np.int32)


def plot_phase_diagram(dom: np.ndarray, freqs: List[float], couplings: List[float], out_no_ext: str, title: str):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    im = ax.imshow(dom, aspect="auto", origin="lower", cmap="Set2", vmin=-0.5, vmax=2.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(["position", "time_trial", "time_lap"])
    cb.set_label("dominant SI coordinate")

    ax.set_title(title)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("external coupling g_ext")

    ax.set_xticks(np.arange(len(freqs)))
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(couplings)))
    ax.set_yticklabels([f"{g:g}" for g in couplings])

    fig.savefig(out_no_ext + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_no_ext + ".svg", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6) Execution & Core Sweep Loop
# ============================================================

def run_2d_sweep_for_mode(cfg: Config, mode: str, freqs: List[float], couplings: List[float], out_dir: str):
    """
    Executes the full 2D parameter sweep for a given comparison mode (slow_component or slow_beat).
    """
    os.makedirs(out_dir, exist_ok=True)
    
    landmark_centers = make_landmarks(cfg.n_landmarks, cfg.track_len)
    si_params = dict(
        n_bins=cfg.si_n_bins,
        discrete_label=cfg.si_discrete_label,
        n_neighbors=cfg.si_n_neighbors,
        num_shuffles=cfg.si_num_shuffles,
        distance_metric=cfg.si_distance_metric,
        verbose=cfg.si_verbose,
    )

    n_g = len(couplings)
    n_f = len(freqs)

    # Sweep grids
    M_pos = np.zeros((n_g, n_f))
    M_tt = np.zeros((n_g, n_f))
    M_tl = np.zeros((n_g, n_f))

    csv_path = os.path.join(out_dir, "results_grid.csv")
    with open(csv_path, "w") as fp:
        fp.write("mode,g_ext,frequency_hz,si_position,si_time_trial,si_time_lap\n")

    for g_idx, g_val in enumerate(couplings):
        for f_idx, f_hz in enumerate(freqs):
            print(f"\n[Mode: {mode}] Run: g_ext={g_val:.2f}, f={f_hz:.3f} Hz")
            
            # 1. Update config parameters
            cfg.g_ext = g_val
            
            # 2. Build model and optimizer
            set_seed(123)
            model = LeakyRNN(cfg).to(cfg.device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2_weight)

            # 3. Train loop
            for epoch in range(1, cfg.n_epochs + 1):
                model.train()
                X, Y = make_batch(cfg, landmark_centers, cfg.batch_size, f=f_hz, mode=mode)
                X, Y = X.to(cfg.device), Y.to(cfg.device)
                
                opt.zero_grad()
                Y_hat, H = model(X)
                loss = loss_fn(Y_hat, Y) + cfg.rate_l2 * H.pow(2).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

                if epoch in (1, cfg.n_epochs // 2, cfg.n_epochs):
                    print(f"  Epoch {epoch:03d}/{cfg.n_epochs} | loss={loss.item():.6f}")

            # 4. Evaluation
            model.eval()
            set_seed(1000)  # separate eval seed
            with torch.no_grad():
                X_val, Y_val = make_batch(cfg, landmark_centers, cfg.eval_batch_size, f=f_hz, mode=mode)
                X_val, Y_val = X_val.to(cfg.device), Y_val.to(cfg.device)
                _, H_val = model(X_val)

            H_np = H_val.detach().cpu().numpy()
            Y_np = Y_val.detach().cpu().numpy()
            B, T, nH = H_np.shape
            N = B * T

            # Reshape for SI calculation
            states_all = H_np.reshape(N, nH)
            pos_all = Y_np[:, :, 0].reshape(N)
            tt_all = Y_np[:, :, 1].reshape(N)   # trial time
            tl_all = Y_np[:, :, 4].reshape(N)   # lap time

            # 5. Compute Structure Index
            si_pos, _, _, _ = compute_structure_index(states_all, pos_all, **si_params)
            si_tt, _, _, _ = compute_structure_index(states_all, tt_all, **si_params)
            si_tl, _, _, _ = compute_structure_index(states_all, tl_all, **si_params)

            M_pos[g_idx, f_idx] = si_pos
            M_tt[g_idx, f_idx] = si_tt
            M_tl[g_idx, f_idx] = si_tl

            print(f"  SI Results -> pos: {si_pos:.4f} | time_trial: {si_tt:.4f} | time_lap: {si_tl:.4f}")

            with open(csv_path, "a") as fp:
                fp.write(f"{mode},{g_val},{f_hz},{si_pos},{si_tt},{si_tl}\n")

    # 6. Save Grid Arrays
    np.save(os.path.join(out_dir, "grid_si_pos.npy"), M_pos)
    np.save(os.path.join(out_dir, "grid_si_time_trial.npy"), M_tt)
    np.save(os.path.join(out_dir, "grid_si_time_lap.npy"), M_tl)

    # 7. Generate Heatmaps
    plot_heatmap(M_pos, freqs, couplings, f"SI Position ({mode})", os.path.join(out_dir, "heatmap_si_pos"), "frequency (Hz)", "g_ext")
    plot_heatmap(M_tt, freqs, couplings, f"SI Trial Time ({mode})", os.path.join(out_dir, "heatmap_si_time_trial"), "frequency (Hz)", "g_ext")
    plot_heatmap(M_tl, freqs, couplings, f"SI Lap Time ({mode})", os.path.join(out_dir, "heatmap_si_time_lap"), "frequency (Hz)", "g_ext")

    # 8. Critical Frequency curves
    plot_critical_freq_multi_thresholds(M_tt, freqs, couplings, [0.5, 0.6, 0.7, 0.75, 0.8], os.path.join(out_dir, "critical_freq_time_trial"), f"Critical Freq (Trial Time, {mode})")

    # 9. Phase / Dominance Diagram
    dom = dominance_map(M_pos, M_tt, M_tl)
    plot_phase_diagram(dom, freqs, couplings, os.path.join(out_dir, "phase_diagram"), f"SI Dominance Map ({mode})")

    print(f"\n[Finished Mode: {mode}] All outputs saved in: {out_dir}")


# ============================================================
# 7) Main Execution
# ============================================================

def main():
    # Sweep lists matching your 2D diagram requirements
    FREQ_LIST = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]  # sweeps infraslow and beating frequencies
    G_EXT_LIST = [0.0, 0.5, 1.0, 2.0, 4.0]               # sweep coupling gain strengths

    # Base configuration setup
    cfg = Config()
    # n_in: speed (1) + landmarks (8) + osc (4) + pulse (1) = 14
    cfg.n_in = 1 + cfg.n_landmarks + 4 + 1
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ROOT_OUT_BASE = "./sim_freq_x_gext_SI"
    
    print("=" * 80)
    print("Unified 2D Sweep: Infraslow Input vs. Oscillator Interference (Beating)")
    print("=" * 80)
    print(f"Device: {cfg.device}")
    print(f"Fs: {cfg.Fs} Hz | T: {cfg.T} steps")
    print(f"Sweeping frequencies: {FREQ_LIST}")
    print(f"Sweeping couplings: {G_EXT_LIST}")

    # Mode 1: Direct Infraslow (Slow Component)
    out_dir_component = os.path.join(ROOT_OUT_BASE, f"sweep_slow_component_{timestamp}")
    run_2d_sweep_for_mode(cfg, "slow_component", FREQ_LIST, G_EXT_LIST, out_dir_component)

    # Mode 2: Oscillator Interference (Slow Beat)
    out_dir_beat = os.path.join(ROOT_OUT_BASE, f"sweep_slow_beat_{timestamp}")
    run_2d_sweep_for_mode(cfg, "slow_beat", FREQ_LIST, G_EXT_LIST, out_dir_beat)

    print("\nUNIFIED EXPERIMENT COMPLETED SUCCESSFULY!")
    print(f"Direct Infraslow outputs: {os.path.abspath(out_dir_component)}")
    print(f"Oscillator Interference outputs: {os.path.abspath(out_dir_beat)}")


if __name__ == "__main__":
    main()
