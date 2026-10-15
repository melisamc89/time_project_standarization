#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rnn_osc_drift_sweep.py

This script performs a 2D parameter sweep (frequency x g_ext) for slow_component and slow_beat modes,
and systematically measures THREE distinct metrics of representational drift used in computational neuroscience literature:

  1) Within-Trial Drift (First-Last Correlation, C_FL):
     Compares the population vector correlation of corresponding spatial positions between the 
     first 20% and the last 20% of a single trial. Because the linear track trajectory is periodic, 
     any decay in correlation represents within-trial representational drift.

  2) Across-Trial Population Vector Correlation (PVC) Decay Rate:
     Constructs 128-dimensional spatial Population Vectors for each trial across 10 spatial bins.
     Computes the B x B trial correlation matrix, calculates the average PVC as a function of 
     trial lag |i - j|, and fits a linear slope to extract the Across-Trial Drift Rate (R_drift).

  3) Decoder Generalization Decay Rate:
     Trains a linear Ridge regression decoder on Trial 0 to decode Position, evaluates it on subsequent
     trials, and measures the decay rate of decoding accuracy (R^2) as a function of trial lag.

Outputs:
  Saves heatmaps of all drift metrics for both conditions in ./sim_freq_x_gext_drift/
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
from sklearn.linear_model import Ridge


# ============================================================
# 0) Config
# ============================================================

@dataclass
class Config:
    Fs: float = 20.0
    dt: float = 1.0 / 20.0
    T: int = 1250
    track_len: float = 1.0

    n_go_back: int = 15
    lap_jitter: float = 0.25

    n_landmarks: int = 8
    landmark_width: float = 0.08
    sensory_noise: float = 0.05
    vel_noise: float = 0.02
    osc_amp: float = 1.0
    osc_noise: float = 0.05

    use_pulse: bool = True
    pulse_mode: str = "lap_end"
    pulse_amp: float = 1.0
    pulse_width_s: float = 0.25

    freq_osc1: float = 8.0

    n_in: int = 14  # speed (1) + landmarks (8) + osc (4) + pulse (1) = 14
    n_hidden: int = 128
    n_out: int = 8  # pos, tau_t, sin_t, cos_t, tau_l, sin_l, cos_l, dir_logit

    tau_rnn: float = 0.1
    w_rec_scale: float = 0.9
    w_in_scale: float = 0.5
    g_ext: float = 1.0

    l2_weight: float = 1e-5
    rate_l2: float = 1e-4

    device: str = "cpu"
    batch_size: int = 64
    eval_batch_size: int = 32  # 32 trials is fast and highly sufficient for drift statistics
    n_epochs: int = 150
    lr: float = 1e-3
    grad_clip: float = 1.0


# ============================================================
# 1) Utilities & Data Generation
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_landmarks(n_landmarks: int, track_len: float):
    return np.linspace(0.05, track_len - 0.05, n_landmarks)


def sensory_features_from_position(x, centers, width):
    x = np.asarray(x)[..., None]
    centers = centers[None, :]
    return np.exp(-0.5 * ((x - centers) / width) ** 2)


def variable_duration_triangle(cfg: Config, rng: np.random.Generator):
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

    assert L.sum() == T
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

    start = 0
    for Li in L:
        end = start + int(Li)
        p[max(start, end - width):end] = cfg.pulse_amp
        start = end
    return p


def generate_trial(cfg: Config, landmark_centers, f: float, mode: str, rng: np.random.Generator):
    T = cfg.T
    dt = cfg.dt

    x, dir_t, lap_id, L, tau_lap = variable_duration_triangle(cfg, rng=rng)

    dx = np.diff(x, prepend=x[0])
    v_true = np.abs(dx / dt).astype(np.float32)
    v_in = (v_true - np.mean(v_true)) / (np.std(v_true) + 1e-6)
    v_in = v_in + cfg.vel_noise * rng.standard_normal(T).astype(np.float32)
    v_in = v_in[:, None]

    s = sensory_features_from_position(x / cfg.track_len, landmark_centers, cfg.landmark_width).astype(np.float32)
    s = s + cfg.sensory_noise * rng.standard_normal(s.shape).astype(np.float32)

    t = np.arange(T, dtype=np.float32) * dt
    phase1 = 2.0 * np.pi * cfg.freq_osc1 * t
    osc1_sin = cfg.osc_amp * np.sin(phase1)
    osc1_cos = cfg.osc_amp * np.cos(phase1)

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
    pulse = build_reward_pulse(cfg, L=L)

    X_in = np.concatenate([v_in, s, osc_in, pulse[:, None]], axis=1).astype(np.float32)

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

    return X_in, Y_out


def make_batch(cfg: Config, landmark_centers, batch_size: int, f: float, mode: str):
    X_list, Y_list = [], []
    seeds = np.random.randint(0, 2**31 - 1, size=batch_size, dtype=np.int64)
    for i in range(batch_size):
        rng = np.random.default_rng(int(seeds[i]))
        X_in, Y_out = generate_trial(cfg, landmark_centers, f, mode, rng=rng)
        X_list.append(X_in)
        Y_list.append(Y_out)

    X = torch.tensor(np.stack(X_list, axis=0), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y_list, axis=0), dtype=torch.float32)
    return X, Y


# ============================================================
# 2) Leaky RNN model
# ============================================================

class LeakyRNN(nn.Module):
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
            x_t[:, 11:13] = x_t[:, 11:13] * self.cfg.g_ext

            pre = x_t @ self.W_in + h @ self.W_rec + self.b
            h_tilde = torch.tanh(pre)
            h = (1.0 - self.alpha) * h + self.alpha * h_tilde

            y_t = self.W_out(h)
            Y_hat.append(y_t)
            H.append(h)

        return torch.stack(Y_hat, dim=1), torch.stack(H, dim=1)


def loss_fn(Y_hat, Y_true):
    pos_loss = F.mse_loss(Y_hat[:, :, 0], Y_true[:, :, 0])
    time_t_loss = F.mse_loss(Y_hat[:, :, 1:4], Y_true[:, :, 1:4])
    time_l_loss = F.mse_loss(Y_hat[:, :, 4:7], Y_true[:, :, 4:7])
    dir_loss = F.binary_cross_entropy_with_logits(Y_hat[:, :, 7], Y_true[:, :, 7])
    return pos_loss + time_t_loss + time_l_loss + 0.5 * dir_loss


# ============================================================
# 3) Representational Drift Measurements
# ============================================================

def compute_within_trial_drift(H_np, Y_np, n_bins=10):
    """
    Measures the population vector correlation of corresponding spatial positions
    between the first 20% and the last 20% of the trial.
    
    H_np: (B, T, nH) hidden states
    Y_np: (B, T, n_out) targets (Y_np[:, :, 0] is normalized position)
    
    Returns:
      C_FL: average correlation coefficient (value in [-1, 1], lower means more drift)
    """
    B, T, nH = H_np.shape
    cutoff = int(0.2 * T)  # 20% of T (samples 0-250 and 1000-1250)

    c_fl_list = []

    for b in range(B):
        h_first = H_np[b, :cutoff, :]   # (250, nH)
        pos_first = Y_np[b, :cutoff, 0] # (250,)
        
        h_last = H_np[b, -cutoff:, :]   # (250, nH)
        pos_last = Y_np[b, -cutoff:, 0] # (250,)

        # Divide space [0, 1] into bins
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        
        pv_first = []
        pv_last = []

        for k in range(n_bins):
            # Mask of when the animal is in bin k
            mask_first = (pos_first >= bin_edges[k]) & (pos_first < bin_edges[k+1])
            mask_last = (pos_last >= bin_edges[k]) & (pos_last < bin_edges[k+1])

            if np.sum(mask_first) > 0 and np.sum(mask_last) > 0:
                pv_first.append(h_first[mask_first].mean(axis=0))
                pv_last.append(h_last[mask_last].mean(axis=0))

        if len(pv_first) > 2:
            pv_first = np.array(pv_first)  # (K, nH)
            pv_last = np.array(pv_last)    # (K, nH)
            
            # Compute Pearson correlation coefficient for corresponding spatial bins
            # Correlate the population vector of bin k in first 20% vs. last 20%
            corrs = []
            for k in range(len(pv_first)):
                v1 = pv_first[k]
                v2 = pv_last[k]
                sd1, sd2 = np.std(v1), np.std(v2)
                if sd1 > 1e-6 and sd2 > 1e-6:
                    r = np.corrcoef(v1, v2)[0, 1]
                    if np.isfinite(r):
                        corrs.append(r)

            if corrs:
                c_fl_list.append(np.mean(corrs))

    return float(np.mean(c_fl_list)) if c_fl_list else 1.0


def compute_crosstrial_pvc_decay(H_np, Y_np, n_bins=10):
    """
    Computes Across-Trial Population Vector Correlation (PVC) decay rate.
    Creates spatial PVs (size nH) for B trials across n_bins.
    Calculates B x B trial correlation matrix, averages across bins,
    plots correlation vs. trial lag |i - j|, and fits a linear slope to get the drift rate.
    
    Returns:
      drift_rate: absolute value of the linear slope (larger means faster drift)
    """
    B, T, nH = H_np.shape
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # PV matrix: (B, n_bins, nH)
    PV = np.zeros((B, n_bins, nH))
    
    for b in range(B):
        states = H_np[b]
        pos = Y_np[b, :, 0]
        for k in range(n_bins):
            mask = (pos >= bin_edges[k]) & (pos < bin_edges[k+1])
            if np.sum(mask) > 0:
                PV[b, k] = states[mask].mean(axis=0)
            else:
                PV[b, k] = np.zeros(nH)

    # Compute correlation matrix across trials: (n_bins, B, B)
    corr_grid = np.zeros((n_bins, B, B))
    for k in range(n_bins):
        vectors = PV[:, k, :]  # (B, nH)
        # Handle zero vectors safely
        active_trials = np.sum(np.abs(vectors), axis=1) > 1e-6
        if np.sum(active_trials) > 4:
            c = np.corrcoef(vectors)
            c[~np.isfinite(c)] = 0.0
            corr_grid[k] = c

    # Average trial-by-trial correlation matrix across spatial bins
    R_ij = np.mean(corr_grid, axis=0)  # (B, B)

    # Average correlation by trial lag
    lags = np.arange(1, B // 2)
    pvc_by_lag = []
    
    for lag in lags:
        vals = []
        for i in range(B - lag):
            vals.append(R_ij[i, i + lag])
        pvc_by_lag.append(np.mean(vals))

    pvc_by_lag = np.array(pvc_by_lag)
    
    # Fit a linear regression slope to the correlation decay: PVC(lag) = slope * lag + intercept
    # Only fit if lag correlation actually exists
    if len(pvc_by_lag) > 3:
        slope, _ = np.polyfit(lags, pvc_by_lag, 1)
        drift_rate = -float(slope)  # convert to positive drift rate (how fast it decays)
    else:
        drift_rate = 0.0

    return max(0.0, drift_rate)


def compute_decoder_drift_rate(H_np, Y_np):
    """
    Trains a spatial decoder (Ridge Regression) on Trial 0 and tests it on Trial k (k = 1...B).
    Computes the decay rate of R^2 accuracy as a function of trial lag.
    
    Returns:
      dec_drift_rate: positive rate of decoding accuracy decay (larger means faster drift)
    """
    B, T, nH = H_np.shape

    # Training decoder on Trial 0
    X_train = H_np[0]      # (T, nH)
    y_train = Y_np[0, :, 0] # (T,) position

    decoder = Ridge(alpha=1.0)
    decoder.fit(X_train, y_train)

    r2_scores = []
    lags = np.arange(1, B)

    for k in lags:
        X_test = H_np[k]
        y_test = Y_np[k, :, 0]
        
        y_pred = decoder.predict(X_test)
        
        # Pearson correlation or R^2 score
        # Using Pearson correlation r as standard mapping of fidelity
        sd_pred, sd_test = np.std(y_pred), np.std(y_test)
        if sd_pred > 1e-6 and sd_test > 1e-6:
            r = np.corrcoef(y_pred, y_test)[0, 1]
            r2_scores.append(r**2 if np.isfinite(r) else 0.0)
        else:
            r2_scores.append(0.0)

    r2_scores = np.array(r2_scores)

    if len(r2_scores) > 3:
        slope, _ = np.polyfit(lags, r2_scores, 1)
        dec_drift_rate = -float(slope)
    else:
        dec_drift_rate = 0.0

    return max(0.0, dec_drift_rate)


# ============================================================
# 4) Main Sweep Loop for Drift Analysis
# ============================================================

def plot_drift_heatmap(M: np.ndarray, xvals: List[float], yvals: List[float],
                       title: str, out_no_ext: str, xlabel: str, ylabel: str, cmap="plasma", vmin=None, vmax=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Drift Metric Value")
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


def run_drift_sweep_for_mode(cfg: Config, mode: str, freqs: List[float], couplings: List[float], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    landmark_centers = make_landmarks(cfg.n_landmarks, cfg.track_len)

    n_g = len(couplings)
    n_f = len(freqs)

    # Drift Grids
    M_cfl = np.zeros((n_g, n_f))   # within-trial first-last correlation
    M_pvc = np.zeros((n_g, n_f))   # across-trial PVC decay rate
    M_dec = np.zeros((n_g, n_f))   # decoder accuracy decay rate

    csv_path = os.path.join(out_dir, "results_drift.csv")
    with open(csv_path, "w") as fp:
        fp.write("mode,g_ext,frequency_hz,c_fl,r_drift_pvc,r_drift_decoder\n")

    for g_idx, g_val in enumerate(couplings):
        for f_idx, f_hz in enumerate(freqs):
            print(f"\n[Mode: {mode}] Run: g_ext={g_val:.2f}, f={f_hz:.3f} Hz")
            cfg.g_ext = g_val
            
            # Train model
            set_seed(123)
            model = LeakyRNN(cfg).to(cfg.device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2_weight)

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

            # Evaluation
            model.eval()
            set_seed(1000)
            with torch.no_grad():
                # Generate 32 trials for highly robust correlation and decoder statistics
                X_val, Y_val = make_batch(cfg, landmark_centers, cfg.eval_batch_size, f=f_hz, mode=mode)
                X_val, Y_val = X_val.to(cfg.device), Y_val.to(cfg.device)
                _, H_val = model(X_val)

            H_np = H_val.detach().cpu().numpy()
            Y_np = Y_val.detach().cpu().numpy()

            # Compute representational drift measurements
            c_fl = compute_within_trial_drift(H_np, Y_np)
            r_pvc = compute_crosstrial_pvc_decay(H_np, Y_np)
            r_dec = compute_decoder_drift_rate(H_np, Y_np)

            M_cfl[g_idx, f_idx] = c_fl
            M_pvc[g_idx, f_idx] = r_pvc
            M_dec[g_idx, f_idx] = r_dec

            print(f"  Drift -> C_FL (Within-Trial): {c_fl:.4f} | R_PVC (Across-Trial): {r_pvc:.5f} | R_DEC (Decoder): {r_dec:.5f}")

            with open(csv_path, "a") as fp:
                fp.write(f"{mode},{g_val},{f_hz},{c_fl},{r_pvc},{r_dec}\n")

    # Save grids
    np.save(os.path.join(out_dir, "grid_c_fl.npy"), M_cfl)
    np.save(os.path.join(out_dir, "grid_r_pvc.npy"), M_pvc)
    np.save(os.path.join(out_dir, "grid_r_dec.npy"), M_dec)

    # Plot Heatmaps
    plot_drift_heatmap(M_cfl, freqs, couplings, f"Within-Trial PV Correlation C_FL ({mode})", 
                       os.path.join(out_dir, "heatmap_within_trial_c_fl"), "frequency (Hz)", "g_ext", cmap="magma", vmin=0.0, vmax=1.0)
    
    plot_drift_heatmap(M_pvc, freqs, couplings, f"Across-Trial PVC Decay Rate R_drift ({mode})", 
                       os.path.join(out_dir, "heatmap_crosstrial_pvc_drift"), "frequency (Hz)", "g_ext", cmap="plasma")
    
    plot_drift_heatmap(M_dec, freqs, couplings, f"Decoder Generalization Decay Rate R_dec ({mode})", 
                       os.path.join(out_dir, "heatmap_decoder_drift"), "frequency (Hz)", "g_ext", cmap="plasma")

    print(f"\n[Finished Mode: {mode}] All drift analysis outputs saved in: {out_dir}")


# ============================================================
# 5) Main
# ============================================================

def main():
    FREQ_LIST = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
    G_EXT_LIST = [0.0, 0.5, 1.0, 2.0, 4.0]

    cfg = Config()
    cfg.n_in = 1 + cfg.n_landmarks + 4 + 1
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ROOT_OUT_BASE = "./sim_freq_x_gext_drift"
    
    print("=" * 80)
    print("Unified 2D Sweep: Representational Drift Analysis")
    print("=" * 80)
    print(f"Device: {cfg.device}")
    print(f"Fs: {cfg.Fs} Hz | T: {cfg.T} steps")
    print(f"Evaluating {cfg.eval_batch_size} trials per cell")

    # Mode 1: Direct Infraslow (Slow Component)
    out_dir_component = os.path.join(ROOT_OUT_BASE, f"sweep_slow_component_{timestamp}")
    run_drift_sweep_for_mode(cfg, "slow_component", FREQ_LIST, G_EXT_LIST, out_dir_component)

    # Mode 2: Oscillator Interference (Slow Beat)
    out_dir_beat = os.path.join(ROOT_OUT_BASE, f"sweep_slow_beat_{timestamp}")
    run_drift_sweep_for_mode(cfg, "slow_beat", FREQ_LIST, G_EXT_LIST, out_dir_beat)

    print("\nUNIFIED DRIFT ANALYSIS COMPLETED SUCCESSFULY!")
    print(f"Direct Infraslow outputs: {os.path.abspath(out_dir_component)}")
    print(f"Oscillator Interference outputs: {os.path.abspath(out_dir_beat)}")


if __name__ == "__main__":
    main()
