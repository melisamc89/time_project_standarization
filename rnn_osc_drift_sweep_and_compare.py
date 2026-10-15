#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rnn_osc_drift_sweep_and_compare.py

This script performs the 2D parameter sweeps for both conditions:
  1) slow_component (Direct Infraslow Input)
  2) slow_beat (Oscillator Interference / Beating)

It then computes:
  - Within-Trial PV Correlation (C_FL)
  - Across-Trial PVC Decay Rate (R_drift)
  - Decoder Generalization Decay Rate (R_dec)

Finally, it generates a comparative figure for EACH metric, containing two subplots 
(Left: slow_component, Right: slow_beat) sharing the EXACT SAME colorbar scale (vmin, vmax)
for seamless visual comparison.

Outputs are saved in:
  ./sim_freq_x_gext_drift/
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
    eval_batch_size: int = 32
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
    B, T, nH = H_np.shape
    cutoff = int(0.2 * T)

    c_fl_list = []
    for b in range(B):
        h_first = H_np[b, :cutoff, :]
        pos_first = Y_np[b, :cutoff, 0]
        h_last = H_np[b, -cutoff:, :]
        pos_last = Y_np[b, -cutoff:, 0]

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        pv_first = []
        pv_last = []

        for k in range(n_bins):
            mask_first = (pos_first >= bin_edges[k]) & (pos_first < bin_edges[k+1])
            mask_last = (pos_last >= bin_edges[k]) & (pos_last < bin_edges[k+1])

            if np.sum(mask_first) > 0 and np.sum(mask_last) > 0:
                pv_first.append(h_first[mask_first].mean(axis=0))
                pv_last.append(h_last[mask_last].mean(axis=0))

        if len(pv_first) > 2:
            pv_first = np.array(pv_first)
            pv_last = np.array(pv_last)
            
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
    B, T, nH = H_np.shape
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    PV = np.zeros((B, n_bins, nH))
    for b in range(B):
        states = H_np[b]
        pos = Y_np[b, :, 0]
        for k in range(n_bins):
            mask = (pos >= bin_edges[k]) & (pos < bin_edges[k+1])
            if np.sum(mask) > 0:
                PV[b, k] = states[mask].mean(axis=0)

    corr_grid = np.zeros((n_bins, B, B))
    for k in range(n_bins):
        vectors = PV[:, k, :]
        active_trials = np.sum(np.abs(vectors), axis=1) > 1e-6
        if np.sum(active_trials) > 4:
            c = np.corrcoef(vectors)
            c[~np.isfinite(c)] = 0.0
            corr_grid[k] = c

    R_ij = np.mean(corr_grid, axis=0)

    lags = np.arange(1, B // 2)
    pvc_by_lag = []
    for lag in lags:
        vals = [R_ij[i, i + lag] for i in range(B - lag)]
        pvc_by_lag.append(np.mean(vals))

    pvc_by_lag = np.array(pvc_by_lag)
    if len(pvc_by_lag) > 3:
        slope, _ = np.polyfit(lags, pvc_by_lag, 1)
        drift_rate = -float(slope)
    else:
        drift_rate = 0.0

    return max(0.0, drift_rate)


def compute_decoder_drift_rate(H_np, Y_np):
    B, T, nH = H_np.shape

    X_train = H_np[0]
    y_train = Y_np[0, :, 0]

    decoder = Ridge(alpha=1.0)
    decoder.fit(X_train, y_train)

    r2_scores = []
    lags = np.arange(1, B)

    for k in lags:
        X_test = H_np[k]
        y_test = Y_np[k, :, 0]
        
        y_pred = decoder.predict(X_test)
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
# 4) Main Sweep and Side-by-Side Plotting
# ============================================================

def plot_comparative_heatmaps(M1: np.ndarray, M2: np.ndarray, xvals: List[float], yvals: List[float],
                              metric_title: str, out_no_ext: str, xlabel: str, ylabel: str, cmap="plasma"):
    """
    Plots a 2-subplot figure comparing slow_component and slow_beat
    with the exact same color range scale (vmin, vmax) and colorbar.
    """
    # Force identical visual range scale
    vmin = min(M1.min(), M2.min())
    vmax = max(M1.max(), M2.max())
    
    # Handle edge case of flat matrices
    if abs(vmax - vmin) < 1e-6:
        vmin = 0.0
        vmax = 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # 1. Left subplot: slow_component
    im1 = ax1.imshow(M1, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f"A. Direct Infraslow (slow_component)", fontsize=13, fontweight="bold")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_xticks(np.arange(len(xvals)))
    ax1.set_xticklabels([f"{f:g}" for f in xvals], rotation=45, ha="right")
    ax1.set_yticks(np.arange(len(yvals)))
    ax1.set_yticklabels([f"{g:g}" for g in yvals])

    # 2. Right subplot: slow_beat
    im2 = ax2.imshow(M2, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f"B. Oscillator Interference (slow_beat)", fontsize=13, fontweight="bold")
    ax2.set_xlabel(xlabel)
    ax2.set_xticks(np.arange(len(xvals)))
    ax2.set_xticklabels([f"{f:g}" for f in xvals], rotation=45, ha="right")

    # Add a unified colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cb = fig.colorbar(im2, cax=cbar_ax)
    cb.set_label(f"{metric_title} Value", fontsize=12, fontweight="bold")

    plt.suptitle(f"Representational Drift Comparison: {metric_title}\n(Exact Same Color Scale: [{vmin:.4f} - {vmax:.4f}])", 
                 fontsize=15, fontweight="bold", y=0.98)

    fig.savefig(out_no_ext + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_no_ext + ".svg", bbox_inches="tight")
    plt.close(fig)


def run_sweep(cfg: Config, mode: str, freqs: List[float], couplings: List[float]):
    landmark_centers = make_landmarks(cfg.n_landmarks, cfg.track_len)
    n_g = len(couplings)
    n_f = len(freqs)

    M_cfl = np.zeros((n_g, n_f))
    M_pvc = np.zeros((n_g, n_f))
    M_dec = np.zeros((n_g, n_f))

    for g_idx, g_val in enumerate(couplings):
        for f_idx, f_hz in enumerate(freqs):
            print(f"[{mode}] training: g_ext={g_val:.2f}, f={f_hz:.3f} Hz")
            cfg.g_ext = g_val
            
            set_seed(123)
            model = LeakyRNN(cfg).to(cfg.device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2_weight)

            # Fast training
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
                X_val, Y_val = make_batch(cfg, landmark_centers, cfg.eval_batch_size, f=f_hz, mode=mode)
                X_val, Y_val = X_val.to(cfg.device), Y_val.to(cfg.device)
                _, H_val = model(X_val)

            H_np = H_val.detach().cpu().numpy()
            Y_np = Y_val.detach().cpu().numpy()

            M_cfl[g_idx, f_idx] = compute_within_trial_drift(H_np, Y_np)
            M_pvc[g_idx, f_idx] = compute_crosstrial_pvc_decay(H_np, Y_np)
            M_dec[g_idx, f_idx] = compute_decoder_drift_rate(H_np, Y_np)

    return M_cfl, M_pvc, M_dec


# ============================================================
# 5) Main
# ============================================================

def main():
    # Sweep configurations
    FREQ_LIST = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
    G_EXT_LIST = [0.0, 0.5, 1.0, 2.0, 4.0]

    cfg = Config()
    cfg.n_in = 1 + cfg.n_landmarks + 4 + 1
    
    ROOT_OUT = "./sim_freq_x_gext_drift"
    os.makedirs(ROOT_OUT, exist_ok=True)
    
    print("=" * 80)
    print("Unified Drift Sweep: Comparative heatmaps (Same Color Scale)")
    print("=" * 80)

    # 1. Run sweeps for slow_component
    print("\n>>> RUNNING DIRECT INFRASLOW SWEEP (slow_component) <<<")
    CFL_comp, PVC_comp, DEC_comp = run_sweep(cfg, "slow_component", FREQ_LIST, G_EXT_LIST)

    # 2. Run sweeps for slow_beat
    print("\n>>> RUNNING OSCILLATOR INTERFERENCE SWEEP (slow_beat) <<<")
    CFL_beat, PVC_beat, DEC_beat = run_sweep(cfg, "slow_beat", FREQ_LIST, G_EXT_LIST)

    # 3. Generate the 2-subplot comparative heatmaps sharing the exact same colorbar scale
    print("\n>>> GENERATING INTEGRATED DRIFT COMPARISON PLOTS <<<")
    
    # A. Within-Trial First-Last segment correlation comparison (C_FL)
    plot_comparative_heatmaps(
        M1=CFL_comp, M2=CFL_beat, xvals=FREQ_LIST, yvals=G_EXT_LIST,
        metric_title="Within-Trial PV Correlation (C_FL)",
        out_no_ext=os.path.join(ROOT_OUT, "comparison_within_trial_c_fl"),
        xlabel="frequency (Hz)", ylabel="g_ext", cmap="magma"
    )

    # B. Across-Trial Population Vector Correlation decay rate (R_drift)
    plot_comparative_heatmaps(
        M1=PVC_comp, M2=PVC_beat, xvals=FREQ_LIST, yvals=G_EXT_LIST,
        metric_title="Across-Trial PVC Drift Rate (R_drift)",
        out_no_ext=os.path.join(ROOT_OUT, "comparison_crosstrial_pvc_drift"),
        xlabel="frequency (Hz)", ylabel="g_ext", cmap="plasma"
    )

    # C. Decoder Generalization decay rate (R_dec)
    plot_comparative_heatmaps(
        M1=DEC_comp, M2=DEC_beat, xvals=FREQ_LIST, yvals=G_EXT_LIST,
        metric_title="Decoder Accuracy Decay Rate (R_dec)",
        out_no_ext=os.path.join(ROOT_OUT, "comparison_decoder_drift"),
        xlabel="frequency (Hz)", ylabel="g_ext", cmap="plasma"
    )

    print(f"\nALL INTEGRATED COMPARISON FIGURES SUCCESSFULY GENERATED!")
    print(f"Figures saved in: {os.path.abspath(ROOT_OUT)}")


if __name__ == "__main__":
    main()
