#!/usr/bin/env python3
"""
Frequency sweep experiment (single leaky RNN + oscillatory input) with:
- Fixed trial length: T=1250 samples (~62.5s at Fs=20 Hz)
- Exactly n_go_back=15 go+back laps per trial
- Variable lap durations (speed variability) while keeping constant sampling rate
- External oscillatory input e(t) injected as an INPUT CHANNEL (in X_in)
- NEW: reward/trial pulse input channel (e.g., reward at lap end)
For each oscillation frequency:
  - train one model (300 epochs)
  - run only ONE evaluation (one eval seed)
  - SAVE:
      * ONE overlay trajectory figure (few trials from eval batch)
      * ONE-trial PCA2D/PCA3D plots (ONLY 1250 points, like your screenshot)
      * SI(position) and SI(time) computed on ALL eval batch points (B*T)
      * MEB radius computed on trial centroids (original hidden space)
  - append results to CSV files
Requires:
- structure_index.py in the same folder, exposing compute_structure_index(states, labels, ...)
Outputs:
  out_root/
    config_base.json
    results_per_eval.csv         (one row per frequency; since NUM_EVAL=1)
    results_summary.csv          (same as per_eval for now)
    freq_0p01Hz/
      model.pt
      config.json
      loss_train_vs_eval.png/svg
      eval_seed_1000/
        eval_traj_overlay_trials.png/svg
        pca_one_trial/
          PCA2D_pos.png/svg
          PCA3D_pos.png/svg
          PCA2D_time.png/svg
          PCA3D_time.png/svg
          PCA2D_lap.png/svg
          PCA3D_lap.png/svg
          PCA2D_ext.png/svg
          PCA3D_ext.png/svg
          PCA2D_pulse.png/svg
          PCA3D_pulse.png/svg
        metrics_summary.txt
"""
import os
import math
import time
import json
import random
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from structure_index import compute_structure_index  # your uploaded code
# =========================
# 0) Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
# =========================
# 1) Tunable parameters
# =========================
@dataclass
class Config:
    # ---- Sampling / trial ----
    Fs: float = 20.0
    dt: float = 1.0 / 20.0
    T: int = 1250                # exactly 1250 points for train & eval
    track_len: float = 1.0
    # Exactly how many go+back laps per trial
    n_go_back: int = 15
    # Lap duration variability (0 -> deterministic). Lognormal sigma.
    lap_jitter: float = 0.25
    # ---- Inputs ----
    n_landmarks: int = 8
    landmark_width: float = 0.08
    sensory_noise: float = 0.05
    vel_noise: float = 0.02
    # External oscillation input channel (in X_in)
    use_external: bool = True
    ext_mode: str = "osc"        # "osc", "piecewise", "constant"
    ext_amp: float = 1.0
    ext_noise: float = 0.05
    # NEW: reward/trial pulse input channel (in X_in)
    use_pulse: bool = True
    pulse_mode: str = "lap_end"  # "lap_end" or "trial_end"
    pulse_amp: float = 1.0
    pulse_width_s: float = 0.25  # duration of pulse window in seconds
    # ---- Learning target flag ----
    learn_time: bool = True  # True: pos+time+phase+dir, False: pos+dir only
    # ---- Model ----
    n_in: int = 0
    n_hidden: int = 128
    n_out: int = 5  # [pos, tau, sin, cos, dir_logit]
    # Dynamics
    tau_rnn: float = 0.1
    w_rec_scale: float = 0.9
    w_in_scale: float = 0.5
    # Regularization
    l2_weight: float = 1e-5
    rate_l2: float = 1e-4
    # Training
    device: str = "cpu"
    batch_size: int = 64
    eval_batch_size: int = 64
    n_epochs: int = 400          # per your request
    lr: float = 1e-3
    grad_clip: float = 1.0
    # ---- Structure Index params ----
    si_n_bins: int = 10
    si_n_neighbors: int = 50
    si_num_shuffles: int = 0
    si_distance_metric: str = "euclidean"
    si_verbose: bool = False
    si_discrete_label: bool = False  # continuous pos/time
    # ---- Colormaps ----
    cmap_pos: str = "magma"
    cmap_tau: str = "YlGn_r"
    cmap_lap: str = "viridis"
    cmap_drive: str = "coolwarm"  # red<->blue
    cmap_pulse: str = "coolwarm"
# =========================
# 2) Data simulation
# =========================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
def save_fig(fig, out_path_no_ext):
    fig.savefig(out_path_no_ext + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_path_no_ext + ".svg", bbox_inches="tight")
    plt.close(fig)
def make_landmarks(n_landmarks: int, track_len: float):
    return np.linspace(0.05, track_len - 0.05, n_landmarks)
def sensory_features_from_position(x, centers, width):
    x = np.asarray(x)[..., None]
    centers = centers[None, :]
    return np.exp(-0.5 * ((x - centers) / width) ** 2)
def generate_external_input(ext_mode, T, dt, amp, freq_hz, noise_std, rng: np.random.Generator):
    t = np.arange(T) * dt
    if ext_mode == "osc":
        e = amp * np.sin(2.0 * np.pi * freq_hz * t)
    elif ext_mode == "piecewise":
        e = np.zeros(T, dtype=np.float32)
        seg_len = max(10, T // 6)
        val = 0.0
        for i in range(T):
            if i % seg_len == 0:
                val = amp * rng.uniform(-1, 1)
            e[i] = val
    elif ext_mode == "constant":
        e = amp * np.ones(T, dtype=np.float32)
    else:
        raise ValueError(f"Unknown ext_mode: {ext_mode}")
    e = e + noise_std * rng.standard_normal(T)
    return e.astype(np.float32)
def variable_duration_triangle(cfg: Config, rng: np.random.Generator):
    """
    Build position x(t) with EXACTLY cfg.n_go_back go+backs,
    but each go+back uses a different number of samples.
    Sum of samples == cfg.T.
    Returns:
      x:     (T,) position in [0, track_len]
      dir_t: (T,) direction (+1 rising, -1 falling)
      lap_id:(T,) lap index 0..n_go_back-1
      L:     (n_go_back,) integer samples per lap
    """
    T = cfg.T
    n = cfg.n_go_back
    # Relative durations (positive, mean ~1)
    w = rng.lognormal(mean=0.0, sigma=cfg.lap_jitter, size=n).astype(np.float64)
    # Convert to integer lengths summing exactly to T
    L = np.floor(T * (w / w.sum())).astype(int)
    diff = T - L.sum()
    if diff > 0:
        L[:diff] += 1
    elif diff < 0:
        idx = np.argsort(-L)
        for k in range(-diff):
            L[idx[k % n]] -= 1
    # Ensure not too short
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
    assert np.all(L >= 4)
    x_list, dir_list, lap_list = [], [], []
    for lap_i, Li in enumerate(L):
        frac = np.linspace(0.0, 1.0, Li, endpoint=False).astype(np.float32)
        p = 1.0 - np.abs(2.0 * frac - 1.0)  # triangle [0,1]
        x_seg = (p * cfg.track_len).astype(np.float32)
        dir_seg = np.where(frac < 0.5, 1.0, -1.0).astype(np.float32)
        lap_seg = np.ones(Li, dtype=np.int32) * lap_i
        x_list.append(x_seg)
        dir_list.append(dir_seg)
        lap_list.append(lap_seg)
    x = np.concatenate(x_list, axis=0)
    dir_t = np.concatenate(dir_list, axis=0)
    lap_id = np.concatenate(lap_list, axis=0)
    return x, dir_t, lap_id, L
def build_reward_pulse(cfg: Config, lap_id: np.ndarray, L: np.ndarray):
    """
    Make a pulse signal (T,) to represent "reward".
    Default: pulse at the END of each lap (last pulse_width_s seconds of that lap).
    Alternative: pulse only at end of whole trial.
    Returns:
      p: (T,) float32
    """
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
    # pulse at end of each lap segment
    # We know lap boundaries from L
    start = 0
    for Li in L:
        end = start + int(Li)
        p[max(start, end - width):end] = cfg.pulse_amp
        start = end
    return p
def generate_trial(cfg: Config, landmark_centers, freq_hz: float, rng: np.random.Generator):
    """
    Returns:
      X_in: (T, n_in) = [speed, landmarks..., ext, pulse]
      Y_out:(T, 5)    = [pos, tau, sin, cos, dir01]
      aux: dict with keys: e, pulse, lap_id
    """
    T = cfg.T
    dt = cfg.dt
    # Variable-duration laps
    x, dir_t, lap_id, L = variable_duration_triangle(cfg, rng=rng)
    # Speed from derivative (absolute)
    dx = np.diff(x, prepend=x[0])
    v_true = np.abs(dx / dt).astype(np.float32)
    v_in = (v_true - np.mean(v_true)) / (np.std(v_true) + 1e-6)
    v_in = v_in + cfg.vel_noise * rng.standard_normal(T).astype(np.float32)
    v_in = v_in[:, None]
    # Landmarks
    s = sensory_features_from_position(x / cfg.track_len, landmark_centers, cfg.landmark_width).astype(np.float32)
    s = s + cfg.sensory_noise * rng.standard_normal(s.shape).astype(np.float32)
    # External
    if cfg.use_external:
        e = generate_external_input(cfg.ext_mode, T, dt, cfg.ext_amp, freq_hz, cfg.ext_noise, rng=rng)
    else:
        e = np.zeros(T, dtype=np.float32)
    # Reward pulse
    pulse = build_reward_pulse(cfg, lap_id=lap_id, L=L)
    # Targets
    tau = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
    phase = 2.0 * np.pi * tau
    sinp = np.sin(phase).astype(np.float32)
    cosp = np.cos(phase).astype(np.float32)
    y_pos = (x / cfg.track_len).astype(np.float32)[:, None]
    dir01 = ((dir_t + 1.0) / 2.0).astype(np.float32)[:, None]
    Y_out = np.concatenate([y_pos, tau, sinp, cosp, dir01], axis=1).astype(np.float32)
    # Inputs: speed + landmarks + ext + pulse
    X_in = np.concatenate([v_in, s, e[:, None], pulse[:, None]], axis=1).astype(np.float32)
    aux = {
        "e": e[:, None].astype(np.float32),
        "pulse": pulse[:, None].astype(np.float32),
        "lap_id": lap_id.astype(np.int32),
    }
    return X_in, Y_out, aux
def make_batch(cfg: Config, landmark_centers, batch_size: int, freq_hz: float):
    X_list, Y_list, e_list, p_list, lap_list = [], [], [], [], []
    seeds = np.random.randint(0, 2**31 - 1, size=batch_size, dtype=np.int64)
    for i in range(batch_size):
        rng = np.random.default_rng(int(seeds[i]))
        X_in, Y_out, aux = generate_trial(cfg, landmark_centers, freq_hz, rng=rng)
        X_list.append(X_in)
        Y_list.append(Y_out)
        e_list.append(aux["e"])
        p_list.append(aux["pulse"])
        lap_list.append(aux["lap_id"])
    X = torch.tensor(np.stack(X_list, axis=0), dtype=torch.float32)   # (B,T,n_in)
    Y = torch.tensor(np.stack(Y_list, axis=0), dtype=torch.float32)   # (B,T,5)
    e = np.stack(e_list, axis=0).astype(np.float32)                   # (B,T,1)
    pulse = np.stack(p_list, axis=0).astype(np.float32)               # (B,T,1)
    lap_id = np.stack(lap_list, axis=0).astype(np.int32)              # (B,T)
    return X, Y, e, pulse, lap_id
# =========================
# 3) Single-population leaky RNN
# =========================
class LeakyRNN(nn.Module):
    """
    "Leaky" means hidden state is a leaky integrator:
      h <- (1-alpha)*h + alpha*tanh(...)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n_in = cfg.n_in
        nH = cfg.n_hidden
        self.W_in = nn.Parameter(torch.randn(n_in, nH) * cfg.w_in_scale / math.sqrt(n_in))
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
            x_t = X[:, t, :]
            pre = x_t @ self.W_in + h @ self.W_rec + self.b
            h_tilde = torch.tanh(pre)
            h = (1.0 - self.alpha) * h + self.alpha * h_tilde
            y_t = self.W_out(h)
            Y_hat.append(y_t)
            H.append(h)
        return torch.stack(Y_hat, dim=1), torch.stack(H, dim=1)
# =========================
# 4) Loss + metrics + plotting helpers
# =========================
def loss_fn(Y_hat, Y, learn_time: bool):
    pos_mse = F.mse_loss(Y_hat[:, :, 0], Y[:, :, 0])
    dir_bce = F.binary_cross_entropy_with_logits(Y_hat[:, :, 4], Y[:, :, 4])
    if learn_time:
        time_mse = F.mse_loss(Y_hat[:, :, 1:4], Y[:, :, 1:4])  # tau,sin,cos
        return pos_mse + time_mse + 0.5 * dir_bce
    return pos_mse + 0.5 * dir_bce
def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))
def trial_centroids(H_btH: np.ndarray):
    # (B,T,nH) -> (B,nH)
    return H_btH.mean(axis=1)
def meb_approx(points: np.ndarray, n_iter: int = 300, tol: float = 1e-7, seed: int = 0):
    """
    Approximate Minimum Enclosing Ball (MEB) via farthest-point updates.
    """
    rng = np.random.default_rng(seed)
    c = points[rng.integers(0, len(points))].copy()
    prev_r = None
    for it in range(n_iter):
        d2 = np.sum((points - c) ** 2, axis=1)
        idx = int(np.argmax(d2))
        p_far = points[idx]
        r = math.sqrt(float(d2[idx]))
        if prev_r is not None and abs(r - prev_r) < tol:
            break
        prev_r = r
        eta = 1.0 / (it + 2.0)
        c = (1.0 - eta) * c + eta * p_far
    d2 = np.sum((points - c) ** 2, axis=1)
    radius = math.sqrt(float(np.max(d2)))
    return c, radius
def plot_eval_overlay_some_trials(cfg, Ytrue, Ypred, out_path_no_ext, f_hz, eval_seed, n_plot=1):
    """
    ONE figure that overlays a subset of trials from the eval batch.
    """
    B, T, _ = Ytrue.shape
    n_plot = min(n_plot, B)
    t_axis = np.arange(T) * cfg.dt
    fig = plt.figure(figsize=(12, 9))
    ax1 = plt.subplot(3, 1, 1)
    for i in range(n_plot):
        ax1.plot(t_axis, Ytrue[i, :, 0], alpha=0.6, linewidth=1)
        ax1.plot(t_axis, Ypred[i, :, 0], alpha=0.6, linewidth=1, linestyle="--")
    ax1.set_ylabel("position (0..1)")
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title(f"Eval overlay (first {n_plot} trials) | f={f_hz} Hz | seed={eval_seed}")
    ax2 = plt.subplot(3, 1, 2)
    for i in range(n_plot):
        ax2.plot(t_axis, Ytrue[i, :, 1], alpha=0.6, linewidth=1)
        ax2.plot(t_axis, Ypred[i, :, 1], alpha=0.6, linewidth=1, linestyle="--")
    ax2.set_ylabel("tau (0..1)")
    ax2.set_ylim(-0.1, 1.1)
    ax3 = plt.subplot(3, 1, 3)
    for i in range(n_plot):
        dir_true = Ytrue[i, :, 4]
        dir_prob = sigmoid_np(Ypred[i, :, 4])
        ax3.plot(t_axis, dir_true, alpha=0.6, linewidth=1, drawstyle="steps-post")
        ax3.plot(t_axis, dir_prob, alpha=0.6, linewidth=1, linestyle="--")
    ax3.set_ylabel("direction")
    ax3.set_xlabel("time (s)")
    ax3.set_ylim(-0.1, 1.1)
    save_fig(fig, out_path_no_ext)
def pca_plot_one_trial(states_TxH, color_T, title, out_path_no_ext, color_label, cmap_name, n_components):
    """
    PCA using ONLY one trial: exactly T=1250 points (like your screenshot).
    """
    assert states_TxH.ndim == 2
    assert color_T.ndim == 1
    assert states_TxH.shape[0] == color_T.shape[0]
    pca = PCA(n_components=n_components, random_state=0)
    Z = pca.fit_transform(states_TxH)
    evr = pca.explained_variance_ratio_
    if n_components == 2:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=color_T, s=18, cmap=cmap_name, alpha=0.95, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(color_label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{title}\nEVR: [{evr[0]:.3f}, {evr[1]:.3f}]")
        save_fig(fig, out_path_no_ext)
    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=color_T, s=18, cmap=cmap_name, alpha=0.95, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.1)
        cb.set_label(color_label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"{title}\nEVR: [{evr[0]:.3f}, {evr[1]:.3f}, {evr[2]:.3f}]")
        save_fig(fig, out_path_no_ext)
    return evr
# =========================
# 5) Main experiment
# =========================
def main():
    # Frequencies you requested
    FREQ_LIST = [0.001,0.005,0.01,0.02,0.0250,0.3,0.035,0.04,0.045,0.05,0.075,0.1]
    # With Fs=20Hz, Nyquist = 10Hz. 15 and 20 Hz will alias.
    # Keeping them anyway because you asked.
    NUM_EVAL = 1
    BASE_TRAIN_SEED = 123
    EVAL_SEED = 1000
    cfg = Config()
    # Inputs: speed + landmarks + ext + pulse
    cfg.n_in = 1 + cfg.n_landmarks + 1 + 1
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join("./sim_freq_sweep_one_rnn_SI", f"freq_sweep_pulse_{timestamp}")
    ensure_dir(out_root)
    with open(os.path.join(out_root, "config_base.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    landmark_centers = make_landmarks(cfg.n_landmarks, cfg.track_len)
    # CSVs
    csv_eval_path = os.path.join(out_root, "results_per_eval.csv")
    csv_sum_path = os.path.join(out_root, "results_summary.csv")
    eval_header = "frequency_hz,eval_seed,si_pos,si_time,meb_radius\n"
    si_params = dict(
        n_bins=cfg.si_n_bins,
        discrete_label=cfg.si_discrete_label,
        n_neighbors=cfg.si_n_neighbors,
        num_shuffles=cfg.si_num_shuffles,
        distance_metric=cfg.si_distance_metric,
        verbose=cfg.si_verbose,
    )
    rows = []
    for f_hz in FREQ_LIST:
        f_tag = str(f_hz).replace(".", "p")
        f_dir = os.path.join(out_root, f"freq_{f_tag}Hz")
        ensure_dir(f_dir)
        with open(os.path.join(f_dir, "config.json"), "w") as fp:
            json.dump(asdict(cfg), fp, indent=2)
        # -------------------------
        # Train
        # -------------------------
        set_seed(BASE_TRAIN_SEED)
        model = LeakyRNN(cfg).to(cfg.device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2_weight)
        history_epoch = []
        history_train = []
        history_eval = []
        for epoch in range(1, cfg.n_epochs + 1):
            model.train()
            X, Y, _e, _p, _lap = make_batch(cfg, landmark_centers, cfg.batch_size, freq_hz=f_hz)
            X, Y = X.to(cfg.device), Y.to(cfg.device)
            opt.zero_grad()
            Y_hat, H = model(X)
            L_train = loss_fn(Y_hat, Y, cfg.learn_time) + cfg.rate_l2 * H.pow(2).mean()
            L_train.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            model.eval()
            with torch.no_grad():
                Xv, Yv, _ev, _pv, _lapv = make_batch(cfg, landmark_centers, cfg.eval_batch_size, freq_hz=f_hz)
                Xv, Yv = Xv.to(cfg.device), Yv.to(cfg.device)
                Yv_hat, Hv = model(Xv)
                L_eval = loss_fn(Yv_hat, Yv, cfg.learn_time) + cfg.rate_l2 * Hv.pow(2).mean()
            history_epoch.append(epoch)
            history_train.append(float(L_train.item()))
            history_eval.append(float(L_eval.item()))
            if epoch in (1, 50, 100, 150, 200, 250, cfg.n_epochs):
                print(f"[{f_hz:>5} Hz] Epoch {epoch:03d}/{cfg.n_epochs} | train={L_train.item():.6f} | eval={L_eval.item():.6f}")
        torch.save(model.state_dict(), os.path.join(f_dir, "model.pt"))
        # Loss plot
        fig = plt.figure(figsize=(7, 4))
        plt.plot(history_epoch, history_train, label="train")
        plt.plot(history_epoch, history_eval, label="eval")
        mode_str = "pos+time" if cfg.learn_time else "pos-only"
        plt.title(f"Loss ({mode_str}) | f={f_hz} Hz")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        save_fig(fig, os.path.join(f_dir, "loss_train_vs_eval"))
        # -------------------------
        # ONE evaluation
        # -------------------------
        for eval_i in range(NUM_EVAL):
            eval_seed = EVAL_SEED
            set_seed(eval_seed)
            eval_dir = os.path.join(f_dir, f"eval_seed_{eval_seed:04d}")
            ensure_dir(eval_dir)
            pca_dir = os.path.join(eval_dir, "pca_one_trial")
            ensure_dir(pca_dir)
            model.eval()
            with torch.no_grad():
                Xp, Yp, e_sig, pulse_sig, lap_id_bt = make_batch(
                    cfg, landmark_centers, cfg.eval_batch_size, freq_hz=f_hz
                )
                Xp, Yp = Xp.to(cfg.device), Yp.to(cfg.device)
                Yhat, H = model(Xp)
            Ytrue = Yp.detach().cpu().numpy()          # (B,T,5)
            Ypred = Yhat.detach().cpu().numpy()        # (B,T,5)
            H_np = H.detach().cpu().numpy()            # (B,T,nH)
            e_np = np.asarray(e_sig)                   # (B,T,1)
            pulse_np = np.asarray(pulse_sig)           # (B,T,1)
            lap_id_bt = np.asarray(lap_id_bt)          # (B,T)
            B, T, _ = Ytrue.shape
            nH = H_np.shape[2]
            N = B * T
            # MEB on trial centroids
            C = trial_centroids(H_np)  # (B,nH)
            _, meb_radius = meb_approx(C, seed=eval_seed)
            # One overlay plot (few trials)
            plot_eval_overlay_some_trials(
                cfg, Ytrue, Ypred,
                out_path_no_ext=os.path.join(eval_dir, "eval_traj_overlay_trials"),
                f_hz=f_hz, eval_seed=eval_seed, n_plot=8
            )
            # ---- PCA plots: ONE trial only (1250 points) ----
            trial_idx = 0
            states_TxH = H_np[trial_idx, :, :]            # (T,nH)
            pos_T = Ytrue[trial_idx, :, 0]                # (T,)
            tau_T = Ytrue[trial_idx, :, 1]                # (T,)
            lap_T = lap_id_bt[trial_idx, :].astype(float) # (T,)
            ext_T = e_np[trial_idx, :, 0]                 # (T,)
            pulse_T = pulse_np[trial_idx, :, 0]           # (T,)
            pca_plot_one_trial(
                states_TxH, tau_T,
                title=f"PCA3D | time | ONE trial | f={f_hz}Hz | seed={eval_seed}",
                out_path_no_ext=os.path.join(pca_dir, "PCA3D_time"),
                color_label="time (tau)", cmap_name=cfg.cmap_tau, n_components=3
            )
            pca_plot_one_trial(
                states_TxH, tau_T,
                title=f"PCA2D | time | ONE trial | f={f_hz}Hz | seed={eval_seed}",
                out_path_no_ext=os.path.join(pca_dir, "PCA2D_time"),
                color_label="time (tau)", cmap_name=cfg.cmap_tau, n_components=2
            )
            pca_plot_one_trial(
                states_TxH, pos_T,
                title=f"PCA3D | position | ONE trial | f={f_hz}Hz | seed={eval_seed}",
                out_path_no_ext=os.path.join(pca_dir, "PCA3D_pos"),
                color_label="position", cmap_name=cfg.cmap_pos, n_components=3
            )
            pca_plot_one_trial(
                states_TxH, pos_T,
                title=f"PCA2D | position | ONE trial | f={f_hz}Hz | seed={eval_seed}",
                out_path_no_ext=os.path.join(pca_dir, "PCA2D_pos"),
                color_label="position", cmap_name=cfg.cmap_pos, n_components=2
            )
            pca_plot_one_trial(
                states_TxH, lap_T,
                title=f"PCA3D | lap id | ONE trial | f={f_hz}Hz | seed={eval_seed}",
                out_path_no_ext=os.path.join(pca_dir, "PCA3D_lap"),
                color_label="lap id", cmap_name=cfg.cmap_lap, n_components=3
            )
            pca_plot_one_trial(
                states_TxH, lap_T,
                title=f"PCA2D | lap id | ONE trial | f={f_hz}Hz | seed={eval_seed}",
                out_path_no_ext=os.path.join(pca_dir, "PCA2D_lap"),
                color_label="lap id", cmap_name=cfg.cmap_lap, n_components=2
            )
            pca_plot_one_trial(
                states_TxH, ext_T,
                title=f"PCA3D | ext e(t) | ONE trial | f={f_hz}Hz | seed={eval_seed}",
                out_path_no_ext=os.path.join(pca_dir, "PCA3D_ext"),
                color_label="e(t)", cmap_name=cfg.cmap_drive, n_components=3
            )
            pca_plot_one_trial(
                states_TxH, ext_T,
                title=f"PCA2D | ext e(t) | ONE trial | f={f_hz}Hz | seed={eval_seed}",
                out_path_no_ext=os.path.join(pca_dir, "PCA2D_ext"),
                color_label="e(t)", cmap_name=cfg.cmap_drive, n_components=2
            )
            pca_plot_one_trial(
                states_TxH, pulse_T,
                title=f"PCA3D | reward pulse | ONE trial | f={f_hz}Hz | seed={eval_seed}",
                out_path_no_ext=os.path.join(pca_dir, "PCA3D_pulse"),
                color_label="pulse", cmap_name=cfg.cmap_pulse, n_components=3
            )
            pca_plot_one_trial(
                states_TxH, pulse_T,
                title=f"PCA2D | reward pulse | ONE trial | f={f_hz}Hz | seed={eval_seed}",
                out_path_no_ext=os.path.join(pca_dir, "PCA2D_pulse"),
                color_label="pulse", cmap_name=cfg.cmap_pulse, n_components=2
            )
            # ---- SI on ALL eval points (B*T) ----
            states_all = H_np.reshape(N, nH)
            pos_all = Ytrue[:, :, 0].reshape(N)
            tau_all = Ytrue[:, :, 1].reshape(N)
            si_pos, _, _, _ = compute_structure_index(states_all, pos_all, **si_params)
            si_time, _, _, _ = compute_structure_index(states_all, tau_all, **si_params)
            with open(os.path.join(eval_dir, "metrics_summary.txt"), "w") as fp:
                fp.write(f"frequency_hz: {f_hz}\n")
                fp.write(f"eval_seed: {eval_seed}\n")
                fp.write(f"SI_position: {si_pos}\n")
                fp.write(f"SI_time: {si_time}\n")
                fp.write(f"MEB_radius_centroids: {meb_radius}\n\n")
                fp.write("SI params:\n")
                fp.write(json.dumps(si_params, indent=2))
                fp.write("\n")
            rows.append(f"{f_hz},{eval_seed},{si_pos},{si_time},{meb_radius}\n")
            # write per-eval CSV incrementally
            with open(csv_eval_path, "w") as fp:
                fp.write(eval_header)
                for r in rows:
                    fp.write(r)
    # Summary (since NUM_EVAL=1, this matches per_eval)
    with open(csv_sum_path, "w") as fp:
        fp.write("frequency_hz,si_pos_mean,si_pos_std,si_time_mean,si_time_std,meb_mean,meb_std\n")
        for line in rows:
            f_hz, eval_seed, si_pos, si_time, meb = line.strip().split(",")
            fp.write(f"{f_hz},{si_pos},0.0,{si_time},0.0,{meb},0.0\n")
    print("\nDONE")
    print(f"Saved experiment to: {os.path.abspath(out_root)}")
    print(f"Per-eval CSV: {os.path.abspath(csv_eval_path)}")
    print(f"Summary  CSV: {os.path.abspath(csv_sum_path)}")
if __name__ == "__main__":
    main()
