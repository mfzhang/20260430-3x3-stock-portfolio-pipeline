"""
stage2_retrain.py — Stage 2 production retrain with Optuna-best config

Heteroscedastic dual-head NN (v2.3.12):
  - Output: (ret_mu, ret_logvar, risk_mu, risk_logvar)
  - Loss:   Gaussian NLL per head (Kendall & Gal 2017)
  - Aleatoric (NN logvar) + epistemic (ensemble variance) uncertainty
  - Y_risk log-transformed for NLL training (Andersen et al. 2003)

Differences vs optuna_search.py:
  - N_ENSEMBLE = 20 (production scale)
  - 5 folds (Fold 1 included)
  - SNDK excluded by default
  - Real-time matplotlib loss plot
  - Saves full per-ticker prediction matrix incl. risk + uncertainty
  - Computes SPY benchmark for ALL folds
  - Aggregates per-snapshot rankings

Usage:
  caffeinate -i python -u stage2_retrain.py 2>&1 | tee stage2_top1.log
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, '.')

from models import HeteroscedasticDualHeadNN, heteroscedastic_loss

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
N_ENSEMBLE_STAGE2 = 20
FOLDS_STAGE2 = [0, 1, 2, 3, 4]
EXCLUDED_TICKERS = {'SNDK'}
N_SELECT = 5
RESULTS_DIR = Path('results/stage2')
PER_SNAPSHOT_BUCKET = 'M'

# Log-transform clamp for volatility target (Andersen et al. 2003, Econometrica).
# log(max(Y_risk, LOG_EPSILON)) handles 22 zero-volatility samples in cache.
LOG_EPSILON = 1e-4

PLOT_UPDATE_EVERY = 50
PLOT_BACKEND = 'MacOSX'


# ============================================================
# REAL-TIME LOSS PLOT
# ============================================================
class LiveLossPlot:
    def __init__(self, n_models, fold_id, config_label):
        import matplotlib
        try:
            matplotlib.use(PLOT_BACKEND)
        except Exception:
            matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.ion()

        self.plt = plt
        self.n_models = n_models
        self.fold_id = fold_id
        self.config_label = config_label

        ncols = 5 if n_models >= 5 else n_models
        nrows = (n_models + ncols - 1) // ncols
        self.fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3 * nrows + 1))
        self.axes = np.array(axes).flatten() if n_models > 1 else [axes]

        for i, ax in enumerate(self.axes):
            if i < n_models:
                ax.set_title(f'NN #{i+1}', fontsize=10)
                ax.set_xlabel('Epoch', fontsize=8)
                ax.set_ylabel('NLL', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)
            else:
                ax.set_visible(False)

        self._set_super_title()
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

        self.lines_train = [None] * n_models
        self.lines_val = [None] * n_models
        self.best_markers = [None] * n_models
        self.stop_lines = [None] * n_models
        self.epochs_data = [[] for _ in range(n_models)]
        self.train_data = [[] for _ in range(n_models)]
        self.val_data = [[] for _ in range(n_models)]

    def _set_super_title(self):
        self.fig.suptitle(
            f'Fold {self.fold_id+1}/5 | {self.config_label}',
            fontsize=12, fontweight='bold'
        )

    def start_model(self, model_idx):
        ax = self.axes[model_idx]
        ax.cla()
        ax.set_title(f'NN #{model_idx+1}', fontsize=10)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('NLL', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        line_t, = ax.plot([], [], 'b-', lw=1.0, label='train', alpha=0.7)
        line_v, = ax.plot([], [], 'r--', lw=1.2, label='val')
        self.lines_train[model_idx] = line_t
        self.lines_val[model_idx] = line_v
        self.best_markers[model_idx] = None
        self.stop_lines[model_idx] = None
        self.epochs_data[model_idx] = []
        self.train_data[model_idx] = []
        self.val_data[model_idx] = []
        ax.legend(loc='upper right', fontsize=7)
        self._draw()

    def update(self, model_idx, epoch, train_loss, val_loss):
        self.epochs_data[model_idx].append(epoch)
        self.train_data[model_idx].append(train_loss)
        self.val_data[model_idx].append(val_loss)

        if epoch % PLOT_UPDATE_EVERY != 0:
            return

        line_t = self.lines_train[model_idx]
        line_v = self.lines_val[model_idx]
        if line_t is None or line_v is None:
            return

        line_t.set_data(self.epochs_data[model_idx], self.train_data[model_idx])
        line_v.set_data(self.epochs_data[model_idx], self.val_data[model_idx])

        ax = self.axes[model_idx]
        ax.relim()
        ax.autoscale_view()
        self._draw()

    def mark_best(self, model_idx, best_epoch, best_val):
        ax = self.axes[model_idx]
        if self.best_markers[model_idx] is not None:
            self.best_markers[model_idx].remove()
        marker, = ax.plot([best_epoch], [best_val], 'r*',
                          markersize=12, zorder=5, label='best')
        self.best_markers[model_idx] = marker
        self._draw()

    def mark_stop(self, model_idx, stop_epoch):
        ax = self.axes[model_idx]
        if self.stop_lines[model_idx] is not None:
            self.stop_lines[model_idx].remove()
        line = ax.axvline(stop_epoch, color='gray', ls=':', lw=1, alpha=0.7)
        self.stop_lines[model_idx] = line
        self._draw()

    def next_fold(self, fold_id, config_label):
        self.fold_id = fold_id
        self.config_label = config_label
        for i in range(self.n_models):
            self.axes[i].cla()
        self._set_super_title()
        self._draw()

    def save_snapshot(self, path):
        self.fig.savefig(path, dpi=80, bbox_inches='tight')

    def _draw(self):
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def close(self):
        try:
            self.plt.close(self.fig)
        except Exception:
            pass


# ============================================================
# CACHE LOADING
# ============================================================
def load_filtered_cache(cache_path='results/backtest_cache.npz'):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cache not found at {cache_path}. Run a backtest first."
        )
    print(f"[Data] Loading {cache_path}...")
    data = np.load(cache_path, allow_pickle=True)
    X = data['X']
    Y_ret = data['Y_ret']
    Y_risk = data['Y_risk']
    meta = data['meta']
    feat_names = data['feat_names']

    if meta.ndim != 2 or meta.shape[1] < 3:
        raise ValueError(
            f"Unexpected meta shape {meta.shape}; expected (N, 3)"
        )

    sample_tickers = meta[:, 0].astype(str)
    sample_dates = meta[:, 2].astype(str)

    n_total = len(X)

    if EXCLUDED_TICKERS:
        mask = ~np.isin(sample_tickers, list(EXCLUDED_TICKERS))
        n_excluded = int((~mask).sum())
        X = X[mask]
        Y_ret = Y_ret[mask]
        Y_risk = Y_risk[mask]
        meta = meta[mask]
        sample_tickers = sample_tickers[mask]
        sample_dates = sample_dates[mask]
        print(f"[Data] Excluded {n_excluded} samples from "
              f"{sorted(EXCLUDED_TICKERS)} (was {n_total}, now {len(X)})")

    sample_tickers = [str(t) for t in sample_tickers]
    sample_dates = [str(d) for d in sample_dates]

    print(f"[Data] {len(X):,} samples × {X.shape[1]} features, "
          f"{len(set(sample_tickers))} tickers, "
          f"date range {min(sample_dates)} ~ {max(sample_dates)}")

    return {
        'X': X, 'Y_ret': Y_ret, 'Y_risk': Y_risk,
        'sample_tickers': sample_tickers,
        'sample_dates': sample_dates,
        'meta': meta,
        'feat_names': feat_names,
    }


# ============================================================
# CONFIG OVERRIDE
# ============================================================
def override_config(config, overrides):
    originals = {}
    for k, v in overrides.items():
        originals[k] = getattr(config, k, None)
        setattr(config, k, v)
    return originals


def restore_config(config, originals):
    for k, v in originals.items():
        if v is None:
            if hasattr(config, k):
                delattr(config, k)
        else:
            setattr(config, k, v)


# ============================================================
# ADAM PATCH
# ============================================================
def patch_adam(weight_decay_override):
    original = optim.Adam.__init__

    def patched(self, params, lr=0.001, betas=(0.9, 0.999),
                eps=1e-8, weight_decay=None, amsgrad=False, **kw):
        if weight_decay is None or abs(weight_decay - 1e-4) < 1e-12:
            weight_decay = weight_decay_override
        return original(self, params, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, **kw)

    optim.Adam.__init__ = patched
    return original


# ============================================================
# FOLD RUNNER
# ============================================================
def run_fold_with_plot(data, train_tickers, test_tickers, fold_id,
                       config_module, live_plot, n_ensemble, n_select,
                       config_label):
    """
    One fold of heteroscedastic ensemble training + per-ticker aggregation.

    Returns dict with rank_corr, selection_alpha, full_ranking incl. risk/sigma,
    per_snapshot rankings, etc.
    """
    from scipy.stats import spearmanr

    X = data['X']
    Y_ret = data['Y_ret']
    Y_risk = data['Y_risk']
    sample_tickers = data['sample_tickers']
    sample_dates = data['sample_dates']

    train_set = set(str(t) for t in train_tickers)
    test_set = set(str(t) for t in test_tickers)
    train_mask = np.array([t in train_set for t in sample_tickers], dtype=bool)
    test_mask = np.array([t in test_set for t in sample_tickers], dtype=bool)

    if train_mask.sum() == 0:
        raise RuntimeError(
            f"Empty train set for fold {fold_id}. "
            f"sample_tickers[:3]={sample_tickers[:3]}, "
            f"train_tickers[:3]={list(train_tickers)[:3]}."
        )
    if test_mask.sum() == 0:
        raise RuntimeError(
            f"Empty test set for fold {fold_id}. "
            f"test_tickers[:3]={list(test_tickers)[:3]}"
        )

    X_tr_full = X[train_mask]
    Y_ret_tr_full = Y_ret[train_mask]
    Y_risk_tr_full = Y_risk[train_mask]
    X_te = X[test_mask]
    Y_ret_te = Y_ret[test_mask]
    Y_risk_te = Y_risk[test_mask]

    # Log-transform volatility for heteroscedastic NLL (Andersen et al. 2003).
    # Training in log-space; inference back-transforms via exp() for display.
    Y_risk_tr_full_log = np.log(np.maximum(Y_risk_tr_full, LOG_EPSILON))

    test_sample_tickers = np.array([sample_tickers[i]
                                    for i in range(len(sample_tickers))
                                    if test_mask[i]])
    test_dates = np.array([sample_dates[i]
                           for i in range(len(sample_dates))
                           if test_mask[i]])

    # Feature selection
    var_thr = getattr(config_module, 'VAR_THRESHOLD', 0.01)
    corr_thr = getattr(config_module, 'CORR_THRESHOLD', 0.05)
    var_per_feat = X_tr_full.var(axis=0)
    keep_var = var_per_feat > var_thr
    corr_per_feat = np.array([
        abs(np.corrcoef(X_tr_full[:, j], Y_ret_tr_full)[0, 1])
        if X_tr_full[:, j].std() > 0 else 0
        for j in range(X_tr_full.shape[1])
    ])
    corr_per_feat = np.nan_to_num(corr_per_feat, nan=0)
    keep_corr = corr_per_feat > corr_thr
    keep = keep_var & keep_corr
    if keep.sum() < 10:
        keep = keep_var
    if keep.sum() < 10:
        keep = np.ones(X.shape[1], dtype=bool)

    X_tr_full = X_tr_full[:, keep]
    X_te = X_te[:, keep]
    n_features = keep.sum()
    print(f"    Features: {n_features} selected (var_thr={var_thr:.4f}, "
          f"corr_thr={corr_thr:.4f})")

    # Train/val split
    rng = np.random.RandomState(SEED + fold_id)
    n_total = len(X_tr_full)
    perm = rng.permutation(n_total)
    n_val = int(n_total * 0.1)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]
    X_fit = X_tr_full[fit_idx]
    Yr_fit = Y_ret_tr_full[fit_idx]
    Yk_fit_log = Y_risk_tr_full_log[fit_idx]
    X_val = X_tr_full[val_idx]
    Yr_val = Y_ret_tr_full[val_idx]
    Yk_val_log = Y_risk_tr_full_log[val_idx]

    print(f"    Train fit: {len(X_fit):,}, val: {len(X_val):,}, "
          f"test: {len(X_te):,}")

    arch = getattr(config_module, 'TRAINING_NN_ARCHITECTURE', [64, 32, 16])
    if isinstance(arch, str):
        arch_map = {'small': [32, 16], 'medium': [64, 32, 16],
                    'large': [128, 64, 32]}
        arch = arch_map.get(arch, [64, 32, 16])
    lr = getattr(config_module, 'TRAINING_LR', 5e-4)
    weight_decay = getattr(config_module, 'TRAINING_WEIGHT_DECAY', 1e-4)
    epochs = getattr(config_module, 'TRAINING_EPOCHS', 5000)
    patience = getattr(config_module, 'EARLY_STOP_PATIENCE', 41)
    batch_size = 256

    # Ensemble: collect 4-tuple predictions per model (risk in log-space)
    ens_ret_mu = []
    ens_ret_sigma = []
    ens_risk_log_mu = []
    ens_risk_log_sigma = []

    for nn_idx in range(n_ensemble):
        torch.manual_seed(SEED + fold_id * 100 + nn_idx)
        np.random.seed(SEED + fold_id * 100 + nn_idx)

        if live_plot is not None:
            live_plot.start_model(nn_idx)

        model = HeteroscedasticDualHeadNN(in_dim=n_features, hidden_dims=arch, dropout=0.2)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        X_fit_t = torch.tensor(X_fit, dtype=torch.float32)
        Yr_fit_t = torch.tensor(Yr_fit, dtype=torch.float32)
        Yk_fit_t = torch.tensor(Yk_fit_log, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        Yr_val_t = torch.tensor(Yr_val, dtype=torch.float32)
        Yk_val_t = torch.tensor(Yk_val_log, dtype=torch.float32)
        X_te_t = torch.tensor(X_te, dtype=torch.float32)

        best_val = float('inf')
        best_epoch = 0
        best_state = None
        wait = 0

        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(X_fit_t))
            losses = []
            for i in range(0, len(order), batch_size):
                bi = order[i:i + batch_size]
                xb = X_fit_t[bi]
                yr = Yr_fit_t[bi]
                yk = Yk_fit_t[bi]
                pred = model(xb)
                loss, _, _ = heteroscedastic_loss(pred, yr, yk)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))

            model.eval()
            with torch.no_grad():
                pred_val = model(X_val_t)
                vl, _, _ = heteroscedastic_loss(pred_val, Yr_val_t, Yk_val_t)
            val_loss = vl.item()

            if live_plot is not None:
                live_plot.update(nn_idx, epoch, train_loss, val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        stop_epoch = epoch
        if live_plot is not None:
            live_plot.mark_best(nn_idx, best_epoch, best_val)
            live_plot.mark_stop(nn_idx, stop_epoch)

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            ret_mu, ret_lv, risk_log_mu, risk_log_lv = model(X_te_t)
        ens_ret_mu.append(ret_mu.numpy())
        ens_ret_sigma.append(torch.exp(0.5 * ret_lv).numpy())
        ens_risk_log_mu.append(risk_log_mu.numpy())
        ens_risk_log_sigma.append(torch.exp(0.5 * risk_log_lv).numpy())

        print(f"    NN #{nn_idx+1}: val_NLL={best_val:.6f} at epoch {best_epoch}, "
              f"stopped at {stop_epoch}")

    # Stack ensemble outputs (risk stays in log-space)
    ret_mu_stack = np.stack(ens_ret_mu)                  # (N_ens, n_test)
    ret_sigma_stack = np.stack(ens_ret_sigma)
    risk_log_mu_stack = np.stack(ens_risk_log_mu)
    risk_log_sigma_stack = np.stack(ens_risk_log_sigma)

    # Return predictions (linear space, unchanged)
    mean_pred = ret_mu_stack.mean(axis=0)
    ret_total_sigma = np.sqrt(ret_mu_stack.var(axis=0) + (ret_sigma_stack ** 2).mean(axis=0))
    ret_aleatoric_sigma = (ret_sigma_stack ** 2).mean(axis=0) ** 0.5

    # Risk in log-space — used for NLL eval, calibration plots (Andersen 2003)
    risk_log_mean = risk_log_mu_stack.mean(axis=0)
    risk_log_total_sigma = np.sqrt(
        risk_log_mu_stack.var(axis=0) + (risk_log_sigma_stack ** 2).mean(axis=0)
    )
    risk_log_aleatoric_sigma = (risk_log_sigma_stack ** 2).mean(axis=0) ** 0.5

    # Risk back-transformed to actual volatility scale for display + MAE.
    # exp(mu) gives the median of the lognormal; Andersen uses this convention.
    risk_mean_pred = np.exp(risk_log_mean)

    # Per-ticker aggregation
    unique_tickers = np.unique(test_sample_tickers)
    ticker_pred_mean = {}
    ticker_pred_std = {}
    ticker_ret_aleatoric = {}
    ticker_risk_mean = {}            # actual volatility scale
    ticker_risk_log_mean = {}        # log-space (for calibration)
    ticker_risk_log_sigma = {}       # log-space total uncertainty
    ticker_risk_log_aleatoric = {}   # log-space aleatoric only
    ticker_actual_mean = {}
    ticker_actual_risk = {}
    ticker_n_snapshots = {}

    for tk in unique_tickers:
        m = test_sample_tickers == tk
        ticker_pred_mean[tk] = float(mean_pred[m].mean())
        ticker_pred_std[tk] = float(ret_total_sigma[m].mean())
        ticker_ret_aleatoric[tk] = float(ret_aleatoric_sigma[m].mean())
        ticker_risk_mean[tk] = float(risk_mean_pred[m].mean())
        ticker_risk_log_mean[tk] = float(risk_log_mean[m].mean())
        ticker_risk_log_sigma[tk] = float(risk_log_total_sigma[m].mean())
        ticker_risk_log_aleatoric[tk] = float(risk_log_aleatoric_sigma[m].mean())
        ticker_actual_mean[tk] = float(Y_ret_te[m].mean())
        ticker_actual_risk[tk] = float(Y_risk_te[m].mean())
        ticker_n_snapshots[tk] = int(m.sum())

    # Rank correlation
    tk_list = sorted(unique_tickers)
    pred_arr = np.array([ticker_pred_mean[tk] for tk in tk_list])
    actual_arr = np.array([ticker_actual_mean[tk] for tk in tk_list])
    rho, p_val = spearmanr(pred_arr, actual_arr)

    # Top-5 / bottom-5 alpha
    pred_rank = np.argsort(-pred_arr)
    top5 = [tk_list[i] for i in pred_rank[:n_select]]
    bot5 = [tk_list[i] for i in pred_rank[-n_select:]]
    top5_actual = np.mean([ticker_actual_mean[tk] for tk in top5])
    bot5_actual = np.mean([ticker_actual_mean[tk] for tk in bot5])
    all_mean = float(np.mean(actual_arr))

    selection_alpha = float(top5_actual - all_mean)
    long_short = float(top5_actual - bot5_actual)

    # Full ranking (Task A) — risk in both linear and log space
    actual_rank_order = np.argsort(-actual_arr)
    actual_rank_map = {tk_list[i]: r for r, i in enumerate(actual_rank_order)}
    pred_rank_map = {tk_list[i]: r for r, i in enumerate(pred_rank)}

    full_ranking = []
    for tk in tk_list:
        full_ranking.append({
            'ticker': tk,
            'pred_ret': ticker_pred_mean[tk],
            'pred_std': ticker_pred_std[tk],
            'pred_aleatoric': ticker_ret_aleatoric[tk],
            'pred_risk': ticker_risk_mean[tk],                       # actual volatility scale
            'pred_risk_log_mean': ticker_risk_log_mean[tk],          # log-space (for calibration)
            'pred_risk_log_sigma': ticker_risk_log_sigma[tk],        # log-space total uncertainty
            'pred_risk_log_aleatoric': ticker_risk_log_aleatoric[tk],
            'actual_ret': ticker_actual_mean[tk],
            'actual_risk': ticker_actual_risk[tk],
            'pred_rank': pred_rank_map[tk] + 1,
            'actual_rank': actual_rank_map[tk] + 1,
            'rank_error': abs(pred_rank_map[tk] - actual_rank_map[tk]),
            'n_snapshots': ticker_n_snapshots[tk],
        })

    # Per-snapshot rankings (Task C)
    per_snapshot = []
    try:
        import pandas as pd
        dates_dt = pd.to_datetime(test_dates)
        buckets = dates_dt.to_period(PER_SNAPSHOT_BUCKET).astype(str)
        for bucket in np.unique(buckets):
            bucket_mask = buckets == bucket
            bucket_preds = mean_pred[bucket_mask]
            bucket_pred_sigmas = ret_total_sigma[bucket_mask]
            bucket_actuals = Y_ret_te[bucket_mask]
            bucket_tks = test_sample_tickers[bucket_mask]
            order = np.argsort(-bucket_preds)
            for rank, idx in enumerate(order):
                per_snapshot.append({
                    'bucket': bucket,
                    'ticker': bucket_tks[idx],
                    'pred_ret': float(bucket_preds[idx]),
                    'pred_sigma': float(bucket_pred_sigmas[idx]),
                    'actual_ret': float(bucket_actuals[idx]),
                    'rank_in_bucket': rank + 1,
                    'n_in_bucket': len(order),
                })
    except Exception as e:
        print(f"    [Warn] Per-snapshot ranking skipped: {e}")

    return {
        'fold_id': fold_id,
        'n_train_tickers': len(train_tickers),
        'n_test_tickers': len(test_tickers),
        'n_test_samples': len(X_te),
        'n_features': int(n_features),
        'rank_corr': float(rho),
        'rank_p': float(p_val),
        'selection_alpha': selection_alpha,
        'long_short_spread': long_short,
        'top5_mean_return': float(top5_actual),
        'bottom5_mean_return': float(bot5_actual),
        'all_mean_return': all_mean,
        'top5_tickers': top5,
        'bottom5_tickers': bot5,
        'full_ranking': full_ranking,
        'per_snapshot': per_snapshot,
        'test_date_range': (str(min(test_dates)), str(max(test_dates))),
    }


# ============================================================
# SPY BENCHMARK
# ============================================================
def fetch_spy_benchmark(date_ranges):
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        print("[Task B] yfinance not available, skipping SPY benchmark")
        return [None] * len(date_ranges)

    results = []
    spy = None
    try:
        spy = yf.Ticker('SPY').history(period='10y')
    except Exception as e:
        print(f"[Task B] SPY fetch failed: {e}")
        return [None] * len(date_ranges)

    if spy is None or len(spy) == 0:
        return [None] * len(date_ranges)

    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)

    for date_range in date_ranges:
        if date_range is None:
            results.append(None)
            continue
        d_start, d_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        returns = []
        for d in pd.date_range(d_start, d_end, freq='W'):
            try:
                p_start_idx = spy.index.searchsorted(d)
                if p_start_idx >= len(spy) - 63:
                    continue
                p_start = spy['Close'].iloc[p_start_idx]
                p_end = spy['Close'].iloc[p_start_idx + 63]
                returns.append((p_end / p_start) - 1)
            except Exception:
                continue
        if returns:
            results.append({
                'mean_3m_return': float(np.mean(returns)),
                'std_3m_return': float(np.std(returns)),
                'n_obs': len(returns),
            })
        else:
            results.append(None)

    return results


# ============================================================
# OUTPUT WRITERS
# ============================================================
def save_outputs(config_label, fold_results, spy_data, total_elapsed):
    out_dir = RESULTS_DIR / config_label
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
    except ImportError:
        print("[Output] pandas not available, writing JSON only")
        pd = None

    for fr in fold_results:
        fold_dir = out_dir / f'fold_{fr["fold_id"]+1}'
        fold_dir.mkdir(exist_ok=True)
        rows = fr['full_ranking']
        if pd is not None:
            df = pd.DataFrame(rows)
            df.to_csv(fold_dir / 'full_ranking.csv', index=False)
        else:
            with open(fold_dir / 'full_ranking.json', 'w') as f:
                json.dump(rows, f, indent=2)

        ps = fr.get('per_snapshot') or []
        if ps and pd is not None:
            pd.DataFrame(ps).to_csv(fold_dir / 'per_snapshot_ranking.csv', index=False)

    spy_summary = []
    for fr, spy in zip(fold_results, spy_data):
        spy_summary.append({
            'fold_id': fr['fold_id'] + 1,
            'fold_date_start': fr['test_date_range'][0] if fr['test_date_range'] else None,
            'fold_date_end': fr['test_date_range'][1] if fr['test_date_range'] else None,
            'spy_mean_3m_return': spy['mean_3m_return'] if spy else None,
            'spy_std_3m_return': spy['std_3m_return'] if spy else None,
            'spy_n_obs': spy['n_obs'] if spy else None,
            'top5_actual_3m': fr['top5_mean_return'],
            'alpha_vs_spy': (fr['top5_mean_return'] - spy['mean_3m_return'])
                           if spy else None,
        })
    if pd is not None:
        pd.DataFrame(spy_summary).to_csv(out_dir / 'spy_benchmark.csv', index=False)
    else:
        with open(out_dir / 'spy_benchmark.json', 'w') as f:
            json.dump(spy_summary, f, indent=2)

    summary = {
        'config_label': config_label,
        'n_ensemble': N_ENSEMBLE_STAGE2,
        'n_folds_used': len(fold_results),
        'fold_ids_used': [fr['fold_id'] + 1 for fr in fold_results],
        'sndk_excluded': 'SNDK' in EXCLUDED_TICKERS,
        'aggregate': {
            'rank_corr_mean': float(np.mean([fr['rank_corr'] for fr in fold_results])),
            'rank_corr_std': float(np.std([fr['rank_corr'] for fr in fold_results])),
            'selection_alpha_mean': float(np.mean(
                [fr['selection_alpha'] for fr in fold_results])),
            'selection_alpha_std': float(np.std(
                [fr['selection_alpha'] for fr in fold_results])),
            'top5_return_mean': float(np.mean(
                [fr['top5_mean_return'] for fr in fold_results])),
            'all_return_mean': float(np.mean(
                [fr['all_mean_return'] for fr in fold_results])),
        },
        'per_fold': [
            {
                'fold_id': fr['fold_id'] + 1,
                'rank_corr': fr['rank_corr'],
                'selection_alpha': fr['selection_alpha'],
                'top5': fr['top5_tickers'],
                'n_test_tickers': fr['n_test_tickers'],
            }
            for fr in fold_results
        ],
        'total_elapsed_min': total_elapsed / 60,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Output] All results written to {out_dir}/")
    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-rank', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--results-json',
                        default='results/optuna_stage1_results.json')
    parser.add_argument('--include-sndk', action='store_true')
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    global EXCLUDED_TICKERS, RESULTS_DIR
    if args.include_sndk:
        EXCLUDED_TICKERS = set()
        RESULTS_DIR = Path('results/stage2_with_sndk')
        print("[Sensitivity mode] SNDK included; output -> results/stage2_with_sndk/")

    with open(args.results_json) as f:
        optuna_results = json.load(f)

    rank_idx = args.config_rank - 1
    if rank_idx >= len(optuna_results['top_3_configs']):
        print(f"[Error] config-rank {args.config_rank} not available")
        sys.exit(1)

    chosen = optuna_results['top_3_configs'][rank_idx]
    params = chosen['params']
    config_label = f"top{args.config_rank}_trial{chosen['trial_number']}"

    print("=" * 70)
    print(f"STAGE 2 RETRAIN — {config_label}")
    print("=" * 70)
    print(f"Optuna Stage 1 mean rank_corr (N=5, 4-fold): {chosen['mean_rank_corr']:+.4f}")
    print(f"Hyperparameters:")
    for k, v in params.items():
        print(f"  {k:18s} = {v}")
    print(f"\nStage 2 settings:")
    print(f"  N_ENSEMBLE  = {N_ENSEMBLE_STAGE2}")
    print(f"  Folds       = {[f+1 for f in FOLDS_STAGE2]}")
    print(f"  Excluded    = {sorted(EXCLUDED_TICKERS)}")
    print(f"  Live plot   = {'OFF' if args.no_plot else 'ON'}")
    print(f"  Loss        = Gaussian NLL (heteroscedastic, Kendall & Gal 2017)")
    print(f"  Risk target = log-space (Andersen et al. 2003)")
    print("=" * 70)

    import config as config_module

    config_overrides = {
        'TRAINING_LR': params['lr'],
        'TRAINING_HUBER_DELTA': params['huber_delta'],
        'TRAINING_NN_ARCHITECTURE': params['architecture'],
        'VAR_THRESHOLD': params['var_threshold'],
        'CORR_THRESHOLD': params['corr_threshold'],
        'N_ENSEMBLE': N_ENSEMBLE_STAGE2,
        'TRAINING_WEIGHT_DECAY': params['weight_decay'],
    }
    originals = override_config(config_module, config_overrides)
    original_adam = patch_adam(params['weight_decay'])

    try:
        data = load_filtered_cache()

        from backtest import _stratified_kfold, _get_ticker_sectors
        unique_tks = sorted(set(data['sample_tickers']))
        ticker_sectors = _get_ticker_sectors(unique_tks, verbose=False)
        folds = _stratified_kfold(unique_tks, ticker_sectors, n_folds=5)

        live_plot = None
        if not args.no_plot:
            param_str = (f"lr={params['lr']:.4f} arch={params['architecture']} "
                         f"N={N_ENSEMBLE_STAGE2}")
            live_plot = LiveLossPlot(
                n_models=N_ENSEMBLE_STAGE2,
                fold_id=FOLDS_STAGE2[0],
                config_label=param_str,
            )

        t_overall = time.time()
        fold_results = []
        for fi, fold_idx in enumerate(FOLDS_STAGE2):
            train_tks, test_tks = folds[fold_idx]
            print(f"\n{'─' * 60}")
            print(f"Fold {fold_idx+1}/5 — Train: {len(train_tks)}, "
                  f"Test: {len(test_tks)} tickers")
            print(f"{'─' * 60}")
            t_fold = time.time()
            if live_plot is not None and fi > 0:
                param_str = (f"lr={params['lr']:.4f} arch={params['architecture']} "
                             f"N={N_ENSEMBLE_STAGE2}")
                live_plot.next_fold(fold_idx, param_str)

            result = run_fold_with_plot(
                data, train_tks, test_tks, fold_idx,
                config_module, live_plot,
                n_ensemble=N_ENSEMBLE_STAGE2,
                n_select=N_SELECT,
                config_label=config_label,
            )
            fold_results.append(result)
            elapsed = (time.time() - t_fold) / 60
            print(f"  Fold {fold_idx+1} done: rank_corr={result['rank_corr']:+.4f}, "
                  f"alpha={result['selection_alpha']*100:+.1f}%p, "
                  f"top5={result['top5_tickers']}, elapsed={elapsed:.1f}min")

            if live_plot is not None:
                snapshot_path = RESULTS_DIR / config_label / f"loss_curves_fold{fold_idx+1}.png"
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                live_plot.save_snapshot(str(snapshot_path))

        total_elapsed = time.time() - t_overall

        print(f"\n[Task B] Computing SPY benchmarks...")
        date_ranges = [fr['test_date_range'] for fr in fold_results]
        spy_data = fetch_spy_benchmark(date_ranges)

        summary = save_outputs(config_label, fold_results, spy_data, total_elapsed)

        print("\n" + "=" * 70)
        print(f"STAGE 2 COMPLETE — {config_label}")
        print("=" * 70)
        agg = summary['aggregate']
        print(f"Mean rank_corr (5 folds, N=20):  {agg['rank_corr_mean']:+.4f}  "
              f"± {agg['rank_corr_std']:.4f}")
        print(f"Mean selection alpha:             "
              f"{agg['selection_alpha_mean']*100:+.1f}%p  "
              f"± {agg['selection_alpha_std']*100:.1f}%p")
        print(f"\nComparison points:")
        print(f"  Optuna Stage 1 (N=5, 4-fold):  "
              f"{chosen['mean_rank_corr']:+.4f}")
        print(f"  v2.3.4 baseline (N=20, 5-fold): +0.5311 (with SNDK, Huber)")
        print(f"  v2.3.8 baseline (N=20, 5-fold): +0.5181 (no SNDK, Huber)")
        print(f"  Stage 2 NLL (N=20, 5-fold):     "
              f"{agg['rank_corr_mean']:+.4f} (no SNDK, Gaussian NLL log-vol)")
        print(f"\nTotal elapsed: {total_elapsed/60:.1f} min "
              f"({total_elapsed/3600:.1f} h)")

        if live_plot is not None:
            print("\nClose plot window manually when done reviewing.")
            try:
                input("Press Enter to close plot and exit... ")
            except (EOFError, KeyboardInterrupt):
                pass
            live_plot.close()

    finally:
        restore_config(config_module, originals)
        optim.Adam.__init__ = original_adam


if __name__ == '__main__':
    main()
