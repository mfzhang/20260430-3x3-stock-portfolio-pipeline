"""
historical.py — historical cross-section training

S&P 500 + NASDAQ-100 (~550 tickers) x 10 years = ~125,000 samples
+ FRED macro features + Fama-French factors + cross-asset data

Usage:
    from historical import build_training_data, walk_forward_splits
    X, Y_ret, Y_risk, meta, feat_names = build_training_data()
"""

import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

from data_auto import compute_technical_features
from io import StringIO


# ============================================================
# UNIVERSE EXPANSION
# ============================================================

def auto_expand_universe(seed_etfs=None):
    """Build training universe. Uses training_universe.py if available, else legacy fallback."""
    try:
        from training_universe import get_training_tickers
        tickers, _ = get_training_tickers(verbose=True)
        if len(tickers) > 50:
            return tickers
        print(f"  [Warning] Only {len(tickers)} tickers from training_universe, using fallback")
    except ImportError:
        print("  [Info] training_universe.py not found, using legacy expansion")
    except Exception as e:
        print(f"  [Warning] training_universe failed: {e}, using fallback")

    return _legacy_expand_universe(seed_etfs)


def _legacy_expand_universe(seed_etfs=None):
    """Legacy fallback: ETF holdings + Wikipedia scraping."""
    import yfinance as yf
    from data_auto import TICKERS
    import pandas as pd
    import urllib.request

    core = set(TICKERS)
    expanded = list(TICKERS)

    print(f"\n[Universe Expansion] Legacy mode (ETF + Wikipedia)")

    # Wikipedia NASDAQ-100
    headers = {"User-Agent": "Mozilla/5.0 (stock-pipeline/2.0)"}
    try:
        req = urllib.request.Request("https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers)
        html = urllib.request.urlopen(req).read().decode()
        tables = pd.read_html(StringIO(html))
        for t in tables:
            if 'Ticker' in t.columns:
                sub_col = [c for c in t.columns if 'Subsector' in c or 'Industry' in c]
                if sub_col:
                    col = sub_col[0]
                    semi = t[t[col].str.contains('Semicon|Electron|Chip', case=False, na=False)]['Ticker'].tolist()
                    health = t[t[col].str.contains('Health|Biotech|Pharma|Medical', case=False, na=False)]['Ticker'].tolist()
                else:
                    semi, health = [], []
                new_semi = [tk for tk in semi if tk not in core][:10]
                new_health = [tk for tk in health if tk not in core][:5]
                expanded.extend(new_semi + new_health)
                core.update(new_semi + new_health)
                print(f"    NASDAQ-100: +{len(new_semi)} semi, +{len(new_health)} health")
                break
    except Exception as e:
        print(f"    NASDAQ-100: failed ({e})")

    # Wikipedia S&P 500
    try:
        req = urllib.request.Request("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
        html = urllib.request.urlopen(req).read().decode()
        tables = pd.read_html(StringIO(html))
        sp500 = tables[0]
        sp_health = sp500[sp500['GICS Sub-Industry'].str.contains('Health|Biotech|Pharma|Medical', case=False, na=False)]['Symbol'].tolist()
        new_sp = [tk.replace('.', '-') for tk in sp_health if tk not in core][:5]
        expanded.extend(new_sp)
        core.update(new_sp)
        print(f"    S&P 500 Health: +{len(new_sp)} tickers")
    except Exception as e:
        print(f"    S&P 500: failed ({e})")

    expanded = list(dict.fromkeys(expanded))
    print(f"\n  Final universe: {len(expanded)} tickers")
    return expanded


# ============================================================
# MACRO FEATURE LOADING
# ============================================================

_MACRO_DF = None
_FF_DF = None
_XASSET_DF = None


def _load_macro_data():
    """Load FRED + Fama-French + cross-asset data once (cached)."""
    global _MACRO_DF, _FF_DF, _XASSET_DF

    if _MACRO_DF is not None:
        return

    # [Ablation] respect USE_MACRO_FEATURES flag — skip all macro loading if disabled
    try:
        import config
        if not getattr(config, 'USE_MACRO_FEATURES', True):
            print("  [Ablation] Macro features DISABLED by config.USE_MACRO_FEATURES=False")
            return
    except Exception:
        pass

    try:
        import config
        fred_key = getattr(config, 'FRED_API_KEY', '')
        if fred_key:
            from training_universe import fetch_fred_data
            _MACRO_DF = fetch_fred_data(fred_key, start_date='2014-01-01', verbose=True)
    except Exception as e:
        print(f"  [FRED] Failed: {e}")

    try:
        from training_universe import fetch_fama_french_factors
        _FF_DF = fetch_fama_french_factors(start_date='2014-01-01', verbose=True)
    except Exception as e:
        print(f"  [Fama-French] Failed: {e}")

    try:
        from training_universe import fetch_cross_asset_data
        _XASSET_DF = fetch_cross_asset_data(period='10y', verbose=True)
    except Exception as e:
        print(f"  [Cross-Asset] Failed: {e}")


def _get_macro_features_at(date_str):
    """Return macro/factor/cross-asset features for a given date as a dict."""
    import pandas as pd

    features = {}

    if _MACRO_DF is not None and not _MACRO_DF.empty:
        try:
            from training_universe import get_macro_features_for_date
            features.update(get_macro_features_for_date(_MACRO_DF, date_str))
        except:
            pass

    if _FF_DF is not None and not _FF_DF.empty:
        try:
            from training_universe import get_ff_features_for_date
            features.update(get_ff_features_for_date(_FF_DF, date_str))
        except:
            pass

    if _XASSET_DF is not None and not _XASSET_DF.empty:
        try:
            from training_universe import get_cross_asset_features
            features.update(get_cross_asset_features(_XASSET_DF, date_str))
        except:
            pass

    return features


# ============================================================
# TRAINING DATA BUILDER
# ============================================================

def build_training_data(tickers=None, period="10y", snapshot_interval=10):
    """
    Build cross-sectional training data across many timestamps.
    Per-ticker snapshots so each ticker uses its full available history.
    Returns: X, Y_ret, Y_risk, meta, feat_names.
    """
    import yfinance as yf

    if tickers is None:
        tickers = auto_expand_universe()

    print(f"\n[historical] Building training data from {period} of history...")
    print(f"  Tickers: {len(tickers)}, Snapshot interval: {snapshot_interval} days")

    # Load macro data
    print(f"\n  Loading macro/factor data...")
    _load_macro_data()
    has_macro = _MACRO_DF is not None or _FF_DF is not None or _XASSET_DF is not None
    if has_macro:
        print(f"  Macro data loaded successfully")
    else:
        print(f"  No macro data available (technical features only)")

    # Download all price data in batches
    all_hist = {}
    failed = 0
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"  Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1} "
              f"({len(batch)} tickers)...")

        for tk in batch:
            try:
                hist = yf.Ticker(tk).history(period=period)
                if hist is not None:
                    hist = hist.dropna(subset=['Close'])
                if hist is not None and len(hist) >= 126:
                    all_hist[tk] = hist
                else:
                    failed += 1
            except:
                failed += 1

        time.sleep(1)  # rate limit between batches

    valid_tickers = list(all_hist.keys())
    print(f"\n  Downloaded: {len(valid_tickers)} valid, {failed} failed")

    if not valid_tickers:
        raise RuntimeError("No valid tickers!")

    # Generate snapshots
    forward_days = 63
    vol_forward = 21
    lookback_min = 63

    X_all, Y_ret_all, Y_risk_all, meta_all = [], [], [], []
    feat_names = None

    total_potential = 0
    for tk in valid_tickers:
        tk_len = len(all_hist[tk]['Close'].values)
        n_snaps = len(range(lookback_min, tk_len - forward_days, snapshot_interval))
        total_potential += n_snaps

    print(f"  Potential samples: {total_potential:,}")
    print(f"  Processing snapshots...")

    progress_interval = max(len(valid_tickers) // 20, 1)

    for ti, tk in enumerate(valid_tickers):
        hist = all_hist[tk]
        close = hist['Close'].values
        high = hist['High'].values
        low = hist['Low'].values
        volume = hist['Volume'].values
        dates = hist.index

        for snap_idx in range(lookback_min, len(close) - forward_days, snapshot_interval):
            try:
                sf = _SliceFrame(
                    close[:snap_idx + 1],
                    high[:snap_idx + 1],
                    low[:snap_idx + 1],
                    volume[:snap_idx + 1]
                )
                feats = compute_technical_features(sf)

                # Add macro features
                if has_macro and snap_idx < len(dates):
                    date_str = str(dates[snap_idx].date())
                    macro_feats = _get_macro_features_at(date_str)
                    feats.update(macro_feats)

                if feat_names is None:
                    feat_names = sorted(feats.keys())

                vec = np.array([feats.get(k, 0) for k in feat_names], dtype=float)
                vec = np.nan_to_num(vec, nan=0, posinf=0, neginf=0)

                # Target: forward 3-month return
                future_price = close[snap_idx + forward_days]
                current_price = close[snap_idx]
                y_ret = (future_price / current_price) - 1

                # Target: forward 1-month realized volatility
                fwd_slice = close[snap_idx:snap_idx + vol_forward + 1]
                fwd_daily = np.diff(fwd_slice) / fwd_slice[:-1]
                y_risk = np.std(fwd_daily) * np.sqrt(252)

                X_all.append(vec)
                Y_ret_all.append(y_ret)
                Y_risk_all.append(y_risk)

                date_str = str(dates[snap_idx].date()) if snap_idx < len(dates) else f"idx_{snap_idx}"
                meta_all.append((tk, snap_idx, date_str))

            except Exception:
                continue

        if (ti + 1) % progress_interval == 0:
            print(f"    {ti+1}/{len(valid_tickers)} tickers, {len(X_all):,} samples so far")

    if not X_all:
        raise RuntimeError("No valid training samples!")

    # Pad to consistent dimension
    max_dim = max(len(x) for x in X_all)
    X_padded = []
    for x in X_all:
        if len(x) < max_dim:
            x = np.concatenate([x, np.zeros(max_dim - len(x))])
        X_padded.append(x)

    X = np.array(X_padded)
    Y_ret = np.array(Y_ret_all)
    Y_risk = np.array(Y_risk_all)

    print(f"\n  Result: {X.shape[0]:,} samples x {X.shape[1]} features")
    print(f"  Y_ret range: [{Y_ret.min():.3f}, {Y_ret.max():.3f}]")
    print(f"  Y_risk range: [{Y_risk.min():.3f}, {Y_risk.max():.3f}]")
    if meta_all:
        print(f"  Date range: {meta_all[0][2]} ~ {meta_all[-1][2]}")

    return X, Y_ret, Y_risk, meta_all, feat_names


# ============================================================
# WALK-FORWARD CV
# ============================================================

def walk_forward_splits(meta, n_splits=4):
    """Time-ordered Walk-Forward cross-validation splits."""
    sorted_indices = sorted(range(len(meta)), key=lambda i: meta[i][1])
    n = len(sorted_indices)
    test_size = n // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        test_start = n - (n_splits - i) * test_size
        test_end = min(test_start + test_size, n)
        train_idx = sorted_indices[:test_start]
        test_idx = sorted_indices[test_start:test_end]
        if len(train_idx) >= 10 and len(test_idx) >= 5:
            splits.append((train_idx, test_idx))

    return splits


def train_and_evaluate(X, Y_ret, Y_risk, splits):
    """Train and evaluate the model on each Walk-Forward split."""
    all_ret_errors = []
    all_risk_errors = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, Y_ret_train, Y_risk_train = X[train_idx], Y_ret[train_idx], Y_risk[train_idx]
        X_test, Y_ret_test, Y_risk_test = X[test_idx], Y_ret[test_idx], Y_risk[test_idx]

        # Feature selection
        feat_var = X_train.var(0)
        feat_corr = np.array([abs(np.corrcoef(X_train[:, d], Y_ret_train)[0, 1])
                              if np.std(X_train[:, d]) > 1e-8 else 0
                              for d in range(X_train.shape[1])])
        keep = (feat_var > 0.01) & (feat_corr > 0.05)
        if keep.sum() < 10:
            keep = feat_var > 0.01
        X_train, X_test = X_train[:, keep], X_test[:, keep]

        # Normalize
        mu = X_train.mean(0)
        sigma = X_train.std(0) + 1e-8
        X_train_n = np.clip((X_train - mu) / sigma, -5, 5)
        X_test_n = np.clip((X_test - mu) / sigma, -5, 5)

        ret_err, risk_err = _train_fold_torch(X_train_n, Y_ret_train, Y_risk_train,
                                               X_test_n, Y_ret_test, Y_risk_test)

        all_ret_errors.append(ret_err)
        all_risk_errors.append(risk_err)
        print(f"    Fold {fold + 1}: train={len(train_idx):,}, test={len(test_idx):,}, "
              f"ret_err={ret_err * 100:.1f}%p, risk_err={risk_err * 100:.1f}%p")

    mean_ret = np.mean(all_ret_errors)
    mean_risk = np.mean(all_risk_errors)

    print(f"\n  Walk-Forward CV Results:")
    print(f"    Mean Return Error:  {mean_ret * 100:.1f}%p")
    print(f"    Mean Risk Error:    {mean_risk * 100:.1f}%p")

    return all_ret_errors, all_risk_errors, mean_ret, mean_risk


def _train_fold_torch(X_tr, Y_ret_tr, Y_risk_tr, X_te, Y_ret_te, Y_risk_te):
    """
    Walk-Forward CV diagnostic with heteroscedastic dual-head NN.

    Uses the same architecture as stage2_retrain.py / backtest.py production
    (v2.3.12) for methodological consistency. Y_risk is log-transformed per
    Andersen et al. (2003) financial volatility convention; MAE is computed
    in the original linear scale.

    Returns (return_MAE, volatility_MAE) on test set.
    """
    import torch
    import config
    from models import HeteroscedasticDualHeadNN, heteroscedastic_loss

    LOG_EPSILON = 1e-4

    D = X_tr.shape[1]

    arch = getattr(config, 'TRAINING_NN_ARCHITECTURE', [64, 32, 16])
    if isinstance(arch, str):
        arch_map = {'small': [32, 16], 'medium': [64, 32, 16],
                    'large': [128, 64, 32]}
        arch = arch_map.get(arch, [64, 32, 16])

    lr = getattr(config, 'TRAINING_LR', 0.0005)
    epochs = getattr(config, 'TRAINING_EPOCHS', 800)
    weight_decay = getattr(config, 'TRAINING_WEIGHT_DECAY', 1e-4)

    Yk_tr_log = np.log(np.maximum(Y_risk_tr, LOG_EPSILON))

    X = torch.tensor(X_tr, dtype=torch.float32)
    yr = torch.tensor(Y_ret_tr, dtype=torch.float32)
    yk_log = torch.tensor(Yk_tr_log, dtype=torch.float32)

    model = HeteroscedasticDualHeadNN(in_dim=D, hidden_dims=arch, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    best_loss = float('inf')
    patience = 0
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss, _, _ = heteroscedastic_loss(pred, yr, yk_log)
        if torch.isnan(loss):
            break
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
            if patience > getattr(config, 'EARLY_STOP_PATIENCE', 41):
                break

    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_te, dtype=torch.float32)
        ret_mu, _, risk_log_mu, _ = model(Xt)
        pr = ret_mu.numpy()
        # Back-transform log-space volatility to actual scale for MAE comparison
        pk = np.exp(risk_log_mu.numpy())

    return np.mean(np.abs(pr - Y_ret_te)), np.mean(np.abs(pk - Y_risk_te))


# ============================================================
# SLICE FRAME (for feature computation)
# ============================================================

class _SliceFrame:
    """Minimal DataFrame-like wrapper for compute_technical_features."""
    def __init__(self, close, high, low, volume):
        self._data = {'Close': close, 'High': high, 'Low': low, 'Volume': volume}

    def __getitem__(self, key):
        class _Col:
            def __init__(self, vals):
                self.values = vals
            def dropna(self):
                v = self.values
                return _Col(v[~np.isnan(v)])
        return _Col(self._data[key])


# ============================================================
# STANDALONE: python historical.py
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HISTORICAL TRAINING + WALK-FORWARD VALIDATION")
    print("  S&P 500 Training Universe + Macro Features")
    print("=" * 70)

    expanded = auto_expand_universe()

    import config
    X, Y_ret, Y_risk, meta, feat_names = build_training_data(
        tickers=expanded,
        period=getattr(config, 'TRAINING_PERIOD', '10y'),
        snapshot_interval=getattr(config, 'TRAINING_SNAPSHOT_INTERVAL', 10),
    )

    splits = walk_forward_splits(meta, n_splits=4)
    print(f"\n  {len(splits)} walk-forward splits:")
    for i, (tr, te) in enumerate(splits):
        print(f"    Split {i + 1}: train={len(tr):,} samples, test={len(te):,} samples")

    print(f"\n[Training + Evaluation]")
    ret_errs, risk_errs, mr, mk = train_and_evaluate(X, Y_ret, Y_risk, splits)

    print(f"\n{'=' * 70}")
    print(f"DONE")
    print(f"  Training samples: {X.shape[0]:,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Walk-Forward Return Error: {mr * 100:.1f}%p")
    print(f"  Walk-Forward Risk Error: {mk * 100:.1f}%p")
    print(f"{'=' * 70}")
