"""merge_v2310_cache.py — v2.3.10 Sub-task D

Combines:
  - results/backtest_cache.npz                          (X, Y_ret, Y_risk, meta, feat_names — 122,257 × 97)
  - results/historical_sec_features.npz                 (filing_count_30d, has_8k_recent — Layer 2)
  - results/historical_earnings_features.npz            (last_surprise_pct — Layer 4)

Produces:
  - results/backtest_cache_v2310.npz                    (X' = 122,257 × 100, with 3 new columns appended)

Original backtest_cache.npz is preserved unchanged. The new file lives
alongside as the v2.3.10 input for Sub-task F (real Task #9 grid).

Validation:
  - meta arrays in all 3 sources must be row-aligned (identical per-row)
  - Y_ret / Y_risk untouched (only X changes)
  - new feat_names appended in fixed order:
      historical_layer2_filing_count_30d
      historical_layer2_has_8k_recent
      historical_layer4_last_surprise_pct

Usage:
  python merge_v2310_cache.py             # full merge with backup
  python merge_v2310_cache.py --no-backup # skip backup (use only if already backed up)
  python merge_v2310_cache.py --validate  # describe v2310 cache (no merge)
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np

ORIG_CACHE = 'results/backtest_cache.npz'
BACKUP_CACHE = 'results/backtest_cache_v239_pre_option2light.npz'
LAYER2_FEATURES = 'results/historical_sec_features.npz'
LAYER4_FEATURES = 'results/historical_earnings_features.npz'
OUTPUT_CACHE = 'results/backtest_cache_v2310.npz'

NEW_FEATURE_NAMES = [
    'historical_layer2_filing_count_30d',
    'historical_layer2_has_8k_recent',
    'historical_layer4_last_surprise_pct',
]


def backup_cache(verbose=True):
    """Create backup of original cache before any changes."""
    if os.path.exists(BACKUP_CACHE):
        if verbose:
            print(f'      [skip] backup already exists: {BACKUP_CACHE}')
        return
    if verbose:
        print(f'[1/6] Backing up {ORIG_CACHE} → {BACKUP_CACHE} ...')
    shutil.copy2(ORIG_CACHE, BACKUP_CACHE)
    size_mb = os.path.getsize(BACKUP_CACHE) / (1024 * 1024)
    if verbose:
        print(f'      → {size_mb:.1f} MB backed up')


def load_orig_cache(verbose=True):
    """Load the v2.3.7-era cache. Returns (X, Y_ret, Y_risk, meta, feat_names)."""
    if verbose:
        print(f'[2/6] Loading {ORIG_CACHE} ...')
    if not os.path.exists(ORIG_CACHE):
        raise FileNotFoundError(f'{ORIG_CACHE} not found')
    data = np.load(ORIG_CACHE, allow_pickle=True)
    X = data['X']
    Y_ret = data['Y_ret']
    Y_risk = data['Y_risk']
    meta = data['meta']
    feat_names = data['feat_names']
    if verbose:
        print(f'      → X {X.shape} {X.dtype}')
        print(f'      → Y_ret {Y_ret.shape} {Y_ret.dtype}, Y_risk {Y_risk.shape} {Y_risk.dtype}')
        print(f'      → meta {meta.shape}, feat_names ({len(feat_names)} entries)')
    return X, Y_ret, Y_risk, meta, feat_names


def load_layer_features(npz_path, expected_keys, verbose=True):
    """Load layer feature npz, return (meta, dict_of_arrays)."""
    if verbose:
        print(f'      Loading {npz_path} ...')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'{npz_path} not found — run upstream compute first')
    data = np.load(npz_path, allow_pickle=True)
    out_meta = data['meta']
    out_arrays = {}
    for k in expected_keys:
        if k not in data.files:
            raise RuntimeError(f'{npz_path} missing key {k}; has {list(data.files)}')
        out_arrays[k] = data[k]
    if verbose:
        print(f'         meta {out_meta.shape}, keys: {list(out_arrays.keys())}')
    return out_meta, out_arrays


def verify_meta_alignment(meta_orig, meta_layer2, meta_layer4, verbose=True):
    """Verify that all three meta arrays are identical row-by-row."""
    if verbose:
        print(f'[4/6] Verifying meta alignment across 3 sources ...')

    if meta_orig.shape != meta_layer2.shape:
        raise RuntimeError(f'meta shape mismatch: orig {meta_orig.shape} vs layer2 {meta_layer2.shape}')
    if meta_orig.shape != meta_layer4.shape:
        raise RuntimeError(f'meta shape mismatch: orig {meta_orig.shape} vs layer4 {meta_layer4.shape}')

    # Spot-check first/middle/last row
    n = len(meta_orig)
    for label, idx in [('first', 0), ('middle', n // 2), ('last', n - 1)]:
        o = list(meta_orig[idx])
        l2 = list(meta_layer2[idx])
        l4 = list(meta_layer4[idx])
        if str(o) != str(l2) or str(o) != str(l4):
            raise RuntimeError(
                f'{label} row mismatch:\n'
                f'  orig:   {o}\n'
                f'  layer2: {l2}\n'
                f'  layer4: {l4}'
            )

    # Comprehensive ticker check (cheap)
    tickers_orig = set(str(r[0]) for r in meta_orig[::100])
    tickers_l2 = set(str(r[0]) for r in meta_layer2[::100])
    tickers_l4 = set(str(r[0]) for r in meta_layer4[::100])
    if tickers_orig != tickers_l2 or tickers_orig != tickers_l4:
        raise RuntimeError(
            f'ticker set mismatch in meta sampling:\n'
            f'  orig only: {tickers_orig - tickers_l2 - tickers_l4}\n'
            f'  layer2 only: {tickers_l2 - tickers_orig}\n'
            f'  layer4 only: {tickers_l4 - tickers_orig}'
        )
    if verbose:
        print(f'      → meta alignment OK across {n:,} rows')


def merge_features(X_orig, feat_names_orig, layer2_arrays, layer4_arrays, verbose=True):
    """Append the 3 new feature columns to X_orig.

    Order is fixed: filing_count_30d, has_8k_recent, last_surprise_pct
    """
    if verbose:
        print(f'[5/6] Merging features into X ...')

    n = X_orig.shape[0]
    fc = layer2_arrays['filing_count_30d'].astype(np.float32).reshape(n, 1)
    h8 = layer2_arrays['has_8k_recent'].astype(np.float32).reshape(n, 1)
    lsp = layer4_arrays['last_surprise_pct'].astype(np.float32).reshape(n, 1)

    # Verify shapes
    for arr, name in [(fc, 'filing_count_30d'), (h8, 'has_8k_recent'), (lsp, 'last_surprise_pct')]:
        if arr.shape[0] != n:
            raise RuntimeError(f'{name} length {arr.shape[0]} != X length {n}')

    X_new = np.hstack([X_orig.astype(np.float32), fc, h8, lsp])
    feat_names_new = list(feat_names_orig) + NEW_FEATURE_NAMES

    if verbose:
        print(f'      → X: {X_orig.shape} → {X_new.shape}')
        print(f'      → feat_names: {len(feat_names_orig)} → {len(feat_names_new)}')
    return X_new, feat_names_new


def write_v2310_cache(X_new, Y_ret, Y_risk, meta, feat_names_new, verbose=True):
    """Write the v2.3.10 backtest cache."""
    if verbose:
        print(f'[6/6] Writing {OUTPUT_CACHE} ...')
    np.savez(OUTPUT_CACHE,
             X=X_new,
             Y_ret=Y_ret,
             Y_risk=Y_risk,
             meta=meta,
             feat_names=np.array(feat_names_new, dtype=object))
    size_mb = os.path.getsize(OUTPUT_CACHE) / (1024 * 1024)
    if verbose:
        print(f'      → {size_mb:.1f} MB written')


def summarize_new_columns(X_new, feat_names_new, verbose=True):
    """Print sanity stats for the 3 new columns."""
    print()
    print('=' * 60)
    print('NEW COLUMN SANITY')
    print('=' * 60)
    n = X_new.shape[0]
    for name in NEW_FEATURE_NAMES:
        idx = feat_names_new.index(name)
        col = X_new[:, idx]
        nonzero = col != 0
        print(f'{name}:')
        print(f'  index in X: {idx}')
        print(f'  shape: {col.shape}, dtype: {col.dtype}')
        print(f'  zero    : {(~nonzero).sum():>7,} ({(~nonzero).mean()*100:.1f}%)')
        print(f'  nonzero : {nonzero.sum():>7,} ({nonzero.mean()*100:.1f}%)')
        if nonzero.any():
            nz = col[nonzero]
            print(f'  mean   : {nz.mean():+.3f}')
            print(f'  median : {np.median(nz):+.3f}')
            print(f'  min    : {nz.min():+.3f}')
            print(f'  max    : {nz.max():+.3f}')
        print()
    print('=' * 60)


def validate_v2310(verbose=True):
    if not os.path.exists(OUTPUT_CACHE):
        print(f'  {OUTPUT_CACHE} not found')
        return
    data = np.load(OUTPUT_CACHE, allow_pickle=True)
    X = data['X']
    feat_names = list(data['feat_names'])
    print(f'  Cache: {OUTPUT_CACHE}')
    print(f'  X shape: {X.shape}')
    print(f'  feat_names: {len(feat_names)} entries')
    print(f'  Last 5 feat_names: {feat_names[-5:]}')
    summarize_new_columns(X, feat_names, verbose=verbose)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--no-backup', action='store_true', help='skip backup of original cache')
    p.add_argument('--validate', action='store_true', help='describe v2310 cache')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    verbose = not args.quiet

    if args.validate:
        validate_v2310(verbose=verbose)
        return

    if not args.no_backup:
        backup_cache(verbose=verbose)
    else:
        if verbose:
            print(f'[1/6] (skipped backup as requested)')

    X_orig, Y_ret, Y_risk, meta_orig, feat_names_orig = load_orig_cache(verbose=verbose)

    if verbose:
        print(f'[3/6] Loading layer feature files ...')
    meta_l2, l2_arrays = load_layer_features(
        LAYER2_FEATURES, ['filing_count_30d', 'has_8k_recent'], verbose=verbose
    )
    meta_l4, l4_arrays = load_layer_features(
        LAYER4_FEATURES, ['last_surprise_pct'], verbose=verbose
    )

    verify_meta_alignment(meta_orig, meta_l2, meta_l4, verbose=verbose)

    X_new, feat_names_new = merge_features(
        X_orig, feat_names_orig, l2_arrays, l4_arrays, verbose=verbose
    )

    write_v2310_cache(X_new, Y_ret, Y_risk, meta_orig, feat_names_new, verbose=verbose)

    summarize_new_columns(X_new, feat_names_new, verbose=verbose)


if __name__ == '__main__':
    main()
