#!/usr/bin/env python
"""Recompute per-fold SPY benchmark from each fold's actual snapshot distribution."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

STAGE2_DIR = Path('results/stage2/top1_trial58')
OUT_FILE = STAGE2_DIR / 'spy_benchmark.csv'
SUMMARY_FILE = STAGE2_DIR / 'summary.json'
FORWARD_DAYS = 63


def fetch_spy():
    print('Fetching SPY (max history)...')
    spy = yf.Ticker('SPY').history(period='max', auto_adjust=True)
    if spy is None or len(spy) == 0:
        raise RuntimeError('SPY fetch failed')
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    spy = spy[['Close']].sort_index().copy()
    spy['close_fwd'] = spy['Close'].shift(-FORWARD_DAYS)
    spy['ret_3m'] = (spy['close_fwd'] / spy['Close']) - 1
    spy = spy.dropna(subset=['ret_3m'])
    print(f'  {len(spy)} rows, {spy.index[0].date()} -> {spy.index[-1].date()}')
    return spy


def build_bucket_lookup(spy):
    # first trading day of each month -> SPY 3m forward return
    lookup = {}
    for ts, row in spy.iterrows():
        bucket = ts.strftime('%Y-%m')
        if bucket not in lookup:
            lookup[bucket] = float(row['ret_3m'])
    return lookup


def main():
    with open(SUMMARY_FILE) as f:
        summary = json.load(f)

    spy = fetch_spy()
    bucket_spy = build_bucket_lookup(spy)
    print(f'  Bucket lookup: {len(bucket_spy)} months '
          f'({min(bucket_spy)} -> {max(bucket_spy)})\n')

    rows = []
    for fold_info in summary['per_fold']:
        fold_id = fold_info['fold_id']
        snap_df = pd.read_csv(STAGE2_DIR / f'fold_{fold_id}' / 'per_snapshot_ranking.csv')
        bucket_counts = snap_df.groupby('bucket').size()

        # bucket-weighted SPY (weight = n_pairs per bucket)
        spy_vals = []
        weights = []
        missing = []
        for bucket, count in bucket_counts.items():
            if bucket in bucket_spy:
                spy_vals.append(bucket_spy[bucket])
                weights.append(count)
            else:
                missing.append(bucket)
        if missing:
            print(f'  Fold {fold_id}: missing SPY for {len(missing)} buckets '
                  f'({missing[:3]}{"..." if len(missing) > 3 else ""})')

        arr = np.array(spy_vals, dtype=float)
        w = np.array(weights, dtype=float)
        w_sum = float(w.sum())
        if w_sum > 0:
            spy_mean = float((arr * w).sum() / w_sum)
            spy_var = float(((arr - spy_mean) ** 2 * w).sum() / w_sum)
            spy_std = float(np.sqrt(spy_var))
        else:
            spy_mean = spy_std = np.nan

        sorted_buckets = sorted(snap_df['bucket'].unique())
        date_start = f'{sorted_buckets[0]}-01'
        date_end = f'{sorted_buckets[-1]}-28'

        full_rank = pd.read_csv(STAGE2_DIR / f'fold_{fold_id}' / 'full_ranking.csv')
        top5 = fold_info['top5']
        top5_actual = float(full_rank[full_rank['ticker'].isin(top5)]['actual_ret'].mean())
        alpha = top5_actual - spy_mean

        rows.append({
            'fold_id': fold_id,
            'fold_date_start': date_start,
            'fold_date_end': date_end,
            'spy_mean_3m_return': spy_mean,
            'spy_std_3m_return': spy_std,
            'spy_n_obs': int(w_sum),
            'top5_actual_3m': top5_actual,
            'alpha_vs_spy': alpha,
        })
        print(f'  Fold {fold_id}: SPY={spy_mean*100:+6.2f}%  '
              f'top5={top5_actual*100:+6.2f}%  alpha={alpha*100:+6.2f}%p  '
              f'({len(spy_vals)} months / {int(w_sum)} pairs)')

    out_df = pd.DataFrame(rows)

    if OUT_FILE.exists():
        backup = OUT_FILE.with_suffix('.csv.bak')
        OUT_FILE.rename(backup)
        print(f'\nBackup: {backup}')

    out_df.to_csv(OUT_FILE, index=False)
    print(f'New file: {OUT_FILE}\n')
    print(out_df.to_string(index=False))


if __name__ == '__main__':
    main()
