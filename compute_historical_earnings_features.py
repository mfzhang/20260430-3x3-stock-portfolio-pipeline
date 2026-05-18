"""compute_historical_earnings_features.py — v2.3.10 Sub-task B.5

Reads:
  - results/backtest_cache.npz                         (meta: 122,257 × 3)
  - results/historical_earnings/{TICKER}.json         (yfinance cache)

Computes per-snapshot Layer 4 feature:
  - last_surprise_pct  float32 : capped (±50%) surprise % from MOST-RECENT
                                 earnings before snapshot date.
                                 0.0 if no earnings within 120-day staleness
                                 window or no estimate available.

Design decisions (v2.3.10 §B-1..B-4):
  - B-1 source: yfinance.Ticker.earnings_dates (25 quarters, ~6 years)
                Pre-2020 snapshots will have last_surprise_pct = 0
                (no historical earnings available from yfinance).
  - B-2 cap: ±50% applied here (cache stores raw)
  - B-3 same-day rule: announce_date <= snapshot_date is INCLUSIVE.
                       Production model uses snapshot-day close prices, so
                       announcements on the same day are in info set.
  - B-4 staleness: surprise discarded if >120 days old. ~90-day quarter
                   period + ~30-day announcement lag = 120-day cap.

Writes:
  - results/historical_earnings_features.npz
        keys: last_surprise_pct, meta

Usage:
  python compute_historical_earnings_features.py             # full compute
  python compute_historical_earnings_features.py --dry-run   # 100 snapshots
  python compute_historical_earnings_features.py --validate  # describe output
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np

CACHE_PATH = 'results/backtest_cache.npz'
EARNINGS_CACHE_DIR = 'results/historical_earnings'
OUTPUT_PATH = 'results/historical_earnings_features.npz'

# B-4 staleness: 120 days
STALENESS_DAYS = 120

# B-2 cap: ±50% (matches sentiment.py v2.3.2 fix)
SURPRISE_CAP = 50.0


def load_earnings_by_ticker(verbose=True):
    """Load all per-ticker earnings JSON.

    Returns (ticker_dates, ticker_pcts) two dicts:
        ticker_dates[T] = ndarray[np.datetime64] of announce dates, sorted
        ticker_pcts[T]  = ndarray[float32] of capped surprise %, parallel
    """
    if verbose:
        print(f'[1/4] Loading earnings from {EARNINGS_CACHE_DIR}/ ...')

    out_dates = {}
    out_pcts = {}
    n_files = 0
    n_records = 0
    n_capped = 0
    n_dropped_nan = 0

    for path in sorted(glob.glob(os.path.join(EARNINGS_CACHE_DIR, '*.json'))):
        if os.path.basename(path).startswith('_'):
            continue
        with open(path) as f:
            d = json.load(f)
        ticker = d['ticker']
        records = d.get('earnings', [])
        if not records:
            continue

        dates = []
        pcts = []
        for entry in records:
            ad = entry.get('announce_date')
            sp_raw = entry.get('eps_surprise_pct_raw')
            if not ad:
                continue
            try:
                dt = np.datetime64(ad)
            except Exception:
                continue

            # B-2: cap at compute time
            if sp_raw is None:
                # No surprise data → set to 0 (matches production sentiment.py default)
                sp_capped = 0.0
                n_dropped_nan += 1
            else:
                if abs(sp_raw) > SURPRISE_CAP:
                    n_capped += 1
                sp_capped = max(-SURPRISE_CAP, min(SURPRISE_CAP, sp_raw))

            dates.append(dt)
            pcts.append(sp_capped)

        if dates:
            out_dates[ticker] = np.array(dates)
            out_pcts[ticker] = np.array(pcts, dtype=np.float32)
            n_files += 1
            n_records += len(dates)

    if verbose:
        print(f'      → {n_files:,} ticker JSONs, {n_records:,} earnings total')
        print(f'      → {n_capped:,} capped at ±{SURPRISE_CAP}% '
              f'({n_capped / max(n_records, 1) * 100:.1f}%)')
        print(f'      → {n_dropped_nan:,} records had null surprise (set to 0)')

    return out_dates, out_pcts


def load_backtest_meta(verbose=True):
    if verbose:
        print(f'[2/4] Loading backtest_cache meta ...')
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f'{CACHE_PATH} missing')
    data = np.load(CACHE_PATH, allow_pickle=True)
    meta = data['meta']
    if meta.ndim != 2 or meta.shape[1] < 3:
        raise RuntimeError(f'Unexpected meta shape {meta.shape}')
    if verbose:
        print(f'      → meta shape: {meta.shape}')
    return meta


def compute_features(meta, ticker_dates, ticker_pcts, dry_run_n=None, verbose=True):
    """For each snapshot:
      1. Find most-recent earnings with announce_date <= snap_date (B-3)
      2. If lag > STALENESS_DAYS: feature = 0 (B-4)
      3. Else: feature = capped surprise %
    """
    n_total = len(meta) if dry_run_n is None else min(dry_run_n, len(meta))
    if verbose:
        scope = 'dry-run' if dry_run_n else 'full'
        print(f'[3/4] Computing features ({scope}: {n_total:,} snapshots) ...')

    last_surprise_pct = np.zeros(n_total, dtype=np.float32)
    skipped_tickers = Counter()
    matched = 0
    stale_dropped = 0
    no_history = 0    # snapshot pre-dates first available earnings (mostly 2016-2020)
    started = time.time()

    staleness_td = np.timedelta64(STALENESS_DAYS, 'D')

    for i in range(n_total):
        row = meta[i]
        ticker = str(row[0]).upper()
        date_str = str(row[2])

        if ticker not in ticker_dates:
            skipped_tickers[ticker] += 1
            continue   # default 0

        try:
            snap_date = np.datetime64(date_str)
        except Exception:
            skipped_tickers[f'{ticker}(bad_date)'] += 1
            continue

        dates = ticker_dates[ticker]
        pcts = ticker_pcts[ticker]

        # B-3: most-recent earnings on or before snapshot_date.
        # searchsorted(side='right') returns insertion point AFTER any equal dates
        idx = np.searchsorted(dates, snap_date, side='right') - 1

        if idx < 0:
            # No earnings before snapshot date — pre-2020 case w/ yfinance,
            # or true pre-IPO snapshot
            no_history += 1
            continue

        recent_date = dates[idx]
        lag = snap_date - recent_date
        if lag > staleness_td:
            stale_dropped += 1
            continue

        last_surprise_pct[i] = pcts[idx]
        matched += 1

        if verbose and (i + 1) % 20000 == 0:
            elapsed = time.time() - started
            rate = (i + 1) / max(elapsed, 0.001)
            print(f'      [{i+1:,}/{n_total:,}] '
                  f'matched={matched:,} stale={stale_dropped:,} '
                  f'no_history={no_history:,} ({rate:,.0f} rows/sec)')

    elapsed = time.time() - started
    if verbose:
        print(f'      → done in {elapsed:.1f}s')

    skipped_log = sorted(skipped_tickers.items(), key=lambda x: -x[1])
    return last_surprise_pct, matched, stale_dropped, no_history, skipped_log


def summarize_features(last_surprise_pct, matched, stale_dropped, no_history, skipped_log):
    print()
    print('=' * 60)
    print('FEATURE STATISTICS')
    print('=' * 60)
    n = len(last_surprise_pct)
    print(f'Snapshots processed: {n:,}')
    print()
    n_skipped_total = sum(c for _, c in skipped_log)
    print('Snapshot disposition:')
    print(f'  matched (<=120d lag):    {matched:>7,} ({matched / n * 100:.1f}%)')
    print(f'  stale (>120d lag):       {stale_dropped:>7,} ({stale_dropped / n * 100:.1f}%)')
    print(f'  no earnings before snap: {no_history:>7,} ({no_history / n * 100:.1f}%)')
    print(f'  no ticker in cache:      {n_skipped_total:>7,} ({n_skipped_total / n * 100:.1f}%)')
    print()

    nonzero = last_surprise_pct != 0
    print('last_surprise_pct distribution:')
    print(f'  zero:    {(~nonzero).sum():>7,} ({(~nonzero).mean() * 100:.1f}%)')
    print(f'  nonzero: {nonzero.sum():>7,} ({nonzero.mean() * 100:.1f}%)')
    if nonzero.any():
        nz = last_surprise_pct[nonzero]
        print(f'  mean   : {nz.mean():+.2f}%')
        print(f'  median : {np.median(nz):+.2f}%')
        print(f'  p1     : {np.percentile(nz, 1):+.2f}%')
        print(f'  p5     : {np.percentile(nz, 5):+.2f}%')
        print(f'  p95    : {np.percentile(nz, 95):+.2f}%')
        print(f'  p99    : {np.percentile(nz, 99):+.2f}%')
        print(f'  min    : {nz.min():+.2f}%')
        print(f'  max    : {nz.max():+.2f}%')
        print(f'  at cap : {(np.abs(nz) >= SURPRISE_CAP - 0.001).sum():,} '
              f'({(np.abs(nz) >= SURPRISE_CAP - 0.001).mean() * 100:.1f}% of nonzero)')

    if skipped_log:
        print()
        print(f'Top 10 missing tickers (no earnings cache):')
        for tk, c in skipped_log[:10]:
            print(f'  {tk:<15} {c:>5} snapshots')
    print('=' * 60)


def validate_existing(verbose=True):
    if not os.path.exists(OUTPUT_PATH):
        print(f'  {OUTPUT_PATH} not found')
        return
    data = np.load(OUTPUT_PATH, allow_pickle=True)
    lsp = data['last_surprise_pct']
    print(f'  Output: {OUTPUT_PATH}')
    print(f'  last_surprise_pct shape: {lsp.shape}, dtype: {lsp.dtype}')
    summarize_features(lsp, 0, 0, 0, [])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', help='process only 100 snapshots')
    p.add_argument('--validate', action='store_true', help='describe existing output')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    verbose = not args.quiet

    if args.validate:
        validate_existing(verbose=verbose)
        return

    ticker_dates, ticker_pcts = load_earnings_by_ticker(verbose=verbose)
    meta = load_backtest_meta(verbose=verbose)

    dry_run_n = 100 if args.dry_run else None
    last_surprise_pct, matched, stale, no_hist, skipped = compute_features(
        meta, ticker_dates, ticker_pcts, dry_run_n=dry_run_n, verbose=verbose
    )

    summarize_features(last_surprise_pct, matched, stale, no_hist, skipped)

    if dry_run_n is None:
        print()
        print(f'[4/4] Writing {OUTPUT_PATH} ...')
        np.savez(OUTPUT_PATH, last_surprise_pct=last_surprise_pct, meta=meta)
        size_kb = os.path.getsize(OUTPUT_PATH) / 1024
        print(f'      → {size_kb:,.1f} KB written')
    else:
        print()
        print(f'(dry-run; not writing {OUTPUT_PATH})')


if __name__ == '__main__':
    main()
