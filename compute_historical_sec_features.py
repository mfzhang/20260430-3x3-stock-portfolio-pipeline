"""compute_historical_sec_features.py — v2.3.10 Sub-task A.5

Reads:
  - results/backtest_cache.npz                 (meta: 122,257 × 3 [ticker, snap_idx, date_str])
  - results/historical_sec/{TICKER}.json       (519 per-ticker filings caches)

Computes per-snapshot Layer 2 features (Track 1: metadata only):
  - filing_count_30d   int   : # of 8-K/10-Q/10-K (incl. /A) in [date-30d, date]
  - has_8k_recent      0/1   : any 8-K in [date-30d, date]

Writes:
  - results/historical_sec_features.npz
        keys: filing_count_30d, has_8k_recent, meta (= same as backtest_cache)

Sanity benchmarks (vs production sentiment.py per v2.3.9 csv diagnostic):
  - filing_count_30d:  ~31% of CURRENT-snapshot tickers were 0 (no filings in
    last 30 days), max=20. Historical distribution should be similar but
    averaged over 10 years.
  - has_8k_recent:     should be nonzero for most snapshots (8-K is the
    most-common filing form: 116K of 155K = 75% of cached filings).

Usage:
  python compute_historical_sec_features.py             # full compute
  python compute_historical_sec_features.py --dry-run   # 100 snapshots only
  python compute_historical_sec_features.py --validate  # describe existing output
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np

CACHE_PATH = 'results/backtest_cache.npz'
SEC_CACHE_DIR = 'results/historical_sec'
OUTPUT_PATH = 'results/historical_sec_features.npz'

WINDOW_DAYS = 30


def load_filings_by_ticker(verbose=True):
    """Load all per-ticker SEC JSON caches.

    Returns dict {TICKER: sorted list of (date_obj, form_str)}.
    Pre-sorted by date for binary-search-friendly windowing later.
    """
    if verbose:
        print(f'[1/4] Loading filings from {SEC_CACHE_DIR}/ ...')
    out = {}
    n_files_loaded = 0
    n_filings_total = 0
    for path in sorted(glob.glob(os.path.join(SEC_CACHE_DIR, '*.json'))):
        if os.path.basename(path).startswith('_'):
            continue
        with open(path) as f:
            d = json.load(f)
        ticker = d['ticker']
        # parse and sort once per ticker
        records = []
        for entry in d.get('filings', []):
            fd = entry['filing_date']     # 'YYYY-MM-DD'
            form = entry['form']
            try:
                dt = datetime.strptime(fd, '%Y-%m-%d').date()
            except ValueError:
                continue
            records.append((dt, form))
        records.sort()
        out[ticker] = records
        n_files_loaded += 1
        n_filings_total += len(records)
    if verbose:
        print(f'      → {n_files_loaded:,} ticker JSONs, {n_filings_total:,} total filings')
    return out


def load_backtest_meta(verbose=True):
    """Returns the meta array (N, 3) from backtest_cache.npz."""
    if verbose:
        print(f'[2/4] Loading backtest_cache meta from {CACHE_PATH} ...')
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f'{CACHE_PATH} missing — run backtest.py first')
    data = np.load(CACHE_PATH, allow_pickle=True)
    meta = data['meta']
    if meta.ndim != 2 or meta.shape[1] < 3:
        raise RuntimeError(f'Unexpected meta shape {meta.shape}')
    if verbose:
        print(f'      → meta shape: {meta.shape}, '
              f'first row: {list(meta[0])}, last row: {list(meta[-1])}')
    return meta


def compute_features(meta, filings_by_ticker, dry_run_n=None, verbose=True):
    """Iterate snapshots and compute per-row features.

    Args:
        meta: (N, 3) ndarray with [ticker, snap_idx, date_str]
        filings_by_ticker: dict {TICKER: sorted [(date, form), ...]}
        dry_run_n: if not None, only process first N snapshots
        verbose: progress printing

    Returns:
        filing_count_30d: (N,) int32
        has_8k_recent:    (N,) int8
        skipped_log: list of dicts describing snapshots without ticker SEC cache
    """
    n_total = len(meta) if dry_run_n is None else min(dry_run_n, len(meta))
    if verbose:
        scope = 'dry-run' if dry_run_n else 'full'
        print(f'[3/4] Computing features ({scope}: {n_total:,} snapshots) ...')

    fc = np.zeros(n_total, dtype=np.int32)
    h8 = np.zeros(n_total, dtype=np.int8)
    skipped_tickers = Counter()
    snapshots_per_ticker = Counter()
    started = time.time()

    # Pre-extract date arrays per ticker for fast windowing via np.searchsorted
    # (much faster than Python list comprehension for 122K snapshots)
    ticker_dates = {}     # TICKER -> ndarray of np.datetime64
    ticker_forms = {}     # TICKER -> list of form strings (parallel to dates)
    for tk, records in filings_by_ticker.items():
        if not records:
            continue
        dates = np.array([np.datetime64(r[0]) for r in records])
        forms = [r[1] for r in records]
        ticker_dates[tk] = dates
        ticker_forms[tk] = forms

    for i in range(n_total):
        row = meta[i]
        ticker = str(row[0]).upper()
        date_str = str(row[2])
        snapshots_per_ticker[ticker] += 1

        if ticker not in ticker_dates:
            skipped_tickers[ticker] += 1
            continue   # fc[i]=0, h8[i]=0 (default)

        try:
            snap_date = np.datetime64(date_str)
        except Exception:
            skipped_tickers[f'{ticker}(bad_date)'] += 1
            continue

        window_start = snap_date - np.timedelta64(WINDOW_DAYS, 'D')
        dates = ticker_dates[ticker]

        # Find indices [window_start, snap_date] inclusive
        # Note: we use right-inclusive on snap_date because production sentiment
        # at snapshot_date includes filings filed THAT day.
        i_lo = np.searchsorted(dates, window_start, side='left')
        i_hi = np.searchsorted(dates, snap_date, side='right')

        if i_hi > i_lo:
            fc[i] = i_hi - i_lo
            forms = ticker_forms[ticker]
            for j in range(i_lo, i_hi):
                f = forms[j]
                if f == '8-K' or f == '8-K/A':
                    h8[i] = 1
                    break

        if verbose and (i + 1) % 20000 == 0:
            elapsed = time.time() - started
            rate = (i + 1) / elapsed
            print(f'      [{i+1:,}/{n_total:,}] processed  '
                  f'({rate:,.0f} rows/sec, {elapsed:.1f}s elapsed)')

    elapsed = time.time() - started
    if verbose:
        print(f'      → done in {elapsed:.1f}s '
              f'({n_total/elapsed:,.0f} rows/sec)')

    skipped_log = sorted(skipped_tickers.items(), key=lambda x: -x[1])
    return fc, h8, skipped_log, snapshots_per_ticker


def summarize_features(fc, h8, skipped_log, snapshots_per_ticker, verbose=True):
    """Print sanity stats."""
    print()
    print('=' * 60)
    print('FEATURE STATISTICS')
    print('=' * 60)
    n = len(fc)
    print(f'Snapshots processed: {n:,}')
    print()
    print('filing_count_30d:')
    print(f'  zero:    {(fc == 0).sum():>7,} ({(fc == 0).mean() * 100:.1f}%)')
    print(f'  nonzero: {(fc > 0).sum():>7,} ({(fc > 0).mean() * 100:.1f}%)')
    print(f'  mean:    {fc.mean():.2f}')
    print(f'  std:     {fc.std():.2f}')
    print(f'  median:  {np.median(fc):.0f}')
    print(f'  max:     {fc.max()}')
    print(f'  p99:     {np.percentile(fc, 99):.0f}')
    print()
    print('has_8k_recent:')
    print(f'  zero:    {(h8 == 0).sum():>7,} ({(h8 == 0).mean() * 100:.1f}%)')
    print(f'  one:     {(h8 == 1).sum():>7,} ({(h8 == 1).mean() * 100:.1f}%)')
    print()
    if skipped_log:
        n_skipped = sum(c for _, c in skipped_log)
        print(f'Snapshots with no SEC cache ticker match: {n_skipped:,} '
              f'({n_skipped / n * 100:.2f}%)')
        print(f'  Top 10 missing tickers:')
        for tk, c in skipped_log[:10]:
            print(f'    {tk:<15} {c:>5} snapshots')
    print('=' * 60)


def validate_existing(verbose=True):
    """Read existing OUTPUT_PATH and summarize."""
    if not os.path.exists(OUTPUT_PATH):
        print(f'  {OUTPUT_PATH} not found')
        return
    data = np.load(OUTPUT_PATH, allow_pickle=True)
    fc = data['filing_count_30d']
    h8 = data['has_8k_recent']
    print(f'  Output: {OUTPUT_PATH}')
    print(f'  filing_count_30d shape: {fc.shape}, dtype: {fc.dtype}')
    print(f'  has_8k_recent  shape: {h8.shape}, dtype: {h8.dtype}')
    summarize_features(fc, h8, [], None, verbose=False)


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

    filings = load_filings_by_ticker(verbose=verbose)
    meta = load_backtest_meta(verbose=verbose)

    dry_run_n = 100 if args.dry_run else None
    fc, h8, skipped_log, sp = compute_features(meta, filings, dry_run_n=dry_run_n, verbose=verbose)

    summarize_features(fc, h8, skipped_log, sp, verbose=verbose)

    # Write output
    if dry_run_n is None:
        print()
        print(f'[4/4] Writing {OUTPUT_PATH} ...')
        np.savez(OUTPUT_PATH,
                 filing_count_30d=fc,
                 has_8k_recent=h8,
                 meta=meta)
        size_kb = os.path.getsize(OUTPUT_PATH) / 1024
        print(f'      → {size_kb:,.1f} KB written')
    else:
        print()
        print(f'(dry-run; not writing {OUTPUT_PATH})')


if __name__ == '__main__':
    main()
