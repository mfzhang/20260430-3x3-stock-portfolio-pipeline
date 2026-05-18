"""historical_earnings_scraper.py — v2.3.10 Sub-task B (yfinance fallback)

Fetches historical earnings (EPS estimate vs actual + announce date) from
yfinance for backtest-universe tickers and caches per-ticker JSON.

WHY yfinance: Alpha Vantage's free tier daily limit is only 25 calls (vs
the 526 ticker universe), making it impractical without a $50/mo premium
subscription. yfinance.Ticker.earnings_dates returns ~25 quarters per
ticker (~6 years), covering 2020-2026 — only 60% of our backtest window.
Pre-2020 snapshots will have last_surprise_pct = 0 (no signal).

If/when v2.3.11 confirms Layer 4 historical alignment is meaningful, the
parallel Alpha Vantage scraper code (preserved at
historical_earnings_scraper_alphavantage.py) can be activated under a
paid plan for full 30-year coverage.

Coverage trade-off:
  - yfinance free, ~5-10 min total fetch, 25 quarters / ticker
  - 2020-2026 (~60% of backtest timeline) covered with surprise signal
  - 2016-2020 (~40%) Layer 4 = 0 (no historical earnings data)
  - Decision rationale: documented in v2.3.10 instruction § Sub-task B

Design decisions (v2.3.10 §B-1..B-4):
  - B-1 source: yfinance.Ticker.earnings_dates (DataFrame: 'EPS Estimate',
                'Reported EPS', 'Surprise(%)' indexed by datetime).
  - B-2 cap: stored RAW; cap applied at compute time (Sub-task B.5).
  - B-3 same-day rule: documented; applied at compute time.
  - B-4 staleness: 120-day cutoff applied at compute time.

Usage:
  python historical_earnings_scraper.py --tickers AAPL MDT ABT     # dry-run
  python historical_earnings_scraper.py --validate                  # describe cache
  python historical_earnings_scraper.py --resume-stats              # show progress
  python historical_earnings_scraper.py                             # full universe

Output:
  results/historical_earnings/_meta.json
  results/historical_earnings/{TICKER}.json
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = 'results/historical_earnings'
META_PATH = os.path.join(CACHE_DIR, '_meta.json')
BACKTEST_CACHE_PATH = 'results/backtest_cache.npz'

# yfinance has internal rate limit but doesn't publish exact threshold.
# Empirically: 50ms between calls works for 100s of tickers. We use 100ms
# for safety (526 ticker × 100ms = 53s of throttle, plus ~0.3s/call latency
# = ~5min total).
SLEEP_BETWEEN_CALLS = 0.1

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 5.0


def get_backtest_universe(verbose=True):
    if not os.path.exists(BACKTEST_CACHE_PATH):
        raise FileNotFoundError(f'{BACKTEST_CACHE_PATH} not found')
    if verbose:
        print(f'      Reading {BACKTEST_CACHE_PATH} for ticker universe ...')
    data = np.load(BACKTEST_CACHE_PATH, allow_pickle=True)
    meta = data['meta']
    if meta.ndim != 2 or meta.shape[1] < 1:
        raise RuntimeError(f'Unexpected meta shape: {meta.shape}')
    tickers = sorted({str(row[0]).upper() for row in meta if row[0]})
    if verbose:
        print(f'      → {len(tickers):,} unique tickers')
    return tickers


def fetch_earnings_for_ticker(ticker):
    """Fetch earnings_dates DataFrame for one ticker.

    Returns list of dicts:
        [{'announce_date': '2024-08-01',
          'eps_estimate': 1.34,
          'eps_actual': 1.40,
          'eps_surprise_pct_raw': 4.36,    # NOT capped (B-2)
         }, ...]
    Sorted oldest-first. Empty list if no data.

    yfinance.Ticker.earnings_dates returns a DataFrame with columns:
        'EPS Estimate', 'Reported EPS', 'Surprise(%)'
    The Surprise(%) is in PERCENT units (3.59 = 3.59%), not fraction.
    """
    tk = yf.Ticker(ticker)
    df = tk.earnings_dates

    if df is None or len(df) == 0:
        return []

    # Column lookup (case-insensitive, robust to slight variants)
    cols = {c.lower().replace(' ', '').replace('_', '').replace('(', '').replace(')', ''): c
            for c in df.columns}
    estimate_col = cols.get('epsestimate')
    actual_col = cols.get('reportedeps') or cols.get('actualeps') or cols.get('epsactual')
    spct_col = cols.get('surprise%') or cols.get('surprisepercent') or cols.get('surprisepct')

    out = []
    for idx, row in df.iterrows():
        try:
            if hasattr(idx, 'strftime'):
                announce_date = idx.strftime('%Y-%m-%d')
            else:
                announce_date = pd.Timestamp(idx).strftime('%Y-%m-%d')
        except Exception:
            continue

        eps_estimate = _safe_float(row.get(estimate_col)) if estimate_col else None
        eps_actual = _safe_float(row.get(actual_col)) if actual_col else None
        eps_spct_raw = _safe_float(row.get(spct_col)) if spct_col else None

        # Skip future-only entries (no actual reported yet)
        if eps_actual is None:
            continue

        # Compute surprise from actual/estimate if not provided
        if eps_spct_raw is None and eps_actual is not None and eps_estimate is not None:
            if eps_estimate != 0:
                eps_spct_raw = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100.0
            else:
                eps_spct_raw = None

        out.append({
            'announce_date': announce_date,
            'eps_estimate': eps_estimate,
            'eps_actual': eps_actual,
            'eps_surprise_pct_raw': eps_spct_raw,    # NOT capped
        })

    out.sort(key=lambda x: x['announce_date'])
    return out


def _safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def list_already_fetched():
    if not os.path.isdir(CACHE_DIR):
        return set()
    out = set()
    for path in glob.glob(os.path.join(CACHE_DIR, '*.json')):
        if os.path.basename(path).startswith('_'):
            continue
        try:
            with open(path) as f:
                d = json.load(f)
            ticker = d.get('ticker') or os.path.basename(path).replace('.json', '')
            out.add(ticker.upper())
        except Exception:
            continue
    return out


def bulk_fetch(tickers, verbose=True, force_refetch=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    already = set() if force_refetch else list_already_fetched()
    if verbose and already:
        print(f'      → {len(already):,} tickers already cached, skipping')
    todo = [t for t in tickers if t not in already]
    if verbose:
        print(f'      → {len(todo):,} tickers to fetch')
    if not todo:
        return [], []

    success = []
    failures = []
    started = time.time()

    for i, ticker in enumerate(todo):
        attempt = 0
        last_exc = None
        records = None

        while attempt < MAX_RETRIES:
            try:
                records = fetch_earnings_for_ticker(ticker)
                break
            except Exception as e:
                ename = type(e).__name__
                msg = str(e)[:100]
                if 'rate' in msg.lower() or 'too many' in msg.lower() or 'YFRateLimit' in ename:
                    if verbose:
                        print(f'  [retry {attempt+1}/{MAX_RETRIES}] {ticker} rate-limit, backing off {RETRY_BACKOFF_SEC * (2**attempt):.0f}s')
                    time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
                    attempt += 1
                    last_exc = e
                else:
                    last_exc = e
                    break

        if records is None:
            failures.append({
                'ticker': ticker,
                'reason': type(last_exc).__name__ if last_exc else 'unknown',
                'detail': str(last_exc)[:200] if last_exc else '',
            })
        else:
            out_path = os.path.join(CACHE_DIR, f'{ticker}.json')
            with open(out_path, 'w') as f:
                json.dump({
                    'ticker': ticker,
                    'n_earnings': len(records),
                    'fetched_at': datetime.now(timezone.utc).isoformat(),
                    'source': 'yfinance.earnings_dates',
                    'earnings': records,
                }, f, indent=1)
            success.append({'ticker': ticker, 'n_earnings': len(records)})

        if i + 1 < len(todo):
            time.sleep(SLEEP_BETWEEN_CALLS)

        if verbose and (i + 1) % 50 == 0:
            elapsed_min = (time.time() - started) / 60.0
            print(f'  [{i+1}/{len(todo)}] success={len(success)} fail={len(failures)} '
                  f'elapsed={elapsed_min:.1f}min')

    elapsed_min = (time.time() - started) / 60.0
    if verbose:
        print(f'      → run complete: {len(success)} succeeded, {len(failures)} failed, '
              f'{elapsed_min:.1f} min')
    return success, failures


def show_resume_stats():
    universe = get_backtest_universe(verbose=False)
    cached = list_already_fetched()
    remaining = sorted(set(universe) - cached)
    print(f'  Universe size: {len(universe)}')
    print(f'  Already cached: {len(cached)}')
    print(f'  Remaining to fetch: {len(remaining)}')
    if remaining:
        print(f'  First 20 remaining: {remaining[:20]}')


def validate_cache(verbose=True):
    if not os.path.isdir(CACHE_DIR):
        print(f'  Cache directory {CACHE_DIR} does not exist')
        return
    files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.json')))
    files = [f for f in files if not os.path.basename(f).startswith('_')]
    print(f'  Cache directory: {CACHE_DIR}')
    print(f'  Per-ticker files: {len(files)}')
    if not files:
        return

    total_earnings = 0
    n_with_data = 0
    n_zero = 0
    earliest, latest = None, None
    n_estimate_zero_or_neg = 0
    surprise_pcts = []

    for path in files:
        with open(path) as fh:
            d = json.load(fh)
        n_e = d.get('n_earnings', 0)
        total_earnings += n_e
        if n_e > 0:
            n_with_data += 1
        else:
            n_zero += 1
        for entry in d.get('earnings', []):
            ad = entry['announce_date']
            if earliest is None or ad < earliest:
                earliest = ad
            if latest is None or ad > latest:
                latest = ad
            est = entry.get('eps_estimate')
            if est is not None and abs(est) < 0.05:
                n_estimate_zero_or_neg += 1
            sp = entry.get('eps_surprise_pct_raw')
            if sp is not None:
                surprise_pcts.append(sp)

    print(f'  Total earnings cached: {total_earnings:,}')
    print(f'  Tickers with earnings data: {n_with_data}')
    print(f'  Tickers with 0 earnings: {n_zero}')
    print(f'  Date range: {earliest} to {latest}')
    if surprise_pcts:
        sp_arr = np.array(surprise_pcts)
        print(f'  Surprise % distribution (raw, uncapped):')
        print(f'    n      : {len(sp_arr):,}')
        print(f'    mean   : {sp_arr.mean():+.2f}%')
        print(f'    median : {np.median(sp_arr):+.2f}%')
        print(f'    p1     : {np.percentile(sp_arr, 1):+.2f}%')
        print(f'    p99    : {np.percentile(sp_arr, 99):+.2f}%')
        print(f'    min    : {sp_arr.min():+.2f}%')
        print(f'    max    : {sp_arr.max():+.2f}%')
        print(f'    abs>50%: {(np.abs(sp_arr) > 50).sum():,} '
              f'({(np.abs(sp_arr) > 50).mean() * 100:.1f}%) '
              f'← capped at compute time')
    print(f'  Records with |estimate| < 0.05 (base-effect risk): {n_estimate_zero_or_neg:,}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', nargs='*', default=None,
                   help='specific tickers (e.g. AAPL MDT ABT) — for dry-run')
    p.add_argument('--validate', action='store_true', help='describe cache')
    p.add_argument('--resume-stats', action='store_true',
                   help='show what is cached vs what is remaining')
    p.add_argument('--force-refetch', action='store_true',
                   help='re-fetch even tickers already cached')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    verbose = not args.quiet

    if args.validate:
        validate_cache(verbose=verbose)
        return
    if args.resume_stats:
        show_resume_stats()
        return

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        if verbose:
            print(f'      Dry-run universe: {tickers}')
    else:
        tickers = get_backtest_universe(verbose=verbose)

    print(f'[1/2] Fetching earnings for {len(tickers)} ticker(s) ...')
    success, failures = bulk_fetch(
        tickers, verbose=verbose, force_refetch=args.force_refetch
    )

    meta = {
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'source': 'yfinance.earnings_dates',
        'n_tickers_attempted': len(tickers),
        'n_success': len(success),
        'n_failures': len(failures),
        'success': success,
        'failures': failures,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'[2/2] Wrote {META_PATH}')

    print()
    print('=' * 60)
    print(f'SUMMARY: {len(success)}/{len(tickers)} successful')
    if failures:
        print(f'Failures (first 10):')
        for fail in failures[:10]:
            print(f'  {fail["ticker"]:<6} {fail["reason"]:<25} {fail["detail"][:50]}')
    if success:
        n_avg = sum(s['n_earnings'] for s in success) / len(success)
        n_zero = sum(1 for s in success if s['n_earnings'] == 0)
        print(f'Avg earnings/ticker: {n_avg:.1f}')
        print(f'Tickers with 0 earnings: {n_zero}')
    print('=' * 60)


if __name__ == '__main__':
    main()
