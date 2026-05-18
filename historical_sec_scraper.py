"""historical_sec_scraper.py — Track 1: metadata only

Fetches SEC EDGAR filings metadata for backtest-universe tickers and caches
to disk for later feature computation.

USAGE:
  python historical_sec_scraper.py --bootstrap-only          # only download CIK map, sanity check
  python historical_sec_scraper.py --tickers AAPL MDT        # dry-run for 2 tickers
  python historical_sec_scraper.py                           # full fetch for all tickers in backtest_cache
  python historical_sec_scraper.py --validate                # show summary of existing cache, no fetching

Output:
  results/historical_sec/_meta.json        # cik map, fetch timestamp, success/failure list
  results/historical_sec/{TICKER}.json     # one file per successful ticker

Rate limit: SEC allows 10 req/sec; we target 8 req/sec for safety margin.

Author: Ki Heon Lee (v2.3.10 / Apr-May 2026)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# SEC_USER_AGENT loaded from config.py (kept empty on GitHub for PII protection)
# SEC EDGAR fair-use policy requires real name + email — set locally before running.
try:
    import config
    SEC_USER_AGENT = config.SEC_USER_AGENT
except (ImportError, AttributeError):
    SEC_USER_AGENT = ''
if not SEC_USER_AGENT:
    raise RuntimeError(
        "SEC_USER_AGENT is empty. Set it in config.py to your "
        "'Real Name email@domain' before running (SEC EDGAR fair-use policy)."
    )
CIK_TICKER_URL = 'https://www.sec.gov/files/company_tickers.json'
SUBMISSIONS_URL_TEMPLATE = 'https://data.sec.gov/submissions/CIK{cik:010d}.json'
PAGINATED_URL_TEMPLATE = 'https://data.sec.gov/submissions/{filename}'

# Form types matching production sentiment.py Layer 2 logic.
# Include amendments (-/A suffix) — they're corrections to prior filings,
# but production sentiment.py likely counts them in filing_count_30d.
RELEVANT_FORMS = {'8-K', '10-Q', '10-K', '8-K/A', '10-Q/A', '10-K/A'}

# Manual CIK overrides for tickers where SEC's company_tickers.json maps
# to a non-canonical entity.
#
# v2.3.10 NOTE: BLK was investigated as a candidate (SEC default → 2012383
# only has 2024-02+ data). However, the alternative CIK 1364742 turned out
# to be "BlackRock Finance, Inc." — the financing subsidiary, NOT the
# operating company. Its 8-K/10-Q/10-K reflect debt activity, not asset-
# management business sentiment. No CIK in SEC's current API contains
# BlackRock's pre-2024 operating-company filings under ticker BLK.
#
# Conclusion: BLK historical SEC sentiment cannot be recovered via
# ticker→CIK mapping alone. This is a known limitation for ~250 of 122,257
# snapshots (0.2%), accepted for v2.3.10 scope.
#
# v2.3.11+ TODO: build historical CIK mapping via EDGAR full-text search
# or a curated reorganization registry to handle BLK and similar cases.
MANUAL_CIK_OVERRIDES = {
    # (empty — see note above)
}

# SEC rate limit: max 10 req/sec, we target 8 (~125ms interval) for safety.
REQ_PER_SEC = 8
MIN_REQ_INTERVAL = 1.0 / REQ_PER_SEC

# Paths
CACHE_DIR = 'results/historical_sec'
META_PATH = os.path.join(CACHE_DIR, '_meta.json')
BACKTEST_CACHE_PATH = 'results/backtest_cache.npz'

# Backoff on transient errors
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0


# ---------------------------------------------------------------------------
# Step 1: CIK mapping bootstrap
# ---------------------------------------------------------------------------

def fetch_cik_map(verbose=True):
    """Download SEC's ticker→CIK mapping. Returns dict {TICKER: int(cik)}.

    Applies MANUAL_CIK_OVERRIDES for known reorganization mismatches
    (e.g., BLK 2024 holding restructure).
    """
    headers = {'User-Agent': SEC_USER_AGENT}
    if verbose:
        print(f'[1/N] Fetching CIK map from {CIK_TICKER_URL} ...')
    resp = requests.get(CIK_TICKER_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # SEC's response is dict-of-dicts: {"0": {cik_str, ticker, title}, "1": {...}, ...}
    cik_map = {}
    for entry in data.values():
        ticker = entry['ticker'].upper()
        cik = int(entry['cik_str'])
        cik_map[ticker] = cik
    # Apply manual overrides (after default mapping built)
    n_overridden = 0
    for ticker, override_cik in MANUAL_CIK_OVERRIDES.items():
        if ticker in cik_map and cik_map[ticker] != override_cik:
            old_cik = cik_map[ticker]
            cik_map[ticker] = override_cik
            n_overridden += 1
            if verbose:
                print(f'      Manual override: {ticker} CIK {old_cik} → {override_cik}')
    if verbose:
        print(f'      → {len(cik_map):,} tickers in SEC CIK map'
              f'{" (" + str(n_overridden) + " overridden)" if n_overridden else ""}')
    return cik_map


# ---------------------------------------------------------------------------
# Step 2: Per-ticker fetch
# ---------------------------------------------------------------------------

def _extract_filings_from_block(forms, dates, accessions, primary_docs):
    """Helper: filter a (forms, dates, ...) block to RELEVANT_FORMS only."""
    out = []
    for i, form in enumerate(forms):
        if form in RELEVANT_FORMS:
            out.append({
                'form': form,
                'filing_date': dates[i],
                'accession_number': accessions[i] if i < len(accessions) else '',
                'primary_doc': primary_docs[i] if i < len(primary_docs) else '',
            })
    return out


def _fetch_paginated_block(filename, headers):
    """Fetch one paginated submissions JSON and return forms/dates/accessions/primary_docs.

    Paginated files have a flat structure: {form: [...], filingDate: [...], ...}
    (no 'recent' nesting like the main submissions JSON).
    """
    url = PAGINATED_URL_TEMPLATE.format(filename=filename)
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return (
        data.get('form', []),
        data.get('filingDate', []),
        data.get('accessionNumber', []),
        data.get('primaryDocument', []),
    )


def fetch_filings_for_ticker(ticker, cik, min_paginated_date='2016-01-01'):
    """Fetch all relevant filings metadata for one ticker, including pagination.

    Strategy:
      1. Fetch main submissions JSON → get inline `recent` block
      2. Inspect `filings.files[]` for older paginated blocks
      3. For each paginated file whose [filingFrom, filingTo] range overlaps
         our backtest window (>= min_paginated_date), fetch and merge.
      4. Skip paginated files entirely older than min_paginated_date (we
         don't need 1994-2006 data for a 2016-2026 backtest).

    Returns:
        filings: list of {'form', 'filing_date', 'accession_number', 'primary_doc'}
                 filtered to RELEVANT_FORMS
        truncation_warning: True if pagination was needed (informational only;
                            with this fix, all relevant filings are recovered).
    """
    headers = {'User-Agent': SEC_USER_AGENT}

    # Step 1: main submissions JSON
    url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data.get('filings', {}).get('recent', {})
    forms = recent.get('form', [])
    dates = recent.get('filingDate', [])
    accessions = recent.get('accessionNumber', [])
    primary_docs = recent.get('primaryDocument', [])

    filings = _extract_filings_from_block(forms, dates, accessions, primary_docs)
    n_inline = len(forms)

    # Step 2: paginated older blocks
    files_meta = data.get('filings', {}).get('files', [])
    paginated_used = []
    for fi in files_meta:
        filing_to = fi.get('filingTo', '')   # most recent date in this file
        # If the entire file is older than our window, skip
        if filing_to and filing_to < min_paginated_date:
            continue
        # Fetch this paginated block
        # Rate-limit: brief sleep before paginated request
        time.sleep(MIN_REQ_INTERVAL)
        try:
            p_forms, p_dates, p_accessions, p_primary = _fetch_paginated_block(
                fi['name'], headers
            )
            paginated_filings = _extract_filings_from_block(
                p_forms, p_dates, p_accessions, p_primary
            )
            filings.extend(paginated_filings)
            paginated_used.append(fi['name'])
        except Exception as e:
            # Paginated fetch failure is non-fatal — log inline as note
            print(f'    WARN: pagination fetch failed for {ticker} ({fi["name"]}): {e}')

    # Truncation flag now means: we needed pagination to recover full history
    truncation_warning = len(paginated_used) > 0

    return filings, truncation_warning


# ---------------------------------------------------------------------------
# Step 3: Bulk fetch with rate limiting and retry
# ---------------------------------------------------------------------------

def bulk_fetch(tickers, cik_map, verbose=True):
    """Iterate tickers, fetch each, write per-ticker JSON cache.

    Returns:
        success: list of {ticker, n_filings, truncated}
        failures: list of {ticker, reason, detail}
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    success = []
    failures = []
    last_request_time = 0.0
    started = time.time()

    for i, ticker in enumerate(tickers):
        cik = cik_map.get(ticker)
        if cik is None:
            failures.append({'ticker': ticker, 'reason': 'no_cik', 'detail': 'ticker not in SEC CIK map'})
            continue

        # Retry loop with exponential backoff on transient errors
        attempt = 0
        last_exc = None
        while attempt < MAX_RETRIES:
            # Rate limit (between requests)
            elapsed = time.time() - last_request_time
            if elapsed < MIN_REQ_INTERVAL:
                time.sleep(MIN_REQ_INTERVAL - elapsed)
            try:
                filings, truncated = fetch_filings_for_ticker(ticker, cik)
                last_request_time = time.time()
                # Write per-ticker cache
                out_path = os.path.join(CACHE_DIR, f'{ticker}.json')
                with open(out_path, 'w') as f:
                    json.dump({
                        'ticker': ticker,
                        'cik': cik,
                        'n_filings': len(filings),
                        'truncated': truncated,
                        'fetched_at': datetime.now(timezone.utc).isoformat(),
                        'filings': filings,
                    }, f, indent=1)
                success.append({'ticker': ticker, 'n_filings': len(filings), 'truncated': truncated})
                break
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else None
                last_exc = e
                if code == 404:
                    # 404 is permanent (CIK exists but no submissions data) — don't retry
                    failures.append({'ticker': ticker, 'reason': 'http_404', 'detail': f'cik={cik}'})
                    last_exc = None
                    break
                if code == 429:
                    # rate limit — back off harder
                    if verbose:
                        print(f'  RATE LIMIT on {ticker} (attempt {attempt+1}/{MAX_RETRIES}) — backing off')
                    time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
                else:
                    if verbose:
                        print(f'  HTTP {code} on {ticker} (attempt {attempt+1}/{MAX_RETRIES})')
                    time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
                attempt += 1
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                if verbose:
                    print(f'  NETWORK error on {ticker} (attempt {attempt+1}/{MAX_RETRIES}): {type(e).__name__}')
                time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
                attempt += 1
            except Exception as e:
                # JSON parse error or unexpected — don't retry
                last_exc = e
                break
        else:
            # Loop exhausted without break — all retries failed
            pass

        if last_exc is not None and (not success or success[-1]['ticker'] != ticker):
            failures.append({
                'ticker': ticker,
                'reason': type(last_exc).__name__,
                'detail': str(last_exc)[:200],
            })

        if verbose and (i + 1) % 50 == 0:
            elapsed_min = (time.time() - started) / 60.0
            print(f'  [{i+1}/{len(tickers)}] processed  '
                  f'(success={len(success)}, fail={len(failures)}, '
                  f'elapsed={elapsed_min:.1f} min)')

    elapsed_min = (time.time() - started) / 60.0
    if verbose:
        print(f'      → {len(success)}/{len(tickers)} succeeded, '
              f'{len(failures)} failed, total {elapsed_min:.1f} min')
    return success, failures


# ---------------------------------------------------------------------------
# Step 4: Discover universe of tickers from backtest_cache
# ---------------------------------------------------------------------------

def get_backtest_universe(verbose=True):
    """Load backtest_cache.npz and extract unique tickers from `meta`.

    Empirically (v2.3.10 diagnosis): `meta` is a (N, 3) object ndarray
    where each row is [ticker_str, snap_idx_int, date_str].
    We extract column 0 and dedupe.
    """
    if not os.path.exists(BACKTEST_CACHE_PATH):
        raise FileNotFoundError(f'{BACKTEST_CACHE_PATH} not found — run backtest.py first')
    if verbose:
        print(f'      Reading {BACKTEST_CACHE_PATH} for ticker universe ...')
    data = np.load(BACKTEST_CACHE_PATH, allow_pickle=True)
    meta = data['meta']
    # meta shape: (N, 3) — columns are [ticker, snap_idx, date_str]
    if meta.ndim == 2 and meta.shape[1] >= 1:
        tickers = sorted({str(row[0]).upper() for row in meta if row[0]})
    else:
        raise RuntimeError(
            f'Unexpected meta shape: {meta.shape}, dtype: {meta.dtype}. '
            f'Expected 2D ndarray with ticker in column 0.'
        )
    if verbose:
        print(f'      → {len(tickers):,} unique tickers in backtest_cache')
    return tickers


# ---------------------------------------------------------------------------
# Validation (read existing cache, summarize)
# ---------------------------------------------------------------------------

def validate_cache(verbose=True):
    """Summarize what's currently in CACHE_DIR. No fetching."""
    if not os.path.isdir(CACHE_DIR):
        print(f'  Cache directory {CACHE_DIR} does not exist')
        return
    files = sorted([f for f in os.listdir(CACHE_DIR) if f.endswith('.json') and not f.startswith('_')])
    print(f'  Cache directory: {CACHE_DIR}')
    print(f'  Per-ticker files: {len(files)}')
    if not files:
        return
    # Summary stats
    total_filings = 0
    form_counts = {}
    earliest_date, latest_date = None, None
    truncated_tickers = []
    for f in files:
        with open(os.path.join(CACHE_DIR, f)) as fh:
            d = json.load(fh)
        total_filings += d.get('n_filings', 0)
        if d.get('truncated'):
            truncated_tickers.append(d['ticker'])
        for filing in d.get('filings', []):
            form = filing['form']
            form_counts[form] = form_counts.get(form, 0) + 1
            fd = filing['filing_date']
            if earliest_date is None or fd < earliest_date:
                earliest_date = fd
            if latest_date is None or fd > latest_date:
                latest_date = fd
    print(f'  Total filings cached: {total_filings:,}')
    print(f'  Date range: {earliest_date} to {latest_date}')
    print(f'  Form-type breakdown:')
    for form in sorted(form_counts.keys()):
        print(f'    {form:<8} {form_counts[form]:>6,}')
    if truncated_tickers:
        print(f'  Truncation warning: {len(truncated_tickers)} ticker(s) hit 1000-filing inline cap:')
        print(f'    {truncated_tickers[:10]}{"..." if len(truncated_tickers) > 10 else ""}')
    # Meta file
    if os.path.exists(META_PATH):
        with open(META_PATH) as fh:
            m = json.load(fh)
        print(f'  Last fetch: {m.get("fetched_at", "n/a")}')
        print(f'  Failures: {len(m.get("failures", []))}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bootstrap-only', action='store_true',
                   help='only fetch CIK map, do not download per-ticker filings')
    p.add_argument('--tickers', nargs='*', default=None,
                   help='specific tickers (e.g. AAPL MDT) — for dry-run; '
                        'default: all tickers in backtest_cache')
    p.add_argument('--validate', action='store_true',
                   help='summarize existing cache, do not fetch')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    verbose = not args.quiet

    if args.validate:
        validate_cache(verbose=verbose)
        return

    # Bootstrap CIK map (always done before any fetch)
    cik_map = fetch_cik_map(verbose=verbose)
    if args.bootstrap_only:
        # Just sanity-check a few well-known tickers
        for t in ['AAPL', 'MSFT', 'NVDA', 'MDT']:
            cik = cik_map.get(t, 'NOT FOUND')
            print(f'  {t:<6} → CIK {cik}')
        return

    # Determine ticker list
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        if verbose:
            print(f'      Dry-run universe: {tickers}')
    else:
        tickers = get_backtest_universe(verbose=verbose)

    # Bulk fetch
    print(f'[2/N] Fetching filings for {len(tickers)} ticker(s) at ~{REQ_PER_SEC} req/sec ...')
    success, failures = bulk_fetch(tickers, cik_map, verbose=verbose)

    # Write meta
    meta = {
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'user_agent': SEC_USER_AGENT,
        'n_tickers_attempted': len(tickers),
        'n_success': len(success),
        'n_failures': len(failures),
        'success': success,
        'failures': failures,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'[3/N] Wrote {META_PATH}')

    # Quick summary
    print()
    print('=' * 60)
    print(f'SUMMARY: {len(success)}/{len(tickers)} successful')
    if failures:
        print(f'Failures (first 10):')
        for fail in failures[:10]:
            print(f'  {fail["ticker"]:<6} {fail["reason"]:<20} {fail["detail"][:50]}')
        if len(failures) > 10:
            print(f'  ... ({len(failures) - 10} more in {META_PATH})')
    if success:
        n_filings_avg = sum(s['n_filings'] for s in success) / len(success)
        n_truncated = sum(1 for s in success if s.get('truncated'))
        print(f'Avg filings/ticker: {n_filings_avg:.1f}')
        if n_truncated:
            print(f'Truncated (≥1000 filings): {n_truncated} tickers')
    print('=' * 60)
    print(f'Run `python {sys.argv[0]} --validate` for full breakdown')


if __name__ == '__main__':
    main()
