#!/usr/bin/env python3
"""
prep_stage2_sentiment.py — collect 4-layer sentiment for backtest universe

Usage:
  caffeinate -i python -u prep_stage2_sentiment.py 2>&1 | tee prep_sentiment.log
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sentiment as senti

CACHE_PATH = 'results/backtest_cache.npz'
OUTPUT_CSV = 'results/stage2_sentiment.csv'
EXCLUDE = {'SNDK'}
CHUNK_SIZE = 50


def main():
    print("="*70)
    print("Sentiment prep — backtest universe")
    print("="*70)

    if not os.path.exists(CACHE_PATH):
        print(f"\n  {CACHE_PATH} not found"); sys.exit(1)

    # Step 1: Universe
    data = np.load(CACHE_PATH, allow_pickle=True)
    tickers = sorted(set(m[0] for m in data['meta']) - EXCLUDE)
    print(f"\n[1] Universe: {len(tickers)} tickers")
    print(f"    Sample: {tickers[:10]}")

    # Step 2: GICS sector map
    print(f"\n[2] Loading GICS sector mapping...")
    t0 = time.time()
    try:
        from screener import _load_index_industry_data
        ticker_meta = _load_index_industry_data(verbose=False)
    except Exception as e:
        print(f"    GICS fetch failed: {e}")
        ticker_meta = {}
    sector_map = {tk: m.get('gics_sector', '') for tk, m in ticker_meta.items() if m}
    covered = sum(1 for tk in tickers if sector_map.get(tk))
    print(f"    Loaded in {time.time()-t0:.1f}s | coverage {covered}/{len(tickers)} ({100*covered/len(tickers):.1f}%)")

    # Step 3: Resume from partial CSV
    done = set(); existing_rows = []
    if os.path.exists(OUTPUT_CSV):
        try:
            prev = pd.read_csv(OUTPUT_CSV)
            existing_rows = prev.to_dict('records')
            done = set(prev['ticker'].tolist())
            print(f"\n[3] Resume: {len(done)} tickers already done")
        except Exception as e:
            print(f"\n[3] Existing CSV unreadable ({e}), starting fresh")

    remaining = [tk for tk in tickers if tk not in done]
    print(f"    Remaining: {len(remaining)} tickers")

    if not remaining:
        print(f"\n  All tickers processed.")
        _print_sanity(pd.DataFrame(existing_rows))
        return

    # Step 4: Process in chunks, save after each
    print(f"\n[4] Processing in chunks of {CHUNK_SIZE} (~{len(remaining)*9/60:.0f} min est)\n")
    start = time.time()
    all_rows = list(existing_rows)
    feat_names = None
    n_chunks = (len(remaining) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for ci, cs in enumerate(range(0, len(remaining), CHUNK_SIZE)):
        chunk = remaining[cs:cs+CHUNK_SIZE]
        c_t0 = time.time()
        c_sec_map = {tk: sector_map.get(tk, '') for tk in chunk}

        try:
            feats_dict, names = senti.collect_all_intelligence(
                chunk, sector_map=c_sec_map, verbose=False)
        except Exception as e:
            print(f"  Chunk {ci+1} failed: {e}")
            print(f"  Saving progress and exiting (re-run to resume)")
            if all_rows: pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)
            sys.exit(1)

        if feat_names is None: feat_names = names

        for tk in chunk:
            f = feats_dict.get(tk, senti.empty_features())
            row = {'ticker': tk}
            row.update({n: f.get(n, 0) for n in feat_names})
            all_rows.append(row)

        pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)

        c_el = time.time() - c_t0
        t_el = time.time() - start
        n_done = cs + len(chunk)
        eta = (t_el / n_done) * (len(remaining) - n_done)
        print(f"  Chunk {ci+1}/{n_chunks}: {len(chunk)} in {c_el:.0f}s "
              f"({c_el/len(chunk):.1f}s/tk) | "
              f"{n_done}/{len(remaining)} ({100*n_done/len(remaining):.1f}%) | "
              f"ETA {eta/60:.0f}min")

    print(f"\n  Done. {len(remaining)} tickers in {(time.time()-start)/60:.1f} min")
    print(f"  Output: {OUTPUT_CSV} ({len(all_rows)} rows)")
    _print_sanity(pd.DataFrame(all_rows))


def _print_sanity(df):
    """Distribution check for key features."""
    print(f"\n[Sanity] feature distributions:")
    for col in ['composite_sentiment', 'event_risk_score', 'filing_count_30d',
                'fda_event_recent', 'news_sentiment_7d']:
        if col in df.columns:
            v = df[col]
            print(f"  {col:25s}: mean={v.mean():+.3f} std={v.std():.3f} "
                  f"min={v.min():+.3f} max={v.max():+.3f}")
    if 'composite_sentiment' in df.columns:
        cs = df['composite_sentiment']
        print(f"\n  composite_sentiment sign: >0:{(cs>0).sum()} <0:{(cs<0).sum()} =0:{(cs==0).sum()}")


if __name__ == "__main__":
    main()
