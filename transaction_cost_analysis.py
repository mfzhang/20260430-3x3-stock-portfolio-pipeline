#!/usr/bin/env python
"""Apply realistic transaction costs to Stage 2 top-5 selections."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

STAGE2_DIR = Path('results/stage2/top1_trial58')
OUT_JSON = Path('results/tc_analysis_summary.json')
OUT_MD = Path('results/tc_analysis_summary.md')
ADV_CACHE = Path('results/tc_adv_cache.json')

# KIS US-routed defaults
TC_PARAMS = {
    'commission_pct': 0.0004,
    'spread_large_cap_pct': 0.0005,
    'spread_small_cap_pct': 0.0020,
    'large_cap_threshold_usd': 10e9,
    'slippage_pct': 0.0005,
    'fx_spread_pct': 0.0010,
    'impact_constant': 0.10,
}

POSITION_SIZES_KRW = [1_000_000, 2_000_000, 5_000_000]
TURNOVER_PER_YEAR = [2, 4, 12]


def fetch_usd_krw():
    """Latest USD/KRW close."""
    hist = yf.Ticker('KRW=X').history(period='5d', auto_adjust=True)
    if hist is None or len(hist) == 0:
        raise RuntimeError('KRW=X fetch returned empty')
    rate = float(hist['Close'].iloc[-1])
    if not (500 < rate < 3000):
        raise RuntimeError(f'USD/KRW rate {rate} out of sane range')
    return rate


def fetch_ticker_aux(ticker):
    """ADV (USD) and market cap for one ticker."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period='3mo', auto_adjust=True)
    if hist is None or len(hist) == 0:
        return None
    adv_shares = float(hist['Volume'].mean())
    avg_close = float(hist['Close'].mean())
    adv_usd = adv_shares * avg_close
    try:
        mcap = float(tk.info.get('marketCap') or 0)
    except Exception:
        mcap = 0.0
    return {'adv_usd': adv_usd, 'market_cap_usd': mcap, 'avg_close_usd': avg_close}


def load_or_fetch_aux(tickers):
    cache = {}
    if ADV_CACHE.exists():
        cache = json.loads(ADV_CACHE.read_text())
    missing = [t for t in tickers if t not in cache]
    if missing:
        print(f'Fetching ADV/MCap for {len(missing)} tickers...')
        for t in missing:
            try:
                aux = fetch_ticker_aux(t)
                if aux:
                    cache[t] = aux
                    print(f"  {t}: ADV=${aux['adv_usd']/1e6:.1f}M  MCap=${aux['market_cap_usd']/1e9:.1f}B")
                time.sleep(0.2)
            except Exception as e:
                print(f'  {t}: failed ({e})')
        ADV_CACHE.write_text(json.dumps(cache, indent=2))
    return cache


def compute_per_leg_tc(position_size_usd, adv_usd, market_cap_usd, params):
    """Single-leg TC as fraction of position notional."""
    commission = params['commission_pct']
    spread = params['spread_large_cap_pct'] if market_cap_usd >= params['large_cap_threshold_usd'] \
             else params['spread_small_cap_pct']
    slippage = params['slippage_pct']
    if adv_usd > 0:
        impact = params['impact_constant'] * (position_size_usd / adv_usd) ** (1/3)
    else:
        impact = params['spread_small_cap_pct']
    return commission + spread + slippage + impact


def compute_fold_net_alpha(fold_info, aux, position_krw, turnover_yr, params, usd_krw):
    position_usd = position_krw / usd_krw
    legs_per_year = 2 * turnover_yr
    per_leg_costs = []
    for tk in fold_info['top5']:
        if tk not in aux:
            per_leg_costs.append(0.005)
            continue
        a = aux[tk]
        per_leg_costs.append(compute_per_leg_tc(position_usd, a['adv_usd'], a['market_cap_usd'], params))
    mean_per_leg = float(np.mean(per_leg_costs))
    annual_tc_pct = legs_per_year * mean_per_leg + 2 * params['fx_spread_pct']
    quarterly_tc_pct = annual_tc_pct / 4
    paper_3m = fold_info['top5_actual_3m']
    paper_alpha = fold_info['alpha_vs_spy']
    return {
        'fold_id': fold_info['fold_id'],
        'mean_per_leg_pct': mean_per_leg,
        'annual_tc_pct': annual_tc_pct,
        'quarterly_tc_pct': quarterly_tc_pct,
        'paper_3m_return': paper_3m,
        'paper_alpha_vs_spy': paper_alpha,
        'net_3m_return': paper_3m - quarterly_tc_pct,
        'net_alpha_vs_spy': paper_alpha - quarterly_tc_pct,
    }


def main():
    summary = json.loads((STAGE2_DIR / 'summary.json').read_text())
    spy_df = pd.read_csv(STAGE2_DIR / 'spy_benchmark.csv')

    usd_krw = fetch_usd_krw()
    print(f'USD/KRW = {usd_krw:.2f}\n')

    fold_records = []
    for fi in summary['per_fold']:
        row = spy_df[spy_df['fold_id'] == fi['fold_id']].iloc[0]
        fold_records.append({
            'fold_id': fi['fold_id'],
            'top5': fi['top5'],
            'top5_actual_3m': float(row['top5_actual_3m']),
            'spy_3m': float(row['spy_mean_3m_return']),
            'alpha_vs_spy': float(row['alpha_vs_spy']),
        })

    all_tickers = sorted({t for fi in fold_records for t in fi['top5']})
    print(f'Unique tickers across folds: {len(all_tickers)}')
    aux = load_or_fetch_aux(all_tickers)

    grid = []
    for pos_krw in POSITION_SIZES_KRW:
        for turn in TURNOVER_PER_YEAR:
            fold_results = [compute_fold_net_alpha(fi, aux, pos_krw, turn, TC_PARAMS, usd_krw)
                            for fi in fold_records]
            net_alphas = [fr['net_alpha_vs_spy'] for fr in fold_results]
            grid.append({
                'position_size_krw': pos_krw,
                'turnover_per_year': turn,
                'mean_quarterly_tc_pct': float(np.mean([fr['quarterly_tc_pct'] for fr in fold_results])),
                'mean_net_alpha_pct': float(np.mean(net_alphas)),
                'std_net_alpha_pct': float(np.std(net_alphas)),
                'per_fold': fold_results,
            })

    output = {
        'tc_params': TC_PARAMS,
        'usd_krw': usd_krw,
        'usd_krw_fetched_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'paper_alpha_baseline': float(spy_df['alpha_vs_spy'].mean()),
        'paper_alpha_std': float(spy_df['alpha_vs_spy'].std()),
        'grid': grid,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f'\nWrote {OUT_JSON}')

    lines = [
        '# Stage 1 — Transaction Cost Sensitivity\n',
        f'Paper alpha baseline (mean across 5 folds): **+{output["paper_alpha_baseline"]*100:.2f}%p / quarter** '
        f'(±{output["paper_alpha_std"]*100:.2f}%p)',
        f'USD/KRW: {usd_krw:.2f} (fetched {output["usd_krw_fetched_at"]})\n',
        '## Sensitivity grid\n',
        '| Position (₩) | Turnover/y | Quarterly TC | Net alpha | StDev |',
        '|---:|---:|---:|---:|---:|',
    ]
    for g in grid:
        lines.append(f"| {g['position_size_krw']:,} | {g['turnover_per_year']} | "
                     f"{g['mean_quarterly_tc_pct']*100:.2f}% | "
                     f"**{g['mean_net_alpha_pct']*100:+.2f}%p** | "
                     f"±{g['std_net_alpha_pct']*100:.2f}%p |")
    OUT_MD.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {OUT_MD}\n')

    print(f"{'Pos (₩M)':>10s} {'Turn/y':>8s} {'Q-TC':>8s} {'Net α':>10s}")
    for g in grid:
        print(f"{g['position_size_krw']/1e6:>9.0f}M {g['turnover_per_year']:>8d} "
              f"{g['mean_quarterly_tc_pct']*100:>7.2f}% "
              f"{g['mean_net_alpha_pct']*100:>+9.2f}%p")


if __name__ == '__main__':
    main()
