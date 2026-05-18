#!/usr/bin/env python3
"""
Composite score coefficient grid search

Recomputes composite scores from Stage 2 cached predictions + sentiment features.
No NN retraining. Evaluates each combo's top-5 selection quality.

Usage:
  python task9_composite_grid.py 2>&1 | tee task9_grid.log
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STAGE2_DIR = 'results/stage2/top1_trial58'
SENTIMENT_CSV = 'results/stage2_sentiment.csv'
OUTPUT_JSON = 'results/task9_composite_grid_results.json'
N_SELECT = 5
N_FOLDS = 5

# Search space
SW_GRID = np.round(np.arange(0.00, 0.31, 0.05), 4)         # 7 values
UP_GRID = np.round(np.arange(1.0, 5.01, 0.5), 4)           # 9 values
ER_GRID = np.round(np.arange(0.0, 4.01, 0.5), 4)           # 9 values

# Production current values
CUR_SW, CUR_UP, CUR_ER = 0.10, 3.0, 2.0


def load_fold_data(stage2_dir, sentiment_df):
    """Merge each fold's Stage 2 predictions with sentiment features."""
    folds = []
    for i in range(1, N_FOLDS + 1):
        path = f'{stage2_dir}/fold_{i}/full_ranking.csv'
        if not os.path.exists(path):
            print(f"  Missing: {path}"); sys.exit(1)
        pred = pd.read_csv(path)
        merged = pred.merge(sentiment_df, on='ticker', how='left')
        # Tickers without sentiment -> neutral fallback
        for col in ['composite_sentiment', 'filing_sentiment', 'fda_sentiment',
                    'fda_event_recent', 'event_risk_score']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        folds.append(merged)
    return folds


def compute_score(df, sw, up, er):
    """Vectorized composite score per row, mirrors sentiment.composite_score_v2."""
    pred_std = np.maximum(df['pred_std'].values, 0.01)
    sharpe = df['pred_ret'].values / pred_std

    sent_boost = 1 + df['composite_sentiment'].values * sw
    filing_boost = 1 + df['filing_sentiment'].values * 0.05
    fda_active = df['fda_event_recent'].values > 0
    fda_bonus = np.where(fda_active, 1 + df['fda_sentiment'].values * 0.08, 1.0)

    num = sharpe * sent_boost * filing_boost * fda_bonus
    denom = 1 + pred_std * up + df['event_risk_score'].values * er
    return num / denom


def evaluate_combo(folds, sw, up, er):
    """Per-fold metrics: rank_corr (Spearman), top-5 alpha vs universe."""
    rank_corrs, alphas, top5_picks = [], [], []
    for df in folds:
        scores = compute_score(df, sw, up, er)
        actual = df['actual_ret'].values

        # Rank correlation
        if len(scores) >= 5 and np.std(scores) > 1e-10:
            rc, _ = spearmanr(scores, actual)
        else:
            rc = 0.0
        rank_corrs.append(rc if not np.isnan(rc) else 0.0)

        # Top-5 alpha
        top5_idx = np.argsort(scores)[-N_SELECT:]
        top5_actual = actual[top5_idx].mean()
        all_actual = actual.mean()
        alphas.append(top5_actual - all_actual)

        # Picks (for inspection of best combos)
        top5_picks.append(df.iloc[top5_idx]['ticker'].tolist())

    return {
        'rank_corrs': rank_corrs,
        'alphas': alphas,
        'top5_picks': top5_picks,
        'mean_rank_corr': np.mean(rank_corrs),
        'mean_alpha': np.mean(alphas),
        'robust_rank_corr': np.mean(rank_corrs[1:]),  # Folds 2-5 (exclude SNDK)
        'robust_alpha': np.mean(alphas[1:]),
    }


def main():
    print("=" * 70)
    print("Task #9 — composite score coefficient grid")
    print("=" * 70)

    # Step 1: Load sentiment features
    if not os.path.exists(SENTIMENT_CSV):
        print(f"\n  {SENTIMENT_CSV} not found. Run prep_stage2_sentiment.py first.")
        sys.exit(1)
    sent_df = pd.read_csv(SENTIMENT_CSV)
    print(f"\n[1] Sentiment: {len(sent_df)} tickers")

    # Step 2: Load Stage 2 fold predictions
    print(f"\n[2] Loading Stage 2 predictions...")
    folds = load_fold_data(STAGE2_DIR, sent_df)
    for i, f in enumerate(folds):
        miss = f['composite_sentiment'].isna().sum() if 'composite_sentiment' in f.columns else 0
        print(f"    Fold {i+1}: {len(f)} tickers (sentiment missing: {miss})")

    # Step 3: Baseline (current production values)
    print(f"\n[3] Baseline (current sw={CUR_SW}, up={CUR_UP}, er={CUR_ER}):")
    baseline = evaluate_combo(folds, CUR_SW, CUR_UP, CUR_ER)
    print(f"    rank_corr (5-fold mean): {baseline['mean_rank_corr']:+.4f}")
    print(f"    rank_corr (robust 2-5): {baseline['robust_rank_corr']:+.4f}")
    print(f"    alpha     (5-fold mean): {baseline['mean_alpha']*100:+.2f}%p")
    print(f"    alpha     (robust 2-5): {baseline['robust_alpha']*100:+.2f}%p")

    # Step 4: Grid search
    n_combos = len(SW_GRID) * len(UP_GRID) * len(ER_GRID)
    print(f"\n[4] Grid: {len(SW_GRID)}x{len(UP_GRID)}x{len(ER_GRID)} = {n_combos} combos")
    t0 = time.time()
    results = []
    for sw in SW_GRID:
        for up in UP_GRID:
            for er in ER_GRID:
                m = evaluate_combo(folds, sw, up, er)
                results.append({
                    'sw': float(sw), 'up': float(up), 'er': float(er),
                    'mean_rank_corr': m['mean_rank_corr'],
                    'robust_rank_corr': m['robust_rank_corr'],
                    'mean_alpha': m['mean_alpha'],
                    'robust_alpha': m['robust_alpha'],
                    'fold_rank_corrs': m['rank_corrs'],
                    'fold_alphas': m['alphas'],
                    'top5_picks': m['top5_picks'],
                })
    print(f"    Done in {time.time()-t0:.1f}s")

    # Step 5: Rank by criteria
    df_r = pd.DataFrame(results)
    print(f"\n[5] Top-10 by robust_rank_corr (folds 2-5):")
    top_rc = df_r.nlargest(10, 'robust_rank_corr')
    print(top_rc[['sw', 'up', 'er', 'robust_rank_corr', 'robust_alpha',
                  'mean_rank_corr', 'mean_alpha']].to_string(index=False,
                  formatters={'robust_alpha': lambda x: f'{x*100:+.2f}%p',
                              'mean_alpha':   lambda x: f'{x*100:+.2f}%p'}))

    print(f"\n[6] Top-10 by robust_alpha (folds 2-5):")
    top_a = df_r.nlargest(10, 'robust_alpha')
    print(top_a[['sw', 'up', 'er', 'robust_alpha', 'robust_rank_corr',
                 'mean_alpha', 'mean_rank_corr']].to_string(index=False,
                 formatters={'robust_alpha': lambda x: f'{x*100:+.2f}%p',
                             'mean_alpha':   lambda x: f'{x*100:+.2f}%p'}))

    # Step 6: Compare to baseline
    best_rc = df_r.loc[df_r['robust_rank_corr'].idxmax()]
    best_a  = df_r.loc[df_r['robust_alpha'].idxmax()]

    print(f"\n[7] Improvement vs baseline (sw={CUR_SW}, up={CUR_UP}, er={CUR_ER}):")
    print(f"    Best rank_corr combo: sw={best_rc['sw']:.2f} up={best_rc['up']:.1f} er={best_rc['er']:.1f}")
    print(f"      robust_rank_corr: {baseline['robust_rank_corr']:+.4f} -> {best_rc['robust_rank_corr']:+.4f} "
          f"(Δ={best_rc['robust_rank_corr']-baseline['robust_rank_corr']:+.4f})")
    print(f"      robust_alpha:     {baseline['robust_alpha']*100:+.2f}%p -> {best_rc['robust_alpha']*100:+.2f}%p "
          f"(Δ={(best_rc['robust_alpha']-baseline['robust_alpha'])*100:+.2f}%p)")

    print(f"    Best alpha combo: sw={best_a['sw']:.2f} up={best_a['up']:.1f} er={best_a['er']:.1f}")
    print(f"      robust_alpha:     {baseline['robust_alpha']*100:+.2f}%p -> {best_a['robust_alpha']*100:+.2f}%p "
          f"(Δ={(best_a['robust_alpha']-baseline['robust_alpha'])*100:+.2f}%p)")

    # Step 7: Robustness — check best combos across all folds
    print(f"\n[8] Per-fold rank_corr for best combos:")
    print(f"    {'Combo':40s} F1     F2     F3     F4     F5")
    print(f"    {'-'*40} ------ ------ ------ ------ ------")
    for label, row in [('Baseline (current)', baseline),
                        (f'Best rank_corr (sw={best_rc.sw:.2f} up={best_rc.up:.1f} er={best_rc.er:.1f})', best_rc),
                        (f'Best alpha     (sw={best_a.sw:.2f} up={best_a.up:.1f} er={best_a.er:.1f})', best_a)]:
        if isinstance(row, pd.Series):
            rcs = row['fold_rank_corrs']
        else:
            rcs = row['rank_corrs']
        rcs_str = ' '.join(f'{r:+.3f}' for r in rcs)
        print(f"    {label:40s} {rcs_str}")

    # Step 8: Save results
    save = {
        'baseline': {'sw': CUR_SW, 'up': CUR_UP, 'er': CUR_ER, **{k: v for k, v in baseline.items() if k != 'top5_picks'}},
        'best_rank_corr': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in best_rc.to_dict().items()},
        'best_alpha':     {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in best_a.to_dict().items()},
        'grid_size': {'sw': len(SW_GRID), 'up': len(UP_GRID), 'er': len(ER_GRID), 'total': n_combos},
        'top10_by_rank_corr': top_rc.drop(columns=['top5_picks']).to_dict('records'),
        'top10_by_alpha':     top_a.drop(columns=['top5_picks']).to_dict('records'),
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(save, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\n[9] Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
