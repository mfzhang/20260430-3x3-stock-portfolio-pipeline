#!/usr/bin/env python3
"""
v2.3.13 pre-launch smoke comparison: standard NLL vs β-NLL.

Per pre-registration amendment 1 (commit 653688c):
  - Fold 2 only
  - N_ENSEMBLE = 1
  - TRAINING_EPOCHS = 2000, patience = 41
  - 3 seeds per loss = 6 total runs
  - Pre-specified metrics computed per run, aggregated across seeds
  - 3-criteria launch decision rule (a/b/c)

Output: smoke_v2313_results.json + console summary.

Does NOT touch v2.3.12 production (results/stage2/top1_trial58/).
"""
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, '.')

# Import stage2_retrain machinery
import stage2_retrain as s2
import config
import backtest
from models import heteroscedastic_loss, heteroscedastic_loss_beta


# ============================================================
# SMOKE CONFIG (matches Amendment 1)
# ============================================================
SMOKE_FOLD = 1                    # 0-indexed fold 1 = Fold 2 (1-indexed)
SMOKE_N_ENSEMBLE = 1
SMOKE_EPOCHS = 2000
SMOKE_SEEDS = [42, 43, 44]
SMOKE_LOSSES = {
    'standard': heteroscedastic_loss,
    'beta_nll': heteroscedastic_loss_beta,
}

OUTPUT_JSON = 'smoke_v2313_results.json'
OUTPUT_LOG = 'smoke_v2313_betanll.log'

# Pre-registered launch decision thresholds
RANK_CORR_TOLERANCE = 0.05       # criterion (b)
Z_STD_DIFFERENTIATION_MIN = 0.10 # criterion (c)


# ============================================================
# METRIC COMPUTATION (pre-specified in amendment 1)
# ============================================================
def compute_calibration_metrics(full_ranking_df):
    """z-score std, |z|<1 coverage, sigma tertile |z| gap."""
    import pandas as pd
    if isinstance(full_ranking_df, list):
        full_ranking_df = pd.DataFrame(full_ranking_df)
    log_pred = full_ranking_df['pred_risk_log_mean'].values
    log_sigma = full_ranking_df['pred_risk_log_sigma'].values
    actual_risk = full_ranking_df['actual_risk'].values
    log_actual = np.log(np.maximum(actual_risk, 1e-4))
    log_residual = log_actual - log_pred
    z = log_residual / np.maximum(log_sigma, 1e-6)
    z_std = float(np.std(z))
    coverage_1 = float(np.mean(np.abs(z) < 1) * 100)
    coverage_2 = float(np.mean(np.abs(z) < 2) * 100)
    q33, q67 = np.percentile(log_sigma, [33, 67])
    z_small = np.abs(z[log_sigma <= q33])
    z_large = np.abs(z[log_sigma >= q67])
    tertile_gap = float(z_large.mean() - z_small.mean())
    log_sigma_range = float(log_sigma.max() - log_sigma.min())
    return {
        'z_std': z_std,
        'z_coverage_1sigma_pct': coverage_1,
        'z_coverage_2sigma_pct': coverage_2,
        'tertile_gap': tertile_gap,
        'log_sigma_range': log_sigma_range,
        'log_sigma_min': float(log_sigma.min()),
        'log_sigma_max': float(log_sigma.max()),
    }


# ============================================================
# SINGLE RUN
# ============================================================
def run_one(loss_name, loss_fn, seed, data, train_tickers, test_tickers, params):
    np.random.seed(seed)
    torch.manual_seed(seed)

    s2.LOSS_FN = loss_fn
    s2.LOSS_LABEL = f'{loss_name} (smoke seed={seed})'

    overrides = {
        'TRAINING_EPOCHS': SMOKE_EPOCHS,
        'TRAINING_LR': params['lr'],
        'TRAINING_HUBER_DELTA': params['huber_delta'],
        'VAR_THRESHOLD': params['var_threshold'],
        'CORR_THRESHOLD': params['corr_threshold'],
    }
    arch_map = {
        'small': [32, 16],
        'medium': [64, 32, 16],
        'large': [128, 64, 32],
    }
    overrides['TRAINING_NN_ARCHITECTURE'] = arch_map[params['architecture']]

    originals = s2.override_config(config, overrides)
    adam_original = s2.patch_adam(params['weight_decay'])

    nan_detected = False
    rank_corr = None
    full_ranking = None
    elapsed = None
    try:
        t0 = time.time()
        result = s2.run_fold_with_plot(
            data=data,
            train_tickers=train_tickers,
            test_tickers=test_tickers,
            fold_id=SMOKE_FOLD,
            config_module=config,
            live_plot=None,
            n_ensemble=SMOKE_N_ENSEMBLE,
            n_select=5,
            config_label=f'smoke_{loss_name}_seed{seed}',
            seed_override=seed,
        )
        elapsed = time.time() - t0
        rank_corr = float(result['rank_corr'])
        full_ranking = result['full_ranking']
        if np.isnan(rank_corr):
            nan_detected = True
    except Exception as e:
        print(f"  [ERROR] {loss_name} seed={seed}: {type(e).__name__}: {e}")
        nan_detected = True
    finally:
        torch.optim.Adam.__init__ = adam_original
        s2.restore_config(config, originals)

    out = {
        'loss': loss_name,
        'seed': seed,
        'nan_or_error': nan_detected,
        'rank_corr': rank_corr,
        'elapsed_sec': elapsed,
    }
    if not nan_detected and full_ranking is not None:
        out.update(compute_calibration_metrics(full_ranking))
        import pandas as pd
        if isinstance(full_ranking, list):
            full_ranking = pd.DataFrame(full_ranking)
        out['top5'] = list(full_ranking.sort_values('pred_rank').head(5)['ticker'])
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    print('=' * 70)
    print('v2.3.13 SMOKE COMPARISON — standard NLL vs β-NLL')
    print('Per pre-registration amendment 1 (commit 653688c)')
    print('=' * 70)
    print(f'Fold: {SMOKE_FOLD+1} (0-idx {SMOKE_FOLD})')
    print(f'N_ENSEMBLE: {SMOKE_N_ENSEMBLE}')
    print(f'Epochs: {SMOKE_EPOCHS}')
    print(f'Seeds: {SMOKE_SEEDS}')
    print(f'Losses: {list(SMOKE_LOSSES.keys())}')
    print(f'Total runs: {len(SMOKE_LOSSES) * len(SMOKE_SEEDS)}')
    print('=' * 70)

    # Load Optuna Trial #58 params (same as production)
    with open('results/optuna_stage1_results.json') as f:
        optuna_d = json.load(f)
    params = optuna_d['top_3_configs'][0]['params']
    print(f'\nTrial #{optuna_d["top_3_configs"][0]["trial_number"]} params loaded.')

    # Load cache (SNDK already excluded by s2.load_filtered_cache via EXCLUDED_TICKERS)
    data = s2.load_filtered_cache('results/backtest_cache.npz')

    # Get fold split
    from backtest import _stratified_kfold, _get_ticker_sectors
    sample_tickers = data['sample_tickers']
    universe = sorted(set(sample_tickers))
    sectors = _get_ticker_sectors(universe)
    folds = _stratified_kfold(universe, sectors, n_folds=5)
    train_tickers, test_tickers = folds[SMOKE_FOLD]
    print(f'Fold {SMOKE_FOLD+1}: train={len(train_tickers)} test={len(test_tickers)}')

    # Run all combinations
    all_runs = []
    t_overall = time.time()
    for loss_name, loss_fn in SMOKE_LOSSES.items():
        for seed in SMOKE_SEEDS:
            print(f'\n--- Run: loss={loss_name}, seed={seed} ---')
            run_result = run_one(
                loss_name, loss_fn, seed,
                data, train_tickers, test_tickers, params,
            )
            all_runs.append(run_result)
            if run_result.get('nan_or_error'):
                print(f'  STATUS: NaN/ERROR')
            else:
                print(f"  rank_corr={run_result['rank_corr']:+.4f}, "
                      f"z_std={run_result['z_std']:.3f}, "
                      f"tertile_gap={run_result['tertile_gap']:+.3f}, "
                      f"|z|<1={run_result['z_coverage_1sigma_pct']:.0f}%, "
                      f"elapsed={run_result['elapsed_sec']/60:.1f}min")
    total_elapsed = time.time() - t_overall

    # ============================================================
    # AGGREGATE
    # ============================================================
    def aggregate(loss_name):
        runs = [r for r in all_runs if r['loss'] == loss_name]
        if any(r['nan_or_error'] for r in runs):
            return {'any_nan_or_error': True, 'n_valid': 0}
        rc = [r['rank_corr'] for r in runs]
        zs = [r['z_std'] for r in runs]
        tg = [r['tertile_gap'] for r in runs]
        lsr = [r['log_sigma_range'] for r in runs]
        cov1 = [r['z_coverage_1sigma_pct'] for r in runs]
        return {
            'any_nan_or_error': False,
            'n_valid': len(runs),
            'rank_corr_mean': float(np.mean(rc)),
            'rank_corr_std': float(np.std(rc)),
            'z_std_mean': float(np.mean(zs)),
            'z_std_std': float(np.std(zs)),
            'tertile_gap_mean': float(np.mean(tg)),
            'log_sigma_range_mean': float(np.mean(lsr)),
            'z_coverage_1sigma_mean': float(np.mean(cov1)),
        }

    summary = {
        'standard': aggregate('standard'),
        'beta_nll': aggregate('beta_nll'),
    }

    # ============================================================
    # DECISION RULE (per amendment 1)
    # ============================================================
    decision = {}
    std_agg = summary['standard']
    bnll_agg = summary['beta_nll']
    decision['a_no_nan'] = (not bnll_agg['any_nan_or_error']) and bnll_agg['n_valid'] == 3
    if decision['a_no_nan'] and not std_agg['any_nan_or_error']:
        decision['b_rank_corr_ok'] = (
            bnll_agg['rank_corr_mean'] >= std_agg['rank_corr_mean'] - RANK_CORR_TOLERANCE
        )
        decision['b_delta_rank_corr'] = bnll_agg['rank_corr_mean'] - std_agg['rank_corr_mean']
        decision['c_z_std_differentiation'] = (
            abs(bnll_agg['z_std_mean'] - std_agg['z_std_mean']) > Z_STD_DIFFERENTIATION_MIN
        )
        decision['c_delta_z_std'] = bnll_agg['z_std_mean'] - std_agg['z_std_mean']
    else:
        decision['b_rank_corr_ok'] = False
        decision['c_z_std_differentiation'] = False

    decision['launch_full'] = (
        decision['a_no_nan']
        and decision['b_rank_corr_ok']
        and decision['c_z_std_differentiation']
    )

    # ============================================================
    # WRITE + PRINT
    # ============================================================
    output = {
        'design_commit': '653688c',
        'patch_commit': '6271f59',
        'total_elapsed_min': total_elapsed / 60,
        'all_runs': all_runs,
        'summary': summary,
        'decision': decision,
        'thresholds_used': {
            'rank_corr_tolerance': RANK_CORR_TOLERANCE,
            'z_std_differentiation_min': Z_STD_DIFFERENTIATION_MIN,
        },
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)

    print('\n' + '=' * 70)
    print('SMOKE SUMMARY')
    print('=' * 70)
    for loss_name in ['standard', 'beta_nll']:
        a = summary[loss_name]
        if a['any_nan_or_error']:
            print(f'{loss_name:10s}: NaN/ERROR detected in at least one seed')
        else:
            print(f'{loss_name:10s}: rank_corr={a["rank_corr_mean"]:+.4f} ± {a["rank_corr_std"]:.4f}  '
                  f'z_std={a["z_std_mean"]:.3f} ± {a["z_std_std"]:.3f}  '
                  f'tertile_gap={a["tertile_gap_mean"]:+.3f}  '
                  f'lsr={a["log_sigma_range_mean"]:.3f}  '
                  f'cov1={a["z_coverage_1sigma_mean"]:.0f}%')

    print('\n' + '-' * 70)
    print('LAUNCH DECISION (per pre-registration amendment 1)')
    print('-' * 70)
    print(f'  (a) No NaN in β-NLL runs:                  {decision["a_no_nan"]}')
    if 'b_delta_rank_corr' in decision:
        print(f'  (b) β-NLL rank_corr ≥ Std - {RANK_CORR_TOLERANCE}:        '
              f'{decision["b_rank_corr_ok"]} (Δ={decision["b_delta_rank_corr"]:+.4f})')
        print(f'  (c) |β-NLL z_std - Std z_std| > {Z_STD_DIFFERENTIATION_MIN}:    '
              f'{decision["c_z_std_differentiation"]} (Δ={decision["c_delta_z_std"]:+.3f})')
    else:
        print(f'  (b) skipped (criterion a failed)')
        print(f'  (c) skipped (criterion a failed)')
    print('-' * 70)
    print(f'  LAUNCH FULL β-NLL RETRAIN: {decision["launch_full"]}')
    print('-' * 70)
    print(f'\nResults written to {OUTPUT_JSON}')
    print(f'Total smoke time: {total_elapsed/60:.1f} min')

    sys.exit(0 if decision['launch_full'] else 1)


if __name__ == '__main__':
    main()
