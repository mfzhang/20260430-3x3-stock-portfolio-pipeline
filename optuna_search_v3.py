"""
optuna_search_v3.py — Stage 1 hyperparameter search v3 (v2.3.15 design)

v3 changes vs optuna_search_v2.py (v2.3.10):
  - Loss function is now heteroscedastic Gaussian NLL (v2.3.12 production),
    NOT Huber. The Huber-specific `huber_delta` search dimension is REMOVED.
  - `dropout` added as a new search dimension (was hardcoded 0.2 in v2.3.6
    and v2.3.12). Now reads from config.TRAINING_DROPOUT, exposed in commit
    92ab700 as a prerequisite.
  - Adam weight_decay monkey-patch REMOVED. backtest.py:311 already reads
    from config.TRAINING_WEIGHT_DECAY (v2.3.8 Defect A fix, commit 7889623),
    so `_override_config()` is sufficient. v3 simplifies the per-trial setup.
  - New study name, storage, output directory. Old v2 / v2310 studies are
    preserved untouched. v3 cache reuses `results/backtest_cache.npz` (97
    features, architecture-independent — confirmed in pre-registration §6).

Pre-registration: pre_registration_v2315_optuna_rerun.md (commits 7288c5b
+ e954025 Amendment 1). All design choices below are locked by that file.

Methodology summary (post-Amendment 1):
  - 5 folds + SNDK ticker-level exclusion at data load (v2.3.10/v2.3.12).
  - 6-dim search: lr, weight_decay, architecture, var_thr, corr_thr, dropout.
  - N=5 ensemble per trial (production retrain at N=20 in v2.3.16).
  - 60 trials, TPE sampler seed=42, n_startup_trials=10.
  - Resume-safe sqlite storage; per-trial 120-min timeout.

Usage:
  # Smoke test (1 trial, separate DB to avoid polluting study)
  caffeinate -i python -u optuna_search_v3.py --smoke 2>&1 | tee optuna_smoke_v2315.log

  # Full run (60 trials, ~45-55h)
  caffeinate -i python -u optuna_search_v3.py 2>&1 | tee optuna_v2315.log

Estimated runtime: ~45-55h (5-fold + 6-dim search + NLL parity per epoch).
"""

import sys
import os
import json
import time
import signal
import traceback
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# STAGE 1 v3 CONFIGURATION (v2.3.15)
# ============================================================

STUDY_NAME = "stage1_5fold_NLL_6dims_no_sndk"
STORAGE = "sqlite:///optuna_storage_v2315.db"
N_TRIALS = 60
N_ENSEMBLE_STAGE1 = 5
N_SELECT = 5
N_FOLDS_TOTAL = 5

# v2.3.10/v2.3.12 methodology (Amendment 1): all 5 folds, SNDK ticker exclusion
STAGE1_FOLD_INDICES = [0, 1, 2, 3, 4]
EXCLUDED_TICKERS = {'SNDK'}

TRIAL_TIMEOUT_SEC = 900 * 60   # [v2.3.15 Amendment 2] 120 → 900 min (15h) — accommodates large arch + cap 20000
CACHE_PATH = "results/backtest_cache.npz"
OUTPUT_DIR = "results/stage1_v2315"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "best_trials.json")

SAMPLER_SEED = 42
N_STARTUP_TRIALS = 10

ARCH_CHOICES = {
    "small":  [32, 16],
    "medium": [64, 32, 16],
    "large":  [128, 64, 32],
}


# Smoke-mode overrides (set by --smoke CLI flag, override module-level constants)
SMOKE_STUDY_NAME = "smoke_v2315"
SMOKE_STORAGE = "sqlite:///optuna_smoke_v2315.db"
SMOKE_N_TRIALS = 1


# ============================================================
# TIMEOUT HANDLING
# ============================================================

class TrialTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TrialTimeoutError(f"Trial exceeded {TRIAL_TIMEOUT_SEC}s")


# ============================================================
# CONFIG OVERRIDE (with restoration)
# ============================================================

def _override_config(config_module, overrides):
    """Override config values; return originals for restoration."""
    originals = {}
    for key, value in overrides.items():
        originals[key] = getattr(config_module, key, None)
        setattr(config_module, key, value)
    return originals


def _restore_config(config_module, originals):
    for key, value in originals.items():
        setattr(config_module, key, value)


# ============================================================
# DATA LOADING (cached, reused across trials; SNDK filtered)
# ============================================================

_DATA_CACHE = {}
_FOLDS_CACHE = None


def _load_data():
    """Load cached training data with SNDK exclusion. Fail loudly if cache missing."""
    if 'X' in _DATA_CACHE:
        return _DATA_CACHE

    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"Cache not found: {CACHE_PATH}\n"
            f"Run `python backtest.py` first to build it."
        )

    print(f"[Data] Loading {CACHE_PATH}...")
    data = np.load(CACHE_PATH, allow_pickle=True)
    X = data['X']
    Y_ret = data['Y_ret']
    Y_risk = data['Y_risk']
    meta_array = data['meta']  # ndarray (N, 3): [ticker, offset, date]
    feat_names = list(data['feat_names'])

    # SNDK ticker-level exclusion (Amendment 1: matches v2.3.10/v2.3.12 production)
    sample_tickers = meta_array[:, 0].astype(str)
    n_total = len(X)

    if EXCLUDED_TICKERS:
        mask = ~np.isin(sample_tickers, list(EXCLUDED_TICKERS))
        n_excluded = int((~mask).sum())
        X = X[mask]
        Y_ret = Y_ret[mask]
        Y_risk = Y_risk[mask]
        meta_array = meta_array[mask]
        sample_tickers = sample_tickers[mask]
        print(f"[Data] Excluded {n_excluded} samples from "
              f"{sorted(EXCLUDED_TICKERS)} (was {n_total}, now {len(X)})")

    # backtest._run_single_fold expects meta as list of tuples
    meta = [tuple(m) for m in meta_array]
    unique_tickers = sorted(set(sample_tickers))

    _DATA_CACHE.update({
        'X': X, 'Y_ret': Y_ret, 'Y_risk': Y_risk,
        'meta': meta, 'feat_names': feat_names,
        'sample_tickers': sample_tickers,
        'unique_tickers': unique_tickers,
    })
    print(f"[Data] {X.shape[0]:,} samples × {X.shape[1]} features, "
          f"{len(unique_tickers)} tickers")
    return _DATA_CACHE


def _get_folds():
    """Stratified K-fold splits (computed once on filtered ticker list, reused)."""
    global _FOLDS_CACHE
    if _FOLDS_CACHE is not None:
        return _FOLDS_CACHE

    from backtest import _stratified_kfold, _get_ticker_sectors

    data = _load_data()
    ticker_sectors = _get_ticker_sectors(data['unique_tickers'], verbose=False)
    folds = _stratified_kfold(data['unique_tickers'], ticker_sectors, N_FOLDS_TOTAL)

    print(f"[Folds] Using indices {STAGE1_FOLD_INDICES} (all 5 folds; "
          f"{sorted(EXCLUDED_TICKERS)} excluded at ticker level)")
    _FOLDS_CACHE = folds
    return folds


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective(trial):
    """Run a single trial; return mean rank_corr across all 5 folds."""
    trial_start = time.time()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TRIAL_TIMEOUT_SEC)

    try:
        # 1. Suggest hyperparameters (6 dims; huber_delta removed, dropout added)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        arch_choice = trial.suggest_categorical("architecture", list(ARCH_CHOICES.keys()))
        var_threshold = trial.suggest_float("var_threshold", 1e-3, 1e-1, log=True)
        corr_threshold = trial.suggest_float("corr_threshold", 1e-3, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        architecture = ARCH_CHOICES[arch_choice]

        print(f"\n{'='*60}")
        print(f"[Trial {trial.number}] {time.strftime('%H:%M:%S')}")
        print(f"  lr={lr:.5f}  wd={weight_decay:.5f}  dropout={dropout:.3f}")
        print(f"  arch={arch_choice} {architecture}")
        print(f"  var_thr={var_threshold:.4f}  corr_thr={corr_threshold:.4f}")
        print(f"{'='*60}")

        # 2. Override config (Adam monkey-patch no longer needed —
        #    backtest.py reads weight_decay/dropout from config since v2.3.8/v2.3.15)
        import config
        overrides = {
            'TRAINING_LR': lr,
            'TRAINING_WEIGHT_DECAY': weight_decay,
            'TRAINING_NN_ARCHITECTURE': architecture,
            'TRAINING_DROPOUT': dropout,
            'VAR_THRESHOLD': var_threshold,
            'CORR_THRESHOLD': corr_threshold,
            'N_ENSEMBLE': N_ENSEMBLE_STAGE1,
        }
        originals = _override_config(config, overrides)

        try:
            data = _load_data()
            folds = _get_folds()

            # 3. Run all 5 folds
            from backtest import _run_single_fold

            fold_rank_corrs = []
            fold_alphas = []

            for fold_idx in STAGE1_FOLD_INDICES:
                train_tickers, test_tickers = folds[fold_idx]
                fold_start = time.time()
                print(f"\n  --- Fold {fold_idx+1}/5 ---")

                result = _run_single_fold(
                    data['X'], data['Y_ret'], data['Y_risk'],
                    data['sample_tickers'], data['meta'],
                    train_tickers, test_tickers, N_SELECT,
                    verbose=True,
                )
                rc = result['rank_corr']
                alpha = result['selection_alpha']
                fold_rank_corrs.append(rc)
                fold_alphas.append(alpha)

                print(f"  Fold {fold_idx+1}: rank_corr={rc:+.3f}  "
                      f"alpha={alpha*100:+.1f}%p  "
                      f"elapsed={(time.time()-fold_start)/60:.1f}min")

            mean_rank_corr = float(np.mean(fold_rank_corrs))
            mean_alpha = float(np.mean(fold_alphas))

            trial.set_user_attr("fold_rank_corrs",
                                [float(v) for v in fold_rank_corrs])
            trial.set_user_attr("fold_alphas",
                                [float(v) for v in fold_alphas])
            trial.set_user_attr("mean_alpha", mean_alpha)
            trial.set_user_attr("elapsed_sec", time.time() - trial_start)

            print(f"\n[Trial {trial.number}] rank_corr={mean_rank_corr:+.4f}  "
                  f"alpha={mean_alpha*100:+.1f}%p  "
                  f"elapsed={(time.time()-trial_start)/60:.1f}min")

            return mean_rank_corr

        finally:
            _restore_config(config, originals)

    except TrialTimeoutError:
        print(f"\n[Trial {trial.number}] TIMEOUT ({TRIAL_TIMEOUT_SEC/60:.0f}min)")
        trial.set_user_attr("timeout", True)
        import optuna
        raise optuna.TrialPruned()

    except Exception as e:
        print(f"\n[Trial {trial.number}] ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        import optuna
        raise optuna.TrialPruned()

    finally:
        signal.alarm(0)


# ============================================================
# MAIN
# ============================================================

def main():
    import optuna

    # CLI args
    parser = argparse.ArgumentParser(
        description="v2.3.15 Stage 1 Optuna re-run under heteroscedastic NN + NLL"
    )
    parser.add_argument(
        '--smoke', action='store_true',
        help="Run 1 trial in separate smoke study (smoke_v2315). "
             "Validates pipeline without polluting main study DB.",
    )
    args = parser.parse_args()

    # Smoke mode: redirect to separate study / DB; cap at 1 trial
    if args.smoke:
        study_name = SMOKE_STUDY_NAME
        storage = SMOKE_STORAGE
        n_trials_target = SMOKE_N_TRIALS
        mode_label = "SMOKE (1 trial)"
    else:
        study_name = STUDY_NAME
        storage = STORAGE
        n_trials_target = N_TRIALS
        mode_label = "FULL"

    print("="*70)
    print(f"OPTUNA STAGE 1 v3 (v2.3.15) — 6-dim NLL search [{mode_label}]")
    print("="*70)
    print(f"Study:     {study_name}")
    print(f"Storage:   {storage}")
    print(f"Trials:    {n_trials_target}  (ensemble N={N_ENSEMBLE_STAGE1}, "
          f"folds={STAGE1_FOLD_INDICES})")
    print(f"Excluded:  {sorted(EXCLUDED_TICKERS)}")
    print(f"Loss:      heteroscedastic Gaussian NLL on log-vol target")
    print(f"Search:    lr, weight_decay, architecture, var_thr, corr_thr, dropout")
    print(f"Removed:   huber_delta (no longer applies under NLL)")
    print(f"Timeout:   {TRIAL_TIMEOUT_SEC/60:.0f}min per trial")
    print("="*70)

    if not os.path.exists(CACHE_PATH):
        print(f"\nERROR: {CACHE_PATH} not found. Run `python backtest.py` first.")
        sys.exit(1)

    sampler = optuna.samplers.TPESampler(
        seed=SAMPLER_SEED,
        n_startup_trials=N_STARTUP_TRIALS,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\n[Study] {len(study.trials)} trials, {completed} completed")

    if completed > 0:
        best_trials = [t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE]
        if best_trials:
            best = max(best_trials, key=lambda t: t.value or -1)
            print(f"[Study] Best: rank_corr={best.value:+.4f}  "
                  f"trial #{best.number}")

    remaining = max(0, n_trials_target - completed)
    if remaining == 0:
        print(f"[Study] Target reached ({n_trials_target}). Nothing to do.")
    else:
        print(f"[Study] Running {remaining} more trials...")
        study.optimize(
            objective,
            n_trials=remaining,
            catch=(Exception,),
            show_progress_bar=False,
        )

    # Save top-3 configs for Stage 2 retrain in v2.3.16 (skip in smoke mode)
    if args.smoke:
        print(f"\n{'='*70}")
        print(f"SMOKE COMPLETE: {len(study.trials)} trials (no results JSON written)")
        print(f"{'='*70}")
        return

    completed_trials = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(completed_trials,
                        key=lambda t: t.value or -1, reverse=True)[:3]

    top_configs = []
    for i, t in enumerate(top_trials):
        top_configs.append({
            "rank": i + 1,
            "trial_number": t.number,
            "mean_rank_corr": float(t.value),
            "mean_alpha": t.user_attrs.get("mean_alpha"),
            "fold_rank_corrs": t.user_attrs.get("fold_rank_corrs"),
            "fold_alphas": t.user_attrs.get("fold_alphas"),
            "elapsed_sec": t.user_attrs.get("elapsed_sec"),
            "params": t.params,
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "study_name": study_name,
            "version": "v2.3.15",
            "loss_function": "heteroscedastic_gaussian_nll",
            "n_trials_total": len(study.trials),
            "n_trials_completed": len(completed_trials),
            "n_trials_target": N_TRIALS,
            "best_mean_rank_corr": float(top_trials[0].value) if top_trials else None,
            "top_3_configs": top_configs,
            "stage1_settings": {
                "n_ensemble": N_ENSEMBLE_STAGE1,
                "fold_indices": STAGE1_FOLD_INDICES,
                "excluded_tickers": sorted(EXCLUDED_TICKERS),
                "timeout_sec": TRIAL_TIMEOUT_SEC,
                "sampler_seed": SAMPLER_SEED,
                "search_space": {
                    "lr": "loguniform [1e-4, 3e-3]",
                    "weight_decay": "loguniform [1e-5, 1e-3]",
                    "architecture": list(ARCH_CHOICES.keys()),
                    "var_threshold": "loguniform [1e-3, 1e-1]",
                    "corr_threshold": "loguniform [1e-3, 1e-1]",
                    "dropout": "uniform [0.1, 0.4]",
                },
            },
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"STAGE 1 v3 COMPLETE: {len(completed_trials)}/{N_TRIALS} trials")
    print(f"Results: {RESULTS_PATH}")
    print(f"{'='*70}")
    for c in top_configs:
        print(f"  #{c['rank']} rank_corr={c['mean_rank_corr']:+.4f}  "
              f"alpha={c['mean_alpha']*100:+.1f}%p  {c['params']}")


if __name__ == "__main__":
    main()
