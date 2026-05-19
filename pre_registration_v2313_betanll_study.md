# Pre-registration: β-NLL vs Standard NLL comparison study

**Date**: 2026-05-19 09:15 KST
**Triggered by**: v2.3.12 production retrain (commit 3b2f7d0) calibration plot review
**Status**: pre-registered before β-NLL training begins

## Hypothesis

The standard Gaussian NLL trained in v2.3.12 production (commit 3b2f7d0) exhibits the calibration pathology described in Seitzer et al. (2022, ICLR): narrow log-sigma dynamic range with large-sigma underlearning. Specifically, on Fold 2:

- z-score std = 0.216 (ideal: 1.0)
- |z|<1 coverage = 100% (ideal: 68%)
- Sigma tertile |z|: 0.25 / 0.27 / 0.49 (small / med / large; pathology: large-sigma underlearning)
- log-sigma range = [0.309, 0.583] (narrow)

The β-NLL variant (Seitzer et al. 2022) is hypothesized to improve aleatoric calibration at the potential cost of mean prediction quality.

## Experimental design

- **Variable**: loss function only (standard `heteroscedastic_loss` vs `heteroscedastic_loss_beta` with β=0.5)
- **Fixed**: all hyperparameters (Trial #58), seed (numpy/torch fixed), data split, fold assignments, SNDK exclusion, N_ENSEMBLE=20, TRAINING_EPOCHS=5000, patience=41
- **Production base case (do not modify)**: `results/stage2/top1_trial58/` (standard NLL)
- **Comparison cell (β-NLL)**: `results/stage2/top1_trial58_betaNLL/`

## Pre-specified comparison metrics

### Primary (production performance)
- Mean rank_corr across 5 folds (without SNDK)
- Threshold for "no degradation": ΔrankCorr ≥ -0.01

### Secondary (calibration quality)
- z-score std on Fold 2 (target: [0.7, 1.3])
- |z|<1 coverage on Fold 2 (target: 60-75%)
- Sigma tertile |z| gap = |z|_large - |z|_small (target: < 0.15; current standard NLL: 0.24)

### Tertiary (deployment impact)
- Top-5 selection identity per fold
- Universe NN risk mean across folds (current standard NLL: 25.6-27.5%; should remain 20-35%)

## Pre-registered decision rule

After β-NLL retrain completes, evaluate against fixed thresholds:

| β-NLL rank_corr | β-NLL calibration | Decision |
|---|---|---|
| ≥ Standard - 0.01 | z-std ∈ [0.7, 1.3] AND tertile gap improved | **Adopt β-NLL as v2.3.13 production** |
| ≥ Standard - 0.01 | Calibration unchanged or worse | Keep standard NLL (parsimony) |
| < Standard - 0.01 | Any | Keep standard NLL (v2.3.12 remains production) |
| Other tied cases | — | Keep standard NLL (default = simpler model) |

The β-NLL result will be archived in either outcome under `results/stage2/top1_trial58_betaNLL/` for reproducibility.

## What is NOT being decided

- Hyperparameter changes (Trial #58 frozen)
- Architecture changes (HeteroscedasticDualHeadNN frozen)
- Feature changes (Optuna-selected feature filter frozen)
- Ensemble size (N=20 frozen)
- SNDK policy (excluded, same as production)

Only the loss function is being varied.

## References

- Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NeurIPS.
- Seitzer et al. (2022) "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks." ICLR.
- Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS. (precedent for head-to-head comparison study design)


---

## Amendment 1 (2026-05-19): Pre-launch smoke comparison design

Before launching the full β-NLL production retrain (5-fold, N=20), a smoke
comparison is run to validate that β-NLL is numerically stable and produces
qualitatively different calibration behavior than standard NLL. This amendment
is committed BEFORE the smoke runs to prevent post-hoc design changes.

### Design

Variable: loss function ∈ {standard NLL, β-NLL with β=0.5}
Other variables (FIXED): Fold 2 only, N_ENSEMBLE=1, all hyperparameters from
Trial #58, TRAINING_EPOCHS=2000, patience=41.
Seeds: numpy/torch seed ∈ {42, 43, 44} for each loss = 6 total runs.

### Pre-specified metrics (Fold 2 test set)

For each (loss, seed) combination, compute:
1. Mean test rank_corr
2. Best val_NLL achieved
3. log-sigma range (max - min over test set)
4. z-score std = std((log_actual - log_pred_mean) / log_pred_sigma)
5. Sigma tertile |z| gap = |z|_large_third - |z|_small_third

### Smoke decision rule

Aggregate over 3 seeds per loss. Compute mean ± std per metric per loss.

Launch full β-NLL production retrain if AND ONLY IF:

(a) No NaN, no divergence in any of 3 β-NLL runs.
(b) Mean β-NLL rank_corr (smoke) ≥ Mean standard NLL rank_corr (smoke) - 0.05.
    Threshold is loose because N=1 smoke is noisy; this prevents launching only
    if β-NLL is catastrophically worse for ranking.
(c) β-NLL z-score std (smoke mean) shows differentiation from standard NLL,
    i.e. |β-NLL z-std - standard z-std| > 0.1 in either direction.
    Rationale: if smoke z-std are identical, β-NLL is not behaving differently
    in any measurable way, and the production retrain would just reproduce
    standard NLL results. Better to abort and save 5 hours.

If any of (a), (b), (c) fails, the production retrain is NOT launched, and
this amendment + smoke results are committed as a null-result finding.

If all three pass, the production retrain is launched.

### Why N=3 seeds and not N=1 or N=5

N=1 gives no variance estimate — cannot tell if a single-seed difference is
signal or noise.
N=3 is the minimum for any variance estimate (~30 min total smoke time).
N=5+ would be ideal for formal hypothesis testing but is reserved for the
production retrain itself (N_ENSEMBLE=20 there provides better variance
characterization across folds).


---

## Amendment 2 (2026-05-19, post-smoke-run-1): seed override patch required

### Findings from smoke run 1 (commit 6271f59)

Smoke run 1 was executed with 3 nominal seeds {42, 43, 44} per loss as per
Amendment 1. Results showed **zero variance across seeds**:

  standard   seed=42: rc=+0.595293 z_std=0.203345
  standard   seed=43: rc=+0.595293 z_std=0.203345
  standard   seed=44: rc=+0.595293 z_std=0.203345
  beta_nll   seed=42: rc=+0.628706 z_std=0.280078
  beta_nll   seed=43: rc=+0.628706 z_std=0.280078
  beta_nll   seed=44: rc=+0.628706 z_std=0.280078

Smoke run 1 results archived at `smoke_v2313_results_run1_seedfail.json`.

### Root cause

Lines 400-401 of `stage2_retrain.py` (in `run_fold_with_plot`):

    torch.manual_seed(SEED + fold_id * 100 + nn_idx)
    np.random.seed(SEED + fold_id * 100 + nn_idx)

Re-seed at each NN training start using a deterministic formula
(`SEED=42 + fold_id*100 + nn_idx`). This is a deliberate design choice to
ensure production reproducibility across runs — but it overrides any
caller-provided seed before NN training. Smoke run 1's `np.random.seed(42/43/44)`
in `run_one()` was effectively discarded.

This is **not a code bug** in the production sense; it is a reproducibility
mechanism that conflicts with our smoke study's variance-estimation goal.

### Decision rule consequence (without amendment 2)

Run 1's effective N=1 measurement gives:

- Criterion (a) No NaN: True ✓
- Criterion (b) rank_corr Δ = +0.033 ≥ -0.05: True ✓
- Criterion (c) |Δz_std| = 0.077 > 0.10: **False ✗**
- LAUNCH: False (pre-registered)

If accepted as-is, this would conclude the study with a null result. But the
variance threshold for criterion (c) was designed assuming N=3 seed estimates;
applying it to an effective-N=1 point estimate is a methodological violation
of Amendment 1's intent.

### Patch plan

Modify `run_fold_with_plot` to accept an optional `seed_override` parameter:

    def run_fold_with_plot(..., seed_override=None):
        ...
        # In the NN training loop (replacing lines 400-401):
        if seed_override is None:
            effective_seed = SEED + fold_id * 100 + nn_idx
        else:
            effective_seed = seed_override * 1000 + fold_id * 100 + nn_idx
        torch.manual_seed(effective_seed)
        np.random.seed(effective_seed)

Production behavior is byte-identical when `seed_override=None` (the
production main() does not pass this argument).

Smoke run 2 will pass `seed_override=42, 43, 44` explicitly, allowing the
formula to produce three genuinely distinct effective seeds.

### Decision rule (unchanged from Amendment 1)

Thresholds remain:
- (a) No NaN in β-NLL runs
- (b) β-NLL rank_corr ≥ standard - 0.05
- (c) |β-NLL z_std mean - standard z_std mean| > 0.10

What changes:
- "z_std mean" now is a real mean across 3 distinct seeds (with non-zero std).
- All other criteria unchanged.

### Honest acknowledgment

Run 1's z_std Δ was +0.077, below the 0.10 threshold. If Run 2 with real N=3
gives Δ in [0.07, 0.13] range, the decision could flip on natural variance.
This is an acceptable scientific outcome — amendment 1's threshold was chosen
ex ante as the magnitude needed to justify a 5-hour production retrain.
Variance-induced uncertainty on the decision is part of the experimental design.

If Run 2 still gives Δ < 0.10 with real variance, the LAUNCH decision remains
False and v2.3.12 production remains final.

### Why this amendment, not a quiet code fix

The Run 1 result is preserved in `smoke_v2313_results_run1_seedfail.json`
and this amendment timestamps the discovery + patch design BEFORE Run 2.
Any future reviewer can verify: Amendment 2 acknowledges Run 1's
specific result and pre-commits to a Run 2 with corrected methodology
under unchanged decision thresholds.

