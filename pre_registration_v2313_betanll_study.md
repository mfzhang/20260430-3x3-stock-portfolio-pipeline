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

