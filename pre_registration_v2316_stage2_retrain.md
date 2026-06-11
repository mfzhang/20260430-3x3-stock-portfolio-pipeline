# Pre-Registration — v2.3.16 Stage 2 Production Retrain at Trial 52 (multi-seed)

**Status**: pre-registered (written before the retrain runs)
**Date**: 2026-06-11
**Depends on**: v2.3.15 Stage 1 verdict (Diverged; T* = Trial 52). See `pre_registration_v2315_optuna_rerun.md` Amendment 3.
**Question**: does the v2.3.15 NLL-search champion (Trial 52) supersede the deployed v2.3.12 production model at N=20 production scale, justifying progression to v2.3.17 walk-forward?

## 1. Purpose

v2.3.15 established (Diverged verdict, 3/6 dimensions) that the v2.3.6-era Trial #58 hyperparameters were tuned under a Huber-loss MLP search and do not transfer to the heteroscedastic dual-head Gaussian-NLL architecture in production since v2.3.12. The pre-registered consequence is to retrain at Trial 52 (the v2.3.15 NLL-search best) at N=20.

The supersession decision is a validation/judgment step, so it is made on **measured variance**, not on a prior. This document fixes — before any run — a multi-seed measurement and a mechanical decision rule whose parity verdict defaults to retaining the incumbent (measurement does not separate the configs → no basis to change).

## 2. Pre-registered run specification (locked)

**Six runs**: two configs × three pre-fixed seeds, paired.

- Seeds: **S ∈ {42, 1, 2}** (pre-registered; 42 is the project-standard seed, the other two arbitrary fixed). Applied identically (paired) to both configs.
- The seed varies model initialization, ensemble-member seeds, and dropout masks only. The **data partition (5 GICS-stratified folds + each fold's train/val split) is held fixed across all six runs**, so the comparison is paired on identical data; seed isolates training stochasticity.

**Config A — champion (v2.3.15 Trial 52)**, from `results/stage1_v2315/best_trials.json` (`--config-rank 1`):
- architecture: large [128, 64, 32]
- lr: 0.001563563963064687
- weight_decay: 4.4856093488331435e-05
- var_threshold: 0.0010819081885486052
- corr_threshold: 0.05584259829572068
- dropout: 0.1146035891599349

**Config B — incumbent baseline (v2.3.12 = old Trial #58)**, from `results/optuna_stage1_results.json` (`--config-rank 1`):
- architecture: medium [64, 32, 16]
- lr: 0.00024955280836145015
- weight_decay: 0.00016413025522015487
- var_threshold: 0.001971224419059394
- corr_threshold: 0.0837536123288791
- dropout: 0.2  (v2.3.12 production default; the old Trial #58 record has no dropout key, so the run falls back to 0.2)

> Note: "old Trial #58" (this baseline) and "Trial 52" (the champion) are from **different studies**. They are not comparable by number.

**Common to all six runs**:
- loss: standard heteroscedastic Gaussian NLL (Kendall & Gal 2017); NOT beta-NLL
- risk target: log-space (Andersen et al. 2003)
- ensemble: N = 20
- folds: 5, GICS-stratified (indices [0,1,2,3,4])
- universe: SNDK excluded
- cache: results/backtest_cache.npz (122,240 × 97 features, 525 tickers, SNDK excluded) — identical to the Stage 1 cache
- epoch cap: 20000 with val-loss early stopping (patience 41)

Per-run primary quantity: **M = mean rank_corr across the 5 folds** (N=20), reported as `aggregate.rank_corr_mean` in each run's `summary.json`.

## 3. Baseline (measured, not assumed)

The decision baseline is the **measured mean of the three Config-B runs**: B̄ = mean over S of M_B(S). It is produced by this study, not carried in.

Sanity reference only: the prior single-seed v2.3.12 figure (0.5021) should land near B̄; a large mismatch flags a setup error. 0.5021 is NOT used in the decision.

The comparison is N=20 vs N=20 throughout. The Stage 1 N=5 figure (0.5732) is the search-scale value and is explicitly NOT a baseline.

## 4. Primary decision rule (mechanical, multi-seed)

Per seed S, paired: **Δ_S = M_A(S) − M_B(S)**.
- mean Δ = mean over S of Δ_S
- R_A = [min, max] of {M_A(S)};  R_B = [min, max] of {M_B(S)}

A **non-parity** verdict (improvement OR regression) requires **all three** gates to hold:
1. |mean Δ| ≥ 0.015
2. sign-consistency: all three Δ_S share the sign of mean Δ
3. range non-overlap: R_A ∩ R_B = ∅

| Verdict | Condition | Action |
|---|---|---|
| **Improvement** | mean Δ ≥ +0.015 AND all Δ_S > 0 AND R_A entirely above R_B | Adopt Trial 52 config as production. Proceed to v2.3.17 walk-forward. |
| **Parity** | any of the three gates fails | **Retain v2.3.12.** Measurement does not separate the configs at this metric/scale → no basis to change. |
| **Regression** | mean Δ ≤ −0.015 AND all Δ_S < 0 AND R_A entirely below R_B | Do NOT adopt. Diagnose (candidates: N=5→N=20 interaction with large architecture; verdict not translating to production scale). v2.3.12 remains production. |

**No prior tie-break.** Unlike a single-seed design, parity here is a *measured* outcome ("configs indistinguishable"), and the conservative response to "indistinguishable" is to retain the incumbent. The v2.3.15 divergence verdict's role was to nominate Trial 52 as the candidate; a measured parity means the candidate is not separable from the incumbent at N=20 — an informative result, not a reason to switch. The decisive deployment gate remains v2.3.17.

## 5. Secondary corroboration (non-binding)

Informs interpretation; does not override §4.

- Selection alpha: 3-seed mean alpha for each config. If §4 yields improvement/parity but A's alpha craters vs B, flag before v2.3.17.
- Per-fold consistency: if a non-parity Δ is driven by a single fold while others move the other way, flag — the aggregate is fragile.

## 6. Outcome consequences

- **Improvement** → update `config.py` to Trial 52 hyperparameters (skip-worktree workflow), commit with the v2.3.16 result, annotate v2.3.7–v2.3.14 docs as "trained under superseded hyperparameters," proceed to v2.3.17 walk-forward.
- **Parity / Regression** → v2.3.12 stays production; do not commit Trial 52 to config.py. (Regression additionally opens a diagnosis sub-task.)

Phase 1 ₩20M deployment remains conditional on v2.3.17 passing its own pre-registered threshold, not this gate.

## 7. Limitations

- n = 3 seeds is a **directional-robustness** check (sign stability + range separation), not an inferential SD; no clean p-value is claimed. The three-gate AND rule (§4) is sized to this limitation — it calls a result only when the signal is large (±0.015), sign-consistent, and range-separated.
- The seed set {42, 1, 2} is pre-fixed; no seed selection after seeing results.
- The data partition is held fixed across runs, so seed variance here is training-stochasticity variance, not data-partition variance. This is deliberate: it isolates the config comparison on identical folds.
- This whole gate is a cross-sectional K-fold rank_corr comparison. The deployment-relevant, out-of-sample, time-ordered test is v2.3.17 walk-forward, which is decisive.

## 8. Commit discipline

Per Asymmetric Validation and the v2315 §11 precedent: this pre-registration is committed **before** the six runs. The result (per-seed M for both configs, mean Δ, range overlap, sign consistency, verdict bucket, adopt/retain/halt) is recorded afterward as **Amendment 1** to this document, in a separate commit that changes no rules.
