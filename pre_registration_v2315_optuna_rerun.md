# Pre-registration: v2.3.15 Optuna Stage 1 re-run under heteroscedastic dual-head NN + Gaussian NLL

**Author**: Ki Heon Lee
**Date**: 2026-05-19 (v2.3.15 open)
**Project**: `keyheon/3x3-stock-portfolio-pipeline`
**Reference HEAD at pre-registration**: `4c8ebe6`
**Governing principle**: `ASYMMETRIC_VALIDATION_PRINCIPLE.md` (Operational rules §1–§7)

This document pre-registers the design, search space, decision rules, and ex ante
predictions for the v2.3.15 Optuna Stage 1 re-run. It must be committed to the main
branch **before** any code changes for `optuna_search_v3.py` are made. Per the
Asymmetric Validation Principle §2, the criteria below are fixed at commit time
and may only be modified by explicit, dated amendments to this file.

---

## 0. Context and motivation

The v2.3.6 Optuna study (`optuna_search.py`, study `stage1_nn_feat_6dims`,
storage `optuna_storage.db`) ran 60 trials of TPE Bayesian optimization over a
6-dim search space under **Huber loss** on raw 3-month volatility targets. The
best trial (#58) hyperparameters have driven all downstream work through v2.3.14.

The v2.3.12 architecture refactor replaced the single-head MLP + Huber loss with
a **heteroscedastic dual-head NN + Gaussian NLL** on log-transformed volatility
targets. From Trial #58, the following hyperparameters are propagated into the
new architecture: `lr=2.5e-4`, `weight_decay=1.64e-4`, `architecture=medium`
([64, 32, 16]), `var_threshold=0.00197`, `corr_threshold=0.0838`. The Trial #58
`huber_delta=0.5` is loaded but ignored (NLL has no delta parameter).

This creates a hyperparameter–loss mismatch: the values came from a loss surface
that is no longer used. Under the Asymmetric Validation Principle §1, the
conservative default is to re-run the search under the production loss. This
study (v2.3.15) executes that re-run.

The v2.3.12 production output at `results/stage2/top1_trial58/` is the validation
target: rank_corr_mean = 0.5021, selection_alpha_mean = +7.61%p, n_ensemble = 20,
n_folds_used = 5, sndk_excluded = True, total_elapsed_min = 249.0.

---

## 1. Hypothesis

- **H0** (alternative): The hyperparameters identified by v2.3.6 Optuna under
  Huber loss (Trial #58) are not optimal under heteroscedastic Gaussian NLL.
  A re-run will identify a meaningfully different configuration on at least
  one dimension.
- **H1** (null): Trial #58 hyperparameters are robust to the loss change.
  The re-run identifies essentially the same configuration on every dimension.

Both outcomes are scientifically informative. H1 retroactively validates the
v2.3.6 hyperparameter selection; H0 reveals that v2.3.7–v2.3.14 results were
conditioned on a misspecified loss and the new best should supersede.

Per the Asymmetric Validation Principle §5, a null result (H1) is accepted as
the result and not subjected to rescue attempts.

---

## 2. Search space (6 dimensions, fixed at this commit)

| # | Parameter | Type | Range / Set | Notes |
|---|---|---|---|---|
| 1 | `lr` | loguniform | [1e-4, 3e-3] | Same as v2.3.6 |
| 2 | `weight_decay` | loguniform | [1e-5, 1e-3] | Same as v2.3.6 |
| 3 | `architecture` | categorical | {`small` [32, 16], `medium` [64, 32, 16], `large` [128, 64, 32]} | Same as v2.3.6 |
| 4 | `var_threshold` | loguniform | [1e-3, 1e-1] | Same as v2.3.6 |
| 5 | `corr_threshold` | loguniform | [1e-3, 1e-1] | Same as v2.3.6 |
| 6 | `dropout` | uniform | [0.1, 0.4] | **NEW**: was hardcoded 0.2 in v2.3.6/v2.3.12. NLL-relevant. |

**Removed from v2.3.6 search space**:
- `huber_delta` — dead parameter under NLL loss. Loaded but unused.

**Intentionally NOT in search space** (deferred to future studies):
- `N_ENSEMBLE` — fixed at 5 for Stage 1, retrain at 20 in v2.3.16.
- `log_sigma_init` — deferred to v2.3.16+ if borderline issues emerge.
- `TRAINING_EPOCHS` — fixed at 5000 (v2.3.6 §54 6000-ep diagnostic).
- `EARLY_STOP_PATIENCE` — fixed at 41 (v2.3.8 unified).
- Optimizer choice — Adam only.
- LR schedule — cosine annealing hardcoded.
- Composite-score coefficients — reserved for future Task #9.

---

## 3. Trial budget, folds, and training configuration

| Parameter | Value | Source |
|---|---|---|
| N_TRIALS | 60 | v2.3.6 convention |
| N_ENSEMBLE_STAGE1 | 5 | v2.3.6 convention; production uses 20 in v2.3.16 |
| FOLDS | `[1, 2, 3, 4]` (0-indexed = Fold 2–5) | v2.3.6 convention. Fold 1 fold-level exclusion to avoid SNDK post-IPO artifact dominating selection. |
| TRAINING_EPOCHS | 5000 | v2.3.6 §54 diagnostic |
| EARLY_STOP_PATIENCE | 41 | v2.3.8 unified |
| TRIAL_TIMEOUT_SEC | 7200 (120 min) | v2.3.6 convention |
| Seed | 42 | Project convention (§3.5.9) |
| Sampler | Optuna TPE | seed=42, n_startup_trials=10 |
| Loss function | `heteroscedastic_loss` (Gaussian NLL on log-vol target) | v2.3.12 production |
| Target | `log(realized 3-month forward volatility)` | v2.3.12 production |
| Architecture base class | `HeteroscedasticDualHeadNN` (mean + log_sigma heads, shared backbone) | v2.3.12 production |

**Note on fold convention**: v2.3.15 Optuna uses fold-level Fold-1 exclusion
(consistent with v2.3.6 hyperparameter search). The downstream v2.3.16 Stage 2
retrain will use all 5 folds + SNDK ticker-level exclusion (consistent with
v2.3.12 production). These differ intentionally — fold-level exclusion is
appropriate for hyperparameter selection robustness; ticker-level exclusion is
appropriate for production evaluation.

---

## 4. Primary metric

**Mean Spearman rank correlation across folds [1, 2, 3, 4]**, identical to v2.3.6
objective. Computed per trial; Optuna maximizes.

Per Asymmetric Validation Principle §3, this is the single mechanical metric
used for trial ranking and best-trial selection. Auxiliary metrics
(selection_alpha, top-5 returns, calibration diagnostics) are computed and
logged but are exploratory per Principle §6 — they may not be substituted for
or combined with rank_corr in the best-trial decision.

---

## 5. Acceptance / comparison rule (mechanical, locked at this commit)

After the 60-trial study completes, identify `T*` = the trial with the highest
mean rank_corr. Compare `T*` hyperparameters dimension-by-dimension against
Trial #58 baseline.

**Trial #58 reference values**:

| Dimension | Trial #58 value |
|---|---|
| lr | 2.5e-4 |
| weight_decay | 1.64e-4 |
| architecture | medium |
| var_threshold | 0.00197 |
| corr_threshold | 0.0838 |
| dropout | 0.2 (was hardcoded — v2.3.15 reference value) |

**Per-dimension "diverged" definition (mechanical)**:

| Dimension | Scale | "Diverged" condition | Robust range |
|---|---|---|---|
| lr | log | `\|log10(T*.lr / 2.5e-4)\| > 0.3` | [1.25e-4, 5.00e-4] |
| weight_decay | log | `\|log10(T*.wd / 1.64e-4)\| > 0.3` | [8.21e-5, 3.27e-4] |
| var_threshold | log | `\|log10(T*.var / 0.00197)\| > 0.3` | [9.87e-4, 3.93e-3] |
| corr_threshold | log | `\|log10(T*.corr / 0.0838)\| > 0.3` | [4.20e-2, 1.67e-1] |
| architecture | categorical | `T*.arch != 'medium'` | {medium} |
| dropout | linear | `\|T*.dropout − 0.2\| > 0.1` | [0.1, 0.3] |

**Aggregate decision rule**:

- **"Robust"**: ALL six dimensions are within their robust range. Conjunction.
- **"Diverged"**: ANY single dimension falls outside its robust range.

**Disjunctive (conservative) framing**: the burden falls on demonstrating robustness
across every dimension. A single divergent dimension flips the verdict to
"Diverged".

**Special clarification (per Phase B Q3 confirmation)**: If a feature-selection
threshold (`var_threshold` or `corr_threshold`) falls outside its robust range
but the **resulting number of selected features** is similar to Trial #58
(e.g., 52 vs 55 features), the verdict is still **"Diverged"**. Different
thresholds may admit different feature sets even at similar counts; the
mechanical rule applies as written. Per Asymmetric Validation Principle §3,
post-hoc reframing is disallowed.

**Decision applied to v2.3.16**:

- **Robust** → v2.3.16 Stage 2 retrain proceeds using `T*` (which should produce
  near-identical results to v2.3.12 production), confirming v2.3.7–v2.3.14
  hyperparameters were indeed loss-robust.
- **Diverged** → v2.3.16 Stage 2 retrain proceeds using `T*`. v2.3.12 production
  is superseded and v2.3.7–v2.3.14 results carry an explicit "trained under
  superseded hyperparameters" annotation in subsequent documentation.

**Unconditional downstream commitment (Phase B Q4)**: v2.3.16 Stage 2 retrain
is run regardless of the Robust/Diverged verdict. Skipping retrain in the
Robust case would leave the actual production model trained under the v2.3.6
Optuna run's specific TPE noise rather than under a coherent NLL-optimized
configuration. The marginal cost (~5–6h) is small relative to the
self-consistency benefit.

---

## 6. Cache, storage, and infrastructure strategy

| Item | Path / Value | Status |
|---|---|---|
| Training cache | `results/backtest_cache.npz` | **Reuse unchanged**. 97 features, architecture-independent. |
| Old Optuna study | `optuna_storage.db` | **Do not modify**. Preserve v2.3.6 study history. |
| Old Optuna script | `optuna_search.py` | **Do not modify**. Preserve v2.3.6 design history. |
| New Optuna study DB | `optuna_storage_v2315.db` | **New**. Already gitignored via `optuna_storage_v*.db`. |
| New Optuna study name | `stage1_nn_feat_6dims_NLL` | 6 dims (huber_delta replaced by dropout). |
| New Optuna script | `optuna_search_v3.py` | **New file**, separate commit after this pre-registration. |
| New Optuna log | `optuna_v2315.log` (full), `optuna_smoke_v2315.log` (smoke) | Gitignored via `*.log`. |
| Stage 1 output dir | `results/stage1_v2315/` | **New**. For top-3 trials JSON dump. Gitignored via `results/`. |
| v2.3.12 production output | `results/stage2/top1_trial58/` | **DO NOT TOUCH**. Validation reference. |
| Loss function | `heteroscedastic_loss` from `models.py` | v2.3.12 production. β-NLL not used (rejected in v2.3.14). |

**Resume safety**: SQLite persistent storage with `load_if_exists=True` means
any interrupted run resumes from the last completed trial. Re-launching the
same command is the recovery procedure.

**Adam weight_decay monkey-patch**: same pattern as v2.3.6 (`backtest.py` line
306 hardcodes `1e-4`; `optuna_search_v3.py` wraps `optim.Adam` to inject the
trial-specific value). Verified post-Stage-1 by inspecting one or two completed
trials' actual training trajectories.

---

## 7. Ex ante observations (predictions before result is known)

Per Asymmetric Validation Principle §6, these are recorded **before** the
study runs to prevent post-hoc narrative construction. They are predictions,
not criteria.

1. **Convergence under NLL**: NLL is well-defined on log-transformed targets
   with the dual-head architecture; no NaN issues expected. If NaN appears in
   smoke testing, the response is to diagnose the bug, not soften the loss
   (Principle §5).

2. **lr and weight_decay**: Expected to land near Trial #58 values (within
   the robust range), because NLL's loss-landscape curvature differs from
   Huber's primarily in tail-error weighting, not in overall scale. Strong
   shift on either dimension would suggest the heteroscedastic head needs
   meaningfully different regularization than the single-head MLP did.

3. **Architecture**: `medium` may not remain optimal. The heteroscedastic
   dual-head benefits from sufficient capacity for the log_sigma head to
   represent per-sample uncertainty; a larger architecture (large [128, 64, 32])
   could now be competitive where it was rejected in v2.3.6 under Huber.

4. **Feature selection thresholds**: Expected to land near v2.3.6 values
   (var_thr ≈ 0.002, corr_thr ≈ 0.084) because these are loss-independent —
   they filter features by their statistical relationship with the target
   before the loss function is invoked.

5. **Dropout**: New search dimension. Prior is unknown. Possible outcomes:
   (a) lands near 0.2 (the v2.3.6/v2.3.12 hardcoded default) — confirms the
   default was reasonable; (b) lands meaningfully higher (~0.3+) — suggests
   heteroscedastic NN benefits from stronger regularization; (c) lands lower
   (~0.1) — suggests the dual-head structure already provides regularization
   and explicit dropout was over-regularizing.

6. **Mean rank_corr at best trial**: Expected in the range [0.49, 0.54] (broad
   band around v2.3.12 production's 0.502). Strong improvement (>0.55) or
   strong regression (<0.47) at the best trial would be surprising and warrant
   careful diagnostic examination before adoption.

7. **TPE convergence pattern**: With 10 startup trials of random sampling
   followed by 50 TPE-guided trials, the best trial is typically found in
   the 30–55 range. Trial #58 of v2.3.6 (the v2.3.6 best) is consistent
   with this pattern.

---

## 8. Sub-question pre-commitments (Phase B decisions)

These decisions were made during v2.3.15 session open before any v3 code was
written. They are locked into this pre-registration:

- **Q1 (dropout in search)**: YES. `uniform [0.1, 0.4]`. Added because v2.3.12
  hardcoded 0.2 without empirical justification under the new architecture.
  Confidence: medium.
- **Q2 (log_sigma_init in search)**: NO. PyTorch default Linear init retained.
  Deferred to v2.3.16+ if v2.3.15 smoke or initial trials reveal NaN/divergence
  issues. Confidence: medium.
- **Q3 (diverged threshold mechanism)**: `|log10(new/old)| > 0.3` for continuous
  log-scale dims, `|Δ| > 0.1` for dropout (linear), any change for architecture
  (categorical). Conjunction across dims for "Robust"; ANY divergence flips
  to "Diverged". Confidence: strong.
- **Q4 (v2.3.16 retrain conditional or unconditional)**: UNCONDITIONAL. v2.3.16
  Stage 2 retrain runs at N=20 with `T*` hyperparameters regardless of v2.3.15
  verdict. Confidence: strong.

---

## 9. What this study cannot answer

Per Asymmetric Validation Principle §4 (reviewer skepticism), the following
questions are explicitly outside v2.3.15's scope. They require follow-up
studies that v2.3.15 may motivate but does not address:

- **Walk-Forward (time-axis) hyperparameter sensitivity**: v2.3.15 uses
  Stratified K-Fold (cross-sectional) splits. Walk-Forward CV re-validation
  is v2.3.17.
- **Composite-score coefficient tuning**: `UNCERTAINTY_PENALTY`,
  `SENTIMENT_WEIGHT_IN_SCORE`, `EVENT_RISK_PENALTY` are held fixed.
  Pending Task #9.
- **Architecture family extensions**: only the three v2.3.6 size presets
  (small/medium/large) are tested. Width-vs-depth tradeoffs, residual
  connections, larger heads — out of scope.
- **Optimizer alternatives**: Adam only. AdamW, RAdam, etc. — out of scope.
- **LR schedule alternatives**: cosine annealing fixed. Constant LR, warmup,
  step decay — out of scope.
- **β-NLL re-test**: Rejected in v2.3.14 per Amendment 3 acceptance rule.
  Not reopened by v2.3.15.

---

## 10. Downstream commitments

If this pre-registration is committed and `optuna_search_v3.py` subsequently
implements the design above:

- **v2.3.16** (conditional on v2.3.15 completion): Stage 2 retrain at N=20 using
  `T*` hyperparameters; ~5–6h wall-clock. Output directory:
  `results/stage2/top1_v2315/` (separate from v2.3.12 production at
  `top1_trial58/`).
- **v2.3.17** (conditional on v2.3.16 completion): Walk-Forward CV re-validation
  using `T*` hyperparameters; ~10h. Confirms that hyperparameters optimal under
  cross-sectional split also generalize across time. If WF-CV result diverges
  materially from K-Fold, that's a finding to document, not to rescue.
- **v2.4.0** (conditional on v2.3.15–v2.3.17 all passing coherently): version
  bump signaling that the full re-audit chain is self-consistent under the
  v2.3.12 architecture/loss.
- **v2.4.x Phase 1 deployment** (conditional on v2.3.17 passing): ₩20M
  brokerage entry.

Per Asymmetric Validation Principle §7, any failure mode at v2.3.15/16/17
that reveals the development direction was wrong (e.g., new best produces
materially worse rank_corr, or WF-CV diverges sharply) triggers an honest
documentation pass and pivot decision — not a relaxation of validation
standards to allow deployment.

---

## 11. Amendment procedure

Per Asymmetric Validation Principle §2, this pre-registration may only be
modified by explicit, dated, numbered amendments appended below this section.
Each amendment must state:

- What was discovered that triggered the amendment
- What rule changes
- Why the change does not undermine the validation logic of the original
  pre-registration

Amendments are themselves Git-committed (each as a separate commit) and
trigger the GitHub 7-point review.

---

## Appendix A: Reference paths and commands

### Local paths (My environment)
```
~/MLDLpythonMath_course/stock_planning/pkg_v2_release/
├── optuna_search.py                          (v2.3.6, do not modify)
├── optuna_search_v3.py                       (v2.3.15, to be created)
├── optuna_storage.db                         (v2.3.6 study, do not modify)
├── optuna_storage_v2315.db                   (v2.3.15 study, gitignored)
├── pre_registration_v2315_optuna_rerun.md    (this file)
├── results/
│   ├── backtest_cache.npz                    (97 features, reused unchanged)
│   ├── stage2/top1_trial58/                  (v2.3.12 production, do not modify)
│   └── stage1_v2315/                         (v2.3.15 output, to be created)
└── config.py                                 (skip-worktree; local API keys preserved)
```

### Launch commands (after smoke test passes)
```bash
# Smoke (1 trial, ~15-30 min)
caffeinate -i python -u optuna_search_v3.py --smoke 2>&1 | tee optuna_smoke_v2315.log

# Full (60 trials, ~30-45h)
caffeinate -i python -u optuna_search_v3.py 2>&1 | tee optuna_v2315.log
```

### Status check (during run)
```python
python -c "
import optuna
from optuna.trial import TrialState
study = optuna.load_study(
    study_name='stage1_nn_feat_6dims_NLL',
    storage='sqlite:///optuna_storage_v2315.db')
completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
print(f'Completed: {len(completed)}/60')
if completed:
    best = max(completed, key=lambda t: t.value if t.value is not None else -1)
    print(f'Best trial: #{best.number}  rank_corr={best.value:+.4f}')
    print(f'Best params: {best.params}')
"
```

---

**End of pre-registration v2315.**
