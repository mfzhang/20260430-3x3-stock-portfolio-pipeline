# 3x3 Portfolio Optimization

A deep-learning pipeline I built to help me allocate ~KRW 20M across US equities. It screens a universe of ~84 tickers across 7 sectors, trains an ensemble on roughly 120K cross-sectional samples drawn from S&P 500 + NASDAQ-100 history, adds macro and sentiment features, and outputs a 3x3 (time horizon × risk tier) allocation matrix for 5 selected stocks.

I come from a neuroscience / cognitive engineering / neuroimaging background, not finance, so this repo is also how I learn quantitative investing from first principles. Treat it accordingly: it's a working system that I actually use, but it's a personal project, not a professional product.

## Latest results (v2.3.7)

Stage 2 production retrain with Optuna-tuned hyperparameters and a 5-fold stratified K-fold backtest at production scale (N=20 ensemble, 526 training tickers). Sensitivity analysis run with and without SNDK (a 26.5σ post-IPO outlier from the 2024-10 WDC spinoff) to isolate its effect.

| Metric | Without SNDK (primary) | With SNDK (sensitivity) |
|---|---|---|
| Rank correlation | 0.518 ± 0.079 | 0.521 ± 0.078 |
| α vs universe equal-weight | +7.1%p ± 1.5%p | +14.8%p ± 14.0%p |
| α vs SPY (cap-weighted) | +8.1%p ± 1.7%p | +16.0%p ± 16.6%p |
| α vs proper momentum | +5.87%p ± 2.63%p | +6.34%p ± 2.57%p |
| NN beats momentum | 5/5 folds | 5/5 folds |

The without-SNDK numbers are reported as primary because the with-SNDK aggregate has 10× wider standard deviation, dominated by Fold 1 alone (where SNDK contributes +37%p of the +44.9%p alpha). The momentum-comparison edge (+5.87 to +6.34%p) is the most robust metric — nearly identical across configurations, 5/5 folds, all with similar standard deviation.

Headline hyperparameters from Optuna (60-trial TPE search over 6 dimensions, Trial #58 best): `medium [64,32,16] architecture`, `lr=2.5e-4`, `huber_delta=0.5`, `weight_decay=1.6e-4`, `var_threshold=0.002`, `corr_threshold=0.084`. See [Hyperparameter optimization](#hyperparameter-optimization-stage-1) and [Stage 2 production retrain](#stage-2-production-retrain) sections below for details.

**Subsequent work**: v2.3.11 adds a [transaction cost analysis](#stage-1--transaction-cost-analysis-v2311) calibrated to the Korean broker route (Korea Investment & Securities). v2.3.12 refactors the NN to a [heteroscedastic dual-head architecture](#v2312--heteroscedastic-nn-with-log-volatility-target) trained with Gaussian NLL on log-transformed volatility, addressing systematic risk over-prediction observed in v2.3.7 production output. Smoke tests passed across two architectures; production retrain in progress at the time of writing.

## What the pipeline does

1. **Screen the investment universe** (`screener.py`). Auto-discovers seeds from S&P 500 + NASDAQ-100 using GICS industry matching, combines with a small list of niche anchor tickers, and filters to ~84 stocks across 7 sectors (AI Compute, Neuromodulation, CNS Pharma, Digital Health, Space/Aerospace, Solar/Clean Energy, ETF benchmarks).
2. **Collect sentiment** (`sentiment.py`). Four layers: news headlines via FinBERT, SEC EDGAR filings, FDA + ClinicalTrials.gov events, and earnings surprises. Produces 22 sentiment features per ticker.
3. **Train on history** (`historical.py`, `training_universe.py`). Builds ~120K training samples from S&P 500 + NASDAQ-100 with 10 years of per-ticker snapshots, augmented with FRED macro series, Fama-French 5 factors, and cross-asset features (VIX, treasuries, gold, oil, USD). Trains an ensemble of 20 PyTorch networks with Huber loss (Optuna-tuned hyperparameters since v2.3.7).
4. **Blend weights data-driven** (`blend_optimizer.py`). Finds the optimal mix of NN predictions and analyst consensus via a multi-window backtest (3m/6m/9m) with regime detection and bounded shrinkage toward a prior.
5. **Select top 5** via a composite score that combines predicted Sharpe, MC Dropout confidence, uncertainty penalty, sentiment boost, and event-risk penalty.
6. **Build the 3x3 allocation matrix** (`models.py`). A small neural network with a differentiable Sinkhorn layer that satisfies row (time horizon) and column (risk tier) marginal constraints, trained end-to-end with a Kahneman-Tversky asymmetric portfolio loss.
7. **(Optional) Stratified K-Fold Portfolio Backtest** (`backtest.py`). Validates whether the model's picks actually outperform on unseen tickers. Ticker-axis K-fold (stratified by GICS sector) plus cross-sector transfer tests.

## Quickstart

```bash
pip install numpy matplotlib yfinance pandas torch transformers scipy optuna
pip install fredapi  # optional, for FRED macro features

# Full pipeline
python run.py --torch --screen --sent

# Portfolio backtest only
python run.py --backtest
# or equivalently:
python backtest.py

# Pipeline + backtest
python run.py --torch --screen --sent --backtest

# Hyperparameter optimization (Stage 1) — ~36-45h
python optuna_search.py

# Stage 2 production retrain with Optuna-best hyperparameters — ~1.9h
python stage2_retrain.py                  # without SNDK (primary)
python stage2_retrain.py --include-sndk   # with SNDK (sensitivity)

# Auxiliary analysis tools
python compute_momentum_baseline.py       # NN vs proper-momentum baseline
python fix_spy_benchmark.py               # SPY 3-month forward benchmark
python transaction_cost_analysis.py       # KIS broker TC sensitivity grid

# Individual modules
python screener.py
python sentiment.py
python training_universe.py
python historical.py       # Walk-Forward CV
```

Before running, get a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) and put it in `config.py` as `FRED_API_KEY`. Without it you lose ~13 macro features but the pipeline still runs.
Optionally, get a free [Finnhub API key](https://finnhub.io) and set `FINNHUB_API_KEY` in `config.py` too. This adds a secondary news source on top of yfinance for the sentiment layer. Without it you still get news from yfinance, SEC filings, FDA events, and clinical trials, so sentiment works fine — Finnhub is nice-to-have, not required.

## Repository layout

```
run.py                          Entry point
config.py                       Hyperparameters
screener.py                     7-sector universe screening with auto seed discovery
sentiment.py                    Multi-layer sentiment (news/SEC/FDA/trials/earnings)
training_universe.py            S&P 500 + NASDAQ-100 + FRED + Fama-French + cross-asset
historical.py                   Training data builder + Walk-Forward CV
blend_optimizer.py              Multi-window backtest + regime detection + shrinkage
backtest.py                     Stratified K-Fold portfolio backtest
optuna_search.py                Stage 1 hyperparameter search (60-trial TPE, 6 dims)
stage2_retrain.py               Stage 2 production retrain (N=20, 5-fold, SNDK exclusion)
compute_momentum_baseline.py    Standalone momentum baseline (apples-to-apples)
fix_spy_benchmark.py            SPY 3-month forward benchmark for stratified K-fold
models.py                       Matrix Network + differentiable Sinkhorn
data_auto.py                    yfinance data collection
visualize.py                    Auto-generated figures
requirements.txt                Dependencies
```

## Why stratified K-fold instead of time-axis backtest

Time-axis splits don't fit this task. My model isn't a time-series forecaster asking "what happens next?" — it's a cross-sectional ranker asking "given current features, which stocks will outperform over the next 3 months?". Training on 2014–2020 and testing on 2024 mixes feature-pattern evaluation with regime change (the AI era didn't even exist before 2023), which confounds the two sources of error.

Stratified K-fold (by GICS sector) holds the time period fixed and splits on the ticker axis: train on ~420 tickers, test on ~107, all spanning the same years. This isolates the question I actually care about — does the model generalize to tickers it hasn't seen? — without conflating it with regime drift.

## Why multi-window blend optimization

A naive single-window optimizer told me to weight NN predictions at 95–100% because analyst rank correlation was around -0.8 in a 3-month window. That number isn't a signal that analysts are bad; it's a structural artifact. Analyst targets are 12-month forward forecasts, but I was measuring them against past 3-month realized returns. Mean reversion means stocks that fell hard recently have the largest apparent upside, which produces a negative Spearman with recent actuals by construction.

`blend_optimizer.py` handles this by running the backtest at 3m / 6m / 9m windows, detecting whether the analyst signal is unstable across windows, and shrinking the optimized weight toward a prior (`w_ret = 0.30`) when instability is detected. Bounded weights (`w_ret ∈ [0.15, 0.60]`) prevent the optimizer from picking extremes when the data is noisy.

## Baseline comparisons

The headline alpha numbers are measured against the test-universe mean, but that's only one baseline. To check whether the model beats naive strategies, the Stage 2 retrain compares NN top-5 picks against four baselines. Results below are from the Stage 2 v2.3.7 measurement (N=20 ensemble, 5-fold stratified K-fold). All baselines are computed apples-to-apples on the same fold splits and same data.

| Baseline | Construction | Edge over baseline (5-fold mean ± std) |
|----------|--------------|---------------------------------------|
| **Random 5** | 1,000 random 5-ticker selections per fold, 95% CI from the empirical distribution (measured in v2.3.3 baseline run, commit 9567fc3) | NN alpha is outside the 95% CI in **5/5 folds** (empirical p < 0.0001 in every fold) |
| **Universe equal-weight** | Mean realized return of all test tickers in the fold | **+7.1%p ± 1.5%p** without SNDK / +14.8%p ± 14.0%p with SNDK |
| **SPY (cap-weighted)** | SPY's mean 3-month forward return averaged over the fold's date range, ~495 weekly anchor dates per fold | **+8.1%p ± 1.7%p** without SNDK / +16.0%p ± 16.6%p with SNDK |
| **Proper momentum top-5** | For each test ticker, split snapshots into early (signal) and late (realized) halves; pick top-5 by early-half mean return; measure realized return on the late half. No look-ahead | **+5.87%p ± 2.63%p** without SNDK (5/5 folds NN wins) / +6.34%p ± 2.57%p with SNDK (5/5 folds NN wins) |

The momentum comparison is the most informative single number. It tells you the NN's contribution *over and above* a trivial "past winners keep winning" heuristic, which is what any skeptical reviewer would try first. Three observations support its robustness:

- **Nearly identical across SNDK configurations**: +5.87 vs +6.34%p. The SNDK ticker isn't responsible for the NN's edge — it's a localized inflation that affects NN and momentum almost equally (NN +14.79 vs momentum +8.45 with SNDK; +7.13 vs +1.26 without).
- **Standard deviation ~2.6%p in both cases**: similar across folds, not driven by one or two outliers.
- **5/5 folds NN wins in both configurations**: not luck.

For comparison with v2.3.3 (default hyperparameters, N=5 ensemble, with SNDK Fold 1 inflation): NN edge over momentum was +6.8%p across 5/5 folds. The v2.3.7 measurement (Optuna-tuned, N=20 production scale) gives essentially the same edge — the per-fold Stage 2 numbers can be found in `results/momentum_baseline_v237.json`.

### Per-fold detail (without SNDK)

| Fold | NN top 5 | NN α | Momentum top 5 | Mom α | NN edge |
|------|----------|------|----------------|-------|---------|
| 1 | SMCI, SHOP, COHR, TRGP, ZS | +7.5%p | SHOP, ZS, GNRC, ZBRA, DECK | -1.7%p | **+9.2%p** |
| 2 | CVNA, PDD, DVN, FANG, DDOG | +6.3%p | CVNA, TPL, PDD, ALGN, AXON | +5.1%p | +1.2%p |
| 3 | MRNA, ARM, MSTR, TTD, PCG | +4.8%p | MRNA, TTD, MSTR, ARM, TEAM | -1.7%p | +6.5%p |
| 4 | TSLA, AMD, LITE, HOOD, FSLR | +7.8%p | TSLA, AMD, MELI, PODD, ASML | +1.0%p | +6.8%p |
| 5 | GEV, INSM, APA, XYZ, PLTR | +9.3%p | GEV, XYZ, CEG, NVDA, PYPL | +3.6%p | +5.7%p |

**Fold 3 is interesting**: 4/5 of NN's top 5 overlap with momentum's top 5 (MRNA, ARM, MSTR, TTD), and NN beats momentum by +6.5%p mainly through the 5th pick (PCG vs TEAM). When momentum and NN converge on the same picks, the NN's edge narrows to ranking precision on a single position. This explains why Fold 3 also has the lowest rank_corr (0.402) — fewer independent NN-only signals.

## Ablation study

To characterize what each feature group contributes, I ran the backtest with three configurations. Results for each are stored in `results/backtest_results_{config}.json`:

| Config | Features | Rank Corr | Alpha (5-fold avg) | Cross-sector Transfer |
|--------|----------|-----------|--------------------|-----------------------|
| Full (macro + sentiment) | 97 | +0.465 | +15.4%p | +0.028 |
| No-macro (tech + sentiment) | 54 | **+0.526** | +15.4%p | **+0.219** |
| Tech-only (technical only) | 54 | **+0.526** | +15.4%p | **+0.219** |

Two findings were counter to my initial expectations.

**Macro features hurt cross-sectional ticker ranking.** Removing the 43 macro features (FRED + Fama-French + cross-asset) *improved* rank correlation from +0.465 to +0.526 and cross-sector transfer from +0.028 to +0.219 — the latter roughly 8× higher. My interpretation: in a cross-sectional split, all tickers at a given snapshot share the same macro values, so macro features carry no inter-ticker signal — only noise that the ensemble partially overfits to. Note this is the opposite of what the Walk-Forward CV (time-axis) shows, where macro features reduce return error from 16.6%p to 11.9%p. The two CV schemes measure different things, and for portfolio *selection* (ticker ranking within a time period), the cross-sectional result is the one that matters.

**Sentiment features don't move backtest metrics but do change the selection.** No-macro and tech-only produce identical backtest numbers by construction — sentiment features are computed only for the current 84 stocks at Stage 2 and are absent from the 527-ticker training cache. Their effect shows up in the top-5 picks instead: tech-only selects `BSX, MDT, MSFT, CRM, SYK`, while adding sentiment swaps SYK out for ADSK (ADSK had a positive news composite of +0.149). One swap out of five is a real but modest effect, and it's not measurable through backtest alpha with the current design.

I haven't restructured the pipeline based on these findings. Macro features are still loaded by default because they're useful inside `blend_optimizer.py`'s regime gate for the Walk-Forward CV and weight-shrinkage logic. Isolating them there — rather than concatenating into the per-ticker feature vector — is in the Future work section.

The v2.3.7 retrain doesn't re-run ablation (focus shifted to hyperparameter optimization). The macro/sentiment toggles still work via `--no-macro`, `--tech-only` flags on `run.py`.

## Hyperparameter optimization (Stage 1)

The earlier README listed "hyperparameters set by trial-and-error" as a known limitation. Stage 1 of v2.3.7 addresses this with a 60-trial Optuna TPE search over 6 dimensions. Results are stored in `optuna_storage.db` (SQLite, resume-safe), with the full trial log in `optuna_stage1.log` and best-3 summary in `results/optuna_stage1_results.json`.

### Search space

| Dimension | Range / Set | Why |
|-----------|-------------|-----|
| `lr` | log-uniform [1e-4, 3e-3] | Adam's typical sweet spot, ~1.5 decades wide |
| `weight_decay` | log-uniform [1e-5, 1e-3] | Tiny to moderate; higher would over-regularize given the training set size |
| `huber_delta` | {0.1, 0.2, 0.3, 0.5, 1.0} | Discrete set covering aggressive (0.1) to MSE-like (1.0) |
| `architecture` | {small [32,16], medium [64,32,16], large [128,64,32]} | v1 default was small; large was rejected at 5,884 samples but may be competitive at 122K |
| `var_threshold` | log-uniform [1e-3, 1e-1] | Previously hardcoded at 0.01 |
| `corr_threshold` | log-uniform [1e-3, 1e-1] | Previously hardcoded at 0.05 |

Composite-score coefficients (`sentiment_weight`, `uncertainty_penalty`, `event_risk_penalty`) are intentionally not in the search space — they're reserved for a separate study because they govern Stage 2 selection rather than NN training.

### Stage 1 design choices

- **N=5 ensemble** during search (vs N=20 in production). 60 trials × 4 folds × 20 NN at production scale would take ~150 days, infeasible. At N=5 it's ~36–45 hours. Hyperparameter ranking is expected to be approximately invariant to ensemble size — Stage 2 below verifies this assumption.
- **4 folds excluded Fold 1** (Stage 1 used `[1,2,3,4]` = Fold 2-5). This was a fast solution to the SNDK post-IPO artifact distorting Fold 1 alpha by ~+35%p. Stage 2 below uses a more precise approach (exclude SNDK at the ticker level, all 5 folds). The `optuna_search.py` script has been updated to match Stage 2's design for any future re-runs, but the v2.3.7 study itself ran on the original 4-fold layout.
- **TRAINING_EPOCHS = 5000** (raised from 800). Epoch cap was determined via a 6000-epoch single-NN diagnostic (`val_trajectory_6000.json`): block-wise validation loss improvement crossed below 0.1% per 500 epochs at epoch 5000. Early stopping (patience=41) catches convergence well before this cap in practice — most NNs stop in 200-400 epochs at the Optuna-best hyperparameters.
- **Persistent SQLite storage**: trials are durable across interruptions. If the run dies, restarting picks up from the last completed trial.
- **120-min per-trial timeout** via `signal.alarm` to prevent pathological configs from stalling the study.

### Stage 1 result

The TPE sampler converged tightly. Top 3 trials all use:

- `medium [64,32,16]` architecture (large/small not in top 3)
- `huber_delta = 0.5` (aggressive 0.1 not preferred at this epoch budget)
- `lr` ~ 2.5e-4 to 5.9e-4 (tight cluster)
- Low `var_threshold` (~0.001-0.002, retains nearly all variance-pass features) + moderate `corr_threshold` (~0.078-0.084, filters by correlation)

| Rank | Trial | Mean rank_corr (N=5, 4-fold) | lr | wd | huber | arch | var_thr | corr_thr |
|------|-------|----------|---|---|---|---|---|---|
| 1 | #58 | **+0.5616** | 2.50e-4 | 1.64e-4 | 0.5 | medium | 0.00197 | 0.0838 |
| 2 | #47 | +0.5604 | 5.94e-4 | 7.22e-5 | 0.5 | medium | 0.00148 | 0.0781 |
| 3 | #52 | +0.5590 | 3.31e-4 | 9.22e-5 | 0.5 | medium | 0.00164 | 0.0783 |

The narrow spread (+0.5616 to +0.5590) and identical structural choices (medium / huber 0.5) suggest the search found a single shared optimum rather than fragile local peaks.

### Why Optuna found these settings

**Huber 0.5 over 0.1**: At delta=0.1, most samples fall in Huber's linear region where gradient magnitude is constant ±1. Loss landscape becomes piecewise-linear, slow to fine-tune. At delta=0.5, most samples sit in the quadratic region with gradients proportional to error — the network converges much faster. Empirically, Optuna-best NNs reach `val_loss ~0.017` in ~250 epochs, vs ~3000+ for the v2.3.6 default huber=0.3 with lr=5e-4.

**Medium over large**: With ~95K training samples and 30-35 selected features, medium has enough capacity (6,700 params) without overfitting to time-synchronous noise that large architecture (23,000 params) chases.

**Low var_threshold + moderate corr_threshold**: Inverts the v1/v2 default. Old behavior filtered by variance (kept variance > 0.01) then weakly by correlation (> 0.05). Optuna prefers retaining most variance-pass features (~all of 97) but filtering more aggressively by target correlation. Result: 30-35 high-signal features per fold rather than the default ~50 mixed.

## Stage 2 production retrain

After Stage 1 completed, the best hyperparameters were re-run at production scale: **N=20 ensemble** (vs Stage 1's N=5), **all 5 folds** (vs Stage 1's 4), and **SNDK excluded at the ticker level** rather than dropping Fold 1 entirely. This serves two purposes:

1. Verify that Stage 1's hyperparameter ranking holds at production ensemble size (the N=5 → N=20 invariance assumption).
2. Generate full per-ticker prediction matrices, per-snapshot rankings, and proper SPY benchmark data for analysis (Tasks A/B/C).

Code: `stage2_retrain.py`, runtime ~1.6-1.9 hours. Outputs in `results/stage2/top1_trial58/`.

### Stage 2 settings

- **Hyperparameters**: Optuna Trial #58 best (see Stage 1 table above)
- **N_ENSEMBLE = 20** (production scale; matches v2.3.4 baseline for direct comparison)
- **5 folds [0,1,2,3,4]** (all folds, stratified by GICS sector)
- **SNDK excluded** (17 samples, 0.014% of training data) — see Sensitivity analysis below
- **TRAINING_EPOCHS = 5000** (effectively governed by patience=41 early stopping)
- **Real-time matplotlib plot** (4×5 grid, one subplot per NN ensemble member) for live monitoring during ~22 min/fold runs

### Stage 2 result (without SNDK, primary)

| Metric | Stage 2 (N=20, 5-fold) | Optuna Stage 1 (N=5, 4-fold) | v2.3.4 baseline (N=20, 5-fold, with SNDK) |
|--------|------------------------|------------------------------|-----------------------------------------|
| Mean rank_corr | **0.518 ± 0.079** | 0.5616 | 0.5311 |
| Mean alpha (universe) | **+7.1%p ± 1.5%p** | — | +15.4%p (Fold 1 SNDK inflated) |
| Mean alpha (vs SPY) | **+8.1%p ± 1.7%p** | — | — |
| Mean alpha (vs momentum) | **+5.87%p ± 2.63%p** (5/5 folds) | — | — |
| Selection Sharpe | 2.71 | — | 2.11 |
| Top-5 Hit Rate | 5/5 folds positive | — | 5/5 folds positive |

The Stage 2 rank_corr (0.518) is below Stage 1's reported 0.5616. This was expected and explained: Stage 1 measured rank_corr only on Fold 2-5 (Fold 1 excluded due to SNDK). Stage 2 measures all 5 folds with SNDK excluded at ticker level — Fold 1 alone has rank_corr 0.483 (the ticker-level fix doesn't restore the inflated Fold 1 alpha that Stage 1 sidestepped). Fold 2-5 alone in Stage 2 averages rank_corr ~0.527, much closer to Stage 1's measurement. The N=5 → N=20 invariance assumption holds.

### Sensitivity analysis: SNDK exclusion

SNDK (SanDisk, 2024-10 WDC spinoff) appeared as an outlier in v2.3.3's Fold 1 analysis: 17 snapshots, mean realized 3-month return +194.7%, max +472.2%. The cache reports 526 tickers with at least 17 snapshots; SNDK is the shortest-history ticker in the set.

**Decision**: exclude SNDK at the ticker level under a uniform criterion: `n_snapshots < 30 AND |return mean − universe mean| > 5σ`. Among 526 tickers in cache, this filter matches **exactly one ticker (SNDK)**.

**Verification**:

| Ticker | n | mean return | Distance from universe mean (+0.046, std 0.174) | Excluded? |
|--------|---|-------------|---------------------------------------------------|-----------|
| SNDK | 17 | +1.947 | **+26.5σ** | ✅ excluded |
| GEV | 40 | +0.309 | +1.5σ | ❌ retained |
| SOLV | 40 | +0.037 | +0.0σ | ❌ retained |
| (universe) | — | +0.046 | — | — |

Two other short-history spinoffs (GEV from GE in 2024-04, SOLV from 3M in 2024-04) have similarly small `n` but their returns sit within the universe distribution. They are retained. Only SNDK, with a 26.5σ deviation from the universe mean over its observed window, qualifies as a structurally non-comparable outlier.

To make this exclusion auditable, Stage 2 was run twice — identical hyperparameters and seeds — once without SNDK (primary) and once with SNDK included (sensitivity, results in `results/stage2_with_sndk/top1_trial58/`).

#### Aggregate comparison

| Configuration | rank_corr | α vs universe | α vs SPY | α vs momentum |
|---|---|---|---|---|
| **Without SNDK** (primary) | 0.518 ± 0.079 | +7.1%p ± 1.5%p | +8.1%p ± 1.7%p | +5.87%p ± 2.63%p (5/5 folds) |
| With SNDK (sensitivity) | 0.521 ± 0.078 | +14.8%p ± 14.0%p | +16.0%p ± 16.6%p | +6.34%p ± 2.57%p (5/5 folds) |

The SNDK effect is localized:

#### Per-fold α vs SPY

| Fold | Without SNDK | With SNDK | Δ | Comment |
|------|--------------|-----------|---|---------|
| 1 | +7.8%p | **+44.9%p** | **+37.1%p** | SNDK is top-1 NN pick (mean +194.7% return) |
| 2 | +7.8%p | +8.6%p | +0.8%p | within noise |
| 3 | +5.5%p | +7.7%p | +2.2%p | within noise |
| 4 | +8.6%p | +7.7%p | -0.9%p | within noise |
| 5 | +10.6%p | +11.3%p | +0.7%p | within noise |

Fold 1 with SNDK has SNDK as its top-1 pick — SNDK's mean realized return contributes ~+37%p to Fold 1's +44.9%p alpha. Other folds (no SNDK in test set) show <±2.2%p difference, confirming SNDK is a localized outlier effect, not a global model behavior.

The momentum-edge metric (last column of the aggregate table) is what the Sensitivity is really telling us: **the NN's edge over momentum is +5.87%p without SNDK vs +6.34%p with — nearly identical**. SNDK doesn't drive the model's selection skill; it only inflates the absolute alpha number that both NN and momentum benefit from. The proper momentum baseline picks SNDK as its top-1 in the with-SNDK config (alpha +34.6%p vs NN's +42.7%p), confirming that SNDK is a momentum-detectable anomaly, not a uniquely model-found one.

We report the **without-SNDK** numbers as primary headlines because:

1. The 26.5σ deviation makes SNDK non-comparable to other tickers under standard backtest evaluation.
2. The with-SNDK aggregate has 10× wider standard deviation (1.7 vs 16.6%p for SPY-alpha) — this is a single-fold distortion masquerading as performance.
3. The without-SNDK numbers are what production-realistic deployment would expect: SNDK is a single observation, not a strategy.

The with-SNDK numbers match v2.3.4's headline alpha (+15.4%p) within rounding, confirming the SNDK effect is reproducible and isolable.

### Tasks A/B/C — full ranking artifacts

Stage 2 produces three analysis artifacts beyond aggregate metrics:

- **Task A: Full per-ticker ranking per fold** (`fold_{1-5}/full_ranking.csv`). All ~100-112 test tickers with predicted return, prediction std (ensemble uncertainty), actual return, predicted rank, actual rank, rank error, and snapshot count. Useful for long-short simulation, decile analysis, or ranking-quality diagnostics.
- **Task B: SPY benchmark across all folds** (`spy_benchmark.csv`). Stage 2 computes SPY's mean 3-month forward return averaged over each fold's date range (~495 weekly anchor dates). Earlier versions only had SPY in 2 of 5 folds (the ones where SPY happened to be a test ticker). The standalone `fix_spy_benchmark.py` script regenerates this if needed.
- **Task C: Per-snapshot rankings** (`fold_{1-5}/per_snapshot_ranking.csv`). Tickers grouped by month, ranked within each bucket. Enables "if I'd run this monthly with a 3-month hold, what would the trajectory look like?" simulation.

These three were initially proposed as analysis asks from a quant friend during v2.3.6, then deferred to Stage 2 to avoid disturbing the running Optuna study. They are now generated automatically by `stage2_retrain.py`.

## Stage 1 — Transaction cost analysis (v2.3.11)

Backtest alpha is paper-alpha. To gate deployment, v2.3.11 adds a transaction cost model calibrated to the Korean retail broker route (Korea Investment & Securities, KIS) for US equities.

### Model

`transaction_cost_analysis.py` decomposes round-trip TC for each ticker as:

- Commission: 0.04% per side (KIS US route)
- FX spread: 0.10% per side (KRW ↔ USD)
- Bid-ask spread: 0.05% per side for large-cap (mcap ≥ $10B), 0.20% for small-cap
- Market impact: cube-root model, `0.10 × (size / ADV)^(1/3)`

ADV (average daily volume) and market cap are fetched via yfinance and cached to `tc_adv_cache.json`. USD/KRW rate is fetched live and cached for the run.

### Sensitivity grid

Three position sizes × three turnover scenarios against the Fold 2-5 mean paper alpha (+8.07%p ± 1.83%p / quarter):

| Position (₩) | Turnover/y | Quarterly TC | Net alpha |
|---:|---:|---:|---:|
| 1,000,000 | 2 | 0.28% | **+7.79%p** |
| 1,000,000 | 4 | 0.52% | **+7.55%p** |
| 1,000,000 | 12 | 1.46% | **+6.62%p** |
| 5,000,000 | 4 | 0.64% | **+7.43%p** |
| 5,000,000 | 12 | 1.83% | **+6.24%p** |

All 9 grid cells (full table in `results/tc_analysis_summary.md`) clear the +3%p deploy gate. The stress case (₩5M position, monthly rebalance) still nets +6.24%p — TC erosion is dominated by FX spread and commission, not market impact (impact only matters for small-cap tickers, which the model rarely picks).

The main scenario (₩1M × turn=4, i.e. quarterly rebalance) carries +7.55%p net alpha. For a ₩20M portfolio that's ~₩1.5M / quarter — material at this position size.

## v2.3.12 — Heteroscedastic NN with log-volatility target

v2.3.7's production NN systematically over-predicted volatility for defensive tickers (RTX, JNJ, MDT predicted 95-200%, realized 16-22%). Two root causes:

1. **Softplus on point-estimate volatility.** No uncertainty quantification — the NN was forced to output a single number for risk, and large ensemble disagreement was hidden inside the mean.
2. **Linear-scale Y_risk target.** Realized 3-month volatility is right-skewed (approximately lognormal); fitting it on a linear scale puts excess weight on the high-vol tail and pulls predictions upward.

v2.3.12 addresses both.

### Architecture: dual-head heteroscedastic NN

`models.py::HeteroscedasticDualHeadNN` outputs `(mu, logvar)` for both return and risk. `logvar` is clamped to `[-10, 5]` for numerical stability; the risk head's logvar bias is initialized to -2.0 (initial sigma ≈ 0.37, in the right order of magnitude for log-vol residuals).

The shared trunk uses BatchNorm + Linear + ReLU + Dropout blocks; two independent heads then project to `(mu, logvar)` for return and risk respectively. All ensemble members in `stage2_retrain.py` and all NN call sites in `run.py`, `backtest.py`, and `historical.py` use this class. The legacy `nn.Sequential` + softplus path is removed.

### Loss: Gaussian negative log-likelihood

`models.py::heteroscedastic_loss` implements the standard Gaussian NLL of Kendall & Gal (2017):

```
L = 0.5 × ( exp(−logvar) · (y − μ)² + logvar )
```

The `exp(−logvar)` term lets the model down-weight high-uncertainty samples, and the `+logvar` term penalizes the model for predicting high uncertainty everywhere. A backup variant `heteroscedastic_loss_beta` (Seitzer et al., 2022) is included but disabled by default — activated only if the calibration plot from the first production retrain shows small-sigma compression pathology.

### Target: log-transformed volatility

`Y_risk → np.log(np.maximum(Y_risk, LOG_EPSILON))` with `LOG_EPSILON = 1e-4`. This handles 22 zero-volatility samples by mapping them to log(1e-4) = -9.2 (treated as outliers by the ensemble + early stopping). Inference back-transforms via `risk_mean = np.exp(risk_log_mu)` — the lognormal median (Andersen, Bollerslev, Diebold & Labys, 2003, *Econometrica*), which is the appropriate point estimate for a right-skewed distribution.

### Pipeline propagation

The log-transform is applied consistently in `stage2_retrain.py`, `historical.py`, `run.py`, and `backtest.py` before tensor conversion. Inference back-transforms via `exp()` for the linear-scale prediction; the log-space mean and sigma are also propagated for downstream uncertainty quantification:

- `pred_risk` — linear-scale (back-transformed) volatility prediction
- `pred_risk_log_mean` / `pred_risk_log_sigma` — log-space mean and per-sample aleatoric uncertainty
- `pred_risk_log_aleatoric` — aleatoric component (sigma from the NLL head)
- Epistemic uncertainty derived from ensemble variance in log-space

These four columns are added to `full_ranking.csv` per fold.

### Validation

Two smoke tests passed 6/6 health checks each:

| Smoke | Architecture | Rank corr | Universe NN risk mean | Defensive ticker recovery |
|---|---|---|---|---|
| Large `[128,64,32]` | sanity check | +0.4009 (p=2.3e-5) | 26.2% | MDT 20.8% vs realized 20.3% |
| Medium `[64,32,16]` | Optuna best (Trial #58) | +0.3906 | 26.3% | LLY 24.7% vs 26.4%; BA 31.5% vs 33.8% |

The broken intermediate v2.3.8 production had universe NN risk mean of 86.2% with defensive tickers in the 95-200% range. v2.3.12 brings both the universe mean and per-ticker calibration into the realistic 20-30% band. The fix is robust across architectures.

### Production retrain protocol

The Optuna best config (Trial #58: medium arch, lr=2.5e-4, wd=1.64e-4) is retrained at `N_ENSEMBLE=20` with the v2.3.12 architecture. Decision flow:

- **Standard NLL (default)** is used for the first retrain.
- A calibration plot (predicted log-sigma vs. realized log-residual) is checked before accepting the result.
- If the plot shows small-sigma compression and large-sigma underlearning (a known pathology of standard NLL — Seitzer et al., 2022), the beta-NLL backup loss is activated and the model is retrained once.
- Otherwise the standard NLL result is final.

## Known limitations

- **Fold 1 outlier inflates the headline alpha when SNDK is included.** SNDK's post-IPO run drives Fold 1 alpha to +44.9%p (vs +7.8%p in the without-SNDK config); the robust estimate across Folds 2–5 is closer to +8.5%p in either configuration. Headlines should be read with the SNDK exclusion noted. See [Sensitivity analysis](#sensitivity-analysis-sndk-exclusion).
- **Hyperparameters were optimized within a 6-dimension search space** (lr, weight_decay, huber_delta, architecture, var_threshold, corr_threshold) using a 60-trial Optuna TPE sampler. Performance is conditional on this search space — broader searches over composite-score coefficients, train/val split ratios, dropout rates, optimizer choice, or LR schedule may yield further improvements. The search also held N_ENSEMBLE=5 fixed for tractability; Stage 2 verifies the N=5 → N=20 invariance assumption but doesn't search ensemble size as a hyperparameter.
- **Methodological asymmetry between Stage 1 and Stage 2.** Stage 1 (Optuna search) ran on 4 folds (`[1,2,3,4]` = Fold 2-5) to avoid SNDK's Fold 1 distortion via fold-level exclusion. Stage 2 (production retrain) ran on all 5 folds with SNDK excluded at the ticker level — a more precise approach. The `optuna_search.py` script has been updated to match Stage 2's design (5 folds + SNDK ticker exclusion) for any future re-runs, but the v2.3.7 hyperparameter search itself ran on the original 4-fold layout. The Stage 2 result demonstrates the asymmetry doesn't materially affect ranking quality (rank_corr 0.518 with vs 0.521 without SNDK at production scale).
- **No transaction cost, slippage, or tax modeling.** All backtest numbers are paper-alpha and will be lower after real-world frictions (typically several %p/year for monthly rebalancing strategies).
- **Survivorship bias in the training universe.** Tickers are sampled from the *current* S&P 500 + NASDAQ-100 composition, so stocks that were delisted or removed from the index during the 10-year window are underrepresented. This biases the training distribution toward survivors.
- **Composite score coefficients are hand-picked.** `sentiment_weight=0.10`, `uncertainty_penalty=3.0`, `event_risk_penalty=2.0` were not tuned via grid search and were intentionally not in the Optuna search space (they govern Stage 2 selection rather than NN training and merit a separate study). The v2.3.3 ablation measured sentiment as a feature group (one swap in top-5); individual coefficient sensitivity is still uncharacterized.
- **No historical fundamentals.** yfinance doesn't expose past PE/ROE/analyst targets, so Stage 1 features are technical + macro only. Fundamentals enter only at Stage 2 via current analyst consensus.
- **Sector concentration in selection step.** No diversification constraint; top 5 often cluster in 2 sectors (typically AI Compute + Neuromodulation). The v2.3.3 ablation showed cross-sector rank-correlation is much lower than within-sector, so forced diversification would pick lower-scoring stocks. Current default accepts the concentration.
- **MC Dropout passes per ensemble member shrunk from 6 to 1 at N=20.** `MC_FORWARD_PASSES = 30` is divided across the ensemble (`30 // N_ENSEMBLE`), so at N=5 each model did 6 passes but at N=20 each model does 1. Uncertainty estimates rely more on ensemble variance than dropout sampling. This is a design choice (ensemble averaging is itself approximate Bayesian, Gal & Ghahramani 2016) but the trade-off isn't characterized empirically.
- **yfinance rate limits.** Heavy S&P 500 batch downloads occasionally cause cross-asset fetches to return truncated history. The tz-safety fix in `training_universe.py` means this degrades gracefully, but ideally the macro loads should happen before the big batch. Separately, `fredapi` lacks an internal socket timeout (`socket.setdefaulttimeout(120)` is set in `fetch_fred_data` to mitigate; addresses a real 30-min stall observed during Stage 1 prep).
- **Risk metric is reported as the lognormal median.** v2.3.12 log-transforms the risk target during training; the inference back-transform `risk_mean = exp(risk_log_mu)` yields the median of the implied lognormal distribution, not the mean. For symmetric ±1.96σ intervals, work in log-space: `lower = exp(μ − 1.96σ)`, `upper = exp(μ + 1.96σ)`. The linear-scale `pred_risk` column in `full_ranking.csv` follows the median convention; the log-space columns (`pred_risk_log_mean`, `pred_risk_log_sigma`) are also exported for downstream analysis.
- **This is not investment advice.** Predictions carry ±12%p MAE on return and ±9%p on risk. Realized Sharpe will likely be 30–35% lower than predicted Sharpe. Monthly tracking against actual realized returns is the only honest validation; backtests don't prove future performance.

## Output

After a full run, `results/` contains:

- `output.json` — selected tickers, 3x3 matrix, per-stock predictions, rationale, blend weights
- `universe.json` — screener output (84 tickers with sector map)
- `fig1_scatter.png` through `fig8_dashboard.png` — auto-generated figures
- `backtest_results.json` — if `--backtest` was run
- `backtest_cache.npz` — cached training matrix for faster re-runs
- `results/tc_analysis_summary.{md,json}` — v2.3.11 transaction cost sensitivity grid
- `tc_adv_cache.json` — cached ADV / market-cap data for the TC model

After Stage 1 + Stage 2 runs:

- `optuna_storage.db` — Optuna trial database (SQLite, resume-safe)
- `optuna_stage1_results.json` — Stage 1 top-3 hyperparameters summary
- `optuna_stage1.log` — full trial log
- `stage2/top1_trial58/` — Stage 2 retrain artifacts (without SNDK):
  - `summary.json` — aggregate metrics
  - `fold_{1-5}/full_ranking.csv` — Task A
  - `fold_{1-5}/per_snapshot_ranking.csv` — Task C
  - `spy_benchmark.csv` — Task B
  - `loss_curves_fold{1-5}.png` — real-time plot snapshots
- `stage2_with_sndk/top1_trial58/` — sensitivity analysis (with SNDK)
- `momentum_baseline_v237.json` — apples-to-apples momentum baseline

## Sector overview

| Sector | Focus | Seed method |
|--------|-------|-------------|
| A: AI Compute | GPU, cloud, AI platforms | auto (GICS industry) |
| B: Neuromodulation | DBS, TMS, BCI, medical devices | 2 anchors + auto |
| C: CNS Pharma | Neurotransmitter-based therapeutics | 1 anchor + auto |
| D: Digital Health | Telemedicine, digital therapeutics | 2 anchors + auto |
| F: Space & Aerospace | Launch, satellites, defense | 6 anchors + auto |
| G: Solar & Clean Energy | Solar, hydrogen, batteries | 10 anchors + auto |
| E: ETF Benchmark | Training benchmarks only (excluded from selection) | fixed 4 |

Anchor tickers are small/niche names that aren't in S&P 500 and therefore can't be auto-discovered. For any large-cap candidate, auto-discovery should find it — if it doesn't, that's usually a GICS classification issue worth investigating rather than patching with an anchor.

## Future work

The current pipeline uses a deep ensemble with MC Dropout, which is an approximate Bayesian method (Gal & Ghahramani, 2016) — the ensemble approximates posterior averaging and MC Dropout approximates variational inference. However, the shrinkage layer in `blend_optimizer.py` and the composite score use hand-picked coefficients rather than proper posterior inference.

Planned upgrades:

- **Composite-score coefficient optimization**: grid search over `sentiment_weight`, `uncertainty_penalty`, `event_risk_penalty`, `EVENT_RISK_PENALTY`. The Optuna Stage 1 search intentionally excluded these to keep the NN training search space focused. Stage 1's discovery that the network converges in 200-400 epochs at the optimal hyperparameters means a follow-up Stage 3 study is feasible at production ensemble size (N=20) within ~24h.
- **Survivorship bias correction**: use point-in-time S&P 500 + NASDAQ-100 composition data (Compustat / CRSP / Bloomberg) instead of current composition. Currently the training universe overrepresents survivors, which biases the model's expectations.
- ~~**Transaction cost + slippage modeling**~~ — implemented in v2.3.11 (see [Stage 1 — Transaction cost analysis](#stage-1--transaction-cost-analysis-v2311)). Korean broker (KIS) calibration with cube-root market impact; main scenario (₩1M × quarterly turnover) yields +7.55%p net alpha. Still future work: live tracking of realized slippage against the model's per-trade prediction to refine the impact constant.
- **Uncertainty calibration**: verify that predicted standard deviations actually match realized errors (calibration plots, temperature scaling if needed, following Guo et al., 2017). At N=20 with 1 MC pass per model, the dropout-based aleatoric component is weak; characterizing whether ensemble variance alone is well-calibrated would inform whether to revisit MC dropout count.
- ~~**Heteroscedastic output + aleatoric / epistemic decomposition**~~ — implemented in v2.3.12 (see [v2.3.12 — Heteroscedastic NN with log-volatility target](#v2312--heteroscedastic-nn-with-log-volatility-target)). Standard Gaussian NLL with log-transformed volatility target; aleatoric (per-sample log-sigma) and epistemic (ensemble variance) propagated separately. Production retrain in progress at N=20 with calibration-plot review before acceptance.
- **Hierarchical Bayesian structure over sectors**: sector-level priors with ticker-level posteriors (analogous to multi-level GLM with ROI-level random effects in fMRI analysis). Expected to improve cross-sector transfer, which is currently weak (+0.027 rank corr in the v2.3.3 ablation; not re-measured at Stage 2).
- **Disentangle macro from the per-ticker feature matrix.** The v2.3.3 ablation showed that macro features hurt cross-sectional rank correlation (+0.465 → +0.526 when removed), because all tickers at a given snapshot share identical macro values — the ensemble partially overfits to time-synchronous signals that carry no inter-ticker information. A cleaner design would route macro features through the blend-optimizer's regime gate only, rather than concatenating them into each ticker's feature vector.
