v2.3.9 Task #9 proxy grid (Apr 30, 2026)

Composite score coefficient grid search using Stage 2 cached predictions
joined with runtime-computed sentiment. Found uncertainty/event_risk
over-penalization (calibration value), sentiment weight inconclusive
due to temporal mismatch between historical predictions and current
sentiment.

Calibration result was committed (e710c0e) for the sentiment-independent
coefficients (UNCERTAINTY_PENALTY 3.0 -> 1.0, EVENT_RISK_PENALTY 2.0 -> 0.0).
SENTIMENT_WEIGHT_IN_SCORE held at 0.10 pending historical sentiment cache.

Files:
  composite_grid.py                   grid search script (567 combos)
  prep_stage2_sentiment.py            525-ticker sentiment fetcher
  stage2_sentiment.csv                525 tickers x 23 features (Apr 30)
  task9_composite_grid_results.json   grid output

Will be redone with historical sentiment cache built via SEC EDGAR
+ earnings_history (planned v2.3.10), and later GDELT news (v2.3.11+).
