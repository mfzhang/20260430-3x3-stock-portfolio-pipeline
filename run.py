#!/usr/bin/env python3
"""
run.py — single entry point

Usage:
  python run.py --torch                       # default universe
  python run.py --torch --screen              # with automated universe screening
  python run.py --torch --screen --sent       # + sentiment intelligence
  python run.py --backtest                    # portfolio backtest only
  python run.py --torch --screen --backtest   # full pipeline + backtest
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def main():
    args = sys.argv[1:]
    use_screen = '--screen' in args or '--auto-screen' in args
    use_sent = '--sent' in args or '--sentiment' in args
    no_sent = '--no-sent' in args
    no_macro = '--no-macro' in args       # [Ablation] disable FRED + FF + cross-asset
    tech_only = '--tech-only' in args     # [Ablation] disable both macro and sentiment
    run_backtest = '--backtest' in args
    custom_tickers = None
    if '--tickers' in args:
        idx = args.index('--tickers')
        custom_tickers = [t.upper() for t in args[idx+1:] if not t.startswith('--')]

    # [Ablation] Apply feature-group disable flags to config
    if tech_only:
        no_macro = True
        no_sent = True
        use_sent = False
    if no_macro:
        config.USE_MACRO_FEATURES = False
    config.USE_SENTIMENT_FEATURES = bool(use_sent and not no_sent)

    # Standalone backtest mode
    if run_backtest and not use_screen:
        from backtest import run_backtest as _bt
        _bt(n_folds=5, n_select=5, verbose=True)
        return

    np.random.seed(config.RANDOM_SEED)
    print("="*70)
    print("3x3 PORTFOLIO PIPELINE")
    print(f"KRW {config.TOTAL_CAPITAL_KRW:,} | {config.N_SELECT} stocks")
    if use_screen: print("  + Automated Universe Screening")
    if use_sent and not no_sent: print("  + Multi-Layer Sentiment Intelligence")
    print("  + Historical Training (S&P 500 expanded)")
    print("="*70)

    # Step 0: Universe Selection
    sector_map = {}
    if use_screen:
        try:
            from screener import build_universe
            print("\n[0] Automated Universe Screening")
            auto_tickers, sector_map, _ = build_universe(verbose=True)
            from data_auto import set_universe
            set_universe(auto_tickers, sector_map)
            custom_tickers = None
        except Exception as e:
            print(f"  Screener failed: {e}")
            print(f"  Falling back to default tickers")

    # Step 0.5: Current Data
    from data_auto import fetch_all_stocks
    data = fetch_all_stocks(custom_tickers)
    tickers, features = data['tickers'], data['features']
    target_ret = np.array([data['targets'][tk]['return'] for tk in tickers])
    target_risk = np.array([data['targets'][tk]['risk'] for tk in tickers])

    N, D = features.shape
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
    f_norm = np.clip((features - features.mean(0)) / (features.std(0) + 1e-8), -5, 5)
    f_norm = np.nan_to_num(f_norm, nan=0, posinf=0, neginf=0)
    print(f"\n  {N} stocks x {D} features")

    # Step 0.7: Sentiment Collection
    sent_features = {}
    if use_sent and not no_sent:
        try:
            from sentiment import collect_all_intelligence, features_to_array
            print("\n[0.5] Multi-Layer Intelligence Analysis")
            sent_features, sent_names = collect_all_intelligence(
                tickers,
                sector_map=sector_map or {tk: 'X' for tk in tickers},
                finnhub_key=getattr(config, 'FINNHUB_API_KEY', ''),
                model_type=getattr(config, 'SENTIMENT_MODEL', 'auto'),
                verbose=True,
            )
            sent_matrix, sent_names = features_to_array(sent_features, tickers)
            features = np.concatenate([features, sent_matrix], axis=1)
            f_norm_sent = np.clip((sent_matrix - sent_matrix.mean(0)) / (sent_matrix.std(0) + 1e-8), -5, 5)
            f_norm_sent = np.nan_to_num(f_norm_sent, nan=0, posinf=0, neginf=0)
            f_norm = np.concatenate([f_norm, f_norm_sent], axis=1)
            N, D = f_norm.shape
            print(f"  Features expanded: {D} (+ {len(sent_names)} sentiment)")
        except Exception as e:
            print(f"  Sentiment collection failed: {e}")
            print(f"  Continuing without sentiment features...")

    # Check PyTorch
    import torch

    # Step 1: Historical Training + Current Prediction
    print(f"\n[1] Historical Training (S&P 500 expanded -> current prediction)")
    mc, train_artifacts = _historical_train_and_predict(tickers, D)

    # Stage 2: Data-driven weight optimization + Fundamental adjustment
    print(f"\n  [Stage 2] Data-Driven Blend Optimization")

    analyst_targets = {tk: data['targets'][tk].get('return', 0) for tk in tickers}
    realized_risks = {tk: data['targets'][tk].get('risk', 0.3) for tk in tickers}

    try:
        from blend_optimizer import optimize_stage2_weights
        blend_result = optimize_stage2_weights(
            tickers=tickers,
            trained_models=train_artifacts['models'],
            keep_mask=train_artifacts['keep'],
            feat_mu=train_artifacts['mu'],
            feat_sigma=train_artifacts['sigma'],
            analyst_targets=analyst_targets,
            realized_risks=realized_risks,
            hist_X_shape_1=train_artifacts['hist_X_shape_1'],
            backtest_days=63,
            verbose=True,
        )
        w_ret = blend_result['w_ret']
        w_risk = blend_result['w_risk']
        print(f"\n    Using optimized weights: tech {w_ret:.0%} / fund {1-w_ret:.0%} (return)")
        print(f"                             NN {w_risk:.0%} / realized {1-w_risk:.0%} (risk)")
    except Exception as e:
        print(f"    Blend optimization failed: {e}")
        print(f"    Using default weights: tech 20% / fund 80%")
        w_ret, w_risk = 0.2, 0.4
        blend_result = None

    print(f"\n  [Stage 2] Fundamental Adjustment (w_ret={w_ret:.2f}, w_risk={w_risk:.2f})")
    for tk in tickers:
        t = data['targets'][tk]
        analyst_ret = t.get('return', 0)
        base_ret = mc[tk]['ret_mean']
        mc[tk]['ret_mean'] = w_ret * base_ret + (1 - w_ret) * analyst_ret
        realized_risk = t.get('risk', mc[tk]['risk_mean'])
        mc[tk]['risk_mean'] = w_risk * mc[tk]['risk_mean'] + (1 - w_risk) * realized_risk
        print(f"    {tk:>6}: tech={base_ret*100:+.1f}% + fund={analyst_ret*100:+.1f}% -> blended={mc[tk]['ret_mean']*100:+.1f}%")

    # Step 2: Selection (Composite Score with sentiment)
    print(f"\n[2] Top {config.N_SELECT} selection")
    etf_tickers = {'VOO', 'QQQ', 'SOXX', 'XBI', 'SPY'}

    if sent_features:
        from sentiment import composite_score_v2
        scores = composite_score_v2(
            mc, sent_features,
            uncertainty_penalty=config.UNCERTAINTY_PENALTY,
            sentiment_weight=getattr(config, 'SENTIMENT_WEIGHT_IN_SCORE', 0.10),
            event_risk_penalty=getattr(config, 'EVENT_RISK_PENALTY', 2.0),
        )
        print("  (Composite Score: + sentiment + event risk)")
    else:
        scores = {}
        for tk, r in mc.items():
            if tk in etf_tickers:
                scores[tk] = -999
                continue
            s = r['ret_mean'] / max(r['risk_mean'], 0.01)
            scores[tk] = (r['conf_mean'] * s) / (1 + r['uncertainty'] * config.UNCERTAINTY_PENALTY)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [tk for tk, _ in ranked[:config.N_SELECT]]

    for i, (tk, sc) in enumerate(ranked):
        r = mc[tk]; sel = " <-" if i < config.N_SELECT else ""
        print(f"  {i+1:>2}. {tk:>6} score={sc:.3f} ret={r['ret_mean']*100:+.0f}% risk={r['risk_mean']*100:.0f}%{sel}")

    # Step 3: Matrix
    print(f"\n[3] End-to-End Matrix Training")
    sel_idx = [tickers.index(tk) for tk in selected]
    sel_feat = f_norm[sel_idx]
    sel_ret = np.array([mc[tk]['ret_mean'] for tk in selected])
    sel_risk = np.array([mc[tk]['risk_mean'] for tk in selected])

    cell_ret, cell_risk = np.zeros((3,3)), np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            w = np.array([np.exp(-((sel_risk[s]-config.RISK_TIER_MIDPOINTS[j])**2)/(2*0.12**2)) for s in range(len(selected))])
            w /= w.sum()+1e-8
            cell_ret[i,j] = sum(w[s]*sel_ret[s]*config.TIME_MULTIPLIERS[i] for s in range(len(selected)))
            cell_risk[i,j] = sum(w[s]*sel_risk[s] for s in range(len(selected)))

    Xm = sel_feat.flatten(); Xm = Xm / (np.abs(Xm) + 1.0)
    from models import MatrixNetwork
    net = MatrixNetwork(len(Xm))
    rm, cm = np.array(config.TIME_MARGINALS), np.array(config.RISK_MARGINALS)

    best_loss, best_W = float('inf'), None
    for ep in range(config.E2E_EPOCHS):
        lr = config.E2E_LR_MIN + 0.5*(config.E2E_LR_MAX-config.E2E_LR_MIN)*(1+np.cos(np.pi*ep/config.E2E_EPOCHS))
        Xi = Xm + np.random.normal(0, config.E2E_NOISE_STD, len(Xm))
        loss, sharpe, pr, pk, W = net.train_step(Xi, cell_ret, cell_risk, rm, cm, lr)
        if loss < best_loss: best_loss, best_W = loss, W.copy()
        if ep % 100 == 0 or ep == config.E2E_EPOCHS-1:
            print(f"    ep={ep:>4} sharpe={sharpe:.3f} ret={pr*100:+.1f}% risk={pk*100:.1f}%")

    # Results
    pr = np.sum(best_W * cell_ret); pk = np.sum(best_W * cell_risk)
    print(f"\n{'='*70}")
    print(f"RESULT: {', '.join(selected)}")
    print(f"{'='*70}")
    for i in range(3):
        vals = "  ".join(f"{best_W[i,j]*100:>6.1f}%" for j in range(3))
        krw = "  ".join(f"W{best_W[i,j]*config.TOTAL_CAPITAL_KRW/1e6:.1f}M" for j in range(3))
        print(f"  {config.TIME_LABELS[i]:>14}: {vals}   ({krw})")
    print(f"\n  Return: {pr*100:+.2f}%  Risk: {pk*100:.2f}%  Sharpe: {pr/max(pk,.001):.2f}")

    # Build detailed rationale for each stock
    rationale = {}
    for tk in tickers:
        r = mc[tk]
        sc = scores.get(tk, -999)
        sf = sent_features.get(tk, {}) if sent_features else {}
        sec = sector_map.get(tk, 'X') if sector_map else 'X'
        is_selected = tk in selected
        rank = next((i+1 for i, (t, _) in enumerate(ranked) if t == tk), 999)

        sharpe_raw = r['ret_mean'] / max(r['risk_mean'], 0.01)
        sent_boost = 1 + sf.get('composite_sentiment', 0) * getattr(config, 'SENTIMENT_WEIGHT_IN_SCORE', 0.10) if sf else 1
        event_risk = sf.get('event_risk_score', 0) if sf else 0

        drivers = []
        if r['ret_mean'] > 0.25: drivers.append(('high_return', f"+{r['ret_mean']*100:.0f}% predicted return"))
        elif r['ret_mean'] > 0.15: drivers.append(('moderate_return', f"+{r['ret_mean']*100:.0f}% predicted return"))
        elif r['ret_mean'] < 0: drivers.append(('negative_return', f"{r['ret_mean']*100:.0f}% negative outlook"))

        if r['risk_mean'] < 0.25: drivers.append(('low_risk', f"{r['risk_mean']*100:.0f}% risk - defensive"))
        elif r['risk_mean'] > 0.5: drivers.append(('high_risk', f"{r['risk_mean']*100:.0f}% risk - volatile"))

        if r.get('conf_mean', 0) > 0.97: drivers.append(('high_confidence', "NN confidence >97%"))
        if r.get('uncertainty', 0) < 0.05: drivers.append(('low_uncertainty', "Low MC Dropout variance"))
        if r.get('uncertainty', 0) > 0.15: drivers.append(('high_uncertainty', "High prediction uncertainty"))

        if sf:
            if sf.get('news_sentiment_7d', 0) > 0.3: drivers.append(('positive_news', f"Strong positive news sentiment"))
            if sf.get('news_sentiment_7d', 0) < -0.3: drivers.append(('negative_news', f"Negative news sentiment"))
            if sf.get('filing_count_30d', 0) >= 3: drivers.append(('active_filings', f"{sf['filing_count_30d']} SEC filings in 30d"))
            if sf.get('fda_event_recent', 0) > 0: drivers.append(('fda_activity', f"Recent FDA events"))
            if sf.get('clinical_trial_active', 0) > 3: drivers.append(('clinical_pipeline', f"{sf['clinical_trial_active']} active trials"))
            if sf.get('earnings_surprise_last', 0) > 10: drivers.append(('earnings_beat', f"+{sf['earnings_surprise_last']:.0f}% earnings surprise"))
            if sf.get('days_to_earnings', 90) < 14: drivers.append(('earnings_imminent', f"Earnings in {sf['days_to_earnings']}d"))

        if sharpe_raw > 1.0: drivers.append(('strong_sharpe', f"Sharpe-like ratio {sharpe_raw:.2f}"))

        if is_selected:
            if not drivers:
                drivers.append(('balanced', 'Balanced return/risk profile'))
            reason = f"Rank #{rank}. {drivers[0][1]}"
            if len(drivers) > 1: reason += f"; {drivers[1][1]}"
        else:
            reason = "Not selected"
            if tk in {'VOO', 'QQQ', 'SOXX', 'XBI', 'SPY'}:
                reason = "ETF - excluded by design"
            elif r['ret_mean'] < 0:
                reason = f"Negative return prediction ({r['ret_mean']*100:.0f}%)"
            elif r['risk_mean'] > 0.7:
                reason = f"Excessive risk ({r['risk_mean']*100:.0f}%)"
            elif rank > config.N_SELECT:
                reason = f"Rank #{rank} - below cutoff (top {config.N_SELECT})"

        rationale[tk] = {
            'rank': rank,
            'selected': is_selected,
            'sector': sec,
            'composite_score': round(sc, 4),
            'score_components': {
                'sharpe_raw': round(sharpe_raw, 4),
                'confidence': round(r.get('conf_mean', 0), 4),
                'uncertainty': round(r.get('uncertainty', 0), 4),
                'sentiment_boost': round(sent_boost, 4),
                'event_risk': round(event_risk, 4),
            },
            'predictions': {
                'return_pct': round(r['ret_mean'] * 100, 2),
                'risk_pct': round(r['risk_mean'] * 100, 2),
                'return_std_pct': round(r.get('ret_std', 0) * 100, 2),
            },
            'drivers': [{'label': d[0], 'description': d[1]} for d in drivers[:5]],
            'reason': reason,
        }
        if sf:
            rationale[tk]['sentiment'] = {
                'news_7d': round(sf.get('news_sentiment_7d', 0), 3),
                'composite': round(sf.get('composite_sentiment', 0), 3),
                'event_risk': round(sf.get('event_risk_score', 0), 3),
                'filings_30d': sf.get('filing_count_30d', 0),
                'fda_events': sf.get('fda_event_recent', 0),
                'clinical_trials': sf.get('clinical_trial_active', 0),
                'days_to_earnings': sf.get('days_to_earnings', 90),
            }

    # Save output
    os.makedirs("results", exist_ok=True)
    output = {
        "selected": selected,
        "matrix": (best_W*100).round(2).tolist(),
        "matrix_krw": [[round(best_W[i,j]*config.TOTAL_CAPITAL_KRW) for j in range(3)] for i in range(3)],
        "metrics": {
            "return": round(pr*100, 2),
            "risk": round(pk*100, 2),
            "sharpe": round(pr/max(pk, .001), 2),
            "capital_krw": config.TOTAL_CAPITAL_KRW,
        },
        "ranking": [(tk, round(s, 4)) for tk, s in ranked],
        "predictions": {tk: {k: round(float(v), 4) for k, v in r.items()} for tk, r in mc.items()},
        "rationale": rationale,
    }
    if use_screen and sector_map:
        output["universe"] = {
            "total": len(tickers),
            "sector_map": {tk: sector_map.get(tk, 'X') for tk in tickers},
        }
    if sent_features:
        output["sentiment"] = {
            "model": getattr(config, 'SENTIMENT_MODEL', 'unknown'),
            "features": {tk: {k: round(float(v), 4) for k, v in sf.items()}
                         for tk, sf in sent_features.items()},
        }
    try:
        if blend_result is not None:
            output["blend_optimization"] = blend_result
    except NameError:
        pass

    json.dump(output, open("results/output.json", "w"), indent=2, ensure_ascii=False)
    print(f"  Saved: results/output.json")

    print(f"\n[Selection Rationale]")
    for tk in selected:
        rt = rationale[tk]
        print(f"  {tk:>6} [{rt['sector']}] score={rt['composite_score']:.3f}")
        print(f"         ret={rt['predictions']['return_pct']:+.1f}% risk={rt['predictions']['risk_pct']:.1f}%")
        for d in rt['drivers'][:3]:
            print(f"         -> {d['label']}: {d['description']}")

    # Auto-generate visualizations
    try:
        from visualize import generate_all_figures
        print(f"\n[4] Generating visualizations...")
        generate_all_figures(output, output_dir="results")
    except Exception as e:
        print(f"  Visualization failed: {e}")
        print(f"  (You can run manually: python visualize.py)")

    # Optional: Portfolio Backtest
    if run_backtest:
        try:
            from backtest import run_backtest as _bt
            _bt(n_folds=5, n_select=5, verbose=True)
        except Exception as e:
            print(f"\n  Backtest failed: {e}")
            print(f"  (You can run separately: python backtest.py)")


def _historical_train_and_predict(tickers, D):
    """Train on historical data (S&P 500 expanded) -> predict on current tickers."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from historical import auto_expand_universe, build_training_data

    print("  [1a] Building historical training data (S&P 500 expanded)...")
    expanded = auto_expand_universe()
    hist_X, hist_Y_ret, hist_Y_risk, hist_meta, _ = build_training_data(
        tickers=expanded,
        period=getattr(config, 'TRAINING_PERIOD', '10y'),
        snapshot_interval=getattr(config, 'TRAINING_SNAPSHOT_INTERVAL', 10),
    )

    # Feature selection + normalization
    feat_var = hist_X.var(0)
    feat_corr = np.array([abs(np.corrcoef(hist_X[:,d], hist_Y_ret)[0,1])
                          if np.std(hist_X[:,d]) > 1e-8 else 0
                          for d in range(hist_X.shape[1])])
    var_thr = getattr(config, 'VAR_THRESHOLD', 0.01)
    corr_thr = getattr(config, 'CORR_THRESHOLD', 0.05)
    keep = (feat_var > var_thr) & (feat_corr > corr_thr)
    if keep.sum() < 10: keep = feat_var > var_thr

    hist_X_sel = hist_X[:, keep]
    mu = hist_X_sel.mean(0); sigma = hist_X_sel.std(0) + 1e-8
    hist_X_n = np.clip((hist_X_sel - mu) / sigma, -5, 5)

    print(f"\n  [1b] Training ensemble on {hist_X_n.shape[0]:,} samples ({hist_X_n.shape[1]} features)...")

    # Log-transform volatility target for heteroscedastic NLL (Andersen et al. 2003)
    LOG_EPSILON = 1e-4
    hist_Y_risk_log = np.log(np.maximum(hist_Y_risk, LOG_EPSILON))

    X_t = torch.tensor(hist_X_n, dtype=torch.float32)
    yr_t = torch.tensor(hist_Y_ret, dtype=torch.float32)
    yk_t = torch.tensor(hist_Y_risk_log, dtype=torch.float32)

    # Train/val split for early stopping (80/20, shared across ensemble members)
    torch.manual_seed(config.RANDOM_SEED)
    N = X_t.shape[0]
    n_val = int(N * 0.2)
    perm = torch.randperm(N)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr_t, X_val_t = X_t[tr_idx], X_t[val_idx]
    yr_tr_t, yr_val_t = yr_t[tr_idx], yr_t[val_idx]
    yk_tr_t, yk_val_t = yk_t[tr_idx], yk_t[val_idx]
    print(f"  Train/val split: {X_tr_t.shape[0]:,}/{X_val_t.shape[0]:,} samples")

    arch = getattr(config, 'TRAINING_NN_ARCHITECTURE', [64, 32, 16])
    lr = getattr(config, 'TRAINING_LR', 0.0005)
    delta = getattr(config, 'TRAINING_HUBER_DELTA', 0.3)
    epochs = getattr(config, 'TRAINING_EPOCHS', 800)

    from models import HeteroscedasticDualHeadNN, heteroscedastic_loss

    models = []
    for seed in range(config.N_ENSEMBLE):
        torch.manual_seed(seed * 77)
        D_sel = hist_X_n.shape[1]
        model = HeteroscedasticDualHeadNN(in_dim=D_sel, hidden_dims=arch, dropout=0.2)
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                       weight_decay=getattr(config, 'TRAINING_WEIGHT_DECAY', 1e-4))

        best_val = float('inf'); patience = 0
        best_state = None
        best_ep = 0
        for ep in range(epochs):
            # Train step (dropout active)
            model.train()
            opt.zero_grad()
            pred = model(X_tr_t)
            train_loss, _, _ = heteroscedastic_loss(pred, yr_tr_t, yk_tr_t)
            if torch.isnan(train_loss): break
            train_loss.backward(); opt.step()

            # Val step (dropout off)
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val_t)
                val_loss, _, _ = heteroscedastic_loss(pred_val, yr_val_t, yk_val_t)
                val_loss_item = val_loss.item()

            if val_loss_item < best_val:
                best_val = val_loss_item
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_ep = ep
                patience = 0
            else:
                patience += 1
                if patience > getattr(config, 'EARLY_STOP_PATIENCE', 41): break

        # Restore best val-loss checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)

        models.append(model)
        print(f"    NN #{seed+1}: val_NLL={best_val:.6f} at epoch {best_ep+1}, stopped at {ep+1}")

    # Predict on CURRENT data
    print(f"\n  [1c] Predicting on current {len(tickers)} stocks...")
    import yfinance as yf
    from historical import _SliceFrame
    from data_auto import compute_technical_features

    current_vecs = []
    for tk in tickers:
        hist = yf.Ticker(tk).history(period="1y")
        hist = hist.dropna(subset=['Close'])
        close = hist['Close'].values
        high = hist['High'].values
        low = hist['Low'].values
        volume = hist['Volume'].values

        sf = _SliceFrame(close, high, low, volume)
        feats = compute_technical_features(sf)
        feat_names = sorted(feats.keys())
        vec = np.array([feats.get(k,0) for k in feat_names], dtype=float)
        vec = np.nan_to_num(vec, nan=0, posinf=0, neginf=0)
        current_vecs.append(vec)

    current_raw = np.array(current_vecs)

    if current_raw.shape[1] < hist_X.shape[1]:
        pad = np.zeros((current_raw.shape[0], hist_X.shape[1] - current_raw.shape[1]))
        current_raw = np.concatenate([current_raw, pad], axis=1)
    elif current_raw.shape[1] > hist_X.shape[1]:
        current_raw = current_raw[:, :hist_X.shape[1]]

    current_sel = current_raw[:, keep]
    current_n = np.clip((current_sel - mu) / sigma, -5, 5)
    X_cur_t = torch.tensor(current_n, dtype=torch.float32)

    mc = {}
    for i, tk in enumerate(tickers):
        all_rets, all_risks_log = [], []
        for model in models:
            model.eval()
            for m in model.modules():
                if isinstance(m, nn.Dropout): m.train()
            passes = config.MC_FORWARD_PASSES // config.N_ENSEMBLE
            for _ in range(passes):
                with torch.no_grad():
                    ret_mu, _, risk_log_mu, _ = model(X_cur_t[i:i+1])
                    all_rets.append(ret_mu[0].item())
                    all_risks_log.append(risk_log_mu[0].item())

        # Back-transform log-volatility to actual scale (Andersen 2003 convention)
        all_risks = np.exp(np.array(all_risks_log)).tolist()

        mc[tk] = {
            'ret_mean': np.mean(all_rets),
            'ret_std': np.std(all_rets),
            'risk_mean': np.mean(all_risks),
            'risk_std': np.std(all_risks),
            'conf_mean': 1.0 - np.std(all_rets),
            'uncertainty': np.std(all_rets) + np.std(all_risks),
        }

    train_artifacts = {
        'models': models,
        'keep': keep,
        'mu': mu,
        'sigma': sigma,
        'hist_X_shape_1': hist_X.shape[1],
    }
    return mc, train_artifacts


if __name__ == "__main__":
    main()
