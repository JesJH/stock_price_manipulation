# Methodology

Full design rationale for the pump-and-dump detection pipeline. Each section explains the problem the step exists to solve, not just what it does.

---

## Step 1 — Observation Window Construction

**Problem:** The model must fire *before* the fraud occurs, not during it. Training on data from the days immediately preceding the fraud event risks using signals that only become visible when manipulation is already underway — too late to act.

**Approach:** Features are computed over a **D-90 to D-5** window, where D is the fraud reference date. The 5-day buffer is the operational lead time requirement, enforced at training time. The 90-day lookback was the original business starting point; an empirical comparison across 30/45/60/90/120-day windows is included in the notebook to validate it.

---

## Step 2 — Feature Engineering

**Problem:** Raw OHLCV price levels are not comparable across securities with different price scales. A single snapshot misses the *temporal dynamics* that characterise manipulation — sustained build-up over weeks, not a high price on one day.

**Approach:** Each security's OHLCV series is transformed into a flat feature vector of statistical summaries. Implemented in `src/features/feature_transformer.py`.

| Feature group | What it captures |
|---|---|
| N-day price changes (1/5/10/15/30/45/90d) | Rate and direction of price movement at multiple horizons |
| Rolling mean, std, positive-day count | Trend consistency and volatility across the window |
| Overnight gap | Information flow between sessions (today's open vs. yesterday's close) |
| Intraday metrics (range, close-minus-open, etc.) | Within-session buy/sell pressure |
| Volume changes and rolling stats | Liquidity build-up and accumulation |
| Order flow proxies: CLV, OBV, CMF, MFI | Approximations of buy vs. sell pressure derived from OHLCV |
| Security metadata (exchange, market cap, country) | Structural context for centroid matching |

**Scaling note:** Features are returned unscaled. Robust scaling (median/IQR) is applied only inside `CentroidSelector` for distance computation. The Decision Tree receives unscaled features so split thresholds remain interpretable in human terms.

**Order flow limitation:** True buy vs. sell volume splits require tick-level data (Polygon, Bloomberg TAQ). The OHLCV-derived proxies above are approximations. In the production system, actual transaction-level order flow was available as a materially stronger signal. See `PARKING_LOT.md` for full discussion.

---

## Step 3 — True Negative Candidate Sourcing

**Problem:** TNs cannot be sampled at random from all equities. Randomly picking large-cap S&P 500 stocks as negatives makes the classifier's job trivial — it separates on market cap or sector, not on the behavioural patterns that matter.

**Approach:** A candidate pool of small-cap and OTC securities is built from public NASDAQ listing files (the same universe our TPs are drawn from). For each TP's D-date, OHLCV is downloaded for all candidates over the *same 90-day window* as that TP. Tagging each TN with the TP's D-date it was evaluated against is essential for leakage prevention in Step 6.

Implemented in `src/data/tn_universe.py` and `src/data/downloader.download_tn_batch()`.

---

## Step 4 — Centroid-Based TN Selection (Model 1)

**Problem:** Even within the small-cap OTC universe, many TN candidates will be genuinely dissimilar to any TP. Including them gives the classifier easy wins that don't generalise to real fraud detection.

**Approach:** Each TP acts as a centroid in robust-scaled feature space. TN candidates are assigned to their nearest TP centroid by Euclidean distance, then downsampled so the classifier sees TPs alongside the *most similar* TNs — securities that look like fraud targets but were not manipulated. Three methods are evaluated:

| Method | Selection rule | Purpose |
|---|---|---|
| Random | Draw N TNs at random from each centroid's pool | Baseline |
| Distance-ranked | Take the N closest TNs per centroid | Maximise match quality; hardest cases |
| Stratified-by-distance | Sample from distance quantile bins | Balance hard near-boundary cases with clearly distinct ones |

The optimal method and TN:TP ratio are identified via grid search inside LOO-CV.

Implemented in `src/models/centroid_selector.py`. Call `matching_report()` after selection to inspect per-centroid distance diagnostics.

---

## Step 5 — Anomaly Scoring (Isolation Forest)

**Problem:** With a small TP set, the Decision Tree has limited labelled signal to learn what "unusual" looks like. Relying on labelled examples alone may miss novel manipulation patterns.

**Approach:** An Isolation Forest is trained on the TN candidate pool as a baseline of "normal" security behaviour. It scores each security on how anomalous its behaviour is relative to normal — higher score means more isolated from typical patterns. This score is appended as a single additional feature column (`if_anomaly_score`) before the Decision Tree sees the data.

This adds unsupervised signal without replacing the interpretable classifier. The IF is re-fit from scratch inside each LOO-CV fold to prevent leakage.

Implemented in `src/models/isolation_forest.py`.

---

## Step 6 — Leakage-Safe Leave-One-Out Cross-Validation

**Problem:** With a small TP set, standard k-fold CV wastes training signal. More importantly, naive LOO-CV creates temporal leakage: TN candidates evaluated in TP_i's 2014 market window could appear as training examples when TP_i is the held-out test case. Any shared temporal signal (market regime, sector trend) would leak into the fold.

**Approach:** LOO-CV holds out one TP per fold. All TN candidates tagged with the held-out TP's D-date are *also* excluded from training and form the negative test set. CentroidSelector, IsolationForest, and the Decision Tree are all re-fit from scratch on training-fold data only.

```
For each TP_i (D-date = D_i):
  Test  : TP_i + all TNs where d_date == D_i
  Train : remaining TPs + all TNs where d_date != D_i
  → fit CentroidSelector on training TPs
  → select TNs from training TN pool
  → fit IsolationForest on selected training TNs
  → fit Decision Tree on training set (with IF score)
  → score test set
```

Out-of-fold predicted probabilities across all folds are used for threshold optimisation.

Implemented in `src/pipeline/training_pipeline.py`.

---

## Step 7 — Threshold Optimisation

**Problem:** A default 0.5 threshold ignores the asymmetric cost of errors. A missed fraud case (false negative) causes direct financial loss and reputational damage. A false alarm (false positive) blocks a legitimate trade, harms customer experience, and wastes analyst time. These costs are not equal.

**Approach:** The classification threshold is grid-searched over [0.01, 0.99] to minimise:

```
cost = FP × fp_cost + FN × fn_cost
```

A placeholder cost ratio of `fn_cost = 5 × fp_cost` is used (1 missed fraud = 5 false alarms). The optimal threshold from out-of-fold predictions is stored in the trained pipeline and used at inference time. The full cost curve, precision/recall vs. threshold, and PR-space scatter are available via `ThresholdOptimizer.plot_*()`.

Implemented in `src/evaluation/threshold_optimizer.py`.
