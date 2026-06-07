# CLAUDE.md — Stock Price Manipulation Detection

Use this file to resume work across devices. It captures design decisions, open questions, implementation status, and key context that isn't derivable from the code alone.

---

## Project Summary

Portfolio project replicating work built in a professional production environment. ML pipeline to detect securities likely to be exploited in pump-and-dump schemes, replacing a rule-based system. True positives are sourced from public SEC data as proxy labels — the production system used internally verified fraud cases.

See README.md for full background, approach, and modelling design.

---

## Repo Structure

```
stock_price_manipulation/
├── CLAUDE.md               ← this file
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/               # downloaded from yfinance / SEC EDGAR
│   ├── processed/         # feature-engineered, split datasets
│   └── external/          # SEC enforcement action list, ticker metadata
├── src/
│   ├── data/
│   │   ├── downloader.py          # fetches OHLCV data via yfinance
│   │   ├── sec_scraper.py         # pulls confirmed P&D tickers from SEC
│   │   └── preprocessor.py       # cleans raw data into processed/
│   ├── features/
│   │   ├── feature_transformer.py # main FeatureTransformer class
│   │   └── feature_config.py      # window sizes, rolling periods, etc.
│   ├── models/
│   │   ├── centroid_selector.py   # Model 1: TN downsampling (3 methods)
│   │   ├── classifier.py          # Model 2: Decision Tree
│   │   └── isolation_forest.py    # Alternative anomaly detection
│   ├── pipeline/
│   │   ├── training_pipeline.py   # end-to-end: data → features → M1 → M2
│   │   └── inference_pipeline.py  # score new securities in production
│   └── evaluation/
│       ├── metrics.py             # precision, recall, FP/FN cost calc
│       └── threshold_optimizer.py # threshold search over FP/FN tradeoff
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── scripts/
│   ├── download_data.py           # CLI: fetch price+volume data
│   ├── train_model.py             # CLI: run full training pipeline
│   └── evaluate_model.py          # CLI: evaluate saved model on holdout
├── tests/
└── outputs/
    ├── models/                    # serialized model artifacts (.pkl/.joblib)
    ├── plots/
    └── reports/
```

---

## Data Strategy (Public Proxy for Portfolio)

Since the original work used proprietary brokerage data, this repo uses publicly sourced proxies:

### True Positives — Confirmed P&D Securities
- **Primary source:** SEC enforcement actions database
  - Litigation releases: https://www.sec.gov/litigation/litreleases.htm
  - Admin proceedings: https://www.sec.gov/divisions/enforce/enforceactions.htm
  - Filter for pump-and-dump / market manipulation cases → extract ticker symbols
- **Secondary source:** Academic labeled datasets (e.g., papers on P&D detection)
- Script: `scripts/download_data.py --source sec` populates `data/external/tp_tickers.csv`

### Price / Volume Data (OHLCV)
- **Source:** Yahoo Finance via `yfinance`
- For each TP ticker: D-90 to D (fraud date from SEC action) OHLCV
- For TN candidates: same-length windows sampled from the same calendar period
- Script: `scripts/download_data.py --source yfinance` populates `data/raw/`

### Security Metadata
- Market cap, exchange, country, currency, sector: from `yfinance` ticker info
- Index membership: from exchange listing files (NASDAQ, NYSE, AMEX CSVs from exchange websites)

### Customer Features
- **Not replicable publicly** — proprietary transaction data
- Placeholder: synthetic customer behavioral features generated from distributions observed in literature, or omitted for the public portfolio version
- Document clearly in notebooks that this feature group was available at the firm but excluded here

---

## Two-Model Architecture

This project contains **two separate models** addressing different sides of the fraud event:

| Model | Predicts | Features | Portfolio status |
|---|---|---|---|
| **Security Model** | Which securities will be used in P&D in next 5 days | Price behavior, volume, metadata, Isolation Forest score | **Implemented here** |
| **Customer Model** | Which customers are likely to have their accounts taken over | Age, tenure, daily balance, country, # of accounts, held position | Documented only — requires proprietary customer data |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Observation window | D-90 to D-5 | 5-day buffer = production lead time; 90 days to be validated (see open questions) |
| TN D date | Same calendar window as matched TP | TNs are assigned to a TP centroid and share that TP's D date |
| TN:TP ratio | 5:1 starting point; grid search to confirm | Chosen by judgment; optimal ratio should be validated empirically |
| CV strategy | Leave-one-out | Small TP set; maximise training signal per fold |
| Feature scaling | Robust scaler (median/IQR) | Outlier-resistant; applied before Model 1 (centroid distances) only |
| Decision tree (unscaled) | Features not scaled for DT | Preserve interpretability for fraud reviewers |
| TN selection wrapping | Model 1 re-runs inside each CV fold | Prevents selection bias leakage into holdout |
| Isolation Forest role | Anomaly score used as a **feature** input to DT | Adds unsupervised signal without replacing the interpretable classifier |

---

## Open Questions (Resolve Before Implementing)

1. **Observation window length** — 90 days was a business decision, not analytically validated. Plan: in `notebooks/01_eda.ipynb`, compare LOO-CV performance at 30, 45, 60, 90, 120-day windows.

2. **Grid search ranges for Model 1:**
   - TN:TP ratio: test 1:1, 3:1, 5:1, 10:1
   - Distance bins Q: 4, 5, 10
   - Max distance cutoff: percentile-based (75th, 90th) or none

3. **FP/FN cost ratio** — Needed for threshold optimization. Not defined yet. Placeholder: 1 FN = 5× cost of 1 FP.

---

## Implementation Status

- [x] Folder structure and boilerplate
- [x] Data layer: `src/data/sec_scraper.py` + `src/data/downloader.py` + `src/data/preprocessor.py`
- [x] Scripts: `scripts/download_data.py`
- [x] Feature engineering: `src/features/feature_transformer.py` + `src/features/feature_config.py`
- [x] Model 1: `src/models/centroid_selector.py` (random, distance_ranked, stratified; + matching_report())
- [x] Isolation Forest: `src/models/isolation_forest.py` (AnomalyScorer — IF score as feature)
- [x] Model 2: `src/models/classifier.py` (FraudClassifier — Decision Tree wrapper)
- [x] Training pipeline: `src/pipeline/training_pipeline.py` (LOO-CV wrapping M1 → IF → M2, leakage-safe)
- [x] Threshold optimizer: `src/evaluation/threshold_optimizer.py`
- [x] Evaluation metrics: `src/evaluation/metrics.py` (FP/FN cost-aware)
- [x] TN universe: `src/data/tn_universe.py` (TNUniverseBuilder) + `downloader.download_tn_batch()`
- [x] Order flow features: CLV, OBV, A/D Line, CMF, MFI added to `feature_transformer.py`
- [ ] Inference pipeline: `src/pipeline/inference_pipeline.py`
- [x] CLI: `scripts/train_model.py`, `scripts/evaluate_model.py`, `scripts/download_data.py --source tn`
- [x] Main notebook: `notebooks/pump_and_dump_detection.ipynb`

---

## Measurement Plan (Offline / Portfolio)

- **Primary:** Precision / Recall at optimised threshold via LOO-CV
- **Baseline:** Legacy rule-based system had a high false-positive flag rate — ML model targets materially higher precision at comparable or better recall
- **Secondary:** FP/FN cost-weighted score using placeholder 5:1 cost ratio
- **Observation window sensitivity:** compare AUC-PR across window lengths (30–120 days)

---

## Environment Setup

**On the primary machine:** Uses the Anaconda base environment — no project-specific venv needed. All packages already installed at `/opt/anaconda3`. Just run `python` or open the notebook and select the Anaconda kernel.

**On a new machine:** Either reuse an existing Anaconda/conda environment that has the packages, or create a fresh one:

```bash
git clone <repo>
cd stock_price_manipulation

# Option A — conda (recommended, matches primary machine)
conda create -n pump_dump python=3.13
conda activate pump_dump
pip install -r requirements.txt

# Option B — plain venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python version: 3.13 (primary machine) — 3.10+ will work.

---

## Session Notes

_Add any important context from each working session here (date + note), e.g.:_
- 2026-05-30: Initial structure created. Open questions above not yet resolved. SEC scraper + yfinance downloader not yet implemented.
