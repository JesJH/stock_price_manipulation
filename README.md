# Stock Price Manipulation Detection

An ML pipeline that identifies securities likely to be used in pump-and-dump fraud schemes, giving a fraud operations team time to place targeted blocks before fraudulent trades execute.

> **Portfolio project** — This replicates the methodology of work built in a professional production environment. Employer details and internal metrics are intentionally omitted. See [Caveats](#caveats) for data and scope limitations.

---

## What We're Trying to Achieve

In a pump-and-dump scheme, a fraudster artificially inflates a security's price through coordinated buying (often using compromised brokerage accounts), then sells their pre-existing position at the peak. The institution absorbs the loss.

The goal is to **flag at-risk securities at least 5 days in advance** — enough lead time for a fraud analyst to review and block. The 5-day buffer is a hard operational constraint that shapes every modelling decision.

The system replaces a legacy rule-based detector that flagged a very high proportion of all securities, producing an unworkable false-positive rate and leaving genuine fraud cases undetected.

---

## Modelling Approach

The pipeline has seven steps. Each addresses a specific challenge in building a fraud model on rare, noisy, imbalanced data.

| Step | What it achieves |
|---|---|
| 1. Observation window | Locks in a D-90 to D-5 window — enforcing the lead-time constraint at training time |
| 2. Feature engineering | Converts raw OHLCV series into statistical summaries that capture manipulation patterns |
| 3. TN candidate sourcing | Builds a pool of "normal" securities from the same universe and time period as TPs |
| 4. Centroid-based TN selection | Matches each TN to its nearest TP in feature space — so the model learns subtle distinctions, not trivial ones |
| 5. Isolation Forest scoring | Adds an unsupervised anomaly signal as a feature, supplementing labelled training signal |
| 6. Decision Tree classification | Produces an interpretable model — fraud reviewers can trace exactly what triggered a flag |
| 7. Threshold optimisation | Tunes the operating point for the asymmetric FP/FN cost tradeoff in a fraud context |

**→ See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full design rationale behind each step.**

---

## Project Structure

```
stock_price_manipulation/
├── data/
│   ├── raw/               # OHLCV parquet files (gitignored)
│   ├── processed/         # Feature matrices (gitignored)
│   └── external/          # tp_tickers.csv, tn_candidates.csv (tracked)
├── src/
│   ├── data/
│   │   ├── downloader.py          # yfinance OHLCV + metadata fetcher
│   │   ├── preprocessor.py        # country + history filters
│   │   ├── sec_scraper.py         # SEC EDGAR TP label sourcing
│   │   └── tn_universe.py         # TN candidate pool builder
│   ├── features/
│   │   ├── feature_transformer.py # OHLCV → flat feature vector
│   │   └── feature_config.py      # window sizes, bins, constants
│   ├── models/
│   │   ├── centroid_selector.py   # TN downsampling (3 methods)
│   │   ├── classifier.py          # Decision Tree wrapper
│   │   └── isolation_forest.py    # Anomaly scorer
│   ├── pipeline/
│   │   └── training_pipeline.py   # LOO-CV: M1 → IF → M2, leakage-safe
│   └── evaluation/
│       ├── metrics.py             # Precision/recall + cost-weighted scoring
│       └── threshold_optimizer.py # FP/FN cost threshold search
├── notebooks/
│   └── pump_and_dump_detection.ipynb   # End-to-end analysis + results
├── scripts/
│   ├── generate_demo_data.py      # Synthetic OHLCV for demonstration
│   ├── download_data.py           # Fetch real TP/TN data from yfinance
│   ├── train_model.py             # CLI training runner
│   └── evaluate_model.py          # CLI evaluation runner
└── outputs/
    ├── models/                    # Saved pipeline artifacts
    └── plots/                     # Evaluation charts
```

---

## Setup

**Requirements:** Python 3.10+, packages in `requirements.txt`.

```bash
git clone <repo>
cd stock_price_manipulation

# Option A — conda (recommended)
conda create -n pump_dump python=3.13
conda activate pump_dump

# Option B — venv
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

### Run the demo (synthetic data — recommended starting point)

Generates synthetic OHLCV series and runs the full pipeline end-to-end in under a minute:

```bash
python scripts/generate_demo_data.py
jupyter notebook notebooks/pump_and_dump_detection.ipynb
```

### Run with real data

Downloads real TN candidates from public NASDAQ listings via yfinance (~40–50 min):

```bash
python scripts/download_data.py --source tn
# Then run the notebook — it detects real data automatically
```

---

## Caveats

**Synthetic demo data** — The default run uses generated price series (pump-and-dump patterns injected for TPs, random walks for TNs). This is for demonstration only. All modelling code runs identically on real downloaded data; the pipeline was designed and validated against real securities.

**Proxy labels** — No formal public body labels a security as pump-and-dumped. SEC enforcement announcements and 8-K disclosures are used as proxies. In the production system, the fraud operations team provided directly verified labels from internal investigation records — a materially cleaner signal.

**Proprietary features excluded** — The production model included customer transaction data (buy/sell volumes, account-level activity) as features. True order flow data requires paid tick-level feeds (Polygon, Bloomberg TAQ). This repo uses OHLCV-derived proxies (CLV, OBV, CMF, MFI) as approximations.

**Country scope** — Analysis is restricted to US and Canadian equities. The original system applied no country filter. The restriction here is a data availability decision: SEC enforcement data is predominantly US-focused and yfinance coverage is most reliable for North American exchanges.

**Customer Model not included** — A second model predicting which customer accounts are at risk of takeover (used to route fraudulent trades into the flagged securities) requires proprietary customer data and is not replicated here.

---

## Further Reading

| Document | Contents |
|---|---|
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Full design rationale for each pipeline step |
| [docs/PARKING_LOT.md](docs/PARKING_LOT.md) | Feature ideas, data limitations, and deferred decisions |
| [CLAUDE.md](CLAUDE.md) | Development context and implementation status |
