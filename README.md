# Stock Price Manipulation Detection

A machine learning pipeline to identify securities likely to be exploited in pump-and-dump schemes, replacing a rule-based fraud detection system and improving coverage while reducing unnecessary trading blocks.

> **About this portfolio project**
>
> This project is inspired by and replicates work built in a professional production environment. It is intended to showcase the end-to-end methodology — problem framing, dataset construction, feature engineering, modelling, and evaluation. Details about the original production system, employer, and internal metrics are intentionally omitted.
>
> **On labelling:** In the production system, the fraud operations team labelled confirmed cases directly from internal transaction records — clean, timely, business-verified ground truth. For this portfolio version, no formal public body authoritatively labels a security as having been used in a pump-and-dump scheme. We use SEC enforcement announcements and company 8-K disclosures as proxy labels. These are reasonable approximations but are not equivalent to internal business labels.
>
> **Data and scope:** All data is sourced from public APIs (SEC EDGAR, yfinance) since proprietary data cannot be shared. The analysis is scoped to US and Canadian equities due to data availability — no such filter existed in production.

---

## Background

Financial institutions can be exposed to pump-and-dump schemes in which compromised customer accounts are used to artificially inflate the price of a target security. Once elevated, the fraudster — holding a pre-existing position — sells at the inflated price, causing direct financial loss to the institution and its customers.

Existing rule-based detection systems often rely on a small number of customer-level signals and can produce high false-positive rates, making them operationally unsustainable and leaving meaningful fraud exposure unaddressed.

---

## Business Goal

Replace the legacy rule-based approach with a model-driven pipeline that flags securities most likely to be exploited for pump-and-dump manipulation — surfacing them with sufficient lead time for the fraud team to review and place targeted blocks before fraudulent trades execute.

---

## Two-Model Architecture

The broader system design comprises two independent models addressing different sides of the same fraud event:

| Model | Question | Features | Portfolio status |
|---|---|---|---|
| **Security Model** *(this repo)* | Will this security be used in a pump-and-dump scheme? | Price behaviour, trading volume, security metadata, anomaly score | Implemented |
| **Customer Model** *(not included)* | Is this customer at risk of account takeover for use in a fraudulent trade? | Demographics, account history, transaction patterns | Requires proprietary data — not replicated |

---

## Data

### True Positives — Confirmed Pump-and-Dump Cases
- Sourced from SEC enforcement announcements and company 8-K disclosures via SEC EDGAR API
- Each case provides a confirmed ticker and a reference date anchoring the observation window
- See [PARKING_LOT.md](PARKING_LOT.md) for a discussion of the labelling limitations of public data

### Price and Volume Data
- Daily OHLCV pulled via `yfinance` for each TP and TN ticker over the observation window
- Features: closing price, open price, intraday high/low, trading volume

### Security Metadata
- Market capitalisation, country of origination, currency, exchange, sector
- One-hot encoded for modelling

### Customer Features *(not replicated)*
- Customer demographics and transaction history were used in the production system
- Excluded here — requires proprietary data

---

## Challenges

**Small labeled dataset** — Confirmed pump-and-dump cases are rare and difficult to source publicly. Standard supervised learning risks overfitting; careful dataset construction and feature engineering are needed to extract signal from limited examples.

**Class imbalance** — Fraudulent securities represent a tiny fraction of all tradeable equities, requiring deliberate sampling strategies to prevent the model from defaulting to predicting everything as negative.

**Noisy price signals** — EDA showed no clean directional price trend leading up to fraud events. Raw price series are unreliable direct inputs; statistical summaries across the observation window are more informative.

**Asymmetric cost function:**
- False Positives (FP): legitimate securities get blocked, harming customer experience and revenue
- False Negatives (FN): fraudulent activity goes undetected, causing financial and reputational loss

---

## Approach

### Observation Window

Features are built over a **D-90 to D-5** observation window — 90 days of security behaviour ending several days before the fraud reference date. The buffer ensures the model must surface risky securities with enough lead time to act, preventing leakage from training on data immediately before the event.

An empirical window-length analysis (see `notebooks/pump_and_dump_detection.ipynb`) compares performance across 30, 45, 60, 90, and 120-day windows.

### Feature Engineering

Three feature groups are constructed over the observation window:

**Price behaviour** — Statistical summaries rather than raw levels: day-over-day changes at multiple intervals (1, 5, 10, 15, 30, 45, 90 days), rolling means, standard deviations, and positive-day counts across close, open, high, and low prices. Overnight gap (today's open vs. yesterday's close) and intraday range metrics are also included.

**Trading volume** — Volume and liquidity can signal early manipulation. Features capture N-day volume changes and rolling statistics across the window.

**Security metadata** — Static attributes (market cap, country, currency, exchange) are one-hot encoded.

All transformations are encapsulated in `FeatureTransformer` to keep the pipeline modular and reproducible.

### Anomaly Scoring (Isolation Forest)

An Isolation Forest is trained on the broader security universe to produce an **anomaly score per security**. This score is appended as an additional feature — adding unsupervised signal about how unusual a security's behaviour is without replacing the interpretable classifier.

### Model 1 — Centroid-Based True Negative Selection

With severe class imbalance, unconstrained negative sampling risks producing a classifier that never needs to distinguish subtle differences. Each confirmed TP security acts as a centroid; candidate TN securities are assigned to their nearest TP centroid by behavioural similarity, so the classifier learns genuine distinctions rather than trivial ones.

Three downsampling methods are evaluated via grid search:

- **Random sampling** — Randomly draw N TNs from each centroid's pool. Baseline.
- **Distance-ranked sampling** — Select the N closest TNs to each centroid. Maximises match quality.
- **Stratified-by-distance sampling** — Bin TNs into equal-count quantile groups by distance, then sample from each bin. Balances hard near-boundary cases with distinct easy cases.

Features are normalised with a robust scaler (median/IQR) before centroid assignment. The scaler is fit on the training fold only inside each cross-validation iteration to prevent leakage. The full pipeline is wrapped in leave-one-out cross-validation given the small TP set.

### Model 2 — Decision Tree Classification

A Decision Tree classifier distinguishes manipulated securities from non-manipulated ones. Decision trees were chosen to preserve interpretability — fraud reviewers need to be able to trace exactly which behavioural conditions triggered a flag.

### Threshold Optimisation

Classification thresholds are tuned explicitly around the FP/FN cost tradeoff rather than raw accuracy. The final threshold reflects the relative business tolerance for each error type.

---

## Results

*To be populated after model training is complete.*

---

## Running the Project

```bash
git clone <repo>
cd stock_price_manipulation
pip install -r requirements.txt

# Download OHLCV data for confirmed P&D cases
python scripts/download_data.py --source yfinance

# Open the analysis notebook
jupyter notebook notebooks/pump_and_dump_detection.ipynb
```

---

## Skills & Tools

`Python` · `scikit-learn` · `pandas` · `yfinance` · `SEC EDGAR` · `Jupyter` · `Centroid Clustering` · `Decision Tree` · `Isolation Forest` · `Machine Learning` · `Fraud Detection` · `Financial Services`
