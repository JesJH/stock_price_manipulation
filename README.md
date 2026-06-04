# Stock Price Manipulation Detection

A machine learning pipeline to identify securities likely to be exploited in pump-and-dump schemes — replacing a rule-based fraud detection system at a brokerage and improving coverage while reducing unnecessary trading blocks.

> **About this portfolio project**
>
> This is a public reproduction of production work built at a brokerage. The purpose is to showcase the end-to-end methodology — problem framing, dataset construction, feature engineering, modelling, and evaluation — not to claim production-ready results on public data.
>
> **Labelling in production vs. this project:**
> In the real system, the brokerage's fraud operations team labelled which securities were used in pump-and-dump schemes based on confirmed account takeover incidents observed directly in internal transaction records. That ground truth was clean, timely, and business-verified.
>
> For this portfolio project, there is no formal public body that authoritatively labels a security as having been used in a pump-and-dump scheme. The SEC charges *individuals* for running schemes — not securities — and those charges often come months or years after the fact. Here we use SEC enforcement announcements, trading suspension disclosures, and company 8-K filings as **proxy labels**: securities that were publicly identified in connection with manipulation activity. These are reasonable proxies but are not equivalent to the internal business labels used in production.
>
> **Other scope differences:**
> All data is sourced from public APIs (SEC EDGAR, yfinance) because proprietary brokerage data cannot be shared. The customer-level model is documented but not implemented (see [Two-Model Architecture](#two-model-architecture)). The analysis is scoped to **US and Canadian equities** because SEC data is US-focused and yfinance coverage is strongest for North American exchanges — in production, no country filter was applied.

---

## Background

A brokerage was experiencing an increasing number of account takeover (ATO) incidents in which compromised customer accounts were used to artificially inflate the price of a target security. Once elevated, the fraudster — holding a pre-existing position in their own account — would sell at the inflated price, profiting at the expense of the brokerage and its customers. These pump-and-dump schemes caused direct financial losses with no systematic way to detect targeted securities in advance.

The existing system relied on three customer-level signals only — account age, account value, and trade frequency over a fixed window — and flagged approximately **50% of all securities** as potentially at risk. This high false-positive rate was operationally unsustainable and provided limited signal on which specific securities to block.

---

## Business Goal

Replace the legacy rule-based approach with a model-driven pipeline that flags securities most likely to be exploited for pump-and-dump manipulation — surfacing them at least 5 days before the anticipated fraud event so the fraud team can review and place targeted blocks before fraudulent trades execute.

---

## Two-Model Architecture

The full production system comprises two independent models addressing different sides of the same fraud event:

| Model | Question | Label | Features |
|---|---|---|---|
| **Security Model** *(this repo)* | Will this security be used in a pump-and-dump scheme in the next 5 days? | Confirmed SEC enforcement cases | Price behavior, trading volume, security metadata, anomaly score |
| **Customer Model** *(not in portfolio)* | Is this customer likely to have their account taken over and used in a fraudulent trade? | ATO-flagged customers | Age, tenure, daily account balance, country of residency, number of accounts, held positions |

Both models must fire for the scheme to complete — a security has to be targeted and a customer account has to be compromised. In production, outputs from both are combined to prioritize which blocks to place.

---

## Data

### True Positives — Confirmed Pump-and-Dump Cases
- Sourced from SEC litigation releases and administrative proceedings (`scripts/download_data.py --source sec`)
- Each case provides a confirmed ticker symbol and a fraud date (D), which anchors the observation window
- Original production work used ~20 confirmed in-house ATO cases over a 5-month period

### Price and Volume Data
- Daily OHLCV pulled via `yfinance` for each TP and TN ticker over the observation window
- Features: closing price, open price, intraday high/low, trading volume

### Security Metadata
- Market capitalisation, country of origination, currency, exchange, sector
- One-hot encoded for modeling

### Customer Features *(original work only — not replicated here)*
- Customer demographics and transaction history aggregated to the security level
- Excluded from this portfolio reproduction — requires proprietary brokerage data

---

## Challenges

**Small labeled dataset** — With only ~20 confirmed true positives in the original dataset (and a larger but still limited set from SEC data in this reproduction), standard supervised learning approaches risk overfitting. Dataset construction and feature engineering required careful design to extract signal from limited examples.

**Class imbalance** — Fraudulent securities represent a tiny fraction of all tradeable equities. The legacy system's 50% flag rate shows how badly uncalibrated a naive approach becomes; deliberate sampling strategies are needed to avoid the model predicting everything as negative.

**Noisy price signals** — EDA of security prices in the observation window showed no clean directional trend leading up to fraud events. Raw price series are unreliable direct inputs; statistical summaries across the window are more informative.

**Asymmetric cost function:**
- False Positives (FP): legitimate securities get blocked, harming customer experience and revenue
- False Negatives (FN): fraudulent activity goes undetected, causing financial and reputational loss

---

## Approach

### Observation Window

Features are built over a **D-90 to D-5** window — 90 days of security behaviour ending 5 days before the fraud date. The 5-day buffer is a hard production constraint: the model must surface risky securities far enough in advance for the fraud team to act. Training up to D would create leakage that does not hold at inference time.

The 90-day window was a business starting point. An empirical window-length analysis (see `notebooks/pump_and_dump_detection.ipynb`) compares model performance across 30, 45, 60, 90, and 120-day windows to validate or refine this choice.

### Feature Engineering

Three feature groups are constructed over the observation window:

**Price behaviour** — Raw price trends showed no consistent pre-fraud pattern in EDA, so features focus on statistical behaviour: day-over-day changes at 1, 3, 5, and 10-day intervals, plus rolling counts, sums, and averages across close, open, high, and low prices.

**Trading volume** — Volume and liquidity can signal early manipulation. Sustained buy-side pressure drives prices up during the pump phase; a net sell spike marks the dump. Features capture these dynamics across the window.

**Security metadata** — Static attributes (market cap, country, currency, exchange) are one-hot encoded.

All transformations are encapsulated in `FeatureTransformer` to keep the pipeline modular and reproducible.

### Anomaly Scoring (Isolation Forest)

Before classification, an Isolation Forest is trained on the full security universe to produce an **anomaly score per security**. This score is appended as an additional feature to the classification dataset — it adds an unsupervised signal about how unusual a security's behaviour is, without replacing the interpretable decision tree.

### Model 1 — Centroid-Based True Negative Selection

With severe class imbalance, unconstrained negative sampling risks training a classifier that never needs to distinguish subtle differences. Each of the 20 TP securities acts as a centroid; all candidate TN securities are assigned to their nearest TP centroid by behavioural similarity, so the classifier learns genuine distinctions rather than trivial ones.

Three downsampling methods are evaluated at a baseline TN:TP ratio of 5:1 (grid-searched across 1:1, 3:1, 5:1, 10:1):

- **Random sampling** — For each TP centroid, N TNs are drawn randomly from the assigned pool. Baseline. May include TNs that are trivially different from the TP.

- **Distance-ranked sampling** — TNs ranked by Euclidean distance to their assigned TP centroid; the N closest are selected. Forces the classifier to learn subtle distinctions between behaviorally similar securities.

- **Stratified-by-distance sampling** — TNs are binned into equal-count quantile groups by distance, then sampled from each bin. Exposes the classifier to both near-boundary hard cases and clearly distinct easy cases — typically produces a more robust decision boundary.

Because Euclidean distance is sensitive to scale, all features are normalised with a robust scaler (median/IQR) before centroid assignment. The scaler is fit on the training fold only inside each cross-validation iteration to prevent leakage.

Model 1 has no intrinsic metric. The optimal configuration is selected via a grid search evaluated on Model 2's held-out performance. The full pipeline (Model 1 selection → Model 2 training) is wrapped in leave-one-out cross-validation to prevent selection bias from leaking into the holdout.

### Model 2 — Decision Tree Classification

A Decision Tree classifier distinguishes manipulated securities from non-manipulated ones on the curated dataset. Decision trees are required by the business: fraud reviewers must be able to trace exactly which behavioural conditions triggered a flag. Black-box models were ruled out.

An 80/20 split (16 TPs for training, 4 for testing) is held constant throughout.

### Threshold Optimisation

Classification thresholds are tuned explicitly around the FP/FN cost tradeoff rather than raw accuracy. The final threshold reflects the business's relative tolerance for each error type and is selected from the LOO-CV performance curve.

---

## Results

*To be populated after model training is complete.*

Baseline to beat: the legacy rule-based system flagged ~50% of securities. The objective is to achieve materially higher precision at comparable or better recall.

---

## Next Steps

- **Observation window validation:** Empirically confirm whether 90 days is optimal or whether a shorter/longer window captures signal better.
- **Channel-based features:** ATO fraud flows predominantly through digital channels (online/mobile). Adding transaction channel as a feature could improve precision without broadening the block list.
- **Retraining pipeline:** As fraud operations surface new confirmed cases, a systematic retraining process is needed to keep the model current as fraudster tactics evolve.
- **Customer model integration:** Combining security-level flags with the customer-level ATO model enables risk scoring at the (customer × security) level, concentrating review effort where both signals fire.

---

## Running the Project

```bash
git clone <repo>
cd stock_price_manipulation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download confirmed P&D tickers from SEC + OHLCV data from yfinance
python scripts/download_data.py --source all

# Open the analysis notebook
jupyter notebook notebooks/pump_and_dump_detection.ipynb
```

---

## Skills & Tools

`Python` · `scikit-learn` · `pandas` · `yfinance` · `SEC EDGAR` · `Jupyter` · `K-Means / Centroid Clustering` · `Decision Tree` · `Isolation Forest` · `Machine Learning` · `Fraud Detection` · `Financial Services`
