# Stock Price Manipulation Detection

A machine learning pipeline to identify securities likely to be exploited in pump-and-dump schemes, replacing a rule-based fraud detection system with a model-driven approach that improves precision and reduces unnecessary trading blocks.

> **About this portfolio project**
>
> This project replicates the methodology of work built in a professional production environment. It is intended to showcase the end-to-end approach — problem framing, dataset construction, feature engineering, modelling, and evaluation. Details about the original production system, employer, and internal metrics are intentionally omitted.

---

## Background

Financial institutions can be exposed to pump-and-dump schemes in which compromised customer accounts are used to artificially inflate the price of a target security. Once elevated, the fraudster — holding a pre-existing long position — sells at the inflated price, causing direct financial loss to the institution and its customers.

Legacy rule-based detection systems often rely on a small number of hard-coded thresholds and produce high false-positive rates, making them operationally unsustainable and leaving meaningful fraud exposure undetected.

---

## Business Goal

Detect securities likely to be used in a pump-and-dump scheme **at least 5 days before the fraud event**, giving the fraud operations team time to review and place targeted blocks before fraudulent trades execute. The 5-day lead time is the key operational constraint shaping every modelling decision.

---

## Two-Model Architecture

The broader system design addresses two different sides of the same fraud event. This repository implements the Security Model only.

| Model | Question answered | Portfolio status |
|---|---|---|
| **Security Model** *(this repo)* | Will this security be used in a pump-and-dump scheme? | Implemented |
| **Customer Model** *(not included)* | Is this customer's account at risk of being taken over for fraudulent trading? | Requires proprietary data — not replicated |

---

## Data

### On labelling

No formal public body authoritatively labels a security as having been used in a pump-and-dump scheme. In the production system, the fraud operations team provided confirmed labels from internal investigation records — clean, timely, business-verified ground truth. For this portfolio version, SEC enforcement announcements and company 8-K disclosures are used as proxy labels: reasonable approximations, but not equivalent to internal labels.

### True Positives — Confirmed Cases

Sourced from SEC EDGAR via enforcement announcements and 8-K filings in which companies disclosed coordinated manipulation of their shares. Each confirmed case provides a ticker and a reference fraud date (D date) that anchors the observation window.

### Price and Volume Data (OHLCV)

Daily Open, High, Low, Close, Volume data pulled via `yfinance` for each confirmed case over its observation window. OHLCV is the only publicly available price data — true order flow (buy vs. sell volume splits) requires paid tick-level data and is not available here. Order flow *proxies* derived from OHLCV are used instead (see Feature Engineering).

### Synthetic Data for Demonstration

Sourcing and downloading real true negative data requires significant API time (~40–50 minutes for 500 candidates across 16 date windows). To allow the full pipeline to run end-to-end in seconds, `scripts/generate_demo_data.py` generates synthetic OHLCV series:

- **TP series**: pump-and-dump pattern injected — quiet baseline → gradual price build-up + volume surge → sharp dump
- **TN series**: geometric random walks with no directional pattern

The synthetic data follows the same schema and file structure as real downloaded data. All modelling code runs identically on either. **The synthetic data is for demonstration only; the methodology and pipeline are designed for and validated against real securities data.**

---

## Methodology

Each step below addresses a specific problem in building a fraud detection model on rare, imbalanced, and noisy data.

### Step 1 — Observation Window Construction

**Problem:** The model must fire before the fraud occurs, not during it. Including data from the days immediately preceding the fraud event risks training on the most obvious manipulation signal, which may not be observable until it is too late to act.

**Approach:** Features are built over a **D-90 to D-5** window — 90 trading days of price and volume behaviour ending 5 days before the fraud reference date. The 5-day buffer enforces the operational lead time requirement at training time, preventing the model from learning signals it would not have access to in production.

---

### Step 2 — Feature Engineering

**Problem:** Raw OHLCV price levels are uninformative across securities with different price scales, and a single point-in-time snapshot misses the temporal dynamics that characterise manipulation (sustained build-up, not just a high price).

**Approach:** Each security's 90-day OHLCV series is transformed into a flat feature vector of statistical summaries:

| Feature group | What it captures |
|---|---|
| N-day price changes (1/5/10/15/30/45/90d) | Rate of price movement at multiple horizons |
| Rolling mean, std, positive-day count | Trend consistency and volatility over the window |
| Intraday metrics (range, close-minus-open, etc.) | Daily buy/sell pressure within each session |
| Volume changes and rolling stats | Accumulation and liquidity build-up |
| Order flow proxies (CLV, OBV, CMF, MFI) | Approximations of buy vs. sell pressure from OHLCV |
| Security metadata (exchange, market cap, country) | Structural context |

All transformations are encapsulated in `FeatureTransformer` — one call converts raw OHLCV into a model-ready row.

---

### Step 3 — Centroid-Based True Negative Selection (Model 1)

**Problem:** With severe class imbalance, randomly selecting true negatives gives the classifier an easy job — a large-cap S&P 500 stock will trivially separate from a micro-cap OTC penny stock on market cap alone. The model learns nothing useful about what distinguishes manipulated securities from similar, non-manipulated ones.

**Approach:** Each confirmed TP security acts as a centroid. TN candidates are assigned to their nearest TP centroid by Euclidean distance in robust-scaled feature space. The classifier is then trained on TPs and the closest TNs — securities that *look similar to the TP* but were not manipulated. Three downsampling variants are compared:

- **Random** — baseline, no assumption about match quality
- **Distance-ranked** — selects the N closest TNs per centroid (hardest cases)
- **Stratified-by-distance** — samples from near and far bins (balances hard and easy cases)

Features are robust-scaled (median/IQR) for centroid distance computation only. The Decision Tree receives unscaled features to preserve interpretability.

---

### Step 4 — Anomaly Scoring (Isolation Forest)

**Problem:** The TP set is small, limiting the Decision Tree's ability to learn what "unusual" looks like from labelled examples alone.

**Approach:** An Isolation Forest is trained on the TN candidate pool as a baseline of "normal" security behaviour. Its anomaly score — how isolated a security is relative to the normal distribution — is appended as a single additional feature. This adds an unsupervised signal about behavioural unusualness without replacing the interpretable classifier.

---

### Step 5 — Decision Tree Classification (Model 2)

**Problem:** A black-box model (gradient boosting, neural network) would likely outperform a decision tree on a larger dataset, but in a fraud context the model output must be explainable. Fraud reviewers need to understand exactly which conditions triggered a flag to make an informed block decision and to defend it to customers or regulators.

**Approach:** A shallow Decision Tree (max depth 4) classifies each security as fraud risk or not. The tree is intentionally depth-limited to keep rules readable. Feature importances and the full tree structure are exposed in the notebook.

---

### Step 6 — Leakage-Safe Leave-One-Out Cross-Validation

**Problem:** With a small TP set, standard k-fold CV wastes training signal. More critically, naively splitting TN candidates independent of which TP they were matched to creates temporal leakage: a TN evaluated in TP_i's 2014 market window could appear as a training example when TP_i is the test case, allowing the model to implicitly learn the conditions of that specific period.

**Approach:** LOO-CV holds out one TP per fold. Crucially, all TN candidates tagged with the held-out TP's D-date are *also* excluded from the training fold — they form the negative test set alongside the held-out TP. CentroidSelector and IsolationForest are re-fit from scratch in each fold using only training-fold data. This prevents both selection bias leakage and temporal leakage.

---

### Step 7 — Threshold Optimisation

**Problem:** Default 0.5 classification thresholds ignore the asymmetric cost of errors in a fraud context. A missed fraud case (false negative) causes direct financial loss; a false alarm (false positive) blocks a legitimate trade and harms customer experience. These costs are not equal.

**Approach:** The classification threshold is grid-searched to minimise `FP × fp_cost + FN × fn_cost`, where the cost ratio reflects the relative business tolerance for each error type. The cost-optimal threshold from out-of-fold predictions is stored in the trained pipeline and used at inference time.

---

## Results

Results are produced by running the notebook end-to-end on synthetic data. See `notebooks/pump_and_dump_detection.ipynb` sections 11–13 for:

- OOF score distributions (TP vs. TN separation)
- Cost curve and precision/recall vs. threshold plots
- Feature importance ranking and decision tree rules

*Quantitative metrics on real data to be populated once real TN OHLCV is sourced.*

---

## Running the Project

```bash
git clone <repo>
cd stock_price_manipulation
pip install -r requirements.txt

# Generate synthetic demo data (runs in ~5 seconds)
python scripts/generate_demo_data.py

# Open the notebook — run all cells top to bottom
jupyter notebook notebooks/pump_and_dump_detection.ipynb
```

To run with real data instead of synthetic:
```bash
# Download real TN candidates from public listings + yfinance (~40-50 min)
python scripts/download_data.py --source tn
```

---

## Skills & Tools

`Python` · `scikit-learn` · `pandas` · `numpy` · `yfinance` · `SEC EDGAR` · `Jupyter` · `Decision Tree` · `Isolation Forest` · `Centroid Clustering` · `Leave-One-Out CV` · `Machine Learning` · `Fraud Detection` · `Financial Services`
