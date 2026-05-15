# Stock Price Manipulation Detection

Replaced a rule-based fraud detection system with a machine learning pipeline to identify securities likely to be used in pump-and-dump schemes — improving coverage while reducing unnecessary trading blocks.

---

## Background

A brokerage firm was experiencing an increasing number of account takeover (ATO) incidents where compromised customer accounts were used to artificially inflate stock prices — a classic pump-and-dump scheme. These unauthorized transactions resulted in direct financial losses for the brokerage and its customers.

The existing detection system relied on a simple rule-based approach using only three customer-level signals: account age, account value, and trade frequency over a fixed window. While interpretable, this approach lacked the breadth to reliably identify the securities being targeted, leaving significant fraud exposure unaddressed.

---

## Business Goal

Replace the legacy rule-based system with a model-driven approach that flags and blocks securities most likely to be exploited for pump-and-dump manipulation — expanding feature coverage beyond customer demographics to include security-level signals, and making blocking decisions before fraudulent trades are executed.

---

## Data

- **Fraud cases:** ~20 confirmed account takeover incidents over a 5-month period, with known securities used for price manipulation
- **Security features:** currency, country of origination, price metrics, date of origination, market capitalization, listed index, trading volume
- **Customer features:** demographics and transaction history

---

## Challenges

**Small labeled dataset** — With only ~20 confirmed true positives, standard supervised learning approaches risk overfitting. Dataset construction and feature engineering required careful design to extract signal from limited examples.

**Class imbalance** — Fraudulent securities represent a tiny fraction of all tradeable equities, requiring deliberate sampling strategies to handle severe imbalance without discarding meaningful negatives.

**Noisy price signals** — EDA of security prices in the observation window showed no clear directional trends leading up to fraud events, making raw price series unreliable as direct inputs.

**Balancing false positives and false negatives** — The cost function is asymmetric:
- False Positives (FP): legitimate securities get blocked, harming customer experience and revenue
- False Negatives (FN): fraudulent activity goes undetected, resulting in financial and reputational loss

---

## Approach

### Feature Engineering
Features were built over a **D-90 to D-5 observation window** — 90 days of security behavior leading up to the fraud date, with a mandatory 5-day buffer before D. The buffer is a production constraint: the model must surface risky securities early enough for the fraud team to review and place trading blocks before fraudulent transactions occur. Training on data right up to D would create leakage that doesn't hold in deployment.

Three categories of features were constructed:

**Price behavior** — Because raw price trends showed no consistent pattern, features focused on statistical behavior across the window: day-over-day changes at 1, 3, 5, and 10-day intervals, plus rolling counts, sums, and averages across closing price, open price, intraday high, and intraday low.

**Trading volume** — Volume and liquidity can signal early manipulation activity. Sustained buy-side volume pressure drives prices up during the pump phase; a spike in net sell-side volume marks the dump. Features were engineered to capture these dynamics across the observation window.

**Security metadata and customer behavior** — Static security attributes (market cap, country of origination, currency, listed index) were one-hot encoded. Customer transaction patterns were included to capture behavioral signals around the time of the fraud.

All transformations were encapsulated in a reusable feature transformation class to keep the pipeline modular and reproducible.

### Modeling

The solution required two models working in sequence:

**Model 1 — K-Means Clustering (negative selection)**
To ensure true negatives were meaningfully comparable to confirmed fraud cases, K-Means was used to cluster securities on their D-90 to D-5 behavior. True negatives were selected from the same clusters as the true positives — meaning the classification model would need to learn genuine behavioral differences, not just distinguish unrelated securities. True negatives were downsampled within clusters to keep the training set challenging but tractable.

**Model 2 — Decision Tree (classification)**
A Decision Tree classifier was trained on the curated dataset to distinguish manipulated securities from non-manipulated ones. The 80/20 split (16 TPs for training, 4 for testing) was used throughout. Decision trees were favored for their interpretability, which was important for explaining flagged securities to the fraud review team.

**Additional experimentation** — An Isolation Forest model was also evaluated as an alternative approach, leveraging its strength in detecting anomalous securities without relying on labeled negatives.

### Threshold Optimization
Classification thresholds were tuned explicitly around the FP/FN cost tradeoff rather than raw accuracy. Blocking a legitimate security (FP) carries a direct revenue and customer experience cost; missing a fraudulent one (FN) carries financial and reputational risk. The final threshold reflects the business's relative tolerance for each error type.

---

## Next Steps

- **Channel-based features:** ATO fraud predominantly flows through digital platforms (online/mobile), not phone. Adding transaction channel as a feature could meaningfully improve precision without broadening the block list.
- **Newly identified TP integration:** As fraud operations surface new confirmed cases, a retraining pipeline is needed to incorporate them systematically — ensuring the model stays current as fraudster tactics evolve.
- **Expand labeled data:** Collaborate with fraud operations to surface additional confirmed cases and improve model generalization.
- **Monitoring and feedback loop:** Track blocked securities post-deployment to measure real-world effectiveness and feed newly identified patterns back into training.

---

## Skills & Tools

`Python` · `scikit-learn` · `pandas` · `imbalanced-learn` · `Jupyter` · `K-Means Clustering` · `Decision Tree` · `Isolation Forest` · `Machine Learning` · `Fraud Detection` · `Financial Services`
