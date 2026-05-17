# Stock Price Manipulation Detection

Replaced a rule-based fraud detection system with a machine learning pipeline to identify securities likely to be used in pump-and-dump schemes — improving coverage while reducing unnecessary trading blocks.

---

## Background

A brokerage firm was experiencing an increasing number of account takeover (ATO) incidents in which compromised customer accounts were used to artificially inflate the price of a target security. Once the price was elevated, the fraudster — who held a pre-existing position in that security in their own account — would sell their shares at the inflated price, profiting at the expense of the brokerage and its customers. These pump-and-dump schemes resulted in direct financial losses with no systematic way to reliably detect the targeted securities in advance.

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

#### Model 1 — Centroid-based True Negative Selection
Since we are working with imbalanced data, we needed some ways to downsample the true negative dataset to ensure the model doesn't predict all securities as true negative.

To ensure true negatives were meaningfully comparable to confirmed fraud cases, each of the 20 TP securities was treated as its own centroid. All candidate TN securities were assigned to their nearest TP centroid based on D-90 to D-5 behavioral characteristics, so the classifier would need to learn genuine behavioral differences rather than simply distinguish unrelated securities.

Three downsampling methods were evaluated:

- **Random sampling** — 
for each TP centroid, N TNs are drawn randomly from the assigned pool. This serves as a baseline. Hyperparameters: TN/TP ratio. It makes no assumption about match quality, which risks including TNs that are trivially different from the TP and inflate apparent classifier performance.

- **Distance-ranked sampling** — 
TNs are ranked by Euclidean distance to their assigned TP centroid in D-90 to D-5 feature space, and the N closest are selected. Hyperparameters: N per centroid, an optional maximum distance cutoff. This optimizes for match quality. The selected TNs are the most behaviorally similar to the TP, forcing the classifier to learn subtle distinctions of securities that are similar in nature, rather than broad ones.

- **Stratified-by-distance sampling** — 
TNs are assigned to equal-count quantile bins (e.g., quartiles, quintiles, or deciles) based on their distance to the TP centroid, then sampled from each bin. Using quantile bins rather than named bands ensures each bin contains a roughly equal number of TNs regardless of how distances distribute within a cluster. Hyperparameters: number of bins Q, per-bin sampling weight (uniform or closer-weighted), and total ratio. This optimizes for training diversity — exposing the classifier to both near-boundary hard cases and clearly distinct easy cases, which typically produces a more robust decision boundary.

    Because Euclidean distance is sensitive to feature scale, all features must be normalized before distance computation. Robust scaling (median / IQR) is preferred over z-score given the likelihood of outliers in price and volume features. The scaler must be fit on the training fold only within each CV iteration to prevent test data from leaking into the distance calculation.

Model 1 has no intrinsic performance metric. The optimal configuration — method, ratio, and distance cutoff — is selected via a grid search evaluated on Model 2's held-out performance. Because Model 1's output is Model 2's training data, cross-validation must wrap the full pipeline: for each fold, Model 1 re-runs selection on the training portion before Model 2 is trained and evaluated. Running CV on Model 2 alone would leak selection bias. Given only 20 TPs, leave-one-out cross-validation is used to maximize the training signal at each fold.


One key thing to note is that since clustering methods are sensitive to data scale, the transformed feature values are scaled prior to being used for clustering.
Decision trees are insensitive to scale and using scaled values would make the decision tree output difficult to interpret directly. Therefore features were not scaled for decision tree modeling. 

#### Model 2 — Decision Tree (Classification)
A Decision Tree classifier was trained on the curated dataset to distinguish manipulated securities from non-manipulated ones. Decision trees were the required model type because the business needed a transparent, rule-based classification process — fraud reviewers must be able to trace exactly which behavioral conditions triggered a flag, which a black-box model cannot provide. The 80/20 split (16 TPs for training, 4 for testing) was used throughout.

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
