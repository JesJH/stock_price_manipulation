# Parking Lot

Ideas, data limitations, and deferred decisions that don't belong in the main codebase yet but are worth tracking.

---

## Data Limitations & Workarounds

### Order Flow (Buy vs. Sell Volume Split)
**What we wanted:** True buy-side vs. sell-side volume per day. In a pump-and-dump scheme, the pump phase is characterised by sustained net buy-side pressure (buyers consistently overwhelming sellers, closing price near the daily high). The dump shows as a sudden reversal — heavy sell volume, price closing near the daily low.

**What's publicly available:** `yfinance` and most free market data APIs only provide **total daily volume** (OHLCV) — they do not split volume into buy-initiated vs. sell-initiated trades. True order flow data requires tick-level or trade-classified data from paid providers (Polygon.io, Refinitiv, Bloomberg TAQ).

**What was used in production:** The original brokerage system had access to actual transaction-level data showing buy vs. sell volumes at the order level — a materially stronger signal than what is available here.

**What we do instead:** Derive order flow *proxies* from OHLCV using standard technical analysis measures that approximate buy vs. sell pressure based on where the close lands relative to the day's high/low range:

| Proxy | Formula / Logic | Captures |
|---|---|---|
| Close Location Value (CLV) | `(2×close − high − low) / (high − low)` | Day-level buy/sell balance; +1 = closed at high (buyers), −1 = closed at low (sellers) |
| Accumulation/Distribution Line | Cumulative sum of `CLV × volume` | Sustained accumulation vs. distribution trend across the window |
| Chaikin Money Flow (CMF) | `sum(CLV × volume) / sum(volume)` over N days | Net money flow direction over a window |
| On-Balance Volume (OBV) | Adds full day volume on up days, subtracts on down days | Directional volume trend; rising = consistent buy pressure |
| Money Flow Index (MFI) | RSI-style oscillator using `typical_price × volume` | Overbought/oversold via volume-weighted momentum |

These proxies are derived quantities, not observed data. They should be treated as approximations — useful signal, but not a substitute for true order flow.

**Status:** Not yet implemented. To add as `_compute_order_flow_features()` in `src/features/feature_transformer.py`.

---

## Feature Ideas (Not Yet Implemented)

### Consecutive Streak Flags
Count-based and streak-based features that capture the *consecutive* nature of coordinated buying — a pattern the rolling `pos_count` features miss:
- `max_consec_up_close` — longest run of days where close > prev close
- `max_consec_up_volume` — longest run of days where volume increased day-over-day
- `current_consec_up_close` — streak length at end of the observation window
- `n_days_close_above_roll20_mean` — count of days price was elevated above rolling average
- `n_days_volume_above_2x_mean` — count of volume spike days
- `n_large_up_days` — count of days with daily return > threshold (e.g. 3%)
- `n_large_intraday_reversals` — days where (high − close) / close exceeded a threshold (sell-into-strength pattern)

Note: with ~20 TPs, keep the threshold list short and motivated by the P&D mechanism to avoid overfitting.

**Status:** Discussed, not implemented. Add as `_compute_flag_features()` in `FeatureTransformer`.

---

## Deferred Modelling Decisions

### Observation Window Length
The 90-day lookback was a business starting point, not empirically validated. A proper comparison across windows (30, 45, 60, 90, 120 days) against held-out classifier performance requires TN data first. Deferred until TN OHLCV is available.

### FP/FN Cost Ratio
Threshold optimisation requires a relative cost weight for false positives vs. false negatives. Not defined yet. Placeholder: 1 FN = 5× the cost of 1 FP. Revisit with business context.

---

## Out of Scope for Portfolio

### Customer Model
The second model — predicting which customers are likely to have their accounts taken over — requires proprietary customer data (demographics, transaction history, account tenure, held positions). Not replicable publicly. Documented in README under Two-Model Architecture.

### Channel-Based Features
ATO fraud flows predominantly through digital channels (online/mobile), not phone. Adding transaction channel as a feature could meaningfully improve precision. Requires internal transaction metadata — not publicly available.
