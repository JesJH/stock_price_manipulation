from typing import Optional

import numpy as np
import pandas as pd

from src.features.feature_config import (
    CHANGE_WINDOWS,
    MARKET_CAP_BINS,
    MARKET_CAP_LABELS,
    ORDER_FLOW_WINDOWS,
    PRICE_COLS,
    ROLLING_WINDOWS,
)


class FeatureTransformer:
    """
    Transforms raw OHLCV time series into a flat feature vector per security.

    Input
    -----
    Long-format OHLCV DataFrame — one row per trading day per ticker,
    covering the observation window [D-lookback, D-buffer].
    Required columns: date, ticker, d_date, open, high, low, close, volume

    Output
    ------
    Wide-format feature matrix — one row per (ticker, d_date) pair,
    with all engineered features as columns.

    Feature groups
    --------------
    1. N-day price changes  — close, open, high, low, overnight gap
       at horizons: 1, 5, 10, 15, 30, 45, 90 trading days
    2. Rolling statistics   — mean, std, positive-day count of daily % changes
       over windows: 5, 10, 15, 30, 45, 90 days (same price series + gap)
    3. Intraday metrics     — rolling mean/std of:
         intraday_range      = (high - low) / close
         close_minus_open    = (close - open) / open
         high_minus_close    = (high - close) / close
         close_minus_low     = (close - low) / close
    4. Volume features      — N-day changes (raw + log-scaled) + rolling stats
    5. Order flow proxies   — OHLCV-derived buy/sell pressure approximations:
         CLV (Close Location Value), OBV, A/D Line changes, CMF, MFI
         See PARKING_LOT.md for discussion of limitations vs. true order flow.
    6. Metadata encoding    — one-hot: country, currency, exchange; binned market cap

    Scaling note
    ------------
    Features are returned UNSCALED. The Decision Tree classifier receives them
    directly to preserve interpretability. Robust scaling (median/IQR) is
    applied only inside CentroidSelector, which needs scale-normalised distances.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self,
        ohlcv_df: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None,
        label_col: str = "label",
    ) -> pd.DataFrame:
        """
        Compute the feature matrix for all tickers in ohlcv_df.

        Parameters
        ----------
        ohlcv_df    : long-format OHLCV with one row per (ticker, date)
        metadata_df : optional wide-format metadata, one row per ticker
        label_col   : name of target column to carry through (if present)

        Returns
        -------
        One row per (ticker, d_date) with all features + label column
        """
        records = []

        for (ticker, d_date), grp in ohlcv_df.groupby(["ticker", "d_date"]):
            grp = grp.sort_values("date").reset_index(drop=True)

            if len(grp) < 5:
                continue

            features = self._compute_price_volume_features(grp)
            features.update(self._compute_order_flow_features(grp))
            features["ticker"] = ticker
            features["d_date"] = pd.Timestamp(d_date)

            if label_col in grp.columns:
                features[label_col] = int(grp[label_col].iloc[0])

            records.append(features)

        feature_df = pd.DataFrame(records)

        if metadata_df is not None and not metadata_df.empty:
            meta_encoded = self._encode_metadata(metadata_df)
            feature_df = feature_df.merge(meta_encoded, on="ticker", how="left")

        return feature_df

    @property
    def feature_names(self) -> list[str]:
        """All engineered feature column names, excluding ticker / d_date / label."""
        names: list[str] = []

        # --- Price + gap: N-day changes ---
        for col in PRICE_COLS + ["gap"]:
            for n in CHANGE_WINDOWS:
                names.append(f"{col}_chg_{n}d")

        # --- Price + gap: rolling stats of daily % change ---
        for col in PRICE_COLS + ["gap"]:
            for r in ROLLING_WINDOWS:
                names += [
                    f"{col}_roll{r}_mean",
                    f"{col}_roll{r}_std",
                    f"{col}_roll{r}_pos_count",
                ]

        # --- Intraday metrics: rolling mean + std ---
        for col in ["intraday_range", "close_minus_open", "high_minus_close", "close_minus_low"]:
            for r in ROLLING_WINDOWS:
                names += [f"{col}_roll{r}_mean", f"{col}_roll{r}_std"]

        # --- Volume: N-day changes ---
        for n in CHANGE_WINDOWS:
            names += [f"volume_chg_{n}d", f"volume_log_chg_{n}d"]

        # --- Volume: rolling stats ---
        for r in ROLLING_WINDOWS:
            names += [
                f"volume_roll{r}_mean",
                f"volume_roll{r}_std",
                f"volume_roll{r}_sum",
                f"volume_roll{r}_pos_count",
            ]

        # --- Order flow proxies ---
        # CLV rolling stats
        for r in ROLLING_WINDOWS:
            names += [f"clv_roll{r}_mean", f"clv_roll{r}_std", f"clv_roll{r}_pos_count"]

        # OBV and A/D Line N-day changes
        for n in CHANGE_WINDOWS:
            names += [f"obv_chg_{n}d", f"ad_chg_{n}d"]

        # CMF and MFI at fixed windows
        for r in ORDER_FLOW_WINDOWS:
            names += [f"cmf_{r}d", f"mfi_{r}d"]

        return names

    # ------------------------------------------------------------------
    # Internal: time-series features
    # ------------------------------------------------------------------

    def _compute_price_volume_features(self, df: pd.DataFrame) -> dict:
        """Compute all numerical features for a single ticker's sorted OHLCV."""
        features: dict = {}
        df = df.copy()

        # ------ Derived daily series ------

        # Overnight gap: today's open vs. yesterday's close
        # Captures pre-market information flow between sessions
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # Intraday metrics (each already a relative measure — no pct_change needed)
        df["intraday_range"]   = (df["high"] - df["low"])   / df["close"]
        df["close_minus_open"] = (df["close"] - df["open"]) / df["open"]
        df["high_minus_close"] = (df["high"] - df["close"]) / df["close"]
        df["close_minus_low"]  = (df["close"] - df["low"])  / df["close"]

        # ------ Price + gap features ------

        for col in PRICE_COLS + ["gap"]:
            series    = df[col]
            daily_pct = series.pct_change(1)

            # Point-to-point N-day changes (last row vs. N rows earlier)
            for n in CHANGE_WINDOWS:
                chg = series.pct_change(n)
                features[f"{col}_chg_{n}d"] = chg.iloc[-1] if len(chg) > n else np.nan

            # Rolling statistics of daily % changes, snapshotted at the last day
            for r in ROLLING_WINDOWS:
                min_p = max(2, r // 3)
                roll  = daily_pct.rolling(r, min_periods=min_p)
                features[f"{col}_roll{r}_mean"]      = roll.mean().iloc[-1]
                features[f"{col}_roll{r}_std"]       = roll.std().iloc[-1]
                features[f"{col}_roll{r}_pos_count"] = (
                    (daily_pct > 0).rolling(r, min_periods=1).sum().iloc[-1]
                )

        # ------ Intraday metrics: rolling stats only ------

        for col in ["intraday_range", "close_minus_open", "high_minus_close", "close_minus_low"]:
            series = df[col]
            for r in ROLLING_WINDOWS:
                min_p = max(2, r // 3)
                roll  = series.rolling(r, min_periods=min_p)
                features[f"{col}_roll{r}_mean"] = roll.mean().iloc[-1]
                features[f"{col}_roll{r}_std"]  = roll.std().iloc[-1]

        # ------ Volume features ------

        vol     = df["volume"].replace(0, np.nan)
        vol_log = np.log1p(vol)

        for n in CHANGE_WINDOWS:
            features[f"volume_chg_{n}d"]     = vol.pct_change(n).iloc[-1]     if len(vol) > n     else np.nan
            features[f"volume_log_chg_{n}d"] = vol_log.pct_change(n).iloc[-1] if len(vol_log) > n else np.nan

        vol_daily_pct = vol.pct_change(1)
        for r in ROLLING_WINDOWS:
            min_p    = max(2, r // 3)
            roll_pct = vol_daily_pct.rolling(r, min_periods=min_p)
            features[f"volume_roll{r}_mean"]      = roll_pct.mean().iloc[-1]
            features[f"volume_roll{r}_std"]       = roll_pct.std().iloc[-1]
            features[f"volume_roll{r}_sum"]       = vol.rolling(r, min_periods=1).sum().iloc[-1]
            features[f"volume_roll{r}_pos_count"] = (
                (vol_daily_pct > 0).rolling(r, min_periods=1).sum().iloc[-1]
            )

        return features

    def _compute_order_flow_features(self, df: pd.DataFrame) -> dict:
        """
        OHLCV-derived proxies for buy/sell pressure.

        True order flow (buy vs. sell volume) is not publicly available —
        these are approximations based on where the close lands within the
        day's high/low range. See PARKING_LOT.md for limitations.
        """
        features: dict = {}
        df = df.copy()

        hl_range = (df["high"] - df["low"]).replace(0, np.nan)

        # Close Location Value: +1 = closed at high (buyers), -1 = closed at low (sellers)
        clv = (2 * df["close"] - df["high"] - df["low"]) / hl_range

        for r in ROLLING_WINDOWS:
            min_p = max(2, r // 3)
            roll  = clv.rolling(r, min_periods=min_p)
            features[f"clv_roll{r}_mean"]      = roll.mean().iloc[-1]
            features[f"clv_roll{r}_std"]        = roll.std().iloc[-1]
            features[f"clv_roll{r}_pos_count"]  = (
                (clv > 0).rolling(r, min_periods=1).sum().iloc[-1]
            )

        # On-Balance Volume: adds volume on up-days, subtracts on down-days
        price_dir = np.sign(df["close"].diff(1)).fillna(0)
        obv = (price_dir * df["volume"]).cumsum()
        for n in CHANGE_WINDOWS:
            features[f"obv_chg_{n}d"] = obv.diff(n).iloc[-1] if len(obv) > n else np.nan

        # Accumulation/Distribution Line: cumulative CLV-weighted volume
        ad_line = (clv.fillna(0) * df["volume"]).cumsum()
        for n in CHANGE_WINDOWS:
            features[f"ad_chg_{n}d"] = ad_line.diff(n).iloc[-1] if len(ad_line) > n else np.nan

        # Chaikin Money Flow: net CLV-weighted volume fraction over window
        clv_vol = clv.fillna(0) * df["volume"]
        for r in ORDER_FLOW_WINDOWS:
            min_p = max(2, r // 3)
            num = clv_vol.rolling(r, min_periods=min_p).sum()
            den = df["volume"].rolling(r, min_periods=min_p).sum().replace(0, np.nan)
            features[f"cmf_{r}d"] = (num / den).iloc[-1]

        # Money Flow Index: volume-weighted RSI-style oscillator (0–100)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        raw_mf = typical_price * df["volume"]
        tp_diff = typical_price.diff(1)
        for r in ORDER_FLOW_WINDOWS:
            min_p = max(2, r // 3)
            pos_mf = raw_mf.where(tp_diff > 0, 0.0).rolling(r, min_periods=min_p).sum()
            neg_mf = raw_mf.where(tp_diff <= 0, 0.0).rolling(r, min_periods=min_p).sum()
            mfi = 100.0 - (100.0 / (1.0 + pos_mf / neg_mf.replace(0, np.nan)))
            features[f"mfi_{r}d"] = mfi.iloc[-1]

        return features

    # ------------------------------------------------------------------
    # Internal: metadata encoding
    # ------------------------------------------------------------------

    def _encode_metadata(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical security attributes and bin market cap.
        Returns a DataFrame keyed on 'ticker'.
        """
        df = metadata_df[["ticker"]].copy()

        if "market_cap" in metadata_df.columns:
            df["market_cap_tier"] = pd.cut(
                metadata_df["market_cap"].fillna(0),
                bins=MARKET_CAP_BINS,
                labels=MARKET_CAP_LABELS,
                right=False,
            ).astype(str)
            df = pd.get_dummies(df, columns=["market_cap_tier"], prefix="mktcap")

        for col in ["country", "currency", "exchange"]:
            if col in metadata_df.columns:
                df[col] = metadata_df[col].fillna("unknown")
                df = pd.get_dummies(df, columns=[col], prefix=col)

        return df
