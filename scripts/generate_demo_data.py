#!/usr/bin/env python3
"""
Generate synthetic demo data for the pump-and-dump detection pipeline.

Creates realistic-looking synthetic OHLCV series so the full pipeline runs
end-to-end without downloading real market data.

  TP securities : pump-and-dump signature injected
                  (quiet accumulation → price build-up + volume surge → dump)
  TN securities : random walks with no directional pattern

Each TN is evaluated once per TP D-date, producing one OHLCV window per
(TN_ticker, TP_D_date) pair — matching the temporal structure that the real
downloader produces and enabling leakage-safe LOO-CV.

Output
------
  data/raw/tp_ohlcv.parquet
  data/raw/tp_metadata.csv
  data/raw/tn_ohlcv.parquet
  data/raw/tn_metadata.csv
  data/external/tn_candidates.csv

Usage
-----
    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --n-tn 50   # smaller run for testing
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR      = Path("data/raw")
EXTERNAL_DIR = Path("data/external")
LOOKBACK_DAYS = 90
BUFFER_DAYS   = 5


# ------------------------------------------------------------------
# OHLCV construction helpers
# ------------------------------------------------------------------

def _make_ohlcv(
    dates: pd.DatetimeIndex,
    closes: np.ndarray,
    volumes: np.ndarray,
    ticker: str,
    d_date,
    label: int,
) -> pd.DataFrame:
    """Derive open / high / low from a close series and return a labelled DataFrame."""
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    n   = len(closes)

    opens         = np.empty(n)
    opens[0]      = closes[0] * (1 + rng.normal(0, 0.004))
    opens[1:]     = closes[:-1] * (1 + rng.normal(0, 0.004, n - 1))

    intraday_move = np.abs(rng.normal(0, 0.008, n))
    highs = np.maximum(closes, opens) * (1 + intraday_move)
    lows  = np.minimum(closes, opens) * (1 - intraday_move)

    return pd.DataFrame({
        "date":   dates,
        "ticker": ticker,
        "d_date": pd.Timestamp(d_date),
        "open":   np.clip(opens,  0.01, None),
        "high":   np.clip(highs,  0.01, None),
        "low":    np.clip(lows,   0.01, None),
        "close":  np.clip(closes, 0.01, None),
        "volume": np.clip(volumes.astype(int), 1, None),
        "label":  label,
    })


def generate_tp_ohlcv(
    ticker: str,
    d_date,
    lookback_days: int = LOOKBACK_DAYS,
    buffer_days: int   = BUFFER_DAYS,
) -> pd.DataFrame:
    """
    Pump-and-dump OHLCV series.

    Four phases over the observation window:
      0–44% of window  : quiet baseline — slight noise, normal volume
      45–71%           : pump building — upward drift, volume 2–5×
      72–87%           : peak pump — steep climb, volume 6–12×
      88–100%          : dump — sharp fall, heavy volume (the exit)
    """
    rng  = np.random.default_rng(abs(hash(ticker)) % (2**32))
    end  = pd.Timestamp(d_date) - pd.Timedelta(days=buffer_days)
    dates = pd.bdate_range(end=end, periods=lookback_days)
    n     = len(dates)

    base_price  = rng.uniform(0.5, 3.0)
    base_volume = rng.integers(50_000, 300_000)

    closes  = np.empty(n)
    volumes = np.empty(n)
    closes[0]  = base_price
    volumes[0] = base_volume

    for i in range(1, n):
        frac = i / n

        if frac < 0.45:
            drift, sigma = 0.001, 0.015
            vol_mult = rng.uniform(0.8, 1.3)
        elif frac < 0.72:
            drift, sigma = 0.007, 0.018
            vol_mult = rng.uniform(2.0, 5.0)
        elif frac < 0.88:
            drift, sigma = 0.013, 0.022
            vol_mult = rng.uniform(6.0, 12.0)
        else:
            drift, sigma = -0.038, 0.030
            vol_mult = rng.uniform(8.0, 18.0)

        ret       = rng.normal(drift, sigma)
        closes[i] = closes[i - 1] * (1 + ret)
        volumes[i] = int(base_volume * vol_mult * np.exp(rng.normal(0, 0.3)))

    return _make_ohlcv(dates, closes, volumes, ticker, d_date, label=1)


def generate_tn_ohlcv(
    ticker: str,
    d_date,
    lookback_days: int = LOOKBACK_DAYS,
    buffer_days: int   = BUFFER_DAYS,
) -> pd.DataFrame:
    """Normal security: geometric random walk, no directional pattern."""
    rng  = np.random.default_rng(abs(hash(ticker + str(d_date))) % (2**32))
    end  = pd.Timestamp(d_date) - pd.Timedelta(days=buffer_days)
    dates = pd.bdate_range(end=end, periods=lookback_days)
    n     = len(dates)

    base_price  = rng.uniform(0.5, 10.0)
    base_volume = rng.integers(10_000, 200_000)

    rets    = rng.normal(0.0002, 0.018, n)
    closes  = base_price * np.cumprod(1 + rets)
    volumes = (base_volume * np.exp(rng.normal(0, 0.4, n))).astype(int)

    return _make_ohlcv(dates, closes, volumes, ticker, d_date, label=0)


def generate_metadata(ticker: str, label: int) -> dict:
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    return {
        "ticker":     ticker,
        "market_cap": int(rng.uniform(1e6, 50e6)),
        "country":    "United States",
        "currency":   "USD",
        "exchange":   rng.choice(["OTC", "PINK", "OTCBB"], p=[0.5, 0.3, 0.2]),
        "sector":     rng.choice(
            ["Technology", "Healthcare", "Energy", "Finance", "Other"],
            p=[0.30, 0.25, 0.20, 0.15, 0.10],
        ),
        "industry":   "Small Cap",
        "quote_type": "EQUITY",
        "short_name": ticker,
        "label":      label,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo data")
    parser.add_argument(
        "--n-tn", type=int, default=150,
        help="Number of synthetic TN candidates (default 150)",
    )
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--buffer",   type=int, default=BUFFER_DAYS)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    tp_csv = EXTERNAL_DIR / "tp_tickers.csv"
    if not tp_csv.exists():
        print(f"ERROR: {tp_csv} not found — this file is tracked in the repo.")
        sys.exit(1)

    cases_df = pd.read_csv(tp_csv, parse_dates=["d_date"])
    print(f"Loaded {len(cases_df)} TP cases from {tp_csv}")

    # --- TP OHLCV ---
    print(f"\nGenerating TP OHLCV ({len(cases_df)} synthetic series)...")
    tp_ohlcv_parts: list[pd.DataFrame] = []
    tp_meta_rows:   list[dict]         = []

    for _, row in cases_df.iterrows():
        ohlcv = generate_tp_ohlcv(row["ticker"], row["d_date"], args.lookback, args.buffer)
        tp_ohlcv_parts.append(ohlcv)
        meta = generate_metadata(row["ticker"], label=1)
        meta["d_date"] = pd.Timestamp(row["d_date"])
        tp_meta_rows.append(meta)

    tp_ohlcv_df = pd.concat(tp_ohlcv_parts, ignore_index=True)
    tp_meta_df  = pd.DataFrame(tp_meta_rows)

    tp_ohlcv_df.to_parquet(RAW_DIR / "tp_ohlcv.parquet", index=False)
    tp_meta_df.to_csv(RAW_DIR / "tp_metadata.csv", index=False)
    print(f"  TP OHLCV  : {len(tp_ohlcv_df):,} rows  → data/raw/tp_ohlcv.parquet")
    print(f"  TP metadata: {len(tp_meta_df)} rows  → data/raw/tp_metadata.csv")

    # --- TN candidates list ---
    n_tn       = args.n_tn
    tn_tickers = [f"DEMO{i:03d}" for i in range(1, n_tn + 1)]
    tn_candidates = pd.DataFrame({
        "ticker":        tn_tickers,
        "security_name": tn_tickers,
        "source":        "synthetic",
    })
    tn_candidates.to_csv(EXTERNAL_DIR / "tn_candidates.csv", index=False)
    print(f"\nTN candidates: {n_tn} synthetic tickers → data/external/tn_candidates.csv")

    # --- TN OHLCV (one window per TP D-date) ---
    tp_dates = sorted(cases_df["d_date"].unique())
    total_tn_series = n_tn * len(tp_dates)
    print(f"\nGenerating TN OHLCV ({n_tn} tickers × {len(tp_dates)} windows = {total_tn_series:,} series)...")

    tn_ohlcv_parts: list[pd.DataFrame] = []
    for d_date in tp_dates:
        for ticker in tn_tickers:
            tn_ohlcv_parts.append(
                generate_tn_ohlcv(ticker, d_date, args.lookback, args.buffer)
            )

    tn_meta_rows = [generate_metadata(t, label=0) for t in tn_tickers]

    tn_ohlcv_df = pd.concat(tn_ohlcv_parts, ignore_index=True)
    tn_meta_df  = pd.DataFrame(tn_meta_rows)

    tn_ohlcv_df.to_parquet(RAW_DIR / "tn_ohlcv.parquet", index=False)
    tn_meta_df.to_csv(RAW_DIR / "tn_metadata.csv", index=False)
    print(f"  TN OHLCV  : {len(tn_ohlcv_df):,} rows  → data/raw/tn_ohlcv.parquet")
    print(f"  TN metadata: {len(tn_meta_df)} rows  → data/raw/tn_metadata.csv")

    total = len(tp_ohlcv_df) + len(tn_ohlcv_df)
    print(f"\nTotal OHLCV rows generated : {total:,}")
    print("Done. Open notebooks/pump_and_dump_detection.ipynb and run all cells.")


if __name__ == "__main__":
    main()
