import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


class MarketDataDownloader:
    """
    Downloads OHLCV price data and security metadata via yfinance.

    For each security (TP or TN), the observation window is [D - lookback_days, D - buffer_days].
    D = fraud date for TPs; for TNs, D is inherited from the matched TP centroid.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_ohlcv(
        self,
        ticker: str,
        d_date: str | pd.Timestamp,
        lookback_days: int = 90,
        buffer_days: int = 5,
    ) -> pd.DataFrame:
        """
        Download daily OHLCV for one security over its observation window.
        Returns empty DataFrame if data is unavailable.
        """
        d = pd.Timestamp(d_date)
        # Add a margin when fetching to account for weekends/holidays,
        # then trim to exactly lookback_days of trading days after download.
        fetch_start = d - timedelta(days=lookback_days + 30)
        fetch_end = d - timedelta(days=buffer_days)

        try:
            raw = yf.download(
                ticker,
                start=fetch_start,
                end=fetch_end,
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                return pd.DataFrame()

            # Flatten multi-level columns that yfinance sometimes returns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            # Keep only the most recent lookback_days trading days
            raw = raw.iloc[-lookback_days:] if len(raw) > lookback_days else raw

            raw = raw.reset_index().rename(columns={"index": "date", "Date": "date"})
            raw.columns = raw.columns.str.lower()
            raw["ticker"] = ticker
            raw["d_date"] = d
            return raw[["date", "ticker", "d_date", "open", "high", "low", "close", "volume"]]

        except Exception as exc:
            print(f"  Warning: could not download {ticker}: {exc}")
            return pd.DataFrame()

    def download_metadata(self, ticker: str) -> dict:
        """Fetch static security attributes from yfinance."""
        try:
            info = yf.Ticker(ticker).info
            return {
                "ticker": ticker,
                "market_cap": info.get("marketCap"),
                "country": info.get("country"),
                "currency": info.get("currency"),
                "exchange": info.get("exchange"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "quote_type": info.get("quoteType"),
                "short_name": info.get("shortName"),
            }
        except Exception as exc:
            print(f"  Warning: could not fetch metadata for {ticker}: {exc}")
            return {"ticker": ticker}

    def download_batch(
        self,
        cases_df: pd.DataFrame,
        lookback_days: int = 90,
        buffer_days: int = 5,
        label: int = 1,
        delay_seconds: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Batch-download OHLCV and metadata for all rows in cases_df.

        cases_df must have columns: ticker, case_date (used as D date).
        label: 1 for true positives, 0 for true negatives.
        Returns (ohlcv_df, metadata_df).
        """
        ohlcv_parts: list[pd.DataFrame] = []
        metadata_rows: list[dict] = []
        total = len(cases_df)

        for i, (_, row) in enumerate(cases_df.iterrows(), 1):
            ticker = row["ticker"]
            d_date = row["case_date"]

            ohlcv = self.download_ohlcv(ticker, d_date, lookback_days, buffer_days)
            if not ohlcv.empty:
                ohlcv["label"] = label
                ohlcv_parts.append(ohlcv)

            meta = self.download_metadata(ticker)
            meta["label"] = label
            meta["d_date"] = pd.Timestamp(d_date)
            metadata_rows.append(meta)

            if i % 20 == 0 or i == total:
                print(f"  {i}/{total} tickers processed")
            time.sleep(delay_seconds)

        ohlcv_df = pd.concat(ohlcv_parts, ignore_index=True) if ohlcv_parts else pd.DataFrame()
        metadata_df = pd.DataFrame(metadata_rows)
        return ohlcv_df, metadata_df

    def save_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.output_dir / filename
        df.to_parquet(path, index=False)
        print(f"  Saved {len(df)} rows to {path}")
        return path

    def save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} rows to {path}")
        return path
