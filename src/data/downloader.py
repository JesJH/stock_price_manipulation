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

    def download_tn_batch(
        self,
        tn_tickers: list[str],
        tp_cases_df: pd.DataFrame,
        lookback_days: int = 90,
        buffer_days: int = 5,
        delay_seconds: float = 1.0,
        chunk_size: int = 100,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        For each TP D-date, bulk-download OHLCV for all TN candidate tickers.

        Each TN ticker is evaluated once per TP D-date, producing one feature
        row per (tn_ticker, tp_d_date) pair. Tagging TNs with the TP's D-date
        is essential for leakage-free LOO-CV: when TP_i is held out, all TN
        rows with d_date == TP_i.d_date are also held out, so no temporal
        signal from TP_i's window can leak into the training fold.

        Uses yfinance's multi-ticker bulk download (one API call per TP window)
        rather than one call per ticker, keeping total requests to N_TP calls.

        Parameters
        ----------
        tn_tickers   : candidate TN ticker symbols
        tp_cases_df  : TP cases — must have a 'd_date' or 'case_date' column
        lookback_days: observation window length
        buffer_days  : days before D to end the window
        chunk_size   : tickers per yfinance batch call (avoids timeouts)

        Returns (ohlcv_df, metadata_df) both with label = 0.
        """
        d_col = "d_date" if "d_date" in tp_cases_df.columns else "case_date"
        tp_dates = sorted(pd.to_datetime(tp_cases_df[d_col]).unique())

        ohlcv_parts: list[pd.DataFrame] = []

        for d_date in tp_dates:
            d = pd.Timestamp(d_date)
            fetch_start = d - timedelta(days=lookback_days + 40)
            fetch_end   = d - timedelta(days=buffer_days)

            print(f"\n  TN window for D={d.date()}: fetching {len(tn_tickers)} candidates ...")

            for i in range(0, len(tn_tickers), chunk_size):
                chunk = tn_tickers[i : i + chunk_size]
                try:
                    raw = yf.download(
                        chunk,
                        start=fetch_start,
                        end=fetch_end,
                        progress=False,
                        auto_adjust=True,
                        group_by="ticker",
                    )
                except Exception as exc:
                    print(f"    Warning: batch download failed for chunk {i}–{i+chunk_size}: {exc}")
                    time.sleep(delay_seconds * 5)
                    continue

                if raw.empty:
                    continue

                for ticker in chunk:
                    try:
                        if isinstance(raw.columns, pd.MultiIndex):
                            # group_by="ticker" → level-0 is ticker, level-1 is field
                            if ticker not in raw.columns.get_level_values(0):
                                continue
                            ticker_df = raw[ticker].copy()
                        else:
                            ticker_df = raw.copy()

                        ticker_df = ticker_df.dropna(how="all")
                        if ticker_df.empty or len(ticker_df) < 10:
                            continue

                        # Keep only the most recent lookback_days trading days
                        ticker_df = ticker_df.iloc[-lookback_days:]
                        ticker_df = (
                            ticker_df.reset_index()
                            .rename(columns={"index": "date", "Date": "date"})
                        )
                        ticker_df.columns = ticker_df.columns.str.lower()

                        required = {"open", "high", "low", "close", "volume"}
                        if not required.issubset(set(ticker_df.columns)):
                            continue

                        ticker_df["ticker"] = ticker
                        ticker_df["d_date"] = d
                        ticker_df["label"]  = 0

                        ohlcv_parts.append(
                            ticker_df[["date", "ticker", "d_date",
                                       "open", "high", "low", "close", "volume", "label"]]
                        )
                    except Exception:
                        continue

                time.sleep(delay_seconds)

            n_so_far = len(ohlcv_parts)
            print(f"    {n_so_far} ticker-window pairs collected so far")

        ohlcv_df = pd.concat(ohlcv_parts, ignore_index=True) if ohlcv_parts else pd.DataFrame()

        # Metadata: one fetch per unique TN ticker (not per TP window)
        print(f"\n  Fetching metadata for {len(tn_tickers)} TN tickers ...")
        meta_rows: list[dict] = []
        for i, ticker in enumerate(tn_tickers, 1):
            meta = self.download_metadata(ticker)
            meta["label"] = 0
            meta_rows.append(meta)
            if i % 50 == 0 or i == len(tn_tickers):
                print(f"    {i}/{len(tn_tickers)} metadata fetched")
            time.sleep(0.1)

        metadata_df = pd.DataFrame(meta_rows)
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
