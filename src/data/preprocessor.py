import pandas as pd

from src.features.feature_config import (
    MIN_TRADING_DAYS,
    SUPPORTED_COUNTRIES,
    SUPPORTED_CURRENCIES,
)


class Preprocessor:
    """
    Cleans and filters raw OHLCV + metadata before feature engineering.

    Responsibilities
    ----------------
    1. Filter to supported countries / currencies (USA and Canada for this demo)
    2. Drop tickers with insufficient trading history in the observation window
    3. Provide a merged view of OHLCV and metadata for analysis

    Country filter note
    -------------------
    In the original production system, country of origination was NOT an
    explicit filter — the model operated across all securities on the
    brokerage's platform. For this public portfolio reproduction we restrict
    to US and Canadian equities because SEC enforcement data is US-focused
    and yfinance coverage is most reliable for North American exchanges.
    This is a data availability decision, not a modelling one.
    """

    def __init__(
        self,
        min_trading_days: int = MIN_TRADING_DAYS,
        supported_countries: list[str] = SUPPORTED_COUNTRIES,
        supported_currencies: list[str] = SUPPORTED_CURRENCIES,
    ):
        self.min_trading_days   = min_trading_days
        self.supported_countries = supported_countries
        self.supported_currencies = supported_currencies

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ohlcv_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the full preprocessing pipeline.

        Returns
        -------
        (filtered_ohlcv, filtered_metadata)
        """
        ohlcv, meta = self.filter_by_country(ohlcv_df, metadata_df)
        ohlcv = self.drop_insufficient_history(ohlcv)
        # Re-align metadata to only tickers that survive the history filter
        surviving_tickers = ohlcv["ticker"].unique()
        meta = meta[meta["ticker"].isin(surviving_tickers)].reset_index(drop=True)
        return ohlcv, meta

    def filter_by_country(
        self,
        ohlcv_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Keep only tickers whose country and currency match the supported scope.
        Tickers missing country/currency metadata are retained with a warning.
        """
        if metadata_df.empty or "country" not in metadata_df.columns:
            return ohlcv_df, metadata_df

        before = metadata_df["ticker"].nunique()

        country_mask  = metadata_df["country"].isin(self.supported_countries) | metadata_df["country"].isna()
        currency_mask = (
            metadata_df["currency"].isin(self.supported_currencies) | metadata_df["currency"].isna()
            if "currency" in metadata_df.columns
            else pd.Series(True, index=metadata_df.index)
        )
        filtered_meta = metadata_df[country_mask & currency_mask].reset_index(drop=True)

        removed = before - filtered_meta["ticker"].nunique()
        if removed > 0:
            print(
                f"  Country filter: removed {removed} ticker(s) outside "
                f"{self.supported_countries} / {self.supported_currencies}"
            )

        kept_tickers = filtered_meta["ticker"].unique()
        filtered_ohlcv = ohlcv_df[ohlcv_df["ticker"].isin(kept_tickers)].reset_index(drop=True)
        return filtered_ohlcv, filtered_meta

    def drop_insufficient_history(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Drop tickers that have fewer than min_trading_days rows in the window."""
        counts = ohlcv_df.groupby("ticker")["date"].count()
        drop = counts[counts < self.min_trading_days].index.tolist()
        if drop:
            print(
                f"  History filter: dropped {len(drop)} ticker(s) with "
                f"<{self.min_trading_days} trading days: {drop}"
            )
        return ohlcv_df[~ohlcv_df["ticker"].isin(drop)].reset_index(drop=True)

    def merge_ohlcv_metadata(
        self,
        ohlcv_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Left-join OHLCV with metadata on ticker for combined analysis."""
        if metadata_df.empty:
            return ohlcv_df
        return ohlcv_df.merge(
            metadata_df.drop(columns=["d_date"], errors="ignore"),
            on="ticker",
            how="left",
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, ohlcv_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
        """Print a concise summary of the filtered dataset."""
        n_tickers  = ohlcv_df["ticker"].nunique()
        n_rows     = len(ohlcv_df)
        n_tp       = ohlcv_df[ohlcv_df["label"] == 1]["ticker"].nunique() if "label" in ohlcv_df.columns else "n/a"
        days_stats = ohlcv_df.groupby("ticker")["date"].count()

        print("=== Preprocessed Dataset ===")
        print(f"  Unique tickers:   {n_tickers}")
        print(f"  TP tickers:       {n_tp}")
        print(f"  Total OHLCV rows: {n_rows:,}")
        print(f"  Trading days per ticker — median: {days_stats.median():.0f}, "
              f"min: {days_stats.min()}, max: {days_stats.max()}")
        if "country" in metadata_df.columns:
            print(f"  Countries:        {sorted(metadata_df['country'].dropna().unique())}")
