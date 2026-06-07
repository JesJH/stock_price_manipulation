"""
TN (True Negative) candidate universe builder.

Pulls small-cap and OTC securities from public NASDAQ listing files,
filters out known TP tickers and non-equity symbols, and saves a curated
candidate pool for downstream OHLCV download and centroid matching.
"""

import io
from pathlib import Path

import pandas as pd
import requests


class TNUniverseBuilder:
    """
    Builds a pool of TN candidate tickers from public NASDAQ listing files.

    Sources
    -------
    NASDAQ trader listing files (publicly available via HTTP):
      - nasdaqlisted.txt  : all NASDAQ-listed securities
      - otherlisted.txt   : NYSE, AMEX, OTC, and other exchange securities

    Filtering
    ---------
    - Exclude ETFs, test issues, warrants, rights, and preferred shares
      (symbols containing special characters or ending in W/R/P)
    - Symbols must be 1–5 characters (rules out most derivative symbols)
    - Exclude all confirmed TP tickers

    Parameters
    ----------
    output_dir   : directory to write tn_candidates.csv
    n_candidates : number of tickers to sample from the filtered pool
    random_state : reproducibility seed for sampling
    """

    NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    OTHER_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    def __init__(
        self,
        output_dir: str | Path = "data/external",
        n_candidates: int = 500,
        random_state: int = 42,
    ):
        self.output_dir   = Path(output_dir)
        self.n_candidates = n_candidates
        self.random_state = random_state
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        tp_tickers: list[str] | None = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch, filter, and sample TN candidate tickers.

        Parameters
        ----------
        tp_tickers : confirmed TP tickers to exclude from the candidate pool
        save       : write tn_candidates.csv to output_dir if True

        Returns
        -------
        DataFrame with columns: ticker, security_name, source
        """
        tp_set = {t.upper() for t in (tp_tickers or [])}

        frames = []
        for url, tag in [
            (self.NASDAQ_LISTED_URL, "nasdaq"),
            (self.OTHER_LISTED_URL,  "other"),
        ]:
            df = self._fetch_listing(url, tag)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            raise RuntimeError(
                "Could not fetch any listing data from NASDAQ trader. "
                "Check network connectivity or supply a manual tn_candidates.csv."
            )

        candidates = pd.concat(frames, ignore_index=True)
        candidates = self._filter(candidates, tp_set)
        print(f"  TNUniverseBuilder: {len(candidates)} symbols after filtering")

        if len(candidates) > self.n_candidates:
            candidates = (
                candidates
                .sample(n=self.n_candidates, random_state=self.random_state)
                .reset_index(drop=True)
            )

        print(f"  TNUniverseBuilder: {len(candidates)} candidate tickers sampled")

        if save:
            path = self.output_dir / "tn_candidates.csv"
            candidates.to_csv(path, index=False)
            print(f"  Saved to {path}")

        return candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_listing(self, url: str, source_tag: str) -> pd.DataFrame | None:
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  Warning: could not fetch {url}: {exc}")
            return None

        # Strip the trailing "File Creation Time" line that NASDAQ appends
        lines = [
            line for line in resp.text.splitlines()
            if not line.startswith("File Creation")
        ]
        text = "\n".join(lines)

        try:
            df = pd.read_csv(io.StringIO(text), sep="|")
        except Exception as exc:
            print(f"  Warning: could not parse listing from {url}: {exc}")
            return None

        if source_tag == "nasdaq":
            # nasdaqlisted.txt columns:
            # Symbol | Security Name | Market Category | Test Issue |
            # Financial Status | Round Lot Size | ETF | NextShares
            if "Symbol" not in df.columns:
                return None
            df = df.rename(columns={"Symbol": "ticker", "Security Name": "security_name"})
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] != "Y"]
            if "ETF" in df.columns:
                df = df[df["ETF"] != "Y"]

        else:
            # otherlisted.txt columns:
            # ACT Symbol | Security Name | Exchange | CQS Symbol |
            # ETF | Round Lot Size | Test Issue | NASDAQ Symbol
            if "ACT Symbol" not in df.columns:
                return None
            df = df.rename(columns={"ACT Symbol": "ticker", "Security Name": "security_name"})
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] != "Y"]
            if "ETF" in df.columns:
                df = df[df["ETF"] != "Y"]

        df["source"] = source_tag
        return df[["ticker", "security_name", "source"]].dropna(subset=["ticker"])

    def _filter(self, df: pd.DataFrame, tp_set: set) -> pd.DataFrame:
        df = df.copy()
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

        # Remove derivatives, warrants, rights, preferred shares
        special_chars = df["ticker"].str.contains(r"[.+$^~%]", regex=True, na=False)
        ends_warrant  = df["ticker"].str.endswith("W")
        ends_right    = df["ticker"].str.endswith("R")
        ends_preferred = df["ticker"].str.endswith("P")
        length_ok     = df["ticker"].str.len().between(1, 5)

        df = df[~special_chars & ~ends_warrant & ~ends_right & ~ends_preferred & length_ok]
        df = df[~df["ticker"].isin(tp_set)]
        return df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
