#!/usr/bin/env python3
"""
CLI script: download TP tickers from SEC and/or fetch OHLCV + metadata via yfinance.

Usage:
    python scripts/download_data.py --source sec
    python scripts/download_data.py --source yfinance
    python scripts/download_data.py --source all --start-date 2010-01-01 --end-date 2023-12-31
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.downloader import MarketDataDownloader
from src.data.sec_scraper import SECPumpDumpScraper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download data for pump-and-dump security detection"
    )
    parser.add_argument(
        "--source",
        choices=["sec", "yfinance", "all"],
        default="all",
        help="Which data to fetch (default: all)",
    )
    parser.add_argument("--start-date", default="2010-01-01", help="SEC search start date")
    parser.add_argument("--end-date", default="2023-12-31", help="SEC search end date")
    parser.add_argument(
        "--lookback-days", type=int, default=90, help="Observation window length in days"
    )
    parser.add_argument(
        "--buffer-days", type=int, default=5, help="Days before D to end the observation window"
    )
    parser.add_argument(
        "--tp-file",
        default="data/external/tp_tickers.csv",
        help="Path to TP tickers CSV (used when --source yfinance)",
    )
    return parser.parse_args()


def run_sec_scrape(args: argparse.Namespace) -> Path:
    print("=== Step 1: Fetching SEC litigation releases ===")
    scraper = SECPumpDumpScraper(output_dir="data/external")
    cases_df = scraper.fetch_confirmed_cases(
        start_date=args.start_date, end_date=args.end_date
    )
    print(f"  Extracted {len(cases_df)} confirmed TP ticker/date pairs")
    return scraper.save(cases_df, "tp_tickers.csv")


def run_yfinance_download(args: argparse.Namespace) -> None:
    import pandas as pd

    tp_path = Path(args.tp_file)
    if not tp_path.exists():
        print(f"ERROR: TP tickers file not found at {tp_path}")
        print("  Run --source sec first to generate it.")
        sys.exit(1)

    cases_df = pd.read_csv(tp_path, parse_dates=["d_date"])
    cases_df = cases_df.rename(columns={"d_date": "case_date"})
    print(f"\n=== Step 2: Downloading market data for {len(cases_df)} TP tickers ===")
    print(f"  Window: D-{args.lookback_days} to D-{args.buffer_days}")

    downloader = MarketDataDownloader(output_dir="data/raw")
    ohlcv_df, meta_df = downloader.download_batch(
        cases_df,
        lookback_days=args.lookback_days,
        buffer_days=args.buffer_days,
        label=1,
    )

    if not ohlcv_df.empty:
        downloader.save_parquet(ohlcv_df, "tp_ohlcv.parquet")
    else:
        print("  Warning: no OHLCV data retrieved")

    if not meta_df.empty:
        downloader.save_csv(meta_df, "tp_metadata.csv")


def main() -> None:
    args = parse_args()

    if args.source in ("sec", "all"):
        run_sec_scrape(args)

    if args.source in ("yfinance", "all"):
        run_yfinance_download(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
