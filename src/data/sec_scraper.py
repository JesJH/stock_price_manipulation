import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path


class SECPumpDumpScraper:
    """
    Scrapes SEC EDGAR litigation releases to extract confirmed pump-and-dump cases.

    Confirmed cases provide real TP tickers with actual fraud dates (D dates),
    which anchor the observation windows used in feature engineering.
    """

    EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
    SEC_BASE = "https://www.sec.gov"

    # Patterns to extract tickers from free-text SEC documents
    TICKER_PATTERNS = [
        r'\((?:ticker|symbol|trading\s+symbol):\s*([A-Z]{1,5})\)',
        r'(?:ticker|trading)\s+symbol[^a-zA-Z]{1,10}([A-Z]{1,5})\b',
        r'\btraded\s+(?:on|under|as)\s+["\']?([A-Z]{1,5})["\']?',
        r'\bstock\s+symbol\s+["\']?([A-Z]{1,5})["\']?',
    ]

    # Common words that look like tickers but aren't
    TICKER_STOPWORDS = {
        "THE", "AND", "FOR", "SEC", "LLC", "INC", "LTD", "USA", "USD", "NYSE",
        "OTC", "CEO", "CFO", "NOT", "BUT", "ALL", "ARE", "WAS", "ITS", "HIS",
        "HER", "WHO", "ACT", "ANY", "NEW", "OLD", "TWO", "ONE", "TEN",
    }

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _search_litigation_releases(
        self,
        query: str,
        start_date: str,
        end_date: str,
        page_size: int = 100,
    ) -> list[dict]:
        """Query EDGAR full-text search for litigation releases matching a query."""
        params = {
            "q": f'"{query}"',
            "forms": "LR",
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "from": 0,
            "size": page_size,
        }
        results = []
        while True:
            resp = requests.get(self.EFTS_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break
            results.extend(hits)
            params["from"] += len(hits)
            total = data.get("hits", {}).get("total", {}).get("value", 0)
            if len(results) >= total or len(hits) < page_size:
                break
            time.sleep(0.5)
        return results

    def _fetch_document_text(self, url: str) -> str:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            return soup.get_text(separator=" ")
        except Exception:
            return ""

    def _extract_tickers(self, text: str) -> list[str]:
        tickers = set()
        for pattern in self.TICKER_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidate = match.group(1).upper()
                if candidate not in self.TICKER_STOPWORDS and 1 <= len(candidate) <= 5:
                    tickers.add(candidate)
        return sorted(tickers)

    def _hit_to_document_url(self, hit: dict) -> str:
        """Construct the SEC document URL from an EDGAR search hit."""
        source = hit.get("_source", {})
        # EDGAR stores the file path; reconstruct the full URL
        file_path = source.get("file_path") or hit.get("_id", "")
        if file_path and not file_path.startswith("http"):
            return f"{self.SEC_BASE}{file_path}"
        return file_path

    def fetch_confirmed_cases(
        self,
        start_date: str = "2005-01-01",
        end_date: str = "2023-12-31",
    ) -> pd.DataFrame:
        """
        Return a DataFrame of confirmed P&D tickers sourced from SEC litigation releases.

        Columns: ticker, company, case_date, case_url
        case_date is the litigation release date — used as a proxy for D (fraud date).
        """
        hits = self._search_litigation_releases("pump and dump", start_date, end_date)
        print(f"  Found {len(hits)} litigation releases mentioning 'pump and dump'")

        records = []
        for i, hit in enumerate(hits, 1):
            source = hit.get("_source", {})
            case_date = source.get("file_date", "")
            entity_names = source.get("display_names") or source.get("entity_names") or []
            company = entity_names[0] if entity_names else ""
            doc_url = self._hit_to_document_url(hit)

            if doc_url:
                text = self._fetch_document_text(doc_url)
                tickers = self._extract_tickers(text)
            else:
                tickers = []

            for ticker in tickers:
                records.append(
                    {
                        "ticker": ticker,
                        "company": company,
                        "case_date": case_date,
                        "case_url": doc_url,
                    }
                )

            if i % 10 == 0:
                print(f"  Processed {i}/{len(hits)} releases...")
            time.sleep(0.3)

        df = pd.DataFrame(records)
        if not df.empty:
            df["case_date"] = pd.to_datetime(df["case_date"], errors="coerce")
            df = df.dropna(subset=["case_date"])
            df = df.drop_duplicates(subset=["ticker", "case_date"])
            df = df.sort_values("case_date").reset_index(drop=True)
        return df

    def save(self, df: pd.DataFrame, filename: str = "tp_tickers.csv") -> Path:
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} records to {path}")
        return path
