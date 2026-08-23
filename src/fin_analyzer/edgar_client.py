"""All HTTP calls to SEC EDGAR, plus the on-disk cache and rate limiting.

Every function here is I/O — no parsing/business logic lives in this module,
so it can stay a thin, boring wrapper around `requests`. Logic that needs to
be unit-tested without a network connection (e.g. picking a CIK out of the
ticker map) lives in cik_lookup.py instead, operating on plain dicts.
"""

import json
import time
from pathlib import Path

import requests

from fin_analyzer.config import Settings

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


def cached_get(url: str, cache_path: Path, settings: Settings) -> str:
    """Return the response body at `url` as text.

    If `cache_path` already exists, read from disk and skip the network
    entirely — this is what makes reruns fast and keeps us off SEC's rate
    limit. Only an actual network call is followed by a sleep.
    """
    if cache_path.exists():
        return cache_path.read_text()

    response = requests.get(
        url,
        headers={"User-Agent": settings.sec_user_agent},
        timeout=30,
    )
    response.raise_for_status()
    time.sleep(settings.request_sleep_seconds)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text)
    return response.text


def fetch_ticker_map(settings: Settings) -> dict:
    """The full ticker -> CIK map SEC publishes. Same file serves every ticker
    in a run, so it's fetched (and cached) once regardless of batch size."""
    cache_path = settings.cache_dir / "company_tickers.json"
    body = cached_get(TICKER_MAP_URL, cache_path, settings)
    return json.loads(body)


def fetch_submissions(cik_padded: str, settings: Settings) -> dict:
    """cik_padded must already be zero-padded to 10 digits, e.g. '0000320193'."""
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    cache_path = settings.cache_dir / "submissions" / f"CIK{cik_padded}.json"
    body = cached_get(url, cache_path, settings)
    return json.loads(body)


def build_filing_url(cik: str, accession_no: str, primary_document: str) -> str:
    """Note the CIK in this URL path is NOT zero-padded (unlike the
    submissions JSON filename above) — that's SEC's convention, not a typo.
    """
    accession_no_nodash = accession_no.replace("-", "")
    cik_unpadded = str(int(cik))
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_unpadded}/{accession_no_nodash}/{primary_document}"
    )


def fetch_filing_html(
    cik: str, accession_no: str, primary_document: str, ticker: str, settings: Settings
) -> str:
    """Download a filing's primary document HTML."""
    url = build_filing_url(cik, accession_no, primary_document)
    cache_path = settings.cache_dir / "filings" / f"{ticker}_{accession_no}.html"
    return cached_get(url, cache_path, settings)
