"""CIK resolution against an in-memory fixture — no network involved.
Mirrors the shape of the real company_tickers.json SEC publishes.
"""

import pytest

from fin_analyzer.cik_lookup import TickerNotFoundError, resolve_cik

SAMPLE_TICKER_MAP = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
}


def test_resolve_cik_pads_to_ten_digits():
    result = resolve_cik("AAPL", SAMPLE_TICKER_MAP)
    assert result == {"cik": "0000320193", "ticker": "AAPL", "name": "Apple Inc."}


def test_resolve_cik_is_case_insensitive():
    result = resolve_cik("aapl", SAMPLE_TICKER_MAP)
    assert result["cik"] == "0000320193"


def test_resolve_cik_unknown_ticker_raises():
    with pytest.raises(TickerNotFoundError):
        resolve_cik("NOPE", SAMPLE_TICKER_MAP)
