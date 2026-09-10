"""fin_analyzer.mcp_server — tool shapes, model-readable errors, and the
stdout-purity guarantee stdio transport depends on.

search()/ask()/companies.list_companies() are mocked at the call site (same
pattern as test_api.py) so these tests spend zero Gemini/Groq quota. Note:
importing fin_analyzer.mcp_server itself runs one real, read-only query
against the production `finanalyzer` database (to build the tool
descriptions' ticker list at server-startup time — see the module's own
docstring for why this is eager rather than deferred). That's a one-time
cost paid once when this test module is first imported, not per test, and
nothing here writes to `finanalyzer`.
"""

import io
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from fin_analyzer.core.exceptions import IndexNotReady, QuotaExhausted, TickerNotFound
from fin_analyzer.core.models import Answer, Source
from fin_analyzer.mcp_server import (
    _friendly_error,
    ask_filings_tool,
    list_companies_tool,
    search_filings_tool,
)
from fin_analyzer.search import Chunk


# --- shape tests ---


@patch("fin_analyzer.mcp_server.companies.list_companies")
def test_list_companies_tool_returns_expected_shape(mock_list_companies):
    from fin_analyzer.companies import Company

    mock_list_companies.return_value = [
        Company(ticker="AAPL", name="Apple Inc.", sector="Electronic Computers", cik="0000320193", latest_filing_date="2025-10-31")
    ]

    result = list_companies_tool()

    assert result == [
        {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Electronic Computers", "cik": "0000320193", "latest_filing_date": "2025-10-31"}
    ]


@patch("fin_analyzer.mcp_server.search")
def test_search_filings_tool_returns_expected_shape(mock_search):
    mock_search.return_value = [
        Chunk(ticker="AAPL", chunk_index=9, text="some risk text", score=0.87, filing_date="2025-10-31")
    ]

    result = search_filings_tool("supply chain risk", ticker="AAPL", limit=5)

    assert result == [
        {"ticker": "AAPL", "chunk_index": 9, "filing_date": "2025-10-31", "score": 0.87, "text": "some risk text"}
    ]


@patch("fin_analyzer.mcp_server.ask")
def test_ask_filings_tool_returns_expected_shape(mock_ask):
    mock_ask.return_value = Answer(
        question="how does Apple describe supply chain risk?",
        answer="Apple relies on outsourcing partners [1].",
        sources=[Source(ticker="AAPL", filing_date="2025-10-31", chunk_index=9, score=0.87, text="full text", cited=True)],
        context_used=1,
    )

    result = ask_filings_tool("how does Apple describe supply chain risk?")

    assert result["answer"] == "Apple relies on outsourcing partners [1]."
    assert result["sources"] == [
        {"ticker": "AAPL", "filing_date": "2025-10-31", "chunk_index": 9, "score": 0.87, "text": "full text", "cited": True}
    ]


# --- model-readable errors, not exception traces ---


@patch("fin_analyzer.mcp_server.companies.list_companies")
def test_friendly_error_names_available_companies_for_ticker_not_found(mock_list_companies):
    from fin_analyzer.companies import Company

    mock_list_companies.return_value = [
        Company(ticker="AAPL", name="Apple Inc.", sector=None, cik="1", latest_filing_date=None),
        Company(ticker="MSFT", name="Microsoft", sector=None, cik="2", latest_filing_date=None),
    ]

    text = _friendly_error(TickerNotFound("TSLA"))

    assert text == "TSLA is not in this corpus. Available companies: AAPL, MSFT."


def test_friendly_error_for_quota_exhausted_gives_a_retry_path():
    text = _friendly_error(QuotaExhausted("rate limited", retry_after_seconds=30.0))
    assert "30 seconds" in text


def test_friendly_error_for_index_not_ready():
    text = _friendly_error(IndexNotReady("not ready"))
    assert "try again" in text.lower()


@patch("fin_analyzer.mcp_server.search")
def test_search_filings_tool_returns_text_not_a_traceback_on_ticker_not_found(mock_search):
    mock_search.side_effect = TickerNotFound("TSLA")

    result = search_filings_tool("risk", ticker="TSLA")

    assert isinstance(result, str)
    assert "TSLA" in result
    assert "Available companies" in result


@patch("fin_analyzer.mcp_server.ask")
def test_ask_filings_tool_returns_text_not_a_traceback_on_ticker_not_found(mock_ask):
    mock_ask.side_effect = TickerNotFound("TSLA")

    result = ask_filings_tool("what are Tesla's risks?", ticker="TSLA")

    assert isinstance(result, str)
    assert "TSLA" in result


# --- stdout purity: stdio transport means stdout IS the protocol ---


@patch("fin_analyzer.mcp_server.search")
def test_search_filings_tool_never_writes_to_stdout(mock_search):
    mock_search.return_value = [
        Chunk(ticker="AAPL", chunk_index=9, text="text", score=0.87, filing_date="2025-10-31")
    ]

    captured = io.StringIO()
    with redirect_stdout(captured):
        search_filings_tool("risk", ticker="AAPL")

    assert captured.getvalue() == ""


@patch("fin_analyzer.mcp_server.search")
def test_search_filings_tool_error_path_never_writes_to_stdout(mock_search):
    mock_search.side_effect = TickerNotFound("TSLA")

    captured = io.StringIO()
    with redirect_stdout(captured):
        search_filings_tool("risk", ticker="TSLA")

    assert captured.getvalue() == ""
