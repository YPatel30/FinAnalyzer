"""MCP server exposing the same three capabilities as the Phase 4 REST API
(list companies, search, ask) to a model instead of a programmer. See
README.md for the full REST-vs-MCP comparison this phase is really about.

Tool descriptions are prompts, not documentation. A REST docstring is read
by a developer who already decided to call the endpoint; an MCP tool
description is read by a model deciding whether to call it at all, with no
other context. Each one below says when to use the tool (with one crisp
positive criterion, not mutual hedging toward the other tool), what its
arguments mean in terms a model can actually supply, and what comes back.

`ingest` is deliberately not exposed here — see the "why ingest isn't an
MCP tool" note in CLAUDE.md; the short version is that REST's 202+poll
pattern relies on the *client* holding durable state across an arbitrary
wait, which a model calling tools inside one bounded reasoning turn has no
reliable equivalent for.

stdio is the transport, so stdout IS the protocol — nothing in this module
or anything it imports may print() to stdout (audited; see CLAUDE.md for
what was found and fixed: embeddings.py and providers/groq_provider.py both
had retry-log prints on stdout, now routed to stderr). Logging here goes to
stderr explicitly, never stdout.
"""

import logging
import sys

from mcp.server.mcpserver import MCPServer

from fin_analyzer import companies
from fin_analyzer.ask import ask
from fin_analyzer.config import get_settings
from fin_analyzer.core.exceptions import IndexNotReady, QuotaExhausted, TickerNotFound
from fin_analyzer.db import get_db
from fin_analyzer.search import search

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("fin_analyzer.mcp")

settings = get_settings()
db = get_db(settings)

mcp = MCPServer("finanalyzer")


def _ticker_list_text() -> str:
    """A live query, not a cached string — used both to build the tool
    descriptions once at startup (below) and inside error messages at call
    time, in case the corpus changed between the two."""
    return ", ".join(c.ticker for c in companies.list_companies(db))


# Computed once at import time (server startup), not hardcoded — the
# description can't go stale as more tickers get ingested. Inlining the
# list is fine at 5 companies; past some size (a few dozen? a few hundred?)
# this should drop the inline list and just point the model at
# list_companies() to look it up — the description itself should stay short
# regardless of how large the corpus gets.
_TICKERS = _ticker_list_text()

LIST_COMPANIES_DESCRIPTION = """Lists every company whose SEC 10-K filings have been ingested. Call this
first if you're unsure whether a company is available — the `ticker`
argument to search_filings and ask_filings only works for companies
returned here.

Returns each company's ticker, name, sector, and most recent 10-K filing
date. Metadata only, no filing content — use search_filings or
ask_filings for that."""

SEARCH_FILINGS_DESCRIPTION = f"""Searches the ingested SEC 10-K filings for passages relevant to `query`
and returns the raw text excerpts — no answer is generated. Use this when
the passages themselves matter: exact wording, comparing multiple
sources, or forming your own read rather than relying on a summary.

Arguments:
- query (required): a natural-language question or topic, e.g. "supply
  chain risk".
- ticker (optional): restrict to one company. Must be one of the ingested
  companies — currently {_TICKERS}. Omit to search across all of them.
- limit (optional, default 5): number of passages to return.

Returns a list of passages, each with ticker, relevance score, filing
date, and full text — excerpts from real 10-K filings (Business, Risk
Factors, MD&A sections). Treat them as source material to read and
reason about, not as a finished answer."""

ASK_FILINGS_DESCRIPTION = f"""Answers a question about the ingested SEC 10-K filings and returns a
written answer with inline [n] citations, plus the source passages those
citations refer to. Retrieval and answer composition happen inside this
tool.

Use this for a direct answer to a specific question. Use search_filings
instead when you want to read source passages and draw your own
conclusion.

If the filings don't cover the question — a company that isn't ingested,
a forward-looking question, or an unrelated topic — it says so explicitly
rather than answering from general knowledge.

Arguments:
- question (required): a natural-language question, e.g. "how does Apple
  describe supply chain risk?"
- ticker (optional): restrict retrieval to one company. Must be one of
  the ingested companies — currently {_TICKERS}. Omit if the question
  isn't about one specific company.

Returns: the answer text (with inline [n] citations) and the source
passages those citations refer to (company, filing date, full text)."""


def _friendly_error(exc: Exception) -> str:
    """Map a core exception to text a model can act on and retry from —
    the same exceptions the REST API maps to HTTP status codes (Phase 4),
    a completely different presentation for a completely different kind
    of caller. A status code is for a program; a model needs a sentence
    that gives it a path to retry correctly.
    """
    if isinstance(exc, TickerNotFound):
        return f"{exc.ticker} is not in this corpus. Available companies: {_ticker_list_text()}."
    if isinstance(exc, QuotaExhausted):
        if exc.retry_after_seconds is not None:
            return f"The answer-generation service is temporarily rate-limited. Try again in about {int(exc.retry_after_seconds)} seconds."
        return "The answer-generation service is temporarily rate-limited. Try again shortly."
    if isinstance(exc, IndexNotReady):
        return "The search index isn't ready yet. Try again in a moment."
    logger.exception("Unexpected error in an MCP tool call")
    return f"An unexpected error occurred: {exc}"


@mcp.tool(name="list_companies", description=LIST_COMPANIES_DESCRIPTION)
def list_companies_tool() -> list[dict]:
    return [
        {
            "ticker": c.ticker,
            "name": c.name,
            "sector": c.sector,
            "cik": c.cik,
            "latest_filing_date": c.latest_filing_date,
        }
        for c in companies.list_companies(db)
    ]


@mcp.tool(name="search_filings", description=SEARCH_FILINGS_DESCRIPTION)
def search_filings_tool(query: str, ticker: str | None = None, limit: int = 5) -> list[dict] | str:
    try:
        chunks = search(query, ticker=ticker, limit=limit, db=db, settings=settings)
    except (TickerNotFound, QuotaExhausted, IndexNotReady) as exc:
        return _friendly_error(exc)
    return [
        {"ticker": c.ticker, "chunk_index": c.chunk_index, "filing_date": c.filing_date, "score": c.score, "text": c.text}
        for c in chunks
    ]


@mcp.tool(name="ask_filings", description=ASK_FILINGS_DESCRIPTION)
def ask_filings_tool(question: str, ticker: str | None = None) -> dict | str:
    try:
        answer = ask(question, ticker=ticker, k=5, db=db, settings=settings)
    except (TickerNotFound, QuotaExhausted, IndexNotReady) as exc:
        return _friendly_error(exc)
    return {
        "answer": answer.answer,
        "sources": [
            {
                "ticker": s.ticker,
                "filing_date": s.filing_date,
                "chunk_index": s.chunk_index,
                "score": s.score,
                "text": s.text,
                "cited": s.cited,
            }
            for s in answer.sources
        ],
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
