# FinAnalyzer

Phase 1: pull SEC 10-K filings into MongoDB. No AI, no API, no MCP server —
those are later phases. See [CLAUDE.md](CLAUDE.md) for the full design
notes and constraints.

## Setup

1. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/) and Python 3.12):

   ```
   uv sync
   ```

2. **Get a MongoDB Atlas connection string.** Free M0 tier is enough:
   Atlas → Database → Connect → Drivers → copy the `mongodb+srv://...` URI.

3. **Create your `.env`:**

   ```
   cp .env.example .env
   ```

   Then edit `.env` and fill in:
   - `MONGODB_URI` — the Atlas connection string from step 2.
   - `SEC_USER_AGENT` — your real name and email, e.g.
     `Jane Doe jane@example.com`. SEC requires this on every request and
     returns `403 Forbidden` without it — see their
     [developer FAQ](https://www.sec.gov/os/webmaster-faq#developers).

   `.env` is gitignored — it will never be committed.

## Run the ingest

```
uv run ingest AAPL MSFT GOOGL
```

This resolves each ticker to a CIK, finds its most recent 10-K, downloads
the filing, extracts and chunks the text, and writes everything to the
`finanalyzer` database. Downloaded filings are cached under `.cache/`
(gitignored), so rerunning is fast and doesn't re-hit SEC.

Rerunning with the same tickers is safe — it updates the existing
company/filing records and replaces that filing's chunks rather than
duplicating them.

At the end it prints 3 random chunks from the database so you can eyeball
extraction quality directly.

### Expected result

- `companies` has one document per ticker.
- `filings` has one document per ticker (its most recent 10-K).
- `chunks` has on the order of 40-60 documents per filing (real 10-K prose —
  Business, Risk Factors, MD&A, footnotes — runs to a few hundred KB per
  filing, which is what 1000-token chunks with 150-token overlap divides
  into). That's dozens of chunks per ticker, not thousands; the total grows
  as more tickers get ingested over time.

## Tests

```
uv run pytest
```

Needs `.env` configured (same file as above) — the idempotency tests write
to a real `finanalyzer_test` database on your Atlas cluster (never
`finanalyzer`) and clean up after themselves. No test makes a real SEC
network call.

## Project layout

```
src/fin_analyzer/
  config.py        # settings, from .env
  db.py             # Mongo client/indexes
  edgar_client.py   # SEC HTTP calls + on-disk cache + rate limiting
  cik_lookup.py     # ticker -> CIK (pure function)
  extract.py        # 10-K HTML -> clean prose
  chunk.py          # prose -> overlapping token chunks
  ingest.py         # orchestrates one ticker, writes to Mongo
  cli.py            # `ingest` entry point
tests/
```
