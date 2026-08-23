# FinAnalyzer

A learning project. The owner is new to FastAPI, MongoDB, MCP, and RAG, but
knows Python. Priority is understanding every piece, not speed of delivery —
favor clear, boring code over clever abstractions. Comment the *why*, not
the *what*. Keep functions small and single-purpose.

## Environment constraints (hard requirements, not preferences)

- 2018 Intel MacBook Pro, x86_64, 8 GB RAM, macOS. **Not** Apple Silicon.
- **Never add PyTorch, sentence-transformers, or onnxruntime, or anything
  that depends on them** — no Intel macOS wheels exist for current versions.
  If a package pulls in torch transitively, pick a different package.
- Python 3.12, managed with **uv** — not conda, not pip directly, not poetry.
- MongoDB Atlas **M0 free tier**: 512 MB storage, ~100 ops/sec, max 3 search
  indexes. Design around these limits (e.g. insert_many over per-doc loops).

## Naming conventions — use exactly these, don't improvise

| Thing | Name |
|---|---|
| Python package | `fin_analyzer` (under `src/`) |
| Mongo database | `finanalyzer` |
| Mongo test database | `finanalyzer_test` |

The git repo folder itself is `FinAnalyzer` (pre-existing, not renamed); the
`pyproject.toml` project name is `fin-analyzer`. Don't conflate these three.

## Phase plan

- **Phase 1 (done)** — ingest SEC 10-K filings into MongoDB. CLI only,
  `uv run ingest TICKER [TICKER ...]`. No AI of any kind. See below.
- **Phase 2+ (not started, don't build yet)** — the eventual system adds:
  RAG over the filings (embeddings + vector search), historical financial
  data, a FastAPI REST API, and an MCP server. None of this exists yet.
  If a task seems to need one of these to finish something in Phase 1,
  stop and say so rather than building it.

## Explicitly out of scope until told otherwise

Embeddings, vector search, any LLM/Gemini call, FastAPI, an MCP server,
Docker, a frontend, async code, authentication, LangChain generally (the one
exception already in use: `langchain-text-splitters`, for chunking only).

## Phase 1 architecture

```
src/fin_analyzer/
  config.py       # pydantic-settings, reads .env
  db.py           # pymongo client/db getters, ensure_indexes()
  edgar_client.py # all SEC HTTP calls + on-disk cache (.cache/) + rate limiting
  cik_lookup.py   # pure function: ticker -> CIK (unit-testable, no network)
  extract.py      # 10-K HTML -> clean prose text (see quality notes below)
  chunk.py        # text -> token-sized overlapping chunks
  ingest.py        # orchestrates one ticker end-to-end, writes to Mongo
  cli.py          # `ingest` entry point
```

Sync `pymongo` throughout, not motor — async was explicitly deferred to a
later phase (if ever).

### SEC EDGAR specifics

- Ticker → CIK map: `https://www.sec.gov/files/company_tickers.json`
- Submissions (per company): `https://data.sec.gov/submissions/CIK##########.json`
  — CIK zero-padded to 10 digits **in this URL only**.
- Filing docs: `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_no_dashes}/{primary_document}`
  — CIK **not** zero-padded here. Easy to mix these two up.
- Every request needs `User-Agent: <Name> <email>` or SEC returns 403.
  Comes from `SEC_USER_AGENT` in `.env` — it's the developer's identity, not
  the app's.
- Rate limit is 10 req/sec; we sleep 0.2s after every real (non-cached)
  request, so effectively ~5 req/sec.
- Everything downloaded gets cached to `.cache/` (gitignored) so reruns
  don't hit SEC again — `edgar_client.cached_get()` is the single chokepoint
  all SEC HTTP goes through.

### Mongo collections

```
companies  # ticker, cik, name, sector (sector = SEC's sicDescription — a SIC
           #  industry description, not a real GICS sector; noted as an
           #  approximation, closest thing SEC's API actually provides)
filings    # ticker, form_type, filing_date, accession_no, source_url
chunks     # ticker, filing_id, chunk_index, text, char_count
```

Indexes: unique on `companies.ticker`; unique on `filings.accession_no`
(the upsert key for filings); compound `{ticker: 1, filing_id: 1}` on
`chunks` (not unique — dedup happens at write time, see below).

### Idempotency strategy

- `companies`: upsert by `ticker`.
- `filings`: upsert by `accession_no` (SEC's permanent ID for one filing),
  via `find_one_and_update(..., return_document=AFTER)` to get the `_id`
  back in one round trip.
- `chunks`: **delete-then-insert** — `delete_many({ticker, filing_id})`
  followed by `insert_many(...)`. Rerunning the same ticker produces the
  same chunk set, not duplicates. This is why chunks doesn't need a unique
  index to enforce dedup.

### Extraction quality (`extract.py`)

10-K HTML is full of huge financial-statement tables, a table of contents,
inline-XBRL tags, and heavy inline formatting (`<span>`, `<font>`, `<b>`)
that splits single sentences across many text nodes. The pipeline:

1. Decompose `<script>`, `<style>`, `<table>` entirely — deliberately drops
   all table content, including any prose that happens to live inside a
   table, as a simplification.
2. Decompose the single `<ix:header>` tag — inline-XBRL documents put *all*
   their non-visible metadata (contexts, units, dimension members, hidden
   duplicate-tagged facts) inside this one container, verified by inspection
   to hold 100% of it. Then *unwrap* (not decompose) every other namespaced
   tag (`ix:nonfraction`, `ix:nonnumeric`, `ix:continuation`, …) — these
   appear directly in the visible body and carry real content: a single
   tagged number sitting inside an MD&A sentence, or (via nonnumeric/
   continuation) a whole footnote paragraph. An earlier version of this
   pipeline decomposed *all* colon-named tags including these, which
   silently deleted real prose — footnote text especially — and undercounted
   chunks by ~30-45%. Caught by comparing extracted character counts before/
   after on real filings; fixed by narrowing the decompose to just the
   header.
3. Unwrap purely inline tags (`span`, `b`, `i`, `a`, `font`, …) rather than
   deleting them, then call `soup.smooth()` to re-merge the text nodes they
   left behind — otherwise `get_text()` would insert line breaks mid-sentence
   wherever a `<span>` used to be.
4. Split on lines, keep only lines that are both long enough (≥40 chars) and
   mostly letters (≥50%) — this is what filters out page numbers, TOC
   leaders (`"Item 1 . . . . . 5"`), and table/nav remnants.

This is a hand-rolled heuristic, not a "smart" extraction library, on
purpose — every rule is visible and tunable. It won't be perfect; eyeball
the sample chunks the CLI prints at the end of a run to judge it.

Real 10-K prose (Business, Risk Factors, MD&A, footnotes) runs to a few
hundred KB per filing, not megabytes — so at 1000 tokens/chunk with 150
overlap, expect roughly 40-60 chunks *per filing*, not thousands. That's a
property of chunk size vs. real prose volume, not a bug; don't "fix" a low
chunk count without checking extracted character counts first.

### Chunking (`chunk.py`)

`RecursiveCharacterTextSplitter.from_tiktoken_encoder` (cl100k_base),
~1000 tokens per chunk, ~150 token overlap — token-based (via tiktoken), not
word-count-based, so the sizes are actually accurate.

## Running things

```
uv run ingest AAPL MSFT GOOGL   # the Phase 1 CLI
uv run pytest                   # tests (needs .env configured — see README)
```
