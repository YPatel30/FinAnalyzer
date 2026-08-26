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
- **Phase 2 (done)** — embeddings (Gemini API) + MongoDB Atlas Vector
  Search over the chunks. `uv run embed`, `uv run search "..."`,
  `uv run eval`. Retrieval only — no LLM call generates or answers
  anything. See below.
- **Phase 3+ (not started, don't build yet)** — the eventual system adds:
  historical financial data, an LLM call that actually answers questions
  using retrieved passages (Phase 2 stops at "retrieve good passages"), a
  FastAPI REST API, and an MCP server. None of this exists yet. If a task
  seems to need one of these to finish something earlier, stop and say so
  rather than building it.

## Explicitly out of scope until told otherwise

Any LLM/chat/generation call (embeddings are fine, Phase 2 uses them via
`google-genai` — *generation* is not), answer generation, prompt
construction, FastAPI, an MCP server, reranking, Docker, a frontend, async
code, authentication, caching layers, hybrid/keyword search, LangChain
generally (the one exception already in use: `langchain-text-splitters`,
for chunking only).

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

## Phase 2 architecture

```
src/fin_analyzer/
  embeddings.py    # Gemini calls: batching, retry-on-429, re-normalization, dimension guard
  vector_index.py  # Atlas Vector Search index create/wait (with the M0 driver-restriction fallback)
  embed.py         # orchestrates: find chunks missing embeddings -> embed -> write -> ensure index
  search.py        # Chunk dataclass + search(query, ticker=None, limit=5)
  eval.py          # recall@5 over eval_data.py's 15 approved questions
  eval_data.py     # the approved questions + expected phrases (see git history for how they were picked)
  cli_embed.py / cli_search.py / cli_eval.py   # `embed` / `search` / `eval` entry points
```

### Embeddings (`embeddings.py`)

- Model: `gemini-embedding-001`, truncated to `EMBEDDING_DIMENSIONS=768`
  (natively 3072-dim) via `output_dimensionality`.
- **task_type is asymmetric and this matters a lot**: chunks embed with
  `RETRIEVAL_DOCUMENT` (in `embed.py`), search queries embed with
  `RETRIEVAL_QUERY` (in `search.py`). Verified empirically (not just per
  docs) that mixing them up doesn't crash — it silently inflates similarity
  scores by ~0.03-0.05 and can flip close rankings. Nothing in this module
  defaults task_type; every caller passes it explicitly so a mix-up can't
  hide.
- **Manual re-normalization is required.** Google's docs: gemini-embedding-001
  only pre-normalizes at the full 3072 dims; a truncated 768-dim vector has
  to be divided by its own L2 norm by hand (`embeddings._normalize`) or
  cosine similarity is subtly wrong. Every vector's dimension is asserted
  before it's allowed near Mongo.
- **Free tier rate limit is tokens-per-minute, not requests-per-minute** —
  found empirically: a 10-chunk call (~8K tokens) succeeded, a 50-chunk
  call (~45K tokens) immediately hit 429. `embed.py` batches chunks by a
  20K-token budget (not a fixed chunk count) and pauses 15s between
  batches, on top of `embeddings.py`'s reactive retry-with-backoff on 429.
- **A 429 does not fall back to per-item retries** — that's reserved for
  genuinely per-item failures (a malformed single input). A 429 is an
  account-level condition; retrying it 100x individually just hits the same
  limit harder, which is what happened the first time this ran for real.
- `embed.py` writes to Mongo incrementally, one token-budgeted batch at a
  time, not all at once at the end — a run that stops partway (rate limit,
  Ctrl-C) doesn't lose already-embedded chunks, and a rerun's "chunks
  missing `embedding`" query picks up exactly where it left off.
- **Oversized-chunk guard**: gemini-embedding-001 caps input at 2048
  tokens/text, and newer embedding models tend to truncate silently rather
  than error on an over-limit input — so this is a hard pre-check (via the
  same tiktoken cl100k_base encoder `chunk.py` uses, an approximation of
  Gemini's real tokenizer but good enough as a trip-wire) before any API
  call, not a try/except. Finding one aborts the whole run — that's a
  Phase 1 chunking issue to fix at the source, not something to paper over
  here.

### Vector index (`vector_index.py`)

- Name: `chunks_vector_index`, on `finanalyzer.chunks`.
- Definition: vector field `embedding` (768 dims, cosine similarity) +
  `ticker` declared as a filter field (so `search()`'s optional ticker
  argument can pre-filter — Atlas requires filter fields declared up front).
- **M0 restricts driver-level index *management*** (`create_search_index()`,
  `list_search_indexes()`) to M10+ clusters, confirmed both via MongoDB's
  own community forum and by testing directly — though empirically, on this
  project's actual cluster, driver-level creation worked fine (the M0
  restriction may not be universal, or Atlas has relaxed it since). Either
  way, `ensure_vector_index()` tries the driver path first and falls back
  to printing the index JSON + manual Atlas UI steps if that's rejected.
- **Readiness is never trusted from a keypress or a listed status field**
  (list_search_indexes() being off-limits on M0 makes that unreliable
  anyway) — `wait_until_ready()` runs an actual trivial `$vectorSearch`
  probe query (using a real embedded chunk's own vector, so it costs zero
  extra Gemini calls) in a backoff retry loop, capped at ~2 minutes. Missing
  index = `OperationFailure`, still-building index = zero hits, ready index
  = a hit — all three are treated explicitly.
- Index creation is sequenced *after* embedding finishes, never before —
  the index shouldn't be built against a field that's still being
  populated, and the readiness probe needs a real vector to already exist.

### Search (`search.py`)

- `search(query, ticker=None, limit=5)` — matches this exact signature so
  it's ergonomic to call directly; `db`/`settings` are keyword-only
  overrides for reusing one connection across many calls (`eval.py`) or
  injecting a fake (tests).
- `$vectorSearch` is always pipeline stage 0. `numCandidates = limit * 15`
  (Atlas's own guidance is roughly 10-20x `limit`).
- Score is projected via `{"$meta": "vectorSearchScore"}`.

### Retrieval eval (`eval.py`, `eval_data.py`)

- 15 questions, grounded in real chunks (read out of the database, not
  invented), spread across AAPL/MSFT/GOOGL and Business/Risk
  Factors/MD&A. Each has an `expected_phrase` verified as an exact
  substring of its source chunk before being approved.
- Deliberately searches the *whole* corpus, no ticker filter, even though
  each question's correct company is known — a real user question doesn't
  pre-declare which company it's about, so filtering by the known-correct
  ticker would test something easier than real usage.
- **Result on the full corpus (5 tickers, 304 chunks) at the time this was
  last run: 12/15 (80%)**, accepted as-is. 2 of the 3 misses were near-ties
  (correct chunk ranked #6, score within 0.001-0.004 of the #5 cutoff) —
  not a real problem, just noise inherent to how similar 10-K risk-factor
  boilerplate reads. The third (Alphabet "Other Bets") was a genuine miss
  at rank #11, diagnosed as a **chunking granularity** issue: that chunk's
  ~1000 tokens bundle one relevant sentence together with several
  paragraphs of unrelated general/AI-competition risk, diluting the pooled
  embedding. Deliberately not treated as a bug to fix by changing chunk
  size — that's a real Phase 1 tradeoff, revisit only if a future eval run
  shows a worse or similar pattern.

### Testing tradeoff: no live vector-search integration test

M0 caps out at **3 total search indexes**; spending a second one on
`finanalyzer_test.chunks` just for tests wasn't worth it. `search()`'s
tests (`tests/test_search.py`) mock `db.chunks.aggregate()` with a canned,
already-descending-score result set — testing that *our* query-building and
result-mapping code is correct, not re-proving Atlas's own documented
`$vectorSearch` ordering guarantee. `embed_missing_chunks()`'s tests use a
fake embedder (no Gemini calls) and no-op `ensure_index_fn`/`wait_ready_fn`
(no real Atlas index management) against real `finanalyzer_test` — those
don't need a vector index at all, just chunk documents.

## Running things

```
uv run ingest AAPL MSFT GOOGL           # the Phase 1 CLI
uv run embed                            # embed any chunks missing vectors (Phase 2)
uv run search "how does Apple describe supply chain risk?"   # semantic search
uv run eval                             # recall@5 over the approved eval questions
uv run pytest                           # tests (needs .env configured — see README)
```
