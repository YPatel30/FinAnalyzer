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
- **Phase 3 (done)** — generation: retrieval + a real LLM call that
  answers with inline citations, this is where it becomes RAG. Answer
  generation and the groundedness judge run on **Groq**, not Gemini (see
  below for why) — Gemini stays embeddings-only. `uv run ask "..."`,
  extends `uv run eval` with groundedness % and refusal rate. See below.
- **Phase 4 (done)** — a FastAPI REST API, a thin layer over the logic
  above. `uv run serve`. See below.
- **Phase 5 (done)** — an MCP server exposing the same three capabilities
  (list companies, search, ask) to a model instead of a programmer — the
  same logic, wrapped twice, on purpose: building it twice is what shows
  where REST and MCP actually differ. `uv run mcp-serve`, or
  `mcp dev src/fin_analyzer/mcp_server.py` for the Inspector. See below,
  and the README's REST-vs-MCP comparison.
- **Phase 6+ (not started, don't build yet)** — the eventual system adds
  historical financial data. None of this exists yet. If a task seems to
  need it to finish something earlier, stop and say so rather than
  building it.

## Explicitly out of scope until told otherwise

New capabilities beyond what Phase 5 exposes, MCP resources/prompts
primitives (tools only), reranking, streaming responses (SSE or MCP
progress notifications — see below for why these don't help the case
Phase 4's job-polling solves), multi-turn conversation or chat history,
tool calling / agents, structured numeric data (Phase 3 demonstrated *why*
this is needed — see the fiscal-2023 numeric-trap note below — but doesn't
build it), Docker, deployment, a frontend, async code (including an async
database driver — sync pymongo throughout, even in the API and MCP server
— see below), authentication (Phase 4's README has two sentences on where
it would hook in, not built), caching layers (eval.py's checkpoint file is
scoped narrowly to its own resumability, not a general cache; same for the
API — no response caching), rate limiting, hybrid/keyword search,
LangChain generally (the one exception already in use:
`langchain-text-splitters`, for chunking only).

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
- **Raises `TickerNotFound`/`IndexNotReady` itself** (Phase 4) — see the
  Phase 4 section below for why ticker validation lives here and not in
  the API route handler.
- `$vectorSearch` is always pipeline stage 0. `numCandidates = limit * 15`
  (Atlas's own guidance is roughly 10-20x `limit`).
- Score is projected via `{"$meta": "vectorSearchScore"}` — in its own
  `$project` immediately after `$vectorSearch`, before the `$lookup` below,
  since that metadata isn't reliably readable in a later stage.
- `Chunk` also carries `filing_date`, joined from `filings` via a `$lookup`
  on `filing_id` (Phase 3 added this — ask()'s prompt needs it). Chose
  `$lookup` over denormalizing `filing_date` onto each chunk at ingest
  time, even though the latter is arguably the better MongoDB instinct
  (filing_date is immutable — no update-anomaly risk, and `search`/`ask`
  is the hot path while `ingest` is rare, so paying the join cost on every
  query to save nothing on writes is backwards at real scale). Reasonable
  for now because there's no real query volume yet and `filing_id` staying
  the only foreign key keeps the door open cheaply if Phase 4+ wants more
  filing-level fields later. Revisit if this ever sees real traffic.

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

## Phase 3 architecture

```
src/fin_analyzer/
  core/
    prompts.py     # SYSTEM_INSTRUCTION, REFUSAL_MESSAGE, context/user templates,
                    #   JUDGE_SYSTEM_INSTRUCTION/JUDGE_PROMPT_TEMPLATE
    models.py      # Answer, Source, GroundednessVerdict (Pydantic)
    citations.py   # parse_cited_numbers() -- pure, regex-based [n] extraction
  providers/
    base.py        # Provider ABC (generate, generate_structured, list_model_names),
                    #   QuotaExhaustedError
    groq_provider.py  # the only concrete Provider right now
    registry.py    # get_provider(name, settings) -> Provider
    validate.py    # validate_configured_models() -- startup check
  generation.py    # generate_answer() / judge_groundedness() -- role-aware,
                    #   settings pick provider+model per role
  ask.py           # retrieve_and_build_prompt() + ask() -- the core function
  eval_checkpoint.py  # JSON checkpoint + CallCounter for `uv run eval`
  refusal_data.py  # the 5 approved refusal-set questions
  cli_ask.py       # `ask` entry point, --show-prompt
```

### Why generation is on Groq, not Gemini (`providers/`)

- The spec asked for `gemini-2.5-flash`. It returned 404 on this project's
  API key — "no longer available to new users" — discovered by actually
  calling it, not from docs (Google's docs still list it as available).
  Google's own error message named `gemini-3.6-flash` as the replacement;
  verified working and adopted.
- `gemini-3.6-flash` then turned out to cap at **20 `generate_content`
  requests per day** on the free tier — a number that appears nowhere in
  public docs, only in the 429 error payload
  (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quotaValue: 20`)
  once hit. A full eval run needs ~35 generation calls; this cannot
  complete in one day on that model, full stop, no amount of backoff fixes
  a daily cap.
- Rather than hunt for a third Gemini model name (the user's own
  assessment: "the published free-tier numbers conflict across sources and
  Google no longer documents them" — verifying more model names would just
  be guessing again), generation moved to a different provider entirely:
  **Groq**. Embeddings stay on Gemini, untouched — that quota is separate
  and has worked reliably throughout.
- This is why `providers/` exists as a real abstraction (`Provider` ABC +
  a registry keyed by a config string), not a second hardcoded call site:
  having been bitten twice by a hardcoded model name at the same place,
  adding or swapping a provider from here on should be one class plus one
  registry line, never a refactor of `ask.py`/`eval.py`.
- **Startup validation is not optional.** The two models this project
  actually planned to use (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`)
  turned out not to exist on this Groq key at all (`list_model_names()`
  confirmed it directly) — Groq's own lineup had moved on. Verified
  replacements, confirmed via the account's own `console.groq.com` billing
  page (free tier, no card) and a live rate-limit-header read (not
  trusted from any blog): **`openai/gpt-oss-120b`** (answers) /
  **`openai/gpt-oss-20b`** (judge) — 1K requests/day, 30/min, 8K tokens/min,
  200K tokens/day each, comfortably above what an eval run needs.
  `cli_ask.py`/`cli_eval.py` both call `validate_configured_models()` at
  startup so a bad model name fails immediately and loudly, not mid-eval
  after quota is already spent — exactly how the Gemini situation was
  first discovered.
- Groq's strict JSON-schema mode (used for the judge's structured output)
  requires `additionalProperties: false` on every object in the schema —
  Pydantic's `model_json_schema()` doesn't set that by default. Found via
  a real 400 from the API; `groq_provider.py`'s `_strict_schema()` walks
  the schema (including `$defs`, for nested models) and adds it.
- `QuotaExhaustedError` (`providers/base.py`) is how a provider tells
  callers "retrying this in-process won't help" — `GroqProvider`
  distinguishes a brief per-minute rate limit (worth a short backoff, read
  from the real `retry-after` response header) from a long-window one
  (>120s away per that same header) by the header value itself, not by
  guessing from the error message text.

### The prompt (`core/prompts.py`)

- Numbered context block: `[n] {ticker} (filed {filing_date}): {text}` —
  full chunk text, never truncated (truncation is a display-time choice,
  see `Source.text` below).
- The system instruction frames scope positively ("you answer questions
  about the filings provided... you have no knowledge beyond them") and
  gives a decision procedure (check company match, topic match, and
  tense/forecast — treat any failure as "not in the context") rather than
  a bare prohibition, then defines **one exact refusal string**
  (`REFUSAL_MESSAGE`) the model must use verbatim on any of those checks
  failing. One string, reused everywhere: `eval.py`'s refusal-rate check
  string-matches against it instead of spending a judge call per refusal
  question, and `ask()` returns the same literal string when retrieval
  comes back empty (see below) — nothing downstream can tell "the model
  refused" apart from "there was nothing to even ask about."
- Citation format is `[2][4]` (adjacent brackets, no comma) specifically so
  parsing is a trivial regex. `core/citations.py`'s `CITATION_RE` also
  matches fullwidth brackets (`【2】`) — a real model response used them
  instead of ASCII ones, which silently zeroed out every citation match
  until this was found and fixed.
- The groundedness judge reuses `CONTEXT_CHUNK_TEMPLATE` to build its own
  context block from the same `Answer.sources`, so it's checking the
  answer against literally the same numbered excerpts it was built from.

### `ask()` (`ask.py`)

- `retrieve_and_build_prompt()` is split out from `ask()` specifically so
  `cli_ask.py --show-prompt` can print the exact prompt without spending a
  generation call — it calls this directly, then separately calls `ask()`
  for the real answer (one extra cheap embedding call when the flag is
  used, in exchange for `ask()` keeping the exact `(question, ticker, k)`
  signature asked for, no debug-only parameters).
- Empty retrieval (`search()` returns nothing) short-circuits straight to
  `REFUSAL_MESSAGE` with `sources=[]`, `context_used=0` — no generation
  call spent on a question with nothing to answer from.
- `Source.text` is the **full** chunk text, not a truncated excerpt —
  the groundedness judge needs the whole passage; a short display excerpt
  is `cli_ask.py`'s choice at print time, the same pattern `cli_search.py`
  already used for `Chunk.text`.

### Eval extensions (`eval.py`, `eval_checkpoint.py`, `refusal_data.py`)

- **Groundedness**: for each of the same 15 `eval_data.py` questions,
  `ask()` generates a real answer, then a *second, independently
  configured* model (`judge_model` — a genuinely different model from
  `generation_model`, not just a different role prompt on the same one)
  checks every factual claim against the same context excerpts. This is a
  model judging a model — printed as a caveat every time, not proof.
- **Refusal set** (`refusal_data.py`): 5 questions spanning company-not-
  ingested, forward-looking, and out-of-domain. Deliberately *not* named
  in the system instruction (would be teaching to the test) — a generic
  decision procedure has to catch them on its own.
- **Score-floor investigation**: before building the refusal mechanism,
  logged top-1 `search()` scores for all 15 answerable + 5 refusal
  questions. The two distributions overlap substantially (answerable
  0.8357-0.8944, refusal 0.7665-0.8632) — forward-looking questions about
  an in-corpus company score in the middle of the answerable range (they
  *are* topically relevant, just not temporally answerable), and only the
  wildly-out-of-domain case is a clean outlier. Conclusion: no score-floor
  short-circuit — it would only catch the easiest case and risks false-
  positive refusals on real low-scoring-but-answerable questions.
- **Checkpointing** (`eval_checkpoint.py`): a full run spends ~35
  generation calls; a crash or quota wall partway through previously lost
  every in-memory result. Each question's result (recall, answer, judge
  verdict) is persisted to `.cache/eval_checkpoint.json` the moment it's
  computed — plain JSON of built-in types, not pickle, so it doesn't
  depend on today's class definitions matching tomorrow's. A rerun skips
  anything already checkpointed. Scoped narrowly to this script's own
  resumability across runs — not a general cache for `ask()` in normal
  use, which stays out of scope. `QuotaExhaustedError` stops the affected
  loop cleanly (whatever completed is kept and reported) instead of a raw
  traceback destroying an otherwise-successful partial run.
- `CallCounter` prints every real (non-cached) API call as it happens and
  is shared across all three eval stages in one run, so the total reflects
  the whole invocation.

### Results at the time this was last run (5 tickers, 304 chunks)

- **recall@5: 12/15 (80%)** — unchanged from Phase 2, as expected
  (retrieval logic didn't change).
- **Groundedness: 13/15 (87% of judged answers)**. Both failures line up
  exactly with 2 of the 3 recall@5 misses (the AAPL culture and GOOGL
  financing near-ties) — a coherent, non-contradictory story: when
  retrieval doesn't surface the right chunk, the model sometimes answers
  anyway from tangential context (or, in the AAPL culture case, appears to
  have answered from general knowledge despite instructions, producing a
  claim with zero textual support), and the judge correctly flags it.
  The third recall miss (GOOGL "Other Bets", the more severe rank-11 one)
  did *not* also fail groundedness — the retrieved chunks included a
  different, legitimately-relevant passage (the segment-reporting note)
  that recall@5's single-expected-phrase check doesn't count as a hit but
  that genuinely supports a correct answer. Recall@5 is a precision proxy,
  not the same thing as "was the final answer actually right."
- **Refusal rate: 5/5 (100%)** — no system-instruction tightening needed.
- **The numeric trap** (`ask "What was Apple's total revenue in fiscal
  2023?"`): the model refused — correctly, but the *reason* is the real
  finding. Retrieval found exactly the right chunk (ranked #1, literally
  "The following table shows net sales by segment for 2025, 2024 and
  2023"), but the actual figures were never extracted into text at all —
  they lived only in the `<table>` Phase 1's `extract.py` deliberately
  drops. This is sharper than "embeddings are fuzzy and might retrieve the
  wrong passage": a prose-oriented chunking pipeline structurally cannot
  represent tabular numeric data as text, no matter how good retrieval or
  generation get. Not fixed, per instruction — this is the argument for
  routing numeric questions to real structured queries in a later phase,
  not an incremental improvement on this one.

## Phase 4 architecture

```
src/fin_analyzer/
  core/
    exceptions.py    # NEW -- TickerNotFound, QuotaExhausted (moved here from
                      #   providers/base.py), IndexNotReady, JobNotFound. Plain
                      #   Python, no fastapi import -- ever, that's the hard rule.
  companies.py        # list_companies(), ticker_exists() -- nothing read the
                       #   companies collection back before this
  jobs.py              # the /ingest job state machine -- a `jobs` Mongo
                        #   collection, not an in-memory dict (see below)
  health.py            # GET /health's logic: Mongo ping + a single-shot
                        #   vector-index probe
  api/
    app.py             # FastAPI(), lifespan (Mongo client setup/teardown),
                        #   CORS, exception handlers, request-ID logging
    dependencies.py    # get_db_dependency()/get_settings_dependency() --
                        #   the shared Mongo client from app.state
    schemas.py         # ALL request/response models -- never core/models.py
                        #   or search.py's Chunk directly
    routes.py          # all 6 routes; `def` not `async def` throughout
  cli_serve.py         # `uv run serve` entry point
```

`api/` is the *only* place allowed to import `fastapi` — not `core/`, not
`providers/`, not any existing business-logic module (`search.py`, `ask.py`,
`ingest.py`, the three new ones above). One-directional dependency: `api/`
imports from everything else, nothing else imports from `api/`. This is
what lets Phase 5's MCP server wrap the same logic without untangling any
HTTP concerns out of it first.

### The routes

```
GET  /health                  liveness + Mongo connectivity + index status
GET  /companies               what's ingested
POST /search                  retrieval only, no generation
POST /ask                     full RAG answer with citations
POST /ingest                  kick off ingestion -> 202 + job_id
GET  /ingest/{job_id}         poll job status
```

`/search` exists separately from `/ask` on purpose — retrieval without
generation is genuinely useful, cheaper, and makes the retrieve-vs-generate
split visible in the API surface, same as the CLI already does with
`search`/`ask` as separate commands.

### Exception → status mapping (`core/exceptions.py` → `api/app.py`)

| Exception | Raised where | Status |
|---|---|---|
| `TickerNotFound` | `search()` itself (see below — not the route) | 404 |
| `JobNotFound` | `jobs.get_job()` | 404 |
| `QuotaExhausted` | `GroqProvider._call()`, with `retry_after_seconds` when Groq's own header gave one | 429 (+ `Retry-After` header if known) |
| `IndexNotReady` | `search()`, wrapping a raw pymongo `OperationFailure` | 503 |
| (unhandled) | anywhere | 500, traceback logged server-side with the request id, never returned to the client |
| Pydantic validation | automatic | 422 |

All five are registered as `@app.exception_handler(...)` in `app.py` —
routes never `try`/`except` any of these themselves, keeping them thin.

### Ticker validation moved into `search()`, not the route handler

Originally proposed as a route-level check; overturned during review, and
rightly — "this company isn't in the corpus" is a domain fact, equally
true for the CLI and for Phase 5's MCP tools, not an HTTP concern.
Validating only in a route handler would've left the CLI's pre-existing
silent-empty-results bug in place (`uv run search "..." --ticker FAKE`
used to just print "No results.") and meant Phase 5 either duplicates the
check or reintroduces the same bug. Fixed at the source instead:
`search()` raises `TickerNotFound` directly when an explicit `ticker` is
given and isn't in `companies`; `ask()` inherits this for free since it
calls `search()` internally; both CLIs (`cli_search.py`, `cli_ask.py`) now
catch it and print a clean message instead of a raw traceback.
`ticker=None` ("search everything") is unaffected — the check only runs
when a ticker is explicitly supplied.

This is a **different mechanism** from Phase 3's refusal behavior, and the
two can coexist without conflict: `ask(question, ticker="TSLA")` raises
`TickerNotFound` before any retrieval happens; `ask("what are Tesla's risk
factors?")` with no ticker filter still runs retrieval (finds nothing
genuinely relevant), and the system instruction's refusal path handles it
as before. One is a hard precondition check on an explicit parameter, the
other is the model reasoning about retrieved content — they don't overlap.

### `search()` also now raises `IndexNotReady`

The real `$vectorSearch` call is wrapped in `try/except OperationFailure`,
re-raised as `IndexNotReady` — so the business-logic layer never lets a
raw pymongo exception escape to a caller that shouldn't need to know
pymongo exists (the API layer, or a future MCP tool).

### `jobs.py`: why a `jobs` Mongo collection, not an in-memory dict

1. **Multiple worker processes.** The moment this API runs with more than
   one worker (`uvicorn --workers 4`, or several instances behind a load
   balancer — the first thing anyone does to handle real concurrent
   traffic), each worker has its own separate memory. `POST /ingest`
   landing on worker A writes to worker A's dict; `GET /ingest/{job_id}`
   landing on worker B (ordinary round-robin routing) finds nothing —
   a false 404 for a job that's actually running fine. Not an edge case;
   the default failure mode the instant you scale past one process.
2. **Process restarts.** A crash or redeploy while a job is in flight
   wipes the dict entirely. The client that started the job can't tell
   "it finished before the restart" from "it never ran" — both look like
   a 404. Mongo survives the process restarting.
3. **Orphan detection.** A job document stuck in `"running"` with an old
   `started_at` and no progress is a real, observable state — the process
   that was running it died mid-work. A dict can't represent this at all:
   it vanishes with the process, so there's no "stuck" state left to
   observe, only silence. `started_at` is stored on every job for exactly
   this reason; actually reaping stale jobs is future work, not built here.

Job id is a fresh `uuid4`, not Mongo's own `_id` — decouples the public API
contract from the storage layer. `jobs.job_id` has a unique index (every
poll looks it up; without one that's a full collection scan, worse as more
ingests accumulate). One job can request several tickers; one ticker
failing doesn't fail the others — `results` carries per-ticker status/
chunk_count/error, and overall job status is `failed` if *any* ticker
failed, `succeeded` only if *all* did (mirrors `ingest.py`'s CLI, which
already continues past a per-ticker failure). `run_ingest_job()` reuses
`ingest_ticker()` (Phase 1) completely unchanged — it's only a state-
tracking wrapper around it.

### `def`, never `async def`

Every route handler is plain `def`. This project uses sync `pymongo`
throughout (a deliberate choice from Phase 1 — see above), including here.
An `async def` handler runs directly on the single event loop thread; a
blocking call inside it (any pymongo call, or the Gemini/Groq HTTP calls
inside `search()`/`ask()`) stalls that one thread every other concurrent
request also depends on, and the whole server serializes under load —
*silently*, since a single request in a manual test looks completely fine
with nothing else on the loop to be blocked by it. A plain `def` handler
runs in FastAPI's threadpool automatically, off the event loop, so a
blocking call blocks only its own thread — the correct default for a sync
driver, not a shortcut. The reasoning is written as a comment at the top
of `routes.py` too, so it survives a future "helpful" `async def` edit.

**Verified, not just asserted**: 10 concurrent `POST /search` requests vs.
10 sequential, same query, against the running server —
sequential 3.55s total (2.82 req/s), concurrent 0.77s total (12.91 req/s),
a 4.58x speedup. Requests measurably don't serialize.

### Lifespan, CORS, request logging

- `app.py` uses the `lifespan` context manager (not the deprecated
  `on_event` hooks) to open **one** `MongoClient` for the process's entire
  lifetime and store it on `app.state` — `db.get_client()` opens a fresh
  connection every call, correct for a short-lived CLI invocation, wrong
  for a long-running server making that call on every request.
- CORS allows `localhost`/`127.0.0.1` on any port (`allow_origin_regex`) —
  no specific frontend or port chosen yet.
- Every request gets a `uuid4` id, set on `request.state` before the route
  runs and returned as an `X-Request-ID` response header; every log line
  for that request (including the 500 handler's traceback log) carries the
  same id, so a client-reported failure can be tied to one specific
  server-side call rather than "something failed around 2:14pm".

### Auth — not built, see README

Two sentences in `README.md` on what would be added (an API-key or JWT
dependency in `api/dependencies.py`, applied per-route) and why that's the
right hook point (keeps the check itself out of `core/`, same principle as
the exception mapping). Not implemented.

## Phase 5 architecture

```
src/fin_analyzer/
  mcp_server.py    # the whole MCP server: MCPServer instance, 3 tools,
                    #   descriptions, model-readable error mapping
```

One file, not a subpackage like `api/` — 3 tools is proportionate to a
single module; `api/` earned the split with 6 routes plus schemas/
dependencies. Same hard rule as Phase 4: only this file is allowed to
import `mcp`. `core/`, `providers/`, `search.py`, `ask.py`, `companies.py`
don't know this server exists, same as they don't know `api/` exists.

### `FastMCP` doesn't exist in the installed `mcp` package

The spec named `FastMCP`; the installed version is `mcp==2.2.0`, where
`FastMCP` was renamed to `MCPServer` (`from mcp.server.mcpserver import
MCPServer`) — discovered via an actual `ImportError` naming the rename
directly, not assumed. Same pattern as Phase 3's Gemini model names:
verify against what's actually installed, don't trust a spec written
against an earlier version. The API is otherwise a straightforward rename
for what this project needs (`@mcp.tool(name=..., description=...)`,
`mcp.run(transport="stdio")`).

### Why `ingest` isn't an MCP tool

REST's `202 Accepted` + poll works because the *client* — a program or a
human — can hold a `job_id` in its own durable state and check back
whenever, with no assumption about how long that takes. MCP has no clean
equivalent: a tool call happens inside one bounded reasoning turn, and a
model has no durable state between calls except the conversation
transcript itself, which can be summarized, truncated, or just not there
in a new session. MCP does have progress notifications for long-running
calls, but they require the call to stay open — they let a model watch
work it's already blocked on, and do nothing for the walk-away case REST's
polling actually solves. Splitting `ingest` into `start`/`poll` tools
would just relocate the problem: the model would have to remember the job
id and decide, unprompted, to check back later, which isn't reliable.

### Tool descriptions are prompts, not documentation

A REST docstring is read by a developer who already decided to call the
endpoint. An MCP tool description is read by a model deciding *whether* to
call it, with no other context — so each one states one crisp positive
criterion for when to use *this* tool (not mutual hedging toward the
other), what its arguments mean in terms a model can supply (`ticker` must
be one of the ingested companies, named explicitly), and what comes back.

The first draft of `ask_filings`'s description included honest self-
critique — "answered by a separate, smaller model", "some nuance may not
survive" — all true, all things said *about* the tool during design, and
all wrong to put *in* the tool itself: a model reading that description
before deciding would route everything to `search_filings`, pre-deciding
the very comparison this phase exists to run empirically. Cut on review;
the refusal paragraph stayed, since that one *is* decision-relevant (it
tells the model the tool is honest about its limits, which isn't the same
as being honest about being the weaker tool).

The ticker list inside `search_filings`/`ask_filings`'s descriptions is
built once at server startup from a real `companies.list_companies()`
query (`_TICKERS` in `mcp_server.py`), not hardcoded — a literal list goes
stale the moment another ticker gets ingested. Inlining the list this way
is fine at 5 companies; past some size (a few dozen? a few hundred?) it
should be dropped from the description entirely in favor of just pointing
the model at `list_companies()` — the description itself should stay
short regardless of how large the corpus gets.

### Errors are read by a model, not a program

REST: `TickerNotFound` → `404`, a status code a program branches on. MCP:
the *same* exception (raised from the *same* place — `search()`/`ask()`
in `core`-adjacent business logic, no duplicate check) is caught in
`mcp_server.py` and turned into a sentence the model can act on:
`"TSLA is not in this corpus. Available companies: AAPL, FISV, GOOGL,
MSFT, NVDA."` — naming the fix, not just the failure. Same exceptions as
Phase 4 (`TickerNotFound`, `QuotaExhausted`, `IndexNotReady`), completely
different presentation, via `mcp_server._friendly_error()`.

### stdio: stdout is the protocol

Audited every module transitively imported by `mcp_server.py` for stray
`print()` calls (`grep -rn "print(" ...` across `search.py`, `ask.py`,
`companies.py`, `embeddings.py`, `generation.py`, `providers/`, `core/`) —
found two real ones, both on the actual call path (not hypothetical):
`embeddings.py`'s and `providers/groq_provider.py`'s 429-retry messages
were printing to stdout, which would corrupt the message stream the
moment a rate limit was hit mid-session. Both now print to `stderr`
explicitly. `vector_index.py` also has print statements in a module
`health.py` (and transitively this server) imports, but the specific
functions containing them (`ensure_vector_index`/`wait_until_ready`) are
never called by anything this server invokes — left alone, since fixing
them wasn't fixing a reachable bug.

### `.env` loading independent of the working directory

`config.py` used to resolve `.env` relative to the process's current
working directory — invisible for every CLI so far, since `uv run`
launches from the project root where a relative path happens to also
work. Claude Desktop launches the MCP server with its own working
directory (not necessarily this project's) and a minimal environment (no
inherited shell `PATH`). Fixed once, for every entry point: `.env`'s path
is now resolved from `config.py`'s own file location
(`Path(__file__).resolve().parent.parent.parent / ".env"`), never the cwd.
Verified by actually running a script from `/tmp` and confirming
`Settings()` still loads correctly.

### Claude Desktop wiring

Config: `~/Library/Application Support/Claude/claude_desktop_config.json`
— merge in (don't overwrite; this file has other real settings):

```json
"mcpServers": {
  "finanalyzer": {
    "command": "/Users/yashpatel/.local/bin/uv",
    "args": ["--directory", "/Users/yashpatel/Downloads/VSCodeProjects/FinAnalyzer", "run", "mcp-serve"]
  }
}
```

Both paths absolute, for the same minimal-environment reason as the `.env`
fix above. Requires a **full quit** of Claude Desktop (Cmd+Q), not just
closing the window — it only reads this file at process startup.

### Verifying it works without a browser

`mcp dev <file>` opens Inspector's browser UI, which can't be driven from
here. Verified instead with a real `mcp.client` session over stdio,
spawned with a deliberately minimal `env` (`PATH` only, no project-related
vars) and `cwd="/"` — the exact conditions Claude Desktop's launch creates
— confirming `list_tools()` returns all 3 tools and each one behaves
correctly (including the `TickerNotFound` → friendly-text path) under
those conditions, not just under a normal shell.

## Running things

```
uv run ingest AAPL MSFT GOOGL           # the Phase 1 CLI
uv run embed                            # embed any chunks missing vectors (Phase 2)
uv run search "how does Apple describe supply chain risk?"   # semantic search
uv run ask "how does Apple describe supply chain risk?"      # cited, grounded answer (Phase 3)
uv run ask "..." --ticker MSFT --k 3 --show-prompt           # print the exact prompt before sending it
uv run eval                             # recall@5, groundedness %, refusal rate
uv run serve                            # FastAPI server on :8000 -- /docs for the Swagger UI (Phase 4)
uv run mcp-serve                        # MCP server over stdio (Phase 5)
mcp dev src/fin_analyzer/mcp_server.py  # MCP Inspector, for interactive testing
uv run pytest                           # tests (needs .env configured — see README)
```
