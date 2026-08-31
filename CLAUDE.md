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
- **Phase 4+ (not started, don't build yet)** — the eventual system adds:
  historical financial data, a FastAPI REST API, and an MCP server. None
  of this exists yet. If a task seems to need one of these to finish
  something earlier, stop and say so rather than building it.

## Explicitly out of scope until told otherwise

FastAPI, an MCP server, reranking, streaming responses, multi-turn
conversation or chat history, tool calling / agents, structured numeric
data (Phase 3 demonstrated *why* this is needed — see the fiscal-2023
numeric-trap note below — but doesn't build it), Docker, a frontend, async
code, authentication, caching layers (eval.py's checkpoint file is scoped
narrowly to its own resumability, not a general cache — see below),
hybrid/keyword search, LangChain generally (the one exception already in
use: `langchain-text-splitters`, for chunking only).

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

## Running things

```
uv run ingest AAPL MSFT GOOGL           # the Phase 1 CLI
uv run embed                            # embed any chunks missing vectors (Phase 2)
uv run search "how does Apple describe supply chain risk?"   # semantic search
uv run ask "how does Apple describe supply chain risk?"      # cited, grounded answer (Phase 3)
uv run ask "..." --ticker MSFT --k 3 --show-prompt           # print the exact prompt before sending it
uv run eval                             # recall@5, groundedness %, refusal rate
uv run pytest                           # tests (needs .env configured — see README)
```
