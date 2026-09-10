# FinAnalyzer

Phase 1: pull SEC 10-K filings into MongoDB. Phase 2: embed the chunks
(Gemini API) and search them with MongoDB Atlas Vector Search. Phase 3:
generate a real, cited answer from those retrieved passages — this is
where it becomes RAG. Phase 4: a FastAPI REST API — a thin layer over the
same logic, nothing new underneath it. Phase 5: an MCP server exposing
the *same* three capabilities to a model instead of a programmer — same
logic again, wrapped a second time, deliberately, to see where the two
designs actually diverge (see the comparison below).
See [CLAUDE.md](CLAUDE.md) for the full design notes and constraints.

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
   - `GEMINI_API_KEY` — for Phase 2 embeddings. Get one at
     [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
     Not needed just to run Phase 1's `ingest`.
   - `GROQ_API_KEY` — for Phase 3 text generation (`ask`/`eval`'s
     groundedness judge). Get one at
     [console.groq.com](https://console.groq.com). Free tier, no card.
     Not needed for `ingest`/`embed`/`search`. If `GENERATION_MODEL`/
     `JUDGE_MODEL` ever 404 (providers change their lineups), see the
     one-liner in `.env.example` to list what's actually available to
     your key — `ask`/`eval` also check this at startup and fail loudly
     before spending anything if a configured model isn't available.

   `.env` is gitignored — it will never be committed. `.env`'s path is
   resolved from `config.py`'s own file location, not the process's
   working directory — matters for Phase 5, where Claude Desktop launches
   the MCP server with its own cwd and a stripped-down environment.

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

## Embed and search (Phase 2)

```
uv run embed
```

Embeds every chunk that doesn't already have a vector (Gemini
`gemini-embedding-001`, truncated to 768 dims, re-normalized), then creates
the `chunks_vector_index` Atlas Vector Search index if it doesn't exist yet
and waits until it's actually ready to query. Rerunning makes zero Gemini
API calls if nothing changed — it's idempotent, so it's safe to run again
after ingesting more tickers.

If your Atlas cluster doesn't allow creating the index via the driver (a
tier restriction on some plans), it prints the index's JSON definition and
step-by-step Atlas UI instructions instead — do that once by hand, then
rerun `uv run embed`.

```
uv run search "how does Apple describe supply chain risk?"
uv run search "what are Microsoft's main risks" --ticker MSFT --limit 3
```

Prints, for each result: a similarity score, ticker, chunk index, and the
first ~300 characters of text.

```
uv run eval
```

Runs 15 pre-approved test questions (see `src/fin_analyzer/eval_data.py`)
through `search()` and reports recall@5 — for how many questions did a
genuinely relevant chunk make it into the top 5 results. Prints each
failure individually, showing what came back instead, so retrieval quality
can be eyeballed rather than trusted blindly.

## Ask (Phase 3)

```
uv run ask "how does Apple describe supply chain risk?"
uv run ask "what does Microsoft say about AI competition?" --ticker MSFT
uv run ask "..." --k 3 --show-prompt
```

Retrieves the top `k` chunks, builds a numbered prompt from them, and asks
an LLM to answer *only* from that context, citing sources inline as
`[1][2]`. Prints the answer, then every source with its score and whether
it was actually cited versus just retrieved. `--show-prompt` prints the
exact system instruction and user prompt before it's sent — no description
of it, the literal text.

Answers only from the provided context, on purpose — if the filings don't
contain the answer (wrong company, a forward-looking question, or
something entirely off-topic), it says so explicitly rather than guessing.
It also can't answer precise numeric questions well: 10-K tables are
dropped during extraction (Phase 1), so a figure that only ever appeared
in a table was never captured as text for anything downstream to find.

```
uv run eval
```

Also reports **groundedness** (a second, different model checks whether
every claim in each answer is actually supported by its sources — a model
judging a model, useful signal, not proof) and **refusal rate** (5
questions the corpus provably can't answer — a good system refuses all 5).

## API server (Phase 4)

```
uv run serve
```

Starts the server at `http://127.0.0.1:8000`. Open
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the Swagger
UI — every route can be exercised from the browser there.

```
GET  /health                  liveness + Mongo connectivity + vector index status
GET  /companies               what's ingested
POST /search                  retrieval only, no generation
POST /ask                     full RAG answer with citations
POST /ingest                  kick off ingestion, returns a job id (202 Accepted)
GET  /ingest/{job_id}         poll that job's status
```

`/ingest` can't be synchronous — a real ingest takes minutes, longer than
most clients will wait. It returns immediately with a job id; poll
`GET /ingest/{job_id}` for `pending` → `running` → `succeeded`/`failed`,
with per-ticker detail (one ticker failing doesn't hide that others
succeeded). Job state lives in a `jobs` Mongo collection, not in server
memory — see `CLAUDE.md` for why that distinction matters the moment this
runs with more than one worker process.

Errors come back as real HTTP status codes, not tracebacks: 404 for an
unknown ticker or an unknown job id, 422 for a
malformed request body (automatic, from the request schemas), 429 (with a
`Retry-After` header when known) if the LLM provider's quota is exhausted,
503 if the vector index isn't ready to query, 500 for anything unexpected
(logged server-side with a request id, never sent to the client as a
traceback).

**Authentication is not built.** If added later, it would be an API-key or
JWT dependency in `src/fin_analyzer/api/dependencies.py`, applied per-route
via FastAPI's `Depends()` — kept out of `core/` and the other business-logic
modules entirely, the same principle behind keeping HTTP status mapping
out of them.

## MCP server (Phase 5)

```
uv run mcp-serve                          # run it directly, stdio transport
mcp dev src/fin_analyzer/mcp_server.py     # run it with MCP Inspector (test this first)
```

Exposes 3 tools — `list_companies`, `search_filings`, `ask_filings` — the
same capabilities as `/companies`, `POST /search`, and `POST /ask`, backed
by the exact same `search()`/`ask()`/`companies.list_companies()` calls.
No `ingest` tool; see `CLAUDE.md` for why REST's 202+poll pattern doesn't
have a clean MCP equivalent.

**Wiring it to Claude Desktop**: edit
`~/Library/Application Support/Claude/claude_desktop_config.json` (merge
in, don't overwrite — it likely has other settings already) and add:

```json
"mcpServers": {
  "finanalyzer": {
    "command": "/absolute/path/to/uv",
    "args": ["--directory", "/absolute/path/to/FinAnalyzer", "run", "mcp-serve"]
  }
}
```

Then **fully quit Claude Desktop (Cmd+Q) and relaunch** — it only reads
this file at startup, and closing the window isn't enough. Both paths
must be absolute: Claude Desktop launches the server with a minimal
environment, not your shell's.

Errors come back as sentences a model can act on, not status codes — e.g.
`"TSLA is not in this corpus. Available companies: AAPL, FISV, GOOGL,
MSFT, NVDA."` — the same exceptions the REST API maps to HTTP statuses,
presented completely differently for a completely different caller.

### REST vs. MCP: the same three capabilities, two different designs

Both wrappers call the exact same business logic — `search.py`'s
`search()`, `ask.py`'s `ask()`, `companies.py`'s `list_companies()` — with
zero duplicated retrieval or generation logic between them. That part
didn't change at all, and is the whole reason building the second wrapper
took an afternoon instead of a rewrite. What changed is everything about
how each interface presents that logic to whoever's calling it:

**Who reads the "documentation," and when.** A REST docstring/OpenAPI
schema is read by a developer who already decided to call the endpoint —
it mostly just needs to describe the request/response shape. An MCP tool
description is read by a model *deciding whether to call the tool at
all*, with no other context. This isn't a stylistic difference — it changed
what got written. The first draft of `ask_filings`'s description included
honest self-critique about the tool ("answered by a smaller model, some
nuance may not survive") that had to be cut, because a model reading that
before choosing a tool would route everything to `search_filings` — the
description had accidentally made a decision that belonged to the
experiment, not to the prompt.

**How an error reaches the caller.** REST: `TickerNotFound` → `404` +
`{"detail": "..."}` — a status code a program branches on, a body a human
reads incidentally. MCP: the *same* exception, caught in the *same* place,
becomes `"TSLA is not in this corpus. Available companies: AAPL, FISV,
GOOGL, MSFT, NVDA."` — because the caller here can read a sentence and act
on it, and a bare "not found" leaves it guessing. One exception hierarchy
(`core/exceptions.py`), two presentation layers, no duplicated logic.

**How an argument gets validated.** REST: Pydantic does strict, mechanical
schema validation before your code ever runs — a wrong type is an
automatic 422, no negotiation. MCP: there's a JSON schema too (derived
from the tool function's type hints), but a model choosing `ticker="TSLA"`
in the first place is a *prompting* outcome — did the description make
clear which values are valid — not a validation outcome. Getting an
argument right is a documentation problem here in a way it just isn't in
REST.

**Long-running work.** REST's `202` + poll works because the client — a
program, or a human at a keyboard — can hold a job id in durable state and
check back whenever. MCP has no equivalent: a tool call happens inside one
bounded reasoning turn, and a model has no durable memory between calls
except the conversation transcript, which can vanish. (MCP does have
progress notifications for a long call, but they require the call to stay
open — useful for watching work you're already blocked on, no help for
the walk-away case REST's polling actually solves.) This is a real
capability gap, not an oversight, and it's why `ingest` only exists on the
REST side.

**Transport-level failure modes that have nothing to do with API design.**
REST's constraints are HTTP semantics — status codes, CORS, request
bodies. MCP over stdio has a constraint one level lower that REST doesn't
have at all: stdout *is* the wire protocol, so one stray `print()`
anywhere in the entire import graph corrupts the message stream. Auditing
for this found two real, live ones (`embeddings.py`, `providers/
groq_provider.py`, both firing on an ordinary rate-limit retry) — a bug
class that simply can't happen to a REST server, where a stray print is
just an ugly log line.

**Startup assumptions.** The REST server is launched by your own shell —
normal `PATH`, normal working directory, no surprises. Claude Desktop
launches the MCP server with a stripped-down environment and an
unpredictable working directory, which forced two real fixes
(`config.py`'s `.env` resolution, the `--directory` flag in the launch
config) that have nothing to do with REST vs. MCP as *interfaces* — they're
about one runtime being a normal process you start and the other being a
process something else starts *for* you.

**What carried over cleanly, and what that's worth to whoever builds a
third wrapper (a second MCP server, or anything else) on this same
business logic:**
- Keep `core/` (and every business-logic module) free of the wrapper
  framework from the *first* wrapper, not the second — the tax for
  skipping this is invisible until you build the second one, which is
  exactly when it's expensive to pay retroactively.
- Write tool descriptions as decision prompts and get them reviewed before
  anything is wired up — a description can bias behavior before a single
  line of tool code runs.
- Audit the whole import graph for stdout writes before touching stdio,
  not just the server file — the real bugs here were two modules deep.
- Resolve config paths relative to the package, never the working
  directory, from the start — don't wait to discover this via a confusing
  failure in someone else's launcher.
- Decide up front which REST capabilities *don't* have a clean MCP
  equivalent (background jobs, here) rather than forcing a shape that
  doesn't fit.

## Tests

```
uv run pytest
```

Needs `.env` configured (same file as above) — the idempotency/API tests
write to a real `finanalyzer_test` database on your Atlas cluster (never
`finanalyzer`) and clean up after themselves. No test makes a real SEC,
Gemini, or Groq network call — `search()`/`ask()`/the SEC calls inside
ingest are all mocked at the boundary the corresponding test is checking.
One exception, documented in `test_mcp_server.py`: importing
`fin_analyzer.mcp_server` runs one real, read-only query against
production `finanalyzer` to build the tool descriptions' ticker list at
"server startup" — a one-time cost when that module is first imported,
not per test, and nothing writes to it.

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
  embeddings.py     # Gemini calls: batching, retry, normalization, dimension guard
  vector_index.py   # Atlas Vector Search index create/wait
  embed.py          # orchestrates: find missing embeddings -> embed -> write -> ensure index
  search.py         # Chunk dataclass + search() -- raises TickerNotFound/IndexNotReady
  eval.py           # recall@5, groundedness, refusal-rate eval
  eval_data.py      # the approved eval questions + expected phrases
  eval_checkpoint.py  # eval's own checkpoint file + call counter
  refusal_data.py   # the 5 approved refusal-set questions
  cli_embed.py / cli_search.py / cli_eval.py / cli_ask.py / cli_serve.py   # entry points
  core/
    prompts.py      # the answer + judge prompt templates
    models.py       # Answer, Source, GroundednessVerdict
    citations.py    # [n] citation parsing
    exceptions.py   # TickerNotFound, QuotaExhausted, IndexNotReady, JobNotFound
  providers/        # Provider abstraction (currently: Groq) for generation
  generation.py     # generate_answer() / judge_groundedness()
  ask.py            # ask() -- the Phase 3 core function
  companies.py      # list_companies(), ticker_exists()
  jobs.py           # the /ingest job state machine (a `jobs` Mongo collection)
  health.py         # GET /health's logic
  api/
    app.py          # FastAPI app: lifespan, CORS, exception handlers, request logging
    dependencies.py # shared Mongo client as a FastAPI dependency
    schemas.py       # API request/response models (separate from core/models.py)
    routes.py        # all 6 routes
  mcp_server.py     # the whole MCP server: 3 tools, descriptions, model-readable errors
tests/
```
