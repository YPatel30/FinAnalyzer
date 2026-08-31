# FinAnalyzer

Phase 1: pull SEC 10-K filings into MongoDB. Phase 2: embed the chunks
(Gemini API) and search them with MongoDB Atlas Vector Search. Phase 3:
generate a real, cited answer from those retrieved passages — this is
where it becomes RAG. No API server, no MCP server — those are later
phases. See [CLAUDE.md](CLAUDE.md) for the full design notes and
constraints.

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

## Tests

```
uv run pytest
```

Needs `.env` configured (same file as above) — the idempotency tests write
to a real `finanalyzer_test` database on your Atlas cluster (never
`finanalyzer`) and clean up after themselves. No test makes a real SEC or
Gemini network call — embedding-related tests use a fake embedder.

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
  search.py         # Chunk dataclass + search()
  eval.py           # recall@5, groundedness, refusal-rate eval
  eval_data.py      # the approved eval questions + expected phrases
  eval_checkpoint.py  # eval's own checkpoint file + call counter
  refusal_data.py   # the 5 approved refusal-set questions
  cli_embed.py / cli_search.py / cli_eval.py / cli_ask.py   # entry points
  core/
    prompts.py      # the answer + judge prompt templates
    models.py       # Answer, Source, GroundednessVerdict
    citations.py    # [n] citation parsing
  providers/        # Provider abstraction (currently: Groq) for generation
  generation.py     # generate_answer() / judge_groundedness()
  ask.py            # ask() -- the Phase 3 core function
tests/
```
