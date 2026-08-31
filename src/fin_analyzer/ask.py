"""ask() — the Phase 3 core function: retrieve -> build prompt -> generate
-> parse citations -> Answer.

retrieve_and_build_prompt() is split out (rather than inlined in ask())
specifically so cli_ask.py's --show-prompt can print the exact prompt
without also spending a generation call — it calls this directly, then
separately calls ask() for the real answer. That means --show-prompt costs
one extra (cheap) embedding call versus not using the flag, in exchange for
ask() keeping the exact signature asked for for (question, ticker, k) with
no debug-only parameters bolted on.
"""

from fin_analyzer.config import Settings, get_settings
from fin_analyzer.core.citations import parse_cited_numbers
from fin_analyzer.core.models import Answer, Source
from fin_analyzer.core.prompts import CONTEXT_CHUNK_TEMPLATE, REFUSAL_MESSAGE, SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE
from fin_analyzer.db import get_db
from fin_analyzer.generation import generate_answer
from fin_analyzer.search import Chunk, search


def retrieve_and_build_prompt(
    question: str,
    ticker: str | None = None,
    k: int = 5,
    *,
    db=None,
    settings: Settings | None = None,
) -> tuple[str | None, list[Chunk]]:
    """Runs retrieval and assembles the user-turn prompt. Returns
    (user_prompt, chunks) — user_prompt is None if nothing was retrieved
    (nothing to send, caller should refuse without calling Gemini at all).
    """
    settings = settings or get_settings()
    db = db if db is not None else get_db(settings)

    chunks = search(question, ticker=ticker, limit=k, db=db, settings=settings)
    if not chunks:
        return None, []

    context_block = "\n\n".join(
        CONTEXT_CHUNK_TEMPLATE.format(n=i + 1, ticker=chunk.ticker, filing_date=chunk.filing_date, text=chunk.text)
        for i, chunk in enumerate(chunks)
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(context_block=context_block, question=question)
    return user_prompt, chunks


def ask(
    question: str,
    ticker: str | None = None,
    k: int = 5,
    *,
    db=None,
    settings: Settings | None = None,
) -> Answer:
    settings = settings or get_settings()
    db = db if db is not None else get_db(settings)

    user_prompt, chunks = retrieve_and_build_prompt(question, ticker=ticker, k=k, db=db, settings=settings)

    if not chunks:
        # Nothing retrieved at all -- refuse without spending a generation
        # call on it. Same exact refusal message the model itself is
        # instructed to use, so nothing downstream can tell "the model
        # refused" apart from "there was nothing to even ask about."
        return Answer(question=question, answer=REFUSAL_MESSAGE, sources=[], context_used=0)

    answer_text = generate_answer(SYSTEM_INSTRUCTION, user_prompt, settings)
    cited_numbers = parse_cited_numbers(answer_text)

    sources = [
        Source(
            ticker=chunk.ticker,
            filing_date=chunk.filing_date,
            chunk_index=chunk.chunk_index,
            score=chunk.score,
            text=chunk.text,
            cited=(i + 1) in cited_numbers,
        )
        for i, chunk in enumerate(chunks)
    ]

    return Answer(question=question, answer=answer_text, sources=sources, context_used=len(chunks))
