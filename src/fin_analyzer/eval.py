"""Retrieval evaluation: recall@5 over the 15 approved questions.

recall@5 = the fraction of questions where at least one of the top 5
search() results contains that question's expected_phrase — i.e. did a
genuinely relevant passage make it into what a user would actually see.

Deliberately searches the *whole* corpus (no ticker filter), even though
each question's correct company is known — a user asking a real question
doesn't pre-tell the system which company they mean, so filtering by the
known-correct ticker would test something easier than real usage. This is
retrieval-only: no LLM, no generation, matching Phase 2 scope. If recall is
poor, the fix is chunking, query phrasing, or task_type — not reaching for
a language model to paper over bad retrieval.
"""

from dataclasses import dataclass

from fin_analyzer.config import Settings, get_settings
from fin_analyzer.db import get_db
from fin_analyzer.eval_data import EVAL_QUESTIONS
from fin_analyzer.search import Chunk, search


@dataclass
class EvalResult:
    question: dict
    hit: bool
    results: list[Chunk]


def run_eval(settings: Settings | None = None, db=None, limit: int = 5) -> list[EvalResult]:
    settings = settings or get_settings()
    db = db if db is not None else get_db(settings)

    results = []
    for q in EVAL_QUESTIONS:
        top_results = search(q["question"], limit=limit, db=db, settings=settings)
        hit = any(q["expected_phrase"] in chunk.text for chunk in top_results)
        results.append(EvalResult(question=q, hit=hit, results=top_results))
    return results
