"""Three separate evaluations, run by `uv run eval`:

1. recall@5 (run_eval) — retrieval only, unchanged in method from Phase 2.
   If recall is poor, the fix is chunking, query phrasing, or task_type —
   not reaching for a language model to paper over bad retrieval.
2. Groundedness (run_groundedness_eval) — for each of the same 15
   questions, ask() generates a real answer, then a SECOND, independent
   call (a different model, judge_model — see config.py) checks every
   factual claim in that answer against the same context excerpts ask()
   used. This is a model judging a model: useful signal, not proof —
   cli_eval.py prints that caveat alongside the result, every time.
3. Refusal rate (run_refusal_eval) — 5 questions the corpus provably cannot
   answer (see refusal_data.py). A system that answers these confidently
   is worse than useless; this is the cheapest possible way to catch it.

All three are checkpointed (eval_checkpoint.py): each question's result is
persisted to disk the moment it's computed, so a run interrupted by quota
exhaustion doesn't lose completed work, and a rerun skips whatever's
already there instead of re-spending scarce generation quota on it. A
QuotaExhaustedError stops the affected loop cleanly — whatever completed
before the wall is kept and reported, not lost to a raw traceback.
"""

from dataclasses import asdict, dataclass

from fin_analyzer.ask import ask
from fin_analyzer.config import Settings, get_settings
from fin_analyzer.core.models import Answer, GroundednessVerdict
from fin_analyzer.core.prompts import (
    CONTEXT_CHUNK_TEMPLATE,
    JUDGE_PROMPT_TEMPLATE,
    JUDGE_SYSTEM_INSTRUCTION,
    REFUSAL_MESSAGE,
)
from fin_analyzer.db import get_db
from fin_analyzer.eval_checkpoint import CallCounter, load_checkpoint, save_checkpoint
from fin_analyzer.eval_data import EVAL_QUESTIONS
from fin_analyzer.generation import judge_groundedness
from fin_analyzer.providers.base import QuotaExhaustedError
from fin_analyzer.refusal_data import REFUSAL_QUESTIONS
from fin_analyzer.search import Chunk, search


@dataclass
class EvalResult:
    question: dict
    hit: bool
    results: list[Chunk]


def run_eval(
    settings: Settings | None = None, db=None, limit: int = 5, counter: CallCounter | None = None
) -> list[EvalResult]:
    settings = settings or get_settings()
    db = db if db is not None else get_db(settings)
    counter = counter or CallCounter()
    checkpoint = load_checkpoint()

    results = []
    for q in EVAL_QUESTIONS:
        key = f"recall::{q['question']}"
        if key in checkpoint:
            cached = checkpoint[key]
            top_results = [Chunk(**c) for c in cached["results"]]
            hit = cached["hit"]
        else:
            top_results = search(q["question"], limit=limit, db=db, settings=settings)
            counter.tick(f"embed (recall): {q['question'][:50]}")
            hit = any(q["expected_phrase"] in chunk.text for chunk in top_results)
            checkpoint[key] = {"hit": hit, "results": [asdict(c) for c in top_results]}
            save_checkpoint(checkpoint)
        results.append(EvalResult(question=q, hit=hit, results=top_results))
    return results


@dataclass
class GroundednessResult:
    question: dict
    answer: Answer
    # None means the model refused rather than answering -- there's no
    # answer to judge, so this question is excluded from the % rather than
    # counted as either grounded or ungrounded.
    verdict: GroundednessVerdict | None


def run_groundedness_eval(
    settings: Settings | None = None, db=None, k: int = 5, counter: CallCounter | None = None
) -> list[GroundednessResult]:
    settings = settings or get_settings()
    db = db if db is not None else get_db(settings)
    counter = counter or CallCounter()
    checkpoint = load_checkpoint()

    results = []
    try:
        for q in EVAL_QUESTIONS:
            answer = _cached_ask(q["question"], checkpoint, counter, k=k, db=db, settings=settings)

            if answer.answer.strip() == REFUSAL_MESSAGE:
                results.append(GroundednessResult(question=q, answer=answer, verdict=None))
                continue

            jkey = f"judge::{q['question']}"
            if jkey in checkpoint:
                verdict = GroundednessVerdict.model_validate(checkpoint[jkey])
            else:
                context_block = "\n\n".join(
                    CONTEXT_CHUNK_TEMPLATE.format(n=i + 1, ticker=s.ticker, filing_date=s.filing_date, text=s.text)
                    for i, s in enumerate(answer.sources)
                )
                judge_prompt = JUDGE_PROMPT_TEMPLATE.format(context_block=context_block, answer=answer.answer)
                verdict = judge_groundedness(JUDGE_SYSTEM_INSTRUCTION, judge_prompt, settings)
                counter.tick(f"judge: {q['question'][:50]}")
                checkpoint[jkey] = verdict.model_dump(mode="json")
                save_checkpoint(checkpoint)

            results.append(GroundednessResult(question=q, answer=answer, verdict=verdict))
    except QuotaExhaustedError as exc:
        print(f"\nQuota exhausted after {len(results)}/{len(EVAL_QUESTIONS)} groundedness questions: {exc}")
        print("Completed results are checkpointed in .cache/eval_checkpoint.json — rerun `uv run eval` to continue.")

    return results


@dataclass
class RefusalResult:
    question: dict
    answer: Answer
    refused: bool


def run_refusal_eval(
    settings: Settings | None = None, db=None, k: int = 5, counter: CallCounter | None = None
) -> list[RefusalResult]:
    settings = settings or get_settings()
    db = db if db is not None else get_db(settings)
    counter = counter or CallCounter()
    checkpoint = load_checkpoint()

    results = []
    try:
        for q in REFUSAL_QUESTIONS:
            answer = _cached_ask(q["question"], checkpoint, counter, k=k, db=db, settings=settings)
            refused = answer.answer.strip() == REFUSAL_MESSAGE
            results.append(RefusalResult(question=q, answer=answer, refused=refused))
    except QuotaExhaustedError as exc:
        print(f"\nQuota exhausted after {len(results)}/{len(REFUSAL_QUESTIONS)} refusal questions: {exc}")
        print("Completed results are checkpointed in .cache/eval_checkpoint.json — rerun `uv run eval` to continue.")

    return results


def _cached_ask(question: str, checkpoint: dict, counter: CallCounter, *, k: int, db, settings: Settings) -> Answer:
    """ask() itself isn't checkpoint-aware (it's a single ad-hoc call in
    normal use, not a batch loop) — this is the thin checkpoint wrapper
    eval.py's two generation-consuming loops share, since both need the
    same "answer this question, or reuse an already-computed answer"
    behavior."""
    key = f"answer::{question}"
    if key in checkpoint:
        return Answer.model_validate(checkpoint[key])

    answer = ask(question, k=k, db=db, settings=settings)
    counter.tick(f"answer: {question[:50]}")
    checkpoint[key] = answer.model_dump(mode="json")
    save_checkpoint(checkpoint)
    return answer
