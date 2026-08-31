"""Pydantic models for ask()'s return value and the groundedness judge's
verdict (eval.py). Pydantic (not a plain dataclass, unlike search.py's
Chunk) because these cross a real boundary — they're what a caller outside
this codebase would consume, and Answer/GroundednessVerdict need to survive
being built from parsed model output, where validation actually matters.
"""

from pydantic import BaseModel


class Source(BaseModel):
    ticker: str
    filing_date: str
    chunk_index: int
    score: float
    # Full chunk text, not truncated. The groundedness judge (eval.py) needs
    # the whole passage to check claims against — truncating here could
    # hide the very sentence that supports (or fails to support) a claim.
    # A short display excerpt is a presentation-time choice, not a data
    # model one — see cli_ask.py, which truncates only when printing, the
    # same pattern cli_search.py already uses for Chunk.text.
    text: str
    # True if the model's answer actually cited this chunk's [n] number —
    # distinct from merely having been retrieved. See core/citations.py.
    cited: bool


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    context_used: int  # how many chunks were retrieved and put in the prompt


class GroundednessVerdict(BaseModel):
    grounded: bool
    unsupported_claims: list[str]
