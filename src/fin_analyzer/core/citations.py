"""Parsing [n] citation markers back out of a generated answer.

Kept separate and pure (no Gemini/Mongo dependency) so it's trivially unit
testable against literal strings, independent of ever calling the model.
"""

import re

# Matches "[2]" -> "2", and also "【2】" (fullwidth brackets) -> "2". The
# prompt instructs plain ASCII "[2][4]" style, but observed in practice
# (a real Groq answer, not a hypothetical): the model sometimes emits
# fullwidth CJK-style brackets instead, which look like a citation to a
# human reader but silently matched nothing under the ASCII-only pattern —
# every source came back "not cited" even though the model clearly meant
# to cite one. Both styles are treated as the same citation instruction.
CITATION_RE = re.compile(r"[\[【](\d+)[\]】]")


def parse_cited_numbers(answer_text: str) -> set[int]:
    """Every distinct chunk number the model cited, e.g. "...[2][4]..." ->
    {2, 4}. A number that doesn't correspond to an actual context chunk
    (the model citing [99] when only 5 were given) is still returned here —
    it's the caller's job (ask.py) to intersect this against the real
    number of chunks sent.
    """
    return {int(n) for n in CITATION_RE.findall(answer_text)}
