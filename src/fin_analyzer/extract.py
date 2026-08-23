"""Turn raw 10-K HTML into clean prose text.

10-K HTML is messy: giant financial-statement tables, a table of contents,
inline-XBRL tags (some wrapping pure metadata, some wrapping real numbers
and footnote paragraphs right in the visible document), and heavy use of
inline formatting tags (<span>, <font>, <b>) that split a single sentence
across many small text nodes. The steps below deal with each problem in order.
"""

import re

from bs4 import BeautifulSoup

# Tags that only ever carry formatting, never paragraph/heading structure.
# We *unwrap* these (drop the tag, keep its text in place) rather than
# decompose them, because decomposing would delete a chunk of a sentence
# that just happens to be, say, bolded.
INLINE_TAGS = ["a", "b", "i", "u", "em", "strong", "span", "font", "sup", "sub", "small", "label"]

# Tags that never contain prose worth keeping — deleted entirely, contents included.
JUNK_TAGS = ["script", "style", "table"]

# A "line" (see get_text below) shorter than this is almost always a heading,
# a page number, or a table/TOC fragment rather than a full sentence.
MIN_LINE_LENGTH = 40

# A prose sentence is mostly letters. TOC leaders ("Item 1 . . . . . . 5"),
# stray numbers, and punctuation-only remnants fail this ratio.
MIN_LETTER_RATIO = 0.5

WHITESPACE_RE = re.compile(r"\s+")


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(JUNK_TAGS):
        tag.decompose()

    # <ix:header> is the one place an inline-XBRL document puts all its
    # non-visible metadata (contexts, units, dimension members, hidden
    # duplicate-tagged facts) — safe to drop as a single block.
    header = soup.find("ix:header")
    if header is not None:
        header.decompose()

    # Inline XBRL also tags content that *is* visible, right in the body:
    # <ix:nonfraction> wraps a single number like "$391,035" sitting inside
    # an MD&A sentence, and <ix:nonnumeric>/<ix:continuation> wrap whole
    # footnote paragraphs (financial statement notes are usually tagged this
    # way). Decomposing these — as opposed to <ix:header> — would silently
    # delete real prose/numbers out of sentences, so unwrap them instead,
    # same as the plain inline formatting tags below.
    for tag in soup.find_all(lambda t: t.name and ":" in t.name):
        tag.unwrap()

    for tag in soup.find_all(INLINE_TAGS):
        tag.unwrap()

    # unwrap() leaves each inline tag's text as its own text node. Without
    # merging them, get_text() below would insert a line break *inside* a
    # sentence wherever it used to have a <span> or <b>. smooth() merges
    # adjacent text nodes back into one, so only real block boundaries
    # (p, div, li, h1-h6, br, ...) produce separate lines.
    soup.smooth()

    raw_text = soup.get_text(separator="\n")
    lines = (_normalize(line) for line in raw_text.split("\n"))
    prose_lines = [line for line in lines if _looks_like_prose(line)]
    return "\n\n".join(prose_lines)


def _normalize(line: str) -> str:
    """Collapse runs of whitespace (including the &nbsp; padding SEC filings
    use heavily for layout) down to single spaces."""
    return WHITESPACE_RE.sub(" ", line).strip()


def _looks_like_prose(line: str) -> bool:
    if len(line) < MIN_LINE_LENGTH:
        return False
    letters = sum(char.isalpha() for char in line)
    return (letters / len(line)) >= MIN_LETTER_RATIO
