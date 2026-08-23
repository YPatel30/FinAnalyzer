"""Split extracted filing text into overlapping, token-sized chunks.

RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
then sentences, then words — only falling back to a hard character cut if a
single paragraph is bigger than one chunk. That keeps chunk boundaries from
landing mid-sentence in the common case. Length is measured in tokens (via
tiktoken's cl100k_base encoding) rather than characters, so "~1000 tokens"
is accurate rather than a word-count guess.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paragraphs in extract.py's output are joined with "\n\n", so that's the
# first (and most common) split point. Falling back through single newlines,
# sentence ends, then words keeps later splits still reasonably sentence-shaped.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(text: str, chunk_size_tokens: int, chunk_overlap_tokens: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=SEPARATORS,
    )
    return splitter.split_text(text)
