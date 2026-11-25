from .text import split_sentences


def chunk_text(text: str, chunk_words: int = 180, overlap: int = 40) -> list[str]:
    """Split text into sentence-aware chunks of at most ``chunk_words`` words.

    Consecutive chunks share up to ``overlap`` words of trailing sentences so that
    an answer straddling a boundary is still retrievable from one chunk.
    """
    if chunk_words <= 0 or overlap < 0 or overlap >= chunk_words:
        raise ValueError("need chunk_words > 0 and 0 <= overlap < chunk_words")

    units: list[str] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        if len(words) <= chunk_words:
            units.append(sentence)
        else:  # a single very long "sentence": hard-split it
            units.extend(
                " ".join(words[i : i + chunk_words]) for i in range(0, len(words), chunk_words)
            )

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        n = len(unit.split())
        if current and current_words + n > chunk_words:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carry_words = 0
            for prev in reversed(current):
                pn = len(prev.split())
                if carry_words + pn > overlap:
                    break
                carry.insert(0, prev)
                carry_words += pn
            if carry_words + n > chunk_words:
                carry, carry_words = [], 0
            current, current_words = carry, carry_words
        current.append(unit)
        current_words += n
    if current:
        chunks.append(" ".join(current))
    return chunks
