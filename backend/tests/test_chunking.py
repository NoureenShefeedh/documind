import pytest

from app.rag.chunking import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("One sentence. Another one.", chunk_words=50, overlap=5) == [
        "One sentence. Another one."
    ]


def test_chunks_respect_max_words():
    text = " ".join(f"Sentence number {i} has a few words." for i in range(100))
    chunks = chunk_text(text, chunk_words=30, overlap=8)
    assert len(chunks) > 5
    assert all(len(c.split()) <= 30 for c in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"Sentence {i} is here." for i in range(40))
    chunks = chunk_text(text, chunk_words=20, overlap=8)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split(". ")[-1].rstrip(".") in nxt or prev.split()[-1] in nxt


def test_no_content_lost():
    sentences = [f"Fact {i} stays." for i in range(60)]
    chunks = chunk_text(" ".join(sentences), chunk_words=25, overlap=6)
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_overlong_sentence_is_hard_split():
    chunks = chunk_text("word " * 500, chunk_words=100, overlap=10)
    assert len(chunks) == 5
    assert all(len(c.split()) <= 100 for c in chunks)


@pytest.mark.parametrize("words,overlap", [(0, 0), (10, 10), (10, -1), (5, 9)])
def test_invalid_parameters(words, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", words, overlap)
