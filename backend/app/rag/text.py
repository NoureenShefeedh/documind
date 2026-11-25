"""Small text helpers shared by chunking, embeddings and the extractive answerer."""
import re

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")
_PARA = re.compile(r"\n\s*\n")

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in into is it its
    me my of on or our she so than that the their them then there these they this to
    us was we were what when where which who why will with you your do does did how
    can could should would about""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens with stopwords removed."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting paragraph breaks."""
    sentences: list[str] = []
    for para in _PARA.split(text.replace("\r\n", "\n")):
        para = " ".join(para.split())
        if para:
            sentences.extend(s for s in _SENT.split(para) if s)
    return sentences
