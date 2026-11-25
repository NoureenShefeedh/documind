from dataclasses import dataclass

from ..repository import DocumentRecord, Repository
from .chunking import chunk_text
from .embeddings import Embedder
from .llm import Answerer, Source
from .loaders import DocumentParseError, extract_text

NO_DOCUMENTS = "You haven't uploaded any documents yet."


@dataclass
class Answer:
    answer: str
    sources: list[Source]


class RagService:
    """Ingest (parse, chunk, embed, store) and answer (retrieve, generate)."""

    def __init__(
        self,
        repo: Repository,
        embedder: Embedder,
        answerer: Answerer,
        *,
        chunk_words: int = 180,
        chunk_overlap: int = 40,
        default_top_k: int = 4,
    ):
        self.repo = repo
        self.embedder = embedder
        self.answerer = answerer
        self.chunk_words = chunk_words
        self.chunk_overlap = chunk_overlap
        self.default_top_k = default_top_k

    def ingest(self, user_id: int, filename: str, data: bytes) -> DocumentRecord:
        text = extract_text(filename, data)
        chunks = chunk_text(text, self.chunk_words, self.chunk_overlap)
        if not chunks:
            raise DocumentParseError("document produced no chunks")
        embeddings = self.embedder.embed(chunks)
        return self.repo.add_document(user_id, filename, chunks, embeddings)

    def ask(self, user_id: int, question: str, top_k: int | None = None) -> Answer:
        query_embedding = self.embedder.embed([question])[0]
        hits = self.repo.search(user_id, query_embedding, top_k or self.default_top_k)
        if not hits:
            return Answer(NO_DOCUMENTS, [])
        sources = [Source(h.filename, h.chunk_index, h.text, round(h.score, 4)) for h in hits]
        return Answer(self.answerer.answer(question, sources), sources)
