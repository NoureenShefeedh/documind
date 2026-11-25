"""Storage interface plus an in-memory implementation (used by tests and quick demos)."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count
from typing import Protocol

import numpy as np


class EmailTakenError(Exception):
    pass


@dataclass
class UserRecord:
    id: int
    email: str
    password_hash: str


@dataclass
class DocumentRecord:
    id: int
    user_id: int
    filename: str
    n_chunks: int
    created_at: datetime


@dataclass
class Hit:
    document_id: int
    filename: str
    chunk_index: int
    text: str
    score: float  # cosine similarity, higher is better


class Repository(Protocol):
    def create_user(self, email: str, password_hash: str) -> UserRecord: ...
    def get_user_by_email(self, email: str) -> UserRecord | None: ...
    def get_user(self, user_id: int) -> UserRecord | None: ...
    def add_document(
        self, user_id: int, filename: str, chunks: list[str], embeddings: np.ndarray
    ) -> DocumentRecord: ...
    def list_documents(self, user_id: int) -> list[DocumentRecord]: ...
    def delete_document(self, user_id: int, document_id: int) -> bool: ...
    def search(self, user_id: int, query_embedding: np.ndarray, top_k: int) -> list[Hit]: ...


@dataclass
class _Chunk:
    document_id: int
    index: int
    text: str
    embedding: np.ndarray


@dataclass
class MemoryRepository:
    users: dict[int, UserRecord] = field(default_factory=dict)
    documents: dict[int, DocumentRecord] = field(default_factory=dict)
    chunks: list[_Chunk] = field(default_factory=list)
    _uid: count = field(default_factory=lambda: count(1))
    _did: count = field(default_factory=lambda: count(1))

    def create_user(self, email: str, password_hash: str) -> UserRecord:
        if self.get_user_by_email(email):
            raise EmailTakenError(email)
        user = UserRecord(next(self._uid), email, password_hash)
        self.users[user.id] = user
        return user

    def get_user_by_email(self, email: str) -> UserRecord | None:
        return next((u for u in self.users.values() if u.email == email), None)

    def get_user(self, user_id: int) -> UserRecord | None:
        return self.users.get(user_id)

    def add_document(
        self, user_id: int, filename: str, chunks: list[str], embeddings: np.ndarray
    ) -> DocumentRecord:
        doc = DocumentRecord(
            next(self._did), user_id, filename, len(chunks), datetime.now(timezone.utc)
        )
        self.documents[doc.id] = doc
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            self.chunks.append(_Chunk(doc.id, i, text, np.asarray(emb, dtype=np.float32)))
        return doc

    def list_documents(self, user_id: int) -> list[DocumentRecord]:
        docs = [d for d in self.documents.values() if d.user_id == user_id]
        return sorted(docs, key=lambda d: d.id, reverse=True)

    def delete_document(self, user_id: int, document_id: int) -> bool:
        doc = self.documents.get(document_id)
        if doc is None or doc.user_id != user_id:
            return False
        del self.documents[document_id]
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        return True

    def search(self, user_id: int, query_embedding: np.ndarray, top_k: int) -> list[Hit]:
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        scored: list[Hit] = []
        for c in self.chunks:
            doc = self.documents[c.document_id]
            if doc.user_id != user_id:
                continue
            cn = np.linalg.norm(c.embedding) or 1.0
            score = float(c.embedding @ q / (cn * qn))
            scored.append(Hit(doc.id, doc.filename, c.index, c.text, score))
        scored.sort(key=lambda h: -h.score)
        return scored[:top_k]
