"""PostgreSQL schema (SQLAlchemy 2.0 + pgvector) and the Postgres-backed repository."""
from datetime import UTC, datetime

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import get_settings
from .repository import DocumentRecord, EmailTakenError, Hit, UserRecord

EMBEDDING_DIM = get_settings().embedding_dim


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    n_chunks: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    idx: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def _user(u: User) -> UserRecord:
    return UserRecord(u.id, u.email, u.password_hash)


def _doc(d: Document) -> DocumentRecord:
    return DocumentRecord(d.id, d.user_id, d.filename, d.n_chunks, d.created_at)


class PgRepository:
    def __init__(self, session_factory: sessionmaker):
        self._sf = session_factory

    def create_user(self, email: str, password_hash: str) -> UserRecord:
        with self._sf() as s:
            user = User(email=email, password_hash=password_hash)
            s.add(user)
            try:
                s.commit()
            except IntegrityError as exc:
                s.rollback()
                raise EmailTakenError(email) from exc
            return _user(user)

    def get_user_by_email(self, email: str) -> UserRecord | None:
        with self._sf() as s:
            u = s.scalar(select(User).where(User.email == email))
            return _user(u) if u else None

    def get_user(self, user_id: int) -> UserRecord | None:
        with self._sf() as s:
            u = s.get(User, user_id)
            return _user(u) if u else None

    def add_document(
        self, user_id: int, filename: str, chunks: list[str], embeddings: np.ndarray
    ) -> DocumentRecord:
        with self._sf() as s:
            doc = Document(user_id=user_id, filename=filename, n_chunks=len(chunks))
            s.add(doc)
            s.flush()
            s.add_all(
                Chunk(
                    document_id=doc.id,
                    user_id=user_id,
                    idx=i,
                    content=content,
                    embedding=np.asarray(emb, dtype=np.float32).tolist(),
                )
                for i, (content, emb) in enumerate(zip(chunks, embeddings, strict=True))
            )
            s.commit()
            return _doc(doc)

    def list_documents(self, user_id: int) -> list[DocumentRecord]:
        with self._sf() as s:
            rows = s.scalars(
                select(Document).where(Document.user_id == user_id).order_by(Document.id.desc())
            )
            return [_doc(d) for d in rows]

    def delete_document(self, user_id: int, document_id: int) -> bool:
        with self._sf() as s:
            owned = s.scalar(
                select(func.count())
                .select_from(Document)
                .where(Document.id == document_id, Document.user_id == user_id)
            )
            if not owned:
                return False
            s.execute(delete(Chunk).where(Chunk.document_id == document_id))
            s.execute(delete(Document).where(Document.id == document_id))
            s.commit()
            return True

    def search(self, user_id: int, query_embedding: np.ndarray, top_k: int) -> list[Hit]:
        q = np.asarray(query_embedding, dtype=np.float32).tolist()
        dist = Chunk.embedding.cosine_distance(q)
        stmt = (
            select(Chunk.document_id, Document.filename, Chunk.idx, Chunk.content, dist.label("d"))
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.user_id == user_id)
            .order_by(dist)
            .limit(top_k)
        )
        with self._sf() as s:
            return [
                Hit(doc_id, filename, idx, content, 1.0 - float(d))
                for doc_id, filename, idx, content, d in s.execute(stmt)
            ]
