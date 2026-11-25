"""Runs against a real Postgres with pgvector. Skipped unless TEST_DATABASE_URL is set.

    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg16
    TEST_DATABASE_URL=postgresql+psycopg://postgres:pw@localhost:5432/postgres pytest -m integration
"""
import os

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.db import PgRepository, init_db, make_engine
from app.repository import EmailTakenError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set"),
]


@pytest.fixture
def repo():
    engine = make_engine(os.environ["TEST_DATABASE_URL"])
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE chunks, documents, users RESTART IDENTITY CASCADE"))
    yield PgRepository(sessionmaker(engine, expire_on_commit=False))
    engine.dispose()


def _unit(i: int, dim: int = 384) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def test_user_roundtrip_and_unique_email(repo):
    user = repo.create_user("a@x.com", "hash")
    assert repo.get_user(user.id).email == "a@x.com"
    assert repo.get_user_by_email("a@x.com").id == user.id
    with pytest.raises(EmailTakenError):
        repo.create_user("a@x.com", "hash")


def test_vector_search_orders_by_similarity_and_scopes_to_user(repo):
    u1, u2 = repo.create_user("1@x.com", "h"), repo.create_user("2@x.com", "h")
    repo.add_document(u1.id, "d.txt", ["match", "other"], np.stack([_unit(0), _unit(1)]))
    repo.add_document(u2.id, "e.txt", ["theirs"], np.stack([_unit(0)]))
    hits = repo.search(u1.id, _unit(0), top_k=5)
    assert [h.text for h in hits] == ["match", "other"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_delete_cascades(repo):
    u = repo.create_user("a@x.com", "h")
    doc = repo.add_document(u.id, "d.txt", ["x"], np.stack([_unit(0)]))
    assert repo.delete_document(u.id, doc.id)
    assert repo.search(u.id, _unit(0), top_k=5) == []
