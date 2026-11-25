import numpy as np
import pytest

from app.repository import EmailTakenError, MemoryRepository


def _vec(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_duplicate_email_rejected():
    repo = MemoryRepository()
    repo.create_user("a@x.com", "h")
    with pytest.raises(EmailTakenError):
        repo.create_user("a@x.com", "h2")


def test_search_ranks_by_cosine_similarity():
    repo = MemoryRepository()
    repo.add_document(1, "d.txt", ["far", "near"], np.stack([_vec(0, 1), _vec(1, 0.1)]))
    hits = repo.search(1, _vec(1, 0), top_k=2)
    assert [h.text for h in hits] == ["near", "far"]
    assert hits[0].score > hits[1].score


def test_search_respects_top_k():
    repo = MemoryRepository()
    repo.add_document(1, "d.txt", ["a", "b", "c"], np.stack([_vec(1, 0), _vec(1, 1), _vec(0, 1)]))
    assert len(repo.search(1, _vec(1, 0), top_k=2)) == 2


def test_users_cannot_see_each_others_chunks():
    repo = MemoryRepository()
    repo.add_document(1, "mine.txt", ["secret"], np.stack([_vec(1, 0)]))
    assert repo.search(2, _vec(1, 0), top_k=5) == []


def test_delete_removes_chunks_and_checks_owner():
    repo = MemoryRepository()
    doc = repo.add_document(1, "d.txt", ["x"], np.stack([_vec(1, 0)]))
    assert repo.delete_document(2, doc.id) is False  # not the owner
    assert repo.delete_document(1, doc.id) is True
    assert repo.search(1, _vec(1, 0), top_k=5) == []
    assert repo.list_documents(1) == []
