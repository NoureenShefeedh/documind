import numpy as np

from app.rag.embeddings import HashingEmbedder


def test_shape_and_unit_norm():
    emb = HashingEmbedder(384).embed(["hello world", "another text here"])
    assert emb.shape == (2, 384)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_deterministic():
    e = HashingEmbedder(128)
    assert np.array_equal(e.embed(["same text"]), e.embed(["same text"]))


def test_empty_text_does_not_produce_nan():
    emb = HashingEmbedder(64).embed(["", "the and of"])
    assert not np.isnan(emb).any()


def test_related_text_scores_higher_than_unrelated():
    e = HashingEmbedder(384)
    q, related, unrelated = e.embed(
        [
            "how do I reset my password",
            "To reset your password open settings and choose reset password.",
            "The quarterly revenue grew by twelve percent in Europe.",
        ]
    )
    assert q @ related > q @ unrelated
