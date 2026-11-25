import hashlib
import math
from collections import Counter
from functools import lru_cache
from typing import Protocol

import numpy as np

from ..config import Settings
from .text import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


class HashingEmbedder:
    """Offline embedder using signed feature hashing over unigrams and bigrams.

    No model download, deterministic, and good enough for lexical retrieval. Swap in
    ``SbertEmbedder`` for real semantic similarity.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _index_sign(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            counts = Counter(tokens)
            counts_bi = Counter(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
            for feat, c in counts.items():
                idx, sign = self._index_sign(feat)
                out[i, idx] += sign * (1.0 + math.log(c))
            for feat, c in counts_bi.items():
                idx, sign = self._index_sign(feat)
                out[i, idx] += 0.5 * sign * (1.0 + math.log(c))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SbertEmbedder:
    """sentence-transformers embedder (all-MiniLM-L6-v2, 384 dims). Optional dependency."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)


@lru_cache
def _cached(provider: str, dim: int) -> Embedder:
    if provider == "hashing":
        return HashingEmbedder(dim)
    if provider == "sbert":
        emb = SbertEmbedder()
        if emb.dim != dim:
            raise ValueError(f"EMBEDDING_DIM={dim} but the sbert model produces {emb.dim}")
        return emb
    raise ValueError(f"unknown EMBEDDING_PROVIDER: {provider!r}")


def get_embedder(settings: Settings) -> Embedder:
    return _cached(settings.embedding_provider, settings.embedding_dim)
