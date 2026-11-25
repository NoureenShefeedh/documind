import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag.embeddings import HashingEmbedder
from app.rag.llm import ExtractiveAnswerer
from app.repository import MemoryRepository


@pytest.fixture
def settings() -> Settings:
    return Settings(jwt_secret="test-secret-at-least-32-bytes-long-xx", max_upload_mb=1, chunk_words=40, chunk_overlap=8)


@pytest.fixture
def client(settings) -> TestClient:
    app = create_app(
        settings,
        repo=MemoryRepository(),
        embedder=HashingEmbedder(settings.embedding_dim),
        answerer=ExtractiveAnswerer(),
    )
    return TestClient(app)


def register(client: TestClient, email: str = "me@example.com", password: str = "password123"):
    resp = client.post("/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def auth(client) -> dict:
    return register(client)
