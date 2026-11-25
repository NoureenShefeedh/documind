from tests.conftest import register

HANDBOOK = (
    b"Employees receive 24 days of paid leave each year. "
    b"The office opens at 9am and closes at 6pm. "
    b"Expense reports must be filed within 30 days of purchase."
)


def upload(client, auth, name="handbook.txt", data=HANDBOOK):
    return client.post("/documents", headers=auth, files={"file": (name, data)})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_register_login_flow(client):
    register(client, "a@example.com", "password123")
    ok = client.post("/auth/login", json={"email": "A@Example.com", "password": "password123"})
    assert ok.status_code == 200 and ok.json()["token_type"] == "bearer"
    bad = client.post("/auth/login", json={"email": "a@example.com", "password": "wrongpass1"})
    assert bad.status_code == 401


def test_duplicate_registration_conflicts(client):
    register(client, "dup@example.com")
    resp = client.post("/auth/register", json={"email": "dup@example.com", "password": "password123"})
    assert resp.status_code == 409


def test_register_validation(client):
    assert client.post("/auth/register", json={"email": "nope", "password": "password123"}).status_code == 422
    assert client.post("/auth/register", json={"email": "a@b.co", "password": "short"}).status_code == 422


def test_endpoints_require_auth(client):
    assert client.get("/documents").status_code == 401
    assert client.post("/query", json={"question": "hi"}).status_code == 401
    assert client.get("/documents", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_upload_query_delete_roundtrip(client, auth):
    resp = upload(client, auth)
    assert resp.status_code == 201
    doc = resp.json()
    assert doc["filename"] == "handbook.txt" and doc["n_chunks"] >= 1

    assert [d["id"] for d in client.get("/documents", headers=auth).json()] == [doc["id"]]

    result = client.post("/query", headers=auth, json={"question": "How many days of paid leave?"})
    assert result.status_code == 200
    body = result.json()
    assert "24 days" in body["answer"]
    assert body["sources"][0]["document"] == "handbook.txt"

    assert client.delete(f"/documents/{doc['id']}", headers=auth).status_code == 204
    assert client.get("/documents", headers=auth).json() == []
    assert client.delete(f"/documents/{doc['id']}", headers=auth).status_code == 404


def test_query_with_no_documents(client, auth):
    body = client.post("/query", headers=auth, json={"question": "anything?"}).json()
    assert "haven't uploaded" in body["answer"] and body["sources"] == []


def test_users_are_isolated(client, auth):
    upload(client, auth)
    other = register(client, "other@example.com")
    assert client.get("/documents", headers=other).json() == []
    body = client.post("/query", headers=other, json={"question": "paid leave days"}).json()
    assert body["sources"] == []
    doc_id = client.get("/documents", headers=auth).json()[0]["id"]
    assert client.delete(f"/documents/{doc_id}", headers=other).status_code == 404


def test_rejects_unsupported_file_type(client, auth):
    assert upload(client, auth, "malware.exe", b"MZ...").status_code == 415


def test_rejects_empty_document(client, auth):
    assert upload(client, auth, "empty.txt", b"   \n  ").status_code == 422


def test_rejects_corrupt_pdf(client, auth):
    assert upload(client, auth, "bad.pdf", b"this is not a pdf").status_code == 422


def test_rejects_oversized_upload(client, auth):
    assert upload(client, auth, "big.txt", b"a " * (1024 * 1024)).status_code == 413


def test_nul_bytes_are_stripped(client, auth):
    assert upload(client, auth, "nul.txt", b"hello\x00 world. More text here.").status_code == 201


def test_query_validation(client, auth):
    assert client.post("/query", headers=auth, json={"question": ""}).status_code == 422
    assert client.post("/query", headers=auth, json={"question": "x", "top_k": 99}).status_code == 422
