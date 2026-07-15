from fastapi.testclient import TestClient

from rag_support.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_answer_shape():
    response = client.post("/ask", json={"question": "How do I restart the device?"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "sources", "context_used"}


def test_ask_validates_input():
    assert client.post("/ask", json={"question": ""}).status_code == 422
