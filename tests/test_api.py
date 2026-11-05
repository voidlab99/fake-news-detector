"""API tests. They need torch, fastapi and a trained ./model, so they skip on a fresh checkout / in CI."""
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

if not (Path(__file__).resolve().parent.parent / "model").exists():
    pytest.skip("no trained model in ./model (run train.py first)", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager runs the lifespan (loads the model)
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_valid_distribution(client):
    r = client.post("/predict", json={"text": "The city council voted on Tuesday to approve the new budget for roads."})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"REAL", "FAKE"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_rejects_too_short_text(client):
    assert client.post("/predict", json={"text": "hi"}).status_code == 422


def test_index_page_served(client):
    assert client.get("/").status_code == 200
