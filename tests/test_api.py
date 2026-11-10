import math

import pytest
from fastapi.testclient import TestClient

from app.main import app, state


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers lifespan -> model load
        yield c


@pytest.fixture(scope="module")
def valid_features(client):
    return state["meta"]["example_input"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["n_features"] == len(body["feature_names"]) == 30
    assert "roc_auc" in body["test_metrics"]


def test_predict_schema_and_probabilities(client, valid_features):
    r = client.post("/predict", json={"features": valid_features})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "label", "probabilities", "model_version"}
    assert body["prediction"] in (0, 1)
    assert isinstance(body["label"], str)
    probs = body["probabilities"]
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-4)
    assert body["label"] == max(probs, key=probs.get)


def test_predict_is_deterministic(client, valid_features):
    a = client.post("/predict", json={"features": valid_features}).json()
    b = client.post("/predict", json={"features": valid_features}).json()
    assert a == b


def test_predict_wrong_feature_count(client):
    r = client.post("/predict", json={"features": [1.0, 2.0]})
    assert r.status_code == 422
    assert "30" in r.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"features": "not-a-list"},
        {"features": ["a"] * 30},
        {"features": []},
        {"wrong_key": [1.0] * 30},
    ],
)
def test_predict_invalid_payloads(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_non_json_body(client):
    r = client.post("/predict", content="garbage", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_batch_predict(client, valid_features):
    r = client.post("/predict/batch", json={"instances": [valid_features, valid_features]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert preds[0] == preds[1]


def test_batch_ragged_rows(client, valid_features):
    r = client.post("/predict/batch", json={"instances": [valid_features, [1.0]]})
    assert r.status_code == 422


def test_batch_empty(client):
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422


def test_unknown_route_and_method(client):
    assert client.get("/nope").status_code == 404
    assert client.get("/predict").status_code == 405


def test_openapi_docs_available(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_process_time_header(client, valid_features):
    r = client.post("/predict", json={"features": valid_features})
    assert float(r.headers["X-Process-Time-Ms"]) >= 0


def test_api_key_enforced_when_configured(client, valid_features, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    payload = {"features": valid_features}
    assert client.post("/predict", json=payload).status_code == 401
    assert client.post("/predict", json=payload, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/predict", json=payload, headers={"X-API-Key": "secret123"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open


def test_api_key_not_required_by_default(client, valid_features, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    assert client.post("/predict", json={"features": valid_features}).status_code == 200


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "ML Inference Console" in r.text


def test_model_info_has_examples(client):
    ex = client.get("/model-info").json()["examples"]
    assert set(ex) == {"malignant", "benign"}
    for label, feats in ex.items():
        pred = client.post("/predict", json={"features": feats}).json()
        assert pred["label"] == label
