"""C6 - the HTTP API requires an API key. Runs where fastapi + httpx are installed (CI)."""
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(registry_root, monkeypatch):
    monkeypatch.setenv("ROAS_API_KEYS", "test-key")
    monkeypatch.delenv("ROAS_ALLOW_NO_AUTH", raising=False)
    from services.api.main import create_app

    return TestClient(create_app(str(registry_root)))


BODY = {"scenarios": [{"game_id": "SG01", "cohort_month": "2025-10", "total_cac": 150000, "forecast_months": 6}]}


def test_refuses_to_start_without_keys(registry_root, monkeypatch):
    monkeypatch.delenv("ROAS_API_KEYS", raising=False)
    monkeypatch.delenv("ROAS_ALLOW_NO_AUTH", raising=False)
    from services.api.main import create_app

    with pytest.raises(RuntimeError):
        create_app(str(registry_root))


def test_requires_key(client):
    assert client.post("/v2/forecast", json=BODY).status_code == 401
    assert client.post("/v2/forecast", json=BODY, headers={"X-API-Key": "wrong"}).status_code == 401


def test_forecast_with_key(client):
    r = client.post("/v2/forecast", json=BODY, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"].startswith("v") and len(body["results"][0]["months"]) == 6
    assert {"roas_bad", "roas_expected", "roas_good"} <= set(body["results"][0]["months"][0])


def test_unknown_channel_is_422_and_unknown_game_404(client):
    h = {"X-API-Key": "test-key"}
    bad = {"scenarios": [{"game_id": "SG01", "cohort_month": "2025-10", "channel_spends": {"MC_Snapchat": 1}}]}
    assert client.post("/v2/forecast", json=bad, headers=h).status_code == 422
    nogame = {"scenarios": [{"game_id": "NOPE", "cohort_month": "2025-10", "total_cac": 1}]}
    assert client.post("/v2/forecast", json=nogame, headers=h).status_code == 404


def test_health_is_open(client):
    assert client.get("/health").json()["status"] == "ok"
