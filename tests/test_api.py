"""API integration tests (isolated workspace, SQLite)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(workspace, synth_dir):
    import os

    os.environ["ENTIQ_WORKSPACE"] = str(workspace)
    from backend.app.main import create_app

    return TestClient(create_app())


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["database"] == "ok"
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_datasets_list(client):
    r = client.get("/api/datasets")
    assert r.status_code == 200 and isinstance(r.json(), list)


def test_upload_rejects_bad_names_and_traversal(client):
    files = [("files", ("../../evil.tsv", b"entity_id\tbusiness_name\n", "text/plain"))]
    r = client.post("/api/datasets/upload", data={"name": "x"}, files=files)
    assert r.status_code == 422
    assert any("expected" in i for i in r.json()["detail"]["issues"])


def test_upload_rejects_wrong_separator(client):
    bad = b"entity_id,business_name,business_address,country\nS1-1,A,B,US\n"
    files = [("files", (f"train_source{k}.tsv", bad, "text/plain")) for k in (1, 2, 3)]
    r = client.post("/api/datasets/upload", data={"name": "csv"}, files=files)
    assert r.status_code == 422 and "tab-separated" in str(r.json())


def test_upload_accepts_valid_split(client, synth_dir):
    files = [("files", (f"test_source{k}.tsv", (synth_dir / "test" / f"test_source{k}.tsv").read_bytes(), "text/plain")) for k in (1, 2, 3)]
    r = client.post("/api/datasets/upload", data={"name": "ok <script>"}, files=files)
    assert r.status_code == 200 and r.json()["splits"] == ["test"]
    names = [d["name"] for d in client.get("/api/datasets").json()]
    assert "ok script" in names  # sanitised


def test_input_validation(client):
    assert client.get("/api/entities/..%2F..%2Fetc").status_code in (404, 422)
    assert client.get("/api/entities", params={"decision": "DROP TABLE"}).status_code == 422
    assert client.get("/api/results/../../x/download").status_code == 404


def test_experiments_and_models_endpoints(client):
    assert client.get("/api/experiments").status_code == 200
    assert client.get("/api/models").status_code == 200
