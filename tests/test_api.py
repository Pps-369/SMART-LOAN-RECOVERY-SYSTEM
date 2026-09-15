import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from loan_recovery.features import REQUIRED_INPUT_COLUMNS  # noqa: E402


@pytest.fixture(scope="module")
def client(trained):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(trained["model_path"]))
    from api.main import app
    with TestClient(app) as c:
        yield c
    mp.undo()


def _payload(small_df, i=0):
    row = small_df.iloc[i]
    body = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in REQUIRED_INPUT_COLUMNS}
    body["Loan_ID"] = row["Loan_ID"]
    return body


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info(client):
    body = client.get("/model-info").json()
    assert "high_risk_threshold" in body and body["segments"]


def test_predict(client, small_df):
    r = client.post("/predict", json=_payload(small_df))
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["P_Loss"] <= 1
    assert body["Risk_Band"] in {"Low", "Medium", "High"}
    assert isinstance(body["Reason_Codes"], list)
    assert body["Loan_ID"] == small_df.iloc[0]["Loan_ID"]


def test_predict_batch(client, small_df):
    records = [_payload(small_df, i) for i in range(20)]
    r = client.post("/predict/batch", json={"records": records})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 20
    assert [x["Priority_Rank"] for x in body["results"]] == list(range(1, 21))


def test_validation_errors(client, small_df):
    bad = _payload(small_df)
    bad["Credit_Score"] = 1200
    assert client.post("/predict", json=bad).status_code == 422
    bad = _payload(small_df)
    bad["Loan_Type"] = "Gold"
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict/batch", json={"records": []}).status_code == 422
