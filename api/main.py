"""REST API for the Smart Loan Recovery System.

    uvicorn api.main:app --reload
    open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from loan_recovery import __version__
from loan_recovery.predict import LoanRecoveryPredictor

EXAMPLE = {
    "Loan_ID": "LN0001234",
    "Borrower_ID": "BRW000042",
    "Age": 38,
    "Employment_Type": "Self-Employed",
    "Monthly_Income": 42000,
    "Num_Dependents": 2,
    "Credit_Score": 598,
    "Loan_Type": "Personal",
    "Loan_Amount": 450000,
    "Loan_Tenure": 48,
    "Interest_Rate": 14.2,
    "Monthly_EMI": 12400,
    "Months_Since_Origination": 20,
    "Collateral_Value": 0,
    "Outstanding_Loan_Amount": 318000,
    "Payment_History": "Delayed",
    "Previous_Defaults": 1,
    "Num_Missed_Payments": 7,
    "Days_Past_Due": 210,
}


class LoanRecord(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE})

    Loan_ID: str | None = None
    Borrower_ID: str | None = None
    Age: int = Field(..., ge=18, le=100)
    Employment_Type: Literal["Salaried", "Self-Employed", "Business Owner", "Unemployed"]
    Monthly_Income: float = Field(..., gt=0)
    Num_Dependents: int = Field(..., ge=0, le=20)
    Credit_Score: float = Field(..., ge=300, le=900)
    Loan_Type: Literal["Home", "Auto", "Personal", "Business"]
    Loan_Amount: float = Field(..., gt=0)
    Loan_Tenure: int = Field(..., gt=0, le=480, description="months")
    Interest_Rate: float = Field(..., ge=0, le=60, description="annual %")
    Monthly_EMI: float = Field(..., gt=0)
    Months_Since_Origination: int = Field(..., ge=0)
    Collateral_Value: float = Field(..., ge=0)
    Outstanding_Loan_Amount: float = Field(..., gt=0)
    Payment_History: Literal["On-Time", "Delayed", "Missed"]
    Previous_Defaults: int = Field(..., ge=0)
    Num_Missed_Payments: int = Field(..., ge=0)
    Days_Past_Due: int = Field(..., ge=0)


class ScoreResult(BaseModel):
    Loan_ID: str | None = None
    Borrower_ID: str | None = None
    P_Loss: float
    P_Recovery: float
    Risk_Band: str
    Segment: str
    Recommended_Action: str
    Action_Detail: str
    Expected_Loss: float
    Value_At_Risk: float
    Net_Escalation_Value: float
    Priority_Rank: int
    Reason_Codes: list[str]


class BatchRequest(BaseModel):
    records: list[LoanRecord] = Field(..., min_length=1, max_length=10000)


class BatchResponse(BaseModel):
    count: int
    total_outstanding: float
    total_expected_loss: float
    risk_band_counts: dict[str, int]
    action_counts: dict[str, int]
    results: list[ScoreResult]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["predictor"] = LoanRecoveryPredictor.load()
    except FileNotFoundError as exc:
        state["predictor"] = None
        state["load_error"] = str(exc)
    yield
    state.clear()


app = FastAPI(
    title="Smart Loan Recovery API",
    description="Predicts whether an overdue loan will be recovered and recommends a recovery action.",
    version=__version__,
    lifespan=lifespan,
)


def _predictor() -> LoanRecoveryPredictor:
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail=state.get("load_error", "Model not loaded"))
    return predictor


def _score(records: list[LoanRecord]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in records])
    return _predictor().score(df)


def _to_results(scored: pd.DataFrame) -> list[ScoreResult]:
    rows = scored.astype({"Risk_Band": str}).to_dict(orient="records")
    results = []
    for row in rows:
        row["Reason_Codes"] = [] if row["Reason_Codes"] == "No major risk flags" else row["Reason_Codes"].split("; ")
        for key in ("Loan_ID", "Borrower_ID"):
            row[key] = row.get(key) if isinstance(row.get(key), str) else None
        results.append(ScoreResult(**row))
    return results


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.get("predictor") is not None}


@app.get("/model-info")
def model_info():
    p = _predictor()
    return {
        "metadata": p.metadata,
        "segments": p.segment_info,
        "high_risk_threshold": p.threshold,
        "medium_risk_threshold": p.medium_threshold,
        "strategy_config": p.strategy_config,
    }


@app.post("/predict", response_model=ScoreResult)
def predict(record: LoanRecord):
    return _to_results(_score([record]))[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    scored = _score(request.records).sort_values("Priority_Rank")
    return BatchResponse(
        count=len(scored),
        total_outstanding=float(scored["Outstanding_Loan_Amount"].sum()),
        total_expected_loss=float(scored["Expected_Loss"].sum()),
        risk_band_counts={str(k): int(v) for k, v in scored["Risk_Band"].value_counts().items()},
        action_counts={str(k): int(v) for k, v in scored["Recommended_Action"].value_counts().items()},
        results=_to_results(scored),
    )
