import numpy as np
import pandas as pd
import pytest

from loan_recovery.features import add_engineered_features
from loan_recovery.strategy import assign_risk_band, escalation_value, optimize_threshold, recommend_actions

CFG = {
    "escalation_cost": 30000, "escalation_recovery_uplift": 0.10, "collateral_haircut": 0.7,
    "min_lgd": 0.1, "high_exposure_amount": 500000, "high_emi_to_income": 0.5,
}


def _loan(**overrides):
    base = {
        "Age": 40, "Employment_Type": "Salaried", "Monthly_Income": 50000, "Num_Dependents": 1,
        "Credit_Score": 700, "Loan_Type": "Personal", "Loan_Amount": 300000, "Loan_Tenure": 36,
        "Interest_Rate": 12.0, "Monthly_EMI": 10000, "Months_Since_Origination": 12, "Collateral_Value": 0,
        "Outstanding_Loan_Amount": 200000, "Payment_History": "On-Time", "Previous_Defaults": 0,
        "Num_Missed_Payments": 1, "Days_Past_Due": 35,
    }
    base.update(overrides)
    return base


def _action(p, band, **loan):
    X = add_engineered_features(pd.DataFrame([_loan(**loan)]))
    return recommend_actions(X, [p], [band], CFG).iloc[0]


def test_risk_bands():
    bands = assign_risk_band([0.05, 0.3, 0.8], high_threshold=0.5, medium_threshold=0.25)
    assert list(bands) == ["Low", "Medium", "High"]


def test_playbook_rules():
    assert _action(0.05, "Low")["Recommended_Action"] == "Automated Reminder"
    assert _action(0.3, "Medium", Monthly_EMI=30000)["Recommended_Action"] == "EMI Restructuring"
    assert _action(0.3, "Medium")["Recommended_Action"] == "Personal Outreach"
    assert _action(0.8, "High", Collateral_Value=400000)["Recommended_Action"] == "Collateral Enforcement"
    assert _action(0.8, "High", Outstanding_Loan_Amount=900000, Loan_Amount=1000000)["Recommended_Action"] == "Settlement Negotiation"
    assert _action(0.8, "High")["Recommended_Action"] == "Agency Referral & Provisioning"


def test_expected_loss_uses_collateral():
    unsecured = _action(0.6, "High")
    secured = _action(0.6, "High", Collateral_Value=200000)
    assert unsecured["Expected_Loss"] == pytest.approx(0.6 * 200000 * 1.0)
    assert secured["Expected_Loss"] < unsecured["Expected_Loss"]
    assert secured["Loss_Given_Default"] >= CFG["min_lgd"]


def test_reason_codes():
    risky = _action(0.9, "High", Days_Past_Due=250, Credit_Score=550, Employment_Type="Unemployed")
    assert "Severely overdue" in risky["Reason_Codes"]
    assert "Credit score below 600" in risky["Reason_Codes"]
    assert "unemployed" in risky["Reason_Codes"]
    assert "Little or no collateral" in risky["Reason_Codes"]


def test_escalation_value_math():
    y = np.array([1, 0, 1])
    p = np.array([0.9, 0.8, 0.1])
    outstanding = np.array([1_000_000, 1_000_000, 1_000_000])
    # flags first two: saves 10% of 1M on the true loss, pays 2 x 30k
    assert escalation_value(y, p, outstanding, 0.5, 30000, 0.10) == 100000 - 60000


def test_optimize_threshold_prefers_value():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.6 + rng.uniform(0, 0.4, 500), 0, 1)   # informative scores
    best, curve = optimize_threshold(y, p, np.full(500, 1_000_000), 30000, 0.10)
    assert curve["net_value"].max() == escalation_value(y, p, np.full(500, 1_000_000), best, 30000, 0.10)
    assert 0.05 <= best <= 0.95
