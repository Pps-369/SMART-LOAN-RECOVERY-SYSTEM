"""Turn loss probabilities into recovery decisions.

A probability on its own does not tell a bank what to do. This module adds the money:

1. Decision threshold - chosen to MAXIMISE NET RECOVERY VALUE on validation data, not accuracy.
   Escalating a loan costs `escalation_cost`; if the loan would otherwise be lost, escalation
   saves `escalation_recovery_uplift * outstanding`. The best threshold balances the two.
2. Risk bands (Low / Medium / High) from that threshold.
3. An action playbook that also looks at affordability, collateral and exposure.
4. Expected loss (for provisioning) and net escalation value (for prioritising the queue).

These rules are a transparent starting point. The cost and uplift numbers in config.yaml are
assumptions; in a real bank they come from past campaign results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACTIONS = {
    "Automated Reminder": "SMS, email and app nudges with a payment link; give the borrower room to self-cure.",
    "Personal Outreach": "Assign a recovery agent for calls and agree a short-term payment plan.",
    "EMI Restructuring": "Offer a tenure extension or reduced EMI - affordability is the core problem.",
    "Collateral Enforcement": "Send a legal demand notice and start collateral recovery; the security covers the dues.",
    "Settlement Negotiation": "Negotiate a one-time settlement backed by a legal notice; exposure is large and under-secured.",
    "Agency Referral & Provisioning": "Refer to a collection agency and provision for the likely loss.",
}
BAND_ORDER = ["Low", "Medium", "High"]


def escalation_value(y_true, p_loss, outstanding, threshold: float, cost: float, uplift: float) -> float:
    """Net value of escalating every loan with p_loss >= threshold."""
    y_true, p_loss, outstanding = map(np.asarray, (y_true, p_loss, outstanding))
    flagged = p_loss >= threshold
    saved = uplift * outstanding[flagged & (y_true == 1)].sum()
    return float(saved - cost * flagged.sum())


def optimize_threshold(y_true, p_loss, outstanding, cost: float, uplift: float) -> tuple[float, pd.DataFrame]:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    y_true, p_loss = np.asarray(y_true), np.asarray(p_loss)
    rows = []
    for t in grid:
        flagged = p_loss >= t
        rows.append(
            {
                "threshold": t,
                "net_value": escalation_value(y_true, p_loss, outstanding, t, cost, uplift),
                "flagged_share": float(flagged.mean()),
                "recall": float((flagged & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)),
                "precision": float((flagged & (y_true == 1)).sum() / max(flagged.sum(), 1)),
            }
        )
    curve = pd.DataFrame(rows)
    best = float(curve.loc[curve["net_value"].idxmax(), "threshold"])
    return best, curve


def assign_risk_band(p_loss, high_threshold: float, medium_threshold: float) -> np.ndarray:
    p_loss = np.asarray(p_loss)
    return np.select([p_loss >= high_threshold, p_loss >= medium_threshold], ["High", "Medium"], default="Low")


def reason_codes(X: pd.DataFrame, emi_limit: float) -> list[str]:
    """Plain-language risk flags. These are business rules, NOT a model explanation."""
    rules = [
        (X["Days_Past_Due"] >= 180, "Severely overdue (180+ days past due)"),
        ((X["Days_Past_Due"] >= 90) & (X["Days_Past_Due"] < 180), "Overdue 90+ days"),
        (X["Num_Missed_Payments"] >= 6, "6 or more missed installments"),
        (X["EMI_to_Income"] >= emi_limit, f"EMI is over {emi_limit:.0%} of monthly income"),
        (X["Collateral_Coverage"] < 0.25, "Little or no collateral cover"),
        (X["Credit_Score"] < 600, "Credit score below 600"),
        (X["Previous_Defaults"] >= 1, "Previous defaults on record"),
        (X["Employment_Type"] == "Unemployed", "Borrower currently unemployed"),
        (X["Payment_History"] == "Missed", "History of missed payments"),
    ]
    codes = [[] for _ in range(len(X))]
    for mask, text in rules:
        for i in np.flatnonzero(mask.fillna(False).to_numpy()):
            codes[i].append(text)
    return ["; ".join(c) if c else "No major risk flags" for c in codes]


def recommend_actions(X: pd.DataFrame, p_loss, bands, cfg: dict) -> pd.DataFrame:
    """X must already contain engineered features."""
    p_loss = np.asarray(p_loss)
    bands = np.asarray(bands)
    outstanding = X["Outstanding_Loan_Amount"].fillna(0).to_numpy()
    coverage = X["Collateral_Coverage"].fillna(0).to_numpy()
    emi_ratio = X["EMI_to_Income"].fillna(0).to_numpy()

    action = np.select(
        [
            bands == "Low",
            (bands == "Medium") & (emi_ratio >= cfg["high_emi_to_income"]),
            bands == "Medium",
            (bands == "High") & (coverage >= 1.0),
            (bands == "High") & (outstanding >= cfg["high_exposure_amount"]),
        ],
        ["Automated Reminder", "EMI Restructuring", "Personal Outreach", "Collateral Enforcement", "Settlement Negotiation"],
        default="Agency Referral & Provisioning",
    )

    lgd = np.clip(1 - coverage * cfg["collateral_haircut"], cfg["min_lgd"], 1.0)
    expected_loss = outstanding * p_loss * lgd
    net_escalation_value = p_loss * cfg["escalation_recovery_uplift"] * outstanding - cfg["escalation_cost"]

    return pd.DataFrame(
        {
            "Recommended_Action": action,
            "Action_Detail": [ACTIONS[a] for a in action],
            "Loss_Given_Default": lgd.round(3),
            "Expected_Loss": expected_loss.round(0),
            "Value_At_Risk": (outstanding * p_loss).round(0),
            "Net_Escalation_Value": net_escalation_value.round(0),
            "Reason_Codes": reason_codes(X.reset_index(drop=True), cfg["high_emi_to_income"]),
        },
        index=X.index,
    )
