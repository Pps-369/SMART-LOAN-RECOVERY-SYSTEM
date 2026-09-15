"""Synthetic overdue-loan portfolio generator.

Real bank recovery data is confidential, so this module simulates a portfolio of loans
that are already overdue. The outcome (Recovery_Status) is driven by a hidden risk score
built from things that genuinely matter in collections: how late the loan is, whether the
borrower can afford the EMI, collateral cover, credit history and employment. Some effects
are non-linear (e.g. very late AND unsecured is much worse than either alone), which gives
tree models something real to learn.

The generator also writes columns that only exist AFTER the bank has started recovering
(Collection_Method, Collection_Attempts, Legal_Action_Taken, Recovered_Amount). They are
deliberately correlated with the outcome, and the training pipeline deliberately excludes
them - see features.LEAKAGE_COLUMNS.

Usage:
    python -m loan_recovery.data_generator --n 12000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import load_config, resolve_path

EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Business Owner", "Unemployed"]
EMPLOYMENT_PROBS = [0.55, 0.22, 0.13, 0.10]
LOAN_TYPES = ["Home", "Auto", "Personal", "Business"]
LOAN_PROBS = [0.22, 0.24, 0.36, 0.18]
PAYMENT_HISTORY = ["On-Time", "Delayed", "Missed"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _emi(principal: np.ndarray, annual_rate: np.ndarray, months: np.ndarray) -> np.ndarray:
    r = annual_rate / 12 / 100
    growth = (1 + r) ** months
    return principal * r * growth / (growth - 1)


def _balance_after(principal, annual_rate, months, paid):
    """Remaining principal after `paid` installments of a standard amortising loan."""
    r = annual_rate / 12 / 100
    g_n = (1 + r) ** months
    g_k = (1 + r) ** paid
    return principal * (g_n - g_k) / (g_n - 1)


def _pick_per_row(rng, options_by_row: list[list[str]]) -> np.ndarray:
    return np.array([rng.choice(opts) for opts in options_by_row])


def generate_loan_data(n: int = 12000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # ---- Borrower profile -------------------------------------------------
    employment = rng.choice(EMPLOYMENT_TYPES, size=n, p=EMPLOYMENT_PROBS)
    age = rng.integers(21, 66, size=n)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.6, 0.4])
    dependents = rng.integers(0, 5, size=n)

    median_income = pd.Series(employment).map(
        {"Salaried": 65000, "Self-Employed": 55000, "Business Owner": 90000, "Unemployed": 18000}
    ).to_numpy()
    income = np.clip(rng.lognormal(np.log(median_income), 0.45), 8000, 800000).round(-2)

    credit_score = np.clip(
        rng.normal(680, 70, n) - 30 * (employment == "Unemployed"), 300, 900
    ).round()

    # ---- Loan terms -------------------------------------------------------
    loan_type = rng.choice(LOAN_TYPES, size=n, p=LOAN_PROBS)
    lt = pd.Series(loan_type)
    mult_lo = lt.map({"Home": 25, "Auto": 5, "Personal": 2, "Business": 8}).to_numpy()
    mult_hi = lt.map({"Home": 70, "Auto": 15, "Personal": 10, "Business": 35}).to_numpy()
    # loan size is set when the borrower was earning a "normal" income
    loan_amount = (median_income * np.exp(rng.normal(0, 0.3, n)) * rng.uniform(mult_lo, mult_hi)).round(-3)
    loan_amount = np.maximum(loan_amount, 25000)

    ten_lo = lt.map({"Home": 10, "Auto": 3, "Personal": 1, "Business": 2}).to_numpy()
    ten_hi = lt.map({"Home": 25, "Auto": 7, "Personal": 5, "Business": 10}).to_numpy()
    tenure = rng.integers(ten_lo, ten_hi + 1) * 12

    base_rate = lt.map({"Home": 8.5, "Auto": 9.5, "Personal": 13.0, "Business": 12.0}).to_numpy()
    interest_rate = np.clip(base_rate + (700 - credit_score) / 100 * 1.5 + rng.normal(0, 0.5, n), 7, 24).round(2)

    emi = _emi(loan_amount, interest_rate, tenure).round()

    # ---- Delinquency state -----------------------------------------------
    months_since_origination = rng.integers(3, tenure)
    days_past_due = np.clip(30 + rng.gamma(1.5, 70, n), 30, 720).round().astype(int)
    missed = np.clip(np.round(days_past_due / 30 + rng.normal(0, 0.8, n)), 1, None).astype(int)
    missed = np.minimum(missed, months_since_origination)

    paid = months_since_origination - missed
    outstanding = _balance_after(loan_amount, interest_rate, tenure, paid) * (1 + 0.015 * missed)
    outstanding = np.clip(outstanding, 1000, loan_amount * 1.3).round()

    # ---- Collateral -------------------------------------------------------
    u = rng.uniform(size=n)
    depreciation = np.clip(1 - 0.10 * months_since_origination / 12, 0.2, 1.0)
    collateral = np.select(
        [loan_type == "Home", loan_type == "Auto", (loan_type == "Business") & (u < 0.55)],
        [
            loan_amount * rng.uniform(1.1, 1.8, n),
            loan_amount * rng.uniform(0.5, 1.0, n) * depreciation,
            loan_amount * rng.uniform(0.3, 1.3, n),
        ],
        default=0.0,
    ).round(-3)

    # ---- Credit behaviour before this delinquency --------------------------
    p_on_time = _sigmoid((credit_score - 650) / 60)
    h = rng.uniform(size=n)
    payment_history = np.where(
        h < p_on_time, "On-Time", np.where(h < p_on_time + (1 - p_on_time) * 0.6, "Delayed", "Missed")
    )
    previous_defaults = np.clip(
        rng.poisson(0.2 + 0.6 * (credit_score < 600) + 0.3 * (payment_history == "Missed")), 0, 5
    )

    # ---- Hidden risk score -> outcome ------------------------------------
    emi_ratio = emi / income
    coverage = collateral / outstanding
    emp_effect = pd.Series(employment).map(
        {"Salaried": 0.0, "Self-Employed": 0.35, "Business Owner": 0.15, "Unemployed": 1.2}
    ).to_numpy()
    hist_effect = pd.Series(payment_history).map({"On-Time": 0.0, "Delayed": 0.25, "Missed": 0.6}).to_numpy()

    z = (
        -1.55
        + 1.10 * np.log(days_past_due / 90)
        + 1.30 * np.clip(emi_ratio - 0.4, -0.4, 2.0)
        - 1.20 * np.clip(coverage, 0, 1.5)
        + emp_effect
        - 0.90 * (credit_score - 650) / 100
        + 0.45 * previous_defaults
        + hist_effect
        + 0.90 * ((days_past_due > 180) & (coverage < 0.3))  # interaction: very late AND unsecured
        + 0.15 * dependents
        + rng.normal(0, 0.6, n)
    )
    p_loss = _sigmoid(z)
    written_off = rng.uniform(size=n) < p_loss

    p_full = _sigmoid(
        1.0 - 0.8 * np.log(days_past_due / 90) + 0.8 * np.clip(coverage, 0, 1.5) - 1.0 * np.clip(emi_ratio - 0.4, -0.4, 2.0)
    )
    fully = rng.uniform(size=n) < p_full
    recovery_status = np.where(written_off, "Written Off", np.where(fully, "Fully Recovered", "Partially Recovered"))

    recovered_amount = np.where(
        written_off,
        outstanding * rng.uniform(0.0, 0.25, n),
        np.where(fully, outstanding, outstanding * rng.uniform(0.35, 0.9, n)),
    ).round()

    # ---- Post-decision columns (leakage: NOT model features) --------------
    methods = _pick_per_row(
        rng,
        [
            ["Legal Notice", "Debt Collectors"] if p > 0.6
            else ["Calls", "Settlement Offer", "Debt Collectors"] if p > 0.3
            else ["Calls", "Settlement Offer"]
            for p in p_loss
        ],
    )
    collection_attempts = 1 + rng.poisson(1 + 4 * p_loss)
    legal_action = np.where(rng.uniform(size=n) < 0.05 + 0.6 * p_loss * (methods == "Legal Notice") + 0.2 * p_loss, "Yes", "No")

    return pd.DataFrame(
        {
            "Borrower_ID": [f"BRW{i:06d}" for i in range(1, n + 1)],
            "Loan_ID": [f"LN{i:07d}" for i in rng.permutation(np.arange(1, n + 1))],
            "Age": age,
            "Gender": gender,
            "Employment_Type": employment,
            "Monthly_Income": income,
            "Num_Dependents": dependents,
            "Credit_Score": credit_score.astype(int),
            "Loan_Type": loan_type,
            "Loan_Amount": loan_amount,
            "Loan_Tenure": tenure,
            "Interest_Rate": interest_rate,
            "Monthly_EMI": emi,
            "Months_Since_Origination": months_since_origination,
            "Collateral_Value": collateral,
            "Outstanding_Loan_Amount": outstanding,
            "Payment_History": payment_history,
            "Previous_Defaults": previous_defaults,
            "Num_Missed_Payments": missed,
            "Days_Past_Due": days_past_due,
            "Collection_Attempts": collection_attempts,
            "Collection_Method": methods,
            "Legal_Action_Taken": legal_action,
            "Recovered_Amount": recovered_amount,
            "Recovery_Status": recovery_status,
        }
    )


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Generate a synthetic overdue-loan portfolio.")
    parser.add_argument("--n", type=int, default=cfg["data"]["n_samples"])
    parser.add_argument("--seed", type=int, default=cfg["project"]["random_state"])
    parser.add_argument("--out", type=str, default=cfg["data"]["raw_path"])
    args = parser.parse_args()

    df = generate_loan_data(args.n, args.seed)
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} loans to {out}")
    print(df["Recovery_Status"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
