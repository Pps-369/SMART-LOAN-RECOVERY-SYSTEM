"""Column contracts and feature engineering.

Three groups of columns are deliberately kept OUT of the model:

* LEAKAGE_COLUMNS - decided or observed after recovery starts. The model's job is to help
  choose the recovery action, so it cannot use the action (or its result) as an input.
  Including them inflates test scores and fails in production.
* FAIRNESS_COLUMNS - protected attributes. Used only for a fairness check in the report.
* ID columns - identifiers carry no signal and would let the model memorise rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET_COLUMN = "Recovery_Status"
LOSS_LABEL = "Written Off"

ID_COLUMNS = ["Borrower_ID", "Loan_ID"]
LEAKAGE_COLUMNS = ["Collection_Attempts", "Collection_Method", "Legal_Action_Taken", "Recovered_Amount"]
FAIRNESS_COLUMNS = ["Gender"]

NUMERIC_BASE = [
    "Age",
    "Monthly_Income",
    "Num_Dependents",
    "Credit_Score",
    "Loan_Amount",
    "Loan_Tenure",
    "Interest_Rate",
    "Monthly_EMI",
    "Months_Since_Origination",
    "Collateral_Value",
    "Outstanding_Loan_Amount",
    "Previous_Defaults",
    "Num_Missed_Payments",
    "Days_Past_Due",
]
CATEGORICAL = ["Employment_Type", "Loan_Type", "Payment_History"]
REQUIRED_INPUT_COLUMNS = NUMERIC_BASE + CATEGORICAL

ENGINEERED = [
    "EMI_to_Income",          # affordability: can the borrower pay at all?
    "Collateral_Coverage",    # security: what does the bank get back if it enforces?
    "Outstanding_Ratio",      # how much of the loan is still unpaid
    "Loan_Age_Ratio",         # early-life delinquency is a stronger warning sign
    "Missed_Payment_Rate",    # missed installments relative to loan age
    "Debt_to_Annual_Income",  # size of the problem relative to earning power
    "Income_per_Dependent",   # household pressure on income
    "Has_Collateral",
    "Log_Days_Past_Due",      # DPD is right-skewed; log makes it friendlier for linear models
]
MODEL_NUMERIC = NUMERIC_BASE + ENGINEERED

CATEGORY_VALUES = {
    "Employment_Type": ["Salaried", "Self-Employed", "Business Owner", "Unemployed"],
    "Loan_Type": ["Home", "Auto", "Personal", "Business"],
    "Payment_History": ["On-Time", "Delayed", "Missed"],
}


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("Input has no rows.")


def make_target(df: pd.DataFrame) -> pd.Series:
    """1 = loan was written off (loss), 0 = fully or partially recovered."""
    return (df[TARGET_COLUMN] == LOSS_LABEL).astype(int).rename("is_loss")


def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Whitelist of columns the model may see. Whitelisting beats blacklisting for leakage."""
    return df[REQUIRED_INPUT_COLUMNS].copy()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_input(df)
    out = df.copy()
    for col in NUMERIC_BASE:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL:
        values = out[col].astype(object)
        cleaned = values.astype(str).str.strip().astype(object)
        cleaned[values.isna().to_numpy()] = np.nan  # keep a real NaN so the imputer can see it
        out[col] = cleaned

    income = out["Monthly_Income"].clip(lower=1)
    outstanding = out["Outstanding_Loan_Amount"].clip(lower=1)
    months = out["Months_Since_Origination"].clip(lower=1)

    out["EMI_to_Income"] = (out["Monthly_EMI"] / income).clip(upper=10)
    out["Collateral_Coverage"] = (out["Collateral_Value"] / outstanding).clip(lower=0, upper=5)
    out["Outstanding_Ratio"] = (out["Outstanding_Loan_Amount"] / out["Loan_Amount"].clip(lower=1)).clip(upper=3)
    out["Loan_Age_Ratio"] = (out["Months_Since_Origination"] / out["Loan_Tenure"].clip(lower=1)).clip(0, 1.5)
    out["Missed_Payment_Rate"] = (out["Num_Missed_Payments"] / months).clip(0, 1)
    out["Debt_to_Annual_Income"] = (out["Outstanding_Loan_Amount"] / (income * 12)).clip(upper=50)
    out["Income_per_Dependent"] = out["Monthly_Income"] / (1 + out["Num_Dependents"].clip(lower=0))
    out["Has_Collateral"] = (out["Collateral_Value"] > 0).astype(int)
    out["Log_Days_Past_Due"] = np.log1p(out["Days_Past_Due"].clip(lower=0))
    return out.replace([np.inf, -np.inf], np.nan)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Pipeline step so training and serving compute features with the exact same code."""

    def fit(self, X: pd.DataFrame, y=None):
        validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_engineered_features(X)
