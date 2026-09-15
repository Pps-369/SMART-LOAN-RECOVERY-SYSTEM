import numpy as np
import pandas as pd
import pytest

from loan_recovery.data_generator import generate_loan_data
from loan_recovery.features import (
    ENGINEERED,
    FAIRNESS_COLUMNS,
    LEAKAGE_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    add_engineered_features,
    make_target,
    model_inputs,
)


def test_generator_is_reproducible():
    a, b = generate_loan_data(300, 1), generate_loan_data(300, 1)
    pd.testing.assert_frame_equal(a, b)


def test_generator_schema_and_ranges(small_df):
    for col in REQUIRED_INPUT_COLUMNS + LEAKAGE_COLUMNS + FAIRNESS_COLUMNS + ["Recovery_Status"]:
        assert col in small_df.columns
    assert small_df[REQUIRED_INPUT_COLUMNS].notna().all().all()
    assert small_df["Days_Past_Due"].min() >= 30          # every loan is overdue
    assert small_df["Credit_Score"].between(300, 900).all()
    assert (small_df["Num_Missed_Payments"] <= small_df["Months_Since_Origination"]).all()
    assert small_df["Loan_ID"].is_unique


def test_loss_rate_is_realistic(small_df):
    assert 0.15 < make_target(small_df).mean() < 0.45


def test_model_inputs_exclude_leakage_and_protected(small_df):
    cols = set(model_inputs(small_df).columns)
    assert not cols & set(LEAKAGE_COLUMNS)
    assert not cols & set(FAIRNESS_COLUMNS)
    assert "Loan_ID" not in cols


def test_engineered_features_are_finite(small_df):
    eng = add_engineered_features(small_df)
    assert set(ENGINEERED).issubset(eng.columns)
    assert np.isfinite(eng[ENGINEERED].to_numpy(dtype=float)).all()


def test_engineered_feature_values():
    row = generate_loan_data(1, 3)
    row.loc[0, ["Monthly_EMI", "Monthly_Income", "Collateral_Value", "Outstanding_Loan_Amount"]] = [20000, 40000, 0, 100000]
    eng = add_engineered_features(row)
    assert eng.loc[0, "EMI_to_Income"] == pytest.approx(0.5)
    assert eng.loc[0, "Collateral_Coverage"] == 0
    assert eng.loc[0, "Has_Collateral"] == 0


def test_zero_income_does_not_create_inf():
    row = generate_loan_data(1, 3)
    row.loc[0, "Monthly_Income"] = 0
    eng = add_engineered_features(row)
    assert np.isfinite(eng.loc[0, "EMI_to_Income"])


def test_missing_column_raises(small_df):
    with pytest.raises(ValueError, match="missing required columns"):
        add_engineered_features(small_df.drop(columns=["Days_Past_Due"]))
