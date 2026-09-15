import numpy as np
import pandas as pd

from loan_recovery.features import REQUIRED_INPUT_COLUMNS


def test_training_produces_artifacts(trained):
    d = trained["dir"]
    assert trained["model_path"].exists()
    for name in ["metrics.json", "training_summary.md", "segment_profiles.csv", "model_comparison.csv"]:
        assert (d / "reports" / name).exists(), name
    assert (d / "reports" / "figures" / "roc_pr_curves.png").exists()
    assert (d / "demo.csv").exists()


def test_model_beats_random_and_dpd_rule(trained):
    r = trained["report"]
    assert r["test_metrics"]["roc_auc"] > 0.70
    assert r["test_metrics"]["roc_auc"] > r["dpd_baseline_roc_auc"]


def test_segments_are_ordered_by_risk(trained):
    segs = sorted(trained["report"]["segments"].values(), key=lambda s: s["rank"])
    rates = [s["train_loss_rate"] for s in segs]
    assert rates == sorted(rates)
    assert 3 <= len(segs) <= 6


def test_score_output_contract(predictor, sample_rows):
    out = predictor.score(sample_rows)
    assert len(out) == len(sample_rows)
    assert out["P_Loss"].between(0, 1).all()
    assert np.allclose(out["P_Loss"] + out["P_Recovery"], 1)
    assert set(out["Risk_Band"].astype(str)) <= {"Low", "Medium", "High"}
    assert out["Recommended_Action"].notna().all()
    assert sorted(out["Priority_Rank"]) == list(range(1, len(out) + 1))
    assert (out["Expected_Loss"] <= out["Outstanding_Loan_Amount"] + 1).all()


def test_scoring_ignores_leakage_columns(predictor, sample_rows):
    a = predictor.score(sample_rows)["P_Loss"]
    tampered = sample_rows.assign(Legal_Action_Taken="Yes", Collection_Attempts=99, Recovered_Amount=0)
    b = predictor.score(tampered)["P_Loss"]
    pd.testing.assert_series_equal(a, b)


def test_scoring_only_required_columns(predictor, sample_rows):
    out = predictor.score(sample_rows[REQUIRED_INPUT_COLUMNS])
    assert "Loan_ID" not in out.columns and len(out) == len(sample_rows)


def test_scoring_handles_missing_values_and_unseen_category(predictor, sample_rows):
    rows = sample_rows.copy()
    rows.loc[rows.index[0], "Credit_Score"] = np.nan
    rows.loc[rows.index[1], "Loan_Type"] = "Gold"
    out = predictor.score(rows)
    assert out["P_Loss"].notna().all()


def test_riskier_loan_scores_higher(predictor, sample_rows):
    base = sample_rows.iloc[[0]].copy()
    safe = base.assign(Days_Past_Due=35, Num_Missed_Payments=1, Credit_Score=800, Previous_Defaults=0,
                       Employment_Type="Salaried", Payment_History="On-Time",
                       Monthly_EMI=base["Monthly_Income"] * 0.1)
    risky = base.assign(Days_Past_Due=400, Num_Missed_Payments=12, Credit_Score=480, Previous_Defaults=3,
                        Employment_Type="Unemployed", Payment_History="Missed", Collateral_Value=0,
                        Monthly_EMI=base["Monthly_Income"] * 1.2)
    risky["Months_Since_Origination"] = risky["Months_Since_Origination"].clip(lower=12)
    assert predictor.score(risky)["P_Loss"].iloc[0] > predictor.score(safe)["P_Loss"].iloc[0]
