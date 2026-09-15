"""Shared fixtures: one small, fast model trained once per test session."""
import pandas as pd
import pytest

from loan_recovery.config import load_config
from loan_recovery.data_generator import generate_loan_data
from loan_recovery.train import run_training


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_df():
    return generate_loan_data(n=2500, random_state=7)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, cfg, small_df):
    out = tmp_path_factory.mktemp("artifacts")
    model_path = out / "model.joblib"
    report = run_training(small_df, cfg, model_path=model_path, reports_dir=out / "reports",
                          demo_path=out / "demo.csv", fast=True)
    return {"model_path": model_path, "report": report, "dir": out}


@pytest.fixture(scope="session")
def predictor(trained):
    from loan_recovery.predict import LoanRecoveryPredictor
    return LoanRecoveryPredictor.load(trained["model_path"])


@pytest.fixture()
def sample_rows(small_df) -> pd.DataFrame:
    return small_df.head(25).copy()
