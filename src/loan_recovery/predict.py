"""Scoring: load the saved bundle and turn raw loan rows into recovery decisions.

    python -m loan_recovery.predict --input data/processed/demo_portfolio.csv --output reports/scored.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd

from .config import load_config, resolve_path
from .features import ID_COLUMNS, add_engineered_features, model_inputs, validate_input
from .strategy import BAND_ORDER, assign_risk_band, recommend_actions


def default_model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH") or resolve_path(load_config()["paths"]["model"]))


class LoanRecoveryPredictor:
    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.segmenter = bundle["segmenter"]
        self.segment_info = {int(k): v for k, v in bundle["segment_info"].items()}
        self.threshold = float(bundle["threshold"])
        self.medium_threshold = float(bundle["medium_threshold"])
        self.strategy_config = bundle["strategy_config"]
        self.metadata = bundle["metadata"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LoanRecoveryPredictor":
        path = Path(path) if path else default_model_path()
        if not path.exists():
            raise FileNotFoundError(f"No model at {path}. Train one first: python -m loan_recovery.train")
        return cls(joblib.load(path))

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_input(df)
        df = df.reset_index(drop=True)
        X = model_inputs(df)
        eng = add_engineered_features(X)

        # isotonic calibration can output exact 0/1; no loan is ever certain
        p_loss = self.model.predict_proba(X)[:, 1].clip(0.001, 0.999)
        bands = assign_risk_band(p_loss, self.threshold, self.medium_threshold)
        seg_ids = self.segmenter.predict(eng)

        out = df[[c for c in ID_COLUMNS if c in df.columns]].copy()
        out["P_Loss"] = p_loss.round(4)
        out["P_Recovery"] = (1 - p_loss).round(4)
        out["Risk_Band"] = pd.Categorical(bands, categories=BAND_ORDER, ordered=True)
        out["Segment_ID"] = seg_ids.astype(int)
        out["Segment"] = [self.segment_info[int(s)]["name"] for s in seg_ids]
        out["Outstanding_Loan_Amount"] = eng["Outstanding_Loan_Amount"]
        out["Days_Past_Due"] = eng["Days_Past_Due"]
        out["Collateral_Coverage"] = eng["Collateral_Coverage"].round(3)
        out["EMI_to_Income"] = eng["EMI_to_Income"].round(3)
        out = pd.concat([out, recommend_actions(eng, p_loss, bands, self.strategy_config)], axis=1)

        # work queue: loans where escalation earns the most come first
        out["Priority_Rank"] = out["Net_Escalation_Value"].rank(ascending=False, method="first").astype(int)
        return out

    def score_one(self, record: dict) -> dict:
        return self.score(pd.DataFrame([record])).iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a CSV of overdue loans.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/scored_portfolio.csv")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    predictor = LoanRecoveryPredictor.load(args.model)
    scored = predictor.score(pd.read_csv(resolve_path(args.input))).sort_values("Priority_Rank")
    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    print(f"Scored {len(scored):,} loans -> {out}")
    print(scored["Recommended_Action"].value_counts().to_string())


if __name__ == "__main__":
    main()
