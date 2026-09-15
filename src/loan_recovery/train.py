"""End-to-end training.

    python -m loan_recovery.train            # full run
    python -m loan_recovery.train --fast     # quick smoke run (fewer models / trees)

Steps
 1. Load data (generate it if missing) and split train / validation / test (stratified).
 2. Segment borrowers with KMeans; choose k by silhouette on TRAIN only.
 3. Compare candidate classifiers with stratified cross-validation on TRAIN.
 4. Calibrate the winner so its probabilities can be used in money calculations.
 5. Pick the escalation threshold that maximises net recovery value on VALIDATION.
 6. Evaluate once on TEST: ranking, calibration, business value, fairness, importance.
 7. Save one artifact bundle + reports.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import evaluation as ev
from .clustering import BorrowerSegmenter, name_segments, profile_segments, select_k
from .config import load_config, resolve_path
from .data_generator import generate_loan_data
from .features import (
    FAIRNESS_COLUMNS,
    LEAKAGE_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    TARGET_COLUMN,
    add_engineered_features,
    make_target,
    model_inputs,
)
from .models import build_candidates, build_pipeline
from .strategy import escalation_value, optimize_threshold


def load_data(cfg: dict, regenerate: bool = False, n_samples: int | None = None) -> pd.DataFrame:
    path = resolve_path(cfg["data"]["raw_path"])
    if regenerate or not path.exists():
        n = n_samples or cfg["data"]["n_samples"]
        print(f"[data] generating {n:,} synthetic loans -> {path}")
        df = generate_loan_data(n, cfg["project"]["random_state"])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
    print(f"[data] loading {path}")
    return pd.read_csv(path)


def run_training(
    df: pd.DataFrame,
    cfg: dict,
    model_path: Path,
    reports_dir: Path,
    demo_path: Path | None = None,
    fast: bool = False,
) -> dict:
    t0 = time.time()
    rs = cfg["project"]["random_state"]
    mcfg, scfg, ccfg = cfg["model"], cfg["strategy"], cfg["clustering"]
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    missing = [c for c in REQUIRED_INPUT_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    print(f"[data] {len(df):,} loans | excluded as leakage: {LEAKAGE_COLUMNS} | fairness-only: {FAIRNESS_COLUMNS}")

    # ---- 1. split -------------------------------------------------------
    y = make_target(df)
    idx_trval, idx_test = train_test_split(df.index, test_size=cfg["data"]["test_size"], stratify=y, random_state=rs)
    idx_train, idx_val = train_test_split(
        idx_trval, test_size=cfg["data"]["val_size"], stratify=y.loc[idx_trval], random_state=rs
    )
    X_all = model_inputs(df)
    X_train, X_val, X_test = X_all.loc[idx_train], X_all.loc[idx_val], X_all.loc[idx_test]
    y_train, y_val, y_test = y.loc[idx_train], y.loc[idx_val], y.loc[idx_test]
    print(f"[split] train={len(X_train):,} val={len(X_val):,} test={len(X_test):,} | loss rate={y.mean():.1%}")

    # ---- 2. segmentation -------------------------------------------------
    k_max = min(ccfg["k_max"], ccfg["k_min"] + 1) if fast else ccfg["k_max"]
    eng_train = add_engineered_features(X_train)
    best_k, k_scores = select_k(eng_train, ccfg["k_min"], k_max, ccfg["silhouette_sample"], rs)
    segmenter = BorrowerSegmenter(best_k, rs).fit(eng_train)
    train_labels = segmenter.predict(eng_train)
    segment_info = name_segments(train_labels, y_train)
    seg_profile = profile_segments(eng_train, train_labels, y_train, segment_info)
    print(f"[segments] k={best_k}\n{seg_profile[['Segment', 'Loans', 'Share', 'Loss_Rate']].to_string()}")
    ev.plot_k_selection(k_scores, best_k, fig_dir / "segment_k_selection.png")
    ev.plot_segments(seg_profile, fig_dir / "segment_profiles.png")
    seg_profile.to_csv(reports_dir / "segment_profiles.csv")

    # ---- 3. model selection ----------------------------------------------
    cv = StratifiedKFold(n_splits=3 if fast else mcfg["cv_folds"], shuffle=True, random_state=rs)
    rows = []
    for name, est in build_candidates(rs, fast).items():
        pipe = build_pipeline(est, best_k, rs)
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=1)
        rows.append({
            "model": name,
            "roc_auc": res["test_roc_auc"].mean(),
            "roc_auc_std": res["test_roc_auc"].std(),
            "average_precision": res["test_average_precision"].mean(),
            "average_precision_std": res["test_average_precision"].std(),
            "fit_seconds": res["fit_time"].mean(),
        })
        print(f"[cv] {name:<24} ROC-AUC {rows[-1]['roc_auc']:.4f}  PR-AUC {rows[-1]['average_precision']:.4f}")
    cv_table = pd.DataFrame(rows).sort_values(mcfg["selection_metric"], ascending=False).round(4)
    cv_table.to_csv(reports_dir / "model_comparison.csv", index=False)
    best_name = cv_table.iloc[0]["model"]
    print(f"[cv] selected: {best_name}")

    # ---- 4. calibrate ---------------------------------------------------
    best_pipe = build_pipeline(build_candidates(rs, fast)[best_name], best_k, rs)
    model = CalibratedClassifierCV(best_pipe, method=mcfg["calibration_method"], cv=mcfg["calibration_cv"])
    model.fit(X_train, y_train)

    # ---- 5. business threshold on validation ------------------------------
    p_val = model.predict_proba(X_val)[:, 1]
    threshold, curve = optimize_threshold(
        y_val, p_val, X_val["Outstanding_Loan_Amount"], scfg["escalation_cost"], scfg["escalation_recovery_uplift"]
    )
    medium_threshold = round(threshold * scfg["medium_band_ratio"], 3)
    curve.to_csv(reports_dir / "threshold_curve.csv", index=False)
    ev.plot_threshold_curve(curve, threshold, fig_dir / "threshold_value_curve.png")
    print(f"[threshold] high-risk >= {threshold:.2f} | medium >= {medium_threshold:.3f}")

    # ---- 6. test evaluation ---------------------------------------------
    p_test = model.predict_proba(X_test)[:, 1]
    metrics = ev.classification_metrics(y_test, p_test, threshold)
    baseline_auc = float(roc_auc_score(y_test, X_test["Days_Past_Due"]))
    out_test = X_test["Outstanding_Loan_Amount"].to_numpy()
    cost, uplift = scfg["escalation_cost"], scfg["escalation_recovery_uplift"]
    business = {
        "model_net_value": escalation_value(y_test, p_test, out_test, threshold, cost, uplift),
        "escalate_all_net_value": escalation_value(y_test, p_test, out_test, 0.0, cost, uplift),
        "escalate_none_net_value": 0.0,
        "dpd_90_rule_net_value": escalation_value(
            y_test, (X_test["Days_Past_Due"] >= 90).astype(float), out_test, 0.5, cost, uplift
        ),
        "perfect_foresight_net_value": escalation_value(y_test, y_test.astype(float), out_test, 0.5, cost, uplift),
        "loans_escalated": int((p_test >= threshold).sum()),
        "expected_loss_test_portfolio": float((out_test * p_test).sum()),
    }
    business = {k: round(v, 0) if isinstance(v, float) else v for k, v in business.items()}

    ev.plot_roc_pr(y_test, p_test, X_test["Days_Past_Due"], fig_dir / "roc_pr_curves.png")
    ev.plot_calibration(y_test, p_test, fig_dir / "calibration_curve.png")
    ev.plot_confusion(metrics["confusion_matrix"], fig_dir / "confusion_matrix.png")

    # fairness check: Gender is never a model input, but outcomes can still differ by group
    fairness = None
    if "Gender" in df.columns:
        g = df.loc[idx_test, "Gender"]
        fairness = (
            pd.DataFrame({"Gender": g.values, "flagged": p_test >= threshold, "actual_loss": y_test.values})
            .groupby("Gender")
            .agg(loans=("flagged", "size"), flagged_rate=("flagged", "mean"), actual_loss_rate=("actual_loss", "mean"))
            .round(4)
        )
        print(f"[fairness]\n{fairness.to_string()}")

    n_perm = min(len(X_test), 1000 if fast else 3000)
    X_perm = X_test.sample(n_perm, random_state=rs)
    perm = permutation_importance(
        model, X_perm, y_test.loc[X_perm.index], scoring="average_precision",
        n_repeats=2 if fast else mcfg["permutation_repeats"], random_state=rs, n_jobs=1,
    )
    importance = pd.DataFrame({
        "feature": X_perm.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).round(5)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    ev.plot_importance(importance, fig_dir / "feature_importance.png")

    # ---- 7. save --------------------------------------------------------
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "best_model": best_name,
        "calibration": mcfg["calibration_method"],
        "n_segments": best_k,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "high_risk_threshold": threshold,
        "medium_risk_threshold": medium_threshold,
        "test_metrics": metrics,
        "dpd_baseline_roc_auc": round(baseline_auc, 4),
        "business_value_test": business,
        "fast_mode": fast,
    }
    bundle = {
        "model": model,
        "segmenter": segmenter,
        "segment_info": segment_info,
        "threshold": threshold,
        "medium_threshold": medium_threshold,
        "strategy_config": scfg,
        "input_columns": REQUIRED_INPUT_COLUMNS,
        "metadata": metadata,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    report = {
        **metadata,
        "cv_comparison": cv_table.to_dict(orient="records"),
        "segment_k_scores": k_scores.round(4).to_dict(orient="records"),
        "segments": segment_info,
        "fairness_by_gender": None if fairness is None else fairness.reset_index().to_dict(orient="records"),
        "top_features": importance.head(10).to_dict(orient="records"),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary(report, reports_dir / "training_summary.md")

    if demo_path is not None:  # held-out loans for the dashboard demo (never used in training)
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        df.loc[idx_test].to_csv(demo_path, index=False)

    print(f"[test] ROC-AUC {metrics['roc_auc']} | PR-AUC {metrics['pr_auc']} | Brier {metrics['brier']} "
          f"| DPD-rule ROC-AUC {baseline_auc:.4f}")
    print(f"[business] net value: model {business['model_net_value']:,.0f} | escalate-all "
          f"{business['escalate_all_net_value']:,.0f} | DPD>=90 rule {business['dpd_90_rule_net_value']:,.0f}")
    print(f"[done] saved {model_path} in {time.time() - t0:.1f}s")
    return report


def _inr_cr(x: float) -> str:
    return f"INR {x / 1e7:,.2f} Cr"


def _write_summary(r: dict, path: Path) -> None:
    m, b = r["test_metrics"], r["business_value_test"]
    lines = [
        "# Training summary",
        "",
        (
            f"Trained {r['trained_at']} - model **{r['best_model']}** ({r['calibration']} calibrated), "
            f"{r['n_segments']} borrower segments."
        ),
        "",
        "## Model comparison (5-fold CV on train)" if not r["fast_mode"] else "## Model comparison (3-fold CV, fast mode)",
        "",
        "| Model | ROC-AUC | PR-AUC |",
        "|---|---|---|",
        *[f"| {c['model']} | {c['roc_auc']:.4f} | {c['average_precision']:.4f} |" for c in r["cv_comparison"]],
        "",
        "## Hold-out test set",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| ROC-AUC | {m['roc_auc']} (days-past-due rule: {r['dpd_baseline_roc_auc']}) |",
        f"| PR-AUC | {m['pr_auc']} (base loss rate: {m['base_loss_rate']}) |",
        f"| Brier score | {m['brier']} |",
        f"| Threshold (value-optimised) | {m['threshold']} |",
        f"| Precision / Recall at threshold | {m['precision']} / {m['recall']} |",
        "",
        "## Business value on the test portfolio",
        "",
        "| Policy | Net recovery value |",
        "|---|---|",
        f"| Model-driven escalation | {_inr_cr(b['model_net_value'])} |",
        f"| Escalate everything | {_inr_cr(b['escalate_all_net_value'])} |",
        f"| Escalate if DPD >= 90 | {_inr_cr(b['dpd_90_rule_net_value'])} |",
        f"| Perfect foresight (ceiling) | {_inr_cr(b['perfect_foresight_net_value'])} |",
        "",
        "Values depend on the cost and uplift assumptions in `config.yaml`.",
        "",
        "## Segments",
        "",
        "| Segment | Train loss rate |",
        "|---|---|",
        *[f"| {s['name']} | {s['train_loss_rate']:.1%} |" for s in sorted(r["segments"].values(), key=lambda s: s["rank"])],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Smart Loan Recovery model.")
    parser.add_argument("--fast", action="store_true", help="quick run with fewer models and trees")
    parser.add_argument("--regenerate", action="store_true", help="regenerate the synthetic dataset")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg["project"]["random_state"])
    df = load_data(cfg, regenerate=args.regenerate, n_samples=args.n_samples)
    run_training(
        df,
        cfg,
        model_path=resolve_path(cfg["paths"]["model"]),
        reports_dir=resolve_path(cfg["paths"]["reports"]),
        demo_path=resolve_path(cfg["data"]["demo_portfolio_path"]),
        fast=args.fast,
    )


if __name__ == "__main__":
    main()
