"""Smart Loan Recovery - collections dashboard.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .`

from loan_recovery.config import load_config, resolve_path  # noqa: E402
from loan_recovery.features import CATEGORY_VALUES, REQUIRED_INPUT_COLUMNS  # noqa: E402
from loan_recovery.predict import LoanRecoveryPredictor  # noqa: E402
from loan_recovery.strategy import ACTIONS, BAND_ORDER  # noqa: E402

st.set_page_config(page_title="Smart Loan Recovery", page_icon="🏦", layout="wide")
CFG = load_config()
REPORTS = resolve_path(CFG["paths"]["reports"])


def inr(x: float) -> str:
    sign, x = ("-" if x < 0 else ""), abs(x)
    if x >= 1e7:
        return f"{sign}₹{x / 1e7:,.2f} Cr"
    if x >= 1e5:
        return f"{sign}₹{x / 1e5:,.2f} L"
    return f"{sign}₹{x:,.0f}"


@st.cache_resource
def get_predictor():
    try:
        return LoanRecoveryPredictor.load()
    except FileNotFoundError:
        return None


@st.cache_data
def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    return get_predictor().score(df)


predictor = get_predictor()
st.title("🏦 Smart Loan Recovery System")
st.caption("Predict which overdue loans will be recovered, segment borrowers and prioritise recovery actions.")

if predictor is None:
    st.warning("No trained model found yet. Train one here (about a minute) or run `python -m loan_recovery.train`.")
    if st.button("Train model now", type="primary"):
        from loan_recovery.train import load_data, run_training

        with st.spinner("Generating data and training... (segmentation, 3 models, calibration)"):
            run_training(
                load_data(CFG),
                CFG,
                model_path=resolve_path(CFG["paths"]["model"]),
                reports_dir=REPORTS,
                demo_path=resolve_path(CFG["data"]["demo_portfolio_path"]),
            )
        get_predictor.clear()
        st.rerun()
    st.stop()

# ---------------- Sidebar: data source ----------------
with st.sidebar:
    st.header("Portfolio data")
    source = st.radio("Source", ["Demo portfolio (held-out test loans)", "Upload CSV"])
    raw = None
    if source.startswith("Demo"):
        demo_path = resolve_path(CFG["data"]["demo_portfolio_path"])
        if demo_path.exists():
            raw = pd.read_csv(demo_path)
        else:
            st.warning("Demo file missing - run training to create it.")
    else:
        upload = st.file_uploader("CSV with the required columns", type="csv")
        with st.expander("Required columns"):
            st.code(", ".join(REQUIRED_INPUT_COLUMNS))
        if upload is not None:
            raw = pd.read_csv(upload)

    st.divider()
    meta = predictor.metadata
    st.subheader("Model")
    st.write(f"**{meta['best_model']}** ({meta['calibration']} calibrated)")
    st.write(f"Trained: {meta['trained_at']}")
    st.write(f"High-risk threshold: **{predictor.threshold:.2f}**")
    st.write(f"Medium-risk threshold: **{predictor.medium_threshold:.3f}**")

tab_overview, tab_queue, tab_segments, tab_single, tab_model = st.tabs(
    ["📊 Portfolio overview", "📋 Collections queue", "🧩 Segments", "🔎 Score a borrower", "📈 Model performance"]
)

scored = None
if raw is not None:
    try:
        scored = score_frame(raw)
    except ValueError as exc:
        st.error(str(exc))

# ---------------- Overview ----------------
with tab_overview:
    if scored is None:
        st.info("Choose or upload a portfolio in the sidebar.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overdue loans", f"{len(scored):,}")
        c2.metric("Total outstanding", inr(scored["Outstanding_Loan_Amount"].sum()))
        c3.metric("Expected loss", inr(scored["Expected_Loss"].sum()))
        c4.metric("High-risk loans", f"{(scored['Risk_Band'] == 'High').mean():.1%}")

        left, right = st.columns(2)
        band = (
            scored.groupby("Risk_Band", observed=False)
            .agg(Loans=("P_Loss", "size"), Expected_Loss_Cr=("Expected_Loss", "sum"))
            .reindex(BAND_ORDER)
            .assign(Expected_Loss_Cr=lambda d: d["Expected_Loss_Cr"] / 1e7)
        )
        left.subheader("Loans by risk band")
        left.bar_chart(band["Loans"])
        right.subheader("Expected loss by risk band (₹ Cr)")
        right.bar_chart(band["Expected_Loss_Cr"])

        action = (
            scored.groupby("Recommended_Action")
            .agg(Loans=("P_Loss", "size"), Outstanding_Cr=("Outstanding_Loan_Amount", "sum"),
                 Expected_Loss_Cr=("Expected_Loss", "sum"), Avg_P_Loss=("P_Loss", "mean"))
            .assign(Outstanding_Cr=lambda d: d["Outstanding_Cr"] / 1e7, Expected_Loss_Cr=lambda d: d["Expected_Loss_Cr"] / 1e7)
            .sort_values("Expected_Loss_Cr", ascending=False)
        )
        st.subheader("Recovery strategy mix")
        st.dataframe(action.style.format({"Outstanding_Cr": "{:.2f}", "Expected_Loss_Cr": "{:.2f}", "Avg_P_Loss": "{:.1%}"}))
        with st.expander("What does each action mean?"):
            for name, detail in ACTIONS.items():
                st.markdown(f"**{name}** - {detail}")

        if "Recovery_Status" in raw.columns:
            st.subheader("Reality check (demo data has known outcomes)")
            check = pd.DataFrame({"Risk_Band": scored["Risk_Band"].astype(str),
                                  "Written_Off": (raw["Recovery_Status"] == "Written Off").to_numpy()})
            st.dataframe(check.groupby("Risk_Band")["Written_Off"].agg(["size", "mean"])
                         .reindex(BAND_ORDER).rename(columns={"size": "Loans", "mean": "Actual loss rate"})
                         .style.format({"Actual loss rate": "{:.1%}"}))

# ---------------- Queue ----------------
with tab_queue:
    if scored is None:
        st.info("Choose or upload a portfolio in the sidebar.")
    else:
        f1, f2, f3 = st.columns(3)
        bands = f1.multiselect("Risk band", BAND_ORDER, default=["High", "Medium"])
        actions = f2.multiselect("Action", sorted(scored["Recommended_Action"].unique()))
        top_n = f3.number_input("Show top N", min_value=10, max_value=max(len(scored), 10), value=min(100, max(len(scored), 10)), step=10)
        q = scored[scored["Risk_Band"].astype(str).isin(bands)] if bands else scored
        if actions:
            q = q[q["Recommended_Action"].isin(actions)]
        q = q.sort_values("Priority_Rank").head(int(top_n))
        cols = [c for c in ["Priority_Rank", "Loan_ID", "Borrower_ID", "Risk_Band", "P_Loss", "Segment",
                            "Outstanding_Loan_Amount", "Days_Past_Due", "Expected_Loss", "Net_Escalation_Value",
                            "Recommended_Action", "Reason_Codes"] if c in q.columns]
        st.dataframe(q[cols], hide_index=True,
                     column_config={"P_Loss": st.column_config.ProgressColumn("P(Loss)", min_value=0.0, max_value=1.0, format="%.2f")})
        st.download_button("⬇️ Download full scored portfolio", scored.sort_values("Priority_Rank").to_csv(index=False),
                           file_name="scored_portfolio.csv", mime="text/csv")

# ---------------- Segments ----------------
with tab_segments:
    st.subheader("Borrower segments (KMeans)")
    prof_path = REPORTS / "segment_profiles.csv"
    if prof_path.exists():
        st.caption("Profiles on the training data")
        st.dataframe(pd.read_csv(prof_path), hide_index=True)
    fig = REPORTS / "figures" / "segment_profiles.png"
    if fig.exists():
        st.image(str(fig))
    if scored is not None:
        st.caption("Current portfolio by segment")
        seg = (scored.groupby("Segment").agg(Loans=("P_Loss", "size"), Avg_P_Loss=("P_Loss", "mean"),
                                             Outstanding_Cr=("Outstanding_Loan_Amount", "sum"),
                                             Expected_Loss_Cr=("Expected_Loss", "sum"))
               .assign(Outstanding_Cr=lambda d: d["Outstanding_Cr"] / 1e7, Expected_Loss_Cr=lambda d: d["Expected_Loss_Cr"] / 1e7))
        st.dataframe(seg.style.format({"Avg_P_Loss": "{:.1%}", "Outstanding_Cr": "{:.2f}", "Expected_Loss_Cr": "{:.2f}"}))
        st.bar_chart(seg["Avg_P_Loss"])

# ---------------- Single borrower ----------------
with tab_single:
    st.subheader("Score one overdue loan")
    with st.form("single"):
        a, b, c = st.columns(3)
        rec = {
            "Age": a.number_input("Age", 18, 100, 38),
            "Employment_Type": a.selectbox("Employment type", CATEGORY_VALUES["Employment_Type"], index=1),
            "Monthly_Income": a.number_input("Monthly income (₹)", 1000, 10_000_000, 42000, step=1000),
            "Num_Dependents": a.number_input("Dependents", 0, 20, 2),
            "Credit_Score": a.number_input("Credit score", 300, 900, 598),
            "Payment_History": a.selectbox("Payment history", CATEGORY_VALUES["Payment_History"], index=1),
            "Loan_Type": b.selectbox("Loan type", CATEGORY_VALUES["Loan_Type"], index=2),
            "Loan_Amount": b.number_input("Loan amount (₹)", 10000, 500_000_000, 450000, step=10000),
            "Loan_Tenure": b.number_input("Tenure (months)", 1, 480, 48),
            "Interest_Rate": b.number_input("Interest rate (%)", 0.0, 60.0, 14.2, step=0.1),
            "Monthly_EMI": b.number_input("Monthly EMI (₹)", 100, 50_000_000, 12400, step=100),
            "Months_Since_Origination": c.number_input("Months since origination", 0, 480, 20),
            "Collateral_Value": c.number_input("Collateral value (₹)", 0, 1_000_000_000, 0, step=10000),
            "Outstanding_Loan_Amount": c.number_input("Outstanding amount (₹)", 1, 500_000_000, 318000, step=10000),
            "Num_Missed_Payments": c.number_input("Missed installments", 0, 480, 7),
            "Days_Past_Due": c.number_input("Days past due", 0, 3650, 210),
            "Previous_Defaults": c.number_input("Previous defaults", 0, 20, 1),
        }
        submitted = st.form_submit_button("Score loan", type="primary")
    if submitted:
        r = predictor.score_one(rec)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Probability of loss", f"{r['P_Loss']:.1%}")
        m2.metric("Risk band", str(r["Risk_Band"]))
        m3.metric("Expected loss", inr(r["Expected_Loss"]))
        m4.metric("Net value of escalating", inr(r["Net_Escalation_Value"]))
        st.success(f"**Recommended action: {r['Recommended_Action']}** - {r['Action_Detail']}")
        st.write(f"**Segment:** {r['Segment']}")
        st.write("**Risk flags:**")
        for flag in r["Reason_Codes"].split("; "):
            st.markdown(f"- {flag}")
        st.caption("Risk flags are business rules that explain the situation; the probability comes from the ML model.")

# ---------------- Model performance ----------------
with tab_model:
    metrics_path = REPORTS / "metrics.json"
    if not metrics_path.exists():
        st.info("Run training to generate reports.")
    else:
        rep = json.loads(metrics_path.read_text(encoding="utf-8"))
        m, bv = rep["test_metrics"], rep["business_value_test"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ROC-AUC (test)", m["roc_auc"], f"{m['roc_auc'] - rep['dpd_baseline_roc_auc']:+.3f} vs DPD rule")
        k2.metric("PR-AUC (test)", m["pr_auc"], f"base rate {m['base_loss_rate']:.1%}", delta_color="off")
        k3.metric("Brier score", m["brier"])
        k4.metric("Net value vs DPD≥90 rule", inr(bv["model_net_value"] - bv["dpd_90_rule_net_value"]))
        st.subheader("Model comparison (cross-validation)")
        st.dataframe(pd.DataFrame(rep["cv_comparison"]), hide_index=True)
        figs = ["roc_pr_curves.png", "threshold_value_curve.png", "calibration_curve.png",
                "confusion_matrix.png", "feature_importance.png", "segment_k_selection.png"]
        cols = st.columns(2)
        for i, name in enumerate(figs):
            path = REPORTS / "figures" / name
            if path.exists():
                cols[i % 2].image(str(path))
