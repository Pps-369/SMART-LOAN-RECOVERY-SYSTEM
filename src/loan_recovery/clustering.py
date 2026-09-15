"""Borrower segmentation (unsupervised).

Why cluster at all when we already have a classifier?
* Collections teams are organised by *portfolio segment*, not by individual probability.
  A handful of named segments with clear profiles is something a business can staff and plan for.
* The distance of a loan to each segment centre is a compact "behavioural profile" that we also
  feed to the classifier (SegmentDistanceAdder). It is fitted inside the pipeline, so during
  cross-validation it is re-fitted on each training fold and never sees validation data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .features import ENGINEERED, add_engineered_features

CLUSTER_FEATURES = [
    "Log_Days_Past_Due",
    "Num_Missed_Payments",
    "EMI_to_Income",
    "Collateral_Coverage",
    "Outstanding_Ratio",
    "Credit_Score",
    "Debt_to_Annual_Income",
    "Previous_Defaults",
]
_LOG_FEATURES = ["EMI_to_Income", "Debt_to_Annual_Income", "Collateral_Coverage"]
RISK_TIERS = ["Low Risk", "Guarded", "Moderate Risk", "Elevated Risk", "High Risk", "Severe Risk"]


def _ensure_engineered(X: pd.DataFrame) -> pd.DataFrame:
    return X if set(ENGINEERED).issubset(X.columns) else add_engineered_features(X)


class BorrowerSegmenter(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        Z = _ensure_engineered(X)[CLUSTER_FEATURES].astype(float).fillna(self.medians_)
        for col in _LOG_FEATURES:
            Z[col] = np.log1p(Z[col].clip(lower=0))  # tame long tails so they don't dominate distances
        return self.scaler_.transform(Z)

    def fit(self, X: pd.DataFrame, y=None):
        X = _ensure_engineered(X)
        self.medians_ = X[CLUSTER_FEATURES].astype(float).median()
        Z = X[CLUSTER_FEATURES].astype(float).fillna(self.medians_)
        for col in _LOG_FEATURES:
            Z[col] = np.log1p(Z[col].clip(lower=0))
        self.scaler_ = StandardScaler().fit(Z)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(self.scaler_.transform(Z))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.kmeans_.predict(self._matrix(X))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Distance of each loan to each segment centre."""
        return self.kmeans_.transform(self._matrix(X))


class SegmentDistanceAdder(BaseEstimator, TransformerMixin):
    """Pipeline step: appends seg_dist_0..k-1 columns to the (engineered) DataFrame."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        self.segmenter_ = BorrowerSegmenter(self.n_clusters, self.random_state).fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dist = self.segmenter_.transform(X)
        cols = [f"seg_dist_{i}" for i in range(self.n_clusters)]
        return pd.concat([X.reset_index(drop=True), pd.DataFrame(dist, columns=cols)], axis=1).set_axis(X.index)


def select_k(X: pd.DataFrame, k_min: int, k_max: int, sample_size: int, random_state: int) -> tuple[int, pd.DataFrame]:
    """Pick k by silhouette score (cluster separation); inertia is kept for the elbow chart."""
    X = _ensure_engineered(X)
    rows = []
    for k in range(k_min, k_max + 1):
        seg = BorrowerSegmenter(k, random_state).fit(X)
        Z = seg._matrix(X)
        labels = seg.kmeans_.labels_
        sil = silhouette_score(Z, labels, sample_size=min(sample_size, len(Z)), random_state=random_state)
        rows.append({"k": k, "silhouette": float(sil), "inertia": float(seg.kmeans_.inertia_)})
    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    return best_k, scores


def name_segments(labels: np.ndarray, y: pd.Series) -> dict[int, dict]:
    """Name clusters by their observed loss rate on TRAINING data (rank 1 = safest)."""
    loss_rate = pd.Series(np.asarray(y)).groupby(labels).mean().sort_values()
    k = len(loss_rate)
    tier_idx = np.round(np.linspace(0, len(RISK_TIERS) - 1, k)).astype(int)
    info = {}
    for rank, (cluster_id, rate) in enumerate(loss_rate.items()):
        info[int(cluster_id)] = {
            "name": f"S{rank + 1} - {RISK_TIERS[tier_idx[rank]]}",
            "rank": rank + 1,
            "train_loss_rate": round(float(rate), 4),
        }
    return info


def profile_segments(X: pd.DataFrame, labels: np.ndarray, y: pd.Series | None, segment_info: dict) -> pd.DataFrame:
    X = _ensure_engineered(X).assign(Segment_ID=labels)
    if y is not None:
        X = X.assign(is_loss=np.asarray(y))
    agg = {
        "Loans": ("Segment_ID", "size"),
        "Avg_Days_Past_Due": ("Days_Past_Due", "mean"),
        "Avg_Missed_Payments": ("Num_Missed_Payments", "mean"),
        "Median_EMI_to_Income": ("EMI_to_Income", "median"),
        "Median_Collateral_Coverage": ("Collateral_Coverage", "median"),
        "Avg_Credit_Score": ("Credit_Score", "mean"),
        "Avg_Previous_Defaults": ("Previous_Defaults", "mean"),
        "Total_Outstanding": ("Outstanding_Loan_Amount", "sum"),
    }
    if y is not None:
        agg["Loss_Rate"] = ("is_loss", "mean")
    prof = X.groupby("Segment_ID").agg(**agg)
    prof.insert(0, "Segment", [segment_info[int(i)]["name"] for i in prof.index])
    prof.insert(2, "Share", prof["Loans"] / prof["Loans"].sum())
    prof["Rank"] = [segment_info[int(i)]["rank"] for i in prof.index]
    return prof.sort_values("Rank").drop(columns="Rank").round(3)
