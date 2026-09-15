"""Model candidates and the end-to-end sklearn pipeline.

Raw loan rows go in, a loss probability comes out. Everything in between (feature engineering,
segment distances, imputation, scaling, encoding) lives inside ONE pipeline object, which is
what gets saved. That removes a whole class of train/serve mismatch bugs.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .clustering import SegmentDistanceAdder
from .features import CATEGORICAL, MODEL_NUMERIC, FeatureEngineer


def build_candidates(random_state: int, fast: bool = False) -> dict:
    """class_weight='balanced' makes each model care about the minority loss class.
    It distorts raw probabilities, which is why the chosen model is calibrated afterwards."""
    candidates = {
        # interpretable baseline - if a complex model can't beat this, don't ship the complex model
        "logistic_regression": LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=100 if fast else 400,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100 if fast else 400,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            class_weight="balanced",
            random_state=random_state,
        ),
    }
    if fast:
        candidates.pop("random_forest")
    return candidates


def build_pipeline(estimator, n_clusters: int, random_state: int) -> Pipeline:
    numeric = MODEL_NUMERIC + [f"seg_dist_{i}" for i in range(n_clusters)]
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, CATEGORICAL)],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            ("segments", SegmentDistanceAdder(n_clusters=n_clusters, random_state=random_state)),
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )
