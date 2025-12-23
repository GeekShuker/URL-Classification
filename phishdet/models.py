"""Model builders for phishing detection ensembles."""
from __future__ import annotations

from typing import List

from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier


def build_en_bag(
    random_state: int,
    n_estimators: int = 200,
    max_depth: int | None = 8,
    min_samples_leaf: int = 1,
    **kwargs,
):
    base = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model = BaggingClassifier(
        estimator=base,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    return model


def build_en_knn(random_state: int, ks: List[int] | None = None, **kwargs):
    ks = ks or [3, 5, 7, 9]
    estimators = []
    for k in ks:
        knn = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k)),
            ]
        )
        estimators.append((f"knn{k}", knn))
    model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    return model


def build_rfe_xgb(
    random_state: int,
    rfe_k: int = 30,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    **kwargs,
):
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric="logloss",
        tree_method="hist",
    )
    model = Pipeline(
        [
            ("rfe", RFE(estimator=DecisionTreeClassifier(random_state=random_state), n_features_to_select=rfe_k)),
            ("xgb", xgb),
        ]
    )
    return model
