"""Classifier definitions and the soft-voting ensemble."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from . import config


def build_models() -> dict[str, object]:
    """Instantiate the three base classifiers."""
    return {
        "XGBoost": xgb.XGBClassifier(**config.XGB_PARAMS),
        "Random Forest": RandomForestClassifier(**config.RF_PARAMS),
        "SVM": LinearSVC(**config.SVM_PARAMS),
    }


def train_all(models: dict[str, object], X_train, y_train, verbose: bool = True) -> dict[str, object]:
    """Fit every model on the training partition."""
    for name, model in models.items():
        if verbose:
            print(f"  training {name} ...", flush=True)
        model.fit(X_train, y_train)
    return models


def soft_vote(models: dict[str, object], X, members: list[str] | None = None) -> np.ndarray:
    """Average the predicted class probabilities of the ensemble members.

    LinearSVC exposes no predict_proba, so the ensemble is formed from the two
    probabilistic models (see config.ENSEMBLE_MEMBERS). This matches the
    ensemble described in the thesis, whose prediction cost is O(N x K x M)
    for N samples, K classes and M base models.
    """
    members = members or config.ENSEMBLE_MEMBERS
    probas = []
    for name in members:
        model = models[name]
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"{name} does not support soft voting.")
        probas.append(model.predict_proba(X))
    mean_proba = np.mean(probas, axis=0)
    return mean_proba


def soft_vote_predict(models: dict[str, object], X, members: list[str] | None = None) -> np.ndarray:
    """Hard class labels from the averaged probabilities."""
    return np.argmax(soft_vote(models, X, members), axis=1)
