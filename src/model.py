"""Model building, training, and evaluation for the ED triage project.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PRIORITY_LEVELS = (2, 3, 4)  # per Dr. De Freitas; looking at the ambiguous middle levels

# Voting-ensemble base learner weights (logreg, mlp, hgb), in _ensemble_base_learners order
ENSEMBLE_WEIGHTS = (2, 1, 2)


class ShiftedLabelClassifier:
    """Shifter for estimators that require labels 0..n-1 (e.g. XGBoost).

    """

    def __init__(self, estimator, offset=1):
        self.estimator = estimator
        self.offset = offset

    def fit(self, X, y, sample_weight=None):
        y_shifted = np.asarray(y) - self.offset
        if sample_weight is None:
            self.estimator.fit(X, y_shifted)
        else:
            self.estimator.fit(X, y_shifted, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.estimator.predict(X) + self.offset

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def _ensemble_base_learners(seed, class_weights=None):
    """Base learners shared by the voting and stacking ensembles.

    """
    logreg = build_model("logreg", seed=seed)
    hgb = build_model("hgb", seed=seed)
    if class_weights is not None:
        # VotingClassifier/StackingClassifier label-encode y to 0..n-1
        # before fitting base learners, so class_weight keys must match
        # the encoded labels (ESI 0-4), not the raw ESI levels.
        encoded_weights = {esi - 1: w for esi, w in class_weights.items()}
        logreg.named_steps["logisticregression"].set_params(class_weight=encoded_weights)
        hgb.set_params(class_weight=encoded_weights)
    return [
        ("logreg", logreg),
        ("mlp", build_model("mlp", seed=seed)),
        ("hgb", hgb),
    ]


def build_model(name, params=None, seed=42, class_weights=None):
    """Construct a model from a config name and parameter dict.

    Supported names: ``dummy``, ``logreg``, ``tree``, ``rf``, ``hgb``,
    ``mlp``, ``catboost``, ``xgboost``, ``voting``, ``stacking``.
    
    Parameters
    ----------
    name : str
        Model key (case-insensitive).
    params : dict, optional
        Keyword arguments forwarded to the estimator
    seed : int
        Random state applied to every estimator.
    class_weights : dict, optional
        {ESI level: weight}. Only used by ``voting``/``stacking``, to
        weight their logreg/hgb base learners; other model types get
        class weighting from the caller instead (see ``fit_model``).

    Returns
    -------
    sklearn estimator or pipeline
    """
    params = dict(params or {})
    name = name.lower()

    if name == "dummy":
        return DummyClassifier(strategy=params.pop("strategy", "stratified"),
                               random_state=seed, **params)
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=params.pop("max_iter", 1000),
                               random_state=seed, **params))
    if name == "tree":
        return DecisionTreeClassifier(
            max_depth=params.pop("max_depth", 30),
            min_samples_leaf=params.pop("min_samples_leaf", 20),
            random_state=seed, **params)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 300),
            random_state=seed, n_jobs=-1, **params)
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_leaf_nodes=params.pop("max_leaf_nodes", 31),
            learning_rate=params.pop("learning_rate", 0.1),
            max_iter=params.pop("max_iter", 1000),
            early_stopping=params.pop("early_stopping", True),
            validation_fraction=params.pop("validation_fraction", 0.1),
            random_state=seed, **params)
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(params.pop("hidden_layer_sizes", (64, 32))),
                alpha=params.pop("alpha", 1e-3),
                max_iter=params.pop("max_iter", 500),
                early_stopping=params.pop("early_stopping", True),
                random_state=seed, **params))

    if name == "catboost":
        from catboost import CatBoostClassifier  # lazy: optional dependency
        return CatBoostClassifier(
            iterations=params.pop("iterations", 400),
            learning_rate=params.pop("learning_rate", 0.2),
            depth=params.pop("depth", 6),
            border_count=params.pop("border_count", 64),
            subsample=params.pop("subsample", 0.8),
            bootstrap_type=params.pop("bootstrap_type", "Bernoulli"),
            rsm=params.pop("rsm", 0.5),
            loss_function=params.pop("loss_function", "MultiClass"),
            early_stopping_rounds=params.pop("early_stopping_rounds", 30),
            thread_count=-1, random_seed=seed, verbose=False, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier  # lazy: optional dependency
        est = XGBClassifier(
            n_estimators=params.pop("n_estimators", 1000),
            learning_rate=params.pop("learning_rate", 0.2),
            max_depth=params.pop("max_depth", 6),
            subsample=params.pop("subsample", 0.8),
            colsample_bytree=params.pop("colsample_bytree", 0.5),
            tree_method=params.pop("tree_method", "hist"),
            objective=params.pop("objective", "multi:softprob"),
            eval_metric=params.pop("eval_metric", "mlogloss"),
            random_state=seed, n_jobs=-1, **params)
        # XGBoost requires labels 0..n-1; ESI labels are 0-4
        return ShiftedLabelClassifier(est, offset=1)

    if name == "voting":
        return VotingClassifier(
            estimators=_ensemble_base_learners(seed, class_weights),
            voting=params.pop("voting", "soft"),
            weights=params.pop("weights", list(ENSEMBLE_WEIGHTS)),
            n_jobs=-1, **params)
    if name == "stacking":
        return StackingClassifier(
            estimators=_ensemble_base_learners(seed, class_weights),
            final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
            stack_method=params.pop("stack_method", "predict_proba"),
            cv=params.pop("cv", 3),
            n_jobs=-1, **params)

    raise ValueError(f"Unknown model name: {name!r}")


def sample_weights(y, class_weights):
    """Map a {class: weight} dict onto a per-row weight array.

    Used for estimators whose ``class_weight`` handling is unreliable with
    1-5 labels (e.g. HistGradientBoostingClassifier re-encodes classes
    internally); ``fit(..., sample_weight=...)`` is mathematically
    equivalent.
    """
    return pd.Series(y).map(class_weights).values


def fit_model(model, X_train, y_train, class_weights=None):
    """Fit a model, optionally with per-class weighting, and time it.

    Parameters
    ----------
    model : sklearn estimator
    X_train, y_train : training data
    class_weights : dict, optional
        {ESI level: weight}. Applied as sample weights.

    Returns
    -------
    (model, train_seconds)
    """
    t0 = time.perf_counter()
    if class_weights is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train,
                  sample_weight=sample_weights(y_train, class_weights))
    return model, time.perf_counter() - t0


def undertriage_rates(y_true, y_pred, levels=PRIORITY_LEVELS):
    """Percentage of true ESI-n patients predicted LESS urgent than n.


    Returns
    -------
    dict mapping level -> percentage (0-100)
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {lvl: float(100 * (y_pred[y_true == lvl] > lvl).mean())
            for lvl in levels}


def evaluate(model, X_test, y_test, levels=PRIORITY_LEVELS, train_seconds=None):
    """Score a fitted model.

    Metrics follow the project hierarchy: per-class recall for ESI 2-4
    first, precision alongside, under-triage rates per class, macro F1 and
    accuracy as context, plus per-patient inference time.

    Returns
    -------
    dict of metric name -> value
    """
    t0 = time.perf_counter()
    pred = np.asarray(model.predict(X_test)).ravel()
    infer_ms = (time.perf_counter() - t0) / len(X_test) * 1000

    recalls = recall_score(y_test, pred, labels=list(levels), average=None,
                           zero_division=0)
    precisions = precision_score(y_test, pred, labels=list(levels),
                                 average=None, zero_division=0)
    under = undertriage_rates(y_test, pred, levels)

    results = {"accuracy": round(float((pred == np.asarray(y_test)).mean()), 3)}
    for lvl, r in zip(levels, recalls):
        results[f"recall_esi{lvl}"] = round(float(r), 3)
    for lvl, p in zip(levels, precisions):
        results[f"precision_esi{lvl}"] = round(float(p), 3)
    for lvl in levels:
        results[f"undertriage_esi{lvl}_pct"] = round(under[lvl], 1)
    results["f1_macro"] = round(float(f1_score(y_test, pred, average="macro")), 3)
    results["f1_weighted"] = round(float(f1_score(y_test, pred, average="weighted")), 3)
    results["infer_ms_per_patient"] = round(infer_ms, 4)
    if train_seconds is not None:
        results["train_seconds"] = round(train_seconds, 1)
    return results


def save_model(model, path):
    """Persist a fitted model (or pipeline) with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path):
    """Load a model previously saved with :func:`save_model`."""
    return joblib.load(path)