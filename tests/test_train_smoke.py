"""Training smoke test
"""

from pathlib import Path

import numpy as np

from src import data, features, model as model_lib
from src.utils import load_config
from conftest import CONFIG_PATH, load_real_sample


def test_smoke_train_predict(tmp_path):
    cfg = load_config(CONFIG_PATH)
    seed = cfg.get("seed", 42)

    raw = load_real_sample(n=50, seed=2)
    df = data.clean_triage(raw)

    # ESI is heavily imbalanced (ESI 1 is <1% of the full dataset), so a
    # 50-row real sample can draw a class with only 1 member; split_data's
    # stratified split requires at least 2 per class, so drop singletons.
    counts = df[data.TARGET].value_counts()
    df = df[df[data.TARGET].isin(counts[counts >= 2].index)]

    base_features = data.select_features(df)
    X = features.encode_demographics(df, base_features,
                                     extra=cfg["features"]["demographics"])
    if cfg["features"].get("clinical_features", True):
        X = features.add_clinical_features(X)
    y = df[data.TARGET]

    X_train, X_test, y_train, y_test = data.split_data(
        X, y, test_size=0.3, seed=seed)

    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name")
    params = model_cfg
    if name == "catboost":
        # keep CatBoost's training logs out of the repo during test runs
        params.setdefault("train_dir", str(tmp_path / "catboost_info"))

    est = model_lib.build_model(name, params, seed=seed)
    est, _ = model_lib.fit_model(est, X_train, y_train)
    preds = np.asarray(est.predict(X_test)).ravel()

    # one prediction per test row, all valid ESI levels
    assert len(preds) == len(y_test)
    assert set(preds.tolist()) <= {1, 2, 3, 4, 5}
