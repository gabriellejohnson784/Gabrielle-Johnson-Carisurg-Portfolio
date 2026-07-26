"""Train and evaluate ED triage models from a config file.

Usage:
    python scripts/train.py --config config.yaml

Reads config.yaml, then calls the src/ modules in order:
load + clean -> select + engineer features -> split -> fit -> evaluate -> save.
"""

import argparse
import sys
from pathlib import Path

# Allow python scripts/train.py from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, features, model as model_lib
from src.utils import format_results, load_config


def main(config_path):
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)

    # load + clean 
    df = data.clean_triage(data.load_raw(cfg["data"]["raw_path"]))

    # select + engineer features 
    base_features = data.select_features(df)
    X = features.encode_demographics(df, base_features,
                                     extra=cfg["features"]["demographics"])
    if cfg["features"].get("clinical_features", True):
        X = features.add_clinical_features(X)
    y = df[data.TARGET]

    # split 
    X_train, X_test, y_train, y_test = data.split_data(
        X, y, test_size=cfg["data"]["test_size"], seed=seed)

    # fit + evaluate the pinned model
    class_weights = cfg.get("class_weights")
    if class_weights is not None:
        class_weights = {int(k): v for k, v in class_weights.items()}
    model_dir = Path(cfg["output"]["model_dir"])

    model_cfg = dict(cfg["model"])
    name = model_cfg.pop("name")
    params = model_cfg

    # CatBoost takes class weights natively as a constructor parameter
    if name == "catboost" and class_weights is not None:
        params.setdefault("class_weights", class_weights)

    est = model_lib.build_model(name, params, seed=seed, class_weights=class_weights)

    if name == "logreg" and class_weights is not None:
        # logreg accepts class_weight natively
        est.named_steps["logisticregression"].set_params(class_weight=class_weights)
        est, train_s = model_lib.fit_model(est, X_train, y_train)
    elif name in ("catboost", "mlp", "dummy", "voting", "stacking"):
        # catboost: weights already in constructor; the rest don't
        # support sample_weight (mlp doesn't, and voting/stacking wrap it)
        est, train_s = model_lib.fit_model(est, X_train, y_train)
    else:
        # tree/rf/hgb/xgboost: apply class weights as sample weights
        est, train_s = model_lib.fit_model(est, X_train, y_train,
                                           class_weights=class_weights)
    results = model_lib.evaluate(est, X_test, y_test, train_seconds=train_s)
    print(f"\n=== {name} ===")
    print(format_results(results))

    saved = model_lib.save_model(est, model_dir / f"model_{name}.joblib")
    print(f"  saved to {saved}")


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml",
                        help="Path to the YAML config file")
    args = parser.parse_args()
    main(args.config)