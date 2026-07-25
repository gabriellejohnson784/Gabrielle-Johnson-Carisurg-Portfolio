"""Generate diagnostic plots for a previously trained ED triage model.

Usage:
    python scripts/visualize.py --config config.yaml --model hgb

Rebuilds the same train/test split train.py used (same config, same seed),
loads the saved model, and saves a confusion matrix PNG next to it.
"""

import argparse
import sys
from pathlib import Path

# Allow python scripts/visualize.py from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, features, model as model_lib, viz
from src.utils import load_config


def main(config_path, model_name):
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)

    df = data.clean_triage(data.load_raw(cfg["data"]["raw_path"]))
    base_features = data.select_features(df)
    X = features.encode_demographics(df, base_features,
                                     extra=cfg["features"]["demographics"])
    if cfg["features"].get("clinical_features", True):
        X = features.add_clinical_features(X)
    y = df[data.TARGET]

    _, X_test, _, y_test = data.split_data(
        X, y, test_size=cfg["data"]["test_size"], seed=seed)

    model_dir = Path(cfg["output"]["model_dir"])
    est = model_lib.load_model(model_dir / f"model_{model_name}.joblib")

    cm_path = viz.save_confusion_matrix(
        est, X_test, y_test, model_dir / f"confusion_matrix_{model_name}.png",
        title=model_name)
    print(f"confusion matrix saved to {cm_path}")


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml",
                        help="Path to the YAML config file")
    parser.add_argument("--model", required=True,
                        help="Model name as used in config.yaml's models: section "
                             "(e.g. hgb) - loads <model_dir>/model_<name>.joblib")
    args = parser.parse_args()
    main(args.config, args.model)
