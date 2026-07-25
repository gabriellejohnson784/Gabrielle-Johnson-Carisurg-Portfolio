# Running the Model Training Pipeline

This walks through training the ED triage models locally, from a clean clone to printed metrics.

## 1. Prerequisites

- Python 3.12 (or similar 3.x)
- The raw triage dataset CSV (not included in this repo — see [Data Privacy](#data-privacy))

## 2. Set up a virtual environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll know it's active when your shell prompt is prefixed with `(.venv)`. Do this every time you open a new terminal to work on this project.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs pandas, numpy, matplotlib, scikit-learn, joblib, pyyaml, and catboost.

## 4. Add the dataset

Place the raw CSV in the `data/` folder:

```bash
mv ~/Downloads/yaleemmlc_admissionprediction_triage.csv data/
```

`data/` is gitignored, so this file never gets committed — see [Data Privacy](#data-privacy) below.

## 5. Check config.yaml

[config.yaml](config.yaml) controls the whole run. Defaults are already set up for a local run:

```yaml
data:
  raw_path: "data/yaleemmlc_admissionprediction_triage.csv"
  test_size: 0.2

models:
  hgb:
    max_leaf_nodes: 31
    learning_rate: 0.1
    max_iter: 1000
  logreg:
    max_iter: 1000
  catboost:
    iterations: 400
    learning_rate: 0.2
    depth: 6

output:
  model_dir: "models"
```

- Add or remove entries under `models:` to control which models get trained.
- `output.model_dir` is where trained `.joblib` files are saved (also gitignored).

## 6. Run training

```bash
python scripts/train.py --config config.yaml
```

## 7. Read the output

For each model listed in `config.yaml`, you'll see a metrics block like:

```
=== hgb ===
  accuracy             : 0.652
  recall_esi2          : 0.832
  recall_esi3          : 0.537
  recall_esi4          : 0.717
  precision_esi2       : 0.603
  precision_esi3       : 0.758
  precision_esi4       : 0.586
  undertriage_esi2_pct : 16.6
  undertriage_esi3_pct : 12.3
  undertriage_esi4_pct : 1.7
  f1_macro             : 0.486
  f1_weighted          : 0.643
  infer_ms_per_patient : 0.0141
  train_seconds        : 12.7
  saved to models/model_hgb.joblib
```

What to look at first:
- **`recall_esiN` / `undertriage_esiN_pct`** — the project's priority metric. Undertriage (predicting a patient *less* urgent than they really are) is the clinically dangerous error direction, so lower undertriage % is better even at some cost to overall accuracy.
- **`accuracy` / `f1_macro` / `f1_weighted`** — general context, not the primary decision metric.
- The trained model is saved to `models/model_<name>.joblib` for later reuse (see [src/model.py](src/model.py)'s `load_model`).

## Data Privacy

The dataset is real (deidentified) patient triage data. `data/` and `models/` are both gitignored except for a `.gitkeep` placeholder — never remove those entries from `.gitignore`, and never commit a CSV or `.joblib` file directly. Anyone cloning this repo needs to supply their own copy of the dataset before training will work.

## Troubleshooting

- **`ModuleNotFoundError`** — you likely forgot to activate `.venv` (step 2) or run `pip install` (step 3).
- **`FileNotFoundError` on the CSV** — check `data.raw_path` in config.yaml matches where you actually placed the file (step 4).
- **Import errors from `src`** — always run `train.py` from the repo root (`python scripts/train.py ...`), not from inside `scripts/`.
