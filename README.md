
# Emergency Department AI-Assisted Triage System 🏥 📝

This project is an amalgamation and reimagining of an AI-assisted triage system built for a low-resource Caribbean hospital, where emergency departments often carry heavy patient loads with limited staff, limited digital infrastructure, and very little room for delay.

## Table of Contents 📖

- [Purpose](#purpose-)
- [Installation](#installation-)
- [Repository Structure](#repository-structure)
- [Usage](#usage-)
- [Data Exploration](#data-exploration-)
- [Model Training](#modelTraining)
- [Handover Document](#handover-document-)
- [Contributing](#contributing-)
- [License](#license)

## Purpose 🩺

The purpose of this project is to explore whether a small but powerful set of routinely collected patient triage variables can help support accurate urgency categorisation in the emergency department where digital data is extremely limited.

Rather than replacing nurses, this system is designed to act as a support tool, a second set of digital aid that helps reduce the burden on triage staff, improve consistency, and support faster decision-making during busy or high pressure periods.

The goal is to identify the least amount of patient information needed to achieve strong triage performance, with a target of approximately 90% accuracy in urgency categorisation and <5 minutes of processing.

## Installation ⚙️

Clone the repository using the command below:

```bash

git clone https://github.com/gabriellejohnson784/Gabrielle-Johnson-Carisurg-Portfolio.git

```

## Repository Structure 🗂️

Model training was refactored out of the notebooks into a modular, config-driven pipeline:

- `src/data.py` - loading and cleaning
- `src/features.py` - feature engineering (demographic encoding, clinical ratios/red flags)
- `src/model.py` - model construction, training, evaluation
- `src/utils.py` - shared helpers (config loading, result formatting)
- `src/diagnostics.py` / `src/viz.py` - optional post-hoc analysis (under-triage profiling, confusion matrices) - not run automatically by training
- `config.yaml` - the single source of truth for hyperparameters, file paths, and the random seed.
- `scripts/train.py` - entry point that reads `config.yaml` and runs the pipeline end-to-end
- `scripts/visualize.py` - loads a saved model and generates a confusion matrix
- `tests/` - pytest schema and training smoke tests

See [docs/HANDOVER.md](docs/HANDOVER.md) for the exact run commands.

## Usage 💻📈

### Pipeline Usage (current)

After running `pip install -r requirements.txt` and placing the dataset at `data/yaleemmlc_admissionprediction_triage.csv`, train the pinned model with:

```bash
python scripts/train.py --config config.yaml
```

See [docs/HANDOVER.md](docs/HANDOVER.md) for the full setup walkthrough (venv, dependencies, dataset placement, run command).

### Notebook Usage (historical)

The Week 0-7 notebooks in this project were completed using Google Colab, before model training was refactored into the pipeline above. Before any data manipulation can begin, the required libraries must be imported and Google Drive must be mounted so the dataset can be accessed.

```python

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from google.colab import drive

drive.mount('/content/drive')

FILE_PATH = '/content/drive/MyDrive/add_your_file_path.csv'

df = pd.read_csv(FILE_PATH)

```
## Data Exploration 🔎

A feasibility analysis of an emergency department triage dataset for a proposed AI assisted triage system at Mercer General ED was uploaded during Week 5.

The analysis reviews 55,121 ED encounters across 225 columns, including demographic variables, arrival information, triage vital signs, chief complaint indicators, and Emergency Severity Index (ESI) labels. The goal is to determine whether the dataset is suitable for train a model that will be used to support Mercer’s ED.

The notebook focuses on data profiling, missingness, sanitizing data, demographic representation, chief complaint and vitial sign patterns and correlation

The main finding is that the dataset may be useful for exploratory analysis but it is not sufficient for Mercer's AI triage model training without additional local validation.

## Model Training 🏋️

In Week 6 we started model training by first producing a dummy/baseline model
that we aim to beat. The dummy classifier predicts using only the ESI class
distribution, any model must outperform it to prove it has learned correlation amongst the features.
The first model we looked at was a logistic regression model, trained on
scaled features. Below is a comparison of the two:

| Model | Accuracy | Weighted F1 | Macro F1 | ESI 1 Recall |
|---|---|---|---|---|
| Dummy (stratified) | 0.38 | — | — | — |
| Logistic Regression | 0.667 | 0.661 | 0.493 | 0.250 |

The dataset was split 80/20, and a `random_state=42`
was used. 

The logistic regression clearly beats the dummy baseline, but the gap between
weighted and macro F1 reflects the severe class imbalance. Next steps
focus on class weighting rather than overall accuracy.

The final pinned model is CatBoost - `config.yaml`'s `model:` block holds the one model, one hyperparameter set actually shipped. See the Handover Document below for the full decision rationale.

## Handover Document 📋

CatBoost was the final model choice. For more information on how exactly to run commands, where the dataset lives and who can access it, and known limitations of the CATBoost for this project you can reference [docs/HANDOVER.md](docs/HANDOVER.md)


## Contributing 🤝

Pull requests are welcome. Any form of contributions that help make the project more practical for low-resource healthcare settings are appreciated.

## License 🧑‍⚖️

This project is not currently licensed.

### By: Gabrielle Johnson Carisurg AI Healthcare Cohort 2026
