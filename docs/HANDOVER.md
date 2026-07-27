# Handover Document

## 1. Project Summary

This project develops an AI-assisted triage system for Mercer's Emergency Department using only information available at the patient's initial assessment. The system is designed to support, rather than replace, the clinical judgement of triage staff. Evaluation focuses primarily on recall and under-triage rates for ESI levels 2-4, with particular attention to preventing dangerous under-triage without causing excessive over-triage. Speed and interpretability are also assessed to ensure that the system can support timely, transparent decision making in time sensitive ED settings.

## 2. Final Model Decision

**CatBoost** meets the three main deployment requirements amongst the models. It produces the lowest under-triage rates for the priority classes (ESI 2 and 4), while maintaining precision comparable to the other leading models. It generates predictions in under 0.01 milliseconds per patient and supports SHAP explanations that can be communicated clearly to triage nurses.

## 3. How to Run the Pipeline

Run the following commands from a terminal:

```bash
git clone https://github.com/gabriellejohnson784/Gabrielle-Johnson-Carisurg-Portfolio.git
cd Gabrielle-Johnson-Carisurg-Portfolio

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

mv <path-to-downloaded-csv> data/yaleemmlc_admissionprediction_triage.csv

python scripts/train.py --config config.yaml
```

The random seed = **42** to support reproducibility.

## 4. Data Location and Governance

The dataset is stored locally in:

```
data/yaleemmlc_admissionprediction_triage.csv
```

The dataset is in `.gitignore` and must not be committed to the repository.

Access to the dataset is restricted to the project author. Anyone receiving the repository must obtain the dataset from its original source.

## 5. Known Limitations

- **Reducing under-triaging decreases precision.** CatBoost's class weighting reduced undertriaging in ESI 2 from 34.6% to 16.1%, but this consequently decreases ESI 2 precision from approximately 0.74 to 0.60.
- **Explanations require an additional component.** Unlike logistic regression, CatBoost does not provide directly readable coefficients. Patient-level explanations require SHAP, adding another software dependency and computation step that must be maintained and validated.
- **External validity has not been established.** The model was developed using data from a single US hospital system. Local and prospective validation is required before clinical use.
- **Missing Data.** Some ESI-2 decisions depend on the patient's history, comorbidities, and nursing judgement that are not fully represented in the available dataset.

## Contact

For questions about implementation, model training, testing or evaluation, contact the project author at gabriellejohnson784@gmail.com.
