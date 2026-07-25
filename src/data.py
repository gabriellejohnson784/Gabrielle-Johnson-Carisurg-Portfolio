"""Dataset loading, cleaning, and splitting for the ED triage project.

"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "esi"

# Vital-sign columns 
VITALS = [
    "triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp",
    "triage_vital_rr", "triage_vital_o2", "triage_vital_temp",
    "triage_glucose",
]


DEMOGRAPHICS = [
    "age", "gender", "ethnicity", "race", "lang", "religion",
    "maritalstatus", "employstatus", "insurance_status",
]

# Administrative / arrival details
ADMIN = ["dep_name", "arrivalmode", "arrivalmonth", "arrivalday", "arrivalhour_bin"]

# Outcomes of the visit - known only AFTER triage, never model inputs
LEAKAGE = ["disposition", "previousdispo"]

# Reference ranges for general adult triage (no paediatric patients).
# Each entry is (low, high, unit).
NORMAL_RANGES = {
    "triage_vital_hr": (60, 100, "bpm"),
    "triage_vital_sbp": (90, 140, "mmHg"),
    "triage_vital_dbp": (60, 90, "mmHg"),
    "triage_vital_rr": (12, 20, "/min"),
    "triage_vital_o2": (95, 100, "%"),
    "triage_vital_temp": (97.0, 99.5, "F"),
    "triage_glucose": (70, 140, "mg/dL"),
}

# "Plausible" anything outside these bounds is treated as a data error 
PLAUSIBLE = {
    "age": (0, 120), "esi": (1, 5),
    "triage_vital_hr": (20, 250), "triage_vital_sbp": (50, 300),
    "triage_vital_dbp": (20, 200), "triage_vital_rr": (4, 60),
    "triage_vital_o2": (50, 100), "triage_vital_temp": (86, 110),
    "triage_glucose": (20, 800),
}


def load_raw(path):
    """Read the raw triage CSV into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Location of the yale triage CSV.

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}.")
    # index_col=0 drops the unnamed export row-number column
    return pd.read_csv(path, index_col=0)


def chief_complaint_columns(df):
    """Return the list of chief-complaint indicator columns"""
    return [col for col in df.columns if col.startswith("cc_")]


def clean_triage(raw, target=TARGET, vitals=None, plausible=None,
                 demographics=None, admin=None, leakage=None):
    """Return a cleaned copy of the raw triage DataFrame.

    Cleaning steps:
    - drop rows with missing/invalid ESI
    - coerce vitals, age, and target to numeric
    - flag implausible values (``<col>_was_impossible``)
    - flag and median-impute missing vitals (``<col>_was_missing``)
    - normalise the oxygen-device column
    - coerce chief-complaint flags to 0/1
    - standardise categorical text columns ("Unknown" for blanks)
    - cast the target to int

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of :func:`load_raw`.
    target, vitals, plausible, demographics, admin, leakage :
        Column configuration; module-level defaults are used when None.

    Returns
    -------
    pandas.DataFrame
    """
    vitals = VITALS if vitals is None else vitals
    plausible = PLAUSIBLE if plausible is None else plausible
    demographics = DEMOGRAPHICS if demographics is None else demographics
    admin = ADMIN if admin is None else admin
    leakage = LEAKAGE if leakage is None else leakage

    d = raw.copy()
    cc_cols = chief_complaint_columns(d)

    # Drop rows with no esi
    d = d[d[target].notna()].copy()

    # Convert expected numeric fields to numeric
    numeric_cols = [c for c in vitals if c in d.columns]
    if "age" in d.columns:
        numeric_cols.append("age")
    numeric_cols.append(target)
    for col in numeric_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Drop rows where ESI became invalid after numeric conversion
    d = d[d[target].notna()].copy()

    # Add impossible value flags for implausible values 
    for col, (low, high) in plausible.items():
        if col in d.columns:
            out_of_range = (d[col] < low) | (d[col] > high)
            if out_of_range.any():
                d[f"{col}_was_impossible"] = out_of_range.astype(int)

    # Add missingness flags for vitals with missing values, then impute
    for col in vitals:
        if col in d.columns and d[col].isna().sum() > 0:
            d[f"{col}_was_missing"] = d[col].isna().astype(int)
            d[col] = d[col].fillna(d[col].median())

    # Oxygen device flag- blank means no device recorded
    if "triage_vital_o2_device" in d.columns:
        d["triage_vital_o2_device"] = (
            pd.to_numeric(d["triage_vital_o2_device"], errors="coerce").fillna(0)
        )

    # Clean chief-complaint flags
    for col in cc_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)
        d[col] = (d[col] == 1).astype(int)

    # Clean categorical text columns
    for col in demographics + admin + leakage:
        if col in d.columns and not pd.api.types.is_numeric_dtype(d[col]):
            d[col] = (
                d[col].astype(str).str.strip()
                .replace({"": "Unknown", "nan": "Unknown", "NaN": "Unknown",
                          "None": "Unknown", "NONE": "Unknown"})
                .fillna("Unknown")
            )

    # Convert target to integer
    d[target] = d[target].round().astype(int)

    return d


def select_features(df, target=TARGET, exclude=None):
    """Return the list of model input columns.

    """
    if exclude is None:
        exclude = LEAKAGE + ADMIN + DEMOGRAPHICS
    return [c for c in df.columns if c != target and c not in exclude]


def split_data(X, y, test_size=0.2, seed=42):
    """Stratified train/test split with the fixed seed.

    Returns
    ------
    (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=seed)