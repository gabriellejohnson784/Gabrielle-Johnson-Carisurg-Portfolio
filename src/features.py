"""Feature engineering and encoding.

"""

import numpy as np
import pandas as pd


DEFAULT_EXTRA = ("age", "gender", "arrivalmode")

RED_FLAG_COLUMNS = [
    "is_tachypneic", "is_hypoxic", "is_febrile", "is_hypothermic",
    "is_bradycardic", "is_tachycardic", "is_hyperglycaemic",
]


def encode_demographics(df, features, extra=DEFAULT_EXTRA, drop_first=True):
    """Return a feature frame with selected demographics one-hot encoded.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset (output of ``data.clean_triage``).
    features : list of str
        Base feature columns (output of ``data.select_features``).
    extra : sequence of str
        Demographic/arrival columns to add. Numeric columns (e.g. ``age``)
        pass through untouched; text columns are one-hot encoded.
    drop_first : bool
        Drop one category per encoded column implied by the others being 0.

    Returns
    -------
    pandas.DataFrame
    """
    extra = list(extra)
    categorical = [c for c in extra if not pd.api.types.is_numeric_dtype(df[c])]
    return pd.get_dummies(df[list(features) + extra], columns=categorical,
                          drop_first=drop_first, dtype=int)


def add_clinical_features(data, glucose_col="triage_glucose"):
    """
    Adds three ratio features (shock index, pulse pressure, SpO2/RR), seven
    red-flag indicators, a combined respiratory-distress flag, and a
    red-flag count. Infinities from divide-by-zero (vitals recorded as 0)
    are replaced with 0.
    """
    out = data.copy()

    #  ratios & combinations 
    out["shock_index"] = out["triage_vital_hr"] / out["triage_vital_sbp"]
    out["pulse_pressure"] = out["triage_vital_sbp"] - out["triage_vital_dbp"]
    out["spo2_rr_ratio"] = out["triage_vital_o2"] / out["triage_vital_rr"]

    #  red-flag indicators 
    out["is_tachypneic"] = (out["triage_vital_rr"] > 20).astype(int)
    out["is_hypoxic"] = (out["triage_vital_o2"] < 92).astype(int)
    out["is_febrile"] = (out["triage_vital_temp"] >= 100.4).astype(int)
    out["is_hypothermic"] = (out["triage_vital_temp"] < 96.8).astype(int)
    out["is_bradycardic"] = (out["triage_vital_hr"] < 60).astype(int)
    out["is_tachycardic"] = (out["triage_vital_hr"] > 100).astype(int)
    out["is_hyperglycaemic"] = (out[glucose_col] > 180).astype(int)

    # combined respiratory red flag
    out["resp_distress"] = (
        (out["is_hypoxic"] == 1) | (out["is_tachypneic"] == 1)
    ).astype(int)

    # severity score: how many red flags fire
    out["red_flag_count"] = out[RED_FLAG_COLUMNS].sum(axis=1)

    # guard against divide-by-zero in the ratios
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)

    return out