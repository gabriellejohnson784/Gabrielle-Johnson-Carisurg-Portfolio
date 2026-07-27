"""does clean_triage produce the shape the model expects?

"""

import pandas as pd

from src import data
from conftest import load_real_sample


def test_clean_triage_produces_valid_schema():
    raw = load_real_sample(n=50, seed=1)
    df = data.clean_triage(raw)

    # target: only valid ESI labels remain, as a gap-free integer column
    assert df[data.TARGET].notna().all()
    assert df[data.TARGET].isin([1, 2, 3, 4, 5]).all()
    assert df[data.TARGET].dtype.kind in "iu"

    # vitals were imputed - no missing values should remain
    for col in data.VITALS:
        if col in df.columns:
            assert df[col].isna().sum() == 0

    # expected columns are present (schema check)
    assert set(data.VITALS).issubset(df.columns)
    assert len(data.chief_complaint_columns(df)) > 0

    # blank categorical text was standardised, never left as raw blanks
    for col in data.DEMOGRAPHICS + data.ADMIN + data.LEAKAGE:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            assert (df[col] == "").sum() == 0
