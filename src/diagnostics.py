""" explaining a model's under-triage errors.
"""

import numpy as np
import pandas as pd


def undertriage_profile(X_test, y_test, pred, level, cols, ref_level, sort_by_effect=True):
    """Compare under-triaged vs. correctly-triaged patients.

    For patients whose true ESI is ``level``, splits them into those the
    model under-triaged vs. those
    correctly triaged, then reports means/medians for ``cols`` alongside
    Cohen's d, which puts every column on the same scale so the features
    that most separate the two groups appear first. 
    ``ref_level`` (the level under-triaged patients were mistaken for) is included for
    context.

    Parameters
    ----------
    cols : list of str
        Which X_test columns to profile? - e.g. raw vitals (data.VITALS) or
        engineered features (e.g. ["shock_index", "red_flag_count"])
    Returns
    -------
    pandas.DataFrame indexed by column in ``cols``.
    """
    y_test, pred = np.asarray(y_test), np.asarray(pred)
    is_level = y_test == level
    if is_level.sum() == 0:
        raise ValueError(f"No patients with true ESI {level} in y_test")

    under = is_level & (pred > level)
    correct = is_level & (pred == level)

    u = X_test.loc[under, cols]
    c = X_test.loc[correct, cols]
    r = X_test.loc[y_test == ref_level, cols]

    nu, nc = len(u), len(c)
    pooled_sd = np.sqrt(((nu - 1) * u.var(ddof=1) + (nc - 1) * c.var(ddof=1)) / (nu + nc - 2))
    cohens_d = (u.mean() - c.mean()) / pooled_sd.replace(0, np.nan)

    tbl = pd.DataFrame({
        "under_triaged_mean": u.mean(),
        "correct_mean": c.mean(),
        f"esi{ref_level}_mean": r.mean(),
        "under_triaged_median": u.median(),
        "correct_median": c.median(),
        "mean_diff": u.mean() - c.mean(),
        "cohens_d": cohens_d,
    }).round(2)

    if sort_by_effect:
        tbl = tbl.reindex(tbl["cohens_d"].abs().sort_values(ascending=False).index)
    return tbl


def cc_enrichment(X_test, y_test, pred, level, min_count=5, top=12):
    """Chief complaints most over-represented among under-triaged patients.

    Within patients whose true ESI is ``level``, compares how often each
    chief-complaint flag fires among under-triaged vs.
    correctly-triaged patients, ranked by the gap between the two rates.

    Returns
    -------
    pandas.DataFrame of cc_* columns, largest gap first.
    """
    y_test, pred = np.asarray(y_test), np.asarray(pred)
    is_level = y_test == level
    under = is_level & (pred > level)
    correct = is_level & (pred == level)
    cc_cols = [c for c in X_test.columns if c.startswith("cc_")]

    tbl = pd.DataFrame({
        "under_triaged_count": X_test.loc[under, cc_cols].sum().astype(int),
        "under_triaged_pct": 100 * X_test.loc[under, cc_cols].mean(),
        "correct_pct": 100 * X_test.loc[correct, cc_cols].mean(),
    })
    tbl["gap"] = tbl["under_triaged_pct"] - tbl["correct_pct"]
    tbl = tbl[tbl["under_triaged_count"] >= min_count]
    return tbl.sort_values("gap", ascending=False).head(top).round(1)
