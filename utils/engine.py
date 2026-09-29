"""
===============================================================================
 FILE : utils/engine.py
 ROLE : THE PRICING ENGINE (loads the saved models and prices customers)
 Two-stage logic, exactly as in lab tasks E1/E2:
   Stage 1 -> classify the customer into Low / Medium / High
   Stage 2 -> price them with THAT tier's own regression
===============================================================================
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
TIERS = ["Low", "Medium", "High"]

FACTOR_LABELS = {
    "age": "Age", "bmi": "BMI", "children": "Children", "exercise_freq": "Exercise / week",
    "sex": "Sex", "smoker": "Smoker", "region": "Region",
}
NUMERIC_FACTORS = ["age", "bmi", "children", "exercise_freq"]
CATEGORICAL_FACTORS = ["sex", "smoker", "region"]


@st.cache_resource(show_spinner="Loading the pricing engine...")
def load_engine():
    """Load the joblib bundle saved by train_model.py (lab task E1)."""
    path = ARTIFACTS / "healthwise_model.joblib"
    if not path.exists():
        st.error("`artifacts/healthwise_model.joblib` is missing. Run `python train_model.py` "
                 "and commit the artifacts folder.")
        st.stop()
    try:
        return joblib.load(path)
    except Exception as exc:  # most common cause: scikit-learn version mismatch
        st.error("The model file could not be loaded. This is almost always a scikit-learn "
                 "version mismatch: `requirements.txt` must pin the SAME scikit-learn version "
                 "that created the file (see README, 'Troubleshooting').\n\n"
                 f"Details: `{exc}`")
        st.stop()


@st.cache_data(show_spinner=False)
def load_report() -> dict:
    """Aggregate statistics computed by train_model.py (no customer-level rows)."""
    path = ARTIFACTS / "report.json"
    if not path.exists():
        st.error("`artifacts/report.json` is missing. Run `python train_model.py`.")
        st.stop()
    return json.loads(path.read_text())


def _frame(customer: dict, M: dict) -> pd.DataFrame:
    row = pd.DataFrame([customer])
    for k in M["num"]:                      # fill anything left blank with the training median
        if k not in row or pd.isna(row.at[0, k]):
            row[k] = M["med"][k]
    return row[M["features"]]


def predict(customer: dict):
    """Single customer -> (tier, premium, probabilities dict). Same logic as the lab's predict()."""
    M = load_engine()
    row = _frame(customer, M)
    tier = M["clf"].predict(row)[0]                                   # Stage 1
    premium = float(M["regs"][tier].predict(row)[0])                  # Stage 2 (that tier's model)
    proba = dict(zip(M["clf"].classes_, M["clf"].predict_proba(row)[0]))
    return tier, round(premium, 2), {t: float(proba.get(t, 0.0)) for t in TIERS}


def flat_quote(customer: dict) -> float:
    """What the OLD one-formula-for-everyone model would quote (the board's 'before')."""
    M = load_engine()
    return float(M["global"].predict(_frame(customer, M))[0])


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised two-stage prediction for many customers. Returns tier + premium columns."""
    M = load_engine()
    X = df.copy()
    for k in M["num"]:
        if k not in X:
            X[k] = M["med"][k]
        X[k] = pd.to_numeric(X[k], errors="coerce").fillna(M["med"][k])
    X = X[M["features"]]
    tiers = M["clf"].predict(X)
    prem = np.zeros(len(X))
    for t in TIERS:
        m = tiers == t
        if m.any():
            prem[m] = M["regs"][t].predict(X[m])
    return pd.DataFrame({"tier": tiers, "premium": prem}, index=df.index)


def contributions(customer: dict, tier: str) -> pd.DataFrame:
    """Break the premium into $ pieces: intercept + coefficient x value for each factor."""
    M = load_engine()
    row = _frame(customer, M)
    pipe = M["regs"][tier]
    Z = pipe.named_steps["pre"].transform(row)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    names = [n.split("__", 1)[1] for n in pipe.named_steps["pre"].get_feature_names_out()]
    coef = pipe.named_steps["model"].coef_
    parts = dict(zip(names, (Z[0] * coef)))
    grouped = {
        "Base rate": float(pipe.named_steps["model"].intercept_),
        "Age": parts["age"], "BMI": parts["bmi"], "Children": parts["children"],
        "Exercise": parts["exercise_freq"], "Sex": parts["sex_male"],
        "Smoker": parts["smoker_yes"],
        "Region": parts["region_north"] + parts["region_south"] + parts["region_west"],
    }
    return pd.DataFrame({"factor": list(grouped), "dollars": list(grouped.values())})


def sweep(base: dict, factor: str, values) -> pd.DataFrame:
    """Vary ONE factor across `values`, hold everything else fixed, price every point."""
    df = pd.DataFrame([dict(base, **{factor: v}) for v in values])
    res = predict_batch(df)
    res[factor] = list(values)
    return res


def grid(base: dict, fx: str, xs, fy: str, ys) -> pd.DataFrame:
    """Two-factor price map."""
    rows = [dict(base, **{fx: x, fy: y}) for y in ys for x in xs]
    df = pd.DataFrame(rows)
    res = predict_batch(df)
    res[fx], res[fy] = df[fx].values, df[fy].values
    return res


def money(x: float, signed: bool = False) -> str:
    if signed:
        return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"
    return f"${x:,.0f}"
