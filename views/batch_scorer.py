"""
===============================================================================
 PAGE 6 : BATCH SCORER   (lab task E3 - score unseen customers, then apply a change)
 1) Upload a CSV (or use the sample file)  ->  predicted tier + premium per customer
 2) Apply a what-if change (e.g. smoker -> yes, +2 children)  ->  the E3 results table
===============================================================================
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from content.qa import QA
from utils.engine import load_report, money, predict_batch
from utils.theme import key_insights, page_header, qa_block, thought

REQUIRED = ["age", "bmi", "children", "exercise_freq", "sex", "smoker", "region"]
VALID = {"sex": {"male", "female"}, "smoker": {"yes", "no"}, "region": {"north", "south", "east", "west"}}
SAMPLE = Path(__file__).resolve().parent.parent / "data_samples" / "sample_customers.csv"

CHANGES = {
    "Make the customer a smoker": ("smoker", lambda s, n: "yes"),
    "Make the customer a non-smoker": ("smoker", lambda s, n: "no"),
    "Add children": ("children", lambda s, n: (s + n).clip(0, 4)),
    "Change age by (years)": ("age", lambda s, n: (s + n).clip(18, 64)),
    "Change BMI by (points)": ("bmi", lambda s, n: (s + n).clip(15, 50)),
    "Change weekly workouts by": ("exercise_freq", lambda s, n: (s + n).clip(0, 7)),
}


def _clean(df: pd.DataFrame):
    """Validate the uploaded file. Returns (clean_df, list_of_problems)."""
    problems = []
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        problems.append(f"Missing columns: {', '.join(missing_cols)}. Required: {', '.join(REQUIRED)}.")
        return df, problems
    for c, allowed in VALID.items():
        df[c] = df[c].astype(str).str.strip().str.lower()
        bad = sorted(set(df[c]) - allowed)
        if bad:
            problems.append(f"Column '{c}' has unexpected values {bad}. Allowed: {sorted(allowed)}.")
    for c in ["age", "bmi", "children", "exercise_freq"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df, problems


def render():
    R = load_report()
    med = R["simulator"]["med"]
    page_header(
        "Batch Scorer",
        "Score a whole file of customers at once, then apply a what-if change (for example: make everyone a smoker) "
        "and see the new premium next to the old one. This is lab task E3.")

    thought(
        "The instructor's unseen customers arrive as a file, so the engine has to work on many rows at once. Missing "
        "numeric values are filled with the training medians (as in the notebook), and every row goes through the "
        "same Stage 1 then Stage 2 logic as the Price Lab.")

    src = st.radio("Customer file", ["Use the sample customers", "Upload my own CSV"], horizontal=True)
    df = None
    if src == "Use the sample customers":
        df = pd.read_csv(SAMPLE)
        st.caption("Sample file: eight made-up customers for demonstration (not from the course dataset).")
    else:
        up = st.file_uploader("CSV with columns: " + ", ".join(REQUIRED) + " (customer_id optional)", type="csv")
        if up:
            df = pd.read_csv(up)

    with st.expander("Download the CSV template"):
        st.download_button("Download template", pd.read_csv(SAMPLE).to_csv(index=False), "customer_template.csv")

    if df is None:
        st.info("Upload a CSV to begin, or switch to the sample customers.")
        key_insights([
            "**Batch scoring runs the same engine as the Price Lab**, so results are consistent.",
            "**Required columns:** " + ", ".join(REQUIRED) + ".",
            "**Blank numeric cells** are filled with training medians: "
            + ", ".join(f"{k} = {v:g}" for k, v in med.items()) + ".",
        ])
        return

    df, problems = _clean(df)
    if problems:
        for p in problems:
            st.error(p)
        st.stop()

    n_blank = int(df[["age", "bmi", "children", "exercise_freq"]].isna().sum().sum())
    idcol = "customer_id" if "customer_id" in df.columns else None
    ids = df[idcol].astype(str) if idcol else pd.Series([f"Row {i + 1}" for i in range(len(df))], index=df.index)
    scored = predict_batch(df)
    out = pd.DataFrame({"Customer": ids, "Predicted tier": scored["tier"], "Annual premium": scored["premium"].round(0)})
    if n_blank:
        st.warning(f"{n_blank} blank numeric value(s) were filled with the training medians.")

    st.subheader("1. Predicted tier and premium")
    st.dataframe(out.style.format({"Annual premium": "${:,.0f}"}), hide_index=True)
    st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")

    # ---------------------------------------------------------------- what-if
    st.subheader("2. Apply a what-if change")
    c1, c2, c3 = st.columns([2, 1, 2])
    change = c1.selectbox("Change", list(CHANGES))
    needs_n = "Add" in change or "by" in change
    amount = c2.number_input("Amount", value=2.0 if "children" in change.lower() else 5.0, step=1.0,
                             disabled=not needs_n)
    who = c3.multiselect("Customers (leave empty for all)", list(ids))
    field, fn = CHANGES[change]
    mask = ids.isin(who) if who else pd.Series(True, index=df.index)
    changed = df.copy()
    changed.loc[mask, field] = fn(changed.loc[mask, field], amount) if needs_n else fn(None, None)
    new_scored = predict_batch(changed)
    label = change if not needs_n else f"{change} {amount:+g}"

    e3 = pd.DataFrame({
        "Customer": ids, "Predicted tier": scored["tier"], "Premium": scored["premium"].round(0),
        "Change applied": [label if m else "none" for m in mask],
        "Updated tier": new_scored["tier"], "Updated premium": new_scored["premium"].round(0)})
    e3["Difference"] = e3["Updated premium"] - e3["Premium"]
    st.dataframe(e3.style.format({"Premium": "${:,.0f}", "Updated premium": "${:,.0f}", "Difference": "{:+,.0f}"}),
                 hide_index=True)
    st.download_button("Download E3 table", e3.to_csv(index=False), "e3_results.csv")

    affected = e3[mask.values]
    crossed = int((affected["Predicted tier"] != affected["Updated tier"]).sum())
    qa_block(QA["E3"], "Lab reflection E3")

    tier_mix = out["Predicted tier"].value_counts()
    key_insights([
        f"**{len(out)} customers scored.** Tier mix: "
        + ", ".join(f"{t} {int(tier_mix.get(t, 0))}" for t in ["Low", "Medium", "High"]) + ".",
        f"**The what-if '{label}'** changes the average premium of the affected customers by "
        f"{money(affected['Difference'].mean(), signed=True)}, and moves {crossed} of {len(affected)} of them into a "
        "different tier.",
        "**Tier changes explain the biggest jumps.** When a change pushes someone into a higher tier, a different "
        "regression prices them, so the increase is larger than the coefficient alone.",
        "**Sense-check every batch:** Low-risk premiums should sit near $5k-$12k, Medium near $11k-$24k and High "
        "near $30k-$59k (the ranges seen in training). Quotes far outside those bands deserve a second look.",
    ])
