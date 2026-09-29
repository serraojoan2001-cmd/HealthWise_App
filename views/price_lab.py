"""
===============================================================================
 PAGE 5 : PRICE LAB   (the live simulator - lab tasks E2 / GUI)
 Change any customer detail and the tier and premium recalculate instantly.
 Everything runs through the SAME two-stage predict() saved in task E1:
     Stage 1 (Decision Tree) picks the tier -> Stage 2 (that tier's regression) prices it.

 What you can do here
   * Presets ........ jump to a typical customer type
   * Sliders ........ change any factor and watch the price
   * Pin baseline ... freeze a customer, change one thing, see the exact dollar effect
   * Why this price . waterfall of where the premium comes from
   * One-factor sweep  vary one factor across its whole range
   * Two-factor map . price and tier for every combination of two factors
===============================================================================
"""
import numpy as np
import pandas as pd
import streamlit as st

from content.qa import QA
from utils import charts
from utils.engine import (FACTOR_LABELS, NUMERIC_FACTORS, contributions, flat_quote, grid, load_report,
                          money, predict, sweep)
from utils.theme import key_insights, page_header, qa_block, thought, tier_chip

TIERS = ["Low", "Medium", "High"]
DEFAULTS = dict(sim_age=40, sim_bmi=27.0, sim_children=1, sim_exercise=3,
                sim_sex="male", sim_smoker="no", sim_region="north")
KEYMAP = dict(age="sim_age", bmi="sim_bmi", children="sim_children", exercise_freq="sim_exercise",
              sex="sim_sex", smoker="sim_smoker", region="sim_region")

PRESETS = {
    "Young, healthy, non-smoker": dict(sim_age=25, sim_bmi=23.0, sim_children=0, sim_exercise=5,
                                       sim_sex="female", sim_smoker="no", sim_region="south"),
    "Typical mid-career": dict(sim_age=45, sim_bmi=28.0, sim_children=2, sim_exercise=4,
                               sim_sex="male", sim_smoker="no", sim_region="north"),
    "Smoker, average build": dict(sim_age=40, sim_bmi=27.0, sim_children=1, sim_exercise=3,
                                  sim_sex="male", sim_smoker="yes", sim_region="north"),
    "Older, high BMI, smoker": dict(sim_age=60, sim_bmi=36.0, sim_children=2, sim_exercise=1,
                                    sim_sex="male", sim_smoker="yes", sim_region="west"),
    "Older, fit, non-smoker": dict(sim_age=58, sim_bmi=24.0, sim_children=1, sim_exercise=5,
                                   sim_sex="female", sim_smoker="no", sim_region="east"),
}
SWEEP_VALUES = {
    "age": list(range(18, 65)), "bmi": [round(x, 1) for x in np.arange(16, 44.5, 0.5)],
    "children": list(range(0, 5)), "exercise_freq": list(range(0, 8)),
    "sex": ["female", "male"], "smoker": ["no", "yes"], "region": ["east", "north", "south", "west"],
}
GRID_VALUES = {"age": np.arange(18, 65, 2), "bmi": np.arange(16, 45, 1.0),
               "children": np.arange(0, 5), "exercise_freq": np.arange(0, 8)}


# ---------------------------------------------------------------- state helpers
def _customer() -> dict:
    s = st.session_state
    return dict(age=s.sim_age, bmi=s.sim_bmi, children=s.sim_children, exercise_freq=s.sim_exercise,
                sex=s.sim_sex, smoker=s.sim_smoker, region=s.sim_region)


def _apply_preset(values: dict):
    for k, v in values.items():
        st.session_state[k] = v


def _pin():
    cust = _customer()
    tier, premium, _ = predict(cust)
    st.session_state["sim_baseline"] = dict(customer=cust, tier=tier, premium=premium)


def _unpin():
    st.session_state.pop("sim_baseline", None)


def render():
    R = load_report()
    sim = R["simulator"]
    coefs = R["regression"]["tier_coefs"]

    # keep the inputs when the user leaves this page and comes back
    saved = st.session_state.get("_sim_saved", DEFAULTS)
    for k, v in saved.items():
        st.session_state.setdefault(k, v)

    page_header(
        "Price Lab",
        "Change any detail about a customer and watch the tier and the premium rebuild instantly. "
        "Pin a baseline first to see the exact dollar effect of a single change.")

    # ---------------------------------------------------------------- presets
    st.markdown("**Start from a typical customer**")
    cols = st.columns(len(PRESETS))
    for col, (name, vals) in zip(cols, PRESETS.items()):
        col.button(name, on_click=_apply_preset, args=(vals,), key=f"preset_{name}", width="stretch")

    # ---------------------------------------------------------------- inputs | result
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("#### Customer details")
        st.slider("Age", 18, 64, key="sim_age", help="Years. The model was trained on ages 18 to 64.")
        st.slider("BMI", 15.0, 50.0, step=0.1, key="sim_bmi",
                  help=f"Body mass index. Training data covered {sim['ranges']['bmi']['min']:.0f} to "
                       f"{sim['ranges']['bmi']['max']:.1f}.")
        st.slider("Children", 0, 4, key="sim_children")
        st.slider("Workouts per week", 0, 7, key="sim_exercise")
        c1, c2 = st.columns(2)
        with c1:
            st.radio("Sex", ["male", "female"], horizontal=True, key="sim_sex")
        with c2:
            st.radio("Smoker", ["no", "yes"], horizontal=True, key="sim_smoker")
        st.selectbox("Region", ["north", "south", "east", "west"], key="sim_region")

        b1, b2 = st.columns(2)
        b1.button("Pin as baseline", on_click=_pin, type="primary", width="stretch",
                  help="Freeze this customer, then change one detail to see its exact dollar effect.")
        b2.button("Clear baseline", on_click=_unpin, width="stretch",
                  disabled="sim_baseline" not in st.session_state)

    cust = _customer()
    tier, premium, proba = predict(cust)
    flat = flat_quote(cust)
    base = st.session_state.get("sim_baseline")

    with right:
        delta_html = ""
        if base:
            diff = premium - base["premium"]
            cls = "hw-delta-up" if diff > 0 else "hw-delta-down"
            move = f" ({base['tier']} to {tier})" if base["tier"] != tier else ""
            if abs(diff) < 0.5:
                delta_html = '<div style="margin-top:.6rem">Same as the pinned baseline.</div>'
            else:
                delta_html = (f'<div style="margin-top:.6rem">vs pinned baseline: '
                              f'<span class="{cls}">{money(diff, signed=True)}</span>{move}</div>')
        st.markdown(
            f'<div class="hw-ticket"><div class="lbl">Estimated annual premium</div>'
            f'<div class="amt">{money(premium)}</div>'
            f'<div style="margin-top:.55rem">{tier_chip(tier)}&nbsp;&nbsp;'
            f'<span style="color:#7B8B91">Old flat-model quote: {money(flat)} '
            f'({money(premium - flat, signed=True)} vs flat)</span></div>{delta_html}</div>',
            unsafe_allow_html=True)
        st.plotly_chart(charts.premium_gauge(sim["tier_ranges"], premium, tier, flat,
                                             base["premium"] if base else None))
        st.plotly_chart(charts.probability_bar(proba))

        # guardrail messages
        top = max(proba.values())
        if top < 0.7:
            st.warning(f"Borderline tier: the tier-sorter is only {top:.0%} sure. In production this customer "
                       "would go to manual review before a quote is issued.")
        for f, lab in [("bmi", "BMI"), ("age", "Age")]:
            r = sim["ranges"][f]
            if cust[f] < r["min"] or cust[f] > r["max"]:
                st.warning(f"{lab} {cust[f]} is outside the range the model was trained on "
                           f"({r['min']:.0f} to {r['max']:.1f}). Treat this quote as an extrapolation.")

    # ---------------------------------------------------------------- explanation tabs
    tabs = st.tabs(["Why this price", "Baseline vs now", "One-factor sweep", "Two-factor map"])

    with tabs[0]:
        thought(
            "The premium is the tier's base rate plus a dollar amount for each factor (its coefficient times the "
            "customer's value). Reading it as a waterfall shows what is pushing this particular price up or down.")
        contrib = contributions(cust, tier)
        st.plotly_chart(charts.waterfall(contrib))
        top_row = contrib.iloc[1:].sort_values("dollars", key=abs, ascending=False).iloc[0]
        st.caption(f"Largest single component right now: **{top_row['factor']}** ({money(top_row['dollars'], signed=True)}). "
                   f"Base rate for the {tier} tier: {money(contrib.iloc[0]['dollars'])}.")

    with tabs[1]:
        if not base:
            st.info("No baseline pinned yet. Set up a customer, press **Pin as baseline**, then change one detail. "
                    "Try: pin the default customer, then set **Smoker = yes**.")
        else:
            rows = []
            for k, lab in FACTOR_LABELS.items():
                b, n = base["customer"][k], cust[k]
                rows.append({"Detail": lab, "Baseline": b, "Now": n, "Changed": "yes" if b != n else ""})
            rows.append({"Detail": "Risk tier", "Baseline": base["tier"], "Now": tier,
                         "Changed": "yes" if base["tier"] != tier else ""})
            rows.append({"Detail": "Annual premium", "Baseline": money(base["premium"]), "Now": money(premium),
                         "Changed": money(premium - base["premium"], signed=True)})
            st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True)
            if base["tier"] != tier:
                st.warning(f"The customer crossed a tier boundary ({base['tier']} to {tier}), so the premium is now "
                           f"computed by a different regression with a different base rate. This is why some single "
                           f"changes move the price by thousands of dollars.")

    with tabs[2]:
        thought(
            "To see how sensitive the price is to one factor, I hold everything else fixed at the current customer "
            "and price every value of that factor. Steps in the line are tier changes.")
        factor = st.selectbox("Factor to vary", list(FACTOR_LABELS), format_func=lambda k: FACTOR_LABELS[k], key="sweep_factor")
        res = sweep(cust, factor, SWEEP_VALUES[factor])
        if factor in NUMERIC_FACTORS:
            st.plotly_chart(charts.sweep_line(res, factor, FACTOR_LABELS[factor], cust[factor]))
            lo, hi = res["premium"].min(), res["premium"].max()
            st.caption(f"Across its whole range, {FACTOR_LABELS[factor]} moves this customer's premium from "
                       f"{money(lo)} to {money(hi)} (a swing of {money(hi - lo)}).")
        else:
            st.plotly_chart(charts.sweep_bars(res, factor, FACTOR_LABELS[factor], cust[factor]))

    with tabs[3]:
        thought(
            "Two factors at once shows interactions the single sweep cannot: for example, BMI matters far more for "
            "an older customer than for a young one, because they sit near a tier boundary.")
        m1, m2, m3 = st.columns(3)
        fx = m1.selectbox("Horizontal axis", NUMERIC_FACTORS, index=0, format_func=lambda k: FACTOR_LABELS[k], key="grid_x")
        options_y = [f for f in NUMERIC_FACTORS if f != fx]
        default_y = options_y.index("bmi") if "bmi" in options_y else 0
        fy = m2.selectbox("Vertical axis", options_y, index=default_y, format_func=lambda k: FACTOR_LABELS[k], key="grid_y")
        smoker_choice = m3.radio("Smoker in the map", ["as current customer", "no", "yes"], key="grid_smoker")
        smoker_map = cust["smoker"] if smoker_choice == "as current customer" else smoker_choice
        xs, ys = GRID_VALUES[fx], GRID_VALUES[fy]
        gres = grid(dict(cust, smoker=smoker_map), fx, xs, fy, ys)
        f1, f2 = charts.grid_maps(gres, fx, fy, xs, ys, FACTOR_LABELS[fx], FACTOR_LABELS[fy], cust[fx], cust[fy])
        g1, g2 = st.columns(2)
        g1.plotly_chart(f1)
        g2.plotly_chart(f2)
        st.caption("The white diamond is the current customer. Sharp colour boundaries on the right are tier changes; "
                   "they are where the premium jumps.")

    # ---------------------------------------------------------------- lab answers + insights
    qa_block(QA["E2"], "Lab reflection E2 (answers based on this simulator)")
    st.session_state["_sim_saved"] = {k: st.session_state[k] for k in DEFAULTS}

    lo_age, hi_age = coefs["Low"]["age"], coefs["High"]["age"]
    key_insights([
        f"**Right now:** this customer is **{tier} risk** at **{money(premium)}** "
        f"({money(premium - flat, signed=True)} vs the old flat quote).",
        f"**Smoking is the biggest single lever.** Flip it and the customer usually crosses a tier boundary, so the "
        f"price rises twice: a higher base rate and a bigger smoker effect "
        f"(+{money(coefs['Low']['smoker_yes'])} in Low, +{money(coefs['High']['smoker_yes'])} in High).",
        f"**Prices move in steps, not smoothly.** Inside a tier a year of age is worth only "
        f"{money(lo_age)}-{money(hi_age)}, but crossing a tier boundary is worth thousands. Two near-identical "
        "customers on either side of a boundary can get very different quotes.",
        f"**Children, exercise and region are small levers** (about {money(coefs['Medium']['children'])} per child in "
        "the Medium tier, and region under a few hundred dollars).",
        f"**Stay inside the training range** (ages {sim['ranges']['age']['min']:.0f}-{sim['ranges']['age']['max']:.0f}, "
        f"BMI {sim['ranges']['bmi']['min']:.0f}-{sim['ranges']['bmi']['max']:.1f}); outside it, quotes are extrapolations.",
    ])
