"""
===============================================================================
 PAGE 4 : PRICE ENGINE   (Stage 2 - predict the annual charge; lab Part C)
 Tabs: C1 flat baseline | C2 one vs many features | C3 tier engine | honest test | C4 tuning | C5 verdict
===============================================================================
"""
import pandas as pd
import streamlit as st

from content.qa import QA
from utils import charts
from utils.engine import load_report
from utils.theme import TIER_COLORS, key_insights, page_header, qa_block, thought

TIERS = ["Low", "Medium", "High"]


def render():
    R = load_report()
    reg = R["regression"]
    g = R["regression_global"]
    c2, c3, c4 = reg["c2"], reg["c3"], reg["c4"]
    e2e = reg["end_to_end"]
    page_header(
        "Price Engine",
        "Stage 2 of the engine: once we know the tier, what should the annual premium be? "
        "Here the flat model is set up as the benchmark, then beaten one step at a time.")

    tabs = st.tabs(["1. The flat baseline", "2. One feature vs many", "3. A model per tier",
                    "4. Honest test", "5. Tuning", "6. Verdict"])

    # ================================================================ C1
    with tabs[0]:
        thought(
            "Before claiming a better idea, I measured today's approach: one linear formula for every customer. "
            "That number is the benchmark everything else has to beat.")
        c = st.columns(3)
        c[0].metric("MAE (typical error)", f"${g['MAE']:,.0f}")
        c[1].metric("RMSE (penalises big misses)", f"${g['RMSE']:,.0f}")
        c[2].metric("R\u00b2 (variance explained)", f"{g['R2']:.3f}")
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(charts.global_error_by_tier(g["by_tier"]))
        with right:
            bt = pd.DataFrame(g["by_tier"]).T.loc[TIERS]
            table = pd.DataFrame({"Customers": bt["n"].astype(int), "Average charge": bt["mean_charge"],
                                  "Typical error": bt["MAE"], "Error as % of charge": bt["MAE_pct"]})
            st.dataframe(table.style.format({"Average charge": "${:,.0f}", "Typical error": "${:,.0f}",
                                             "Error as % of charge": "{:.0f}%"}))
            st.caption(f"The flat model even quoted a negative premium for {g['n_negative_preds']} customer(s) "
                       f"(lowest: ${g['min_pred']:,.0f}), which is impossible.")
        st.markdown(
            f"R\u00b2 = {g['R2']:.2f} sounds decent, but a typical quote is off by about ${g['MAE']:,.0f}, "
            f"roughly {g['MAE'] / g['mean_charge_test']:.0%} of an average charge. The errors are systematic: the "
            "formula drags everyone towards the middle.")
        qa_block(QA["C1"], "Lab reflection C1")

    # ================================================================ C2
    with tabs[1]:
        thought(
            "Inside one tier I compared a simple model (only the single strongest feature, chosen from the "
            "correlations) with a multiple model (all features). The size of the jump tells me how many factors "
            "drive cost in that tier.")
        st.plotly_chart(charts.simple_vs_multi(c2))
        tbl = pd.DataFrame([{"Tier": t, "Strongest single feature": c2[t]["single"],
                             "Correlation": c2[t]["single_corr"], "R\u00b2 one feature": c2[t]["r2_simple"],
                             "R\u00b2 all features": c2[t]["r2_multi"], "Jump": c2[t]["jump"]} for t in TIERS])
        st.dataframe(tbl.style.format({"Correlation": "{:+.2f}", "R\u00b2 one feature": "{:.3f}",
                                       "R\u00b2 all features": "{:.3f}", "Jump": "{:+.3f}"}), hide_index=True)
        st.info("In the High tier one feature explains very little and the jump is biggest: cost there comes from "
                "several factors acting together. In the Low tier, age alone already explains a lot.")
        qa_block(QA["C2"], "Lab reflection C2")

    # ================================================================ C3
    with tabs[2]:
        thought(
            "If customers really differ by tier, then a separate regression per tier should fit better than one "
            "for everybody. I trained Low, Medium and High separately and pooled their predictions to compare with "
            "the flat model.")
        left, right = st.columns([2, 3])
        with left:
            st.plotly_chart(charts.tier_fit(c3))
        with right:
            st.dataframe(pd.DataFrame([{"Tier": t, "Customers": c3[t]["n"], "R\u00b2": c3[t]["R2"],
                                        "Typical error": c3[t]["MAE"], "Charge spread (std)": c3[t]["charge_std"]}
                                       for t in TIERS]).style.format({"R\u00b2": "{:.3f}", "Typical error": "${:,.0f}",
                                                                      "Charge spread (std)": "${:,.0f}"}), hide_index=True)
            comp = pd.DataFrame({"Model": ["Global flat model (C1)", "Two-stage, perfect tier (C3)"],
                                 "MAE": [g["MAE"], reg["c3_all"]["MAE"]], "R\u00b2": [g["R2"], reg["c3_all"]["R2"]]})
            st.dataframe(comp.style.format({"MAE": "${:,.0f}", "R\u00b2": "{:.3f}"}), hide_index=True)
        st.caption("Why the pooled R\u00b2 (0.99) is higher than any single tier's (0.75-0.80): the three tiers sit at very "
                   "different price levels, so simply knowing the tier explains most of the total variation. Read "
                   "the per-tier R\u00b2 and the dollar error for the real quality of the fit.")
        qa_block(QA["C3"], "Lab reflection C3")

    # ================================================================ Honest test
    with tabs[3]:
        thought(
            "The C3 result quietly assumes Stage 1 never makes a mistake: each customer is priced by their true "
            "tier. A real quote does not know the true tier. So I re-ran the comparison with the classifiers "
            "choosing the tier, on one shared set of unseen customers.")
        c = st.columns(2)
        with c[0]:
            st.plotly_chart(charts.metric_bars(e2e, "MAE", "Average pricing error (lower is better)"))
        with c[1]:
            st.plotly_chart(charts.metric_bars(e2e, "R2", "R\u00b2 (higher is better)"))
        tab = pd.DataFrame(e2e).T
        st.dataframe(tab.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}", "R2": "{:.3f}"}))
        dep = e2e["Two-stage, Decision-Tree router (deployed)"]
        flat = e2e["Global flat model"]
        st.success(f"With the deployed Decision Tree choosing the tier, the engine cuts average error by "
                   f"{1 - dep['MAE'] / flat['MAE']:.0%} (${flat['MAE']:,.0f} to ${dep['MAE']:,.0f}). "
                   "A weaker tier-sorter (Logistic Regression) would keep the MAE gain but lose most of the R\u00b2 "
                   "gain: quality of Stage 1 matters.")

    # ================================================================ C4
    with tabs[4]:
        thought(
            "Even after removing junk, features overlap and each tier has few customers. Ridge and Lasso add a "
            "penalty (alpha) that shrinks unstable coefficients. I swept alpha to find where accuracy holds, and "
            "used Lasso's coefficients to rank the real drivers.")
        tier = st.radio("Tier", TIERS, index=2, horizontal=True)
        d = c4[tier]
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.alpha_sweep(d["sweep"], d["best_alpha"]))
        with right:
            st.plotly_chart(charts.lasso_coefs(d["lasso_coefs"], f"Lasso coefficients, {tier} tier (alpha = {d['best_alpha']})"))
        sw = pd.DataFrame(d["sweep"]).rename(columns={"alpha": "alpha", "ridge": "Ridge R\u00b2", "lasso": "Lasso R\u00b2"})
        st.dataframe(sw.style.format({"Ridge R\u00b2": "{:.3f}", "Lasso R\u00b2": "{:.3f}"}), hide_index=True)
        zeroed = [k for k, v in d["lasso_coefs"].items() if abs(v) < 1e-6]
        st.caption(f"Plain linear regression scores R\u00b2 {d['ols_r2']:.3f} in this tier. "
                   + (f"Lasso set these to exactly zero: {', '.join(zeroed)}." if zeroed else
                      "Lasso kept every feature at this alpha."))
        st.markdown("**What each factor is worth in dollars, tier by tier** (from the deployed models)")
        st.plotly_chart(charts.dollars_by_tier(reg["tier_coefs"]))
        st.dataframe(pd.DataFrame(reg["tier_coefs"]).rename(index={
            "intercept": "Base rate", "age": "Per year of age", "bmi": "Per BMI point", "children": "Per child",
            "exercise_freq": "Per weekly workout", "sex_male": "Male (vs female)", "smoker_yes": "Smoker (vs not)",
            "region_north": "Region north (vs east)", "region_south": "Region south (vs east)",
            "region_west": "Region west (vs east)"})[TIERS].style.format("${:,.0f}"))
        qa_block(QA["C4"], "Lab reflection C4")

    # ================================================================ C5
    with tabs[5]:
        h = c4["High"]
        thought("The model ladder shows what each step of effort actually bought, so I can see where the gain came from.")
        ladder = pd.DataFrame([
            {"Model (High tier)": f"Simple linear ({c2['High']['single']})", "R\u00b2": c2["High"]["r2_simple"],
             "What it tells you": "One feature explains little"},
            {"Model (High tier)": "Multiple linear (all features)", "R\u00b2": c2["High"]["r2_multi"],
             "What it tells you": "Several real drivers together explain most of the variance"},
            {"Model (High tier)": f"Ridge (tuned, alpha = 1)", "R\u00b2": h["ridge_tuned"],
             "What it tells you": "No accuracy gain: the data is clean"},
            {"Model (High tier)": f"Lasso (tuned, alpha = {h['best_alpha']})", "R\u00b2": h["lasso_tuned"],
             "What it tells you": "Same accuracy, and it drops an overlapping feature"}])
        st.dataframe(ladder.style.format({"R\u00b2": "{:.3f}"}), hide_index=True)
        st.success("Biggest gains: adding the real features and splitting customers into tiers. Removing junk and tuning "
                   "alpha changed accuracy very little; they made the model simpler and safer, not more accurate.")
        qa_block(QA["C5"], "Lab reflection C5")

    dep = e2e["Two-stage, Decision-Tree router (deployed)"]
    flat = e2e["Global flat model"]
    key_insights([
        f"**One flat formula is too blunt:** typical error ${g['MAE']:,.0f} "
        f"(about {g['by_tier']['Low']['MAE_pct']:.0f}% of a Low-tier customer's charge), and it over-charges cheap "
        "tiers while under-charging High-risk customers.",
        f"**Separate models per tier win:** error falls to ${reg['c3_all']['MAE']:,.0f} with the right tier, and to "
        f"${dep['MAE']:,.0f} ({1 - dep['MAE'] / flat['MAE']:.0%} lower) when the app picks the tier itself.",
        f"**High-risk customers are the hardest to price** (typical error ${c3['High']['MAE']:,.0f}); Low-risk the easiest "
        f"(${c3['Low']['MAE']:,.0f}).",
        f"**The value of a factor depends on the tier:** smoking adds ${reg['tier_coefs']['Low']['smoker_yes']:,.0f} in "
        f"Low but ${reg['tier_coefs']['High']['smoker_yes']:,.0f} in High. That is exactly why one coefficient for "
        "everyone fails.",
        "**Regularisation confirmed the drivers rather than improving accuracy:** BMI, smoker and age lead; region is "
        "negligible.",
    ])
