"""
===============================================================================
 PAGE 2 : DATA DETECTIVE   (lab Part A: understand, clean and select features)
 Tabs follow the lab tasks: A1-A2 | A3 | A4 | A5 | A6-A7 | A8
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
    page_header(
        "Data Detective",
        "Before any modelling: what is in the data, what is broken, and which columns deserve to be "
        "predictors. Every decision here is backed by evidence, not hunches.")

    tabs = st.tabs(["1. Inspect and clean", "2. Explore", "3. Feature audit",
                    "4. Hypothesis tests", "5. Redundancy", "6. Lasso cross-check"])

    # ================================================================ A1-A2
    with tabs[0]:
        ins, cl = R["inspect"], R["cleaning"]
        c = st.columns(5)
        c[0].metric("Rows loaded", f"{ins['n_rows_raw']:,}")
        c[1].metric("Columns", ins["n_cols"])
        c[2].metric("Duplicates removed", cl["duplicates_removed"])
        c[3].metric("Missing values filled", sum(cl["missing_before"].values()))
        c[4].metric("Clean rows", f"{cl['n_rows_clean']:,}")
        thought(
            "I started by asking what one row is (one customer), then checked types, duplicates and gaps. "
            "Duplicates would double-count customers and can leak into the test set. Gaps were small "
            "(18 in bmi, 18 in exercise_freq), so I filled them with the median, which is not thrown off by "
            f"outliers. In this data bmi is nearly symmetric (mean {cl['bmi']['mean']:.2f}, median "
            f"{cl['bmi']['median']:.1f}), so either choice gives almost the same fill.")
        rows = [{"Column": k, "Type": v["kind"], "Distinct values": v["n_unique"], "Missing": v["missing"]}
                for k, v in ins["columns"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"Duplicate customers found: {', '.join(cl['duplicate_ids'])}. "
                   f"After cleaning, all {cl['unique_ids']:,} customer IDs are unique.")
        qa_block(QA["A1"] + QA["A2"], "Lab reflections A1 and A2")

    # ================================================================ A3
    with tabs[1]:
        e = R["eda"]
        thought(
            "The target is the first thing to understand. Its shape tells me what a linear model will struggle "
            "with, and the smoker and tier splits tell me where the money is.")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.hist_charge(e["hist"]))
            st.caption(f"Skew = {e['skew']:.2f}. Mean ${e['mean']:,.0f} sits above the median ${e['median']:,.0f}: "
                       "a long tail of expensive customers.")
        with right:
            st.plotly_chart(charts.box_groups(
                e["smoker_box"], ["no", "yes"], {"no": "#7B8B91", "yes": TIER_COLORS["High"]},
                "Charge by smoker status", labels={"no": "Non-smoker", "yes": "Smoker"}))
            g = e["smoker_groups"]
            st.caption(f"Smokers average ${g['yes']['mean']:,.0f}; non-smokers ${g['no']['mean']:,.0f}.")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.box_groups(e["tier_box"], TIERS, TIER_COLORS, "Charge by risk tier"))
        with right:
            ct = pd.DataFrame(e["smoker_tier_crosstab"]).T[TIERS]
            ct.index = ["Non-smoker", "Smoker"]
            share = pd.Series(e["tier_share"])[TIERS]
            st.markdown("**Customers by tier and smoker status**")
            st.dataframe(ct)
            st.caption("Tier shares: " + ", ".join(f"{t} {share[t]:.0%}" for t in TIERS) +
                       ". A model that always answers 'Medium' would already be right about half the time.")
        qa_block(QA["A3"], "Lab reflection A3")

    # ================================================================ A4
    with tabs[2]:
        A = R["audit"]
        thought(
            "Every column went into one of three buckets: useful, redundant, or irrelevant. First I asked whether "
            "it could plausibly affect medical cost. Then I checked the evidence: correlations for numbers, and "
            "group-average spreads plus an ANOVA test for categories.")
        v = pd.DataFrame([{"Column": k, "Decision": d["verdict"], "Evidence": d["evidence"]}
                          for k, d in A["verdicts"].items()])
        st.dataframe(v, hide_index=True)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.corr_bars(A["corr_target"]))
            st.caption(f"lucky_number correlates at {A['corr_target']['lucky_number']:+.3f}: a random number "
                       "says nothing about health.")
        with right:
            col = st.selectbox("Inspect a categorical column", list(A["categorical"]), index=4,
                               format_func=lambda x: x.replace("_", " "))
            info = A["categorical"][col]
            st.plotly_chart(charts.group_means(info, R["eda"]["mean"], f"Average charge by {col.replace('_', ' ')}"))
            verdict = "no real effect" if info["anova_p"] > 0.05 else "a statistically real effect"
            st.caption(f"Spread {info['spread_pct']:.1f}% across {info['n_categories']} categories "
                       f"(smallest group n = {info['min_group_n']}). ANOVA p = {info['anova_p']:.3f}: {verdict}.")
        st.info("**The zodiac trap.** Zodiac sign shows the biggest spread of the junk columns, but it has 12 groups "
                "with as few as 83 customers each, so some look high or low by chance alone. Always check group sizes "
                "and run a significance test before trusting a pattern.")
        qa_block(QA["A4"], "Lab reflection A4")

    # ================================================================ A5
    with tabs[3]:
        T = R["tests"]
        thought(
            "The business question is 'do smokers really pay more, or is it chance?'. That is a comparison of two "
            "averages, so I used a two-sample t-test (Welch version, which does not assume equal spread).")
        c = st.columns(4)
        c[0].metric("Smokers / non-smokers", f"{T['n_smokers']} / {T['n_nonsmokers']}")
        c[1].metric("Average gap", f"${T['gap']:,.0f}")
        c[2].metric("t-statistic", f"{T['t']:.1f}")
        c[3].metric("p-value", f"{T['p']:.1e}")
        left, right = st.columns(2)
        with left:
            st.markdown(
                "**Hypotheses**  \n"
                "H0: smokers and non-smokers have the same average annual charge.  \n"
                "H1: their average annual charge is different.\n\n"
                "**Decision at alpha = 0.05:** reject H0.")
        with right:
            st.markdown(
                "**Reading the p-value in plain language.** It answers: *if smoking really made no difference, how "
                "often would luck alone produce a gap this large?* Here the answer is essentially never "
                f"(about {T['p']:.1e}; a notebook prints this as 0.00000). So the "
                f"${T['gap']:,.0f} gap is real.")
        st.markdown(f"**Age and charge:** correlation r = {T['age_r']:.2f} (p = {T['age_p']:.1e}). Older customers "
                    f"pay more, but age alone explains only about {T['age_r'] ** 2:.0%} of the variation.")
        qa_block(QA["A5"], "Lab reflection A5")

    # ================================================================ A6-A7
    with tabs[4]:
        D = R["redundancy"]
        thought(
            "Two pairs of columns carry the same information. Duplicates make regression coefficients unstable "
            "because the model cannot tell which column deserves the credit. I measured it with correlation and VIF, "
            "then kept the more interpretable column of each pair.")
        c = st.columns(2)
        c[0].metric("bmi vs weight_kg correlation", f"{D['bmi_weight_corr']:.3f}")
        c[1].metric("age vs date_of_birth correlation", f"{D['age_dob_corr']:.2f}")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(charts.corr_heatmap(D["corr_matrix"]["labels"], D["corr_matrix"]["values"]))
        with right:
            st.plotly_chart(charts.vif_bars(D["vif_before"], D["vif_after"]))
        st.success("Decision: keep **age** (drop date_of_birth) and keep **bmi** (drop weight_kg). "
                   "After dropping weight, every VIF is below 3.")
        st.markdown("**Encoding categories (A7).** Sex, smoker and region become 0/1 columns, dropping one category "
                    "each (`drop_first=True`) because it can always be worked out from the others. Without that, the "
                    "columns would be perfect duplicates (the dummy variable trap).")
        inputs = [c_ for c_ in R["modelling_frame"]["columns"] if c_ not in ("annual_charge", "risk_tier")]
        st.code("Final model inputs: " + ", ".join(inputs), language="text")
        qa_block(QA["A6"] + QA["A7"], "Lab reflections A6 and A7")

    # ================================================================ A8
    with tabs[5]:
        L = R["lasso_audit"]
        thought(
            "My audit used judgement. Lasso is an independent check: it shrinks useless coefficients to exactly "
            "zero. I gave it every column including the junk and raised the penalty (alpha) to see who survives.")
        rows = pd.DataFrame([{
            "alpha": r["alpha"], "R\u00b2 on unseen customers": round(r["r2"], 3),
            "Features kept (of %d)" % L["n_features_total"]: r["n_kept"],
            "Junk columns still present": len(r["junk_left"]),
            "Survivors at this alpha": ", ".join(r["kept"]) if r["n_kept"] <= 6 else "(most columns)"}
            for r in L["rows"]])
        st.dataframe(rows, hide_index=True)
        st.plotly_chart(charts.lasso_audit_chart(L["rows"]))
        last = L["rows"][-1]
        st.success(f"At the strongest penalty only {', '.join(last['kept'])} survive, R\u00b2 barely moves "
                   f"({L['rows'][0]['r2']:.3f} with everything vs {last['r2']:.3f} with four features), "
                   "and no junk column is left.")
        qa_block(QA["A8"], "Lab reflection A8")

    L = R["lasso_audit"]
    key_insights([
        f"**The data needed little repair:** {R['cleaning']['duplicates_removed']} duplicate rows and "
        f"{sum(R['cleaning']['missing_before'].values())} missing values out of {R['inspect']['n_rows_raw']:,} rows.",
        f"**Charges are right-skewed (skew {R['eda']['skew']:.2f}) and tiers are imbalanced** "
        f"(Medium {R['eda']['tier_share']['Medium']:.0%}), so accuracy alone can mislead and a straight line will "
        "under-predict the largest bills.",
        f"**Smoking is real, large and not chance:** a ${R['tests']['gap']:,.0f} average gap (p = {R['tests']['p']:.0e}).",
        "**Five columns are pure noise** (lucky number, colour, zodiac, contact, marketing opt-in). Correlation, ANOVA "
        "and Lasso all agree, and dropping them costs no accuracy.",
        f"**Two columns are duplicates:** weight (VIF {R['redundancy']['vif_before']['weight_kg']:.0f}) and date of birth. "
        "Keeping bmi and age gives the same information with stable coefficients.",
    ])
