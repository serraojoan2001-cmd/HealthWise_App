"""
===============================================================================
 PAGE 7 : GUARDRAILS   (risks, limits and the model card)
 Source : lab Bonus 1 (error propagation), Bonus 2 (PCA), task E1 (saving the model)
 Tabs   : when the tier is wrong | limits and fairness | model card
===============================================================================
"""
import pandas as pd
import streamlit as st

from content.qa import QA
from utils import charts
from utils.engine import load_report, money
from utils.theme import key_insights, page_header, qa_block, thought

TIERS = ["Low", "Medium", "High"]


def render():
    R = load_report()
    risk, cls, meta = R["risk"], R["classification"], R["meta"]
    tree = cls["tree"]
    page_header(
        "Guardrails",
        "Where the engine can go wrong, what that would cost, and what to watch. "
        "A pricing model is only as safe as its weakest step.")

    tabs = st.tabs(["1. When the tier is wrong", "2. Limits and fairness", "3. Model card"])

    # ================================================================ Bonus 1
    with tabs[0]:
        thought(
            "Stage 2 trusts Stage 1 completely. If the tier is wrong, the customer is priced by the wrong "
            "regression. I measured what that does to the quote, using unseen customers and the tree that is "
            "deployed.")
        c = st.columns(3)
        c[0].metric("Customers sent to the wrong tier", f"{risk['share_misrouted']:.1%}")
        c[1].metric("Typical quote error, right tier", money(risk["mae_correct"]))
        c[2].metric("Typical quote error, wrong tier", money(risk["mae_wrong"]),
                    f"{risk['mae_wrong'] / risk['mae_correct']:.0f}x larger", delta_color="inverse")
        st.plotly_chart(charts.routing_bars(risk["routes"]))
        rt = pd.DataFrame([r for r in risk["routes"] if r["actual"] != r["predicted"]]).rename(columns={
            "actual": "Actual tier", "predicted": "Sent to", "n": "Customers", "mean_true": "Average true charge",
            "mean_quote": "Average quote", "mean_error": "Average error"})
        st.dataframe(rt[["Actual tier", "Sent to", "Customers", "Average true charge", "Average quote", "Average error"]]
                     .style.format({"Average true charge": "${:,.0f}", "Average quote": "${:,.0f}",
                                    "Average error": "{:+,.0f}"}), hide_index=True)
        st.caption("Small counts: read these as direction and size, not precise averages.")
        st.warning("The costly case is a High-risk customer priced as Medium (or Low): the quote is thousands of "
                   "dollars below their true cost, year after year.")
        conf = tree["confidence"]
        st.markdown(
            "**Safeguards to add**\n"
            f"- **Confidence gate.** Calls where the sorter is under {conf['threshold']:.0%} sure are only "
            f"{conf['acc_low_conf']:.0%} accurate (vs {conf['acc_high_conf']:.0%} otherwise). Send them to manual review.\n"
            "- **Lean up at the boundary.** For borderline customers, price with the higher tier's model or add a margin.\n"
            "- **Monitor High-to-Medium errors separately** from other mistakes, and retrain on a schedule.")
        qa_block(QA["Bonus"][:1], "Lab Bonus 1")

    # ================================================================ limits
    with tabs[1]:
        thought("Good results on a course dataset are not permission to price real people. These are the limits I would "
                "put in front of a board before deployment.")
        st.markdown(
            "**1. The tiers are defined by charge bands.** In this data, each risk tier covers a distinct band of annual "
            "charge. So the tier label already 'knows' something about the answer. In a real business the tier "
            "would have to come from information available *before* claims, or the engine risks circular logic.\n\n"
            "**2. Prices move in steps.** Crossing a tier boundary switches to a different regression, so two nearly "
            "identical customers can be quoted very differently (see the Price Lab). A smoother blend of tiers "
            "would avoid the cliff.\n\n"
            "**3. Limited range.** The models were trained on ages 18-64 and BMI 16-44. Outside that, quotes are "
            "extrapolations.\n\n"
            "**4. Straight-line assumption.** Charges are right-skewed and each tier is priced with a linear model, "
            "so the most extreme bills will be under-predicted.\n\n"
            "**5. Fairness and regulation.** Sex and region are used as rating factors here because the lab includes "
            "them. Many jurisdictions restrict or forbid some rating factors (sex is banned in EU insurance "
            "pricing, for example), so legal review comes before any real deployment. Region has almost no effect "
            "in this data, which makes it an easy factor to remove.\n\n"
            "**6. Small data.** About 1,200 customers, and each tier has only a few hundred. Retrain regularly "
            "and re-check accuracy as the customer base changes.")
        qa_block(QA["Bonus"][1:], "Lab Bonus 2: when would you use PCA?")

    # ================================================================ model card
    with tabs[2]:
        thought("A model card records exactly what was built so that anyone can reproduce or audit it.")
        left, right = st.columns(2)
        with left:
            st.markdown(
                "**Data**  \n"
                f"- {R['inspect']['n_rows_raw']:,} rows loaded, {R['cleaning']['n_rows_clean']:,} after removing "
                f"{R['cleaning']['duplicates_removed']} duplicates  \n"
                f"- Missing bmi and exercise_freq filled with medians ({meta['n_rows']:,} customers used)  \n"
                "- Random seed 42, 80/20 train-test splits\n\n"
                "**Inputs used by the engine**  \n"
                "- Numeric: age, bmi, children, exercise_freq  \n"
                "- Categorical: sex, smoker, region\n\n"
                "**Dropped**  \n"
                "- Irrelevant: lucky_number, favorite_color, zodiac_sign, preferred_contact, marketing_opt_in  \n"
                "- Redundant: date_of_birth (age), weight_kg (bmi)  \n"
                "- Identifier: customer_id")
        with right:
            st.markdown(
                "**Models**  \n"
                "- Stage 1: Decision Tree classifier, depth 5  \n"
                "- Stage 2: one Linear Regression per tier  \n"
                "- Benchmark: one global Linear Regression\n\n"
                "**Software versions (must match when deploying)**  \n"
                f"- scikit-learn {meta['sklearn']}  \n- pandas {meta['pandas']}  \n- numpy {meta['numpy']}")
        st.markdown("**Files the dashboard reads**")
        st.code("artifacts/healthwise_model.joblib   # trained models (lab task E1)\n"
                "artifacts/report.json              # every number and chart input on these pages\n"
                "train_model.py                     # rebuilds both from healthwise.csv", language="text")
        qa_block(QA["E1"], "Lab reflection E1")

    r_hm = next((r for r in risk["routes"] if r["actual"] == "High" and r["predicted"] == "Medium"), None)
    key_insights([
        f"**A wrong tier is expensive:** the typical quote error jumps from {money(risk['mae_correct'])} to "
        f"{money(risk['mae_wrong'])} when the customer is sent to the wrong tier, and "
        f"{risk['share_misrouted']:.1%} of unseen customers were.",
        (f"**The costly error is High-risk customers priced as Medium:** in the test, {r_hm['n']} of them were quoted "
         f"about {money(abs(r_hm['mean_error']))} too low on average." if r_hm else
         "**No High-risk customers were priced as Medium in the test set.**"),
        f"**Cheap safeguard:** low-confidence calls ({tree['confidence']['share_low_conf']:.0%} of customers) are only "
        f"{tree['confidence']['acc_low_conf']:.0%} accurate, so review them by hand.",
        "**Before real use:** check that tiers do not depend on outcomes, smooth the tier-boundary cliff, keep inputs "
        "inside the training range, and get legal review of the rating factors.",
    ])
