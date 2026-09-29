"""
===============================================================================
 PAGE 1 : BOARDROOM BRIEF   (the executive summary)
 Source : lab Part D (board memo) + headline numbers from Parts B and C
===============================================================================
"""
import streamlit as st

from content.qa import QA
from utils import charts
from utils.engine import load_report
from utils.theme import key_insights, page_header, qa_block, thought

TIERS = ["Low", "Medium", "High"]


def render():
    R = load_report()
    cls, reg = R["classification"], R["regression"]
    e2e = reg["end_to_end"]
    flat_lab = R["regression_global"]
    two_lab = reg["c3_all"]
    flat_e2e = e2e["Global flat model"]
    deployed = e2e["Two-stage, Decision-Tree router (deployed)"]
    tests = R["tests"]
    tree_acc = cls["tree"]["test_accuracy"]
    cut_lab = 1 - two_lab["MAE"] / flat_lab["MAE"]
    cut_honest = 1 - deployed["MAE"] / flat_e2e["MAE"]

    page_header(
        "Boardroom Brief",
        "One flat price for everyone overcharges safe customers and undercharges risky ones. "
        "The Smart Premium Engine fixes that in two steps: sort each customer into a risk tier, "
        "then price them inside that tier.")

    # ---------------------------------------------------------------- headline numbers
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Old flat model: average error", f"${flat_lab['MAE']:,.0f}",
              help="Lab task C1. One linear formula for all customers.")
    c2.metric("Two-stage engine: average error", f"${two_lab['MAE']:,.0f}", f"-{cut_lab:.0%} vs flat",
              delta_color="inverse", help="Lab task C3. Each customer priced by the correct tier's model.")
    c3.metric("Honest test: tier chosen by the app", f"${deployed['MAE']:,.0f}", f"-{cut_honest:.0%} vs flat",
              delta_color="inverse",
              help="Same held-out customers for both models. The Decision Tree picks the tier, "
                   "so its mistakes are included.")
    c4.metric("Tier-sorter accuracy", f"{tree_acc:.1%}",
              help="Decision Tree used in the app, measured on customers it never saw in training.")
    st.caption(
        "Average error = MAE, the typical dollar gap between the quote and the customer's true annual charge. "
        "The middle figure assumes Stage 1 always finds the right tier; the right-hand figure does not, "
        "and is the one to quote to the board.")

    # ---------------------------------------------------------------- how it works
    st.subheader("How the engine works")
    a, b, c = st.columns([1, 1, 1])
    with a:
        st.markdown("**1. Customer profile**  \nAge, BMI, children, exercise, sex, smoker, region.")
    with b:
        st.markdown("**2. Stage 1: sort into a tier**  \nA Decision Tree labels the customer "
                    "Low, Medium or High risk.")
    with c:
        st.markdown("**3. Stage 2: price inside the tier**  \nThat tier's own regression turns the profile "
                    "into an annual premium.")
    thought(
        "The board's problem is that one formula prices a young non-smoker and an older heavy smoker the same way. "
        "I first proved the flat model is weak (average error about $5,700), then asked whether customers behave "
        "differently enough to justify separate models. They do: being a smoker is worth about "
        f"${reg['tier_coefs']['Low']['smoker_yes']:,.0f} in the Low tier but "
        f"${reg['tier_coefs']['High']['smoker_yes']:,.0f} in the High tier, so a single coefficient cannot be right for both.")

    # ---------------------------------------------------------------- what drives cost
    st.subheader("What drives cost")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.dollars_by_tier(reg["tier_coefs"]))
    with right:
        st.markdown(
            f"**Smoking is the biggest lever.** Smokers pay about **${tests['gap']:,.0f}** more per year than "
            f"non-smokers ({tests['ratio']:.1f} times as much), and the gap is statistically certain "
            f"(p = {tests['p']:.1e}).\n\n"
            "**Top three drivers:** smoker, BMI and age. Number of children and sex matter a little. "
            "Region hardly matters.\n\n"
            "**Worthless columns removed:** lucky number, favourite colour, zodiac sign, preferred contact and "
            "marketing opt-in, plus duplicates of information already held (weight, date of birth).")

    # ---------------------------------------------------------------- memo
    st.subheader("Memo to the HealthWise board")
    with st.container(border=True):
        st.markdown(
            f"The two-stage engine first sorts each customer into a Low, Medium or High risk tier, then predicts "
            f"their premium using a model built only for that tier. The average pricing error fell from "
            f"**${flat_lab['MAE']:,.0f} to ${two_lab['MAE']:,.0f}** and R\u00b2 rose from {flat_lab['R2']:.2f} to "
            f"{two_lab['R2']:.2f}, which clearly shows it beats the single flat model. When the engine chooses the tier "
            f"itself, the honest improvement is still **{cut_honest:.0%}** "
            f"(${flat_e2e['MAE']:,.0f} to ${deployed['MAE']:,.0f}).\n\n"
            "Cost is driven mainly by three factors: smoking status, BMI and age. Number of children and sex matter a "
            "little less, and region hardly matters. Columns with no link to cost were dropped: favourite colour, "
            "zodiac sign, lucky number, preferred contact and marketing opt-in. Columns that duplicate other data "
            "(weight, date of birth) and the customer ID were also dropped. Removing them did not reduce accuracy.\n\n"
            f"**Recommendation: deploy the engine.** Smokers pay about ${tests['gap']:,.0f} more per year than "
            f"non-smokers, so pricing everyone with one formula loses money on risky customers and overcharges "
            f"low-risk ones. The main risk is the tier-sorting step. The Decision Tree in the app is right "
            f"{tree_acc:.0%} of the time on unseen customers, and most of its mistakes label High-risk customers as "
            f"Medium, which would under-price them. Monitor those cases, send borderline classifications to manual "
            f"review, and retrain the models regularly.")

    qa_block(QA["D"], "Lab reflection D: the board questions and my answers")
    qa_block(QA["C3"][1:2], "Lab reflection C3.2: does two-stage beat the flat model?")

    key_insights([
        f"**The engine works.** Average pricing error drops from ${flat_lab['MAE']:,.0f} to ${two_lab['MAE']:,.0f} "
        f"in the lab test ({cut_lab:.0%}), and by {cut_honest:.0%} in the stricter test where the app picks the tier itself.",
        f"**Smoking is the single biggest lever.** Smokers pay about ${tests['gap']:,.0f} more a year, "
        f"and the effect grows with risk: ${reg['tier_coefs']['Low']['smoker_yes']:,.0f} in the Low tier, "
        f"${reg['tier_coefs']['High']['smoker_yes']:,.0f} in the High tier.",
        "**Three drivers explain most of the price:** smoker, BMI and age. Five noise columns and two duplicates "
        "were removed with no loss of accuracy.",
        "**Recommendation: deploy, with guardrails.** Watch High-risk customers being sorted into Medium, and "
        "route low-confidence classifications to a human (see the Guardrails page).",
    ])
