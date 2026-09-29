"""
===============================================================================
 FILE : app.py            <- STREAMLIT ENTRY POINT (set this as "Main file path")
 ROLE : Builds the navigation and runs whichever page the visitor picks.

 Pages (each lives in views/):
   Boardroom Brief  - executive summary and board memo
   Data Detective   - Part A: cleaning, EDA, feature audit, tests, VIF, Lasso
   Risk Sorter      - Part B: Stage 1 classifier
   Price Engine     - Part C: Stage 2 regression per tier
   Price Lab        - live simulator: change a factor, watch the price
   Batch Scorer     - Part E3: score a CSV and apply what-if changes
   Guardrails       - risks, limits, model card, bonus questions
===============================================================================
"""
import streamlit as st

st.set_page_config(page_title="HealthWise Smart Premium Engine", page_icon=":material/health_and_safety:",
                   layout="wide", initial_sidebar_state="expanded")

from utils.engine import load_report  # noqa: E402  (after set_page_config on purpose)
from views import (batch_scorer, boardroom_brief, data_detective, guardrails,  # noqa: E402
                   price_engine, price_lab, risk_sorter)

pages = {
    "Overview": [
        st.Page(boardroom_brief.render, title="Boardroom Brief", icon=":material/summarize:",
                url_path="boardroom-brief", default=True),
    ],
    "The analysis": [
        st.Page(data_detective.render, title="Data Detective", icon=":material/search:", url_path="data-detective"),
        st.Page(risk_sorter.render, title="Risk Sorter", icon=":material/category:", url_path="risk-sorter"),
        st.Page(price_engine.render, title="Price Engine", icon=":material/payments:", url_path="price-engine"),
    ],
    "Try it": [
        st.Page(price_lab.render, title="Price Lab", icon=":material/science:", url_path="price-lab"),
        st.Page(batch_scorer.render, title="Batch Scorer", icon=":material/upload_file:", url_path="batch-scorer"),
    ],
    "Reference": [
        st.Page(guardrails.render, title="Guardrails", icon=":material/shield:", url_path="guardrails"),
    ],
}

nav = st.navigation(pages)

with st.sidebar:
    meta = load_report()["meta"]
    st.caption("HealthWise Insurance - Smart Premium Engine")
    st.caption(f"Machine Learning lab dashboard. Trained on {meta['n_rows']:,} customers with scikit-learn "
               f"{meta['sklearn']}. Educational project: not for pricing real policies.")

nav.run()
