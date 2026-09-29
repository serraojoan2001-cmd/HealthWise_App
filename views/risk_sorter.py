"""
===============================================================================
 PAGE 3 : RISK SORTER   (Stage 1 - classify the customer; lab Part B)
 Tabs follow the lab tasks: B1 build | B2 accuracy and mistakes | B3 overfit check | what it looks at
 Two models appear side by side:
   * Logistic Regression  -> the model evaluated in lab tasks B1-B3
   * Decision Tree        -> the model actually DEPLOYED in the app (lab task E1)
===============================================================================
"""
import pandas as pd
import streamlit as st

from content.qa import QA
from utils import charts
from utils.engine import load_report
from utils.theme import key_insights, page_header, qa_block, thought

TIERS = ["Low", "Medium", "High"]
LOGREG = "Logistic Regression (lab Part B)"
TREE = "Decision Tree (deployed in the app)"


def _report_table(block):
    rep = block["report"]
    return pd.DataFrame(
        [{"Tier": t, "Precision": rep[t]["precision"], "Recall": rep[t]["recall"],
          "F1": rep[t]["f1-score"], "Customers": int(rep[t]["support"])} for t in TIERS]
    ).style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"})


def render():
    R = load_report()
    cls = R["classification"]
    lr, tree = cls["logreg"], cls["tree"]
    page_header(
        "Risk Sorter",
        "Stage 1 of the engine: given a customer's profile, which risk tier do they belong to? "
        "A wrong answer here sends the customer to the wrong price list in Stage 2.")

    c = st.columns(4)
    c[0].metric("Logistic Regression", f"{lr['test_accuracy']:.1%}", help="Accuracy on unseen customers (lab B2).")
    c[1].metric("Decision Tree (deployed)", f"{tree['test_accuracy']:.1%}",
                f"+{(tree['test_accuracy'] - lr['test_accuracy']) * 100:.1f} pts")
    c[2].metric("Always guess 'Medium'", f"{lr['majority_baseline']:.1%}", help="The bar any model must clear.")
    c[3].metric("Tree mistakes", f"{1 - tree['test_accuracy']:.1%}", help="Share of unseen customers put in the wrong tier.")

    tabs = st.tabs(["1. Build the classifier", "2. Accuracy and mistakes", "3. Overfit check", "4. What it looks at"])

    # ================================================================ B1
    with tabs[0]:
        thought(
            "I wanted a classifier that is easy to explain and gives probabilities, so I started with Logistic "
            "Regression. The tiers are imbalanced (Medium is the majority), so I split the data with "
            "<code>stratify</code>, which keeps the same tier mix in the training and test sets. The lab's "
            "deployment step (E1) uses a Decision Tree, so I evaluated that one on exactly the same split to see "
            "which is better.")
        td, tr = lr["test_distribution"], lr["train_distribution"]
        df = pd.DataFrame({"Training customers": tr, "Test customers": td}).loc[TIERS]
        df["Test share"] = df["Test customers"] / df["Test customers"].sum()
        st.markdown("**Stratified split: the tier mix is preserved**")
        st.dataframe(df.style.format({"Test share": "{:.1%}"}))
        qa_block(QA["B1"], "Lab reflection B1")

    # ================================================================ B2
    with tabs[1]:
        thought(
            "Accuracy is a single number; the confusion matrix shows where the mistakes are. What matters to "
            "HealthWise is not just how many mistakes, but which ones: calling a High-risk customer Low means "
            "collecting a small premium against big claims.")
        choice = st.radio("Model", [TREE, LOGREG], horizontal=True)
        block = tree if choice == TREE else lr
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(charts.confusion(block["confusion"], block["labels"], f"Confusion matrix: {choice.split(' (')[0]}"))
        with right:
            st.markdown("**Per-tier scorecard**")
            st.dataframe(_report_table(block), hide_index=True)
            cm = block["confusion"]
            st.markdown(
                f"- Overall accuracy: **{block['accuracy']:.1%}**\n"
                f"- High-risk customers sent to Low (the costly error): **{cm[2][0]}**\n"
                f"- High-risk customers sent to Medium: **{cm[2][1]}**\n"
                f"- Low-risk customers sent to High (overcharged): **{cm[0][2]}**")
        st.plotly_chart(charts.recall_bars(lr["report"], tree["report"], "Logistic Regression", "Decision Tree"))
        st.warning("**Which mistake is worse?** High labelled Low. HealthWise collects a low premium from someone who "
                   "will file expensive claims, losing money every year. Low labelled High overcharges a safe "
                   "customer, who may leave: a smaller loss.")
        qa_block(QA["B2"], "Lab reflection B2 (answers refer to Logistic Regression)")

    # ================================================================ B3
    with tabs[2]:
        thought(
            "Comparing training accuracy with test accuracy tells me whether the model memorised the training data "
            "(big gap: overfitting) or is too simple (both low: underfitting).")
        df = pd.DataFrame({
            "Model": ["Logistic Regression", "Decision Tree (depth 5)"],
            "Training accuracy": [lr["train_accuracy"], tree["train_accuracy"]],
            "Test accuracy": [lr["test_accuracy"], tree["test_accuracy"]]})
        df["Gap (train minus test)"] = df["Training accuracy"] - df["Test accuracy"]
        st.dataframe(df.style.format({"Training accuracy": "{:.3f}", "Test accuracy": "{:.3f}",
                                      "Gap (train minus test)": "{:+.3f}"}), hide_index=True)
        st.success("Neither model overfits: both gaps are tiny and test accuracy is not below training accuracy. "
                   "Logistic Regression (about 80%) is a little too simple for the tier boundaries; the depth-5 "
                   "tree captures them better.")
        st.caption("The Decision Tree is deliberately limited to depth 5. An unlimited tree would memorise the "
                   "training customers; the depth limit is what keeps its test accuracy honest.")
        qa_block(QA["B3"], "Lab reflection B3")

    # ================================================================ importance + safeguard
    with tabs[3]:
        thought(
            "Because a tree makes explicit yes/no splits, I can ask which features it relies on. I also checked "
            "how sure it is of each call, because a borderline customer is the right place to add a human check.")
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(charts.importance_bars(cls["tree_importances"]))
        with right:
            conf = tree["confidence"]
            st.markdown(
                f"**Confidence check.** On unseen customers, calls where the tree was at least {conf['threshold']:.0%} "
                f"sure were right **{conf['acc_high_conf']:.0%}** of the time. The {conf['share_low_conf']:.1%} of "
                f"customers below that line ({conf['n_low_conf']} people) were right only "
                f"**{conf['acc_low_conf']:.0%}** of the time.\n\n"
                "So a simple safeguard is to send low-confidence customers to manual review. The Price Lab shows this "
                "warning live.")
        st.caption(f"Training accuracy of the final deployed tree (fit on all customers): "
                   f"{cls['deployed_train_accuracy']:.1%}. The honest figure is the held-out one above.")

    key_insights([
        f"**The deployed Decision Tree is the better sorter:** {tree['test_accuracy']:.1%} on unseen customers vs "
        f"{lr['test_accuracy']:.1%} for Logistic Regression, against a {lr['majority_baseline']:.0%} 'always Medium' baseline.",
        f"**Mistakes are almost always neighbours.** Low and High are rarely confused; errors sit at the Medium "
        f"boundary, and the costly one is High sent to Medium ({tree['confusion'][2][1]} of "
        f"{sum(tree['confusion'][2])} High-risk customers in the test set).",
        "**No overfitting:** train and test accuracy are within a few points for both models.",
        f"**BMI, age and smoker status do almost all the sorting** "
        f"({cls['tree_importances']['bmi']:.0%}, {cls['tree_importances']['age']:.0%} and "
        f"{cls['tree_importances']['smoker_yes']:.0%} of the tree's decisions).",
        f"**Add a safeguard:** the {tree['confidence']['share_low_conf']:.0%} of calls made with low confidence are "
        f"only {tree['confidence']['acc_low_conf']:.0%} accurate. Route those to a human before quoting.",
    ])
