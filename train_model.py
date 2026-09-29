"""
===============================================================================
 FILE   : train_model.py
 ROLE   : BUILD STEP (run once, on your own computer or in Colab - NOT on Streamlit)
 INPUT  : healthwise.csv  (the course dataset - keep it OUT of GitHub)
 OUTPUT : artifacts/healthwise_model.joblib   <- the trained two-stage engine
          artifacts/report.json               <- every number/chart-input the dashboard shows
-------------------------------------------------------------------------------
 Everything here mirrors HealthWise_Lab_Joan (same cleaning, same drops, same
 RANDOM_STATE = 42, same models) so the dashboard reproduces your lab results.
 The dashboard itself never touches the raw CSV - it only reads the two files above,
 which contain trained models and aggregate statistics (no customer-level rows).

 USAGE :  pip install -r requirements-train.txt
          python train_model.py --csv healthwise.csv
===============================================================================
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TIERS = ["Low", "Medium", "High"]
NUM = ["age", "bmi", "children", "exercise_freq"]
CAT = ["sex", "smoker", "region"]
FEATURES = NUM + CAT
JUNK = ["lucky_number", "favorite_color", "zodiac_sign", "marketing_opt_in", "preferred_contact"]
CATEGORICAL_ALL = ["sex", "smoker", "region", "favorite_color", "zodiac_sign",
                   "preferred_contact", "marketing_opt_in"]


# ----------------------------------------------------------------------------- helpers
def jsonable(o):
    """Convert numpy / pandas objects into plain python so json.dump works."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (pd.Series,)):
        return jsonable(o.to_dict())
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def make_pre():
    """A FRESH encoder for every pipeline (avoids sharing one fitted object across models)."""
    return ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT),
    ])


def box_stats(values):
    """Five-number summary + mean, ready for a plotly Box trace (no raw rows needed)."""
    v = np.asarray(values, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo = v[v >= q1 - 1.5 * iqr].min()
    hi = v[v <= q3 + 1.5 * iqr].max()
    return dict(q1=q1, median=med, q3=q3, lowerfence=lo, upperfence=hi,
                mean=v.mean(), min=v.min(), max=v.max(), n=len(v))


def metrics(y_true, y_pred):
    return dict(MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                R2=r2_score(y_true, y_pred))


def cls_block(y_true, y_pred, labels=TIERS):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    return dict(labels=labels, accuracy=accuracy_score(y_true, y_pred),
                confusion=cm.tolist(), report=rep)


# ----------------------------------------------------------------------------- main
def main(csv_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(csv_path)
    R = {}   # the report

    # ======================================================= A1  LOAD & INSPECT
    col_types = {}
    for c in raw.columns:
        if c in ("customer_id",):
            kind = "identifier"
        elif c == "date_of_birth":
            kind = "date"
        elif pd.api.types.is_numeric_dtype(raw[c]):
            kind = "numeric"
        else:
            kind = "categorical"
        col_types[c] = dict(kind=kind, n_unique=int(raw[c].nunique()),
                            missing=int(raw[c].isna().sum()))
    R["inspect"] = dict(n_rows_raw=len(raw), n_cols=raw.shape[1], columns=col_types)

    # ======================================================= A2  CLEAN
    n_dup = int(raw.duplicated().sum())
    dup_ids = raw.loc[raw.duplicated(keep=False), "customer_id"].unique().tolist()
    df = raw.drop_duplicates().copy()
    missing = df.isna().sum()
    missing = {k: int(v) for k, v in missing.items() if v > 0}
    bmi_stats = dict(mean=df.bmi.mean(), median=df.bmi.median(), skew=df.bmi.skew())
    ex_stats = dict(mean=df.exercise_freq.mean(), median=df.exercise_freq.median())
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df["exercise_freq"] = df["exercise_freq"].fillna(df["exercise_freq"].median())
    R["cleaning"] = dict(duplicates_removed=n_dup, duplicate_ids=dup_ids, n_rows_clean=len(df),
                         missing_before=missing, remaining_missing=int(df.isna().sum().sum()),
                         bmi=bmi_stats, exercise=ex_stats, unique_ids=int(df.customer_id.nunique()))

    # ======================================================= A3  EDA
    y = df["annual_charge"]
    hist_counts, hist_edges = np.histogram(y, bins=40)
    smoker_box = {s: box_stats(df.loc[df.smoker == s, "annual_charge"]) for s in ["no", "yes"]}
    tier_box = {t: box_stats(df.loc[df.risk_tier == t, "annual_charge"]) for t in TIERS}
    tier_counts = df.risk_tier.value_counts().reindex(TIERS)
    crosstab = pd.crosstab(df.smoker, df.risk_tier).reindex(columns=TIERS)
    R["eda"] = dict(
        skew=y.skew(), mean=y.mean(), median=y.median(), std=y.std(), min=y.min(), max=y.max(),
        hist=dict(counts=hist_counts, edges=hist_edges),
        smoker_box=smoker_box, tier_box=tier_box,
        tier_counts=tier_counts, tier_share=(tier_counts / len(df)),
        smoker_tier_crosstab=crosstab.to_dict(orient="index"),
        smoker_groups=df.groupby("smoker").annual_charge.agg(["mean", "median", "count"]).to_dict(orient="index"),
    )

    # ======================================================= A4  FEATURE AUDIT
    num_cols = ["age", "bmi", "weight_kg", "children", "exercise_freq", "lucky_number"]
    corr_target = df[num_cols + ["annual_charge"]].corr()["annual_charge"].drop("annual_charge")
    cat_audit = {}
    for c in CATEGORICAL_ALL:
        grp = df.groupby(c)["annual_charge"]
        means, counts = grp.mean(), grp.count()
        spread = (means.max() - means.min()) / df["annual_charge"].mean()
        p_anova = stats.f_oneway(*[g.values for _, g in grp])[1]
        cat_audit[c] = dict(spread_pct=spread * 100, n_categories=int(df[c].nunique()),
                            min_group_n=int(counts.min()), anova_p=p_anova,
                            group_means=means.to_dict(), group_counts=counts.to_dict())
    verdicts = {
        "customer_id": ("Dropped - identifier", "Unique label, no predictive meaning"),
        "age": ("Kept - useful", f"r = {corr_target['age']:.2f} with charge"),
        "sex": ("Kept - useful", f"ANOVA p = {cat_audit['sex']['anova_p']:.3f}"),
        "bmi": ("Kept - useful", f"r = {corr_target['bmi']:.2f} with charge"),
        "weight_kg": ("Dropped - redundant", "Duplicates bmi (r = 0.99)"),
        "children": ("Kept - useful", "Small but real family-size effect"),
        "smoker": ("Kept - useful", "Largest single driver of cost"),
        "region": ("Kept - useful", "Plausible factor; weak effect (kept for the lab's Stage-2 design)"),
        "exercise_freq": ("Kept - useful", f"r = {corr_target['exercise_freq']:.2f} with charge"),
        "date_of_birth": ("Dropped - redundant", "Same information as age (r = 1.00)"),
        "favorite_color": ("Dropped - irrelevant", f"ANOVA p = {cat_audit['favorite_color']['anova_p']:.2f}"),
        "zodiac_sign": ("Dropped - irrelevant", f"12 tiny groups; ANOVA p = {cat_audit['zodiac_sign']['anova_p']:.2f}"),
        "lucky_number": ("Dropped - irrelevant", f"r = {corr_target['lucky_number']:.3f}"),
        "preferred_contact": ("Dropped - irrelevant", f"ANOVA p = {cat_audit['preferred_contact']['anova_p']:.2f}"),
        "marketing_opt_in": ("Dropped - irrelevant", f"ANOVA p = {cat_audit['marketing_opt_in']['anova_p']:.2f}"),
        "annual_charge": ("Target - Stage 2", "What the regression predicts"),
        "risk_tier": ("Target - Stage 1", "What the classifier predicts"),
    }
    R["audit"] = dict(corr_target=corr_target, categorical=cat_audit,
                      verdicts={k: dict(verdict=v[0], evidence=v[1]) for k, v in verdicts.items()})

    # ======================================================= A5  HYPOTHESIS TESTS
    s = df.loc[df.smoker == "yes", "annual_charge"]
    n_ = df.loc[df.smoker == "no", "annual_charge"]
    t_stat, p_val = stats.ttest_ind(s, n_, equal_var=False)
    r_age, p_age = stats.pearsonr(df["age"], df["annual_charge"])
    R["tests"] = dict(n_smokers=len(s), n_nonsmokers=len(n_), mean_smokers=s.mean(),
                      mean_nonsmokers=n_.mean(), gap=s.mean() - n_.mean(), ratio=s.mean() / n_.mean(),
                      t=t_stat, p=p_val, age_r=r_age, age_p=p_age)

    # ======================================================= A6  REDUNDANCY / VIF
    age_dob = 2026 - pd.to_datetime(df["date_of_birth"]).dt.year
    num_feats = ["age", "bmi", "weight_kg", "children", "exercise_freq"]
    Xv = df[num_feats].dropna()
    vif_before = {f: variance_inflation_factor(Xv.values, i) for i, f in enumerate(num_feats)}
    kept = ["age", "bmi", "children", "exercise_freq"]
    Xv2 = df[kept].dropna()
    vif_after = {f: variance_inflation_factor(Xv2.values, i) for i, f in enumerate(kept)}
    R["redundancy"] = dict(age_dob_corr=df["age"].corr(age_dob), bmi_weight_corr=df["bmi"].corr(df["weight_kg"]),
                           vif_before=vif_before, vif_after=vif_after,
                           corr_matrix=dict(labels=num_feats, values=df[num_feats].corr().values))

    # ---- the modelling frame used from here on (same as the lab's df_model)
    drop_cols = ["customer_id"] + JUNK + ["weight_kg", "date_of_birth"]
    df_model = pd.get_dummies(df.drop(columns=drop_cols), columns=CAT, drop_first=True)
    R["modelling_frame"] = dict(columns=list(df_model.columns), dropped=drop_cols)

    # ======================================================= A8  LASSO AUDIT
    audit = df.drop(columns=["customer_id", "date_of_birth", "annual_charge", "risk_tier"])
    audit = pd.get_dummies(audit, columns=CATEGORICAL_ALL, drop_first=True)
    feat_names = list(audit.columns)
    Xa, ya = audit.values.astype(float), df["annual_charge"].values
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.2, random_state=RANDOM_STATE)
    sc = StandardScaler().fit(Xtr)
    junk_prefixes = tuple(JUNK)
    lasso_rows = []
    for a in [10, 100, 400, 800]:
        l = Lasso(alpha=a, max_iter=50000).fit(sc.transform(Xtr), ytr)
        kept_f = [f for f, c in zip(feat_names, l.coef_) if abs(c) > 1]
        lasso_rows.append(dict(alpha=a, r2=r2_score(yte, l.predict(sc.transform(Xte))),
                               n_kept=len(kept_f), kept=kept_f,
                               junk_left=[f for f in kept_f if f.startswith(junk_prefixes)]))
    R["lasso_audit"] = dict(n_features_total=len(feat_names), rows=lasso_rows)

    # ======================================================= B  STAGE 1 CLASSIFIER
    # -- B1/B2/B3 exactly as in the lab: Logistic Regression on the dummy-encoded frame
    Xc = df_model.drop(columns=["annual_charge", "risk_tier"])
    yc = df_model["risk_tier"]
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=RANDOM_STATE, stratify=yc)
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    pred = lr.predict(Xte)
    logreg = cls_block(yte, pred)
    logreg.update(train_accuracy=accuracy_score(ytr, lr.predict(Xtr)),
                  test_accuracy=accuracy_score(yte, pred),
                  majority_baseline=float((yte == yte.value_counts().idxmax()).mean()),
                  test_distribution=yte.value_counts().reindex(TIERS).to_dict(),
                  train_distribution=ytr.value_counts().reindex(TIERS).to_dict())
    # -- the classifier that is actually DEPLOYED (lab task E1): Decision Tree, depth 5.
    #    Evaluate it on the SAME held-out split so the app can report its honest accuracy.
    Xf, yf = df[FEATURES], df["risk_tier"]
    Xtr_f, Xte_f, ytr_f, yte_f = train_test_split(Xf, yf, test_size=0.2, random_state=RANDOM_STATE, stratify=yf)
    tree_eval = Pipeline([("pre", make_pre()), ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))])
    tree_eval.fit(Xtr_f, ytr_f)
    tree_pred = tree_eval.predict(Xte_f)
    tree = cls_block(yte_f, tree_pred)
    tree.update(train_accuracy=accuracy_score(ytr_f, tree_eval.predict(Xtr_f)),
                test_accuracy=accuracy_score(yte_f, tree_pred))
    # confidence safeguard evidence: how accurate are low-confidence calls?
    proba = tree_eval.predict_proba(Xte_f)
    conf = proba.max(axis=1)
    correct = (tree_pred == yte_f.values)
    tree["confidence"] = {
        "threshold": 0.7,
        "share_low_conf": float((conf < 0.7).mean()),
        "acc_low_conf": float(correct[conf < 0.7].mean()) if (conf < 0.7).any() else None,
        "acc_high_conf": float(correct[conf >= 0.7].mean()),
        "n_low_conf": int((conf < 0.7).sum()),
    }

    # ======================================================= C  STAGE 2 REGRESSION
    # -- C1 global baseline (lab)
    Xr = df_model.drop(columns=["annual_charge", "risk_tier"])
    yr = df_model["annual_charge"]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=RANDOM_STATE)
    g = LinearRegression().fit(Xtr, ytr)
    gp = g.predict(Xte)
    c1 = metrics(yte, gp)
    tier_of_test = df_model.loc[Xte.index, "risk_tier"]
    err = pd.DataFrame({"tier": tier_of_test, "true": yte, "pred": gp})
    err["abs"] = (err.true - err.pred).abs()
    err["signed"] = err.pred - err.true
    by_tier = err.groupby("tier").agg(n=("true", "size"), mean_charge=("true", "mean"),
                                      MAE=("abs", "mean"), mean_bias=("signed", "mean")).reindex(TIERS)
    by_tier["MAE_pct"] = by_tier.MAE / by_tier.mean_charge * 100
    R["regression_global"] = dict(**c1, mean_charge_test=float(yte.mean()),
                                  by_tier=by_tier.to_dict(orient="index"),
                                  n_negative_preds=int((gp < 0).sum()), min_pred=float(gp.min()))

    # -- C2 simple vs multiple (single = strongest correlated feature inside the tier)
    c2 = {}
    for t in TIERS:
        d = df_model[df_model.risk_tier == t].drop(columns=["risk_tier"])
        corr_t = d.astype(float).corr()["annual_charge"].drop("annual_charge")
        single = corr_t.abs().idxmax()
        Xd, yd = d.drop(columns=["annual_charge"]), d["annual_charge"]
        Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, random_state=RANDOM_STATE)
        simple = LinearRegression().fit(Xtr[[single]], ytr)
        multi = LinearRegression().fit(Xtr, ytr)
        r2s, r2m = r2_score(yte, simple.predict(Xte[[single]])), r2_score(yte, multi.predict(Xte))
        c2[t] = dict(single=single, single_corr=corr_t[single], r2_simple=r2s, r2_multi=r2m,
                     jump=r2m - r2s, corr=corr_t.sort_values(key=abs, ascending=False).to_dict(), n=len(d))

    # -- C3 regression inside each tier
    c3, ot, op = {}, [], []
    for t in TIERS:
        d = df_model[df_model.risk_tier == t].drop(columns=["risk_tier"])
        Xd, yd = d.drop(columns=["annual_charge"]), d["annual_charge"]
        Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, random_state=RANDOM_STATE)
        m = LinearRegression().fit(Xtr, ytr)
        p = m.predict(Xte)
        ot.extend(yte); op.extend(p)
        c3[t] = dict(R2=r2_score(yte, p), MAE=mean_absolute_error(yte, p), n=len(d),
                     charge_std=float(yd.std()), charge_mean=float(yd.mean()))
    c3_all = dict(MAE=mean_absolute_error(ot, op), R2=r2_score(ot, op))

    # -- C4 ridge / lasso sweep in every tier + lasso coefficients (standardised)
    alphas = [0.01, 0.1, 1, 10, 50, 100, 200]
    c4 = {}
    for t in TIERS:
        d = df_model[df_model.risk_tier == t].drop(columns=["risk_tier"])
        Xd, yd = d.drop(columns=["annual_charge"]), d["annual_charge"]
        Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, random_state=RANDOM_STATE)
        scl = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scl.transform(Xtr), scl.transform(Xte)
        rows = []
        for a in alphas:
            rr = Ridge(alpha=a).fit(Xtr_s, ytr)
            ll = Lasso(alpha=a, max_iter=10000).fit(Xtr_s, ytr)
            rows.append(dict(alpha=a, ridge=r2_score(yte, rr.predict(Xte_s)), lasso=r2_score(yte, ll.predict(Xte_s))))
        best_alpha = 10
        lasso = Lasso(alpha=best_alpha, max_iter=10000).fit(Xtr_s, ytr)
        coefs = pd.Series(lasso.coef_, index=Xd.columns).sort_values(key=abs, ascending=False)
        ols_r2 = r2_score(yte, LinearRegression().fit(Xtr, ytr).predict(Xte))
        c4[t] = dict(sweep=rows, best_alpha=best_alpha, lasso_coefs=coefs.to_dict(), ols_r2=ols_r2,
                     ridge_tuned=r2_score(yte, Ridge(alpha=1).fit(Xtr_s, ytr).predict(Xte_s)),
                     lasso_tuned=r2_score(yte, lasso.predict(Xte_s)))

    # ======================================================= END-TO-END (honest) EVALUATION
    # Same 80/20 stratified split for every model; Stage 2 trained on TRAIN rows only.
    Xtr_f, Xte_f, ytr_f, yte_f, ttr, tte = train_test_split(
        df[FEATURES], df["annual_charge"], df["risk_tier"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["risk_tier"])
    glob = Pipeline([("pre", make_pre()), ("model", LinearRegression())]).fit(Xtr_f, ytr_f)
    regs_tr = {t: Pipeline([("pre", make_pre()), ("model", LinearRegression())]).fit(Xtr_f[ttr == t], ytr_f[ttr == t])
               for t in TIERS}
    tree_r = Pipeline([("pre", make_pre()), ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))]).fit(Xtr_f, ttr)
    lr_r = Pipeline([("pre", make_pre()), ("model", LogisticRegression(max_iter=1000))]).fit(Xtr_f, ttr)

    def route(tiers_pred):
        out_ = np.zeros(len(Xte_f))
        for t in TIERS:
            m_ = np.asarray(tiers_pred) == t
            if m_.any():
                out_[m_] = regs_tr[t].predict(Xte_f[m_])
        return out_

    e2e = {
        "Global flat model": metrics(yte_f, glob.predict(Xte_f)),
        "Two-stage, perfect tier (lab C3 style)": metrics(yte_f, route(tte.values)),
        "Two-stage, Decision-Tree router (deployed)": metrics(yte_f, route(tree_r.predict(Xte_f))),
        "Two-stage, Logistic router": metrics(yte_f, route(lr_r.predict(Xte_f))),
    }

    # Bonus 1 - error propagation: what does a WRONG tier do to the quote?
    tp = tree_r.predict(Xte_f)
    quote = route(tp)
    mis = pd.DataFrame({"actual": tte.values, "pred": tp, "true": yte_f.values, "quote": quote})
    mis["err"] = mis.quote - mis.true
    misrows = []
    for a in TIERS:
        for p_ in TIERS:
            sub = mis[(mis.actual == a) & (mis.pred == p_)]
            if len(sub):
                misrows.append(dict(actual=a, predicted=p_, n=len(sub), mean_true=sub.true.mean(),
                                    mean_quote=sub.quote.mean(), mean_error=sub.err.mean(),
                                    mean_abs_error=sub.err.abs().mean()))
    R["risk"] = dict(routes=misrows, share_misrouted=float((mis.actual != mis.pred).mean()),
                     mae_correct=float(mis[mis.actual == mis.pred].err.abs().mean()),
                     mae_wrong=float(mis[mis.actual != mis.pred].err.abs().mean()),
                     n_test=len(mis))

    # ======================================================= E1  FINAL DEPLOYED MODELS (full data)
    med = {c: float(df[c].median()) for c in NUM}
    clf = Pipeline([("pre", make_pre()), ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))])
    clf.fit(df[FEATURES], df["risk_tier"])
    regs = {}
    for t in TIERS:
        sub = df[df["risk_tier"] == t]
        regs[t] = Pipeline([("pre", make_pre()), ("model", LinearRegression())]).fit(sub[FEATURES], sub["annual_charge"])
    global_model = Pipeline([("pre", make_pre()), ("model", LinearRegression())]).fit(df[FEATURES], df["annual_charge"])

    names = list(clf.named_steps["pre"].get_feature_names_out())
    importances = dict(zip([n.split("__", 1)[1] for n in names], clf.named_steps["model"].feature_importances_))
    tier_coefs = {}
    for t in TIERS:
        m = regs[t].named_steps["model"]
        tier_coefs[t] = dict(intercept=float(m.intercept_),
                             **{n.split("__", 1)[1]: float(c) for n, c in zip(names, m.coef_)})
    tier_ranges = {t: box_stats(df.loc[df.risk_tier == t, "annual_charge"]) for t in TIERS}

    joblib.dump({"clf": clf, "regs": regs, "global": global_model, "features": FEATURES,
                 "num": NUM, "cat": CAT, "med": med,
                 "meta": {"sklearn": sklearn.__version__, "random_state": RANDOM_STATE}},
                out / "healthwise_model.joblib")

    R["classification"] = dict(logreg=logreg, tree=tree,
                               deployed_train_accuracy=accuracy_score(df["risk_tier"], clf.predict(df[FEATURES])),
                               tree_importances=importances)
    R["regression"] = dict(c1=c1, c2=c2, c3=c3, c3_all=c3_all, c4=c4, end_to_end=e2e,
                           tier_coefs=tier_coefs, ladder_tier="High")
    R["simulator"] = dict(
        med=med, tier_ranges=tier_ranges,
        ranges={c: dict(min=float(df[c].min()), max=float(df[c].max())) for c in NUM},
        mean_charge=float(df.annual_charge.mean()),
        categories={c: sorted(df[c].unique().tolist()) for c in CAT})
    R["meta"] = dict(sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__,
                     random_state=RANDOM_STATE, n_rows=len(df))

    (out / "report.json").write_text(json.dumps(jsonable(R), indent=1))
    print(f"Saved {out/'healthwise_model.joblib'}  (scikit-learn {sklearn.__version__})")
    print(f"Saved {out/'report.json'}")
    print("Stage-1 (LogReg) test acc :", round(logreg['test_accuracy'], 3))
    print("Stage-1 (Tree)   test acc :", round(tree['test_accuracy'], 3), "| deployed train acc:",
          round(R['classification']['deployed_train_accuracy'], 3))
    print("Global   :", {k: round(v, 3) for k, v in c1.items()})
    print("C3 all   :", {k: round(v, 3) for k, v in c3_all.items()})
    for k, v in e2e.items():
        print(f"E2E {k:45} MAE={v['MAE']:.0f} R2={v['R2']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="healthwise.csv")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    main(a.csv, a.out)
