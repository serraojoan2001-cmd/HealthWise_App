"""
===============================================================================
 FILE : utils/charts.py
 ROLE : Every Plotly chart in the dashboard, built from report.json / model output.
        Pages call these functions, so all charts share one visual style.
===============================================================================
"""
import numpy as np
import plotly.graph_objects as go

from utils.theme import GREY, INK, RULE, TEAL, TIER_COLORS, TIER_SOFT

TIERS = ["Low", "Medium", "High"]


def _base(fig, height=360, title=None, legend=True):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        title=dict(text=title, font=dict(size=16, color=INK)) if title else None,
        font=dict(family="Source Sans 3, Segoe UI, sans-serif", color=INK),
        showlegend=legend, legend=dict(orientation="h", y=-0.18),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ------------------------------------------------------------------ Data Detective (Part A)
def hist_charge(hist, title="How much customers pay per year"):
    edges = np.array(hist["edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=hist["counts"], width=np.diff(edges) * 0.95, marker_color=TEAL))
    fig.update_xaxes(title="Annual charge ($)", tickprefix="$", tickformat=",")
    fig.update_yaxes(title="Customers")
    return _base(fig, 330, title, legend=False)


def box_groups(boxes: dict, order, colors=None, title="", ytitle="Annual charge ($)", labels=None):
    fig = go.Figure()
    for k in order:
        b = boxes[k]
        name = (labels or {}).get(k, k)
        fig.add_trace(go.Box(
            x=[name], q1=[b["q1"]], median=[b["median"]], q3=[b["q3"]],
            lowerfence=[b["lowerfence"]], upperfence=[b["upperfence"]], mean=[b["mean"]],
            name=name, marker_color=(colors or {}).get(k, TEAL), boxmean=True))
    fig.update_yaxes(title=ytitle, tickprefix="$", tickformat=",")
    return _base(fig, 340, title, legend=False)


def corr_bars(corr: dict, title="Correlation with annual charge"):
    items = sorted(corr.items(), key=lambda kv: abs(kv[1]))
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
        marker_color=[TEAL if abs(v) > 0.1 else RULE for _, v in items],
        text=[f"{v:+.2f}" for _, v in items], textposition="outside"))
    fig.update_xaxes(range=[-0.45, 0.55], zeroline=True, zerolinecolor=GREY)
    return _base(fig, 320, title, legend=False)


def group_means(audit_col: dict, overall_mean: float, title=""):
    """Bar per category with n shown - the 'zodiac trap' chart."""
    names = list(audit_col["group_means"].keys())
    means = [audit_col["group_means"][n] for n in names]
    counts = [audit_col["group_counts"][n] for n in names]
    order = np.argsort(means)
    names, means, counts = [names[i] for i in order], [means[i] for i in order], [counts[i] for i in order]
    fig = go.Figure(go.Bar(x=names, y=means, marker_color=RULE,
                           text=[f"n={c}" for c in counts], textposition="outside"))
    fig.add_hline(y=overall_mean, line_dash="dash", line_color=TEAL,
                  annotation_text="overall average", annotation_position="top left")
    fig.update_yaxes(title="Average charge ($)", tickprefix="$", tickformat=",", rangemode="tozero")
    return _base(fig, 340, title, legend=False)


def vif_bars(before: dict, after: dict, title="Variance inflation factor (VIF)"):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before dropping weight_kg", x=list(before), y=list(before.values()),
                         marker_color=TIER_COLORS["High"], text=[f"{v:.1f}" for v in before.values()],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="After dropping weight_kg", x=list(after), y=list(after.values()),
                         marker_color=TEAL, text=[f"{v:.1f}" for v in after.values()],
                         textposition="outside"))
    fig.add_hline(y=10, line_dash="dash", line_color=GREY, annotation_text="VIF = 10 (danger zone)")
    fig.update_layout(barmode="group")
    return _base(fig, 340, title)


def corr_heatmap(labels, values, title="Correlation between numeric features"):
    fig = go.Figure(go.Heatmap(z=values, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu_r",
                               text=np.round(values, 2), texttemplate="%{text}"))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 360, title, legend=False)


def lasso_audit_chart(rows):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"alpha = {r['alpha']}" for r in rows], y=[r["n_kept"] for r in rows],
                         name="Features kept", marker_color=RULE,
                         text=[f"R\u00b2 {r['r2']:.3f}" for r in rows], textposition="outside"))
    fig.update_yaxes(title="Features that survive (of 29)")
    return _base(fig, 320, "Lasso audit: fewer features, same accuracy", legend=False)


# ------------------------------------------------------------------ Risk Sorter (Stage 1)
def confusion(cm, labels, title):
    z = np.array(cm)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Predicted {l}" for l in labels], y=[f"Actual {l}" for l in labels],
        colorscale=[[0, "#FFFFFF"], [1, TEAL]], showscale=False,
        text=z, texttemplate="%{text}", textfont=dict(size=18)))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 330, title, legend=False)


def recall_bars(report_a: dict, report_b: dict, name_a: str, name_b: str):
    fig = go.Figure()
    for rep, name, col in [(report_a, name_a, RULE), (report_b, name_b, TEAL)]:
        fig.add_trace(go.Bar(name=name, x=TIERS, y=[rep[t]["recall"] for t in TIERS], marker_color=col,
                             text=[f"{rep[t]['recall']:.0%}" for t in TIERS], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%", title="Recall (share of the tier caught)")
    return _base(fig, 330, "Who gets caught? Recall by tier")


def importance_bars(imp: dict, title="What the deployed tier-sorter looks at"):
    nice = {"age": "Age", "bmi": "BMI", "children": "Children", "exercise_freq": "Exercise",
            "sex_male": "Sex", "smoker_yes": "Smoker", "region_north": "Region: north",
            "region_south": "Region: south", "region_west": "Region: west"}
    items = sorted(imp.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[nice.get(k, k) for k, _ in items], orientation="h",
                           marker_color=TEAL, text=[f"{v:.0%}" for _, v in items], textposition="outside"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(v for _, v in items) * 1.2])
    return _base(fig, 340, title, legend=False)


def probability_bar(proba: dict):
    fig = go.Figure()
    for t in TIERS:
        fig.add_trace(go.Bar(y=["Stage 1 confidence"], x=[proba[t]], orientation="h", name=t,
                             marker_color=TIER_COLORS[t],
                             text=[f"{t} {proba[t]:.0%}" if proba[t] >= 0.08 else ""], textposition="inside"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    return _base(fig, 130, None, legend=False)


# ------------------------------------------------------------------ Price Engine (Stage 2)
def metric_bars(models: dict, key="MAE", title="Average pricing error (MAE)"):
    names = list(models)
    vals = [models[n][key] for n in names]
    colors = [GREY if "flat" in n.lower() else TEAL for n in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           text=[f"${v:,.0f}" if key == "MAE" else f"{v:.3f}" for v in vals],
                           textposition="outside"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, max(vals) * 1.25])
    return _base(fig, 300, title, legend=False)


def global_error_by_tier(by_tier: dict):
    fig = go.Figure(go.Bar(
        x=TIERS, y=[by_tier[t]["mean_bias"] for t in TIERS],
        marker_color=[TIER_COLORS[t] for t in TIERS],
        text=[f"{by_tier[t]['mean_bias']:+,.0f}" for t in TIERS], textposition="outside"))
    fig.add_hline(y=0, line_color=GREY)
    fig.update_yaxes(title="Average quote minus true charge ($)", tickprefix="$", tickformat=",")
    return _base(fig, 320, "The flat model over-charges cheap tiers and under-charges High", legend=False)


def simple_vs_multi(c2: dict):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="One feature", x=TIERS, y=[c2[t]["r2_simple"] for t in TIERS], marker_color=RULE,
                         text=[f"{c2[t]['r2_simple']:.2f}<br>({c2[t]['single']})" for t in TIERS],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="All features", x=TIERS, y=[c2[t]["r2_multi"] for t in TIERS], marker_color=TEAL,
                         text=[f"{c2[t]['r2_multi']:.2f}" for t in TIERS], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1], title="R\u00b2 on held-out customers")
    return _base(fig, 340, "One feature vs all features, inside each tier")


def tier_fit(c3: dict):
    fig = go.Figure(go.Bar(x=TIERS, y=[c3[t]["MAE"] for t in TIERS],
                           marker_color=[TIER_COLORS[t] for t in TIERS],
                           text=[f"${c3[t]['MAE']:,.0f}<br>R\u00b2 {c3[t]['R2']:.2f}" for t in TIERS],
                           textposition="outside"))
    fig.update_yaxes(title="Average error inside the tier ($)", tickprefix="$", tickformat=",")
    fig.update_yaxes(range=[0, max(c3[t]["MAE"] for t in TIERS) * 1.3])
    return _base(fig, 320, "Which tier is easiest to price?", legend=False)


def alpha_sweep(rows, best_alpha=None):
    a = [r["alpha"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a, y=[r["ridge"] for r in rows], name="Ridge", mode="lines+markers",
                             line=dict(color=GREY)))
    fig.add_trace(go.Scatter(x=a, y=[r["lasso"] for r in rows], name="Lasso", mode="lines+markers",
                             line=dict(color=TEAL)))
    if best_alpha:
        fig.add_vline(x=best_alpha, line_dash="dot", line_color=INK,
                      annotation_text=f"chosen alpha = {best_alpha}")
    fig.update_xaxes(type="log", title="alpha (penalty strength, log scale)")
    fig.update_yaxes(title="R\u00b2 on held-out customers", range=[0, 0.9])
    return _base(fig, 340, "Too much penalty makes the model underfit")


def lasso_coefs(coefs: dict, title="Lasso: what drives cost (standardised)"):
    items = sorted(coefs.items(), key=lambda kv: abs(kv[1]))
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=[TEAL if abs(v) > 1 else RULE for _, v in items],
                           text=[f"{v:,.0f}" for _, v in items], textposition="outside"))
    fig.update_xaxes(title="Coefficient ($ per 1 standard deviation)")
    return _base(fig, 360, title, legend=False)


def dollars_by_tier(tier_coefs: dict):
    keys = [("smoker_yes", "Smoker (yes vs no)"), ("sex_male", "Sex (male vs female)"),
            ("children", "Each child"), ("bmi", "Each BMI point"), ("age", "Each year of age"),
            ("exercise_freq", "Each workout / week")]
    fig = go.Figure()
    for t in TIERS:
        fig.add_trace(go.Bar(name=t, x=[lab for _, lab in keys], y=[tier_coefs[t][k] for k, _ in keys],
                             marker_color=TIER_COLORS[t]))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Added annual premium ($)", tickprefix="$", tickformat=",")
    return _base(fig, 360, "What each factor is worth, tier by tier")


# ------------------------------------------------------------------ Price Lab (live simulator)
def premium_gauge(tier_ranges: dict, premium: float, tier: str, flat: float = None, baseline: float = None):
    """The signature visual: where this quote sits inside the tier price bands."""
    fig = go.Figure()
    for t in TIERS:
        b = tier_ranges[t]
        fig.add_shape(type="rect", x0=b["min"], x1=b["max"], y0=0.15, y1=0.85, line_width=0,
                      fillcolor=TIER_SOFT[t], layer="below")
        fig.add_annotation(x=(b["min"] + b["max"]) / 2, y=0.5, text=f"<b>{t}</b><br>${b['min']/1000:.0f}k-${b['max']/1000:.0f}k",
                           showarrow=False, font=dict(color=TIER_COLORS[t], size=12))
    if baseline is not None:
        fig.add_trace(go.Scatter(x=[baseline], y=[0.5], mode="markers", name="Pinned baseline",
                                 marker=dict(symbol="circle-open", size=17, color=INK, line=dict(width=2))))
    if flat is not None:
        fig.add_trace(go.Scatter(x=[flat], y=[0.9], mode="markers+text", name="Old flat-model quote",
                                 text=["old flat quote"], textposition="top center",
                                 marker=dict(symbol="x", size=11, color=GREY)))
    fig.add_trace(go.Scatter(x=[premium], y=[0.5], mode="markers", name="This quote",
                             marker=dict(symbol="diamond", size=20, color=TIER_COLORS[tier],
                                         line=dict(width=2, color=INK))))
    fig.update_xaxes(range=[0, 62000], tickprefix="$", tickformat=",", showgrid=False)
    fig.update_yaxes(visible=False, range=[0, 1.15])
    return _base(fig, 190, None, legend=True)


def waterfall(contrib):
    vals = contrib["dollars"].tolist()
    fig = go.Figure(go.Waterfall(
        x=contrib["factor"].tolist() + ["Premium"], measure=["absolute"] + ["relative"] * (len(vals) - 1) + ["total"],
        y=vals[:1] + vals[1:] + [0],
        text=[f"${v:,.0f}" for v in vals] + [f"${sum(vals):,.0f}"], textposition="outside",
        increasing=dict(marker_color=TIER_COLORS["High"]), decreasing=dict(marker_color=TIER_COLORS["Low"]),
        totals=dict(marker_color=TEAL), connector=dict(line=dict(color=RULE))))
    fig.update_yaxes(title="Annual premium ($)", tickprefix="$", tickformat=",")
    return _base(fig, 380, "Where this premium comes from (inside the customer's tier)", legend=False)


def sweep_line(df, factor, label, current, tier_ranges=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[factor], y=df["premium"], mode="lines", line=dict(color=RULE, width=2),
                             showlegend=False, hoverinfo="skip"))
    for t in TIERS:
        d = df[df.tier == t]
        if len(d):
            fig.add_trace(go.Scatter(x=d[factor], y=d["premium"], mode="markers", name=f"{t} tier",
                                     marker=dict(color=TIER_COLORS[t], size=8)))
    fig.add_vline(x=current, line_dash="dot", line_color=INK, annotation_text="current")
    fig.update_xaxes(title=label)
    fig.update_yaxes(title="Premium ($)", tickprefix="$", tickformat=",")
    return _base(fig, 360, f"Premium as {label} changes (everything else held fixed)")


def sweep_bars(df, factor, label, current):
    fig = go.Figure(go.Bar(
        x=[str(v) for v in df[factor]], y=df["premium"],
        marker_color=[TIER_COLORS[t] for t in df["tier"]],
        text=[f"${p:,.0f}<br>{t}" for p, t in zip(df["premium"], df["tier"])], textposition="outside"))
    fig.update_yaxes(title="Premium ($)", tickprefix="$", tickformat=",", rangemode="tozero")
    fig.update_yaxes(range=[0, df["premium"].max() * 1.25])
    return _base(fig, 340, f"Premium by {label} (current: {current})", legend=False)


def grid_maps(res, fx, fy, xs, ys, lx, ly, cur_x, cur_y):
    """Two side-by-side maps: dollars, and which tier the customer lands in."""
    z = res["premium"].values.reshape(len(ys), len(xs))
    tier_idx = res["tier"].map({"Low": 0, "Medium": 1, "High": 2}).values.reshape(len(ys), len(xs))
    f1 = go.Figure(go.Heatmap(z=z, x=xs, y=ys, colorscale="Tealgrn", colorbar=dict(title="$", tickformat=",")))
    f2 = go.Figure(go.Heatmap(
        z=tier_idx, x=xs, y=ys, zmin=0, zmax=2, showscale=False,
        colorscale=[[0, TIER_COLORS["Low"]], [0.33, TIER_COLORS["Low"]], [0.34, TIER_COLORS["Medium"]],
                    [0.66, TIER_COLORS["Medium"]], [0.67, TIER_COLORS["High"]], [1, TIER_COLORS["High"]]]))
    for f, title in [(f1, "Premium ($)"), (f2, "Which tier? (green = Low, amber = Medium, red = High)")]:
        f.add_trace(go.Scatter(x=[cur_x], y=[cur_y], mode="markers", name="current customer",
                               marker=dict(symbol="diamond", size=15, color="white", line=dict(width=2, color=INK))))
        f.update_xaxes(title=lx)
        f.update_yaxes(title=ly)
        _base(f, 380, title, legend=False)
    return f1, f2


def routing_bars(routes):
    """Bonus 1: what a wrong tier does to the quote (mean quote error, in dollars)."""
    wrong = [r for r in routes if r["actual"] != r["predicted"]]
    labels = [f"{r['actual']} sent to {r['predicted']}<br>(n={r['n']})" for r in wrong]
    fig = go.Figure(go.Bar(x=labels, y=[r["mean_error"] for r in wrong],
                           marker_color=[TIER_COLORS[r["actual"]] for r in wrong],
                           text=[f"{r['mean_error']:+,.0f}" for r in wrong], textposition="outside"))
    fig.add_hline(y=0, line_color=GREY)
    fig.update_yaxes(title="Average quote minus true charge ($)", tickprefix="$", tickformat=",")
    return _base(fig, 360, "What a wrong tier does to the quote", legend=False)
