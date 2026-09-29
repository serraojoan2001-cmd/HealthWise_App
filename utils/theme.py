"""
===============================================================================
 FILE : utils/theme.py
 ROLE : LOOK & FEEL (colours, fonts, and the small UI building blocks every page reuses)
===============================================================================
"""
import html

import streamlit as st

INK = "#12303B"        # deep teal-navy text
TEAL = "#0E7C86"       # primary accent
MIST = "#F1F5F6"       # quiet panel background
RULE = "#D5DEE1"       # hairlines
GREY = "#7B8B91"
TIER_COLORS = {"Low": "#3A9D6E", "Medium": "#E0A02B", "High": "#C8453F"}
TIER_SOFT = {"Low": "#E4F3EB", "Medium": "#FBF0D9", "High": "#F8E2E0"}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Source+Sans+3:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label {{
    font-family: 'Source Sans 3', 'Segoe UI', system-ui, sans-serif;
}}
h1, h2, h3, .hw-title {{
    font-family: 'Newsreader', Georgia, 'Times New Roman', serif !important;
    color: {INK}; letter-spacing: -0.01em;
}}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}

/* page header */
.hw-title {{ font-size: 2.35rem; font-weight: 600; line-height: 1.12; margin: 0 0 .35rem 0; }}
.hw-sub {{ color: {GREY}; font-size: 1.05rem; margin: 0 0 1.4rem 0; max-width: 70ch; }}

/* key insights: the closing note on every page */
.hw-insights {{ border-left: 4px solid {TEAL}; background: {MIST}; padding: 1.05rem 1.3rem 1.05rem 1.4rem;
    margin-top: 2.4rem; border-radius: 0 6px 6px 0; }}
.hw-insights h4 {{ font-family: 'Newsreader', Georgia, serif; margin: 0 0 .5rem 0; font-size: 1.25rem; color: {INK}; }}
.hw-insights ul {{ margin: 0; padding-left: 1.15rem; }}
.hw-insights li {{ margin: .32rem 0; line-height: 1.5; }}

/* thought-process note */
.hw-thought {{ border: 1px solid {RULE}; border-radius: 6px; padding: .8rem 1.1rem; margin: .4rem 0 1rem 0; background: #fff; }}
.hw-thought b {{ color: {TEAL}; }}

/* tier chip + premium ticket (Price Lab) */
.hw-chip {{ display:inline-block; padding: .12rem .7rem; border-radius: 999px; font-weight: 600; font-size: .95rem; }}
.hw-ticket {{ border: 1px solid {RULE}; border-radius: 8px; padding: 1.1rem 1.4rem; background: #fff; }}
.hw-ticket .amt {{ font-family: 'Newsreader', Georgia, serif; font-size: 3.1rem; font-weight: 600; line-height: 1; color: {INK}; }}
.hw-ticket .lbl {{ color: {GREY}; font-size: .95rem; margin-bottom: .3rem; }}
.hw-delta-up {{ color: {TIER_COLORS['High']}; font-weight: 600; }}
.hw-delta-down {{ color: {TIER_COLORS['Low']}; font-weight: 600; }}

/* KPI tiles */
div[data-testid="stMetric"] {{ background: #fff; border: 1px solid {RULE}; border-radius: 6px; padding: .8rem 1rem; }}
div[data-testid="stMetricValue"] {{ font-family: 'Newsreader', Georgia, serif; color: {INK}; }}
</style>
"""


def apply_theme():
    """Inject the stylesheet once per page run."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str):
    apply_theme()
    st.markdown(f'<div class="hw-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hw-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def thought(text: str, label: str = "How I thought about it"):
    """A short 'thought process' note (rendered above the evidence it explains)."""
    st.markdown(f'<div class="hw-thought"><b>{html.escape(label)}.</b> {text}</div>',
                unsafe_allow_html=True)


def key_insights(bullets):
    """The closing 'Key insights' box - call this LAST on every page."""
    items = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f'<div class="hw-insights"><h4>Key insights</h4><ul>{items}</ul></div>',
                unsafe_allow_html=True)


def tier_chip(tier: str) -> str:
    return (f'<span class="hw-chip" style="background:{TIER_SOFT[tier]};color:{TIER_COLORS[tier]};'
            f'border:1px solid {TIER_COLORS[tier]}">{tier} risk</span>')


def qa_block(items, title="Lab questions and my answers"):
    """items = [(question, answer_markdown), ...] shown in a collapsed expander."""
    with st.expander(title):
        for q, a in items:
            st.markdown(f"**{q}**")
            st.markdown(a)
            st.divider()
