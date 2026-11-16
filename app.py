"""Interactive DCF + Monte Carlo explorer.

pip install -e ".[app]"
streamlit run app.py
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import streamlit as st

from dcf_valuation import load_config, plots
from dcf_valuation.analysis import LABELS
from dcf_valuation.config import MCVariable
from dcf_valuation.thesis import run_model

EXAMPLES = Path(__file__).parent / "examples"
NAMES = {
    "Model A: pre-revenue software start-up": "startup",
    "Model B: furniture manufacturer (anonymised)": "company_b",
}

st.set_page_config(page_title="DCF + Monte Carlo", layout="wide")
st.title("DCF valuation with Monte Carlo simulation")
st.caption(
    "Companion app to the master's thesis *Company Valuation Using the Discounted Cash Flow Method and "
    "Monte Carlo Simulation* (Milan Stamenović, 2026). Change the assumptions on the left; the model is "
    "recomputed instantly."
)

choice = st.sidebar.selectbox("Company", list(NAMES))
cfg = load_config(EXAMPLES / f"{NAMES[choice]}.yaml")
coc = cfg.cost_of_capital
if NAMES[choice] == "company_b":
    st.info("Company B is anonymised: euro amounts are scaled by an undisclosed factor. Ratios are unchanged.")

st.sidebar.header("Cost of capital")
rf = st.sidebar.slider("Risk-free rate", 0.0, 0.08, coc.risk_free_rate, 0.0005, format="%.4f")
erp = st.sidebar.slider("Mature market equity risk premium", 0.02, 0.08, coc.equity_risk_premium, 0.0005, format="%.4f")
crp = st.sidebar.slider("Country risk premium", 0.0, 0.08, coc.country_risk_premium, 0.0005, format="%.4f")
beta = st.sidebar.slider("Unlevered beta", 0.3, 2.0, coc.unlevered_beta, 0.01)
scp = st.sidebar.slider("Specific risk premium", 0.0, 0.20, coc.specific_risk_premium, 0.005, format="%.3f")
use_override = coc.wacc_override is not None and st.sidebar.checkbox("Use the rounded WACC of the thesis", True)

st.sidebar.header("Terminal value")
g = st.sidebar.slider("Terminal growth rate g", 0.0, 0.04, cfg.terminal.growth, 0.0025, format="%.4f")

st.sidebar.header("Monte Carlo")
runs = st.sidebar.select_slider("Runs", [5_000, 20_000, 50_000, 100_000], 20_000)
seed = st.sidebar.number_input("Seed", value=cfg.monte_carlo.seed or 0, step=1)
new_vars = []
for v in cfg.monte_carlo.variables:
    p = dict(v.params)
    if "sd" in p:
        p["sd"] = st.sidebar.slider(
            f"{LABELS[v.name]}: standard deviation",
            0.0,
            float(p["sd"]) * 3,
            float(p["sd"]),
            float(p["sd"]) / 20,
            format="%.4f",
        )
    new_vars.append(MCVariable(v.name, v.distribution, p, v.clip))

cfg = dataclasses.replace(
    cfg,
    cost_of_capital=dataclasses.replace(
        coc,
        risk_free_rate=rf,
        equity_risk_premium=erp,
        country_risk_premium=crp,
        unlevered_beta=beta,
        specific_risk_premium=scp,
        specific_risk_components={},
        wacc_override=coc.wacc_override if use_override else None,
    ),
    terminal=dataclasses.replace(cfg.terminal, growth=g),
    monte_carlo=dataclasses.replace(cfg.monte_carlo, variables=new_vars),
)

try:
    run = run_model(cfg, rng=int(seed), runs=int(runs))
except ValueError as exc:
    st.error(f"Cannot value the company with these inputs: {exc}")
    st.stop()

s, base = run.stats, run.base.equity_value
c1, c2, c3, c4 = st.columns(4)
c1.metric("WACC", f"{run.base.wacc.wacc:.2%}")
c2.metric("Base-case equity value", f"{base:,.0f}")
c3.metric("90 % interval", f"{s['p5'] / 1e6:.2f}M - {s['p95'] / 1e6:.2f}M")
c4.metric("Coefficient of variation", f"{s['cv']:.2f}")

left, right = st.columns(2)
eq = run.simulation.equity_value
left.pyplot(plots.histogram(eq, base, "Distribution of equity value"))
right.pyplot(plots.tornado_chart(run.tornado, base, "What drives the value (P10-P90 of each input)"))
left.pyplot(plots.percentiles(eq, base, "Percentiles"))
sens = cfg.sensitivity
right.pyplot(plots.heatmap(run.grid, sens.wacc, sens.terminal_growth, "Equity value by WACC and g"))
st.caption("The heatmap uses the WACC grid of the thesis; the other charts use the inputs on the left.")
