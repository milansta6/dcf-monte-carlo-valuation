"""Free cash flow projection and DCF valuation (thesis Chapter 1, eqs. 1-3).

Every function here is vectorised: shocks may be scalars (one scenario) or
arrays of length ``n`` (``n`` Monte Carlo runs). Results always carry a
leading scenario axis of length ``n``, so the deterministic base case is
simply the ``n = 1`` special case of the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ValuationConfig
from .wacc import WaccResult, compute_wacc


def _col(x: float | np.ndarray | None, default: float) -> np.ndarray:
    """Shape a scalar or a 1-D array of shocks as a column vector (n, 1)."""
    if x is None:
        x = default
    return np.atleast_1d(np.asarray(x, dtype=float))[:, None]


@dataclass
class Projection:
    """Projected operating figures, each of shape (n_scenarios, n_years)."""

    revenue: np.ndarray
    ebitda: np.ndarray
    ebit: np.ndarray
    nopat: np.ndarray
    depreciation: np.ndarray
    capex: np.ndarray
    nwc_investment: np.ndarray  # + = cash outflow
    fcf: np.ndarray


def project(
    cfg: ValuationConfig,
    *,
    revenue_deviation: float | np.ndarray | None = None,
    revenue_growth_deviation: float | np.ndarray | None = None,
    fixed_cost_deviation: float | np.ndarray | None = None,
    ebit_margin_shift: float | np.ndarray | None = None,
) -> Projection:
    """Project free cash flow to the firm, optionally under shocked assumptions.

    FCFF = EBIT x (1 - T) + D&A - CapEx - increase in NWC   (thesis eq. 1)

    The tax rate is applied to EBIT in every year, also when EBIT is
    negative; this is the convention of both thesis models (Section 4.5).
    """
    p = cfg.projection
    t = np.arange(1, p.n_years + 1)
    plan = p.revenue[None, :]

    # --- revenue
    if revenue_growth_deviation is not None:
        # planned growth path from the base year, shifted by a constant number of pp
        prev = np.concatenate([[p.base_year_revenue], p.revenue[:-1]])
        plan_growth = p.revenue / prev - 1.0
        growth = plan_growth[None, :] + _col(revenue_growth_deviation, 0.0)
        revenue = p.base_year_revenue * np.cumprod(1.0 + growth, axis=1)
        factor = revenue / plan
    elif revenue_deviation is not None:
        # a deviation d leaves revenue (1+d)^t times the plan in year t
        factor = (1.0 + _col(revenue_deviation, 0.0)) ** t[None, :]
        revenue = plan * factor
    else:
        factor = np.ones_like(plan)
        revenue = plan * factor
    n = revenue.shape[0]

    # --- operating profit
    da = np.broadcast_to(p.depreciation, (n, p.n_years))
    if p.method == "cost_build_up":
        variable = p.variable_costs[None, :] * factor  # variable costs move with revenue
        fixed = p.fixed_costs[None, :] * (1.0 + _col(fixed_cost_deviation, 0.0))
        ebitda = revenue - variable - fixed
        ebit = ebitda - da
    else:  # ebit_margin
        margin = (p.ebit / p.revenue)[None, :] + _col(ebit_margin_shift, 0.0)
        ebit = revenue * margin
        ebitda = ebit + da
    nopat = ebit * (1.0 - cfg.tax_rate)

    # --- reinvestment
    capex = np.broadcast_to(p.capex, (n, p.n_years))
    if p.nwc_method == "flows":
        nwc_inv = p.nwc_investment[None, :] * factor  # working capital flows move with revenue
    else:
        nwc = revenue * (p.nwc_levels / p.revenue)[None, :]  # constant NWC / revenue ratios
        opening = np.full((n, 1), p.opening_nwc)
        nwc_inv = nwc - np.concatenate([opening, nwc[:, :-1]], axis=1)

    fcf = nopat + da - capex - nwc_inv
    return Projection(revenue, ebitda, ebit, nopat, da, capex, nwc_inv, fcf)


@dataclass
class Valuation:
    """DCF valuation results, each of shape (n_scenarios,) except ``pv_by_year``."""

    pv_by_year: np.ndarray
    pv_fcf: np.ndarray
    terminal_fcf: np.ndarray
    terminal_value: np.ndarray
    pv_terminal_value: np.ndarray
    enterprise_value: np.ndarray
    equity_value: np.ndarray
    discount_factors: np.ndarray
    terminal_exponent: float

    @property
    def tv_share(self) -> np.ndarray:
        return self.pv_terminal_value / self.enterprise_value


def discount(
    cfg: ValuationConfig,
    fcf: np.ndarray,
    wacc: float | np.ndarray,
    g: float | np.ndarray,
    *,
    terminal_fcf: float | None = None,
) -> Valuation:
    """Discount FCFF with the mid-year convention and add a Gordon terminal value.

    TV = FCF_(n+1) / (WACC - g)                       (thesis eq. 2)
    EV = sum FCF_t / (1+WACC)^(t-0.5) + PV(TV)         (thesis eq. 3)

    ``terminal_fcf`` replaces the derived terminal-year FCF (used for the
    deterministic base case of Model A, whose spreadsheet built the 2032 cash
    flow line by line).
    """
    fcf = np.atleast_2d(fcf)
    n_years = fcf.shape[1]
    wacc = np.atleast_1d(np.asarray(wacc, dtype=float))
    g = np.atleast_1d(np.asarray(g, dtype=float))
    if np.any(wacc <= g):
        raise ValueError("WACC must exceed the terminal growth rate")

    exponents = np.arange(1, n_years + 1) - 0.5  # mid-year convention
    disc = (1.0 + wacc)[:, None] ** (-exponents[None, :])
    pv_by_year = fcf * disc
    pv_fcf = pv_by_year.sum(axis=1)

    term = cfg.terminal
    if terminal_fcf is not None:
        tfcf = np.full(fcf.shape[0], float(terminal_fcf))
    elif term.fcf_growth is not None:
        tfcf = fcf[:, -1] * (1.0 + term.fcf_growth)
    else:
        tfcf = fcf[:, -1] * (1.0 + g)
    tv = tfcf / (wacc - g)
    # Gordon gives the value one period before the first terminal cash flow, i.e. at
    # the date of the last explicit cash flow (exponent n - 0.5). Model A's spreadsheet
    # discounts one year further (n + 0.5); both conventions are discussed in Section 5.4.
    exponent = n_years - 0.5 if term.timing == "same_as_last_year" else n_years + 0.5
    pv_tv = tv * (1.0 + wacc) ** (-exponent)
    ev = pv_fcf + pv_tv
    equity = ev + cfg.bridge_total
    return Valuation(pv_by_year, pv_fcf, tfcf, tv, pv_tv, ev, equity, disc, exponent)


@dataclass
class BaseCase:
    """Deterministic base-case valuation of one configuration."""

    wacc: WaccResult
    growth: float
    projection: Projection
    valuation: Valuation

    @property
    def equity_value(self) -> float:
        return float(self.valuation.equity_value[0])

    @property
    def enterprise_value(self) -> float:
        return float(self.valuation.enterprise_value[0])


def base_case(cfg: ValuationConfig, *, wacc: float | None = None, g: float | None = None) -> BaseCase:
    """Value the company with all assumptions at their planned values."""
    wr = compute_wacc(cfg.cost_of_capital, cfg.tax_rate)
    w = wr.wacc if wacc is None else wacc
    growth = cfg.terminal.growth if g is None else g
    proj = project(cfg)
    val = discount(cfg, proj.fcf, w, growth, terminal_fcf=cfg.terminal.fcf_override)
    return BaseCase(wr, growth, proj, val)


def equity_value(
    cfg: ValuationConfig,
    *,
    wacc: float | np.ndarray | None = None,
    terminal_growth: float | np.ndarray | None = None,
    **shocks: float | np.ndarray | None,
) -> np.ndarray:
    """Equity value under shocked assumptions, as used by the simulation and the tornado.

    The terminal FCF is always derived from the last projected FCF here (the
    ``fcf_override`` of the base case is not used), exactly as in the thesis
    simulation code.
    """
    w = compute_wacc(cfg.cost_of_capital, cfg.tax_rate).wacc if wacc is None else wacc
    g = cfg.terminal.growth if terminal_growth is None else terminal_growth
    proj = project(cfg, **shocks)
    return discount(cfg, proj.fcf, w, g).equity_value
