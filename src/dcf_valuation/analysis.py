"""Summaries of a valuation: statistics, sensitivity tables, tornado and rank correlations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from .config import ValuationConfig
from .dcf import base_case, equity_value
from .montecarlo import Simulation
from .wacc import compute_wacc

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)

# Human-readable labels for the simulated inputs
LABELS = {
    "revenue_deviation": "Revenue vs plan (compounding)",
    "revenue_growth_deviation": "Revenue growth vs plan",
    "fixed_cost_deviation": "Fixed operating costs",
    "ebit_margin_shift": "EBIT margin",
    "wacc": "Discount rate (WACC)",
    "terminal_growth": "Terminal growth rate g",
}


def summarize(values: np.ndarray, base: float, thresholds: list[float] | None = None) -> dict[str, float]:
    """Descriptive statistics of simulated values (population standard deviation, as in the thesis)."""
    x = np.asarray(values, dtype=float)
    q = np.percentile(x, PERCENTILES)
    out = {
        "base": float(base),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "cv": float(x.std() / x.mean()),
        "min": float(x.min()),
        "max": float(x.max()),
        **{f"p{p}": float(v) for p, v in zip(PERCENTILES, q, strict=True)},
        "p_negative": float((x < 0).mean()),
        "p_below_base": float((x < base).mean()),
    }
    for t in thresholds or []:
        out[f"p_below_{t:.0f}"] = float((x < t).mean())
        out[f"p_above_{t:.0f}"] = float((x > t).mean())
    return out


def spearman(sim: Simulation) -> dict[str, float]:
    """Spearman rank correlation between each simulated input and the equity value."""
    return {k: float(spearmanr(v, sim.equity_value).statistic) for k, v in sim.inputs.items()}


@dataclass
class TornadoBar:
    name: str
    label: str
    input_low: float
    input_high: float
    value_at_low: float
    value_at_high: float

    @property
    def width(self) -> float:
        return abs(self.value_at_high - self.value_at_low)


def tornado(cfg: ValuationConfig, sim: Simulation) -> list[TornadoBar]:
    """One-at-a-time sensitivity (thesis Figures 8 and 14).

    Each input is set to its simulated 10th and 90th percentile (configurable)
    while all others stay at their base values. Bars are sorted by width.
    """
    lo_p, hi_p = cfg.monte_carlo.tornado_percentiles
    bars = []
    for name, x in sim.inputs.items():
        lo, hi = np.percentile(x, [lo_p, hi_p])
        if name == "wacc":
            v_lo, v_hi = equity_value(cfg, wacc=lo), equity_value(cfg, wacc=hi)
        elif name == "terminal_growth":
            v_lo, v_hi = equity_value(cfg, terminal_growth=lo), equity_value(cfg, terminal_growth=hi)
        else:
            v_lo, v_hi = equity_value(cfg, **{name: lo}), equity_value(cfg, **{name: hi})
        bars.append(TornadoBar(name, LABELS[name], float(lo), float(hi), float(v_lo[0]), float(v_hi[0])))
    return sorted(bars, key=lambda b: b.width, reverse=True)


def sensitivity_grid(cfg: ValuationConfig, waccs: list[float], growths: list[float]) -> np.ndarray:
    """Base-case equity value for every combination of WACC (rows) and terminal growth (columns)."""
    return np.array([[base_case(cfg, wacc=w, g=g).equity_value for g in growths] for w in waccs])


def wacc_table(cfg: ValuationConfig, offsets: list[float]) -> list[dict[str, float]]:
    """One-dimensional sensitivity to the discount rate (thesis Table 7)."""
    base_w = compute_wacc(cfg.cost_of_capital, cfg.tax_rate).wacc
    rows = []
    for off in offsets:
        bc = base_case(cfg, wacc=base_w + off)
        v = bc.valuation
        rows.append(
            {
                "wacc": base_w + off,
                "offset": off,
                "pv_fcf": float(v.pv_fcf[0]),
                "pv_tv": float(v.pv_terminal_value[0]),
                "ev": float(v.enterprise_value[0]),
                "equity": float(v.equity_value[0]),
                "tv_share": float(v.tv_share[0]),
            }
        )
    return rows


def scenario_grid_stats(values: np.ndarray) -> dict[str, float]:
    """Statistics of a scenario grid; sample standard deviation as in Excel's STDEV (thesis Table 14)."""
    x = np.ravel(values)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "max": float(x.max()),
    }
