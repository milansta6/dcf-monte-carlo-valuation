"""Reproduction of the numbers reported in the thesis.

The thesis ran both simulations from *one* random number generator
(seed 20260927): first the 100,000 runs of Model A, then the 100,000 runs of
Model B, continuing the same stream. :func:`reproduce` does the same, so the
Monte Carlo statistics match the thesis exactly, not only statistically.

Model A is compared with the thesis in euros. Company B is confidential and
its amounts in ``examples/company_b.yaml`` are scaled by an undisclosed factor,
so Company B is compared through ratios (value / base case, shares,
probabilities, correlations), which scaling leaves unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analysis import (
    TornadoBar,
    scenario_grid_stats,
    sensitivity_grid,
    spearman,
    summarize,
    tornado,
    wacc_table,
)
from .config import ValuationConfig, load_config
from .dcf import BaseCase, base_case
from .montecarlo import Simulation, simulate

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
THESIS_SEED = 20260927


@dataclass
class ModelRun:
    """Everything computed for one model."""

    config: ValuationConfig
    base: BaseCase
    simulation: Simulation
    stats: dict[str, float]
    spearman: dict[str, float]
    tornado: list[TornadoBar]
    grid: np.ndarray
    wacc_rows: list[dict[str, float]] = field(default_factory=list)
    scenario_stats: dict[str, float] | None = None


def run_model(cfg: ValuationConfig, rng: np.random.Generator | int | None = None, runs: int | None = None) -> ModelRun:
    """Base case, simulation and all sensitivity analyses for one configuration."""
    base = base_case(cfg)
    sim = simulate(cfg, runs=runs, rng=rng)
    s = cfg.sensitivity
    grid = sensitivity_grid(cfg, s.wacc, s.terminal_growth) if s.wacc and s.terminal_growth else np.empty((0, 0))
    scen = None
    if s.scenario_wacc and s.scenario_growth:
        scen = scenario_grid_stats(sensitivity_grid(cfg, s.scenario_wacc, s.scenario_growth))
    return ModelRun(
        config=cfg,
        base=base,
        simulation=sim,
        stats=summarize(sim.equity_value, base.equity_value, cfg.monte_carlo.thresholds),
        spearman=spearman(sim),
        tornado=tornado(cfg, sim),
        grid=grid,
        wacc_rows=wacc_table(cfg, s.wacc_offsets) if s.wacc_offsets else [],
        scenario_stats=scen,
    )


def reproduce(examples_dir: str | Path = EXAMPLES, runs: int | None = None) -> dict[str, ModelRun]:
    """Run both thesis examples on one shared random stream, in the thesis order."""
    examples_dir = Path(examples_dir)
    rng = np.random.default_rng(THESIS_SEED)
    out = {}
    for name in ("startup", "company_b"):
        cfg = load_config(examples_dir / f"{name}.yaml")
        out[name] = run_model(cfg, rng=rng, runs=runs)
    return out


# --------------------------------------------------------------------------- reference values
@dataclass
class Check:
    model: str
    quantity: str
    source: str
    thesis: float
    code: float
    tol: float
    relative: bool = False

    @property
    def diff(self) -> float:
        return self.code - self.thesis

    @property
    def passed(self) -> bool:
        if self.relative:
            return abs(self.diff) <= self.tol * abs(self.thesis)
        return abs(self.diff) <= self.tol


# Model A: values as printed in the thesis (EUR). Tolerances reflect the rounding of the print.
STARTUP_DETERMINISTIC = {
    # quantity: (thesis value, tolerance, source)
    "WACC": (0.2411, 0.00005, "Table 5"),
    "FCFF 2027": (-35270, 1, "Table 6"),
    "FCFF 2028": (-41832, 1, "Table 6"),
    "FCFF 2029": (183688, 1, "Table 6"),
    "FCFF 2030": (378702, 1, "Table 6"),
    "FCFF 2031": (661471, 1, "Table 6"),
    "PV of FCFF 2027-2031": (473202, 1, "Table 6"),
    "Terminal value": (3084835, 1, "Table 6"),
    "PV of terminal value": (940365, 1, "Table 6"),
    "Enterprise value": (1413568, 1, "Table 6"),
    "Equity value": (1457054, 1, "Table 6"),
    "TV / EV": (0.665, 0.0005, "Table 6"),
}
STARTUP_WACC_TABLE = [2791797, 2318794, 1959868, 1680028, 1457054, 1276191, 1127275, 1003091, 898386]  # Table 7
STARTUP_MC = {
    "mean": (1674382, "Table 9"),
    "p50": (1467570, "Table 9"),
    "std": (1102647, "Table 9"),
    "p5": (304751, "Table 9"),
    "p25": (890890, "Table 9"),
    "p75": (2224765, "Table 9"),
    "p95": (3757977, "Table 9"),
}
STARTUP_MC_PROB = {
    "cv": (0.66, 0.005, "Table 9"),
    "p_negative": (0.009, 0.0005, "Table 9"),
    "p_below_500000": (0.101, 0.0005, "Table 9"),
    "p_above_2000000": (0.311, 0.0005, "Table 9"),
    "p_below_base": (0.495, 0.0005, "Table 9"),
}
STARTUP_SPEARMAN = {"revenue_deviation": 0.96, "wacc": -0.24, "fixed_cost_deviation": -0.06, "terminal_growth": 0.02}
STARTUP_TORNADO = {  # EUR million, Section 4.7
    "revenue_deviation": (0.51, 2.97),
    "wacc": (1.84, 1.18),
    "fixed_cost_deviation": (1.53, 1.38),
}

# Company B: ratios to the base-case equity value (scale-free), Sections 5.4-5.6.
COMPANY_B_DETERMINISTIC = {
    "Levered beta": (1.00, 0.005, "Table 11"),
    "Cost of equity": (0.1593, 0.00005, "Table 11"),
    "WACC (computed)": (0.1096, 0.00005, "Table 11"),
    "WACC (used)": (0.110, 0.00001, "Table 11"),
}
COMPANY_B_RATIOS = {  # thesis value / thesis base case; tolerance is relative, covers the rounding of the scaled inputs
    "TV / EV": (1.557, "Table 13"),
    "EV / equity": (0.6588, "Table 13"),
    "MC mean / base": (0.9938, "Table 16"),
    "MC median / base": (0.9982, "Table 16"),
    "MC P5 / base": (0.5704, "Table 16"),
    "MC P25 / base": (0.8261, "Table 16"),
    "MC P75 / base": (1.1659, "Table 16"),
    "MC P95 / base": (1.4004, "Table 16"),
    "MC std / base": (0.2530, "Table 16"),
    "Grid min / base (WACC 12%, g 1%)": (0.7655, "Figure 12"),
    "Grid max / base (WACC 10%, g 3%)": (1.3686, "Figure 12"),
    "Scenario grid mean / base": (1.0030, "Table 14 (49 scenarios)"),
    "Scenario grid std / base": (0.06036, "Table 14 (49 scenarios)"),
    "Scenario grid min / base": (0.8880, "Table 14 (49 scenarios)"),
    "Scenario grid max / base": (1.1369, "Table 14 (49 scenarios)"),
}
COMPANY_B_MC_PROB = {
    "cv": (0.2546, 0.003, "Table 16"),
    "p_below_base": (0.503, 0.003, "Table 16"),
    "P(value < 0.799 x base)": (0.218, 0.005, "Table 16 (EUR 6.0m)"),
    "P(value > 1.198 x base)": (0.210, 0.005, "Table 16 (EUR 9.0m)"),
}
COMPANY_B_SPEARMAN = {
    "ebit_margin_shift": 0.75,
    "revenue_growth_deviation": -0.52,
    "wacc": -0.31,
    "terminal_growth": 0.12,
}
COMPANY_B_TORNADO = {  # value / base at the 10th and 90th percentile of each input, Section 5.6
    "ebit_margin_shift": (0.7557, 1.2451),
    "revenue_growth_deviation": (1.1548, 0.8061),
    "wacc": (1.1130, 0.9029),
    "terminal_growth": (0.9604, 1.0426),
}
RATIO_TOL = 0.01  # 1 % relative: rounding Company B's scaled inputs to EUR 10,000 moves ratios by up to ~0.5 %
PRINT_EPS = 1e-9  # a value exactly on the rounding boundary of the print (e.g. 0.3105 -> "31.1%") still passes


def checks(results: dict[str, ModelRun], exact_stream: bool = True) -> list[Check]:
    """Compare computed results with the thesis.

    With ``exact_stream=False`` (independent seeds or fewer runs), Monte Carlo
    statistics are compared within a sampling tolerance instead of exactly.
    """
    out: list[Check] = []
    mc_tol = 1.0 if exact_stream else 0.03  # EUR exactly, or 3 % relative
    prob_slack = 0.0 if exact_stream else 0.01

    # ---------------- Model A
    a = results["startup"]
    v = a.base.valuation
    code_a = {
        "WACC": a.base.wacc.wacc,
        **{f"FCFF {y}": float(f) for y, f in zip(a.config.projection.years, a.base.projection.fcf[0], strict=True)},
        "PV of FCFF 2027-2031": float(v.pv_fcf[0]),
        "Terminal value": float(v.terminal_value[0]),
        "PV of terminal value": float(v.pv_terminal_value[0]),
        "Enterprise value": float(v.enterprise_value[0]),
        "Equity value": float(v.equity_value[0]),
        "TV / EV": float(v.tv_share[0]),
    }
    for q, (ref, tol, src) in STARTUP_DETERMINISTIC.items():
        out.append(Check("A", q, src, ref, code_a[q], tol))
    for row, ref in zip(a.wacc_rows, STARTUP_WACC_TABLE, strict=True):
        out.append(Check("A", f"Equity at WACC {row['wacc']:.1%}", "Table 7", ref, row["equity"], 1))
    if a.grid.size:
        out.append(Check("A", "Grid minimum (EUR m)", "Section 4.6", 0.97, a.grid.min() / 1e6, 0.005))
        out.append(Check("A", "Grid maximum (EUR m)", "Section 4.6", 2.56, a.grid.max() / 1e6, 0.005))
    for k, (ref, src) in STARTUP_MC.items():
        out.append(Check("A", f"MC {k}", src, ref, a.stats[k], mc_tol, relative=not exact_stream))
    for k, (ref, tol, src) in STARTUP_MC_PROB.items():
        out.append(Check("A", f"MC {k}", src, ref, a.stats[k], tol + prob_slack + PRINT_EPS))
    for k, ref in STARTUP_SPEARMAN.items():
        out.append(Check("A", f"Spearman {k}", "Section 4.7", ref, a.spearman[k], 0.005 + prob_slack))
    bars = {b.name: b for b in a.tornado}
    for k, (lo, hi) in STARTUP_TORNADO.items():
        out.append(
            Check("A", f"Tornado {k} @P10 (EUR m)", "Section 4.7", lo, bars[k].value_at_low / 1e6, 0.005 + prob_slack)
        )
        out.append(
            Check("A", f"Tornado {k} @P90 (EUR m)", "Section 4.7", hi, bars[k].value_at_high / 1e6, 0.005 + prob_slack)
        )

    # ---------------- Company B (scale-free)
    b = results["company_b"]
    w = b.base.wacc
    code_b = {
        "Levered beta": w.levered_beta,
        "Cost of equity": w.cost_of_equity,
        "WACC (computed)": w.wacc_computed,
        "WACC (used)": w.wacc,
    }
    for q, (ref, tol, src) in COMPANY_B_DETERMINISTIC.items():
        out.append(Check("B", q, src, ref, code_b[q], tol))
    base = b.base.equity_value
    vb = b.base.valuation
    ratio_tol = RATIO_TOL if exact_stream else 0.04
    code_r = {
        "TV / EV": float(vb.tv_share[0]),
        "EV / equity": b.base.enterprise_value / base,
        "MC mean / base": b.stats["mean"] / base,
        "MC median / base": b.stats["p50"] / base,
        "MC P5 / base": b.stats["p5"] / base,
        "MC P25 / base": b.stats["p25"] / base,
        "MC P75 / base": b.stats["p75"] / base,
        "MC P95 / base": b.stats["p95"] / base,
        "MC std / base": b.stats["std"] / base,
        "Grid min / base (WACC 12%, g 1%)": b.grid.min() / base,
        "Grid max / base (WACC 10%, g 3%)": b.grid.max() / base,
        "Scenario grid mean / base": b.scenario_stats["mean"] / base,
        "Scenario grid std / base": b.scenario_stats["std"] / base,
        "Scenario grid min / base": b.scenario_stats["min"] / base,
        "Scenario grid max / base": b.scenario_stats["max"] / base,
    }
    for q, (ref, src) in COMPANY_B_RATIOS.items():
        tol = ratio_tol if q.startswith("MC") else RATIO_TOL
        out.append(Check("B", q, src, ref, code_r[q], tol, relative=True))
    eq = b.simulation.equity_value
    code_p = {
        "cv": b.stats["cv"],
        "p_below_base": b.stats["p_below_base"],
        "P(value < 0.799 x base)": float((eq < 0.79894 * base).mean()),
        "P(value > 1.198 x base)": float((eq > 1.19842 * base).mean()),
    }
    for q, (ref, tol, src) in COMPANY_B_MC_PROB.items():
        out.append(Check("B", f"MC {q}", src, ref, code_p[q], tol + prob_slack))
    for k, ref in COMPANY_B_SPEARMAN.items():
        out.append(Check("B", f"Spearman {k}", "Section 5.6", ref, b.spearman[k], 0.005 + prob_slack))
    bars = {x.name: x for x in b.tornado}
    for k, (lo, hi) in COMPANY_B_TORNADO.items():
        out.append(
            Check("B", f"Tornado {k} @P10 / base", "Section 5.6", lo, bars[k].value_at_low / base, 0.01, relative=True)
        )
        out.append(
            Check("B", f"Tornado {k} @P90 / base", "Section 5.6", hi, bars[k].value_at_high / base, 0.01, relative=True)
        )
    return out
