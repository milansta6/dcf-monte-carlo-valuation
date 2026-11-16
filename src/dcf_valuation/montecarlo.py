"""Monte Carlo simulation of the DCF model (thesis Chapter 3.3 and Sections 4.7 / 5.6).

All runs are computed at once with NumPy array operations. The random inputs
are drawn in the order in which they appear in the configuration file, so a
fixed seed reproduces the same results on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MCVariable, ValuationConfig
from .dcf import discount, project
from .wacc import compute_wacc


def _resolve(value: float | str, base: float) -> float:
    """Allow ``mean: base`` / ``mode: base`` in the YAML to refer to the base-case value."""
    return base if value == "base" else float(value)


def draw(var: MCVariable, n: int, rng: np.random.Generator, base: float = 0.0) -> np.ndarray:
    """Draw ``n`` values of one input from its distribution."""
    p = var.params
    if var.distribution == "normal":
        x = rng.normal(_resolve(p.get("mean", 0.0), base), float(p["sd"]), n)
    elif var.distribution == "triangular":
        x = rng.triangular(float(p["min"]), _resolve(p["mode"], base), float(p["max"]), n)
    elif var.distribution == "uniform":
        x = rng.uniform(float(p["min"]), float(p["max"]), n)
    else:  # pragma: no cover - rejected by the config parser
        raise ValueError(var.distribution)
    if var.clip is not None:
        x = np.clip(x, *var.clip)
    return x


@dataclass
class Simulation:
    """Inputs and outputs of a simulation; every array has one entry per run."""

    inputs: dict[str, np.ndarray]
    equity_value: np.ndarray
    enterprise_value: np.ndarray
    tv_share: np.ndarray
    revenue_last_year: np.ndarray
    runs: int


def simulate(
    cfg: ValuationConfig,
    runs: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> Simulation:
    """Run the Monte Carlo simulation.

    ``rng`` may be a ``numpy.random.Generator`` (to share one random stream
    between several models, as the thesis did) or an integer seed; by default
    the seed from the configuration is used.
    """
    n = cfg.monte_carlo.runs if runs is None else int(runs)
    if rng is None or isinstance(rng, int | np.integer):
        rng = np.random.default_rng(cfg.monte_carlo.seed if rng is None else int(rng))

    base_wacc = compute_wacc(cfg.cost_of_capital, cfg.tax_rate).wacc
    bases = {"wacc": base_wacc, "terminal_growth": cfg.terminal.growth}
    inputs = {v.name: draw(v, n, rng, bases.get(v.name, 0.0)) for v in cfg.monte_carlo.variables}

    wacc = inputs.get("wacc", np.full(n, base_wacc))
    g = inputs.get("terminal_growth", np.full(n, cfg.terminal.growth))
    shocks = {k: v for k, v in inputs.items() if k not in ("wacc", "terminal_growth")}
    proj = project(cfg, **shocks)
    if proj.fcf.shape[0] != n:  # no operating shocks: same cash flows in every run
        proj.fcf = np.broadcast_to(proj.fcf, (n, proj.fcf.shape[1]))
        proj.revenue = np.broadcast_to(proj.revenue, (n, proj.revenue.shape[1]))
    val = discount(cfg, proj.fcf, wacc, g)
    return Simulation(
        inputs=inputs,
        equity_value=val.equity_value,
        enterprise_value=val.enterprise_value,
        tv_share=val.tv_share,
        revenue_last_year=proj.revenue[:, -1],
        runs=n,
    )
