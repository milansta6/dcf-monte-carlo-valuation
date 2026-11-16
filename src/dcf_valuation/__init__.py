"""DCF valuation with Monte Carlo simulation.

Companion code to the master's thesis "Company Valuation Using the Discounted
Cash Flow Method and Monte Carlo Simulation" (Milan Stamenović, School of
Computing, Union University, Belgrade, 2026).
"""

from .analysis import scenario_grid_stats, sensitivity_grid, spearman, summarize, tornado, wacc_table
from .config import ConfigError, ValuationConfig, load_config
from .dcf import base_case, discount, equity_value, project
from .montecarlo import simulate
from .wacc import compute_wacc, cost_of_equity, levered_beta

__version__ = "1.0.0"

__all__ = [
    "ConfigError",
    "ValuationConfig",
    "base_case",
    "compute_wacc",
    "cost_of_equity",
    "discount",
    "equity_value",
    "levered_beta",
    "load_config",
    "project",
    "scenario_grid_stats",
    "sensitivity_grid",
    "simulate",
    "spearman",
    "summarize",
    "tornado",
    "wacc_table",
]
