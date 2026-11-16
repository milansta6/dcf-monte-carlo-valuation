"""Loading and validating valuation configurations from YAML files.

A configuration describes one company: the operating forecast, the cost of
capital inputs, the terminal value convention, the equity bridge and the
Monte Carlo set-up. See ``examples/startup.yaml`` for an annotated example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

OPERATING_METHODS = ("cost_build_up", "ebit_margin")
NWC_METHODS = ("flows", "levels")
TERMINAL_TIMINGS = ("same_as_last_year", "one_year_after_last_year")
DISTRIBUTIONS = ("normal", "triangular", "uniform")

# Which stochastic inputs exist and what they do to the model.
MC_VARIABLES = {
    "revenue_deviation": "proportional deviation from planned revenue, compounding each year: x (1+d)^t",
    "revenue_growth_deviation": "additive deviation from the planned revenue growth rate in every year",
    "fixed_cost_deviation": "proportional deviation of fixed operating costs from plan: x (1+k)",
    "ebit_margin_shift": "additive shift of the planned EBIT margin path",
    "wacc": "discount rate",
    "terminal_growth": "perpetual growth rate g of the terminal value",
}


class ConfigError(ValueError):
    """Raised when a configuration file is incomplete or inconsistent."""


@dataclass(frozen=True)
class CostOfCapitalInputs:
    risk_free_rate: float
    equity_risk_premium: float
    country_risk_premium: float
    unlevered_beta: float
    debt_to_equity: float = 0.0
    specific_risk_premium: float = 0.0
    pre_tax_cost_of_debt: float = 0.0
    wacc_override: float | None = None
    specific_risk_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Projection:
    years: list[int]
    method: str
    revenue: np.ndarray
    depreciation: np.ndarray
    capex: np.ndarray
    nwc_method: str
    # cost_build_up
    variable_costs: np.ndarray | None = None
    fixed_costs: np.ndarray | None = None
    # ebit_margin
    ebit: np.ndarray | None = None
    base_year_revenue: float | None = None
    # net working capital
    nwc_investment: np.ndarray | None = None  # flows: + = cash outflow
    nwc_levels: np.ndarray | None = None  # levels: closing balances
    opening_nwc: float | None = None

    @property
    def n_years(self) -> int:
        return len(self.years)


@dataclass(frozen=True)
class TerminalValue:
    growth: float
    timing: str
    fcf_growth: float | None = None  # None -> terminal FCF = last FCF x (1 + g)
    fcf_override: float | None = None  # used in the deterministic base case only


@dataclass(frozen=True)
class MCVariable:
    name: str
    distribution: str
    params: dict[str, float | str]
    clip: tuple[float, float] | None = None


@dataclass(frozen=True)
class MonteCarloSettings:
    runs: int
    seed: int | None
    variables: list[MCVariable]
    tornado_percentiles: tuple[float, float] = (10.0, 90.0)
    thresholds: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class SensitivitySettings:
    wacc: list[float]
    terminal_growth: list[float]
    wacc_offsets: list[float] = field(default_factory=list)
    scenario_wacc: list[float] = field(default_factory=list)
    scenario_growth: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ValuationConfig:
    name: str
    short_name: str
    currency: str
    valuation_date: str
    tax_rate: float
    cost_of_capital: CostOfCapitalInputs
    projection: Projection
    terminal: TerminalValue
    bridge: dict[str, float]
    monte_carlo: MonteCarloSettings
    sensitivity: SensitivitySettings
    notes: str = ""
    source: Path | None = None

    @property
    def bridge_total(self) -> float:
        return float(sum(self.bridge.values()))


# --------------------------------------------------------------------------- helpers
def _require(d: dict, key: str, where: str) -> Any:
    if key not in d or d[key] is None:
        raise ConfigError(f"missing required key '{key}' in {where}")
    return d[key]


def _array(values: Any, n: int, where: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise ConfigError(f"{where} must have {n} values (one per projection year), got {arr.shape}")
    return arr


def _parse_projection(d: dict) -> Projection:
    years = [int(y) for y in _require(d, "years", "projection")]
    n = len(years)
    method = _require(d, "method", "projection")
    if method not in OPERATING_METHODS:
        raise ConfigError(f"projection.method must be one of {OPERATING_METHODS}, got '{method}'")
    kw: dict[str, Any] = {
        "years": years,
        "method": method,
        "revenue": _array(_require(d, "revenue", "projection"), n, "projection.revenue"),
        "depreciation": _array(d.get("depreciation", 0.0), n, "projection.depreciation"),
        "capex": _array(d.get("capex", 0.0), n, "projection.capex"),
    }
    if method == "cost_build_up":
        kw["variable_costs"] = _array(d.get("variable_costs", 0.0), n, "projection.variable_costs")
        kw["fixed_costs"] = _array(d.get("fixed_costs", 0.0), n, "projection.fixed_costs")
    else:
        kw["ebit"] = _array(_require(d, "ebit", "projection"), n, "projection.ebit")
    if d.get("base_year_revenue") is not None:
        kw["base_year_revenue"] = float(d["base_year_revenue"])

    nwc = _require(d, "net_working_capital", "projection")
    nwc_method = _require(nwc, "method", "projection.net_working_capital")
    if nwc_method not in NWC_METHODS:
        raise ConfigError(f"net_working_capital.method must be one of {NWC_METHODS}")
    kw["nwc_method"] = nwc_method
    if nwc_method == "flows":
        kw["nwc_investment"] = _array(
            _require(nwc, "investment", "net_working_capital"), n, "net_working_capital.investment"
        )
    else:
        kw["nwc_levels"] = _array(_require(nwc, "levels", "net_working_capital"), n, "net_working_capital.levels")
        kw["opening_nwc"] = float(_require(nwc, "opening", "net_working_capital"))
    return Projection(**kw)


def _parse_mc(d: dict) -> MonteCarloSettings:
    variables = []
    for name, spec in _require(d, "variables", "monte_carlo").items():
        if name not in MC_VARIABLES:
            raise ConfigError(f"unknown Monte Carlo variable '{name}'; known: {sorted(MC_VARIABLES)}")
        dist = _require(spec, "distribution", f"monte_carlo.variables.{name}")
        if dist not in DISTRIBUTIONS:
            raise ConfigError(f"distribution of '{name}' must be one of {DISTRIBUTIONS}")
        params = {k: v for k, v in spec.items() if k not in ("distribution", "clip")}
        clip = tuple(float(x) for x in spec["clip"]) if spec.get("clip") is not None else None
        variables.append(MCVariable(name, dist, params, clip))  # type: ignore[arg-type]
    seed = d.get("seed")
    return MonteCarloSettings(
        runs=int(d.get("runs", 100_000)),
        seed=None if seed is None else int(seed),
        variables=variables,
        tornado_percentiles=tuple(float(x) for x in d.get("tornado_percentiles", (10, 90))),  # type: ignore[arg-type]
        thresholds=[float(x) for x in d.get("thresholds", [])],
    )


def parse_config(raw: dict, source: Path | None = None) -> ValuationConfig:
    """Build a :class:`ValuationConfig` from a parsed YAML mapping."""
    coc = dict(_require(raw, "cost_of_capital", "config"))
    components = coc.pop("specific_risk_components", {}) or {}
    try:
        coc_inputs = CostOfCapitalInputs(**coc, specific_risk_components=components)
    except TypeError as exc:  # unknown key
        raise ConfigError(f"cost_of_capital: {exc}") from exc
    if components and abs(sum(components.values()) - coc_inputs.specific_risk_premium) > 1e-9:
        raise ConfigError("specific_risk_components do not add up to specific_risk_premium")

    term = _require(raw, "terminal_value", "config")
    timing = term.get("discounting", "same_as_last_year")
    if timing not in TERMINAL_TIMINGS:
        raise ConfigError(f"terminal_value.discounting must be one of {TERMINAL_TIMINGS}")
    terminal = TerminalValue(
        growth=float(_require(term, "growth", "terminal_value")),
        timing=timing,
        fcf_growth=None if term.get("fcf_growth") is None else float(term["fcf_growth"]),
        fcf_override=None if term.get("fcf_override") is None else float(term["fcf_override"]),
    )

    sens = raw.get("sensitivity", {}) or {}
    sensitivity = SensitivitySettings(
        wacc=[float(x) for x in sens.get("wacc", [])],
        terminal_growth=[float(x) for x in sens.get("terminal_growth", [])],
        wacc_offsets=[float(x) for x in sens.get("wacc_offsets", [])],
        scenario_wacc=[float(x) for x in sens.get("scenario_wacc", [])],
        scenario_growth=[float(x) for x in sens.get("scenario_growth", [])],
    )

    projection = _parse_projection(_require(raw, "projection", "config"))
    mc = _parse_mc(_require(raw, "monte_carlo", "config"))
    names = {v.name for v in mc.variables}
    if projection.method != "cost_build_up" and "fixed_cost_deviation" in names:
        raise ConfigError("fixed_cost_deviation requires projection.method = cost_build_up")
    if projection.method != "ebit_margin" and "ebit_margin_shift" in names:
        raise ConfigError("ebit_margin_shift requires projection.method = ebit_margin")
    if "revenue_growth_deviation" in names and projection.base_year_revenue is None:
        raise ConfigError("revenue_growth_deviation requires projection.base_year_revenue")
    if {"revenue_deviation", "revenue_growth_deviation"} <= names:
        raise ConfigError("use either revenue_deviation or revenue_growth_deviation, not both")

    return ValuationConfig(
        name=str(_require(raw, "name", "config")),
        short_name=str(raw.get("short_name", source.stem if source else "model")),
        currency=str(raw.get("currency", "EUR")),
        valuation_date=str(raw.get("valuation_date", "")),
        tax_rate=float(_require(raw, "tax_rate", "config")),
        cost_of_capital=coc_inputs,
        projection=projection,
        terminal=terminal,
        bridge={str(k): float(v) for k, v in (raw.get("equity_bridge") or {}).items()},
        monte_carlo=mc,
        sensitivity=sensitivity,
        notes=str(raw.get("notes", "")),
        source=source,
    )


def load_config(path: str | Path) -> ValuationConfig:
    """Read a YAML configuration file."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} does not contain a YAML mapping")
    return parse_config(raw, source=path)
