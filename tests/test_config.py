import copy
from pathlib import Path

import pytest
import yaml

from dcf_valuation.config import ConfigError, load_config, parse_config

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def raw():
    return yaml.safe_load((EXAMPLES / "startup.yaml").read_text(encoding="utf-8"))


def test_examples_load():
    for f in EXAMPLES.glob("*.yaml"):
        cfg = load_config(f)
        assert cfg.projection.n_years == 5
        assert cfg.monte_carlo.runs == 100_000
        assert cfg.monte_carlo.seed == 20260927


def test_specific_risk_components_must_add_up(raw):
    raw["cost_of_capital"]["specific_risk_components"]["niche_market"] = 0.03
    with pytest.raises(ConfigError, match="add up"):
        parse_config(raw)


def test_wrong_number_of_years(raw):
    raw["projection"]["revenue"] = [1, 2, 3]
    with pytest.raises(ConfigError, match="5 values"):
        parse_config(raw)


@pytest.mark.parametrize(
    ("path", "value", "msg"),
    [
        (("projection", "method"), "magic", "projection.method"),
        (("terminal_value", "discounting"), "never", "discounting"),
        (("monte_carlo", "variables", "wacc", "distribution"), "cauchy", "distribution"),
    ],
)
def test_invalid_choices(raw, path, value, msg):
    d = raw
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    with pytest.raises(ConfigError, match=msg):
        parse_config(raw)


def test_unknown_variable_and_incompatible_combinations(raw):
    bad = copy.deepcopy(raw)
    bad["monte_carlo"]["variables"]["inflation"] = {"distribution": "normal", "sd": 0.01}
    with pytest.raises(ConfigError, match="unknown"):
        parse_config(bad)
    bad = copy.deepcopy(raw)
    bad["monte_carlo"]["variables"]["ebit_margin_shift"] = {"distribution": "normal", "sd": 0.01}
    with pytest.raises(ConfigError, match="ebit_margin"):
        parse_config(bad)


def test_missing_key(raw):
    del raw["tax_rate"]
    with pytest.raises(ConfigError, match="tax_rate"):
        parse_config(raw)
