import dataclasses

import numpy as np
import pytest

from dcf_valuation import base_case, equity_value, project, simulate, summarize
from dcf_valuation.config import MCVariable
from dcf_valuation.montecarlo import draw


@pytest.mark.parametrize("name", ["startup", "company_b"])
def test_vectorised_simulation_equals_run_by_run(name, request):
    """Each simulated value equals a separate evaluation of the model with that run's inputs."""
    cfg = request.getfixturevalue(name)
    sim = simulate(cfg, runs=50, rng=7)
    for i in range(50):
        x = {k: v[i] for k, v in sim.inputs.items()}
        w, g = x.pop("wacc"), x.pop("terminal_growth")
        single = equity_value(cfg, wacc=w, terminal_growth=g, **x)[0]
        assert sim.equity_value[i] == pytest.approx(single, rel=1e-12)


def test_fixed_seed_is_reproducible(startup):
    a = simulate(startup, runs=1000, rng=123).equity_value
    b = simulate(startup, runs=1000, rng=123).equity_value
    c = simulate(startup, runs=1000, rng=124).equity_value
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_seed_from_config_reproduces_thesis_model_a(startup):
    """Model A is first on the thesis random stream, so its own seed reproduces Table 9."""
    s = summarize(simulate(startup).equity_value, 1457054)
    assert s["mean"] == pytest.approx(1674382, abs=1)
    assert s["p5"] == pytest.approx(304751, abs=1)
    assert s["p95"] == pytest.approx(3757977, abs=1)


def test_distributions():
    rng = np.random.default_rng(0)
    n = draw(MCVariable("wacc", "normal", {"mean": "base", "sd": 0.025}, (0.15, 0.35)), 200_000, rng, base=0.24)
    assert n.min() >= 0.15
    assert n.max() <= 0.35
    assert n.mean() == pytest.approx(0.24, abs=1e-3)
    t = draw(MCVariable("terminal_growth", "triangular", {"min": 0.01, "mode": 0.02, "max": 0.03}), 200_000, rng)
    assert t.min() >= 0.01
    assert t.max() <= 0.03
    assert t.mean() == pytest.approx(0.02, abs=1e-3)
    u = draw(MCVariable("terminal_growth", "uniform", {"min": 0.0, "max": 1.0}), 200_000, rng)
    assert u.mean() == pytest.approx(0.5, abs=5e-3)


def test_revenue_deviation_compounds(startup):
    """A deviation of -10 % leaves 2031 revenue 41 % below the plan (thesis Section 4.7)."""
    p = project(startup, revenue_deviation=-0.10)
    ratio = p.revenue[0] / startup.projection.revenue
    np.testing.assert_allclose(ratio, 0.9 ** np.arange(1, 6))
    assert 1 - ratio[-1] == pytest.approx(0.41, abs=0.005)


def test_simulation_without_operating_shocks(startup):
    only_wacc = [v for v in startup.monte_carlo.variables if v.name == "wacc"]
    cfg = dataclasses.replace(startup, monte_carlo=dataclasses.replace(startup.monte_carlo, variables=only_wacc))
    sim = simulate(cfg, runs=100, rng=1)
    assert sim.equity_value.shape == (100,)
    assert len(np.unique(sim.revenue_last_year)) == 1


def test_independent_seed_is_close_to_thesis(company_b):
    """Company B on its own seed (not the shared thesis stream) agrees within sampling error."""
    base = base_case(company_b).equity_value
    s = summarize(simulate(company_b, rng=2024).equity_value, base)
    assert s["cv"] == pytest.approx(0.2546, abs=0.01)
    assert s["p5"] / base == pytest.approx(0.5704, rel=0.03)
    assert s["p95"] / base == pytest.approx(1.4004, rel=0.03)
