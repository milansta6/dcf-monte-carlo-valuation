"""Every key number of the thesis, recomputed by the package.

Model A is compared in euros (deterministic values to the euro, Monte Carlo
statistics exactly, because the shared random stream of the thesis is
replayed). Company B is anonymised and compared through scale-free ratios.
"""

import pytest

from dcf_valuation.thesis import checks


def test_all_thesis_numbers_reproduced(thesis_results):
    failed = [c for c in checks(thesis_results) if not c.passed]
    msg = "\n".join(f"{c.model} {c.quantity} ({c.source}): thesis {c.thesis}, code {c.code}" for c in failed)
    assert not failed, "numbers that do not match the thesis:\n" + msg


@pytest.mark.parametrize("model", ["A", "B"])
def test_every_model_is_checked(thesis_results, model):
    assert sum(c.model == model for c in checks(thesis_results)) >= 25


def test_drivers_of_value(thesis_results):
    """Main findings: revenue drives the start-up, the margin drives the manufacturer, growth destroys value."""
    a, b = thesis_results["startup"], thesis_results["company_b"]
    assert a.tornado[0].name == "revenue_deviation"
    assert b.tornado[0].name == "ebit_margin_shift"
    assert b.spearman["revenue_growth_deviation"] < -0.5
    assert a.stats["cv"] > 2.5 * b.stats["cv"]


def test_scenario_grid_2d(thesis_results):
    """Table 14, 49-scenario grid (WACC 10.7-11.3 %, g 1.4-2.6 %), relative to the base case."""
    b = thesis_results["company_b"]
    s, base = b.scenario_stats, b.base.equity_value
    assert s["median"] == pytest.approx(base)
    assert s["mean"] / base == pytest.approx(1.00296, rel=1e-3)
    assert s["std"] / base == pytest.approx(0.06036, rel=1e-2)
