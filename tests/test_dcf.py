import dataclasses

import numpy as np
import pytest

from dcf_valuation import base_case, discount, project, wacc_table


def test_mid_year_discounting_and_gordon(startup):
    """Hand-computed example: three flows of 100, WACC 10 %, g 2 %."""
    cfg = dataclasses.replace(
        startup,
        terminal=dataclasses.replace(startup.terminal, fcf_growth=None, fcf_override=None, timing="same_as_last_year"),
        bridge={},
    )
    v = discount(cfg, np.array([100.0, 100.0, 100.0]), 0.10, 0.02)
    pv = sum(100 / 1.1 ** (t - 0.5) for t in (1, 2, 3))
    tv = 100 * 1.02 / 0.08
    assert v.pv_fcf[0] == pytest.approx(pv)
    assert v.terminal_value[0] == pytest.approx(tv)
    assert v.pv_terminal_value[0] == pytest.approx(tv / 1.1**2.5)
    assert v.equity_value[0] == pytest.approx(pv + tv / 1.1**2.5)


def test_terminal_value_timing_conventions(startup):
    """Model A discounts the TV one year further than Model B (thesis Sections 4.5 and 5.4)."""
    later = base_case(startup)
    same_cfg = dataclasses.replace(startup, terminal=dataclasses.replace(startup.terminal, timing="same_as_last_year"))
    same = base_case(same_cfg)
    assert later.valuation.terminal_exponent == 5.5
    assert same.valuation.terminal_exponent == 4.5
    ratio = same.valuation.pv_terminal_value[0] / later.valuation.pv_terminal_value[0]
    assert ratio == pytest.approx(1.241099)
    # thesis: "with an exponent of 4.5 ... the equity value EUR 1.68 million instead of EUR 1.46 million"
    assert same.equity_value == pytest.approx(1.68e6, abs=5e3)


def test_startup_base_case_matches_table_6(startup):
    bc = base_case(startup)
    np.testing.assert_allclose(bc.projection.fcf[0], [-35270, -41832, 183688, 378702, 661471], atol=0.5)
    np.testing.assert_allclose(bc.projection.nopat[0], [-37445, 40885, 140849, 330755, 600777], atol=0.5)
    assert bc.valuation.pv_fcf[0] == pytest.approx(473202, abs=1)
    assert bc.valuation.terminal_value[0] == pytest.approx(3084835, abs=1)
    assert bc.enterprise_value == pytest.approx(1413568, abs=1)
    assert bc.equity_value == pytest.approx(1457054, abs=1)


def test_startup_terminal_fcf_rule(startup):
    """Simulation convention FCF_2032 = 1.03 x FCF_2031 changes the base by < 0.1 % (thesis Section 3.4)."""
    exact = base_case(startup).equity_value
    cfg = dataclasses.replace(startup, terminal=dataclasses.replace(startup.terminal, fcf_override=None))
    rule = base_case(cfg).equity_value
    assert rule == pytest.approx(1456035, abs=1)
    assert abs(rule / exact - 1) < 0.001


def test_wacc_table_matches_table_7(startup):
    rows = wacc_table(startup, startup.sensitivity.wacc_offsets)
    table_7_ev = [2748311, 2275307, 1916382, 1636542, 1413568, 1232705, 1083789, 959605, 854900]
    np.testing.assert_allclose([r["ev"] for r in rows], table_7_ev, atol=1)  # spreadsheet FCFF in whole euros
    assert [round(r["tv_share"], 3) for r in rows][0::4] == [0.773, 0.665, 0.573]


def test_company_b_working_capital_levels(company_b):
    p = project(company_b)
    levels = company_b.projection.nwc_levels
    expected = np.diff(np.concatenate([[company_b.projection.opening_nwc], levels]))
    np.testing.assert_allclose(p.nwc_investment[0], expected)
    # depreciation equals capex, so FCFF = NOPAT - increase in NWC
    np.testing.assert_allclose(p.fcf[0], p.nopat[0] - expected)
    assert p.fcf[0][0] < 0  # the 2025 working capital normalisation makes the first FCFF negative


def test_company_b_terminal_value_exceeds_ev(company_b):
    bc = base_case(company_b)
    assert bc.valuation.tv_share[0] == pytest.approx(1.557, rel=0.01)  # thesis: 156 %
    assert bc.valuation.pv_fcf[0] < 0


def test_shocks_are_neutral_at_zero(startup, company_b):
    for cfg, shocks in (
        (startup, {"revenue_deviation": 0.0, "fixed_cost_deviation": 0.0}),
        (company_b, {"revenue_growth_deviation": 0.0, "ebit_margin_shift": 0.0}),
    ):
        np.testing.assert_allclose(project(cfg, **shocks).fcf, project(cfg).fcf, rtol=1e-12, atol=1e-6)


def test_wacc_below_growth_is_rejected(startup):
    with pytest.raises(ValueError):
        discount(startup, np.ones(5), 0.02, 0.03)
