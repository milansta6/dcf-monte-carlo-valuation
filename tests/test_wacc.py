import pytest

from dcf_valuation import compute_wacc, cost_of_equity, levered_beta
from dcf_valuation.wacc import after_tax_cost_of_debt, weights_from_debt_to_equity


def test_hamada_relevering():
    # thesis Table 11: 0.56 x [1 + (1 - 15%) x 0.92] = 1.00
    assert levered_beta(0.56, 0.92, 0.15) == pytest.approx(0.99792)
    assert levered_beta(1.23, 0.0, 0.15) == 1.23


def test_build_up_cost_of_equity():
    # thesis Table 5: 3.11% + 1.23 x (4.23% + 3.90%) + 11% = 24.11%
    assert cost_of_equity(0.0311, 1.23, 0.0423, 0.039, 0.11) == pytest.approx(0.241099)


def test_cost_of_debt_and_weights():
    assert after_tax_cost_of_debt(0.06558, 0.15) == pytest.approx(0.055743)
    we, wd = weights_from_debt_to_equity(0.92212)
    assert wd == pytest.approx(0.4797, abs=1e-4)
    assert we + wd == pytest.approx(1.0)


def test_startup_wacc_equals_cost_of_equity(startup):
    w = compute_wacc(startup.cost_of_capital, startup.tax_rate)
    assert w.weight_debt == 0
    assert w.wacc == pytest.approx(w.cost_of_equity)
    assert w.wacc == pytest.approx(0.241099)
    assert not w.is_overridden


def test_company_b_wacc(company_b):
    # thesis Table 11: Ce 15.93 %, Cd 5.57 %, WACC 10.96 % rounded to 11.0 %
    w = compute_wacc(company_b.cost_of_capital, company_b.tax_rate)
    assert w.levered_beta == pytest.approx(0.99893, abs=1e-5)
    assert w.cost_of_equity == pytest.approx(0.15934, abs=1e-5)
    assert w.after_tax_cost_of_debt == pytest.approx(0.05574, abs=1e-5)
    assert w.wacc_computed == pytest.approx(0.10964, abs=1e-5)
    assert w.wacc == 0.11
    assert w.is_overridden
