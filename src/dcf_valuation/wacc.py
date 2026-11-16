"""Cost of capital: CAPM build-up, Hamada relevering and the WACC (thesis Chapter 2)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CostOfCapitalInputs


def levered_beta(unlevered_beta: float, debt_to_equity: float, tax_rate: float) -> float:
    """Hamada: beta_L = beta_U x [1 + (1 - T) x D/E]  (thesis eq. 6)."""
    return unlevered_beta * (1.0 + (1.0 - tax_rate) * debt_to_equity)


def cost_of_equity(
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
    country_risk_premium: float,
    specific_risk_premium: float = 0.0,
) -> float:
    """Build-up CAPM: Ce = Rf + beta_L x (ERP + CRP) + SCP  (thesis eq. 5)."""
    return risk_free_rate + beta * (equity_risk_premium + country_risk_premium) + specific_risk_premium


def after_tax_cost_of_debt(pre_tax_rate: float, tax_rate: float) -> float:
    """Cd = Rd x (1 - T)."""
    return pre_tax_rate * (1.0 - tax_rate)


def weights_from_debt_to_equity(debt_to_equity: float) -> tuple[float, float]:
    """Return (E/V, D/V) implied by a debt-to-equity ratio."""
    wd = debt_to_equity / (1.0 + debt_to_equity)
    return 1.0 - wd, wd


@dataclass(frozen=True)
class WaccResult:
    levered_beta: float
    cost_of_equity: float
    after_tax_cost_of_debt: float
    weight_equity: float
    weight_debt: float
    wacc_computed: float
    wacc: float  # the rate actually used (equals wacc_computed unless overridden)

    @property
    def is_overridden(self) -> bool:
        return self.wacc != self.wacc_computed


def compute_wacc(inputs: CostOfCapitalInputs, tax_rate: float) -> WaccResult:
    """WACC = E/V x Ce + D/V x Rd (1 - T)  (thesis eq. 7)."""
    beta_l = levered_beta(inputs.unlevered_beta, inputs.debt_to_equity, tax_rate)
    ce = cost_of_equity(
        inputs.risk_free_rate,
        beta_l,
        inputs.equity_risk_premium,
        inputs.country_risk_premium,
        inputs.specific_risk_premium,
    )
    cd = after_tax_cost_of_debt(inputs.pre_tax_cost_of_debt, tax_rate)
    we, wd = weights_from_debt_to_equity(inputs.debt_to_equity)
    computed = we * ce + wd * cd
    used = computed if inputs.wacc_override is None else float(inputs.wacc_override)
    return WaccResult(beta_l, ce, cd, we, wd, computed, used)
