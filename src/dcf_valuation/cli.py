"""Command line interface.

python -m dcf_valuation --config examples/startup.yaml
python -m dcf_valuation --config examples/startup.yaml --runs 20000 --seed 1 --out output
python -m dcf_valuation --reproduce            # compare every key number with the thesis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import LABELS
from .config import ConfigError, load_config
from .thesis import ModelRun, checks, reproduce, run_model


def _m(x: float) -> str:
    return f"{x:>15,.0f}"


def report(run: ModelRun) -> str:
    """Plain-text report of one valuation."""
    cfg, bc = run.config, run.base
    w, v, p = bc.wacc, bc.valuation, bc.projection
    cur = cfg.currency
    lines = [f"\n{cfg.name}", "=" * len(cfg.name), ""]
    lines += [
        "Cost of capital",
        f"  Levered beta                 {w.levered_beta:8.3f}",
        f"  Cost of equity               {w.cost_of_equity:8.2%}",
        f"  After-tax cost of debt       {w.after_tax_cost_of_debt:8.2%}   weight of debt {w.weight_debt:.1%}",
        f"  WACC (computed)              {w.wacc_computed:8.2%}",
    ]
    if w.is_overridden:
        lines.append(f"  WACC used in the model       {w.wacc:8.2%}   (override from the config)")
    lines += [
        "",
        f"Free cash flow to the firm ({cur})",
        f"  {'':<12}" + "".join(f"{y:>13}" for y in cfg.projection.years),
    ]
    for name, row in (
        ("Revenue", p.revenue[0]),
        ("EBIT", p.ebit[0]),
        ("NOPAT", p.nopat[0]),
        ("FCFF", p.fcf[0]),
        ("PV of FCFF", v.pv_by_year[0]),
    ):
        lines.append(f"  {name:<12}" + "".join(f"{x:>13,.0f}" for x in row))
    lines += [
        "",
        f"Base-case valuation ({cur})",
        f"  PV of explicit FCFF       {_m(v.pv_fcf[0])}",
        f"  Terminal value            {_m(v.terminal_value[0])}   (discount exponent {v.terminal_exponent})",
        f"  PV of terminal value      {_m(v.pv_terminal_value[0])}   ({v.tv_share[0]:.1%} of EV)",
        f"  Enterprise value          {_m(v.enterprise_value[0])}",
    ]
    lines += [f"  {'+ ' if a >= 0 else '- '}{k.replace('_', ' '):<24}{_m(abs(a))}" for k, a in cfg.bridge.items() if a]
    lines += [f"  Equity value              {_m(bc.equity_value)}", ""]
    s = run.stats
    lines += [
        f"Monte Carlo simulation ({run.simulation.runs:,} runs)",
        f"  Mean                      {_m(s['mean'])}",
        f"  Median                    {_m(s['p50'])}",
        f"  Standard deviation        {_m(s['std'])}   (coefficient of variation {s['cv']:.2f})",
        f"  5th - 95th percentile     {_m(s['p5'])} - {s['p95']:,.0f}",
        f"  25th - 75th percentile    {_m(s['p25'])} - {s['p75']:,.0f}",
        f"  P(value < 0)              {s['p_negative']:15.1%}",
        f"  P(value < base case)      {s['p_below_base']:15.1%}",
    ]
    for t in cfg.monte_carlo.thresholds:
        side, key = (">", "p_above") if t > bc.equity_value else ("<", "p_below")
        label = f"P(value {side} {t:,.0f})"
        lines.append(f"  {label:<26}{s[f'{key}_{t:.0f}']:15.1%}")
    lines += ["", "Drivers of value (Spearman rank correlation with equity value)"]
    for k, r in sorted(run.spearman.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"  {LABELS[k]:<32}{r:+.2f}")
    lines += ["", "Tornado (each input at its 10th / 90th percentile, others at base)"]
    for b in run.tornado:
        lines.append(f"  {b.label:<32}{b.value_at_low:>13,.0f} {b.value_at_high:>13,.0f}")
    return "\n".join(lines)


def check_report(results: dict[str, ModelRun], exact_stream: bool = True) -> tuple[str, bool]:
    rows = checks(results, exact_stream=exact_stream)
    lines = [f"{'':5}{'Model':<6}{'Quantity':<48}{'Source':<24}{'Thesis':>14}{'Code':>16}"]
    for c in rows:
        fmt = (lambda x: f"{x:,.0f}") if abs(c.thesis) >= 100 else (lambda x: f"{x:.4f}")
        lines.append(
            f"{'OK' if c.passed else 'FAIL':<5}{c.model:<6}{c.quantity:<48}{c.source:<24}"
            f"{fmt(c.thesis):>14}{fmt(c.code):>16}"
        )
    n_ok = sum(c.passed for c in rows)
    lines.append(f"\n{n_ok} of {len(rows)} checks passed.")
    lines.append("Company B is compared through scale-free ratios because its amounts are anonymised.")
    return "\n".join(lines), n_ok == len(rows)


def main(argv: list[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")  # files only, no windows
    ap = argparse.ArgumentParser(
        prog="python -m dcf_valuation", description="DCF valuation with Monte Carlo simulation"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", type=Path, help="YAML configuration of the company to value")
    g.add_argument(
        "--reproduce", action="store_true", help="reproduce both thesis examples and compare with the thesis"
    )
    ap.add_argument("--runs", type=int, help="number of Monte Carlo runs (default: from the config)")
    ap.add_argument("--seed", type=int, help="random seed (default: from the config)")
    ap.add_argument("--out", type=Path, default=Path("output"), help="folder for charts (default: output/)")
    ap.add_argument("--no-charts", action="store_true", help="do not save charts")
    args = ap.parse_args(argv)

    if args.reproduce:
        results = reproduce(runs=args.runs)
        for run in results.values():
            print(report(run))
        text, ok = check_report(results, exact_stream=args.runs is None)
        print("\n\nComparison with the thesis\n" + text)
        if not args.no_charts:
            from . import plots

            for name, run in results.items():
                plots.all_charts(run, args.out, name)
            plots.compare_cdf(
                {
                    "Model A: start-up": (
                        results["startup"].simulation.equity_value,
                        results["startup"].base.equity_value,
                    ),
                    "Model B: manufacturer": (
                        results["company_b"].simulation.equity_value,
                        results["company_b"].base.equity_value,
                    ),
                },
                args.out,
                "compare_cdf.png",
            )
            print(f"\nCharts saved to {args.out}/")
        return 0 if ok else 1

    try:
        cfg = load_config(args.config)
    except (OSError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    run = run_model(cfg, rng=args.seed, runs=args.runs)
    print(report(run))
    if not args.no_charts:
        from . import plots

        paths = plots.all_charts(run, args.out, cfg.short_name)
        print(f"\nCharts saved to {args.out}/: " + ", ".join(p.name for p in paths))
    return 0
