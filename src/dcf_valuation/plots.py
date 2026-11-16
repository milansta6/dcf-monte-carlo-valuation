"""Matplotlib charts in the style of the thesis figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

from .analysis import TornadoBar

# Palette: categorical slots 1-2 plus a sequential blue ramp; ink tokens for text.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
}
WIDTH = 7.0


def money(x: float, _pos: object = None) -> str:
    """Compact money format: 1.46M, 940k."""
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def _save(fig: plt.Figure, out: Path | None, name: str) -> plt.Figure:
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / name)
        plt.close(fig)
    return fig


def histogram(
    values: np.ndarray,
    base: float,
    title: str,
    out: Path | None = None,
    name: str = "hist.png",
    clip_pct: tuple[float, float] = (0.2, 99.6),
    bins: int = 120,
) -> plt.Figure:
    """Distribution of simulated equity value with base case, median and 90 % interval."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
        lo, hi = np.percentile(values, clip_pct)
        ax.hist(values, bins=bins, range=(lo, hi), color=BLUE, edgecolor=SURFACE, linewidth=0.3)
        p5, p50, p95 = np.percentile(values, [5, 50, 95])
        ax.axvspan(p5, p95, color=BLUE, alpha=0.07, lw=0)
        ax.axvline(base, color=ORANGE, lw=2)
        ax.axvline(p50, color=INK, lw=1.2, ls="--")
        for v in (p5, p95):
            ax.axvline(v, color=MUTED, lw=1, ls=":")
        ymax = ax.get_ylim()[1]
        ax.text(p5, ymax * 0.97, f" P5\n {money(p5)}", va="top", ha="left", fontsize=7.5, color=INK2)
        ax.text(p95, ymax * 0.97, f" P95\n {money(p95)}", va="top", ha="left", fontsize=7.5, color=INK2)
        ax.set_ylim(0, ymax * 1.12)
        sub = (
            f"Base case {money(base)} (orange)  ·  median {money(p50)} (dashed)  ·  mean {money(values.mean())}"
            f"  ·  shaded: 90% interval"
        )
        ax.text(0, 1.02, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=INK2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(money))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: f"{x:,.0f}"))
        ax.set_xlabel("Equity value")
        ax.set_ylabel("Number of runs")
        ax.set_title(title, pad=18)
        ax.grid(axis="x", visible=False)
        return _save(fig, out, name)


def percentiles(
    values: np.ndarray, base: float, title: str, out: Path | None = None, name: str = "percentiles.png"
) -> plt.Figure:
    """Percentile ladder: P5...P95 of simulated equity value against the base case."""
    ps = [5, 10, 25, 50, 75, 90, 95]
    q = np.percentile(values, ps)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
        y = np.arange(len(ps))
        ax.hlines(y, 0, q, color=GRID, lw=1)
        ax.scatter(q, y, s=46, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        for yi, v in zip(y, q, strict=True):
            ax.text(v, yi + 0.28, money(v), ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.axvline(base, color=ORANGE, lw=2, label=f"Base case {money(base)}")
        ax.set_yticks(y, [f"P{p}" for p in ps])
        ax.set_ylim(-0.6, len(ps) - 0.1)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(money))
        ax.set_xlabel("Equity value")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        return _save(fig, out, name)


def heatmap(
    grid: np.ndarray,
    waccs: list[float],
    growths: list[float],
    title: str,
    out: Path | None = None,
    name: str = "heatmap.png",
    base: tuple[float, float] | None = None,
    scale: float = 1e6,
    fmt: str = "{:.2f}M",
) -> plt.Figure:
    """Equity value for combinations of WACC (rows) and terminal growth (columns)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH, 0.42 * len(waccs) + 1.4))
        m = grid / scale
        ax.imshow(m, cmap=SEQ, aspect="auto")
        ax.set_xticks(range(len(growths)), [f"{g:.1%}" for g in growths])
        ax.set_yticks(range(len(waccs)), [f"{w:.1%}" for w in waccs])
        ax.set_xlabel("Terminal growth rate g")
        ax.set_ylabel("Discount rate (WACC)")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        rng = (m.max() - m.min()) or 1.0
        for i in range(len(waccs)):
            for j in range(len(growths)):
                dark = (m[i, j] - m.min()) / rng > 0.5
                is_base = base is not None and np.isclose(waccs[i], base[0]) and np.isclose(growths[j], base[1])
                ax.text(
                    j,
                    i,
                    fmt.format(m[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if dark else INK,
                    fontweight="bold" if is_base else "normal",
                )
                if is_base:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2))
        ax.set_title(title)
        return _save(fig, out, name)


def tornado_chart(
    bars: list[TornadoBar], base: float, title: str, out: Path | None = None, name: str = "tornado.png"
) -> plt.Figure:
    """One-at-a-time sensitivity: each input at its 10th (orange) and 90th (blue) percentile."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH, 0.55 * len(bars) + 1.3))
        y = np.arange(len(bars))[::-1]
        for yi, b in zip(y, bars, strict=True):
            ax.barh(
                yi,
                b.value_at_low - base,
                left=base,
                color=ORANGE,
                height=0.56,
                label="Input at its 10th percentile" if yi == y[0] else None,
            )
            ax.barh(
                yi,
                b.value_at_high - base,
                left=base,
                color=BLUE,
                height=0.56,
                label="Input at its 90th percentile" if yi == y[0] else None,
            )
            lo, hi = sorted((b.value_at_low, b.value_at_high))
            ax.text(lo, yi, money(lo) + " ", ha="right", va="center", fontsize=7.5, color=INK2)
            ax.text(hi, yi, " " + money(hi), ha="left", va="center", fontsize=7.5, color=INK2)
        ax.axvline(base, color=INK, lw=1)
        ax.set_yticks(y, [b.label for b in bars])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(money))
        ax.set_xlabel(f"Equity value (base case {money(base)})")
        ax.margins(x=0.16)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32 if len(bars) < 4 else -0.22), ncol=2, fontsize=8)
        ax.set_title(title)
        return _save(fig, out, name)


def fcf_chart(
    years: list[int], fcf: np.ndarray, pv: np.ndarray, title: str, out: Path | None = None, name: str = "fcf.png"
) -> plt.Figure:
    """Free cash flow to the firm and its present value by year."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
        x = np.arange(len(years))
        ax.bar(x - 0.2, fcf, 0.38, color=BLUE, label="FCFF")
        ax.bar(x + 0.2, pv, 0.38, color=ORANGE, label="Present value")
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_xticks(x, [str(y) for y in years])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(money))
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        return _save(fig, out, name)


def compare_cdf(
    series: dict[str, tuple[np.ndarray, float]], out: Path | None = None, name: str = "compare_cdf.png"
) -> plt.Figure:
    """Cumulative distributions of simulated value / base-case value for several models (thesis Figure 15)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH, 3.2))
        for (label, (values, base)), color in zip(series.items(), (BLUE, ORANGE), strict=False):
            r = np.sort(values / base)
            ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=color, lw=2, label=label)
        ax.axvline(1, color=INK, lw=0.8, ls="--")
        ax.axhspan(0.05, 0.95, color=GRID, alpha=0.35, lw=0)
        ax.set_xlim(-0.25, 3)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.set_xlabel("Simulated equity value / base-case equity value")
        ax.set_ylabel("Cumulative probability")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Relative uncertainty of the two valuations")
        return _save(fig, out, name)


def all_charts(run, out: Path, prefix: str) -> list[Path]:
    """Save the standard set of charts for one :class:`~dcf_valuation.thesis.ModelRun`."""
    cfg, base = run.config, run.base.equity_value
    eq = run.simulation.equity_value
    label = cfg.name.split(" - ")[0] if " - " in cfg.name else cfg.name
    n = f"{run.simulation.runs:,}"
    histogram(eq, base, f"{label}: Monte Carlo distribution of equity value ({n} runs)", out, f"{prefix}_mc_hist.png")
    percentiles(eq, base, f"{label}: percentiles of simulated equity value", out, f"{prefix}_percentiles.png")
    tornado_chart(
        run.tornado, base, f"{label}: one-at-a-time sensitivity (P10-P90 of each input)", out, f"{prefix}_tornado.png"
    )
    pv = run.base.valuation.pv_by_year[0]
    fcf_chart(
        cfg.projection.years,
        run.base.projection.fcf[0],
        pv,
        f"{label}: free cash flow and present value",
        out,
        f"{prefix}_fcf.png",
    )
    names = [f"{prefix}_mc_hist.png", f"{prefix}_percentiles.png", f"{prefix}_tornado.png", f"{prefix}_fcf.png"]
    if run.grid.size:
        s = cfg.sensitivity
        heatmap(
            run.grid,
            s.wacc,
            s.terminal_growth,
            f"{label}: equity value by WACC and terminal growth",
            out,
            f"{prefix}_heatmap.png",
            base=(run.base.wacc.wacc, cfg.terminal.growth),
        )
        names.append(f"{prefix}_heatmap.png")
    return [out / x for x in names]
