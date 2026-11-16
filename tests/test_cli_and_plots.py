from pathlib import Path

import pytest

from dcf_valuation.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("name", ["startup", "company_b"])
def test_cli_with_charts(name, tmp_path, capsys):
    assert main(["--config", str(EXAMPLES / f"{name}.yaml"), "--runs", "2000", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Equity value" in out
    assert "Monte Carlo simulation (2,000 runs)" in out
    for suffix in ("mc_hist", "percentiles", "tornado", "fcf", "heatmap"):
        assert (tmp_path / f"{name}_{suffix}.png").stat().st_size > 10_000


def test_cli_missing_config(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.yaml"), "--no-charts"]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_reproduce(capsys):
    assert main(["--reproduce", "--no-charts"]) == 0
    assert "checks passed" in capsys.readouterr().out
