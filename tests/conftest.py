from pathlib import Path

import pytest

from dcf_valuation import load_config
from dcf_valuation.thesis import reproduce

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.fixture(scope="session")
def startup():
    return load_config(EXAMPLES / "startup.yaml")


@pytest.fixture(scope="session")
def company_b():
    return load_config(EXAMPLES / "company_b.yaml")


@pytest.fixture(scope="session")
def thesis_results():
    """Both examples on the shared thesis random stream (100,000 runs each)."""
    return reproduce(EXAMPLES)
