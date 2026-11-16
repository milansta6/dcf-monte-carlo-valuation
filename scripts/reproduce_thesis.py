"""Reproduce every key number of the thesis and compare it with the printed value.

    python scripts/reproduce_thesis.py              # exact reproduction (shared random stream)
    python scripts/reproduce_thesis.py --out docs/images

Exits with status 1 if any number does not match.
"""

import sys

from dcf_valuation.cli import main

if __name__ == "__main__":
    sys.exit(main(["--reproduce", *sys.argv[1:]]))
