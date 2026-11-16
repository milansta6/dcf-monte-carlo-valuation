"""Execute every notebook in notebooks/ and fail on the first error.

python scripts/run_notebooks.py            # execute, keep the files unchanged
python scripts/run_notebooks.py --save     # execute and store the outputs in the notebooks
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def main() -> int:
    save = "--save" in sys.argv
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        NotebookClient(
            nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS)}}
        ).execute()
        if save:
            nbformat.write(nb, path)
        print(f"ok  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
