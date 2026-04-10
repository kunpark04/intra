"""Execute analysis.ipynb for each iteration, injecting os.chdir for correct cwd."""
import nbformat
from nbclient import NotebookClient
from pathlib import Path
import sys


def execute_analysis(version: str) -> None:
    path = Path("iterations") / version / "analysis.ipynb"
    abs_cwd = str(path.parent.resolve())

    with open(path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Inject chdir as first line of cell 0 so kernel finds local CSV/JSON files
    chdir_line = f"import os; os.chdir({abs_cwd!r})\n"
    original_src = nb.cells[0].source
    nb.cells[0].source = chdir_line + original_src

    print(f"Executing {path}  cwd={abs_cwd}")
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    client.execute()

    # Strip injected chdir from saved notebook
    nb.cells[0].source = original_src

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    versions = sys.argv[1:] if len(sys.argv) > 1 else ["V1", "V2"]
    for v in versions:
        execute_analysis(v)
