"""Execute every chapter notebook in place (outputs + images are regenerated).

    python scripts/run_notebooks.py            # all chapters
    python scripts/run_notebooks.py 05 07      # only chapters whose folder starts with 05 / 07
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


def run(nb_path: Path, timeout: int = 1200) -> None:
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(nb_path.parent)}})
    t0 = time.time()
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"  ok  {nb_path.relative_to(ROOT)}  ({time.time() - t0:.0f}s)")


def main(prefixes: list[str]) -> None:
    chapters = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit())
    if prefixes:
        chapters = [c for c in chapters if any(c.name.startswith(p) for p in prefixes)]
    for chapter in chapters:
        for nb_path in sorted(chapter.glob("*.ipynb")):
            run(nb_path)


if __name__ == "__main__":
    main(sys.argv[1:])
