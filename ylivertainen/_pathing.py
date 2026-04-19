"""
_pathing.py — Ylivertainen Pathing Helper
=========================================
Finds the root of the repository and inserts it into the Python path.
"""

from pathlib import Path
import sys

def setup_repo_path() -> Path:
    def find_root(marker: str = "predictive_modeling.py") -> Path:
        p = Path.cwd().resolve()
        for folder in [p, *p.parents]:
            if (folder / marker).is_file():
                return folder
        raise FileNotFoundError(f"Could not find {marker} above {p}")
    root = find_root() # Path to .../ylivertainen
    print("Current location:", root.name)
    print(root)
    repo_root = root.parent  # .../TheLibraryOfCode
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    print("Inserted (times):", sys.path.count(str(root)))
    print("Current location:", repo_root.name)
    return root