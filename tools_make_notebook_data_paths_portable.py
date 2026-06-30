import json
from pathlib import Path


BASE = Path(r"D:\feature-selection-experiments\notebooks")

PATH_CELL = '''from pathlib import Path


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "X_train.csv").exists():
            return candidate
    windows_project = Path(r"D:\\feature-selection-experiments")
    if (windows_project / "data" / "X_train.csv").exists():
        return windows_project
    raise FileNotFoundError("Could not find project data directory containing X_train.csv")


PROJECT_ROOT = find_project_root(Path.cwd().resolve())
DATA_DIR = PROJECT_ROOT / "data"
print(f"Using DATA_DIR: {DATA_DIR}")
'''

REPLACEMENTS = {
    '"D:/feature-selection-experiments/data/X_train.csv"': 'str(DATA_DIR / "X_train.csv")',
    '"D:/feature-selection-experiments/data/X_val.csv"': 'str(DATA_DIR / "X_val.csv")',
    '"D:/feature-selection-experiments/data/X_test.csv"': 'str(DATA_DIR / "X_test.csv")',
    '"D:/feature-selection-experiments/data/y_train.csv"': 'str(DATA_DIR / "y_train.csv")',
    '"D:/feature-selection-experiments/data/y_val.csv"': 'str(DATA_DIR / "y_val.csv")',
    '"D:/feature-selection-experiments/data/y_test.csv"': 'str(DATA_DIR / "y_test.csv")',
}


def main():
    for path in sorted(BASE.glob("*_with_metrics.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))

        cells = notebook.get("cells", [])
        cells = [
            cell for cell in cells
            if "def find_project_root(start: Path) -> Path:" not in "".join(cell.get("source", []))
        ]
        cells.insert(0, {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": PATH_CELL.splitlines(keepends=True),
        })
        notebook["cells"] = cells

        paths_changed = False
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            original = source
            for old, new in REPLACEMENTS.items():
                source = source.replace(old, new)
            if source != original:
                cell["source"] = source.splitlines(keepends=True)
                paths_changed = True

        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"updated {path.name} paths_changed={paths_changed}")


if __name__ == "__main__":
    main()
