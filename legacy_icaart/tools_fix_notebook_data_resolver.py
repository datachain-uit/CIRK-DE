import json
from pathlib import Path


BASE = Path(r"D:\feature-selection-experiments\notebooks")

RESOLVER = '''from pathlib import Path


def resolve_data_path(filename: str) -> str:
    candidates = [
        Path.cwd() / "data" / filename,
        Path.cwd().parent / "data" / filename,
        Path(r"D:\\feature-selection-experiments\\data") / filename,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    checked = "\\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked:\\n{checked}")


DATA_DIR = Path(resolve_data_path("X_train.csv")).parent
print(f"Using DATA_DIR: {DATA_DIR}")
'''

OLD_PATH_CELL_MARKERS = (
    "def find_project_root(start: Path) -> Path:",
    "PROJECT_ROOT = find_project_root(Path.cwd().resolve())",
)

REPLACEMENTS = {
    'str(DATA_DIR / "X_train.csv")': 'resolve_data_path("X_train.csv")',
    'str(DATA_DIR / "X_val.csv")': 'resolve_data_path("X_val.csv")',
    'str(DATA_DIR / "X_test.csv")': 'resolve_data_path("X_test.csv")',
    'str(DATA_DIR / "y_train.csv")': 'resolve_data_path("y_train.csv")',
    'str(DATA_DIR / "y_val.csv")': 'resolve_data_path("y_val.csv")',
    'str(DATA_DIR / "y_test.csv")': 'resolve_data_path("y_test.csv")',
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/X_train.csv"': 'resolve_data_path("X_train.csv")',
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/X_val.csv"': 'resolve_data_path("X_val.csv")',
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/X_test.csv"': 'resolve_data_path("X_test.csv")',
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/y_train.csv"': 'resolve_data_path("y_train.csv")',
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/y_val.csv"': 'resolve_data_path("y_val.csv")',
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/y_test.csv"': 'resolve_data_path("y_test.csv")',
    '"D:/feature-selection-experiments/data/X_train.csv"': 'resolve_data_path("X_train.csv")',
    '"D:/feature-selection-experiments/data/X_val.csv"': 'resolve_data_path("X_val.csv")',
    '"D:/feature-selection-experiments/data/X_test.csv"': 'resolve_data_path("X_test.csv")',
    '"D:/feature-selection-experiments/data/y_train.csv"': 'resolve_data_path("y_train.csv")',
    '"D:/feature-selection-experiments/data/y_val.csv"': 'resolve_data_path("y_val.csv")',
    '"D:/feature-selection-experiments/data/y_test.csv"': 'resolve_data_path("y_test.csv")',
}


def cell_source(cell):
    return "".join(cell.get("source", []))


def needs_resolver(source: str) -> bool:
    return any(f'resolve_data_path("{name}")' in source for name in [
        "X_train.csv", "X_val.csv", "X_test.csv",
        "y_train.csv", "y_val.csv", "y_test.csv",
    ])


def main():
    for path in sorted(BASE.glob("*_with_metrics.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        cells = [
            cell for cell in cells
            if not any(marker in cell_source(cell) for marker in OLD_PATH_CELL_MARKERS)
            and "def resolve_data_path(filename: str) -> str:" not in cell_source(cell)
        ]

        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            source = cell_source(cell)
            for old, new in REPLACEMENTS.items():
                source = source.replace(old, new)
            cell["source"] = source.splitlines(keepends=True)

        cells.insert(0, {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": RESOLVER.splitlines(keepends=True),
        })

        resolver_lines = RESOLVER.splitlines(keepends=True)
        for cell in cells[1:]:
            if cell.get("cell_type") != "code":
                continue
            source = cell_source(cell)
            if needs_resolver(source):
                cell["source"] = resolver_lines + ["\n"] + source.splitlines(keepends=True)

        notebook["cells"] = cells
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"fixed {path.name}")


if __name__ == "__main__":
    main()
