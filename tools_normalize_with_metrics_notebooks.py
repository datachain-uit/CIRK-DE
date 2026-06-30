import json
from pathlib import Path


BASE = Path(r"D:\feature-selection-experiments\notebooks")

MOUNT_CELL = '''try:
    from google.colab import drive
    drive.mount("/content/drive")
except ModuleNotFoundError:
    print("Not running in Google Colab; skipping Drive mount.")
'''

RESOLVER_CELL = '''from pathlib import Path
import os


def resolve_data_path(filename: str) -> str:
    """
    Resolve data files in local Jupyter or Google Colab.
    In Colab, mount Google Drive first and place data under:
    /content/drive/MyDrive/feature-selection-experiments/data
    """
    candidates = []

    env_data_dir = os.environ.get("FEATURE_SELECTION_DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir) / filename)

    cwd = Path.cwd()
    candidates.extend([
        cwd / "data" / filename,
        cwd.parent / "data" / filename,
        Path("/content/data") / filename,
        Path("/content/feature-selection-experiments/data") / filename,
        Path("/content/drive/MyDrive/feature-selection-experiments/data") / filename,
        Path("/content/drive/MyDrive/data") / filename,
        Path(r"D:\\feature-selection-experiments\\data") / filename,
    ])

    for path in candidates:
        if path.exists():
            return str(path)

    search_roots = [
        Path("/content"),
        Path("/content/drive/MyDrive"),
    ]
    for root in search_roots:
        if not root.exists():
            continue
        matches = list(root.glob(f"**/{filename}"))
        for match in matches:
            if match.parent.name == "data":
                return str(match)
        if matches:
            return str(matches[0])

    checked = "\\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find {filename}. Checked:\\n{checked}\\n\\n"
        "Upload the data folder to /content/data, or mount Google Drive and place it at "
        "/content/drive/MyDrive/feature-selection-experiments/data."
    )


DATA_DIR = Path(resolve_data_path("X_train.csv")).parent
print(f"Using DATA_DIR: {DATA_DIR}")
'''

DATA_REPLACEMENTS = {
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/X_train.csv"': 'resolve_data_path("X_train.csv")',
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/X_val.csv"': 'resolve_data_path("X_val.csv")',
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/X_test.csv"': 'resolve_data_path("X_test.csv")',
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/y_train.csv"': 'resolve_data_path("y_train.csv")',
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/y_val.csv"': 'resolve_data_path("y_val.csv")',
    '"/kaggle/input/datasets/lfreedom2750/data-time-complexity/y_test.csv"': 'resolve_data_path("y_test.csv")',
    "'/kaggle/input/datasets/lfreedom2750/data-time-complexity/X_train.csv'": 'resolve_data_path("X_train.csv")',
    "'/kaggle/input/datasets/lfreedom2750/data-time-complexity/X_val.csv'": 'resolve_data_path("X_val.csv")',
    "'/kaggle/input/datasets/lfreedom2750/data-time-complexity/X_test.csv'": 'resolve_data_path("X_test.csv")',
    "'/kaggle/input/datasets/lfreedom2750/data-time-complexity/y_train.csv'": 'resolve_data_path("y_train.csv")',
    "'/kaggle/input/datasets/lfreedom2750/data-time-complexity/y_val.csv'": 'resolve_data_path("y_val.csv")',
    "'/kaggle/input/datasets/lfreedom2750/data-time-complexity/y_test.csv'": 'resolve_data_path("y_test.csv")',
    '"D:/feature-selection-experiments/data/X_train.csv"': 'resolve_data_path("X_train.csv")',
    '"D:/feature-selection-experiments/data/X_val.csv"': 'resolve_data_path("X_val.csv")',
    '"D:/feature-selection-experiments/data/X_test.csv"': 'resolve_data_path("X_test.csv")',
    '"D:/feature-selection-experiments/data/y_train.csv"': 'resolve_data_path("y_train.csv")',
    '"D:/feature-selection-experiments/data/y_val.csv"': 'resolve_data_path("y_val.csv")',
    '"D:/feature-selection-experiments/data/y_test.csv"': 'resolve_data_path("y_test.csv")',
}

OTHER_REPLACEMENTS = {
    '"/kaggle/working/selected_indices.csv"': '"selected_indices.csv"',
    "'/kaggle/working/selected_indices.csv'": '"selected_indices.csv"',
    '"/kaggle/input/datasets/lfreedom2750/selected-features/selected_features_bmo.csv"': '"selected_features_bmo.csv"',
    "'/kaggle/input/datasets/lfreedom2750/selected-features/selected_features_bmo.csv'": '"selected_features_bmo.csv"',
    'SAVE_DIR   = "/kaggle/working"': 'SAVE_DIR   = "/content/working"',
}


def cell_source(cell):
    return "".join(cell.get("source", []))


def is_injected_cell(cell):
    source = cell_source(cell)
    return (
        ("from google.colab import drive" in source and "drive.mount" in source)
        or "def resolve_data_path(filename: str) -> str:" in source
        or "def find_project_root(start: Path) -> Path:" in source
    )


def main():
    for path in sorted(BASE.glob("*_with_metrics.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = [cell for cell in notebook.get("cells", []) if not is_injected_cell(cell)]

        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            source = cell_source(cell)
            for old, new in {**DATA_REPLACEMENTS, **OTHER_REPLACEMENTS}.items():
                source = source.replace(old, new)
            cell["source"] = source.splitlines(keepends=True)

        mount_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": MOUNT_CELL.splitlines(keepends=True),
        }
        resolver_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": RESOLVER_CELL.splitlines(keepends=True),
        }
        notebook["cells"] = [mount_cell, resolver_cell] + cells
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"normalized {path.name}")


if __name__ == "__main__":
    main()
