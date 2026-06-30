import json
from pathlib import Path


BASE = Path(r"D:\feature-selection-experiments\notebooks")

RESOLVER = '''from pathlib import Path
import os


def resolve_data_path(filename: str) -> str:
    """
    Resolve data files in local Jupyter or Google Colab.

    Colab cannot access Windows paths such as D:\\... directly. Upload the data
    folder to Colab, or put it in Google Drive and mount Drive first.
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
        "If you are running in Colab, upload the data folder to /content/data, "
        "or mount Google Drive and place it at "
        "/content/drive/MyDrive/feature-selection-experiments/data."
    )


DATA_DIR = Path(resolve_data_path("X_train.csv")).parent
print(f"Using DATA_DIR: {DATA_DIR}")
'''


def cell_source(cell):
    return "".join(cell.get("source", []))


def is_old_resolver(cell):
    source = cell_source(cell)
    return (
        "def resolve_data_path(filename: str) -> str:" in source
        or "def find_project_root(start: Path) -> Path:" in source
    )


def main():
    for path in sorted(BASE.glob("*_with_metrics.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = [cell for cell in notebook.get("cells", []) if not is_old_resolver(cell)]

        resolver_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": RESOLVER.splitlines(keepends=True),
        }
        notebook["cells"] = [resolver_cell] + cells
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"updated {path.name}")


if __name__ == "__main__":
    main()
