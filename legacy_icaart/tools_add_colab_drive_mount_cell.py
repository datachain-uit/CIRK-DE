import json
from pathlib import Path


BASE = Path(r"D:\feature-selection-experiments\notebooks")

MOUNT_CELL = '''try:
    from google.colab import drive
    drive.mount("/content/drive")
except ModuleNotFoundError:
    print("Not running in Google Colab; skipping Drive mount.")
'''


def cell_source(cell):
    return "".join(cell.get("source", []))


def is_mount_cell(cell):
    source = cell_source(cell)
    return "from google.colab import drive" in source and "drive.mount" in source


def main():
    for path in sorted(BASE.glob("*_with_metrics.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = [cell for cell in notebook.get("cells", []) if not is_mount_cell(cell)]
        cells.insert(0, {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": MOUNT_CELL.splitlines(keepends=True),
        })
        notebook["cells"] = cells
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"updated {path.name}")


if __name__ == "__main__":
    main()
