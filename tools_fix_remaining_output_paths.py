import json
from pathlib import Path


BASE = Path(r"D:\feature-selection-experiments\notebooks")

OLD_OUTPUT_BLOCK = '''DATA_DIR = Path(resolve_data_path("X_train.csv")).parent
print(f"Using DATA_DIR: {DATA_DIR}")'''

NEW_OUTPUT_BLOCK = '''DATA_DIR = Path(resolve_data_path("X_train.csv")).parent
OUTPUT_DIR = Path("/content/working") if Path("/content").exists() else Path.cwd()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Using DATA_DIR: {DATA_DIR}")
print(f"Using OUTPUT_DIR: {OUTPUT_DIR}")'''


def main():
    for path in sorted(BASE.glob("*_with_metrics.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for cell in notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            new_source = source.replace(OLD_OUTPUT_BLOCK, NEW_OUTPUT_BLOCK)
            new_source = new_source.replace(
                'out_path = "/kaggle/working/model_comparison.csv"',
                'out_path = str(OUTPUT_DIR / "model_comparison.csv")',
            )
            if new_source != source:
                cell["source"] = new_source.splitlines(keepends=True)
                changed = True
        if changed:
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"updated {path.name}")


if __name__ == "__main__":
    main()
