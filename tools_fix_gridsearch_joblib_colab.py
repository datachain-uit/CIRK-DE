import json
from pathlib import Path


BASE = Path(r"D:\feature-selection-experiments\notebooks")

TARGETS = list(BASE.glob("*_with_metrics.ipynb")) + [
    BASE / "ml-baseline-no-feature-selection.ipynb"
]


def main():
    for path in TARGETS:
        if not path.exists():
            continue

        notebook = json.loads(path.read_text(encoding="utf-8"))
        changed = False

        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue

            source = "".join(cell.get("source", []))
            original = source

            # Colab can get BrokenProcessPool / numpy._core errors with loky
            # after package changes. Keep GridSearchCV single-process.
            source = source.replace("n_jobs=-1", "n_jobs=1")
            source = source.replace("n_jobs = -1", "n_jobs = 1")

            # Some notebooks use LabelEncoder in load_data but did not import it.
            if (
                "from sklearn.model_selection import train_test_split" in source
                and "LabelEncoder" not in source
            ):
                source = source.replace(
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\n",
                )

            if source != original:
                cell["source"] = source.splitlines(keepends=True)
                changed = True

        if changed:
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"fixed {path.name}")


if __name__ == "__main__":
    main()
