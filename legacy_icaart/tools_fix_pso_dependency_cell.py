import json
from pathlib import Path


NOTEBOOK = Path(r"D:\feature-selection-experiments\notebooks\pso-dt_with_metrics.ipynb")

DEPENDENCY_CELL = '''import importlib
import subprocess
import sys


def ensure_package(import_name: str, pip_name: str | None = None) -> None:
    try:
        importlib.import_module(import_name)
    except Exception:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            pip_name or import_name,
        ])


# Do not uninstall/re-pin numpy or pandas in Colab; that can break scipy/sklearn.
ensure_package("mealpy")

try:
    import numpy
    import pandas
    import scipy
    import sklearn
    print("Dependency check OK")
    print("numpy:", numpy.__version__)
    print("pandas:", pandas.__version__)
    print("scipy:", scipy.__version__)
    print("sklearn:", sklearn.__version__)
except Exception as exc:
    print("sklearn/scipy stack looks broken:", repr(exc))
    print("Repairing numpy/pandas/scipy/scikit-learn stack. Restart runtime after this cell if import still fails.")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "threadpoolctl",
        "mealpy",
    ])
'''


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    # Replace old mealpy install cell.
    cells[2]["source"] = DEPENDENCY_CELL.splitlines(keepends=True)

    # Remove the old numpy/pandas uninstall/reinstall cell if present.
    cells[:] = [
        cell for cell in cells
        if "!pip uninstall -y numpy pandas" not in "".join(cell.get("source", []))
    ]

    # Add the missing LabelEncoder import.
    for cell in cells:
        source = "".join(cell.get("source", []))
        if "from sklearn.model_selection import train_test_split" in source and "LabelEncoder" not in source:
            source = source.replace(
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\n",
            )
            cell["source"] = source.splitlines(keepends=True)
            break

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"fixed {NOTEBOOK}")


if __name__ == "__main__":
    main()
