import json
import re
from pathlib import Path


BASE = Path(r"D:\feature-selection-experiments\notebooks")

SLUGS = {
    "bmo-svm-v1_with_metrics.ipynb": "bmo-svm-v1",
    "de-qlearning-spark_with_metrics.ipynb": "de-qlearning-spark",
    "de-qlearning_with_metrics.ipynb": "de-qlearning",
    "deql-spark-diversity_with_metrics.ipynb": "deql-spark-diversity",
    "gwo-pso-knn_with_metrics.ipynb": "gwo-pso-knn",
    "ibabc-cgo_with_metrics.ipynb": "ibabc-cgo",
    "pso-dt_with_metrics.ipynb": "pso-dt",
    "sga-rf-h100_with_metrics.ipynb": "sga-rf-h100",
}

RESULT_BLOCK_RE = re.compile(
    r'DATA_DIR = Path\(resolve_data_path\("X_train\.csv"\)\)\.parent\n'
    r'(?:OUTPUT_DIR = .+\nOUTPUT_DIR\.mkdir\(parents=True, exist_ok=True\)\n)?'
    r'print\(f"Using DATA_DIR: \{DATA_DIR\}"\)\n'
    r'(?:print\(f"Using OUTPUT_DIR: \{OUTPUT_DIR\}"\)\n)?'
)

SIMPLE_OUTPUT_RE = re.compile(
    r'(?P<prefix>\.(?:to_csv|to_excel|to_json)\()(?P<quote>f?["\'])(?P<name>[^"\']+\.(?:csv|xlsx|json))(?P=quote)'
)

SIMPLE_READ_RE = re.compile(
    r'(?P<prefix>pd\.read_csv\()(?P<quote>["\'])(?P<name>(?:selected_[^"\']+|.+_selected[^"\']*)\.csv)(?P=quote)'
)

SIMPLE_OPEN_RE = re.compile(
    r'(?P<prefix>open\()(?P<quote>["\'])(?P<name>[^"\']+\.(?:csv|txt|json|pt|pth|pkl))(?P=quote)'
)


def result_block(slug: str) -> str:
    return f'''DATA_DIR = Path(resolve_data_path("X_train.csv")).parent
if Path("/content/drive/MyDrive").exists():
    RESULT_DIR = Path("/content/drive/MyDrive/feature-selection-experiments/results/{slug}")
else:
    RESULT_DIR = Path.cwd() / "results" / "{slug}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = RESULT_DIR


def output_path(filename: str) -> str:
    return str(RESULT_DIR / filename)


print(f"Using DATA_DIR: {{DATA_DIR}}")
print(f"Using RESULT_DIR: {{RESULT_DIR}}")
'''


def cell_source(cell):
    return "".join(cell.get("source", []))


def route_simple_outputs(source: str) -> str:
    def replace_write(match):
        name = match.group("name")
        if "output_path(" in match.group(0) or "/" in name or "\\" in name:
            return match.group(0)
        return f'{match.group("prefix")}output_path({match.group("quote")}{name}{match.group("quote")})'

    def replace_read(match):
        name = match.group("name")
        if "output_path(" in match.group(0) or "/" in name or "\\" in name:
            return match.group(0)
        return f'{match.group("prefix")}output_path({match.group("quote")}{name}{match.group("quote")})'

    def replace_open(match):
        name = match.group("name")
        if "output_path(" in match.group(0) or "/" in name or "\\" in name:
            return match.group(0)
        return f'{match.group("prefix")}output_path({match.group("quote")}{name}{match.group("quote")})'

    source = SIMPLE_OUTPUT_RE.sub(replace_write, source)
    source = SIMPLE_READ_RE.sub(replace_read, source)
    source = SIMPLE_OPEN_RE.sub(replace_open, source)
    return source


def route_specific_outputs(source: str) -> str:
    replacements = {
        'out_path = str(OUTPUT_DIR / "model_comparison.csv")': 'out_path = output_path("model_comparison.csv")',
        'SAVE_DIR   = "/content/working"': 'SAVE_DIR   = str(RESULT_DIR)',
        'SAVE_DIR   = "/kaggle/working"': 'SAVE_DIR   = str(RESULT_DIR)',
        'output_dir="mamba_search_results"': 'output_dir=str(RESULT_DIR / "mamba_search_results")',
        'output_dir = "mamba_search_results"': 'output_dir = str(RESULT_DIR / "mamba_search_results")',
        'output_dir="mamba_search_results"': 'output_dir=str(RESULT_DIR / "mamba_search_results")',
        'output_dir="mamba_search_results"': 'output_dir=str(RESULT_DIR / "mamba_search_results")',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def normalize_notebook(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    slug = SLUGS[path.name]

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell_source(cell)

        if 'DATA_DIR = Path(resolve_data_path("X_train.csv")).parent' in source:
            source = RESULT_BLOCK_RE.sub(result_block(slug), source)

        source = route_simple_outputs(source)
        source = route_specific_outputs(source)
        cell["source"] = source.splitlines(keepends=True)

    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"routed {path.name} -> results/{slug}")


def main():
    for path in sorted(BASE.glob("*_with_metrics.ipynb")):
        if path.name in SLUGS:
            normalize_notebook(path)


if __name__ == "__main__":
    main()
