import json
from pathlib import Path


NOTEBOOK = Path(r"D:\feature-selection-experiments\notebooks\sga-rf-h100_with_metrics.ipynb")


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if "diversity_binary_hamming" not in source:
            continue

        source = source.replace('        "diversity_binary_hamming": diversity_formula,\n', "")
        source = source.replace(
            '        "diversity_binary_hamming_source": ["pairwise_hamming_on_feature_masks"] * n_epochs,\n',
            "",
        )

        cell["source"] = source.splitlines(keepends=True)
        break

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
