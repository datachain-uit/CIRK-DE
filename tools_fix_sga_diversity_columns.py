import json
from pathlib import Path


NOTEBOOK = Path(r"D:\feature-selection-experiments\notebooks\sga-rf-h100_with_metrics.ipynb")


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if "def save_sga_history(model):" not in source:
            continue

        source = source.replace(
            """    exploration_mealpy = align_history(model.history.list_exploration, n_epochs)
    exploitation_mealpy = align_history(model.history.list_exploitation, n_epochs)
""",
            """    diversity_mealpy = align_history(model.history.list_diversity, n_epochs)
    exploration_mealpy = align_history(model.history.list_exploration, n_epochs)
    exploitation_mealpy = align_history(model.history.list_exploitation, n_epochs)
""",
        )

        source = source.replace(
            """        "diversity": diversity_formula,
        "exploration": exploration_mealpy,
        "exploitation": exploitation_mealpy,
        "exploration_source": ["mealpy"] * n_epochs,
        "exploitation_source": ["mealpy"] * n_epochs,
        "diversity_source": ["pairwise_hamming"] * n_epochs,
""",
            """        "diversity": diversity_mealpy,
        "diversity_binary_hamming": diversity_formula,
        "exploration": exploration_mealpy,
        "exploitation": exploitation_mealpy,
        "diversity_source": ["mealpy"] * n_epochs,
        "diversity_binary_hamming_source": ["pairwise_hamming_on_feature_masks"] * n_epochs,
        "exploration_source": ["mealpy"] * n_epochs,
        "exploitation_source": ["mealpy"] * n_epochs,
""",
        )

        cell["source"] = source.splitlines(keepends=True)
        break

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"fixed {NOTEBOOK}")


if __name__ == "__main__":
    main()
