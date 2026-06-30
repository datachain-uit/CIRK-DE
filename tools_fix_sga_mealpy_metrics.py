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
            """def compute_formula_metrics_from_history_populations(
    populations,
    threshold: float = 0.5,
    min_features: int = 1,
    similarity_threshold: float = 0.1
) -> Tuple[List[float], List[float], List[float]]:
    \"\"\"Compute diversity/exploration/exploitation for each epoch from stored populations.\"\"\"
    diversities = []
    explorations = []
    exploitations = []
    previous_masks = None

    for population in populations:
        population_matrix = extract_population_matrix(population)
        current_masks = continuous_population_to_masks(
            population_matrix,
            threshold=threshold,
            min_features=min_features
        )
        diversities.append(compute_population_diversity(current_masks))

        if previous_masks is None:
            explorations.append(0.0)
            exploitations.append(0.0)
        else:
            exploration, exploitation = compute_exploration_exploitation(
                new_masks=current_masks,
                previous_masks=previous_masks,
                similarity_threshold=similarity_threshold
            )
            explorations.append(exploration)
            exploitations.append(exploitation)

        previous_masks = current_masks.copy()

    return diversities, explorations, exploitations

def save_sga_history(model):
    diversity_formula, exploration_formula, exploitation_formula = compute_formula_metrics_from_history_populations(
        model.history.list_population,
        threshold=0.5,
        min_features=1,
        similarity_threshold=0.1
    )
""",
            """def compute_diversity_from_history_populations(
    populations,
    threshold: float = 0.5,
    min_features: int = 1
) -> List[float]:
    \"\"\"Compute only diversity from stored populations using average pairwise Hamming distance.\"\"\"
    diversities = []

    for population in populations:
        population_matrix = extract_population_matrix(population)
        current_masks = continuous_population_to_masks(
            population_matrix,
            threshold=threshold,
            min_features=min_features
        )
        diversities.append(compute_population_diversity(current_masks))

    return diversities


def align_history(values, n: int, fill_value: float = 0.0) -> List[float]:
    values = list(values)
    if len(values) >= n:
        return values[:n]
    return values + [fill_value] * (n - len(values))


def save_sga_history(model):
    n_epochs = len(model.history.list_global_best_fit)
    diversity_formula = align_history(
        compute_diversity_from_history_populations(
            model.history.list_population,
            threshold=0.5,
            min_features=1
        ),
        n_epochs
    )
    exploration_mealpy = align_history(model.history.list_exploration, n_epochs)
    exploitation_mealpy = align_history(model.history.list_exploitation, n_epochs)
""",
        )

        source = source.replace(
            """        "diversity": diversity_formula,
        "exploration": exploration_formula,
        "exploitation": exploitation_formula,
        "exploration_threshold": [0.1] * len(diversity_formula),
        "epoch_time": model.history.list_epoch_time
""",
            """        "diversity": diversity_formula,
        "exploration": exploration_mealpy,
        "exploitation": exploitation_mealpy,
        "exploration_source": ["mealpy"] * n_epochs,
        "exploitation_source": ["mealpy"] * n_epochs,
        "diversity_source": ["pairwise_hamming"] * n_epochs,
        "epoch_time": align_history(model.history.list_epoch_time, n_epochs)
""",
        )

        cell["source"] = source.splitlines(keepends=True)
        break

    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"fixed {NOTEBOOK}")


if __name__ == "__main__":
    main()
