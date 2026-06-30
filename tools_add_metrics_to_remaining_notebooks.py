import json
from pathlib import Path


BASE = Path(r"D:\feature-selection-experiments\notebooks")


METRIC_HELPERS = r'''

def hamming_distance_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return the normalized Hamming distance between two binary masks."""
    mask_a = np.asarray(mask_a).astype(int)
    mask_b = np.asarray(mask_b).astype(int)
    if mask_a.shape != mask_b.shape:
        raise ValueError("Two masks must have the same shape.")
    return float(np.mean(mask_a != mask_b))


def continuous_population_to_masks(
    population: np.ndarray,
    threshold: float = 0.5,
    min_features: int = 1,
    use_sigmoid: bool = False
) -> np.ndarray:
    """Convert a continuous population to binary feature masks for metric tracking."""
    values = 1.0 / (1.0 + np.exp(-population)) if use_sigmoid else np.asarray(population)
    masks = (values > threshold).astype(int)

    if min_features > 0:
        for i in range(masks.shape[0]):
            if masks[i].sum() < min_features:
                top_idx = np.argsort(values[i])[-min_features:]
                masks[i, top_idx] = 1

    return masks


def compute_population_diversity(population_masks: np.ndarray) -> float:
    """Average pairwise Hamming distance among all masks in the current population."""
    population_masks = np.asarray(population_masks).astype(int)
    n_individuals = population_masks.shape[0]
    if n_individuals < 2:
        return 0.0

    total_distance = 0.0
    pair_count = 0
    for i in range(n_individuals):
        for j in range(i + 1, n_individuals):
            total_distance += hamming_distance_ratio(population_masks[i], population_masks[j])
            pair_count += 1

    return float(total_distance / pair_count)


def compute_exploration_exploitation(
    new_masks: np.ndarray,
    previous_masks: np.ndarray,
    similarity_threshold: float = 0.1
) -> Tuple[float, float]:
    """
    Exploration is the fraction of new masks whose nearest previous neighbor
    is farther than the threshold. Exploitation is the complementary fraction.
    """
    new_masks = np.asarray(new_masks).astype(int)
    previous_masks = np.asarray(previous_masks).astype(int)
    if len(new_masks) == 0 or len(previous_masks) == 0:
        return 0.0, 0.0

    exploration_count = 0
    for new_mask in new_masks:
        nearest_distance = min(
            hamming_distance_ratio(new_mask, previous_mask)
            for previous_mask in previous_masks
        )
        if nearest_distance > similarity_threshold:
            exploration_count += 1

    exploration = exploration_count / len(new_masks)
    exploitation = 1.0 - exploration
    return float(exploration), float(exploitation)


def extract_solution_vector(individual):
    """Extract a solution vector from common mealpy Agent/history representations."""
    if hasattr(individual, "solution"):
        return np.asarray(individual.solution, dtype=float)
    if isinstance(individual, dict):
        for key in ("solution", "position", "pos"):
            if key in individual:
                return np.asarray(individual[key], dtype=float)
    if isinstance(individual, (list, tuple)) and len(individual) > 0:
        first = individual[0]
        if np.asarray(first).ndim > 0:
            return np.asarray(first, dtype=float)
    return np.asarray(individual, dtype=float)


def extract_population_matrix(population):
    """Convert a mealpy history population entry to a 2D numpy matrix."""
    return np.asarray([extract_solution_vector(ind) for ind in population], dtype=float)


def compute_formula_metrics_from_history_populations(
    populations,
    threshold: float = 0.5,
    min_features: int = 1,
    similarity_threshold: float = 0.1
) -> Tuple[List[float], List[float], List[float]]:
    """Compute diversity/exploration/exploitation for each epoch from stored populations."""
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
'''


def read_notebook(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def write_notebook(name, notebook):
    (BASE / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def source(cell):
    return "".join(cell.get("source", []))


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def replace_once(text, old, new, context):
    if old not in text:
        raise RuntimeError(f"Could not find replacement target: {context}")
    return text.replace(old, new, 1)


def patch_mealpy_notebook(input_name, output_name, save_cell_idx, csv_name):
    nb = read_notebook(input_name)
    src = source(nb["cells"][save_cell_idx])
    src = METRIC_HELPERS + "\n" + src
    src = replace_once(
        src,
        "    history_data = {",
        "    diversity_formula, exploration_formula, exploitation_formula = compute_formula_metrics_from_history_populations(\n        model.history.list_population,\n        threshold=0.5,\n        min_features=1,\n        similarity_threshold=0.1\n    )\n\n    history_data = {",
        f"{input_name} formula metric computation",
    )
    src = replace_once(
        src,
        '        "diversity": model.history.list_diversity,\n        "exploration": model.history.list_exploration,\n        "exploitation": model.history.list_exploitation,',
        '        "diversity": diversity_formula,\n        "exploration": exploration_formula,\n        "exploitation": exploitation_formula,\n        "exploration_threshold": [0.1] * len(diversity_formula),',
        f"{input_name} history dataframe fields",
    )
    src = replace_once(
        src,
        f'    history_df.to_csv(f"{"optimization_metrics_sga.csv" if "sga" in input_name else "optimization_metrics_pso.csv"}", index=False)',
        f'    history_df.to_csv("{csv_name}", index=False)\n    print("[Metrics] Saved {csv_name}")',
        f"{input_name} csv save",
    )
    set_source(nb["cells"][save_cell_idx], src)
    write_notebook(output_name, nb)


def patch_ibabc():
    nb = read_notebook("ibabc-cgo.ipynb")
    src = source(nb["cells"][4])
    src = METRIC_HELPERS + "\n" + src
    src = replace_once(
        src,
        "    for iteration in range(max_iter):\n        epoch_start = time.time()",
        "    for iteration in range(max_iter):\n        epoch_start = time.time()\n        previous_masks_for_metrics = continuous_population_to_masks(\n            population,\n            threshold=0.5,\n            min_features=1,\n            use_sigmoid=True\n        )",
        "ibabc previous masks",
    )
    src = replace_once(
        src,
        "        diversity = np.mean(np.std(population, axis=0))\n        exploration = np.linalg.norm(np.mean(population, axis=0) - global_best)\n        exploitation = np.linalg.norm(population[current_best_idx] - global_best)",
        "        current_masks_for_metrics = continuous_population_to_masks(\n            population,\n            threshold=0.5,\n            min_features=1,\n            use_sigmoid=True\n        )\n        diversity = compute_population_diversity(current_masks_for_metrics)\n        exploration, exploitation = compute_exploration_exploitation(\n            new_masks=current_masks_for_metrics,\n            previous_masks=previous_masks_for_metrics,\n            similarity_threshold=0.1\n        )",
        "ibabc metric formulas",
    )
    src = replace_once(
        src,
        '        print(f"Iteration {iteration:03d} | Best Fitness: {global_best_fitness:.5f} | Epoch Time: {epoch_end - epoch_start:.2f}s")',
        '        print(f"Iteration {iteration:03d} | Best Fitness: {global_best_fitness:.5f} | Diversity: {diversity:.4f} | Exploration: {exploration:.4f} | Exploitation: {exploitation:.4f} | Epoch Time: {epoch_end - epoch_start:.2f}s")',
        "ibabc print",
    )
    set_source(nb["cells"][4], src)

    src = source(nb["cells"][3])
    src = replace_once(
        src,
        '        "exploitation": history["list_exploitation"],\n        "epoch_time": history["list_epoch_time"]',
        '        "exploitation": history["list_exploitation"],\n        "exploration_threshold": [0.1] * len(history["list_global_best_fit"]),\n        "epoch_time": history["list_epoch_time"]',
        "ibabc threshold column",
    )
    src = replace_once(
        src,
        '    df.to_csv(f"optimization_metrics_ibabc_cgo.csv", index=False)',
        '    df.to_csv("optimization_metrics_ibabc_cgo_with_formulas.csv", index=False)\n    print("[Metrics] Saved optimization_metrics_ibabc_cgo_with_formulas.csv")',
        "ibabc csv name",
    )
    set_source(nb["cells"][3], src)
    write_notebook("ibabc-cgo_with_metrics.ipynb", nb)


def main():
    patch_mealpy_notebook(
        "sga-rf-h100.ipynb",
        "sga-rf-h100_with_metrics.ipynb",
        save_cell_idx=6,
        csv_name="optimization_metrics_sga_with_formulas.csv",
    )
    patch_mealpy_notebook(
        "pso-dt.ipynb",
        "pso-dt_with_metrics.ipynb",
        save_cell_idx=5,
        csv_name="optimization_metrics_pso_with_formulas.csv",
    )
    patch_ibabc()


if __name__ == "__main__":
    main()
