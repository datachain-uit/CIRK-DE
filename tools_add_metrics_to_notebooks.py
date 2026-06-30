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
'''


def read_notebook(name):
    path = BASE / name
    return json.loads(path.read_text(encoding="utf-8"))


def write_notebook(name, notebook):
    path = BASE / name
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def source(cell):
    return "".join(cell.get("source", []))


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def replace_once(text, old, new, context):
    if old not in text:
        raise RuntimeError(f"Could not find replacement target: {context}")
    return text.replace(old, new, 1)


def patch_bmo():
    nb = read_notebook("bmo-svm-v1.ipynb")
    src = source(nb["cells"][3])
    src = replace_once(src, "class BMOFeatureSelector:", METRIC_HELPERS + "\nclass BMOFeatureSelector:", "BMO helpers")
    src = replace_once(
        src,
        "        self.history_: List[Dict] = []\n        self.feature_selection_time_: float = 0.0",
        "        self.history_: List[Dict] = []\n        self.optimization_metrics_: Optional[pd.DataFrame] = None\n        self.feature_selection_time_: float = 0.0",
        "BMO optimization_metrics_ attribute",
    )
    src = replace_once(
        src,
        "        for gen in range(self.config.max_generations):\n            gen_start = time.perf_counter()\n\n            new_population = []",
        "        for gen in range(self.config.max_generations):\n            gen_start = time.perf_counter()\n            previous_masks_for_metrics = masks.copy()\n\n            new_population = []",
        "BMO previous masks",
    )
    src = replace_once(
        src,
        "            new_masks = np.array(new_masks)\n\n            # =========================",
        "            new_masks = np.array(new_masks)\n            exploration_threshold = getattr(self.config, \"exploration_threshold\", 0.1)\n            exploration, exploitation = compute_exploration_exploitation(\n                new_masks=new_masks,\n                previous_masks=previous_masks_for_metrics,\n                similarity_threshold=exploration_threshold\n            )\n\n            # =========================",
        "BMO exploration metrics",
    )
    src = replace_once(
        src,
        "            gen_time = time.perf_counter() - gen_start\n\n            self.history_.append({",
        "            diversity = compute_population_diversity(masks)\n            gen_time = time.perf_counter() - gen_start\n\n            self.history_.append({",
        "BMO diversity metric",
    )
    src = replace_once(
        src,
        '                "mean_num_features": float(np.mean(num_features)),\n                "gen_time_sec": gen_time',
        '                "mean_num_features": float(np.mean(num_features)),\n                "diversity": diversity,\n                "exploration": exploration,\n                "exploitation": exploitation,\n                "exploration_threshold": exploration_threshold,\n                "gen_time_sec": gen_time',
        "BMO history fields",
    )
    src = replace_once(
        src,
        '                f"Selected features = {num_features[0]} | "\n                f"Time = {gen_time:.4f}s"',
        '                f"Selected features = {num_features[0]} | "\n                f"Diversity = {diversity:.4f} | "\n                f"Exploration = {exploration:.4f} | "\n                f"Exploitation = {exploitation:.4f} | "\n                f"Time = {gen_time:.4f}s"',
        "BMO print fields",
    )
    src = replace_once(
        src,
        "        print(f\"\\n[Timing] Total feature selection time: {self.feature_selection_time_:.4f} seconds\")\n\n        return self",
        "        self.optimization_metrics_ = pd.DataFrame(self.history_)\n        self.optimization_metrics_.to_csv(\"optimization_metrics_bmo_with_formulas.csv\", index=False)\n        print(\"[Metrics] Saved optimization_metrics_bmo_with_formulas.csv\")\n        print(f\"\\n[Timing] Total feature selection time: {self.feature_selection_time_:.4f} seconds\")\n\n        return self",
        "BMO save metrics",
    )
    set_source(nb["cells"][3], src)
    write_notebook("bmo-svm-v1_with_metrics.ipynb", nb)


def patch_gwo_pso():
    nb = read_notebook("gwo-pso-knn.ipynb")
    src = source(nb["cells"][4])
    if "compute_population_diversity" not in src:
        src = METRIC_HELPERS + "\n" + src
        set_source(nb["cells"][4], src)

    src = source(nb["cells"][5])
    src = replace_once(
        src,
        "    for iteration in range(max_iter):\n        start_time = time.time()",
        "    for iteration in range(max_iter):\n        start_time = time.time()\n        previous_masks_for_metrics = continuous_population_to_masks(\n            positions,\n            threshold=0.5,\n            min_features=1,\n            use_sigmoid=True\n        )",
        "GWO previous masks",
    )
    src = replace_once(
        src,
        "        # Calculate metrics\n        diversity = np.mean(np.std(positions, axis=0))\n        exploration = np.mean(np.linalg.norm(positions - alpha_pos, axis=1))\n        exploitation = np.mean(np.linalg.norm(positions - positions.mean(axis=0), axis=1))",
        "        # Calculate metrics using the paper formulas on binary feature masks.\n        current_masks_for_metrics = continuous_population_to_masks(\n            positions,\n            threshold=0.5,\n            min_features=1,\n            use_sigmoid=True\n        )\n        diversity = compute_population_diversity(current_masks_for_metrics)\n        exploration, exploitation = compute_exploration_exploitation(\n            new_masks=current_masks_for_metrics,\n            previous_masks=previous_masks_for_metrics,\n            similarity_threshold=0.1\n        )",
        "GWO metric formulas",
    )
    src = replace_once(
        src,
        "        print(f\"Iteration {iteration + 1} - Best Fitness: {alpha_score:.4f} | Best Current Fitness: {current_best_score:.5f} | Time: {history['list_epoch_time'][-1]:.4f}s\")",
        "        print(f\"Iteration {iteration + 1} - Best Fitness: {alpha_score:.4f} | Best Current Fitness: {current_best_score:.5f} | Diversity: {diversity:.4f} | Exploration: {exploration:.4f} | Exploitation: {exploitation:.4f} | Time: {history['list_epoch_time'][-1]:.4f}s\")",
        "GWO print metrics",
    )
    src = replace_once(
        src,
        "    df.to_csv(f\"optimization_metrics_gwopso.csv\", index=False)\n    print(f\"Saved optimization_metrics_gwopso.csv\")",
        "    df.to_csv(f\"optimization_metrics_gwopso_with_formulas.csv\", index=False)\n    print(f\"Saved optimization_metrics_gwopso_with_formulas.csv\")",
        "GWO csv name",
    )
    set_source(nb["cells"][5], src)
    write_notebook("gwo-pso-knn_with_metrics.ipynb", nb)


def patch_qlde_local():
    nb = read_notebook("de-qlearning.ipynb")
    src = source(nb["cells"][3])
    src = replace_once(src, "\ndef select_action_softmax(", METRIC_HELPERS + "\ndef select_action_softmax(", "QLDE local helpers")
    src = replace_once(
        src,
        "        self.history_: List[Dict] = []\n        self.q_table_: Optional[np.ndarray] = None",
        "        self.history_: List[Dict] = []\n        self.optimization_metrics_: Optional[pd.DataFrame] = None\n        self.q_table_: Optional[np.ndarray] = None",
        "QLDE local attribute",
    )
    src = replace_once(
        src,
        "        for gen in range(self.config.max_generations):\n            new_population = population.copy()",
        "        for gen in range(self.config.max_generations):\n            previous_masks_for_metrics = continuous_population_to_masks(\n                population,\n                threshold=self.config.threshold,\n                min_features=self.config.min_features\n            )\n            trial_masks_for_metrics = np.zeros_like(previous_masks_for_metrics)\n\n            new_population = population.copy()",
        "QLDE local previous masks",
    )
    src = replace_once(
        src,
        "                trial_fitness, trial_f1, trial_nf = evaluate_feature_subset(\n                    X, y, trial_mask, self.config\n                )",
        "                trial_fitness, trial_f1, trial_nf = evaluate_feature_subset(\n                    X, y, trial_mask, self.config\n                )\n                trial_masks_for_metrics[i] = trial_mask",
        "QLDE local trial masks",
    )
    src = replace_once(
        src,
        "            population = new_population\n            fitnesses = new_fitnesses\n            macro_f1s = new_macro_f1s\n            num_features = new_num_features\n\n            self.history_.append({",
        "            population = new_population\n            fitnesses = new_fitnesses\n            macro_f1s = new_macro_f1s\n            num_features = new_num_features\n\n            current_masks_for_metrics = continuous_population_to_masks(\n                population,\n                threshold=self.config.threshold,\n                min_features=self.config.min_features\n            )\n            diversity = compute_population_diversity(current_masks_for_metrics)\n            exploration_threshold = getattr(self.config, \"exploration_threshold\", 0.1)\n            exploration, exploitation = compute_exploration_exploitation(\n                new_masks=trial_masks_for_metrics,\n                previous_masks=previous_masks_for_metrics,\n                similarity_threshold=exploration_threshold\n            )\n\n            self.history_.append({",
        "QLDE local compute metrics",
    )
    src = replace_once(
        src,
        '                "best_num_features": int(num_features[np.argmax(fitnesses)]),\n                "improvements": gen_improvements',
        '                "best_num_features": int(num_features[np.argmax(fitnesses)]),\n                "improvements": gen_improvements,\n                "diversity": diversity,\n                "exploration": exploration,\n                "exploitation": exploitation,\n                "exploration_threshold": exploration_threshold',
        "QLDE local history",
    )
    src = replace_once(
        src,
        '                f"Selected features = {num_features[np.argmax(fitnesses)]} | "\n                f"Improvements = {gen_improvements}"',
        '                f"Selected features = {num_features[np.argmax(fitnesses)]} | "\n                f"Improvements = {gen_improvements} | "\n                f"Diversity = {diversity:.4f} | "\n                f"Exploration = {exploration:.4f} | "\n                f"Exploitation = {exploitation:.4f}"',
        "QLDE local print",
    )
    src = replace_once(
        src,
        "        self.q_table_ = q_table\n        return self",
        "        self.optimization_metrics_ = pd.DataFrame(self.history_)\n        self.optimization_metrics_.to_csv(\"optimization_metrics_de_qlearning_with_formulas.csv\", index=False)\n        print(\"[Metrics] Saved optimization_metrics_de_qlearning_with_formulas.csv\")\n        self.q_table_ = q_table\n        return self",
        "QLDE local save",
    )
    set_source(nb["cells"][3], src)
    write_notebook("de-qlearning_with_metrics.ipynb", nb)


def patch_qlde_spark(name, output_name, csv_name, diversity_variant=False):
    nb = read_notebook(name)
    src = source(nb["cells"][4])
    if "def compute_population_diversity(" not in src:
        insertion_point = "\ndef select_action_softmax("
        if diversity_variant:
            insertion_point = "\ndef compute_archive_novelty("
        src = replace_once(src, insertion_point, METRIC_HELPERS + insertion_point, f"{name} helpers")
    src = replace_once(
        src,
        "        self.history_: List[Dict] = []\n        self.q_table_: Optional[np.ndarray] = None",
        "        self.history_: List[Dict] = []\n        self.optimization_metrics_: Optional[pd.DataFrame] = None\n        self.q_table_: Optional[np.ndarray] = None",
        f"{name} attribute",
    )
    src = replace_once(
        src,
        "            new_population = population.copy()\n            new_fitnesses = fitnesses.copy()",
        "            previous_masks_for_metrics = continuous_population_to_masks(\n                population,\n                threshold=self.config.threshold,\n                min_features=self.config.min_features\n            )\n            trial_masks_for_metrics = np.zeros_like(previous_masks_for_metrics)\n\n            new_population = population.copy()\n            new_fitnesses = fitnesses.copy()",
        f"{name} previous masks",
    )
    src = replace_once(
        src,
        "                    trial_fitness = res[\"trial_fitness\"]\n                    trial_f1 = res[\"trial_f1\"]",
        "                    trial_fitness = res[\"trial_fitness\"]\n                    trial_masks_for_metrics[i] = trial_mask\n                    trial_f1 = res[\"trial_f1\"]",
        f"{name} spark trial masks",
    )
    src = replace_once(
        src,
        "                    trial_fitness, trial_f1, trial_nf = evaluate_feature_subset(\n                        X_df_full, y, trial_mask, self.config\n                    )",
        "                    trial_fitness, trial_f1, trial_nf = evaluate_feature_subset(\n                        X_df_full, y, trial_mask, self.config\n                    )\n                    trial_masks_for_metrics[i] = trial_mask",
        f"{name} local trial masks",
    )
    src = replace_once(
        src,
        "            population = new_population\n            fitnesses = new_fitnesses\n            macro_f1s = new_macro_f1s\n            num_features = new_num_features",
        "            population = new_population\n            fitnesses = new_fitnesses\n            macro_f1s = new_macro_f1s\n            num_features = new_num_features\n\n            current_masks_for_metrics = continuous_population_to_masks(\n                population,\n                threshold=self.config.threshold,\n                min_features=self.config.min_features\n            )\n            diversity = compute_population_diversity(current_masks_for_metrics)\n            exploration_threshold = getattr(self.config, \"exploration_threshold\", 0.1)\n            exploration, exploitation = compute_exploration_exploitation(\n                new_masks=trial_masks_for_metrics,\n                previous_masks=previous_masks_for_metrics,\n                similarity_threshold=exploration_threshold\n            )",
        f"{name} compute metrics",
    )
    src = replace_once(
        src,
        '                "improvements": gen_improvements,',
        '                "improvements": gen_improvements,\n                "diversity": diversity,\n                "exploration": exploration,\n                "exploitation": exploitation,\n                "exploration_threshold": exploration_threshold,',
        f"{name} history",
    )
    src = replace_once(
        src,
        '                f"Improvements = {gen_improvements} | "\n',
        '                f"Improvements = {gen_improvements} | "\n                f"Diversity = {diversity:.4f} | "\n                f"Exploration = {exploration:.4f} | "\n                f"Exploitation = {exploitation:.4f} | "\n',
        f"{name} print",
    )
    src = replace_once(
        src,
        "        self.q_table_ = q_table\n        self.feature_selection_time_ = time.perf_counter() - fs_start",
        f"        self.optimization_metrics_ = pd.DataFrame(self.history_)\n        self.optimization_metrics_.to_csv(\"{csv_name}\", index=False)\n        print(\"[Metrics] Saved {csv_name}\")\n        self.q_table_ = q_table\n        self.feature_selection_time_ = time.perf_counter() - fs_start",
        f"{name} save",
    )
    set_source(nb["cells"][4], src)
    write_notebook(output_name, nb)


def main():
    patch_bmo()
    patch_gwo_pso()
    patch_qlde_local()
    patch_qlde_spark(
        "de-qlearning-spark.ipynb",
        "de-qlearning-spark_with_metrics.ipynb",
        "optimization_metrics_de_qlearning_spark_with_formulas.csv",
    )
    patch_qlde_spark(
        "deql-spark-diversity.ipynb",
        "deql-spark-diversity_with_metrics.ipynb",
        "optimization_metrics_deql_spark_diversity_with_formulas.csv",
        diversity_variant=True,
    )


if __name__ == "__main__":
    main()

