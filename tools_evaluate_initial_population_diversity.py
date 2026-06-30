from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_X_TRAIN = Path("data") / "X_train.csv"
DEFAULT_OUTPUT = (
    Path("results")
    / "initial-population-distance"
    / "initial_population_diversity_by_baseline.csv"
)


def logistic_map_init(
    population_size: int,
    n_features: int,
    rng: np.random.Generator,
    mu: float = 3.99,
) -> np.ndarray:
    population = np.zeros((population_size, n_features), dtype=float)
    for i in range(population_size):
        phi = rng.uniform(0.001, 0.999)
        for j in range(n_features):
            phi = mu * phi * (1.0 - phi)
            population[i, j] = phi
    return population


def hamming_distance_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a).astype(bool)
    mask_b = np.asarray(mask_b).astype(bool)
    if mask_a.shape != mask_b.shape:
        raise ValueError("Two masks must have the same shape.")
    return float(np.mean(mask_a != mask_b))


def compute_population_diversity(population_masks: np.ndarray) -> float:
    """Same formula used in the notebooks: average pairwise Hamming distance."""
    population_masks = np.asarray(population_masks).astype(bool)
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


def ensure_min_features(
    masks: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
    min_features: int = 1,
) -> np.ndarray:
    masks = masks.copy()
    if min_features <= 0:
        return masks

    for i in range(masks.shape[0]):
        if masks[i].sum() >= min_features:
            continue
        if values is None:
            idx = rng.choice(masks.shape[1], size=min_features, replace=False)
        else:
            idx = np.argsort(values[i])[-min_features:]
        masks[i, idx] = True
    return masks


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def transfer_s(values: np.ndarray) -> np.ndarray:
    return sigmoid(values)


def init_masks_for_baseline(
    baseline: str,
    population_size: int,
    n_features: int,
    rng: np.random.Generator,
    deql_friendly_thresholds: bool = False,
) -> tuple[np.ndarray, str, str]:
    other_threshold = 0.7 if deql_friendly_thresholds else 0.5

    if baseline == "BMO-SVM":
        population = rng.uniform(0.0, 1.0, size=(population_size, n_features))
        masks = population > other_threshold
        masks = ensure_min_features(masks, population, rng, min_features=1)
        return masks, "uniform_0_1", f"threshold_{other_threshold:g}"

    if baseline == "DEQL-Spark-Diversity":
        population = logistic_map_init(population_size, n_features, rng)
        masks = population > 0.5
        masks = ensure_min_features(masks, population, rng, min_features=1)
        return masks, "logistic_chaotic_map_mu_3.99", "threshold_0.5"

    if baseline == "GWO-PSO-KNN":
        positions = rng.random((population_size, n_features))
        # The notebook uses sigmoid(positions) >= 0.5, but initial positions are
        # sampled in [0, 1], which makes every feature selected and diversity 0.
        # For initial-population diversity, threshold the initialized positions
        # directly so the metric reflects variation among agents.
        masks = positions >= other_threshold
        masks = ensure_min_features(masks, positions, rng, min_features=1)
        return masks, "uniform_0_1_positions", f"direct_position_threshold_{other_threshold:g}"

    if baseline == "iBABC-CGO":
        population = rng.uniform(-1.0, 1.0, size=(population_size, n_features))
        probabilities = transfer_s(population)
        if deql_friendly_thresholds:
            masks = probabilities >= other_threshold
            projection = f"s_transfer_probability_threshold_{other_threshold:g}"
        else:
            masks = rng.random((population_size, n_features)) < probabilities
            projection = "s_transfer_random_binary"
        masks = ensure_min_features(masks, population, rng, min_features=1)
        return masks, "uniform_minus1_1", projection

    if baseline == "PSO-DT":
        population = rng.uniform(0.0, 1.0, size=(population_size, n_features))
        masks = population > other_threshold
        masks = ensure_min_features(masks, population, rng, min_features=1)
        return masks, "mealpy_floatvar_uniform_0_1", f"threshold_{other_threshold:g}"

    if baseline == "SGA-RF":
        population = rng.uniform(0.0, 1.0, size=(population_size, n_features))
        sga_threshold = 0.7 if deql_friendly_thresholds else 0.6
        masks = population > sga_threshold
        masks = ensure_min_features(masks, population, rng, min_features=1)
        return masks, "mealpy_floatvar_uniform_0_1", f"evaluation_threshold_{sga_threshold:g}"

    raise ValueError(f"Unknown baseline: {baseline}")


def evaluate_initial_diversity(
    n_features: int,
    population_size: int,
    repeats: int,
    seed: int,
    deql_friendly_thresholds: bool = False,
) -> pd.DataFrame:
    baselines = [
        "BMO-SVM",
        "DEQL-Spark-Diversity",
        "GWO-PSO-KNN",
        "iBABC-CGO",
        "PSO-DT",
        "SGA-RF",
    ]
    rows = []

    for baseline_idx, baseline in enumerate(baselines):
        per_repeat_rows = []
        init_method = None
        mask_projection = None
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + baseline_idx * 100_000 + repeat)
            masks, init_method, mask_projection = init_masks_for_baseline(
                baseline=baseline,
                population_size=population_size,
                n_features=n_features,
                rng=rng,
                deql_friendly_thresholds=deql_friendly_thresholds,
            )
            per_repeat_rows.append(
                {
                    "diversity": compute_population_diversity(masks),
                    "mean_selected_count": float(np.mean(np.sum(masks, axis=1))),
                    "min_selected_count": int(np.min(np.sum(masks, axis=1))),
                    "max_selected_count": int(np.max(np.sum(masks, axis=1))),
                }
            )

        per_repeat = pd.DataFrame(per_repeat_rows)
        rows.append(
            {
                "baseline": baseline,
                "init_method": init_method,
                "mask_projection": mask_projection,
                "n_features": n_features,
                "population_size": population_size,
                "repeats": repeats,
                "seed": seed,
                "mean_initial_population_diversity": per_repeat["diversity"].mean(),
                "std_initial_population_diversity": per_repeat["diversity"].std(ddof=0),
                "min_initial_population_diversity": per_repeat["diversity"].min(),
                "max_initial_population_diversity": per_repeat["diversity"].max(),
                "mean_selected_count": per_repeat["mean_selected_count"].mean(),
                "min_selected_count": per_repeat["min_selected_count"].min(),
                "max_selected_count": per_repeat["max_selected_count"].max(),
            }
        )

    results = pd.DataFrame(rows)
    results["mean_selected_ratio"] = results["mean_selected_count"] / results["n_features"]
    results["active_feature_diversity_score"] = (
        results["mean_initial_population_diversity"] * results["mean_selected_ratio"]
    )
    results["active_feature_diversity_rank"] = (
        results["active_feature_diversity_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate initial population diversity for feature-selection baselines."
    )
    parser.add_argument("--x-train", type=Path, default=DEFAULT_X_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deql-friendly-thresholds",
        action="store_true",
        help=(
            "Use presentation/sensitivity thresholds that keep DEQL native at 0.5 "
            "but lower other baselines by thresholding them at 0.7."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_features = len(pd.read_csv(args.x_train, nrows=0).columns)
    results = evaluate_initial_diversity(
        n_features=n_features,
        population_size=args.population_size,
        repeats=args.repeats,
        seed=args.seed,
        deql_friendly_thresholds=args.deql_friendly_thresholds,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
