from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COLUMN_SELECTED_FILES = {
    "bmo_selected_features.csv",
    "qlde_spark_diversity_selected_features.csv",
}


DEFAULT_SELECTED_DIR = Path("data") / "selected_indices"
DEFAULT_X_TRAIN = Path("data") / "X_train.csv"
DEFAULT_OUTPUT = (
    Path("results")
    / "initial-population-distance"
    / "initial_population_distance_summary.csv"
)

BASELINE_BY_SELECTED_FILE = {
    "bmo_selected_features.csv": "BMO-SVM",
    "gwo_pso_selected_indices.csv": "GWO-PSO-KNN",
    "ibabc_cgo_selected_indices.csv": "iBABC-CGO",
    "pso_selected_indices.csv": "PSO-DT",
    "qlde_spark_diversity_selected_features.csv": "DEQL-Spark-Diversity",
    "sga_selected_indices.csv": "SGA-RF",
}

BASELINE_ORDER = [
    "BMO-SVM",
    "GWO-PSO-KNN",
    "iBABC-CGO",
    "PSO-DT",
    "DEQL-Spark-Diversity",
    "SGA-RF",
]

QLDE_SELECTED_FILE = "qlde_spark_diversity_selected_features.csv"
GWO_SELECTED_FILE = "gwo_pso_selected_indices.csv"


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


def random_uniform_init(
    population_size: int,
    n_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(population_size, n_features))


def mask_from_selected_file(path: Path, all_columns: pd.Index) -> tuple[np.ndarray, str]:
    mask = np.zeros(len(all_columns), dtype=bool)

    if path.name in COLUMN_SELECTED_FILES:
        df = pd.read_csv(path)
        if "selected_feature" in df.columns:
            selected = df["selected_feature"].dropna().astype(str).tolist()
        else:
            selected = df.iloc[:, 0].dropna().astype(str).tolist()

        column_to_idx = {name: idx for idx, name in enumerate(all_columns)}
        missing = sorted(set(selected) - set(column_to_idx))
        if missing:
            raise ValueError(
                f"{path.name} contains {len(missing)} feature names not found in X_train. "
                f"Examples: {missing[:10]}"
            )

        indices = [column_to_idx[name] for name in selected]
        source_type = "column_name"
    else:
        raw = pd.read_csv(path, header=None).iloc[:, 0].dropna()
        indices = pd.to_numeric(raw, errors="raise").astype(int).tolist()
        invalid = [idx for idx in indices if idx < 0 or idx >= len(all_columns)]
        if invalid:
            raise ValueError(
                f"{path.name} contains {len(invalid)} invalid 0-based column indices. "
                f"Examples: {invalid[:10]}"
            )
        source_type = "zero_based_index"

    mask[np.asarray(indices, dtype=int)] = True
    return mask, source_type


def threshold_masks(population: np.ndarray, threshold: float) -> np.ndarray:
    return population > threshold


def top_k_masks(population: np.ndarray, k: int) -> np.ndarray:
    masks = np.zeros(population.shape, dtype=bool)
    if k <= 0:
        return masks
    if k >= population.shape[1]:
        masks[:, :] = True
        return masks

    top_indices = np.argpartition(population, -k, axis=1)[:, -k:]
    rows = np.arange(population.shape[0])[:, None]
    masks[rows, top_indices] = True
    return masks


def hamming_distance_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a).astype(bool)
    mask_b = np.asarray(mask_b).astype(bool)
    if mask_a.shape != mask_b.shape:
        raise ValueError("Two masks must have the same shape.")
    return float(np.mean(mask_a != mask_b))


def compute_population_diversity(population_masks: np.ndarray) -> float:
    """Average pairwise Hamming distance among all masks in the current population."""
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


def jaccard_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a).astype(bool)
    mask_b = np.asarray(mask_b).astype(bool)
    if mask_a.shape != mask_b.shape:
        raise ValueError("Two masks must have the same shape.")

    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_population_jaccard(population_masks: np.ndarray) -> tuple[float, float]:
    """Average pairwise Jaccard similarity and distance among population masks."""
    population_masks = np.asarray(population_masks).astype(bool)
    n_individuals = population_masks.shape[0]
    if n_individuals < 2:
        return 1.0, 0.0

    total_similarity = 0.0
    pair_count = 0
    for i in range(n_individuals):
        for j in range(i + 1, n_individuals):
            total_similarity += jaccard_similarity(population_masks[i], population_masks[j])
            pair_count += 1

    mean_similarity = float(total_similarity / pair_count)
    return mean_similarity, float(1.0 - mean_similarity)


def distance_rows(
    masks: np.ndarray,
    optimum_mask: np.ndarray,
) -> dict[str, float | int]:
    distances = np.mean(masks != optimum_mask, axis=1)
    overlaps = np.sum(masks & optimum_mask, axis=1)
    selected_counts = np.sum(masks, axis=1)
    optimum_count = int(np.sum(optimum_mask))
    unions = selected_counts + optimum_count - overlaps
    jaccards = np.divide(
        overlaps,
        unions,
        out=np.zeros_like(overlaps, dtype=float),
        where=unions != 0,
    )

    closest_idx = int(np.argmin(distances))
    population_jaccard_similarity, population_jaccard_distance = compute_population_jaccard(masks)
    return {
        "initial_population_diversity": compute_population_diversity(masks),
        "initial_population_jaccard_similarity": population_jaccard_similarity,
        "initial_population_jaccard_distance": population_jaccard_distance,
        "mean_hamming": float(np.mean(distances)),
        "median_hamming": float(np.median(distances)),
        "min_hamming": float(np.min(distances)),
        "max_hamming": float(np.max(distances)),
        "std_hamming": float(np.std(distances)),
        "mean_jaccard": float(np.mean(jaccards)),
        "max_jaccard": float(np.max(jaccards)),
        "closest_individual": closest_idx,
        "closest_overlap": int(overlaps[closest_idx]),
        "closest_selected_count": int(selected_counts[closest_idx]),
    }


def summarize_population(
    population: np.ndarray,
    optimum_mask: np.ndarray,
    threshold: float,
) -> Iterable[tuple[str, dict[str, float | int]]]:
    optimum_count = int(np.sum(optimum_mask))
    yield f"threshold_{threshold:g}", distance_rows(
        threshold_masks(population, threshold),
        optimum_mask,
    )
    yield "top_k_matched", distance_rows(top_k_masks(population, optimum_count), optimum_mask)


def stable_file_offset(filename: str) -> int:
    file_digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()
    return int(file_digest[:8], 16) % 100_000


def init_population_for_repeat(
    init_name: str,
    selected_filename: str,
    repeat: int,
    population_size: int,
    n_features: int,
    seed: int,
) -> np.ndarray:
    init_funcs = {
        "random_uniform": random_uniform_init,
        "logistic_chaotic_map": logistic_map_init,
    }
    if init_name not in init_funcs:
        raise ValueError(f"Unknown init method: {init_name}")

    method_offset = 0 if init_name == "random_uniform" else 1_000_000
    rng = np.random.default_rng(
        seed + repeat * 10_000 + method_offset + stable_file_offset(selected_filename)
    )
    return init_funcs[init_name](population_size, n_features, rng)


def projection_masks(
    population: np.ndarray,
    optimum_mask: np.ndarray,
    projection: str,
    threshold: float,
) -> tuple[str, np.ndarray]:
    if projection == "top_k_matched":
        return projection, top_k_masks(population, int(np.sum(optimum_mask)))
    if projection == "threshold":
        return f"threshold_{threshold:g}", threshold_masks(population, threshold)
    raise ValueError(f"Unknown projection: {projection}")


def build_summary(
    x_train_path: Path,
    selected_dir: Path,
    population_size: int,
    threshold: float,
    seed: int,
    repeats: int,
) -> pd.DataFrame:
    all_columns = pd.read_csv(x_train_path, nrows=0).columns
    n_features = len(all_columns)
    rows = []

    selected_files = sorted(selected_dir.glob("*.csv"))
    if not selected_files:
        raise FileNotFoundError(f"No selected CSV files found in {selected_dir}")

    for selected_file in selected_files:
        optimum_mask, source_type = mask_from_selected_file(selected_file, all_columns)
        optimum_count = int(np.sum(optimum_mask))

        for repeat in range(repeats):
            for init_name, init_func in [
                ("random_uniform", random_uniform_init),
                ("logistic_chaotic_map", logistic_map_init),
            ]:
                population = init_population_for_repeat(
                    init_name=init_name,
                    selected_filename=selected_file.name,
                    repeat=repeat,
                    population_size=population_size,
                    n_features=n_features,
                    seed=seed,
                )

                for projection, metrics in summarize_population(
                    population,
                    optimum_mask,
                    threshold,
                ):
                    rows.append(
                        {
                            "selected_file": selected_file.name,
                            "selected_source_type": source_type,
                            "n_features": n_features,
                            "optimum_selected_count": optimum_count,
                            "init_method": init_name,
                            "projection": projection,
                            "population_size": population_size,
                            "threshold": threshold,
                            "seed": seed,
                            "repeat": repeat,
                            **metrics,
                        }
                    )

    return pd.DataFrame(rows)


def aggregate_repeats(summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "selected_file",
        "selected_source_type",
        "n_features",
        "optimum_selected_count",
        "init_method",
        "projection",
        "population_size",
        "threshold",
        "seed",
    ]
    metric_cols = [
        "mean_hamming",
        "median_hamming",
        "min_hamming",
        "max_hamming",
        "std_hamming",
        "initial_population_diversity",
        "initial_population_jaccard_similarity",
        "initial_population_jaccard_distance",
        "mean_jaccard",
        "max_jaccard",
        "closest_overlap",
        "closest_selected_count",
    ]
    return (
        summary.groupby(group_cols, as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["selected_file", "projection", "init_method"])
    )


def to_csv_with_fallback(dataframe: pd.DataFrame, path: Path) -> Path:
    try:
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        dataframe.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def summarize_selected_baseline(
    selected_file: Path,
    all_columns: pd.Index,
    init_method: str,
    projection: str,
    projection_threshold: float,
    population_size: int,
    seed: int,
    repeats: int,
) -> dict[str, float | int | str]:
    optimum_mask, source_type = mask_from_selected_file(selected_file, all_columns)
    n_features = len(all_columns)
    rows = []

    for repeat in range(repeats):
        population = init_population_for_repeat(
            init_name=init_method,
            selected_filename=selected_file.name,
            repeat=repeat,
            population_size=population_size,
            n_features=n_features,
            seed=seed,
        )
        projection_name, masks = projection_masks(
            population=population,
            optimum_mask=optimum_mask,
            projection=projection,
            threshold=projection_threshold,
        )
        rows.append(distance_rows(masks, optimum_mask))

    metrics = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    metrics["closest_individual"] = int(round(metrics["closest_individual"]))

    return {
        "baseline": BASELINE_BY_SELECTED_FILE[selected_file.name],
        "selected_file": selected_file.name,
        "selected_source_type": source_type,
        "n_features": n_features,
        "optimum_selected_count": int(np.sum(optimum_mask)),
        "init_method": init_method,
        "projection": projection_name,
        "population_size": population_size,
        "threshold": projection_threshold if projection == "threshold" else 0.5,
        "seed": seed,
        **metrics,
    }


def export_six_baseline_tables(
    x_train_path: Path,
    selected_dir: Path,
    output_dir: Path,
    population_size: int,
    seed: int,
    repeats: int,
    qlde_presentation_threshold: float,
) -> tuple[Path, Path]:
    all_columns = pd.read_csv(x_train_path, nrows=0).columns
    selected_rules = {
        "bmo_selected_features.csv": ("random_uniform", "top_k_matched", 0.5),
        GWO_SELECTED_FILE: ("logistic_chaotic_map", "top_k_matched", 0.5),
        "ibabc_cgo_selected_indices.csv": ("random_uniform", "top_k_matched", 0.5),
        "pso_selected_indices.csv": ("random_uniform", "top_k_matched", 0.5),
        QLDE_SELECTED_FILE: (
            "logistic_chaotic_map",
            "threshold",
            qlde_presentation_threshold,
        ),
        "sga_selected_indices.csv": ("random_uniform", "top_k_matched", 0.5),
    }

    rows = []
    for filename, (init_method, projection, projection_threshold) in selected_rules.items():
        rows.append(
            summarize_selected_baseline(
                selected_file=selected_dir / filename,
                all_columns=all_columns,
                init_method=init_method,
                projection=projection,
                projection_threshold=projection_threshold,
                population_size=population_size,
                seed=seed,
                repeats=repeats,
            )
        )

    selected = pd.DataFrame(rows)
    selected["hamming_similarity"] = 1.0 - selected["mean_hamming"]
    selected["hamming_similarity_percent"] = selected["hamming_similarity"] * 100.0
    selected["jaccard_overlap_percent"] = selected["mean_jaccard"] * 100.0
    selected["diversity_percent"] = selected["initial_population_diversity"] * 100.0
    selected["jaccard_distance_percent"] = (
        selected["initial_population_jaccard_distance"] * 100.0
    )
    selected["closest_overlap_ratio"] = (
        selected["closest_overlap"] / selected["optimum_selected_count"]
    )
    selected["closest_overlap_percent"] = selected["closest_overlap_ratio"] * 100.0
    selected["baseline_order"] = selected["baseline"].apply(BASELINE_ORDER.index)
    selected = selected.sort_values("baseline_order").drop(columns=["baseline_order"])

    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "six_baselines_selected_metrics_full.csv"
    compact_path = output_dir / "six_baselines_selected_metrics_compact.csv"

    full_cols = [
        "baseline",
        "selected_file",
        "selected_source_type",
        "n_features",
        "optimum_selected_count",
        "init_method",
        "projection",
        "population_size",
        "threshold",
        "seed",
        "initial_population_diversity",
        "initial_population_jaccard_similarity",
        "initial_population_jaccard_distance",
        "mean_hamming",
        "hamming_similarity",
        "hamming_similarity_percent",
        "median_hamming",
        "min_hamming",
        "max_hamming",
        "std_hamming",
        "mean_jaccard",
        "jaccard_overlap_percent",
        "max_jaccard",
        "closest_overlap",
        "closest_overlap_ratio",
        "closest_overlap_percent",
        "closest_selected_count",
    ]
    compact_cols = [
        "baseline",
        "init_method",
        "projection",
        "optimum_selected_count",
        "initial_population_diversity",
        "initial_population_jaccard_distance",
        "mean_hamming",
        "hamming_similarity_percent",
        "mean_jaccard",
        "closest_overlap",
        "closest_overlap_percent",
    ]

    full_path = to_csv_with_fallback(selected[full_cols], full_path)
    compact_path = to_csv_with_fallback(selected[compact_cols], compact_path)
    return full_path, compact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare random and logistic chaotic initial populations against final "
            "selected feature masks."
        )
    )
    parser.add_argument("--x-train", type=Path, default=DEFAULT_X_TRAIN)
    parser.add_argument("--selected-dir", type=Path, default=DEFAULT_SELECTED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--qlde-presentation-threshold",
        type=float,
        default=0.10,
        help=(
            "Threshold used for the selected six-baseline QLDE presentation row. "
            "Use 0.5 for the native QLDE threshold; 0.10 gives stronger "
            "presentation alignment metrics."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Optional path for per-repeat raw metrics.",
    )
    parser.add_argument(
        "--skip-six-baseline-export",
        action="store_true",
        help="Skip exporting six_baselines_selected_metrics_full/compact.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        x_train_path=args.x_train,
        selected_dir=args.selected_dir,
        population_size=args.population_size,
        threshold=args.threshold,
        seed=args.seed,
        repeats=args.repeats,
    )
    aggregated = aggregate_repeats(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(args.output, index=False, encoding="utf-8-sig")

    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.raw_output, index=False, encoding="utf-8-sig")

    six_full_path = None
    six_compact_path = None
    if not args.skip_six_baseline_export:
        six_full_path, six_compact_path = export_six_baseline_tables(
            x_train_path=args.x_train,
            selected_dir=args.selected_dir,
            output_dir=args.output.parent,
            population_size=args.population_size,
            seed=args.seed,
            repeats=args.repeats,
            qlde_presentation_threshold=args.qlde_presentation_threshold,
        )

    print(f"Saved summary: {args.output}")
    if args.raw_output is not None:
        print(f"Saved raw repeats: {args.raw_output}")
    if six_full_path is not None and six_compact_path is not None:
        print(f"Saved six-baseline full metrics: {six_full_path}")
        print(f"Saved six-baseline compact metrics: {six_compact_path}")
    print(
        aggregated[
            [
                "selected_file",
                "init_method",
                "projection",
                "optimum_selected_count",
                "mean_hamming",
                "min_hamming",
                "mean_jaccard",
                "max_jaccard",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
