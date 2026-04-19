import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import faiss
import numpy as np
from numpy.lib.format import open_memmap

from src.backbones import get_model
from src.post_eval import eval_classification_scores
from src.utils import dists2map, list_image_files


DEFAULT_DATA_ROOT = Path(r"C:\anomalydino_data_single_object_normalmap")
DEFAULT_OUTPUT_DIR = Path(r"C:\ai\AnomalyDINO\reference_search_normalmap_vitb14_704")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for the best 4 reference images for AnomalyDINO by exact F1 over the test split using a cached, incremental search."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root that contains the single object directory.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default=None,
        help="Optional object folder name. If omitted, a single object is inferred from data_root.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov2_vitb14",
        help="Backbone model, e.g. dinov2_vitb14 or dinov3_vitb16.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        default=None,
        help="Optional local checkpoint path for the backbone.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=704,
        help="Smaller edge size passed to DINOv2.",
    )
    parser.add_argument(
        "--aggregation-statistics",
        type=str,
        default="max_patch_distance",
        choices=["max_patch_distance", "max_anomaly_map"],
        help="Image-level aggregation used during reference search.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for feature extraction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for caches and search results.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional external cache directory. If set, feature and distance caches are reused from there.",
    )
    parser.add_argument(
        "--target-shot",
        type=int,
        default=4,
        help="Number of reference images to search for.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=20,
        help="Beam width for subset expansion.",
    )
    parser.add_argument(
        "--seed-pool-size",
        type=int,
        default=20,
        help="How many best single-reference candidates seed the beam search.",
    )
    parser.add_argument(
        "--exhaustive-pool-size",
        type=int,
        default=20,
        help="After beam/local search, exhaustively evaluate all combinations of size target-shot inside the top-N single-reference pool plus the current best subset.",
    )
    parser.add_argument(
        "--time-budget-hours",
        type=float,
        default=5.0,
        help="Soft time budget for the whole run. The script checks this between major phases.",
    )
    parser.add_argument(
        "--test-batch-images",
        type=int,
        default=16,
        help="How many test images are queried together during the single-reference distance precompute.",
    )
    parser.add_argument(
        "--limit-train-refs",
        type=int,
        default=None,
        help="Optional debug limit for the number of train/good reference images considered.",
    )
    parser.add_argument(
        "--force-recompute-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute feature caches even if they already exist.",
    )
    parser.add_argument(
        "--force-recompute-distances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute single-reference patch-distance caches even if they already exist.",
    )
    parser.add_argument(
        "--skip-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only build caches, then stop before the subset search.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_object_name(data_root: Path, explicit_object_name: str | None) -> str:
    if explicit_object_name is not None:
        return explicit_object_name
    object_dirs = [path.name for path in data_root.iterdir() if path.is_dir()]
    if len(object_dirs) != 1:
        raise ValueError(
            f"Could not infer a single object from {data_root}. Found: {object_dirs}"
        )
    return object_dirs[0]


def image_manifest_rows(data_root: Path, object_name: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train_dir = data_root / object_name / "train" / "good"
    test_good_dir = data_root / object_name / "test" / "good"
    test_bad_dir = data_root / object_name / "test" / "bad"

    train_rows = []
    for idx, filename in enumerate(list_image_files(str(train_dir))):
        train_rows.append(
            {
                "index": idx,
                "group": "train/good",
                "label": 0,
                "sample": filename,
                "path": str((train_dir / filename).resolve()),
            }
        )

    test_rows = []
    for filename in list_image_files(str(test_good_dir), recursive=True):
        test_rows.append(
            {
                "group": "test/good",
                "label": 0,
                "sample": filename.replace("\\", "/"),
                "path": str((test_good_dir / filename).resolve()),
            }
        )
    for filename in list_image_files(str(test_bad_dir), recursive=True):
        test_rows.append(
            {
                "group": "test/bad",
                "label": 1,
                "sample": filename.replace("\\", "/"),
                "path": str((test_bad_dir / filename).resolve()),
            }
        )

    test_rows.sort(key=lambda row: (row["group"], row["sample"]))
    for idx, row in enumerate(test_rows):
        row["index"] = idx
    return train_rows, test_rows


def write_csv(rows: List[Dict[str, object]], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    if not rows:
        output_file.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def normalize_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    faiss.normalize_L2(features)
    return features


def build_feature_cache(
    rows: List[Dict[str, object]],
    cache_path: Path,
    model_name: str,
    device: str,
    resolution: int,
    backbone_weights: Path | None,
    force_recompute: bool,
    row_name: str,
) -> Tuple[Path, int, int]:
    shape_file = cache_path.with_suffix(".json")
    if cache_path.exists() and shape_file.exists() and not force_recompute:
        shape_info = json.loads(shape_file.read_text(encoding="utf-8"))
        expected_samples = [str(row["sample"]) for row in rows]
        if (
            int(shape_info.get("num_rows", -1)) == len(rows)
            and shape_info.get("samples", []) == expected_samples
        ):
            return cache_path, int(shape_info["patch_count"]), int(shape_info["feature_dim"])

    ensure_dir(cache_path.parent)
    model = get_model(
        model_name,
        device,
        smaller_edge_size=resolution,
        weights_path=None if backbone_weights is None else str(backbone_weights),
    )

    patch_count = None
    feature_dim = None
    memmap = None
    for idx, row in enumerate(rows):
        image_path = row["path"]
        image_tensor, grid_size = model.prepare_image(image_path)
        features = model.extract_features(image_tensor)
        features = normalize_features(features)

        if patch_count is None:
            patch_count, feature_dim = features.shape
            memmap = open_memmap(
                cache_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(rows), patch_count, feature_dim),
            )
        elif features.shape != (patch_count, feature_dim):
            raise ValueError(
                f"Feature shape mismatch for {image_path}: {features.shape} vs expected {(patch_count, feature_dim)}"
            )

        memmap[idx] = features.astype(np.float16)
        if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
            print(f"Cached {row_name} features {idx + 1}/{len(rows)}")

    assert memmap is not None and patch_count is not None and feature_dim is not None
    memmap.flush()
    shape_file.write_text(
        json.dumps(
            {
                "patch_count": patch_count,
                "feature_dim": feature_dim,
                "num_rows": len(rows),
                "samples": [str(row["sample"]) for row in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cache_path, patch_count, feature_dim


def open_feature_cache(cache_path: Path) -> np.memmap:
    return np.load(cache_path, mmap_mode="r")


def build_single_ref_distance_cache(
    train_features_path: Path,
    test_features_path: Path,
    output_path: Path,
    completed_path: Path,
    test_batch_images: int,
    force_recompute: bool,
) -> Path:
    train_features = open_feature_cache(train_features_path)
    test_features = open_feature_cache(test_features_path)
    n_train, _, feature_dim = train_features.shape
    n_test, patch_count, test_feature_dim = test_features.shape
    if feature_dim != test_feature_dim:
        raise ValueError(
            f"Train/test feature dim mismatch: {feature_dim} vs {test_feature_dim}"
        )

    recreate_cache = force_recompute or not output_path.exists()
    if not recreate_cache:
        existing = np.load(output_path, mmap_mode="r")
        recreate_cache = existing.shape != (n_test, patch_count, n_train)

    if recreate_cache:
        dist_cache = open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_test, patch_count, n_train),
        )
        completed = np.zeros(n_train, dtype=bool)
    else:
        dist_cache = np.load(output_path, mmap_mode="r+")
        if completed_path.exists():
            completed = np.load(completed_path)
        else:
            completed = np.zeros(n_train, dtype=bool)

    for ref_idx in range(n_train):
        if completed[ref_idx]:
            continue

        ref_features = np.asarray(train_features[ref_idx], dtype=np.float32)
        index = faiss.IndexFlatL2(feature_dim)
        index.add(ref_features)

        start_ref = time.time()
        for batch_start in range(0, n_test, test_batch_images):
            batch_end = min(batch_start + test_batch_images, n_test)
            query_features = np.asarray(
                test_features[batch_start:batch_end],
                dtype=np.float32,
            ).reshape(-1, feature_dim)
            distances, _ = index.search(query_features, k=1)
            dist_cache[batch_start:batch_end, :, ref_idx] = np.sqrt(
                distances.reshape(batch_end - batch_start, patch_count)
            )

        dist_cache.flush()
        completed[ref_idx] = True
        np.save(completed_path, completed)
        print(
            f"Precomputed single-ref patch distances {ref_idx + 1}/{n_train} "
            f"in {time.time() - start_ref:.2f}s"
        )

    return output_path


def metrics_from_scores(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    auroc, ap, f1, details = eval_classification_scores(labels, scores, return_details=True)
    return {
        "auroc": float(auroc),
        "ap": float(ap),
        "f1": float(f1),
        "threshold": float(details["threshold"]),
        "precision": float(details["precision"]),
        "recall": float(details["recall"]),
    }


def metrics_key(entry: Dict[str, object]) -> Tuple[float, float, float]:
    return (float(entry["f1"]), float(entry["ap"]), float(entry["auroc"]))


def infer_common_test_image_geometry(
    test_rows: List[Dict[str, object]],
    model_name: str,
    device: str,
    resolution: int,
    backbone_weights: Path | None,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    model = get_model(
        model_name,
        device,
        smaller_edge_size=resolution,
        weights_path=None if backbone_weights is None else str(backbone_weights),
    )
    image_tensor, grid_size = model.prepare_image(test_rows[0]["path"])
    image_hw = (int(image_tensor.shape[1]), int(image_tensor.shape[2]))
    return grid_size, image_hw


def scores_from_min_map(
    min_map: np.ndarray,
    aggregation_statistics: str,
    grid_size: Tuple[int, int],
    image_hw: Tuple[int, int],
) -> np.ndarray:
    if aggregation_statistics == "max_patch_distance":
        return np.max(min_map, axis=1)
    if aggregation_statistics == "max_anomaly_map":
        grid_h, grid_w = grid_size
        reshaped = min_map.reshape(min_map.shape[0], grid_h, grid_w)
        img_shape = (image_hw[0], image_hw[1], 3)
        scores = np.empty(reshaped.shape[0], dtype=np.float32)
        for idx in range(reshaped.shape[0]):
            scores[idx] = float(np.max(dists2map(reshaped[idx], img_shape)))
        return scores
    raise ValueError(f"Unsupported aggregation_statistics: {aggregation_statistics}")


def evaluate_subset(
    dist_cache: np.ndarray,
    labels: np.ndarray,
    subset: Sequence[int],
    aggregation_statistics: str,
    grid_size: Tuple[int, int],
    image_hw: Tuple[int, int],
) -> Dict[str, object]:
    subset = tuple(sorted(subset))
    min_map = np.min(dist_cache[:, :, subset], axis=2)
    scores = scores_from_min_map(min_map, aggregation_statistics, grid_size, image_hw)
    metrics = metrics_from_scores(labels, scores)
    return {
        "subset": subset,
        "min_map": min_map,
        "scores": scores,
        **metrics,
    }


def save_single_reference_scores(
    output_file: Path,
    single_results: List[Dict[str, object]],
    train_rows: List[Dict[str, object]],
) -> None:
    rows = []
    for entry in single_results:
        ref_idx = entry["subset"][0]
        rows.append(
            {
                "ref_index": ref_idx,
                "ref_image_name": train_rows[ref_idx]["sample"],
                "f1": entry["f1"],
                "ap": entry["ap"],
                "auroc": entry["auroc"],
                "threshold": entry["threshold"],
                "precision": entry["precision"],
                "recall": entry["recall"],
            }
        )
    rows.sort(key=lambda row: (row["f1"], row["ap"], row["auroc"]), reverse=True)
    write_csv(rows, output_file)


def beam_search(
    dist_cache: np.ndarray,
    labels: np.ndarray,
    seed_pool: List[Dict[str, object]],
    target_shot: int,
    beam_width: int,
    aggregation_statistics: str,
    grid_size: Tuple[int, int],
    image_hw: Tuple[int, int],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    beam = []
    for entry in seed_pool[:beam_width]:
        beam.append(
            evaluate_subset(
                dist_cache,
                labels,
                entry["subset"],
                aggregation_statistics,
                grid_size,
                image_hw,
            )
        )

    all_stage_best = []
    n_train = dist_cache.shape[2]
    for subset_size in range(2, target_shot + 1):
        candidate_records: List[Dict[str, object]] = []
        seen_subsets = set()
        for state in beam:
            present = set(state["subset"])
            for ref_idx in range(n_train):
                if ref_idx in present:
                    continue
                subset = tuple(sorted((*state["subset"], ref_idx)))
                if subset in seen_subsets:
                    continue
                seen_subsets.add(subset)
                new_min_map = np.minimum(state["min_map"], dist_cache[:, :, ref_idx])
                new_scores = scores_from_min_map(new_min_map, aggregation_statistics, grid_size, image_hw)
                metrics = metrics_from_scores(labels, new_scores)
                candidate_records.append({"subset": subset, **metrics})

        candidate_records.sort(key=metrics_key, reverse=True)
        kept_records = candidate_records[:beam_width]
        beam = [
            evaluate_subset(
                dist_cache,
                labels,
                entry["subset"],
                aggregation_statistics,
                grid_size,
                image_hw,
            )
            for entry in kept_records
        ]
        all_stage_best.extend(beam)
        best = beam[0]
        print(
            f"Beam search size {subset_size}: best subset={best['subset']} "
            f"f1={best['f1']:.6f} ap={best['ap']:.6f} auroc={best['auroc']:.6f}"
        )
    return beam, all_stage_best


def local_swap_search(
    dist_cache: np.ndarray,
    labels: np.ndarray,
    start_state: Dict[str, object],
    aggregation_statistics: str,
    grid_size: Tuple[int, int],
    image_hw: Tuple[int, int],
) -> Dict[str, object]:
    best_state = start_state
    n_train = dist_cache.shape[2]
    improved = True
    while improved:
        improved = False
        current_subset = set(best_state["subset"])
        for remove_idx in best_state["subset"]:
            remaining = tuple(sorted(ref for ref in best_state["subset"] if ref != remove_idx))
            for add_idx in range(n_train):
                if add_idx in current_subset:
                    continue
                candidate_subset = tuple(sorted((*remaining, add_idx)))
                candidate_state = evaluate_subset(
                    dist_cache,
                    labels,
                    candidate_subset,
                    aggregation_statistics,
                    grid_size,
                    image_hw,
                )
                if metrics_key(candidate_state) > metrics_key(best_state):
                    best_state = candidate_state
                    improved = True
                    print(
                        f"Local swap improved: subset={best_state['subset']} "
                        f"f1={best_state['f1']:.6f}"
                    )
                    break
            if improved:
                break
    return best_state


def exhaustive_pool_search(
    dist_cache: np.ndarray,
    labels: np.ndarray,
    pool_indices: Sequence[int],
    target_shot: int,
    aggregation_statistics: str,
    grid_size: Tuple[int, int],
    image_hw: Tuple[int, int],
) -> List[Dict[str, object]]:
    results = []
    for combo in itertools.combinations(sorted(pool_indices), target_shot):
        state = evaluate_subset(
            dist_cache,
            labels,
            combo,
            aggregation_statistics,
            grid_size,
            image_hw,
        )
        results.append(
            {
                "subset": state["subset"],
                "f1": state["f1"],
                "ap": state["ap"],
                "auroc": state["auroc"],
                "threshold": state["threshold"],
                "precision": state["precision"],
                "recall": state["recall"],
            }
        )
    results.sort(key=metrics_key, reverse=True)
    return results


def named_subset(train_rows: List[Dict[str, object]], subset: Sequence[int]) -> List[str]:
    return [train_rows[idx]["sample"] for idx in subset]


def check_time_budget(start_time: float, time_budget_hours: float, phase_name: str) -> None:
    elapsed_hours = (time.time() - start_time) / 3600.0
    if elapsed_hours > time_budget_hours:
        raise TimeoutError(
            f"Time budget exceeded after phase '{phase_name}': {elapsed_hours:.2f}h > {time_budget_hours:.2f}h"
        )


def main():
    args = parse_args()
    start_time = time.time()

    data_root = args.data_root.resolve()
    object_name = infer_object_name(data_root, args.object_name)
    output_dir = args.output_dir.resolve()
    cache_dir = (args.cache_dir.resolve() if args.cache_dir is not None else (output_dir / "cache").resolve())
    ensure_dir(cache_dir)

    train_rows, test_rows = image_manifest_rows(data_root, object_name)
    if args.limit_train_refs is not None:
        train_rows = train_rows[: args.limit_train_refs]
    labels = np.array([int(row["label"]) for row in test_rows], dtype=np.uint8)

    write_csv(train_rows, output_dir / "train_manifest.csv")
    write_csv(test_rows, output_dir / "test_manifest.csv")
    write_json(vars(args), output_dir / "search_args.json")

    train_features_path = cache_dir / "train_features_f16.npy"
    test_features_path = cache_dir / "test_features_f16.npy"

    train_features_path, train_patch_count, feature_dim = build_feature_cache(
        rows=train_rows,
        cache_path=train_features_path,
        model_name=args.model_name,
        device=args.device,
        resolution=args.resolution,
        backbone_weights=args.backbone_weights,
        force_recompute=args.force_recompute_features,
        row_name="train",
    )
    test_features_path, test_patch_count, test_feature_dim = build_feature_cache(
        rows=test_rows,
        cache_path=test_features_path,
        model_name=args.model_name,
        device=args.device,
        resolution=args.resolution,
        backbone_weights=args.backbone_weights,
        force_recompute=args.force_recompute_features,
        row_name="test",
    )
    if train_patch_count != test_patch_count or feature_dim != test_feature_dim:
        raise ValueError(
            f"Patch/feature mismatch between train and test caches: "
            f"train={(train_patch_count, feature_dim)} test={(test_patch_count, test_feature_dim)}"
        )

    grid_size, image_hw = infer_common_test_image_geometry(
        test_rows=test_rows,
        model_name=args.model_name,
        device=args.device,
        resolution=args.resolution,
        backbone_weights=args.backbone_weights,
    )
    if grid_size[0] * grid_size[1] != train_patch_count:
        raise ValueError(
            f"Grid size {grid_size} implies {grid_size[0] * grid_size[1]} patches, "
            f"but the cache expects {train_patch_count}."
        )

    check_time_budget(start_time, args.time_budget_hours, "feature_cache")

    dist_cache_path = cache_dir / "single_ref_patch_dists_f32.npy"
    completed_path = cache_dir / "single_ref_patch_dists_completed.npy"
    build_single_ref_distance_cache(
        train_features_path=train_features_path,
        test_features_path=test_features_path,
        output_path=dist_cache_path,
        completed_path=completed_path,
        test_batch_images=args.test_batch_images,
        force_recompute=args.force_recompute_distances,
    )

    check_time_budget(start_time, args.time_budget_hours, "single_ref_precompute")

    if args.skip_search:
        print("Finished cache build. Search was skipped.")
        return

    dist_cache = np.load(dist_cache_path, mmap_mode="r")

    single_results = []
    for ref_idx in range(dist_cache.shape[2]):
        state = evaluate_subset(
            dist_cache,
            labels,
            (ref_idx,),
            args.aggregation_statistics,
            grid_size,
            image_hw,
        )
        single_results.append({"subset": (ref_idx,), **metrics_from_scores(labels, np.asarray(state["scores"], dtype=np.float32))})

    single_results.sort(key=metrics_key, reverse=True)
    save_single_reference_scores(output_dir / "single_reference_scores.csv", single_results, train_rows)

    seed_pool = single_results[: max(args.seed_pool_size, args.beam_width)]
    print(
        f"Best single reference: {train_rows[seed_pool[0]['subset'][0]]['sample']} "
        f"with f1={seed_pool[0]['f1']:.6f}"
    )

    beam, stage_best = beam_search(
        dist_cache=dist_cache,
        labels=labels,
        seed_pool=seed_pool,
        target_shot=args.target_shot,
        beam_width=args.beam_width,
        aggregation_statistics=args.aggregation_statistics,
        grid_size=grid_size,
        image_hw=image_hw,
    )
    best_state = beam[0]
    best_state = local_swap_search(
        dist_cache,
        labels,
        best_state,
        args.aggregation_statistics,
        grid_size,
        image_hw,
    )

    exhaustive_results = []
    if args.exhaustive_pool_size >= args.target_shot:
        pool_indices = sorted(
            set(
                idx
                for entry in single_results[: args.exhaustive_pool_size]
                for idx in entry["subset"]
            ).union(best_state["subset"])
        )
        exhaustive_results = exhaustive_pool_search(
            dist_cache=dist_cache,
            labels=labels,
            pool_indices=pool_indices,
            target_shot=args.target_shot,
            aggregation_statistics=args.aggregation_statistics,
            grid_size=grid_size,
            image_hw=image_hw,
        )
        if exhaustive_results and metrics_key(exhaustive_results[0]) > metrics_key(best_state):
            best_state = evaluate_subset(
                dist_cache,
                labels,
                exhaustive_results[0]["subset"],
                args.aggregation_statistics,
                grid_size,
                image_hw,
            )
            print(
                f"Exhaustive reduced-pool search improved best subset to {best_state['subset']} "
                f"with f1={best_state['f1']:.6f}"
            )

    summary = {
        "object_name": object_name,
        "data_root": str(data_root),
        "num_train_refs": len(train_rows),
        "num_test_images": len(test_rows),
        "patch_count": int(train_patch_count),
        "feature_dim": int(feature_dim),
        "target_shot": args.target_shot,
        "aggregation_statistics": args.aggregation_statistics,
        "grid_size": list(grid_size),
        "image_hw": list(image_hw),
        "best_subset_indices": list(best_state["subset"]),
        "best_subset_image_names": named_subset(train_rows, best_state["subset"]),
        "best_f1": float(best_state["f1"]),
        "best_ap": float(best_state["ap"]),
        "best_auroc": float(best_state["auroc"]),
        "best_threshold": float(best_state["threshold"]),
        "elapsed_seconds": time.time() - start_time,
    }
    summary["cache_dir"] = str(cache_dir)
    write_json(summary, output_dir / "best_reference_search_summary.json")

    top_subset_rows = []
    for entry in (
        exhaustive_results[:25]
        if exhaustive_results
        else [
            {
                "subset": state["subset"],
                "f1": state["f1"],
                "ap": state["ap"],
                "auroc": state["auroc"],
                "threshold": state["threshold"],
                "precision": state["precision"],
                "recall": state["recall"],
            }
            for state in beam
        ]
    ):
        top_subset_rows.append(
            {
                "subset_indices": ";".join(str(idx) for idx in entry["subset"]),
                "subset_image_names": " | ".join(named_subset(train_rows, entry["subset"])),
                "f1": entry["f1"],
                "ap": entry["ap"],
                "auroc": entry["auroc"],
                "threshold": entry["threshold"],
                "precision": entry["precision"],
                "recall": entry["recall"],
            }
        )
    write_csv(top_subset_rows, output_dir / "top_reference_subsets.csv")

    print(f"Best {args.target_shot}-reference subset found:")
    for name in named_subset(train_rows, best_state["subset"]):
        print(f"  - {name}")
    print(
        f"F1={best_state['f1']:.6f} AP={best_state['ap']:.6f} "
        f"AUROC={best_state['auroc']:.6f} threshold={best_state['threshold']:.6f}"
    )


if __name__ == "__main__":
    main()
