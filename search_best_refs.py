import argparse
import csv
import json
import math
import os
from pathlib import Path

import faiss
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm
from PIL import Image

from src.backbones import get_model
from src.post_eval import max_anomaly_map
from src.utils import get_dataset_info, list_image_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CUSTOM")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--object_name", type=str, default="buttons")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--resolution", type=int, default=704)
    parser.add_argument("--preprocess", type=str, default="force_no_mask_no_rotation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--trials_per_shot", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--faiss_on_cpu", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_dir", type=str, default="search_results")
    parser.add_argument("--save_feature_cache", default=False, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def l2_normalize_copy(features):
    features = np.array(features, dtype=np.float32, copy=True)
    faiss.normalize_L2(features)
    return features


def build_knn_index(features_ref, faiss_on_cpu=True):
    features_ref = np.asarray(features_ref, dtype=np.float32)
    if faiss_on_cpu:
        index = faiss.IndexFlatL2(features_ref.shape[1])
    else:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])
    index.add(features_ref)
    return index


def best_f1_and_threshold(labels, scores):
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return float(f1_scores[best_idx]), float(thresholds[best_idx]), float(precisions[best_idx]), float(recalls[best_idx])


def gather_train_and_test_files(data_root, object_name):
    train_dir = os.path.join(data_root, object_name, "train", "good")
    train_files = list_image_files(train_dir)

    test_dir = os.path.join(data_root, object_name, "test")
    test_entries = []
    for anomaly_type in sorted(os.listdir(test_dir)):
        anomaly_dir = os.path.join(test_dir, anomaly_type)
        if not os.path.isdir(anomaly_dir):
            continue
        recursive_scan = any(os.path.isdir(os.path.join(anomaly_dir, entry)) for entry in os.listdir(anomaly_dir))
        for rel_file in list_image_files(anomaly_dir, recursive=recursive_scan):
            label = 0 if anomaly_type == "good" else 1
            test_entries.append({
                "rel_path": f"{anomaly_type}/{rel_file}".replace("\\", "/"),
                "full_path": os.path.join(anomaly_dir, rel_file),
                "label": label,
            })
    return train_files, test_entries


def extract_train_features(model, train_dir, train_files):
    train_features = []
    for rel_name in tqdm(train_files, desc="Extract train features"):
        full_path = os.path.join(train_dir, rel_name)
        image_tensor, grid_size = model.prepare_image(full_path)
        features = model.extract_features(image_tensor)
        train_features.append({
            "name": rel_name,
            "features": np.asarray(features, dtype=np.float32),
            "grid_size": tuple(grid_size),
        })
    return train_features


def extract_test_features(model, test_entries):
    test_features = []
    for entry in tqdm(test_entries, desc="Extract test features"):
        image_tensor, grid_size = model.prepare_image(entry["full_path"])
        features = model.extract_features(image_tensor)
        image_shape = tuple(np.asarray(Image.open(entry["full_path"])).shape)
        test_features.append({
            "name": entry["rel_path"],
            "label": entry["label"],
            "features": np.asarray(features, dtype=np.float32),
            "grid_size": tuple(grid_size),
            "image_shape": image_shape,
        })
    return test_features


def evaluate_subset(train_features, test_features, subset_indices, faiss_on_cpu=True):
    subset_names = [train_features[idx]["name"] for idx in subset_indices]
    ref_features = np.concatenate([train_features[idx]["features"] for idx in subset_indices], axis=0).astype(np.float32)
    ref_features = l2_normalize_copy(ref_features)
    index = build_knn_index(ref_features, faiss_on_cpu=faiss_on_cpu)

    labels = []
    scores = []
    for test_item in test_features:
        query_features = l2_normalize_copy(test_item["features"])
        distances, _ = index.search(query_features, k=1)
        distances = distances.squeeze() / 2.0
        patch_map = distances.reshape(test_item["grid_size"])
        score = max_anomaly_map(patch_map, test_item["image_shape"])
        labels.append(test_item["label"])
        scores.append(float(score))

    auroc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    f1, threshold, precision, recall = best_f1_and_threshold(labels, scores)
    return {
        "subset_indices": list(subset_indices),
        "subset_names": subset_names,
        "shot": len(subset_indices),
        "classification_AUROC": auroc,
        "classification_AP": ap,
        "classification_F1": f1,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
    }


def generate_random_subsets(num_items, shot, trials, rng):
    if shot > num_items:
        return []

    max_unique = min(trials, math.comb(num_items, shot) if shot <= num_items else 0)
    subsets = set()
    while len(subsets) < max_unique:
        subset = tuple(sorted(rng.choice(num_items, size=shot, replace=False).tolist()))
        subsets.add(subset)
    return [list(subset) for subset in sorted(subsets)]


def save_feature_cache(output_dir, train_features, test_features):
    cache_dir = Path(output_dir) / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump([{"name": item["name"], "grid_size": item["grid_size"]} for item in train_features], f, indent=2)
    with open(cache_dir / "test_manifest.json", "w", encoding="utf-8") as f:
        json.dump([
            {
                "name": item["name"],
                "label": item["label"],
                "grid_size": item["grid_size"],
                "image_shape": item["image_shape"],
            }
            for item in test_features
        ], f, indent=2)
    for idx, item in enumerate(train_features):
        np.save(cache_dir / f"train_{idx:03d}.npy", item["features"])
    for idx, item in enumerate(test_features):
        np.save(cache_dir / f"test_{idx:03d}.npy", item["features"])


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    objects, _, masking_default, rotation_default = get_dataset_info(
        args.dataset,
        args.preprocess,
        data_path=args.data_root,
    )
    if args.object_name not in objects:
        raise ValueError(f"Unknown object '{args.object_name}', available objects: {objects}")
    if masking_default[args.object_name]:
        raise ValueError("This search script currently assumes masking is disabled.")
    if rotation_default[args.object_name]:
        raise ValueError("This search script currently assumes rotation is disabled.")

    train_files, test_entries = gather_train_and_test_files(args.data_root, args.object_name)
    print(f"Train images: {len(train_files)}")
    print(f"Test images: {len(test_entries)}")

    model = get_model(args.model_name, "cuda", smaller_edge_size=args.resolution)
    train_dir = os.path.join(args.data_root, args.object_name, "train", "good")
    train_features = extract_train_features(model, train_dir, train_files)
    test_features = extract_test_features(model, test_entries)

    if args.save_feature_cache:
        save_feature_cache(output_dir, train_features, test_features)

    rng = np.random.default_rng(args.seed)
    results = []
    best_per_shot = {}
    csv_path = output_dir / "search_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "shot",
            "trial",
            "classification_AUROC",
            "classification_AP",
            "classification_F1",
            "threshold",
            "precision",
            "recall",
            "subset_names",
        ])

        for shot in args.shots:
            print(f"=== Search shot={shot} ===")
            if shot == 1:
                subsets = [[idx] for idx in range(len(train_features))]
            else:
                subsets = generate_random_subsets(len(train_features), shot, args.trials_per_shot, rng)

            shot_best = None
            for trial_idx, subset_indices in enumerate(tqdm(subsets, desc=f"Evaluate shot={shot}")):
                result = evaluate_subset(
                    train_features=train_features,
                    test_features=test_features,
                    subset_indices=subset_indices,
                    faiss_on_cpu=args.faiss_on_cpu,
                )
                result["trial"] = trial_idx
                results.append(result)
                writer.writerow([
                    result["shot"],
                    trial_idx,
                    result["classification_AUROC"],
                    result["classification_AP"],
                    result["classification_F1"],
                    result["threshold"],
                    result["precision"],
                    result["recall"],
                    "|".join(result["subset_names"]),
                ])
                csv_file.flush()

                if shot_best is None or result["classification_F1"] > shot_best["classification_F1"]:
                    shot_best = result

            best_per_shot[str(shot)] = shot_best
            print(
                f"Best shot={shot}: F1={shot_best['classification_F1']:.6f}, "
                f"AUROC={shot_best['classification_AUROC']:.6f}, AP={shot_best['classification_AP']:.6f}"
            )

    overall_best = max(results, key=lambda item: item["classification_F1"])
    summary = {
        "dataset": args.dataset,
        "object_name": args.object_name,
        "model_name": args.model_name,
        "resolution": args.resolution,
        "preprocess": args.preprocess,
        "aggregation_statistics": "max_anomaly_map",
        "search_seed": args.seed,
        "shots": args.shots,
        "trials_per_shot": args.trials_per_shot,
        "overall_best": overall_best,
        "best_per_shot": best_per_shot,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Overall best ===")
    print(json.dumps(overall_best, indent=2))


if __name__ == "__main__":
    main()
