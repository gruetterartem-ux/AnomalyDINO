from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler

from extract_labeled_roi_overthreshold_multilayer_maxminmean_features import DEFAULT_EXPERIMENT_DIR


DEFAULT_FEATURES_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_overthreshold_overlap_multilayer_l1to12_maxminmean_features_labeled"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Boruta-style wrapper feature selection on the max+min+mean ROI feature set, "
            "using an approximate mRMR ranking as a prefilter."
        )
    )
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefilter-k", type=int, default=2000)
    parser.add_argument("--mrmr-prefilter-top", type=int, default=4096)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--shadow-percentile", type=float, default=100.0)
    parser.add_argument("--n-estimators", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--class-weight", type=str, default="balanced_subsample")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--valid-labels", nargs="*", default=("2D", "3D"))
    parser.add_argument("--ignore-labels", nargs="*", default=("skip", "unclear", "unknown"))
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


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


def default_output_dir(features_dir: Path, explicit_output_dir: Path | None, prefilter_k: int) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (features_dir / f"boruta_mrmr_prefilter{int(prefilter_k)}").resolve()


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def write_progress(progress: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    output_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().lower()


def load_inputs(features_dir: Path, valid_labels: list[str], ignore_labels: list[str]) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    features_file = features_dir / "roi_features_mean.npy"
    metadata_file = features_dir / "roi_feature_table.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Feature table not found: {metadata_file}")

    X = np.load(features_file).astype(np.float32)
    table = pd.read_csv(metadata_file)
    if len(X) != len(table):
        raise ValueError(f"Length mismatch: {len(X)} features vs {len(table)} metadata rows")

    table = table.copy()
    table["label_clean"] = table["label"].map(clean_label)
    valid_set = {clean_label(label) for label in valid_labels}
    ignore_set = {clean_label(label) for label in ignore_labels}
    mask = table["label_clean"].isin(valid_set) & ~table["label_clean"].isin(ignore_set)
    table = table.loc[mask].reset_index(drop=True)
    X = X[mask.to_numpy()]

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(valid_set))
    y_labels = table["label_clean"].to_numpy()
    y = label_encoder.transform(y_labels)
    class_names = list(label_encoder.classes_)
    return X, table, y, y_labels, class_names


def safe_abs_corrcoef(matrix: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    return corr


def greedy_mrmr_rank(relevance: np.ndarray, redundancy_matrix: np.ndarray, max_select: int) -> np.ndarray:
    selected_local: list[int] = []
    selected_mask = np.zeros(relevance.shape[0], dtype=bool)
    redundancy_sum = np.zeros(relevance.shape[0], dtype=np.float32)
    for iteration in range(max_select):
        if iteration == 0:
            scores = relevance.copy()
        else:
            scores = relevance - (redundancy_sum / float(iteration))
        scores[selected_mask] = -np.inf
        best_local = int(np.argmax(scores))
        selected_local.append(best_local)
        selected_mask[best_local] = True
        redundancy_sum += redundancy_matrix[:, best_local]
    return np.asarray(selected_local, dtype=np.int32)


def decode_feature_index(feature_index: int, layer_dim: int = 768) -> dict[str, object]:
    components = ("max", "min", "mean")
    per_layer = layer_dim * len(components)
    layer_zero = int(feature_index) // per_layer
    remainder = int(feature_index) % per_layer
    component_index = remainder // layer_dim
    within_component_dim = remainder % layer_dim
    return {
        "layer_index": int(layer_zero + 1),
        "aggregation_component": components[int(component_index)],
        "within_component_dim": int(within_component_dim),
    }


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir.resolve()
    output_dir = default_output_dir(features_dir, args.output_dir, int(args.prefilter_k))
    ensure_dir(output_dir)
    progress_file = output_dir / "progress.json"
    start_time = time.time()

    X_raw, table, y, y_labels, class_names = load_inputs(
        features_dir=features_dir,
        valid_labels=list(args.valid_labels),
        ignore_labels=list(args.ignore_labels),
    )
    feature_dim = int(X_raw.shape[1])
    prefilter_k = int(min(max(1, int(args.prefilter_k)), feature_dim))
    mrmr_prefilter_top = int(min(max(prefilter_k, int(args.mrmr_prefilter_top)), feature_dim))

    write_progress(
        {
            "status": "running",
            "phase": "initializing",
            "features_dir": str(features_dir),
            "output_dir": str(output_dir),
            "num_samples": int(len(table)),
            "feature_dim": feature_dim,
            "prefilter_k": int(prefilter_k),
            "mrmr_prefilter_top": int(mrmr_prefilter_top),
            "elapsed_seconds": 0.0,
        },
        progress_file,
    )

    print("[Boruta] standardizing features for mRMR prefilter", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    write_progress(
        {
            "status": "running",
            "phase": "mrmr_relevance",
            "feature_dim": feature_dim,
            "mrmr_prefilter_top": int(mrmr_prefilter_top),
            "elapsed_seconds": float(time.time() - start_time),
        },
        progress_file,
    )
    print("[Boruta] computing mRMR relevance", flush=True)
    relevance = mutual_info_classif(X_scaled, y, discrete_features=False, random_state=int(args.random_state))
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mrmr_seed_indices = np.argsort(-relevance, kind="stable")[:mrmr_prefilter_top]

    write_progress(
        {
            "status": "running",
            "phase": "mrmr_redundancy",
            "mrmr_prefilter_top": int(mrmr_prefilter_top),
            "elapsed_seconds": float(time.time() - start_time),
        },
        progress_file,
    )
    print(f"[Boruta] computing redundancy matrix on top-{mrmr_prefilter_top} relevance features", flush=True)
    redundancy = safe_abs_corrcoef(X_scaled[:, mrmr_seed_indices])
    ranked_local = greedy_mrmr_rank(relevance[mrmr_seed_indices], redundancy, prefilter_k)
    prefilter_indices = mrmr_seed_indices[ranked_local]

    prefilter_rows: list[dict[str, object]] = []
    for rank, feature_index in enumerate(prefilter_indices, start=1):
        row = {
            "mrmr_rank": int(rank),
            "feature_index": int(feature_index),
            "mutual_information": float(relevance[feature_index]),
        }
        row.update(decode_feature_index(int(feature_index)))
        prefilter_rows.append(row)
    write_csv(prefilter_rows, output_dir / "mrmr_prefilter_top_features.csv")
    np.save(output_dir / "mrmr_prefilter_feature_indices.npy", prefilter_indices.astype(np.int32))

    X_prefilter = X_raw[:, prefilter_indices].astype(np.float32)
    feature_names = [f"feature_{int(idx)}" for idx in prefilter_indices]
    n_features = int(X_prefilter.shape[1])
    statuses = np.zeros(n_features, dtype=np.int8)  # 0 tentative, 1 confirmed, -1 rejected
    hits = np.zeros(n_features, dtype=np.int32)
    real_importance_sum = np.zeros(n_features, dtype=np.float64)
    real_importance_max = np.zeros(n_features, dtype=np.float64)
    shadow_threshold_history: list[float] = []
    iteration_rows: list[dict[str, object]] = []
    no_change_rounds = 0

    for iteration in range(1, int(args.max_iter) + 1):
        iter_start = time.time()
        tentative_mask = statuses == 0
        num_tentative = int(tentative_mask.sum())
        num_confirmed = int((statuses == 1).sum())
        num_rejected = int((statuses == -1).sum())

        if num_tentative == 0:
            write_progress(
                {
                    "status": "completed",
                    "phase": "boruta_done",
                    "iteration": int(iteration - 1),
                    "max_iter": int(args.max_iter),
                    "num_confirmed": num_confirmed,
                    "num_rejected": num_rejected,
                    "num_tentative": num_tentative,
                    "elapsed_seconds": float(time.time() - start_time),
                    "eta_seconds": 0.0,
                    "eta_human": "0s",
                    "stop_reason": "no_tentative_features_left",
                },
                progress_file,
            )
            break

        shadow = X_prefilter[:, tentative_mask].copy()
        rng = np.random.default_rng(int(args.random_state) + iteration)
        for column_index in range(shadow.shape[1]):
            rng.shuffle(shadow[:, column_index])

        X_boruta = np.concatenate([X_prefilter, shadow], axis=1)
        model = RandomForestClassifier(
            n_estimators=int(args.n_estimators),
            max_depth=None if args.max_depth is None else int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            max_features=str(args.max_features),
            class_weight=None if args.class_weight == "none" else str(args.class_weight),
            n_jobs=int(args.n_jobs),
            random_state=int(args.random_state) + iteration,
        )
        model.fit(X_boruta, y)
        importances = model.feature_importances_.astype(np.float64)
        real_importances = importances[:n_features]
        shadow_importances = importances[n_features:]
        threshold = float(np.percentile(shadow_importances, float(args.shadow_percentile)))
        shadow_threshold_history.append(threshold)

        hits += (real_importances > threshold).astype(np.int32)
        real_importance_sum += real_importances
        real_importance_max = np.maximum(real_importance_max, real_importances)

        changed = False
        bonferroni = max(1, num_tentative)
        for feature_idx in np.where(tentative_mask)[0]:
            p_accept = binomtest(int(hits[feature_idx]), iteration, 0.5, alternative="greater").pvalue
            p_reject = binomtest(int(hits[feature_idx]), iteration, 0.5, alternative="less").pvalue
            if p_accept < (float(args.alpha) / float(bonferroni)):
                statuses[feature_idx] = 1
                changed = True
            elif p_reject < (float(args.alpha) / float(bonferroni)):
                statuses[feature_idx] = -1
                changed = True

        no_change_rounds = 0 if changed else (no_change_rounds + 1)
        elapsed = time.time() - start_time
        mean_iter = elapsed / float(iteration)
        eta_seconds = mean_iter * float(max(0, int(args.max_iter) - iteration))
        iter_seconds = time.time() - iter_start

        num_confirmed = int((statuses == 1).sum())
        num_rejected = int((statuses == -1).sum())
        num_tentative = int((statuses == 0).sum())
        iteration_rows.append(
            {
                "iteration": int(iteration),
                "shadow_threshold": float(threshold),
                "num_confirmed": num_confirmed,
                "num_rejected": num_rejected,
                "num_tentative": num_tentative,
                "iteration_seconds": float(iter_seconds),
                "elapsed_seconds": float(elapsed),
                "eta_seconds": float(eta_seconds),
            }
        )
        write_csv(iteration_rows, output_dir / "boruta_iteration_log.csv")
        write_progress(
            {
                "status": "running",
                "phase": "boruta_iteration",
                "iteration": int(iteration),
                "max_iter": int(args.max_iter),
                "num_confirmed": num_confirmed,
                "num_rejected": num_rejected,
                "num_tentative": num_tentative,
                "shadow_threshold": float(threshold),
                "iteration_seconds": float(iter_seconds),
                "elapsed_seconds": float(elapsed),
                "eta_seconds": float(eta_seconds),
                "eta_human": format_seconds(eta_seconds),
                "no_change_rounds": int(no_change_rounds),
                "patience": int(args.patience),
            },
            progress_file,
        )
        print(
            f"[Boruta] iter {iteration}/{args.max_iter} | confirmed={num_confirmed} | "
            f"rejected={num_rejected} | tentative={num_tentative} | ETA {format_seconds(eta_seconds)}",
            flush=True,
        )

        if num_tentative == 0:
            break
        if int(args.patience) > 0 and no_change_rounds >= int(args.patience):
            write_progress(
                {
                    "status": "completed",
                    "phase": "boruta_done",
                    "iteration": int(iteration),
                    "max_iter": int(args.max_iter),
                    "num_confirmed": num_confirmed,
                    "num_rejected": num_rejected,
                    "num_tentative": num_tentative,
                    "elapsed_seconds": float(elapsed),
                    "eta_seconds": 0.0,
                    "eta_human": "0s",
                    "stop_reason": "patience_exhausted",
                },
                progress_file,
            )
            break

    mean_importance = real_importance_sum / float(max(1, len(iteration_rows)))
    status_name = {1: "confirmed", 0: "tentative", -1: "rejected"}
    all_feature_rows: list[dict[str, object]] = []
    for local_idx, feature_index in enumerate(prefilter_indices):
        row = {
            "prefilter_rank": int(local_idx + 1),
            "feature_index": int(feature_index),
            "status": status_name[int(statuses[local_idx])],
            "hits": int(hits[local_idx]),
            "hit_rate": float(hits[local_idx] / max(1, len(iteration_rows))),
            "mean_importance": float(mean_importance[local_idx]),
            "max_importance": float(real_importance_max[local_idx]),
            "mutual_information": float(relevance[feature_index]),
        }
        row.update(decode_feature_index(int(feature_index)))
        all_feature_rows.append(row)

    confirmed_rows = [row for row in all_feature_rows if row["status"] == "confirmed"]
    tentative_rows = [row for row in all_feature_rows if row["status"] == "tentative"]
    rejected_rows = [row for row in all_feature_rows if row["status"] == "rejected"]
    confirmed_rows.sort(key=lambda row: (float(row["hit_rate"]), float(row["mean_importance"])), reverse=True)
    tentative_rows.sort(key=lambda row: (float(row["hit_rate"]), float(row["mean_importance"])), reverse=True)
    rejected_rows.sort(key=lambda row: (float(row["hit_rate"]), float(row["mean_importance"])), reverse=True)

    write_csv(all_feature_rows, output_dir / "boruta_all_features.csv")
    write_csv(confirmed_rows, output_dir / "boruta_confirmed_features.csv")
    write_csv(tentative_rows, output_dir / "boruta_tentative_features.csv")
    write_csv(rejected_rows, output_dir / "boruta_rejected_features.csv")
    np.save(
        output_dir / "boruta_confirmed_feature_indices.npy",
        np.asarray([row["feature_index"] for row in confirmed_rows], dtype=np.int32),
    )
    np.save(
        output_dir / "boruta_tentative_feature_indices.npy",
        np.asarray([row["feature_index"] for row in tentative_rows], dtype=np.int32),
    )

    layer_counter: dict[str, dict[str, int]] = {}
    component_counter: dict[str, int] = {}
    for row in confirmed_rows:
        layer_key = str(row["layer_index"])
        component = str(row["aggregation_component"])
        layer_counter.setdefault(layer_key, {"max": 0, "min": 0, "mean": 0})
        layer_counter[layer_key][component] += 1
        component_counter[component] = component_counter.get(component, 0) + 1

    summary = {
        "features_dir": str(features_dir),
        "output_dir": str(output_dir),
        "num_samples": int(len(table)),
        "class_names": class_names,
        "class_counts": {
            class_name: int((table["label_clean"] == class_name).sum())
            for class_name in class_names
        },
        "feature_dim": feature_dim,
        "selector": "boruta_shadow_random_forest",
        "mrmr_prefilter_selector": "mrmr_mi_relevance_abs_corr_redundancy",
        "prefilter_k": int(prefilter_k),
        "mrmr_prefilter_top": int(mrmr_prefilter_top),
        "boruta_max_iter": int(args.max_iter),
        "boruta_alpha": float(args.alpha),
        "shadow_percentile": float(args.shadow_percentile),
        "rf_n_estimators": int(args.n_estimators),
        "rf_max_depth": None if args.max_depth is None else int(args.max_depth),
        "rf_min_samples_leaf": int(args.min_samples_leaf),
        "rf_max_features": str(args.max_features),
        "rf_class_weight": None if args.class_weight == "none" else str(args.class_weight),
        "num_iterations_run": int(len(iteration_rows)),
        "num_confirmed": int(len(confirmed_rows)),
        "num_tentative": int(len(tentative_rows)),
        "num_rejected": int(len(rejected_rows)),
        "confirmed_component_counts": component_counter,
        "confirmed_layer_component_counts": layer_counter,
        "mean_shadow_threshold": float(np.mean(shadow_threshold_history)) if shadow_threshold_history else 0.0,
        "elapsed_seconds": float(time.time() - start_time),
    }
    write_json(summary, output_dir / "summary.json")

    write_progress(
        {
            "status": "completed",
            "phase": "done",
            "num_iterations_run": int(len(iteration_rows)),
            "num_confirmed": int(len(confirmed_rows)),
            "num_tentative": int(len(tentative_rows)),
            "num_rejected": int(len(rejected_rows)),
            "elapsed_seconds": float(time.time() - start_time),
            "eta_seconds": 0.0,
            "eta_human": "0s",
            "summary_json": str(output_dir / "summary.json"),
        },
        progress_file,
    )

    print(
        f"[Boruta] completed | confirmed={len(confirmed_rows)} | tentative={len(tentative_rows)} | "
        f"rejected={len(rejected_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
