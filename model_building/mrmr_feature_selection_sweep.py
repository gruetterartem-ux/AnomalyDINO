from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model_building.sklearn_eval_utils import (
    build_estimator,
    evaluate_subset,
    generate_default_k_values,
    score_value,
)
from model_building.roi_sklearn_groupcv import clean_label, load_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate mRMR feature selection with MI relevance and correlation redundancy, "
            "then sweep top-k feature subsets and evaluate classifiers with StratifiedGroupKFold."
        )
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--classifiers",
        nargs="*",
        default=("svm_rbf",),
        choices=("logreg", "svm_linear", "svm_rbf", "rf"),
    )
    parser.add_argument("--k-values", nargs="*", type=int, default=None)
    parser.add_argument(
        "--score-key",
        type=str,
        default="macro_f1",
        choices=("macro_f1", "macro_recall", "accuracy", "3d_f1", "3d_recall"),
    )
    parser.add_argument("--prefilter-top", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--rf-class-weight", type=str, default="balanced_subsample")
    parser.add_argument("--rf-n-estimators", type=int, default=500)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rf-max-features", type=str, default="sqrt")
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


def default_output_dir(features_dir: Path, explicit_output_dir: Path | None, score_key: str) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (features_dir / f"mrmr_topk_sweep_{score_key}").resolve()


def write_progress(progress: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    output_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")


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


def auto_prefilter_top(feature_dim: int, max_k: int) -> int:
    target = max(2048, max_k * 8)
    return int(min(feature_dim, target))


def safe_abs_corrcoef(matrix: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    return corr


def greedy_mrmr_ranking(
    relevance: np.ndarray,
    redundancy_matrix: np.ndarray,
    max_select: int,
    progress_file: Path,
    start_time: float,
) -> np.ndarray:
    prefilter_top = int(relevance.shape[0])
    selected_local: list[int] = []
    selected_mask = np.zeros(prefilter_top, dtype=bool)
    redundancy_sum = np.zeros(prefilter_top, dtype=np.float32)

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

        elapsed = time.time() - start_time
        mean_per_step = elapsed / float(iteration + 1)
        remaining = max(0, max_select - (iteration + 1))
        eta_seconds = mean_per_step * float(remaining)
        write_progress(
            {
                "status": "running",
                "phase": "mrmr_greedy_selection",
                "selected_features": int(iteration + 1),
                "target_selected_features": int(max_select),
                "elapsed_seconds": float(elapsed),
                "eta_seconds": float(eta_seconds),
                "eta_human": format_seconds(eta_seconds),
                "latest_selected_local_index": int(best_local),
                "latest_selected_relevance": float(relevance[best_local]),
                "latest_selected_score": float(scores[best_local]),
            },
            progress_file,
        )
        if (iteration + 1) % 10 == 0 or iteration + 1 == max_select:
            print(
                f"[mRMR] selected {iteration + 1}/{max_select} features | ETA {format_seconds(eta_seconds)}",
                flush=True,
            )

    return np.asarray(selected_local, dtype=np.int32)


def main() -> None:
    args = parse_args()
    features_dir = args.features_dir.resolve()
    labels_file = (args.labels_file or (features_dir / "roi_feature_table.csv")).resolve()
    output_dir = default_output_dir(features_dir, args.output_dir, args.score_key)
    ensure_dir(output_dir)
    progress_file = output_dir / "progress.json"
    start_time = time.time()

    features, table = load_inputs(features_dir, labels_file)
    table = table.copy()
    table["label"] = table["label"].map(clean_label)
    ignore_labels = {clean_label(label) for label in args.ignore_labels}
    valid_labels = [clean_label(label) for label in args.valid_labels]
    valid_mask = table["label"].isin(valid_labels) & ~table["label"].isin(ignore_labels)
    labeled_table = table.loc[valid_mask].copy()
    if labeled_table.empty:
        raise ValueError("No labeled ROIs found after filtering labels.")

    labeled_features = features[labeled_table["feature_index"].to_numpy()].astype(np.float32)
    groups = labeled_table["group_id"].astype(str).to_numpy()
    label_encoder = LabelEncoder()
    label_encoder.fit(valid_labels)
    y = label_encoder.transform(labeled_table["label"].to_numpy())
    y_labels = label_encoder.inverse_transform(y)
    class_names = list(label_encoder.classes_)

    k_values = args.k_values if args.k_values else generate_default_k_values(labeled_features.shape[1])
    k_values = sorted({int(k) for k in k_values if 1 <= int(k) <= labeled_features.shape[1]})
    if not k_values:
        raise ValueError("No valid k values to evaluate.")
    max_k = max(k_values)
    prefilter_top = int(args.prefilter_top) if args.prefilter_top is not None else auto_prefilter_top(labeled_features.shape[1], max_k)
    if prefilter_top < max_k:
        raise ValueError(f"prefilter_top={prefilter_top} must be >= max_k={max_k}")

    write_progress(
        {
            "status": "running",
            "phase": "initializing",
            "features_dir": str(features_dir),
            "output_dir": str(output_dir),
            "num_labeled_rois": int(len(labeled_table)),
            "feature_dim": int(labeled_features.shape[1]),
            "k_values": [int(k) for k in k_values],
            "prefilter_top": int(prefilter_top),
            "elapsed_seconds": 0.0,
        },
        progress_file,
    )

    print("[mRMR] standardizing features for relevance/redundancy estimation", flush=True)
    selector_scaler = StandardScaler()
    X_scaled = selector_scaler.fit_transform(labeled_features).astype(np.float32)

    write_progress(
        {
            "status": "running",
            "phase": "mutual_information",
            "feature_dim": int(labeled_features.shape[1]),
            "prefilter_top": int(prefilter_top),
            "elapsed_seconds": float(time.time() - start_time),
        },
        progress_file,
    )
    print("[mRMR] computing mutual information relevance", flush=True)
    relevance = mutual_info_classif(X_scaled, y, discrete_features=False, random_state=int(args.random_state))
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    prefilter_indices = np.argsort(-relevance, kind="stable")[:prefilter_top]
    prefilter_relevance = relevance[prefilter_indices]

    relevance_rows = [
        {
            "rank_by_relevance": int(rank + 1),
            "feature_index": int(feature_index),
            "mutual_information": float(relevance[feature_index]),
        }
        for rank, feature_index in enumerate(prefilter_indices)
    ]
    write_csv(relevance_rows, output_dir / "mrmr_prefilter_relevance.csv")

    write_progress(
        {
            "status": "running",
            "phase": "redundancy_matrix",
            "prefilter_top": int(prefilter_top),
            "elapsed_seconds": float(time.time() - start_time),
        },
        progress_file,
    )
    print(f"[mRMR] computing redundancy matrix on top-{prefilter_top} relevance features", flush=True)
    X_prefilter = X_scaled[:, prefilter_indices]
    redundancy_matrix = safe_abs_corrcoef(X_prefilter)

    ranked_local = greedy_mrmr_ranking(
        relevance=prefilter_relevance,
        redundancy_matrix=redundancy_matrix,
        max_select=max_k,
        progress_file=progress_file,
        start_time=start_time,
    )
    ranked_indices = prefilter_indices[ranked_local]

    all_rows: list[Dict[str, object]] = []
    best_runs: dict[str, dict[str, object]] = {
        classifier: {
            "score": -np.inf,
            "accuracy": -np.inf,
            "k": None,
            "metrics": None,
            "fold_rows": None,
            "oof_rows": None,
            "selected_indices": None,
        }
        for classifier in args.classifiers
    }

    total_eval_steps = len(args.classifiers) * len(k_values)
    completed_eval_steps = 0
    for classifier in args.classifiers:
        print(f"[mRMR] evaluating classifier={classifier}", flush=True)
        for k in k_values:
            write_progress(
                {
                    "status": "running",
                    "phase": "subset_evaluation",
                    "classifier": classifier,
                    "current_k": int(k),
                    "completed_subset_evals": int(completed_eval_steps),
                    "total_subset_evals": int(total_eval_steps),
                    "elapsed_seconds": float(time.time() - start_time),
                },
                progress_file,
            )
            selected_indices = ranked_indices[:k]
            X_subset = labeled_features[:, selected_indices]
            overall_metrics, fold_rows, oof_rows = evaluate_subset(
                X=X_subset,
                y=y,
                y_labels=y_labels,
                groups=groups,
                class_names=class_names,
                classifier=classifier,
                args=args,
            )

            row = {
                "classifier": classifier,
                "k": int(k),
                "accuracy": overall_metrics["accuracy"],
                "macro_precision": overall_metrics["macro_precision"],
                "macro_recall": overall_metrics["macro_recall"],
                "macro_f1": overall_metrics["macro_f1"],
                "2d_precision": float(overall_metrics["classification_report"]["2d"]["precision"]),
                "2d_recall": float(overall_metrics["classification_report"]["2d"]["recall"]),
                "2d_f1": float(overall_metrics["classification_report"]["2d"]["f1-score"]),
                "3d_precision": float(overall_metrics["classification_report"]["3d"]["precision"]),
                "3d_recall": float(overall_metrics["classification_report"]["3d"]["recall"]),
                "3d_f1": float(overall_metrics["classification_report"]["3d"]["f1-score"]),
            }
            all_rows.append(row)

            primary_score = score_value(overall_metrics, args.score_key)
            current_best = best_runs[classifier]
            if (
                primary_score > current_best["score"]
                or (
                    np.isclose(primary_score, current_best["score"])
                    and overall_metrics["accuracy"] > current_best["accuracy"]
                )
                or (
                    np.isclose(primary_score, current_best["score"])
                    and np.isclose(overall_metrics["accuracy"], current_best["accuracy"])
                    and (current_best["k"] is None or int(k) < int(current_best["k"]))
                )
            ):
                current_best.update(
                    {
                        "score": float(primary_score),
                        "accuracy": float(overall_metrics["accuracy"]),
                        "k": int(k),
                        "metrics": overall_metrics,
                        "fold_rows": fold_rows,
                        "oof_rows": oof_rows,
                        "selected_indices": selected_indices.copy(),
                    }
                )

            completed_eval_steps += 1
            elapsed = time.time() - start_time
            eta_seconds = ((elapsed / completed_eval_steps) * max(0, total_eval_steps - completed_eval_steps)) if completed_eval_steps > 0 else None
            print(
                f"[mRMR] {classifier} | k={k} | {args.score_key}={primary_score:.4f} | ETA {format_seconds(eta_seconds)}",
                flush=True,
            )
            write_csv(all_rows, output_dir / "results.csv")
            write_progress(
                {
                    "status": "running",
                    "phase": "subset_evaluation",
                    "classifier": classifier,
                    "current_k": int(k),
                    "completed_subset_evals": int(completed_eval_steps),
                    "total_subset_evals": int(total_eval_steps),
                    "elapsed_seconds": float(elapsed),
                    "eta_seconds": eta_seconds,
                    "eta_human": format_seconds(eta_seconds),
                    "current_score": float(primary_score),
                    "current_accuracy": float(overall_metrics["accuracy"]),
                },
                progress_file,
            )

    results_csv = output_dir / "results.csv"
    top_results_csv = output_dir / "top_results.csv"
    best_summary_json = output_dir / "best_by_classifier.json"
    config_json = output_dir / "config.json"

    sorted_rows = sorted(
        all_rows,
        key=lambda row: (float(row[args.score_key]), float(row["accuracy"]), -int(row["k"])),
        reverse=True,
    )
    write_csv(all_rows, results_csv)
    write_csv(sorted_rows, top_results_csv)

    best_summary: dict[str, object] = {
        "features_dir": str(features_dir),
        "labels_file": str(labels_file),
        "score_key": args.score_key,
        "selector": "mrmr_mi_relevance_abs_corr_redundancy",
        "prefilter_top": int(prefilter_top),
        "classifiers": list(args.classifiers),
        "k_values": [int(k) for k in k_values],
        "num_labeled_rois": int(len(labeled_table)),
        "num_groups": int(pd.Series(groups).nunique()),
        "class_names": class_names,
        "class_counts": {
            class_name: int((labeled_table["label"] == class_name).sum())
            for class_name in class_names
        },
        "best": {},
    }

    for classifier, info in best_runs.items():
        if info["k"] is None:
            continue
        classifier_dir = output_dir / f"best_{classifier}"
        ensure_dir(classifier_dir)

        selected_indices = np.asarray(info["selected_indices"], dtype=np.int32)
        feature_rows = [
            {
                "rank": int(rank + 1),
                "feature_index": int(feature_index),
                "mutual_information": float(relevance[feature_index]),
            }
            for rank, feature_index in enumerate(selected_indices)
        ]
        write_csv(feature_rows, classifier_dir / "selected_features.csv")
        np.save(classifier_dir / "selected_feature_indices.npy", selected_indices)
        write_csv(info["fold_rows"], classifier_dir / "fold_metrics.csv")

        oof_rows: list[Dict[str, object]] = []
        labeled_table_reset = labeled_table.reset_index(drop=True)
        for base_row, pred_row in zip(labeled_table_reset.to_dict(orient="records"), info["oof_rows"]):
            merged = dict(base_row)
            merged.update(pred_row)
            oof_rows.append(merged)
        write_csv(oof_rows, classifier_dir / "oof_predictions.csv")

        summary = {
            "classifier": classifier,
            "score_key": args.score_key,
            "best_k": int(info["k"]),
            "overall": info["metrics"],
            "folds": info["fold_rows"],
        }
        write_json(summary, classifier_dir / "summary.json")
        best_summary["best"][classifier] = summary

    write_json(
        {
            "features_dir": str(features_dir),
            "labels_file": str(labels_file),
            "selector": "mrmr_mi_relevance_abs_corr_redundancy",
            "prefilter_top": int(prefilter_top),
            "score_key": args.score_key,
            "classifiers": list(args.classifiers),
            "k_values": [int(k) for k in k_values],
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "c_value": float(args.c_value),
            "max_iter": int(args.max_iter),
            "gamma": args.gamma,
            "class_weight": None if args.class_weight == "none" else args.class_weight,
            "rf_class_weight": None if args.rf_class_weight == "none" else args.rf_class_weight,
        },
        config_json,
    )
    write_json(best_summary, best_summary_json)

    write_progress(
        {
            "status": "completed",
            "phase": "done",
            "elapsed_seconds": float(time.time() - start_time),
            "best_summary_json": str(best_summary_json),
        },
        progress_file,
    )

    print(f"Saved results: {results_csv}")
    print(f"Saved top results: {top_results_csv}")
    print(f"Saved best summary: {best_summary_json}")
    for classifier, info in best_runs.items():
        if info["k"] is not None:
            print(
                f"{classifier}: best_k={info['k']} | {args.score_key}={info['score']:.4f} | "
                f"accuracy={info['accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()
