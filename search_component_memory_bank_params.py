from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from component_labeling_app.session_io import load_session
from component_memory_bank.data_io import load_patch_features, load_patch_scores, load_run_samples
from component_memory_bank.export import write_csv, write_json
from component_memory_bank.inference import compute_patch_class_scores, summarize_components
from component_memory_bank.memory_bank import (
    ComponentLabelRecord,
    ManualPatchLabelRecord,
    build_memory_banks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search the component memory-bank kNN decision parameters "
            "with fold-safe part-level cross-validation."
        )
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-group", type=str, default="test/bad")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 10],
    )
    parser.add_argument(
        "--anomaly-threshold-values",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--tau-s-values",
        type=float,
        nargs="+",
        default=[-0.1, -0.05, 0.0, 0.05, 0.1, 0.15],
    )
    parser.add_argument(
        "--tau-n-values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--tau-p-values",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "f1_3d", "f1_2d", "accuracy", "macro_recall", "recall_3d", "precision_3d"],
    )
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def _component_label_records(df: pd.DataFrame) -> list[ComponentLabelRecord]:
    labeled = df[df["label"].fillna("").astype(str).isin(["2D", "3D", "skip"])].copy()
    records: list[ComponentLabelRecord] = []
    for row in labeled.itertuples(index=False):
        records.append(
            ComponentLabelRecord(
                object_name=str(row.object_name),
                sample=str(row.sample).replace("\\", "/"),
                component_id=int(row.component_id),
                anomaly_threshold=float(row.anomaly_threshold),
                top_k=int(row.top_k),
                label=str(row.label),
            )
        )
    return records


def _manual_patch_label_records(df: pd.DataFrame) -> list[ManualPatchLabelRecord]:
    if df.empty:
        return []
    labeled = df[df["label"].fillna("").astype(str).isin(["2D", "3D"])].copy()
    records: list[ManualPatchLabelRecord] = []
    for row in labeled.itertuples(index=False):
        records.append(
            ManualPatchLabelRecord(
                object_name=str(row.object_name),
                sample=str(row.sample).replace("\\", "/"),
                row=int(row.row),
                col=int(row.col),
                patch_index=int(row.patch_index),
                anomaly_score=float(row.anomaly_score),
                label=str(row.label),
            )
        )
    return records


def _binary_metrics_from_cm(cm: np.ndarray) -> dict[str, float | list[list[int]]]:
    cm = np.asarray(cm, dtype=np.int64)
    a, b = int(cm[0, 0]), int(cm[0, 1])
    c, d = int(cm[1, 0]), int(cm[1, 1])
    total = max(a + b + c + d, 1)

    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        return float(precision), float(recall), float(f1)

    precision_2d, recall_2d, f1_2d = _prf(a, c, b)
    precision_3d, recall_3d, f1_3d = _prf(d, b, c)
    accuracy = float((a + d) / total)
    macro_precision = float((precision_2d + precision_3d) / 2.0)
    macro_recall = float((recall_2d + recall_3d) / 2.0)
    macro_f1 = float((f1_2d + f1_3d) / 2.0)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "precision_2d": precision_2d,
        "recall_2d": recall_2d,
        "f1_2d": f1_2d,
        "precision_3d": precision_3d,
        "recall_3d": recall_3d,
        "f1_3d": f1_3d,
        "confusion_matrix": [[a, b], [c, d]],
    }


def _objective_value(metrics: dict[str, float | list[list[int]]], objective: str) -> float:
    return float(metrics[objective])


def _default_threshold_grid(default_threshold: float) -> list[float]:
    raw = [
        max(default_threshold * 0.70, 1e-6),
        max(default_threshold * 0.85, 1e-6),
        default_threshold,
        default_threshold * 1.15,
        default_threshold * 1.35,
    ]
    grid = sorted({round(float(value), 6) for value in raw})
    return grid


def _slug_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()

    context, component_df, part_df, manual_patch_df, run_args, sample_inventory = load_session(args.session_dir)
    samples = load_run_samples(context.experiment_dir, seed=context.seed)
    sample_map = {sample.sample: sample for sample in samples}

    eval_parts = part_df[
        (part_df["evaluation_group"] == args.evaluation_group)
        & (part_df["part_label"].fillna("").astype(str).isin(["2D", "3D"]))
    ].copy()
    if eval_parts.empty:
        raise ValueError("No evaluation parts with part_label in {2D, 3D} were found.")

    valid_samples = set(eval_parts["sample"].tolist())
    component_df = component_df[component_df["sample"].isin(valid_samples)].copy()
    manual_patch_df = manual_patch_df[manual_patch_df["sample"].isin(valid_samples)].copy()

    eval_parts.sort_values("sample", inplace=True)
    eval_parts.reset_index(drop=True, inplace=True)
    part_samples = eval_parts["sample"].tolist()
    y = eval_parts["part_label"].astype(str).tolist()
    y_index = np.array([0 if label == "2D" else 1 for label in y], dtype=np.int8)

    default_threshold = float(context.anomaly_threshold)
    anomaly_threshold_values = (
        sorted({round(float(v), 6) for v in args.anomaly_threshold_values})
        if args.anomaly_threshold_values is not None
        else _default_threshold_grid(default_threshold)
    )
    k_values = sorted({int(v) for v in args.k_values})
    tau_s_values = sorted({float(v) for v in args.tau_s_values})
    tau_n_values = sorted({int(v) for v in args.tau_n_values})
    tau_p_values = sorted({float(v) for v in args.tau_p_values})

    k_thr_pairs = list(product(k_values, anomaly_threshold_values))
    tau_triplets = list(product(tau_s_values, tau_n_values, tau_p_values))
    full_combos = [
        (k_value, anomaly_threshold, tau_s, tau_n, tau_p)
        for (k_value, anomaly_threshold) in k_thr_pairs
        for (tau_s, tau_n, tau_p) in tau_triplets
    ]
    combo_counts = np.zeros((len(full_combos), 2, 2), dtype=np.int32)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(part_samples, y), start=1):
        train_samples = {part_samples[idx] for idx in train_idx}
        valid_samples_fold = [part_samples[idx] for idx in valid_idx]
        valid_true_idx = {sample_name: int(y_index[idx]) for idx, sample_name in zip(valid_idx, valid_samples_fold)}

        train_component_records = _component_label_records(
            component_df[component_df["sample"].isin(train_samples)].copy()
        )
        train_manual_records = _manual_patch_label_records(
            manual_patch_df[manual_patch_df["sample"].isin(train_samples)].copy()
        )

        bundle = build_memory_banks(samples, train_component_records, train_manual_records)
        bank_2d = bundle.features_2d.astype(np.float32, copy=False)
        bank_3d = bundle.features_3d.astype(np.float32, copy=False)

        cached_component_arrays: dict[tuple[str, int, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for sample_name in valid_samples_fold:
            sample = sample_map[sample_name]
            patch_features, grid_size = load_patch_features(sample)
            anomaly_scores = load_patch_scores(sample)
            if tuple(anomaly_scores.shape) != tuple(grid_size):
                raise ValueError(
                    f"Grid mismatch for {sample.sample!r}: feature cache {grid_size}, anomaly grid {anomaly_scores.shape}."
                )

            for k_value, anomaly_threshold in k_thr_pairs:
                patch_scores = compute_patch_class_scores(
                    patch_features=patch_features,
                    anomaly_scores=anomaly_scores,
                    bank_2d=bank_2d,
                    bank_3d=bank_3d,
                    k_neighbors=k_value,
                    anomaly_threshold=anomaly_threshold,
                )
                component_summaries = summarize_components(
                    anomaly_scores=anomaly_scores,
                    active_mask=patch_scores["active_mask"],
                    margin_c=patch_scores["margin_c"],
                    weighted_margin_z=patch_scores["weighted_margin_z"],
                )
                score_s = np.asarray([float(summary["score_s"]) for summary in component_summaries], dtype=np.float32)
                size_n = np.asarray([int(summary["size_n"]) for summary in component_summaries], dtype=np.int32)
                peak_p = np.asarray([float(summary["peak_p"]) for summary in component_summaries], dtype=np.float32)
                cached_component_arrays[(sample_name, k_value, anomaly_threshold)] = (score_s, size_n, peak_p)

        for sample_name in valid_samples_fold:
            true_idx = valid_true_idx[sample_name]
            for pair_idx, (k_value, anomaly_threshold) in enumerate(k_thr_pairs):
                score_s, size_n, peak_p = cached_component_arrays[(sample_name, k_value, anomaly_threshold)]
                base_offset = pair_idx * len(tau_triplets)

                if score_s.size == 0:
                    combo_counts[base_offset : base_offset + len(tau_triplets), true_idx, 0] += 1
                    continue

                for tau_idx, (tau_s, tau_n, tau_p) in enumerate(tau_triplets):
                    has_3d = bool(np.any(((score_s > tau_s) & (size_n >= tau_n)) | (peak_p > tau_p)))
                    pred_idx = 1 if has_3d else 0
                    combo_counts[base_offset + tau_idx, true_idx, pred_idx] += 1

        print(
            f"Fold {fold_idx}/{args.n_splits} complete. "
            f"Train bank sizes: 2D={bank_2d.shape[0]}, 3D={bank_3d.shape[0]}."
        )

    result_rows: list[dict[str, object]] = []
    for combo_idx, combo in enumerate(full_combos):
        k_value, anomaly_threshold, tau_s, tau_n, tau_p = combo
        metrics = _binary_metrics_from_cm(combo_counts[combo_idx])
        row = {
            "k_neighbors": int(k_value),
            "anomaly_threshold": float(anomaly_threshold),
            "tau_s": float(tau_s),
            "tau_n": int(tau_n),
            "tau_p": float(tau_p),
            "objective": float(_objective_value(metrics, args.objective)),
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "precision_2d": metrics["precision_2d"],
            "recall_2d": metrics["recall_2d"],
            "f1_2d": metrics["f1_2d"],
            "precision_3d": metrics["precision_3d"],
            "recall_3d": metrics["recall_3d"],
            "f1_3d": metrics["f1_3d"],
            "cm_2d_pred_2d": int(metrics["confusion_matrix"][0][0]),
            "cm_2d_pred_3d": int(metrics["confusion_matrix"][0][1]),
            "cm_3d_pred_2d": int(metrics["confusion_matrix"][1][0]),
            "cm_3d_pred_3d": int(metrics["confusion_matrix"][1][1]),
        }
        result_rows.append(row)

    results_df = pd.DataFrame(result_rows)
    results_df.sort_values(
        by=[args.objective, "macro_f1", "f1_3d", "accuracy"],
        ascending=[False, False, False, False],
        inplace=True,
    )
    results_df.reset_index(drop=True, inplace=True)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (
            context.session_dir
            / (
                f"cv_param_search_obj_{args.objective}"
                f"_k{len(k_values)}"
                f"_thr{len(anomaly_threshold_values)}"
                f"_ts{len(tau_s_values)}"
                f"_tn{len(tau_n_values)}"
                f"_tp{len(tau_p_values)}"
            )
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "results.csv", index=False)
    top_n = max(1, int(args.top_n))
    results_df.head(top_n).to_csv(output_dir / "top_results.csv", index=False)

    best_row = results_df.iloc[0].to_dict()
    runtime_sec = float(time.perf_counter() - started_at)
    summary = {
        "session_dir": str(context.session_dir),
        "experiment_dir": str(context.experiment_dir),
        "seed": int(args.seed),
        "evaluation_group": args.evaluation_group,
        "objective": args.objective,
        "n_splits": int(args.n_splits),
        "num_parts_total": int(len(eval_parts)),
        "num_parts_2d": int((eval_parts["part_label"] == "2D").sum()),
        "num_parts_3d": int((eval_parts["part_label"] == "3D").sum()),
        "k_values": [int(v) for v in k_values],
        "anomaly_threshold_values": [float(v) for v in anomaly_threshold_values],
        "tau_s_values": [float(v) for v in tau_s_values],
        "tau_n_values": [int(v) for v in tau_n_values],
        "tau_p_values": [float(v) for v in tau_p_values],
        "num_combinations": int(len(full_combos)),
        "runtime_sec": runtime_sec,
        "best_result": best_row,
        "output_dir": str(output_dir),
    }
    write_json(summary, output_dir / "summary.json")

    print(
        f"Parameter search complete. Output: {output_dir}\n"
        f"Best {args.objective}: {best_row[args.objective]:.6f} "
        f"(k={int(best_row['k_neighbors'])}, thr={best_row['anomaly_threshold']:.6f}, "
        f"tau_s={best_row['tau_s']:.6f}, tau_n={int(best_row['tau_n'])}, tau_p={best_row['tau_p']:.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
