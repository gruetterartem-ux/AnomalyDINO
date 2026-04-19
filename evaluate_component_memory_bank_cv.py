from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from component_labeling_app.session_io import load_session
from component_memory_bank.data_io import load_patch_features, load_patch_scores, load_run_samples
from component_memory_bank.export import write_csv, write_json
from component_memory_bank.inference import (
    classify_components,
    classify_part,
    compute_patch_class_scores,
    summarize_components,
)
from component_memory_bank.memory_bank import (
    ComponentLabelRecord,
    ManualPatchLabelRecord,
    build_memory_banks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the component memory-bank kNN logic with fold-safe part-level cross-validation."
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-group", type=str, default="test/bad")
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--anomaly-threshold", type=float, default=None)
    parser.add_argument("--tau-s", type=float, default=0.0)
    parser.add_argument("--tau-n", type=int, default=2)
    parser.add_argument("--tau-p", type=float, default=0.05)
    return parser.parse_args()


def _component_label_records(df: pd.DataFrame) -> list[ComponentLabelRecord]:
    labeled = df[df["label"].fillna("").astype(str).isin(["2D", "3D", "skip"])].copy()
    records: list[ComponentLabelRecord] = []
    for row in labeled.itertuples(index=False):
        label = str(row.label)
        if not label:
            continue
        records.append(
            ComponentLabelRecord(
                object_name=str(row.object_name),
                sample=str(row.sample).replace("\\", "/"),
                component_id=int(row.component_id),
                anomaly_threshold=float(row.anomaly_threshold),
                top_k=int(row.top_k),
                label=label,
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


def _compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    labels = ["2D", "3D"]
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(labels)
        },
    }


def main() -> int:
    args = parse_args()

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

    default_threshold = float(context.anomaly_threshold)
    anomaly_threshold = float(args.anomaly_threshold) if args.anomaly_threshold is not None else default_threshold

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    y_true_all: list[str] = []
    y_pred_all: list[str] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(part_samples, y), start=1):
        train_samples = {part_samples[idx] for idx in train_idx}
        valid_samples_fold = [part_samples[idx] for idx in valid_idx]

        train_component_records = _component_label_records(
            component_df[component_df["sample"].isin(train_samples)].copy()
        )
        train_manual_records = _manual_patch_label_records(
            manual_patch_df[manual_patch_df["sample"].isin(train_samples)].copy()
        )

        bundle = build_memory_banks(samples, train_component_records, train_manual_records)
        bank_2d = bundle.features_2d.astype(np.float32, copy=False)
        bank_3d = bundle.features_3d.astype(np.float32, copy=False)

        fold_y_true: list[str] = []
        fold_y_pred: list[str] = []
        for sample_name in valid_samples_fold:
            sample = sample_map[sample_name]
            part_true = str(eval_parts.loc[eval_parts["sample"] == sample_name, "part_label"].iloc[0])
            patch_features, grid_size = load_patch_features(sample)
            anomaly_scores = load_patch_scores(sample)
            if tuple(anomaly_scores.shape) != tuple(grid_size):
                raise ValueError(
                    f"Grid mismatch for {sample.sample!r}: feature cache {grid_size}, anomaly grid {anomaly_scores.shape}."
                )

            patch_scores = compute_patch_class_scores(
                patch_features=patch_features,
                anomaly_scores=anomaly_scores,
                bank_2d=bank_2d,
                bank_3d=bank_3d,
                k_neighbors=args.k_neighbors,
                anomaly_threshold=anomaly_threshold,
            )
            component_summaries = summarize_components(
                anomaly_scores=anomaly_scores,
                active_mask=patch_scores["active_mask"],
                margin_c=patch_scores["margin_c"],
                weighted_margin_z=patch_scores["weighted_margin_z"],
            )
            component_decisions = classify_components(
                sample=sample.sample,
                component_summaries=component_summaries,
                tau_s=args.tau_s,
                tau_n=args.tau_n,
                tau_p=args.tau_p,
            )
            part_decision = classify_part(sample.sample, component_decisions)

            fold_y_true.append(part_true)
            fold_y_pred.append(part_decision.predicted_label)
            y_true_all.append(part_true)
            y_pred_all.append(part_decision.predicted_label)

            comp_3d_candidates = [decision for decision in component_decisions if decision.is_3d]
            best_3d_component = max(comp_3d_candidates, key=lambda d: d.peak_p, default=None)
            oof_rows.append(
                {
                    "fold": fold_idx,
                    "sample": sample.sample,
                    "evaluation_group": sample.evaluation_group,
                    "true_label": part_true,
                    "predicted_label": part_decision.predicted_label,
                    "correct": int(part_true == part_decision.predicted_label),
                    "num_components": part_decision.num_components,
                    "num_3d_components": part_decision.num_3d_components,
                    "best_3d_component_id": best_3d_component.component_id if best_3d_component else "",
                    "best_3d_component_peak_p": best_3d_component.peak_p if best_3d_component else "",
                    "best_3d_component_score_s": best_3d_component.score_s if best_3d_component else "",
                    "best_3d_component_size_n": best_3d_component.size_n if best_3d_component else "",
                }
            )
            for decision in component_decisions:
                component_rows.append(
                    {
                        "fold": fold_idx,
                        "sample": sample.sample,
                        "true_part_label": part_true,
                        "component_id": decision.component_id,
                        "predicted_component_label": "3D" if decision.is_3d else "2D",
                        "score_s": decision.score_s,
                        "size_n": decision.size_n,
                        "peak_p": decision.peak_p,
                        "max_margin_c": decision.max_margin_c,
                    }
                )

        fold_metrics = _compute_metrics(fold_y_true, fold_y_pred)
        fold_rows.append(
            {
                "fold": fold_idx,
                "num_train_parts": len(train_samples),
                "num_valid_parts": len(valid_samples_fold),
                "num_train_component_records": len(train_component_records),
                "num_train_manual_records": len(train_manual_records),
                "train_bank_patches_2d": int(bank_2d.shape[0]),
                "train_bank_patches_3d": int(bank_3d.shape[0]),
                "accuracy": fold_metrics["accuracy"],
                "macro_precision": fold_metrics["macro_precision"],
                "macro_recall": fold_metrics["macro_recall"],
                "macro_f1": fold_metrics["macro_f1"],
                "precision_2d": fold_metrics["per_class"]["2D"]["precision"],
                "recall_2d": fold_metrics["per_class"]["2D"]["recall"],
                "f1_2d": fold_metrics["per_class"]["2D"]["f1"],
                "precision_3d": fold_metrics["per_class"]["3D"]["precision"],
                "recall_3d": fold_metrics["per_class"]["3D"]["recall"],
                "f1_3d": fold_metrics["per_class"]["3D"]["f1"],
            }
        )

    summary = _compute_metrics(y_true_all, y_pred_all)
    summary.update(
        {
            "session_dir": str(args.session_dir.resolve()),
            "experiment_dir": str(context.experiment_dir),
            "seed": int(args.seed),
            "evaluation_group": args.evaluation_group,
            "n_splits": int(args.n_splits),
            "num_parts_total": int(len(eval_parts)),
            "num_parts_2d": int((eval_parts["part_label"] == "2D").sum()),
            "num_parts_3d": int((eval_parts["part_label"] == "3D").sum()),
            "k_neighbors": int(args.k_neighbors),
            "anomaly_threshold": float(anomaly_threshold),
            "tau_s": float(args.tau_s),
            "tau_n": int(args.tau_n),
            "tau_p": float(args.tau_p),
            "output_dir": str(args.output_dir.resolve()),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(fold_rows, args.output_dir / "fold_metrics.csv")
    write_csv(oof_rows, args.output_dir / "oof_part_predictions.csv")
    write_csv(component_rows, args.output_dir / "oof_component_predictions.csv")
    write_json(summary, args.output_dir / "summary.json")

    print(f"Cross-validation complete. Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
