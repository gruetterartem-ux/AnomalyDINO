import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


DEFAULT_OOF_PREDICTIONS = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413\cls_roi_features_labeled\svm_rbf_groupcv_results\oof_predictions.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate ROI-level predictions to part/sample level with an OR rule: "
            "if any ROI is predicted as 3D, the whole part is 3D."
        )
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=DEFAULT_OOF_PREDICTIONS,
        help="ROI-level OOF predictions CSV from the classifier training pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <oof_dir>/part_level_any3d.",
    )
    parser.add_argument(
        "--positive-label",
        type=str,
        default="3d",
        help="Label that triggers a positive part-level decision when present in any ROI.",
    )
    parser.add_argument(
        "--negative-label",
        type=str,
        default="2d",
        help="Fallback part-level label if no ROI is positive.",
    )
    parser.add_argument(
        "--negative-confidence-threshold",
        type=float,
        default=None,
        help=(
            "Optional confidence threshold for the negative class. If set, an ROI is treated as "
            "negative only when P(negative_label) >= threshold; otherwise it is treated as positive."
        ),
    )
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


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().lower()


def default_output_dir(
    oof_predictions: Path,
    explicit_output_dir: Path | None,
    negative_confidence_threshold: float | None,
) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    suffix = "part_level_any3d"
    if negative_confidence_threshold is not None:
        suffix = f"{suffix}_negconf{int(round(negative_confidence_threshold * 100)):02d}"
    return (oof_predictions.resolve().parent / suffix).resolve()


def effective_roi_predictions(
    table: pd.DataFrame,
    positive_label: str,
    negative_label: str,
    negative_confidence_threshold: float | None,
) -> pd.DataFrame:
    table = table.copy()
    if negative_confidence_threshold is None:
        table["effective_predicted_label"] = table["predicted_label"]
        return table

    negative_proba_column = f"proba_{negative_label}"
    if negative_proba_column not in table.columns:
        raise ValueError(f"Missing probability column: {negative_proba_column}")

    negative_probs = table[negative_proba_column].astype(float)
    table["effective_predicted_label"] = np.where(
        negative_probs >= float(negative_confidence_threshold),
        negative_label,
        positive_label,
    )
    table["negative_probability"] = negative_probs
    return table


def part_level_rows(
    table: pd.DataFrame,
    positive_label: str,
    negative_label: str,
    negative_confidence_threshold: float | None,
) -> List[Dict[str, object]]:
    positive_proba_column = f"proba_{positive_label}"
    if positive_proba_column not in table.columns:
        raise ValueError(f"Missing probability column: {positive_proba_column}")

    rows: List[Dict[str, object]] = []
    grouped = table.sort_values(["sample", "roi_index"]).groupby("sample", sort=True)
    for sample, sample_rows in grouped:
        true_positive = (sample_rows["label"] == positive_label).any()
        pred_positive = (sample_rows["effective_predicted_label"] == positive_label).any()
        max_positive_idx = sample_rows[positive_proba_column].astype(float).idxmax()
        max_positive_row = sample_rows.loc[max_positive_idx]
        if "negative_probability" in sample_rows.columns:
            max_negative_idx = sample_rows["negative_probability"].astype(float).idxmax()
            max_negative_row = sample_rows.loc[max_negative_idx]
            max_negative_probability = float(max_negative_row["negative_probability"])
            max_negative_roi_index = int(max_negative_row["roi_index"])
            max_negative_roi_nummer = f"roi{int(max_negative_row['roi_index'])}"
        else:
            max_negative_probability = float("nan")
            max_negative_roi_index = -1
            max_negative_roi_nummer = ""

        rows.append(
            {
                "sample": sample,
                "num_rois": int(len(sample_rows)),
                "true_label": positive_label if true_positive else negative_label,
                "predicted_label": positive_label if pred_positive else negative_label,
                "decision_rule": (
                    f"{negative_label} only if P({negative_label}) >= {negative_confidence_threshold:.2f}"
                    if negative_confidence_threshold is not None
                    else f"any ROI predicted as {positive_label} => {positive_label}"
                ),
                "max_positive_probability": float(max_positive_row[positive_proba_column]),
                "max_positive_roi_index": int(max_positive_row["roi_index"]),
                "max_positive_roi_nummer": f"roi{int(max_positive_row['roi_index'])}",
                "max_positive_roi_predicted_label": str(max_positive_row["predicted_label"]),
                "max_negative_probability": max_negative_probability,
                "max_negative_roi_index": max_negative_roi_index,
                "max_negative_roi_nummer": max_negative_roi_nummer,
                "num_true_positive_rois": int((sample_rows["label"] == positive_label).sum()),
                "num_pred_positive_rois": int((sample_rows["effective_predicted_label"] == positive_label).sum()),
                "all_true_roi_labels": ";".join(sample_rows["label"].astype(str).tolist()),
                "all_pred_roi_labels": ";".join(sample_rows["predicted_label"].astype(str).tolist()),
                "all_effective_pred_roi_labels": ";".join(sample_rows["effective_predicted_label"].astype(str).tolist()),
            }
        )
    return rows


def metrics_from_part_rows(rows: List[Dict[str, object]], positive_label: str, negative_label: str) -> Dict[str, object]:
    table = pd.DataFrame(rows)
    y_true = table["true_label"].to_numpy()
    y_pred = table["predicted_label"].to_numpy()
    y_score = table["max_positive_probability"].astype(float).to_numpy()
    y_true_binary = (y_true == positive_label).astype(int)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[negative_label, positive_label],
        average="macro",
        zero_division=0,
    )
    pos_precision, pos_recall, pos_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[positive_label],
        average="binary",
        pos_label=positive_label,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "positive_precision": float(pos_precision),
        "positive_recall": float(pos_recall),
        "positive_f1": float(pos_f1),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[negative_label, positive_label],
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[negative_label, positive_label],
            output_dict=True,
            zero_division=0,
        ),
    }

    if len(np.unique(y_true_binary)) == 2:
        metrics["auroc_from_max_positive_probability"] = float(roc_auc_score(y_true_binary, y_score))
        metrics["average_precision_from_max_positive_probability"] = float(
            average_precision_score(y_true_binary, y_score)
        )

    return metrics


def main():
    args = parse_args()
    oof_predictions = args.oof_predictions.resolve()
    output_dir = default_output_dir(oof_predictions, args.output_dir, args.negative_confidence_threshold)
    ensure_dir(output_dir)

    table = pd.read_csv(oof_predictions)
    required = {"sample", "roi_index", "label", "predicted_label"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns in {oof_predictions}: {sorted(missing)}")

    positive_label = clean_label(args.positive_label)
    negative_label = clean_label(args.negative_label)

    table = table.copy()
    table["label"] = table["label"].map(clean_label)
    table["predicted_label"] = table["predicted_label"].map(clean_label)
    table["sample"] = table["sample"].astype(str)

    table = table[table["label"].isin({positive_label, negative_label})].copy()
    if table.empty:
        raise ValueError("No ROI rows remain after filtering to the requested labels.")

    table = effective_roi_predictions(
        table=table,
        positive_label=positive_label,
        negative_label=negative_label,
        negative_confidence_threshold=args.negative_confidence_threshold,
    )
    rows = part_level_rows(
        table,
        positive_label,
        negative_label,
        args.negative_confidence_threshold,
    )
    metrics = metrics_from_part_rows(rows, positive_label, negative_label)

    sample_predictions_file = output_dir / "sample_predictions.csv"
    summary_file = output_dir / "summary.json"
    write_csv(rows, sample_predictions_file)
    write_json(
        {
            "oof_predictions": str(oof_predictions),
            "positive_label": positive_label,
            "negative_label": negative_label,
            "negative_confidence_threshold": args.negative_confidence_threshold,
            "num_roi_rows": int(len(table)),
            "num_samples": int(len(rows)),
            "rule": (
                f"ROI is {negative_label} only if P({negative_label}) >= {args.negative_confidence_threshold:.2f}; "
                f"otherwise ROI is {positive_label}. Part is {positive_label} if any ROI is {positive_label}."
                if args.negative_confidence_threshold is not None
                else f"predict {positive_label} if any ROI is predicted as {positive_label}, else {negative_label}"
            ),
            "metrics": metrics,
            "sample_predictions_file": str(sample_predictions_file),
        },
        summary_file,
    )

    print(f"Saved sample predictions: {sample_predictions_file}")
    print(f"Saved summary: {summary_file}")
    print(
        f"Part-level macro F1: {metrics['macro_f1']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"{positive_label.upper()} recall: {metrics['positive_recall']:.4f}"
    )


if __name__ == "__main__":
    main()
