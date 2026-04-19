import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)
DEFAULT_BASE_TABLE = (
    DEFAULT_EXPERIMENT_DIR
    / "hq_sam_outputs_batch"
    / "roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1_seed=0_sam_hq_vit_tiny_maskpooled_features"
    / "roi_feature_table.csv"
)
DEFAULT_LABELS_FILE = Path(r"C:\ai\AnomalyDINO\labeling_tables\dinov3_res688_roi_labels.xlsx")
DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search crop/bounding-box settings that maximize ROI classification performance on "
            "already labeled ROIs. This keeps the ROI identities fixed and sweeps crop source and padding."
        )
    )
    parser.add_argument("--base-table", type=Path, default=DEFAULT_BASE_TABLE)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--classifier", choices=("logreg", "svm_linear", "svm_rbf"), default="svm_rbf")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--sources", nargs="*", default=("raw", "sam"), choices=("raw", "sam"))
    parser.add_argument("--padding-px-list", nargs="*", type=int, default=(0, 8, 16, 24, 32))
    parser.add_argument("--size-multiple", type=int, default=16)
    parser.add_argument("--valid-labels", nargs="*", default=("2D", "3D"))
    parser.add_argument("--hf-token-env", type=str, default="HF_TOKEN")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
        json.dump(data, handle, indent=2)


def normalize_bildname(value: object) -> str:
    return str(value).strip().replace("\\", "/").split("/")[-1]


def normalize_roi_nummer(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("roi"):
        return text
    if text.isdigit():
        return f"roi{text}"
    return text


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def load_table_file(table_file: Path) -> pd.DataFrame:
    suffix = table_file.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(table_file)
    if suffix in {".csv", ".txt", ".tsv"}:
        return pd.read_csv(table_file, sep=None, engine="python")
    raise ValueError(f"Unsupported table format: {table_file}")


def load_base_table(base_table_file: Path) -> pd.DataFrame:
    table = pd.read_csv(base_table_file)
    required = {
        "roi_uid",
        "sample",
        "roi_index",
        "image_path",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "sam_mask_bbox_x_min",
        "sam_mask_bbox_y_min",
        "sam_mask_bbox_x_max",
        "sam_mask_bbox_y_max",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Base table is missing required columns: {sorted(missing)}")

    table = table.copy()
    table["bildname"] = table["image_path"].map(normalize_bildname)
    table["roi_nummer"] = "roi" + table["roi_index"].astype(int).astype(str)
    return table


def load_labeled_rows(base_table: pd.DataFrame, labels_file: Path, valid_labels: Iterable[str], limit: int | None) -> pd.DataFrame:
    labels = load_table_file(labels_file)
    if not {"bildname", "roi_nummer", "label"}.issubset(labels.columns):
        raise ValueError("Labels file must contain bildname, roi_nummer, and label columns.")

    labels = labels.copy()
    labels["bildname"] = labels["bildname"].map(normalize_bildname)
    labels["roi_nummer"] = labels["roi_nummer"].map(normalize_roi_nummer)
    labels["label"] = labels["label"].map(clean_label)
    labels = labels[labels["label"] != ""].copy()

    valid_lookup = {label.lower(): label.lower() for label in valid_labels}
    labels["label_lower"] = labels["label"].str.lower()
    labels = labels[labels["label_lower"].isin(valid_lookup.keys())].copy()
    labels["label"] = labels["label_lower"]
    labels = labels.drop(columns=["label_lower"])

    merged = base_table.merge(
        labels[["bildname", "roi_nummer", "label"]],
        on=["bildname", "roi_nummer"],
        how="inner",
        suffixes=("", "_label_file"),
    )
    if merged.empty:
        raise ValueError("No labeled rows matched the base ROI table.")

    if "label_label_file" in merged.columns:
        merged["label"] = merged["label_label_file"]
        merged = merged.drop(columns=["label_label_file"])

    merged = merged.sort_values(["sample", "roi_index"]).reset_index(drop=True)
    if limit is not None:
        merged = merged.iloc[:limit].copy()
    return merged


def resolve_device(device_name: str) -> str:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to cpu.")
        return "cpu"
    return device_name


def import_transformers():
    from transformers import AutoImageProcessor, AutoModel

    return AutoImageProcessor, AutoModel


def load_dinov3(model_id: str, device: str, token_env_name: str):
    AutoImageProcessor, AutoModel = import_transformers()
    token = None
    try:
        import os

        token = os.getenv(token_env_name)
    except Exception:
        token = None

    try:
        processor = AutoImageProcessor.from_pretrained(model_id, token=token, local_files_only=True)
        model = AutoModel.from_pretrained(model_id, token=token, local_files_only=True)
        print("Loaded DINOv3 from local Hugging Face cache.")
    except Exception:
        processor = AutoImageProcessor.from_pretrained(model_id, token=token)
        model = AutoModel.from_pretrained(model_id, token=token)

    model = model.to(device)
    model.eval()
    return processor, model


def choose_axis_bounds(mask_min: int, mask_max: int, desired_min: int, desired_max: int, image_size: int, size_multiple: int) -> Tuple[int, int]:
    desired_size = desired_max - desired_min
    mask_size = mask_max - mask_min
    if desired_size <= 0 or mask_size <= 0:
        raise ValueError(f"Invalid sizes for axis selection: desired_size={desired_size}, mask_size={mask_size}")

    if size_multiple <= 1:
        target_size = desired_size
    else:
        shrunken_size = (desired_size // size_multiple) * size_multiple
        if shrunken_size >= mask_size and shrunken_size > 0:
            target_size = shrunken_size
        else:
            target_size = ((mask_size + size_multiple - 1) // size_multiple) * size_multiple

    if target_size > image_size:
        target_size = (image_size // size_multiple) * size_multiple if size_multiple > 1 else image_size
    if target_size < mask_size or target_size <= 0:
        target_size = image_size

    allowed_min = max(0, mask_max - target_size)
    allowed_max = min(mask_min, image_size - target_size)
    if allowed_min > allowed_max:
        start = max(0, min(desired_min, image_size - target_size))
        return start, start + target_size

    desired_center = (desired_min + desired_max) / 2.0
    preferred_min = round(desired_center - target_size / 2.0)
    final_min = max(allowed_min, min(preferred_min, allowed_max))
    return final_min, final_min + target_size


def multiple_aligned_crop_box(x_min: int, y_min: int, x_max: int, y_max: int, padding_px: int, width: int, height: int, size_multiple: int) -> Tuple[int, int, int, int]:
    padded_x_min = max(0, x_min - padding_px)
    padded_y_min = max(0, y_min - padding_px)
    padded_x_max = min(width, x_max + padding_px)
    padded_y_max = min(height, y_max + padding_px)

    crop_x_min, crop_x_max = choose_axis_bounds(
        mask_min=x_min,
        mask_max=x_max,
        desired_min=padded_x_min,
        desired_max=padded_x_max,
        image_size=width,
        size_multiple=size_multiple,
    )
    crop_y_min, crop_y_max = choose_axis_bounds(
        mask_min=y_min,
        mask_max=y_max,
        desired_min=padded_y_min,
        desired_max=padded_y_max,
        image_size=height,
        size_multiple=size_multiple,
    )
    return crop_x_min, crop_y_min, crop_x_max, crop_y_max


def crop_coordinates(row: pd.Series, source: str) -> Tuple[int, int, int, int]:
    if source == "raw":
        return int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
    if source == "sam":
        return (
            int(row["sam_mask_bbox_x_min"]),
            int(row["sam_mask_bbox_y_min"]),
            int(row["sam_mask_bbox_x_max"]),
            int(row["sam_mask_bbox_y_max"]),
        )
    raise ValueError(f"Unsupported source: {source}")


def extract_cls_features_for_setting(
    labeled_rows: pd.DataFrame,
    source: str,
    padding_px: int,
    size_multiple: int,
    processor,
    model,
    device: str,
    batch_size: int,
) -> np.ndarray:
    features: List[np.ndarray] = []
    for start in range(0, len(labeled_rows), batch_size):
        batch_rows = labeled_rows.iloc[start : start + batch_size]
        images: List[Image.Image] = []
        for _, row in batch_rows.iterrows():
            image_path = Path(str(row["image_path"]))
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
                x_min, y_min, x_max, y_max = crop_coordinates(row, source)
                crop_box = multiple_aligned_crop_box(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    padding_px=padding_px,
                    width=image.width,
                    height=image.height,
                    size_multiple=size_multiple,
                )
                crop = image.crop(crop_box)
                images.append(crop.copy())

        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
        features.append(cls_embeddings.astype(np.float32))
    return np.concatenate(features, axis=0).astype(np.float32)


def build_pipeline(classifier: str, c_value: float, max_iter: int, class_weight: str | None) -> Pipeline:
    if classifier == "logreg":
        classifier_step = LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            C=c_value,
            class_weight=class_weight,
        )
    elif classifier == "svm_linear":
        classifier_step = SVC(
            kernel="linear",
            C=c_value,
            class_weight=class_weight,
            probability=True,
            max_iter=max_iter,
        )
    elif classifier == "svm_rbf":
        classifier_step = SVC(
            kernel="rbf",
            C=c_value,
            class_weight=class_weight,
            probability=True,
            max_iter=max_iter,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier_step),
        ]
    )


def evaluate_setting(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    class_names: List[str],
    classifier: str,
    c_value: float,
    max_iter: int,
    class_weight: str | None,
    n_splits: int,
    random_state: int,
) -> Dict[str, object]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipeline = build_pipeline(
        classifier=classifier,
        c_value=c_value,
        max_iter=max_iter,
        class_weight=class_weight,
    )

    oof_pred = np.full_like(labels, fill_value=-1)
    for train_idx, val_idx in splitter.split(features, labels, groups):
        pipeline.fit(features[train_idx], labels[train_idx])
        oof_pred[val_idx] = pipeline.predict(features[val_idx])

    if np.any(oof_pred < 0):
        raise RuntimeError("Some out-of-fold predictions were not filled.")

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        oof_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, oof_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(labels, oof_pred, labels=np.arange(len(class_names))).tolist(),
    }


def main():
    args = parse_args()
    base_table_path = args.base_table.resolve()
    experiment_dir = base_table_path.parents[2]
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (experiment_dir / f"bbox_setting_search_{args.classifier}")
    ensure_dir(output_dir)

    base_table = load_base_table(base_table_path)
    labeled_rows = load_labeled_rows(
        base_table=base_table,
        labels_file=args.labels_file.resolve(),
        valid_labels=args.valid_labels,
        limit=args.limit,
    )

    label_encoder = LabelEncoder()
    class_names = [str(label).lower() for label in args.valid_labels]
    label_encoder.fit(class_names)
    y = label_encoder.transform(labeled_rows["label"].str.lower().to_numpy())
    groups = labeled_rows["sample"].astype(str).to_numpy()

    class_weight = None if args.class_weight == "none" else args.class_weight
    device = resolve_device(args.device)
    processor, model = load_dinov3(args.model_id, device, args.hf_token_env)

    result_rows: List[Dict[str, object]] = []
    for source in args.sources:
        for padding_px in args.padding_px_list:
            setting_name = f"{source}_pad{int(padding_px):02d}"
            print(f"Evaluating {setting_name} ...")
            features = extract_cls_features_for_setting(
                labeled_rows=labeled_rows,
                source=source,
                padding_px=int(padding_px),
                size_multiple=int(args.size_multiple),
                processor=processor,
                model=model,
                device=device,
                batch_size=int(args.batch_size),
            )
            metrics = evaluate_setting(
                features=features,
                labels=y,
                groups=groups,
                class_names=class_names,
                classifier=args.classifier,
                c_value=float(args.c_value),
                max_iter=int(args.max_iter),
                class_weight=class_weight,
                n_splits=int(args.n_splits),
                random_state=int(args.random_state),
            )
            row = {
                "setting_name": setting_name,
                "source": source,
                "padding_px": int(padding_px),
                "size_multiple": int(args.size_multiple),
                "classifier": args.classifier,
                "num_labeled_rois": int(len(labeled_rows)),
                "num_groups": int(pd.Series(groups).nunique()),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            }
            result_rows.append(row)
            print(
                f"  -> macro_f1={row['macro_f1']:.4f} | accuracy={row['accuracy']:.4f}"
            )

    results_table = pd.DataFrame(result_rows).sort_values(
        ["macro_f1", "accuracy", "macro_precision"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary = {
        "base_table": str(args.base_table.resolve()),
        "labels_file": str(args.labels_file.resolve()),
        "model_id": args.model_id,
        "classifier": args.classifier,
        "num_labeled_rois": int(len(labeled_rows)),
        "num_groups": int(pd.Series(groups).nunique()),
        "class_names": class_names,
        "size_multiple": int(args.size_multiple),
        "sources": list(args.sources),
        "padding_px_list": [int(value) for value in args.padding_px_list],
        "best_setting": results_table.iloc[0].to_dict() if not results_table.empty else None,
    }

    results_csv = output_dir / "results.csv"
    summary_json = output_dir / "summary.json"
    results_table.to_csv(results_csv, index=False)
    write_json(summary, summary_json)

    print(f"Saved results: {results_csv}")
    print(f"Saved summary: {summary_json}")
    if not results_table.empty:
        best = results_table.iloc[0]
        print(
            f"Best setting: {best['setting_name']} | macro_f1={best['macro_f1']:.4f} | "
            f"accuracy={best['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
