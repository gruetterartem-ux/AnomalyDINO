from __future__ import annotations

import argparse
import csv
import json
import os
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from component_memory_bank.data_io import load_run_samples
from component_memory_bank.export import write_json


DEFAULT_MEMORY_BANK_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413\component_memory_bank_backend\session_full\memory_bank_export"
)
DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed 64x64 translated crops around labeled memory-bank patches, "
            "extract DINOv3 CLS tokens, and train a patch-level logistic regression."
        )
    )
    parser.add_argument("--memory-bank-dir", type=Path, default=DEFAULT_MEMORY_BANK_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument(
        "--translations",
        type=int,
        nargs="+",
        default=[-8, 0, 8],
        help="Translations in resized-image pixels applied in x and y around the base patch center.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--hf-token-env", type=str, default="HF_TOKEN")
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


def import_transformers():
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency `transformers`. Install `transformers` in .venvAnomalyDINO."
        ) from exc
    return AutoImageProcessor, AutoModel


def resolve_device(device_name: str) -> str:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to cpu.")
        return "cpu"
    return device_name


def load_dinov3(model_id: str, device: str, token_env_name: str):
    AutoImageProcessor, AutoModel = import_transformers()
    token = os.getenv(token_env_name)
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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))).tolist(),
        "per_class": {
            class_name: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, class_name in enumerate(class_names)
        },
    }


def _load_memory_bank_context(memory_bank_dir: Path) -> tuple[pd.DataFrame, Path, int]:
    summary_path = memory_bank_dir / "summary.json"
    patches_path = memory_bank_dir / "selected_patches.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing memory-bank summary: {summary_path}")
    if not patches_path.exists():
        raise FileNotFoundError(f"Missing selected patches CSV: {patches_path}")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    experiment_dir = Path(summary["experiment_dir"]).resolve()
    seed = int(summary["seed"])
    patches_df = pd.read_csv(patches_path)
    patches_df["sample"] = patches_df["sample"].astype(str).str.replace("\\", "/", regex=False)
    patches_df["component_label"] = patches_df["component_label"].astype(str)
    patches_df["source_patch_uid"] = (
        patches_df["sample"]
        + "__patch_"
        + patches_df["patch_index"].astype(int).astype(str)
    )
    return patches_df, experiment_dir, seed


def _sample_cache_info(feature_cache_path: Path) -> dict[str, object]:
    with np.load(feature_cache_path) as cache_data:
        original_w, original_h = [int(v) for v in cache_data["original_size"].tolist()]
        resized_w, resized_h = [int(v) for v in cache_data["resized_size"].tolist()]
        patch_size = int(np.asarray(cache_data["patch_size"]).reshape(-1)[0])
        image_path = Path(str(cache_data["image_path"].tolist()))
    return {
        "original_h": original_h,
        "original_w": original_w,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "patch_size": patch_size,
        "image_path": image_path,
    }


def _crop_rect_with_shift(
    image: Image.Image,
    center_x: float,
    center_y: float,
    crop_w: int,
    crop_h: int,
) -> Image.Image:
    img_w, img_h = image.size
    left = int(round(center_x - crop_w / 2.0))
    top = int(round(center_y - crop_h / 2.0))
    left = max(0, min(left, img_w - crop_w))
    top = max(0, min(top, img_h - crop_h))
    right = left + crop_w
    bottom = top + crop_h
    return image.crop((left, top, right, bottom))


def build_augmented_crops(
    patches_df: pd.DataFrame,
    sample_map: dict[str, object],
    output_dir: Path,
    crop_size: int,
    translations: list[int],
) -> pd.DataFrame:
    crops_dir = output_dir / "crops64"
    ensure_dir(crops_dir)

    sample_info_cache: dict[str, dict[str, object]] = {}
    image_cache: dict[str, Image.Image] = {}
    rows: list[dict[str, object]] = []

    translation_pairs = list(product(translations, translations))

    for patch_row in patches_df.itertuples(index=False):
        sample_name = str(patch_row.sample)
        sample = sample_map[sample_name]
        if sample_name not in sample_info_cache:
            sample_info_cache[sample_name] = _sample_cache_info(sample.feature_cache_path)
        info = sample_info_cache[sample_name]
        if sample_name not in image_cache:
            image_cache[sample_name] = Image.open(info["image_path"]).convert("RGB")
        image = image_cache[sample_name]

        scale_x = float(info["resized_w"]) / float(info["original_w"])
        scale_y = float(info["resized_h"]) / float(info["original_h"])
        patch_size = int(info["patch_size"])

        center_x_resized = (int(patch_row.col) + 0.5) * patch_size
        center_y_resized = (int(patch_row.row) + 0.5) * patch_size
        center_x_orig = center_x_resized / scale_x
        center_y_orig = center_y_resized / scale_y

        crop_w_orig = max(1, int(round(crop_size / scale_x)))
        crop_h_orig = max(1, int(round(crop_size / scale_y)))

        stem = (
            f"{Path(sample_name).stem}"
            f"__p{int(patch_row.patch_index):04d}"
            f"__{str(patch_row.component_label)}"
        )

        for aug_idx, (dx, dy) in enumerate(translation_pairs):
            shifted_center_x = center_x_orig + (float(dx) / scale_x)
            shifted_center_y = center_y_orig + (float(dy) / scale_y)
            crop = _crop_rect_with_shift(
                image=image,
                center_x=shifted_center_x,
                center_y=shifted_center_y,
                crop_w=crop_w_orig,
                crop_h=crop_h_orig,
            )
            crop = crop.resize((crop_size, crop_size), resample=Image.Resampling.BICUBIC)

            file_name = f"{stem}__dx{dx:+03d}_dy{dy:+03d}.png".replace("+", "p").replace("-", "m")
            label_dir = crops_dir / str(patch_row.component_label)
            ensure_dir(label_dir)
            crop_path = label_dir / file_name
            crop.save(crop_path)

            rows.append(
                {
                    "augment_index": len(rows),
                    "source_patch_uid": str(patch_row.source_patch_uid),
                    "label": str(patch_row.component_label),
                    "sample": sample_name,
                    "object_name": str(patch_row.object_name),
                    "component_id": patch_row.component_id,
                    "source_type": str(patch_row.source_type),
                    "patch_index": int(patch_row.patch_index),
                    "row": int(patch_row.row),
                    "col": int(patch_row.col),
                    "anomaly_score": float(patch_row.anomaly_score),
                    "anomaly_threshold": float(patch_row.anomaly_threshold) if pd.notna(patch_row.anomaly_threshold) else "",
                    "translation_dx": int(dx),
                    "translation_dy": int(dy),
                    "raw_image_path": str(info["image_path"]),
                    "crop_path": str(crop_path),
                    "crop_size": int(crop_size),
                    "patch_size": int(patch_size),
                    "center_x_resized": float(center_x_resized),
                    "center_y_resized": float(center_y_resized),
                    "center_x_original": float(center_x_orig),
                    "center_y_original": float(center_y_orig),
                    "crop_w_original": int(crop_w_orig),
                    "crop_h_original": int(crop_h_orig),
                }
            )

    augmented_df = pd.DataFrame(rows)
    augmented_df.to_csv(output_dir / "augmented_patch_crops.csv", index=False)
    return augmented_df


def extract_cls_embeddings(
    augmented_df: pd.DataFrame,
    output_dir: Path,
    model_id: str,
    device: str,
    batch_size: int,
    hf_token_env: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    processor, model = load_dinov3(model_id, device, hf_token_env)
    normalize = transforms.Normalize(mean=tuple(processor.image_mean), std=tuple(processor.image_std))
    to_tensor = transforms.ToTensor()

    all_features: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    crop_paths = [Path(path) for path in augmented_df["crop_path"].tolist()]
    for start in range(0, len(crop_paths), batch_size):
        batch_paths = crop_paths[start : start + batch_size]
        batch_images: list[torch.Tensor] = []
        for crop_path in batch_paths:
            with Image.open(crop_path) as image_handle:
                image = image_handle.convert("RGB")
                batch_images.append(normalize(to_tensor(image)))

        pixel_values = torch.stack(batch_images, dim=0).to(device)
        with torch.inference_mode():
            outputs = model(pixel_values=pixel_values, return_dict=True)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()

        all_features.append(cls_embeddings.astype(np.float32))
        batch_rows = augmented_df.iloc[start : start + len(batch_paths)]
        for batch_idx, (_, row) in enumerate(batch_rows.iterrows()):
            meta = row.to_dict()
            meta["feature_index"] = int(start + batch_idx)
            meta["feature_type"] = "dinov3_cls_token_64x64_augmented"
            meta["group_id"] = meta["source_patch_uid"]
            meta["model_id"] = model_id
            meta["embedding_dim"] = int(cls_embeddings.shape[1])
            meta_rows.append(meta)

        print(f"Extracted CLS features for {min(start + len(batch_paths), len(crop_paths))}/{len(crop_paths)} crops")

    feature_array = np.concatenate(all_features, axis=0).astype(np.float32)
    np.save(output_dir / "augmented_patch_cls_features.npy", feature_array)
    metadata_df = pd.DataFrame(meta_rows)
    metadata_df.to_csv(output_dir / "augmented_patch_cls_feature_table.csv", index=False)
    return feature_array, metadata_df


def train_grouped_logreg(
    features: np.ndarray,
    metadata_df: pd.DataFrame,
    output_dir: Path,
    n_splits: int,
    random_state: int,
    c_value: float,
    max_iter: int,
    class_weight: str | None,
) -> None:
    y_labels = metadata_df["label"].astype(str).to_numpy()
    class_names = ["2D", "3D"]
    y = np.array([0 if label == "2D" else 1 for label in y_labels], dtype=np.int32)
    groups = metadata_df["group_id"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=max_iter,
                    C=c_value,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(features, y, groups), start=1):
        pipeline.fit(features[train_idx], y[train_idx])
        y_pred = pipeline.predict(features[val_idx])
        y_proba = pipeline.predict_proba(features[val_idx]).astype(np.float32)
        oof_pred[val_idx] = y_pred
        oof_proba[val_idx] = y_proba

        metrics = _compute_metrics(y[val_idx], y_pred, class_names)
        fold_rows.append(
            {
                "fold": fold_idx,
                "num_train_crops": int(len(train_idx)),
                "num_val_crops": int(len(val_idx)),
                "num_train_source_patches": int(pd.Series(groups[train_idx]).nunique()),
                "num_val_source_patches": int(pd.Series(groups[val_idx]).nunique()),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "f1_2d": metrics["per_class"]["2D"]["f1"],
                "f1_3d": metrics["per_class"]["3D"]["f1"],
            }
        )

    if np.any(oof_pred < 0):
        raise RuntimeError("Some OOF predictions are missing.")

    overall = _compute_metrics(y, oof_pred, class_names)
    oof_rows: list[dict[str, object]] = []
    for idx, row in metadata_df.reset_index(drop=True).iterrows():
        out = row.to_dict()
        out["true_label"] = y_labels[idx]
        out["predicted_label"] = class_names[oof_pred[idx]]
        out["correct"] = int(y_labels[idx] == class_names[oof_pred[idx]])
        out["proba_2D"] = float(oof_proba[idx, 0])
        out["proba_3D"] = float(oof_proba[idx, 1])
        oof_rows.append(out)

    write_csv(fold_rows, output_dir / "logreg_groupcv_fold_metrics.csv")
    write_csv(oof_rows, output_dir / "logreg_groupcv_oof_predictions.csv")
    write_json(
        {
            "classifier": "logreg",
            "cv_type": "StratifiedGroupKFold",
            "grouping": "source_patch_uid",
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "class_weight": class_weight,
            "c_value": float(c_value),
            "max_iter": int(max_iter),
            "num_augmented_crops_total": int(len(metadata_df)),
            "num_augmented_crops_2d": int((metadata_df["label"] == "2D").sum()),
            "num_augmented_crops_3d": int((metadata_df["label"] == "3D").sum()),
            "num_source_patches_total": int(metadata_df["group_id"].nunique()),
            "num_source_patches_2d": int(metadata_df.loc[metadata_df["label"] == "2D", "group_id"].nunique()),
            "num_source_patches_3d": int(metadata_df.loc[metadata_df["label"] == "3D", "group_id"].nunique()),
            "overall": overall,
            "folds": fold_rows,
        },
        output_dir / "logreg_groupcv_summary.json",
    )


def default_output_dir(memory_bank_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (memory_bank_dir / "aug64_t8_cls_logreg").resolve()


def main() -> int:
    args = parse_args()
    memory_bank_dir = args.memory_bank_dir.resolve()
    output_dir = default_output_dir(memory_bank_dir, args.output_dir)
    ensure_dir(output_dir)

    patches_df, experiment_dir, seed = _load_memory_bank_context(memory_bank_dir)
    samples = load_run_samples(experiment_dir, seed=seed)
    sample_map = {sample.sample: sample for sample in samples}

    translations = sorted({int(v) for v in args.translations})
    device = resolve_device(args.device)
    class_weight = None if args.class_weight == "none" else args.class_weight

    augmented_df = build_augmented_crops(
        patches_df=patches_df,
        sample_map=sample_map,
        output_dir=output_dir,
        crop_size=int(args.crop_size),
        translations=translations,
    )
    features, metadata_df = extract_cls_embeddings(
        augmented_df=augmented_df,
        output_dir=output_dir,
        model_id=args.model_id,
        device=device,
        batch_size=int(args.batch_size),
        hf_token_env=args.hf_token_env,
    )
    train_grouped_logreg(
        features=features,
        metadata_df=metadata_df,
        output_dir=output_dir,
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
        c_value=float(args.c_value),
        max_iter=int(args.max_iter),
        class_weight=class_weight,
    )

    write_json(
        {
            "memory_bank_dir": str(memory_bank_dir),
            "experiment_dir": str(experiment_dir),
            "seed": int(seed),
            "model_id": args.model_id,
            "device": device,
            "crop_size": int(args.crop_size),
            "translations": translations,
            "num_translation_variants_per_source_patch": int(len(translations) ** 2),
            "num_source_patches": int(patches_df["source_patch_uid"].nunique()),
            "num_augmented_crops": int(len(augmented_df)),
            "output_dir": str(output_dir),
            "augmented_crops_csv": str(output_dir / "augmented_patch_crops.csv"),
            "features_file": str(output_dir / "augmented_patch_cls_features.npy"),
            "feature_table_csv": str(output_dir / "augmented_patch_cls_feature_table.csv"),
            "logreg_summary_json": str(output_dir / "logreg_groupcv_summary.json"),
        },
        output_dir / "summary.json",
    )

    print(f"Augmented patch CLS pipeline complete. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
