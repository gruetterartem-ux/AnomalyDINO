import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_ROI_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov2_vitb14_704\16-shot_preprocess=force_no_mask_no_rotation_all16_test_maxpatch_random\roi_crops_peak_seeds\seed=0"
)
DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract global DINOv3 CLS embeddings for ROI crops."
    )
    parser.add_argument(
        "--roi-dir",
        type=Path,
        default=DEFAULT_ROI_DIR,
        help="Directory that contains ROI images and optionally roi_metadata.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <roi-dir>/dinov3_vitb16_cls_embeddings.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id for DINOv3.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of ROI crops processed per batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0 or cpu.",
    )
    parser.add_argument(
        "--hf-token-env",
        type=str,
        default="HF_TOKEN",
        help="Environment variable that stores the Hugging Face token for gated DINOv3 checkpoints.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of ROI images.",
    )
    return parser.parse_args()


def import_transformers():
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency `transformers`. Install `transformers` and `huggingface_hub` "
            "in .venvAnomalyDINO before using this script."
        ) from exc
    return AutoImageProcessor, AutoModel


def resolve_device(device_name: str) -> str:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to cpu.")
        return "cpu"
    return device_name


def list_roi_images(roi_dir: Path) -> List[Path]:
    image_paths = [
        path for path in roi_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def load_roi_metadata(roi_dir: Path) -> Dict[str, Dict[str, str]]:
    metadata_file = roi_dir / "roi_metadata.csv"
    if not metadata_file.exists():
        return {}

    rows: Dict[str, Dict[str, str]] = {}
    with metadata_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            crop_path = row.get("crop_path")
            if not crop_path:
                continue
            try:
                resolved_path = str(Path(crop_path).resolve())
            except OSError:
                continue
            rows[resolved_path] = row
    return rows


def load_dinov3(model_id: str, device: str, token_env_name: str):
    AutoImageProcessor, AutoModel = import_transformers()
    token = os.getenv(token_env_name)

    try:
        processor = AutoImageProcessor.from_pretrained(model_id, token=token)
        model = AutoModel.from_pretrained(model_id, token=token)
    except Exception as exc:
        try:
            processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
            model = AutoModel.from_pretrained(model_id, local_files_only=True)
            print("Loaded DINOv3 from local Hugging Face cache.")
        except Exception as local_exc:
            raise RuntimeError(
                f"Could not load model {model_id!r}. The checkpoint is gated on Hugging Face. "
                f"Accept access for the model page and authenticate via `huggingface-cli login` "
                f"or set the token in the environment variable {token_env_name!r}."
            ) from local_exc

    model = model.to(device)
    model.eval()
    return processor, model


def load_image_batch(image_paths: List[Path]) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
    images: List[Image.Image] = []
    sizes: List[Tuple[int, int]] = []
    for image_path in image_paths:
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            sizes.append(image.size)
            images.append(image.copy())
    return images, sizes


def extract_cls_embeddings(image_paths: List[Path], processor, model, device: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    images, sizes = load_image_batch(image_paths)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

    return cls_embeddings.float().cpu().numpy(), sizes


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_metadata(metadata_rows: List[Dict[str, object]], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    if not metadata_rows:
        output_file.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in metadata_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)


def main():
    args = parse_args()
    roi_dir = args.roi_dir.resolve()
    if not roi_dir.exists():
        raise FileNotFoundError(f"ROI directory not found: {roi_dir}")

    output_dir = (args.output_dir or (roi_dir / "dinov3_vitb16_cls_embeddings")).resolve()
    ensure_dir(output_dir)

    device = resolve_device(args.device)
    roi_image_paths = list_roi_images(roi_dir)
    if args.limit is not None:
        roi_image_paths = roi_image_paths[:args.limit]

    if not roi_image_paths:
        raise FileNotFoundError(f"No ROI images found in {roi_dir}")

    roi_metadata = load_roi_metadata(roi_dir)
    processor, model = load_dinov3(args.model_id, device, args.hf_token_env)

    all_embeddings: List[np.ndarray] = []
    metadata_rows: List[Dict[str, object]] = []

    for start in range(0, len(roi_image_paths), args.batch_size):
        batch_paths = roi_image_paths[start:start + args.batch_size]
        batch_embeddings, batch_sizes = extract_cls_embeddings(batch_paths, processor, model, device)
        all_embeddings.append(batch_embeddings)

        for batch_index, image_path in enumerate(batch_paths):
            resolved_path = str(image_path.resolve())
            width, height = batch_sizes[batch_index]
            row: Dict[str, object] = {
                "embedding_index": start + batch_index,
                "image_path": resolved_path,
                "relative_path": str(image_path.relative_to(roi_dir)).replace("\\", "/"),
                "width": width,
                "height": height,
                "model_id": args.model_id,
                "embedding_dim": int(batch_embeddings.shape[1]),
            }

            source_metadata = roi_metadata.get(resolved_path)
            if source_metadata is not None:
                for key, value in source_metadata.items():
                    row[f"roi_{key}"] = value

            metadata_rows.append(row)

        print(f"Processed {min(start + len(batch_paths), len(roi_image_paths))}/{len(roi_image_paths)} ROI images")

    embeddings = np.concatenate(all_embeddings, axis=0)
    embeddings_file = output_dir / "embeddings_cls.npy"
    metadata_file = output_dir / "embedding_metadata.csv"
    np.save(embeddings_file, embeddings)
    write_metadata(metadata_rows, metadata_file)

    print(f"Saved embeddings: {embeddings_file}")
    print(f"Saved metadata: {metadata_file}")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
