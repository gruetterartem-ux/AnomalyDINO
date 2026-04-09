import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_ROI_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov2_vitb14_704\16-shot_preprocess=force_no_mask_no_rotation_all16_test_maxpatch_random\roi_crops_peak_seeds\seed=0"
)
DEFAULT_MODEL_NAME = "dinov2_vitb14"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract global DINOv2 CLS embeddings for ROI crops."
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
        help="Optional output directory. Defaults to <roi-dir>/<model-name>_cls_embeddings.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        help="Official torch.hub DINOv2 model name.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=448,
        help="Resize smaller image edge before the patch-size crop.",
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
        "--half-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run inference in fp16 on CUDA to reduce memory usage.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of ROI images.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> str:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to cpu.")
        return "cpu"
    return device_name


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
                rows[str(Path(crop_path).resolve())] = row
            except OSError:
                continue
    return rows


def load_dinov2_model(model_name: str, device: str):
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    model = model.to(device)
    return model


def build_transform(resolution: int):
    return transforms.Compose(
        [
            transforms.Resize(
                size=resolution,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def prepare_image(image: Image.Image, transform, patch_size: int) -> torch.Tensor:
    image_tensor = transform(image)
    height, width = image_tensor.shape[1:]
    cropped_height = height - (height % patch_size)
    cropped_width = width - (width % patch_size)
    if cropped_height <= 0 or cropped_width <= 0:
        raise ValueError(
            f"Image became empty after patch-size crop for patch_size={patch_size}: "
            f"input tensor shape={tuple(image_tensor.shape)}"
        )
    return image_tensor[:, :cropped_height, :cropped_width]


def load_image_batch(
    image_paths: List[Path],
    transform,
    patch_size: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[Tuple[int, int]]]:
    tensors: List[torch.Tensor] = []
    original_sizes: List[Tuple[int, int]] = []
    processed_sizes: List[Tuple[int, int]] = []

    for image_path in image_paths:
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            original_sizes.append(image.size)
            image_tensor = prepare_image(image, transform, patch_size)
            processed_sizes.append((image_tensor.shape[2], image_tensor.shape[1]))
            tensors.append(image_tensor)

    batch = torch.stack(tensors, dim=0)
    return batch, original_sizes, processed_sizes


def extract_cls_embeddings(
    image_paths: List[Path],
    model,
    transform,
    device: str,
    half_precision: bool,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    batch, original_sizes, processed_sizes = load_image_batch(image_paths, transform, model.patch_size)
    if half_precision and device.startswith("cuda"):
        batch = batch.half()
    batch = batch.to(device)

    with torch.inference_mode():
        features = model.forward_features(batch)
        cls_embeddings = features["x_norm_clstoken"]

    return cls_embeddings.float().cpu().numpy(), original_sizes, processed_sizes


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

    output_dir = (args.output_dir or (roi_dir / f"{args.model_name}_cls_embeddings")).resolve()
    ensure_dir(output_dir)

    device = resolve_device(args.device)
    roi_image_paths = list_roi_images(roi_dir)
    if args.limit is not None:
        roi_image_paths = roi_image_paths[:args.limit]

    if not roi_image_paths:
        raise FileNotFoundError(f"No ROI images found in {roi_dir}")

    roi_metadata = load_roi_metadata(roi_dir)
    model = load_dinov2_model(args.model_name, device)
    transform = build_transform(args.resolution)

    all_embeddings: List[np.ndarray] = []
    metadata_rows: List[Dict[str, object]] = []

    for start in range(0, len(roi_image_paths), args.batch_size):
        batch_paths = roi_image_paths[start:start + args.batch_size]
        batch_embeddings, original_sizes, processed_sizes = extract_cls_embeddings(
            batch_paths,
            model,
            transform,
            device,
            args.half_precision,
        )
        all_embeddings.append(batch_embeddings)

        for batch_index, image_path in enumerate(batch_paths):
            resolved_path = str(image_path.resolve())
            orig_w, orig_h = original_sizes[batch_index]
            proc_w, proc_h = processed_sizes[batch_index]
            row: Dict[str, object] = {
                "embedding_index": start + batch_index,
                "image_path": resolved_path,
                "relative_path": str(image_path.relative_to(roi_dir)).replace("\\", "/"),
                "original_width": orig_w,
                "original_height": orig_h,
                "processed_width": proc_w,
                "processed_height": proc_h,
                "model_name": args.model_name,
                "resolution": args.resolution,
                "patch_size": int(model.patch_size),
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
