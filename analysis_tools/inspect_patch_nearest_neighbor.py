import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from src.backbones import get_model
from src.utils import augment_image, get_dataset_info


ROTATION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a single test patch from a finished AnomalyDINO run and "
            "find its nearest reference patch in the reconstructed memory bank."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Finished run directory containing args.yaml.")
    parser.add_argument("--sample", type=str, required=True, help="Sample path as it appears in measurements_seed=*.csv, e.g. 'bad/2D/foo.png'.")
    parser.add_argument("--patch-row", type=int, default=None, help="Patch row in the model grid.")
    parser.add_argument("--patch-col", type=int, default=None, help="Patch column in the model grid.")
    parser.add_argument("--x", type=int, default=None, help="Original-image x pixel coordinate.")
    parser.add_argument("--y", type=int, default=None, help="Original-image y pixel coordinate.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the finished run. Default: 0.")
    parser.add_argument("--object-name", type=str, default=None, help="Object name if the run contains multiple objects.")
    parser.add_argument("--top-k", type=int, default=1, help="How many nearest reference patches to report. Default: 1.")
    parser.add_argument("--device", type=str, default=None, help="Override device from args.yaml, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <run-dir>/patch_nn_inspection/seed=<seed>/...",
    )
    return parser.parse_args()


def normalize_sample_path(sample):
    return sample.replace("\\", "/").lstrip("/")


def load_run_args(run_dir):
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Could not find args.yaml in {run_dir}")
    with args_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def infer_object_name(run_dir, seed, explicit_name=None):
    if explicit_name:
        return explicit_name
    reference_files = sorted(run_dir.glob(f"reference_images_*_seed={seed}.json"))
    if len(reference_files) == 1:
        stem = reference_files[0].stem
        return stem.split("reference_images_", 1)[1].rsplit(f"_seed={seed}", 1)[0]
    if not reference_files:
        raise FileNotFoundError(f"No reference_images_*_seed={seed}.json found in {run_dir}")
    raise ValueError(
        f"Multiple objects found in {run_dir}. Please provide --object-name. "
        f"Candidates: {[f.name for f in reference_files]}"
    )


def load_reference_images(run_dir, object_name, seed):
    ref_path = run_dir / f"reference_images_{object_name}_seed={seed}.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"Could not find {ref_path}")
    with ref_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rgb_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_processed_rgb_for_display(model, rgb_image):
    return get_resize_and_crop_info(model, rgb_image)["processed_rgb"]


def get_resize_and_crop_info(model, rgb_image):
    original_height, original_width = rgb_image.shape[:2]
    patch_size = getattr(model, "patch_size", None)
    resize_transform = getattr(model, "resize_transform", None)
    if patch_size is None or resize_transform is None:
        return {
            "original_height": int(original_height),
            "original_width": int(original_width),
            "resized_height": int(original_height),
            "resized_width": int(original_width),
            "processed_height": int(original_height),
            "processed_width": int(original_width),
            "patch_size": patch_size,
            "processed_rgb": rgb_image.copy(),
        }

    pil_image = Image.fromarray(rgb_image)
    resized = resize_transform(pil_image)
    resized_rgb = np.array(resized)
    crop_h = resized_rgb.shape[0] - resized_rgb.shape[0] % patch_size
    crop_w = resized_rgb.shape[1] - resized_rgb.shape[1] % patch_size
    return {
        "original_height": int(original_height),
        "original_width": int(original_width),
        "resized_height": int(resized_rgb.shape[0]),
        "resized_width": int(resized_rgb.shape[1]),
        "processed_height": int(crop_h),
        "processed_width": int(crop_w),
        "patch_size": int(patch_size),
        "processed_rgb": resized_rgb[:crop_h, :crop_w].copy(),
    }


def patch_bbox(patch_row, patch_col, patch_size):
    left = patch_col * patch_size
    top = patch_row * patch_size
    right = left + patch_size
    bottom = top + patch_size
    return left, top, right, bottom


def draw_patch_box(image_rgb, patch_row, patch_col, patch_size, outline, label):
    canvas = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = patch_bbox(patch_row, patch_col, patch_size)
    draw.rectangle([left, top, right - 1, bottom - 1], outline=outline, width=3)
    draw.text((left + 4, max(4, top - 18)), label, fill=outline)
    return canvas


def concatenate_side_by_side(left_image, right_image, left_text, right_text, title_text):
    gap = 24
    header_h = 72
    width = left_image.width + right_image.width + gap
    height = max(left_image.height, right_image.height) + header_h
    canvas = Image.new("RGB", (width, height), color="white")
    canvas.paste(left_image, (0, header_h))
    canvas.paste(right_image, (left_image.width + gap, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), title_text, fill="black")
    draw.text((16, 38), left_text, fill="black")
    draw.text((left_image.width + gap + 16, 38), right_text, fill="black")
    return canvas


def compute_distances(query_feature, features_ref, knn_metric):
    query = query_feature.astype(np.float32, copy=True)
    refs = features_ref.astype(np.float32, copy=True)

    if knn_metric == "L2_normalized":
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise ValueError("Selected query patch has zero norm and cannot be normalized.")
        query = query / query_norm

        ref_norms = np.linalg.norm(refs, axis=1, keepdims=True)
        ref_norms[ref_norms == 0] = 1.0
        refs = refs / ref_norms

        squared_l2 = np.sum((refs - query[None, :]) ** 2, axis=1)
        return squared_l2 / 2.0

    if knn_metric == "L2":
        return np.linalg.norm(refs - query[None, :], axis=1)

    raise ValueError(f"Unsupported knn_metric: {knn_metric}")


def build_memory_bank(model, args_yaml, object_name, reference_images, masking_default, rotation_default):
    data_root = Path(args_yaml["data_root"])
    ref_dir = data_root / object_name / "train" / "good"
    features_ref = []
    metadata = []

    masking = masking_default[object_name]
    rotation = rotation_default[object_name]
    mask_ref_images = bool(args_yaml.get("mask_ref_images", False))

    for ref_name in reference_images:
        ref_path = ref_dir / ref_name
        ref_rgb = load_rgb_image(ref_path)
        if rotation:
            augmented_images = augment_image(ref_rgb, angles=ROTATION_ANGLES)
            augmentation_labels = [f"rot_{angle}" for angle in ROTATION_ANGLES]
        else:
            augmented_images = [ref_rgb]
            augmentation_labels = ["identity"]

        for aug_idx, (aug_rgb, aug_label) in enumerate(zip(augmented_images, augmentation_labels)):
            image_tensor, grid_size = model.prepare_image(aug_rgb)
            patch_features = model.extract_features(image_tensor)
            patch_mask = model.compute_background_mask(
                patch_features,
                grid_size,
                threshold=10,
                masking_type=(mask_ref_images and masking),
            ).astype(bool)

            coords = np.argwhere(patch_mask.reshape(grid_size))
            features_ref.append(patch_features[patch_mask])

            processed_rgb = get_processed_rgb_for_display(model, aug_rgb)
            processed_h, processed_w = processed_rgb.shape[:2]

            for row, col in coords:
                metadata.append(
                    {
                        "reference_image_name": ref_name,
                        "reference_image_path": str(ref_path),
                        "augmentation_index": aug_idx,
                        "augmentation_label": aug_label,
                        "patch_row": int(row),
                        "patch_col": int(col),
                        "grid_rows": int(grid_size[0]),
                        "grid_cols": int(grid_size[1]),
                        "processed_height": int(processed_h),
                        "processed_width": int(processed_w),
                    }
                )

    if not features_ref:
        raise ValueError("Memory bank reconstruction produced no reference features.")

    return np.concatenate(features_ref, axis=0).astype(np.float32), metadata


def resolve_sample_image_path(args_yaml, object_name, sample):
    sample_path = Path(normalize_sample_path(sample))
    data_root = Path(args_yaml["data_root"])

    if args_yaml.get("eval_remaining_train_good", False):
        raise NotImplementedError("This inspector currently supports standard runs, not eval_remaining_train_good runs.")

    split_name = args_yaml.get("inference_split", "test")
    full_path = data_root / object_name / split_name / sample_path
    if not full_path.exists():
        raise FileNotFoundError(f"Could not resolve sample path {sample!r} to {full_path}")
    return full_path


def resolve_saved_patch_map_path(run_dir, seed, object_name, args_yaml, sample):
    sample_path = Path(normalize_sample_path(sample))
    if args_yaml.get("eval_remaining_train_good", False):
        output_group_root = "custom_eval"
    else:
        output_group_root = args_yaml.get("inference_split", "test")
    return run_dir / "anomaly_maps" / f"seed={seed}" / object_name / output_group_root / sample_path.parent / f"{sample_path.stem}.npy"


def default_output_dir(run_dir, seed, object_name, sample, patch_row, patch_col):
    sample_path = Path(normalize_sample_path(sample))
    return (
        run_dir
        / "patch_nn_inspection"
        / f"seed={seed}"
        / object_name
        / sample_path.parent
        / f"{sample_path.stem}__patch_r{patch_row:02d}_c{patch_col:02d}"
    )


def resolve_patch_coordinates(cli_args, model, sample_rgb, grid_size):
    uses_patch_indices = cli_args.patch_row is not None or cli_args.patch_col is not None
    uses_original_pixels = cli_args.x is not None or cli_args.y is not None

    if uses_patch_indices and uses_original_pixels:
        raise ValueError("Use either --patch-row/--patch-col or --x/--y, not both.")
    if not uses_patch_indices and not uses_original_pixels:
        raise ValueError("Provide either --patch-row/--patch-col or --x/--y.")

    grid_rows, grid_cols = int(grid_size[0]), int(grid_size[1])
    geometry = get_resize_and_crop_info(model, sample_rgb)

    if uses_patch_indices:
        if cli_args.patch_row is None or cli_args.patch_col is None:
            raise ValueError("Both --patch-row and --patch-col are required together.")
        patch_row = int(cli_args.patch_row)
        patch_col = int(cli_args.patch_col)
        selection_info = {
            "selection_mode": "patch_indices",
            "original_x": None,
            "original_y": None,
            "resized_x": None,
            "resized_y": None,
            "original_width": geometry["original_width"],
            "original_height": geometry["original_height"],
            "resized_width": geometry["resized_width"],
            "resized_height": geometry["resized_height"],
            "processed_width": geometry["processed_width"],
            "processed_height": geometry["processed_height"],
        }
    else:
        if cli_args.x is None or cli_args.y is None:
            raise ValueError("Both --x and --y are required together.")
        if geometry["patch_size"] is None:
            raise NotImplementedError("Original-image x/y lookup is currently implemented for dense patch backbones only.")

        original_x = int(cli_args.x)
        original_y = int(cli_args.y)
        if (
            original_x < 0
            or original_x >= geometry["original_width"]
            or original_y < 0
            or original_y >= geometry["original_height"]
        ):
            raise ValueError(
                f"Original-image point ({original_x}, {original_y}) is outside image size "
                f"{geometry['original_width']}x{geometry['original_height']}."
            )

        resized_x = min(
            geometry["resized_width"] - 1,
            int(np.floor(original_x * geometry["resized_width"] / geometry["original_width"])),
        )
        resized_y = min(
            geometry["resized_height"] - 1,
            int(np.floor(original_y * geometry["resized_height"] / geometry["original_height"])),
        )
        if resized_x >= geometry["processed_width"] or resized_y >= geometry["processed_height"]:
            raise ValueError(
                f"Original-image point ({original_x}, {original_y}) maps to resized point "
                f"({resized_x}, {resized_y}), which falls inside the cropped-away border. "
                f"Processed image size is {geometry['processed_width']}x{geometry['processed_height']}."
            )

        patch_size = geometry["patch_size"]
        patch_row = resized_y // patch_size
        patch_col = resized_x // patch_size
        selection_info = {
            "selection_mode": "original_image_pixels",
            "original_x": original_x,
            "original_y": original_y,
            "resized_x": int(resized_x),
            "resized_y": int(resized_y),
            "original_width": geometry["original_width"],
            "original_height": geometry["original_height"],
            "resized_width": geometry["resized_width"],
            "resized_height": geometry["resized_height"],
            "processed_width": geometry["processed_width"],
            "processed_height": geometry["processed_height"],
        }

    if patch_row < 0 or patch_row >= grid_rows or patch_col < 0 or patch_col >= grid_cols:
        raise ValueError(f"Patch ({patch_row}, {patch_col}) is outside grid {grid_rows}x{grid_cols}.")

    return patch_row, patch_col, selection_info, geometry


def main():
    cli_args = parse_args()
    run_dir = cli_args.run_dir.resolve()
    args_yaml = load_run_args(run_dir)

    object_name = infer_object_name(run_dir, cli_args.seed, cli_args.object_name)
    reference_images = load_reference_images(run_dir, object_name, cli_args.seed)

    device = cli_args.device or args_yaml.get("device", "cuda:0")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    objects, _, masking_default, rotation_default = get_dataset_info(
        args_yaml["dataset"],
        args_yaml["preprocess"],
        data_path=args_yaml["data_root"],
    )
    if object_name not in objects:
        raise ValueError(f"Object {object_name!r} was not inferred from data root {args_yaml['data_root']}")

    model = get_model(args_yaml["model_name"], device, smaller_edge_size=args_yaml["resolution"])

    with torch.inference_mode():
        features_ref, reference_metadata = build_memory_bank(
            model,
            args_yaml,
            object_name,
            reference_images,
            masking_default,
            rotation_default,
        )

        sample_path = resolve_sample_image_path(args_yaml, object_name, cli_args.sample)
        sample_rgb = load_rgb_image(sample_path)
        sample_tensor, grid_size = model.prepare_image(sample_rgb)
        sample_features = model.extract_features(sample_tensor)
        masking = masking_default[object_name]
        sample_mask = model.compute_background_mask(
            sample_features,
            grid_size,
            threshold=10,
            masking_type=masking,
        ).astype(bool)

    patch_row, patch_col, selection_info, geometry = resolve_patch_coordinates(
        cli_args, model, sample_rgb, grid_size
    )
    grid_rows, grid_cols = int(grid_size[0]), int(grid_size[1])

    flat_index = patch_row * grid_cols + patch_col
    if not sample_mask.reshape(grid_size)[patch_row, patch_col]:
        raise ValueError(
            f"Patch ({patch_row}, {patch_col}) is excluded by the background mask for this run."
        )

    query_feature = sample_features[flat_index]
    distances = compute_distances(query_feature, features_ref, args_yaml["knn_metric"])
    top_k = max(1, min(cli_args.top_k, distances.shape[0]))
    top_indices = np.argsort(distances)[:top_k]

    sample_patch_map_path = resolve_saved_patch_map_path(run_dir, cli_args.seed, object_name, args_yaml, cli_args.sample)
    saved_patch_distance = None
    if sample_patch_map_path.exists():
        saved_patch_map = np.load(sample_patch_map_path)
        saved_patch_distance = float(saved_patch_map[patch_row, patch_col])

    output_dir = (cli_args.output_dir or default_output_dir(
        run_dir, cli_args.seed, object_name, cli_args.sample, patch_row, patch_col
    )).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_sample_rgb = geometry["processed_rgb"]
    patch_size = geometry["patch_size"] or (processed_sample_rgb.shape[0] // grid_rows)
    test_annotated = draw_patch_box(
        processed_sample_rgb,
        patch_row,
        patch_col,
        patch_size,
        outline="red",
        label=f"test ({patch_row},{patch_col})",
    )
    test_annotated_path = output_dir / "test_patch_context.png"
    test_annotated.save(test_annotated_path)

    results = {
        "run_dir": str(run_dir),
        "object_name": object_name,
        "sample": normalize_sample_path(cli_args.sample),
        "sample_image_path": str(sample_path),
        "selection_mode": selection_info["selection_mode"],
        "original_x": selection_info["original_x"],
        "original_y": selection_info["original_y"],
        "resized_x": selection_info["resized_x"],
        "resized_y": selection_info["resized_y"],
        "original_width": selection_info["original_width"],
        "original_height": selection_info["original_height"],
        "resized_width": selection_info["resized_width"],
        "resized_height": selection_info["resized_height"],
        "processed_width": selection_info["processed_width"],
        "processed_height": selection_info["processed_height"],
        "patch_row": patch_row,
        "patch_col": patch_col,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "knn_metric": args_yaml["knn_metric"],
        "saved_patch_distance_from_run": saved_patch_distance,
        "reference_images": reference_images,
        "neighbors": [],
    }

    for rank, ref_index in enumerate(top_indices, start=1):
        ref_meta = reference_metadata[int(ref_index)]
        ref_rgb = load_rgb_image(Path(ref_meta["reference_image_path"]))
        if ref_meta["augmentation_label"] != "identity":
            aug_idx = ref_meta["augmentation_index"]
            ref_rgb = augment_image(ref_rgb, angles=ROTATION_ANGLES)[aug_idx]
        processed_ref_rgb = get_processed_rgb_for_display(model, ref_rgb)
        ref_patch_size = processed_ref_rgb.shape[0] // ref_meta["grid_rows"]
        ref_annotated = draw_patch_box(
            processed_ref_rgb,
            ref_meta["patch_row"],
            ref_meta["patch_col"],
            ref_patch_size,
            outline="blue",
            label=f"ref ({ref_meta['patch_row']},{ref_meta['patch_col']})",
        )

        ref_path = output_dir / f"reference_rank{rank:02d}_patch_context.png"
        ref_annotated.save(ref_path)

        side_by_side = concatenate_side_by_side(
            test_annotated,
            ref_annotated,
            left_text=(
                f"test patch ({patch_row},{patch_col})"
                + (f"  saved_dist={saved_patch_distance:.5f}" if saved_patch_distance is not None else "")
            ),
            right_text=(
                f"{ref_meta['reference_image_name']}  patch ({ref_meta['patch_row']},{ref_meta['patch_col']})"
                f"  dist={float(distances[ref_index]):.5f}"
            ),
            title_text=f"Nearest Neighbor Rank {rank}",
        )
        side_by_side_path = output_dir / f"comparison_rank{rank:02d}.png"
        side_by_side.save(side_by_side_path)

        left, top, right, bottom = patch_bbox(patch_row, patch_col, patch_size)
        query_patch_crop = Image.fromarray(processed_sample_rgb[top:bottom, left:right])
        query_patch_crop_path = output_dir / f"test_patch_crop_rank{rank:02d}.png"
        query_patch_crop.save(query_patch_crop_path)

        ref_left, ref_top, ref_right, ref_bottom = patch_bbox(
            ref_meta["patch_row"], ref_meta["patch_col"], ref_patch_size
        )
        ref_patch_crop = Image.fromarray(processed_ref_rgb[ref_top:ref_bottom, ref_left:ref_right])
        ref_patch_crop_path = output_dir / f"reference_patch_crop_rank{rank:02d}.png"
        ref_patch_crop.save(ref_patch_crop_path)

        neighbor_entry = {
            "rank": rank,
            "distance": float(distances[ref_index]),
            "reference_image_name": ref_meta["reference_image_name"],
            "reference_image_path": ref_meta["reference_image_path"],
            "augmentation_label": ref_meta["augmentation_label"],
            "patch_row": ref_meta["patch_row"],
            "patch_col": ref_meta["patch_col"],
            "grid_rows": ref_meta["grid_rows"],
            "grid_cols": ref_meta["grid_cols"],
            "processed_height": ref_meta["processed_height"],
            "processed_width": ref_meta["processed_width"],
            "processed_patch_bbox_xyxy": list(patch_bbox(ref_meta["patch_row"], ref_meta["patch_col"], ref_patch_size)),
            "reference_context_png": str(ref_path),
            "comparison_png": str(side_by_side_path),
            "reference_patch_crop_png": str(ref_patch_crop_path),
            "test_patch_crop_png": str(query_patch_crop_path),
        }
        results["neighbors"].append(neighbor_entry)

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(result_path)


if __name__ == "__main__":
    main()
