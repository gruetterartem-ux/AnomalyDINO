import argparse
import csv
import json
import shutil
from pathlib import Path

from src.visualize import export_anomaly_map_pngs


MODALITIES = ("buttons", "normalmap", "albedo")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect patch-grid PNGs for test images where exactly one modality "
            "predicts the image correctly and the other two do not."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\anomalydino_data\patchgrid_exactly_one_correct_20260410"),
        help="Directory where comparison folders and manifests will be written.",
    )
    return parser.parse_args()


def ensure_patch_grid_exports(run_dir: Path, data_root: Path):
    anomaly_maps_dir = run_dir / "anomaly_maps" / "seed=0"
    has_patch_grids = any(
        path.name == "Patch-Gitter_png"
        for path in anomaly_maps_dir.rglob("*")
        if path.is_dir()
    )
    if not has_patch_grids:
        export_anomaly_map_pngs(str(anomaly_maps_dir), str(data_root))


def load_predictions(measurements_file: Path):
    rows = {}
    threshold = None
    with measurements_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if threshold is None and row.get("Threshold"):
                threshold = float(row["Threshold"])
            sample = row["Sample"]
            score = float(row["Anomaly_Score"])
            label = row["Label"].strip()
            true_anomaly = label == "1"
            pred_anomaly = score >= threshold
            rows[sample] = {
                "sample": sample,
                "score": score,
                "threshold": threshold,
                "true_anomaly": true_anomaly,
                "pred_anomaly": pred_anomaly,
                "correct": pred_anomaly == true_anomaly,
                "evaluation_group": row["Evaluation_Group"],
            }
    if threshold is None:
        raise ValueError(f"Could not infer threshold from {measurements_file}")
    return rows


def patch_grid_png_path(run_dir: Path, modality: str, sample: str):
    sample_path = Path(sample)
    return (
        run_dir
        / "anomaly_maps"
        / "seed=0"
        / modality
        / "test"
        / sample_path.parent
        / "Patch-Gitter_png"
        / f"{sample_path.stem}.png"
    )


def reset_output_dir(output_dir: Path):
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    run_dirs = {
        "buttons": Path(
            r"C:\ai\AnomalyDINO\results_CUSTOM\dinov2_vitb14_704"
            r"\16-shot_preprocess=force_no_mask_no_rotation_all16_test_maxpatch_random_20260410"
        ),
        "normalmap": Path(
            r"C:\ai\AnomalyDINO\results_CUSTOM\dinov2_vitb14_704"
            r"\16-shot_preprocess=force_no_mask_no_rotation_all16_test_maxpatch_buttonsrefs_normalmap_20260410"
        ),
        "albedo": Path(
            r"C:\ai\AnomalyDINO\results_CUSTOM\dinov2_vitb14_704"
            r"\16-shot_preprocess=force_no_mask_no_rotation_all16_test_maxpatch_buttonsrefs_albedo_20260410"
        ),
    }
    data_roots = {
        "buttons": Path(r"C:\anomalydino_data"),
        "normalmap": Path(r"C:\anomalydino_data_single_object_normalmap"),
        "albedo": Path(r"C:\anomalydino_data_single_object_albedo"),
    }

    for modality in MODALITIES:
        ensure_patch_grid_exports(run_dirs[modality], data_roots[modality])

    predictions = {
        modality: load_predictions(run_dirs[modality] / "measurements_seed=0.csv")
        for modality in MODALITIES
    }
    all_samples = set(predictions["buttons"].keys())
    for modality in MODALITIES[1:]:
        if set(predictions[modality].keys()) != all_samples:
            raise ValueError(f"Sample mismatch between buttons and {modality}")

    output_dir = args.output_dir
    reset_output_dir(output_dir)

    manifests = {modality: [] for modality in MODALITIES}

    for sample in sorted(all_samples):
        correctness = {modality: predictions[modality][sample]["correct"] for modality in MODALITIES}
        correct_modalities = [modality for modality, is_correct in correctness.items() if is_correct]
        if len(correct_modalities) != 1:
            continue

        winner = correct_modalities[0]
        case_root = output_dir / f"only_{winner}_correct" / Path(sample).parent / Path(sample).stem
        case_root.mkdir(parents=True, exist_ok=True)

        manifest_row = {
            "sample": sample,
            "true_label": "anomal" if predictions[winner][sample]["true_anomaly"] else "good",
            "winning_modality": winner,
        }

        for modality in MODALITIES:
            source_png = patch_grid_png_path(run_dirs[modality], modality, sample)
            if not source_png.exists():
                raise FileNotFoundError(f"Missing patch-grid PNG: {source_png}")

            target_png = case_root / f"{modality}_Patch-Gitter.png"
            shutil.copy2(source_png, target_png)

            pred = predictions[modality][sample]
            manifest_row[f"{modality}_score"] = f"{pred['score']:.5f}"
            manifest_row[f"{modality}_threshold"] = f"{pred['threshold']:.5f}"
            manifest_row[f"{modality}_pred"] = "anomal" if pred["pred_anomaly"] else "good"
            manifest_row[f"{modality}_correct"] = pred["correct"]
            manifest_row[f"{modality}_png"] = str(target_png)

        manifests[winner].append(manifest_row)

    summary = {}
    for modality in MODALITIES:
        case_dir = output_dir / f"only_{modality}_correct"
        case_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = case_dir / "manifest.csv"
        rows = manifests[modality]
        fieldnames = [
            "sample",
            "true_label",
            "winning_modality",
            "buttons_score",
            "buttons_threshold",
            "buttons_pred",
            "buttons_correct",
            "buttons_png",
            "normalmap_score",
            "normalmap_threshold",
            "normalmap_pred",
            "normalmap_correct",
            "normalmap_png",
            "albedo_score",
            "albedo_threshold",
            "albedo_pred",
            "albedo_correct",
            "albedo_png",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        summary[f"only_{modality}_correct"] = {
            "count": len(rows),
            "manifest": str(manifest_path),
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
