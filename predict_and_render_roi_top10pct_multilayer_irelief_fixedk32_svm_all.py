from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from extract_labeled_roi_toppercent_multilayer_softmax_patch_features import (
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_LABELS_FILE,
    DEFAULT_ROI_METADATA_CSV,
    build_multilayer_run_context,
    concatenated_patch_features,
    load_labels_table,
    load_multilayer_cache,
    load_patch_scores,
    load_roi_table,
    prepare_labeled_roi_table,
)
from extract_labeled_roi_toppercent_pca_softmax_patch_features import (
    bbox_patch_window,
    select_roi_patches_center_in_box,
    select_roi_patches_overlap,
)
from fit_roi_irelief_cosine import (
    build_weighted_feature_set,
    estimate_sigma,
    fit_irelief_cosine,
    l2_normalize_rows,
    pairwise_cosine_distance,
)


DEFAULT_OUTPUT_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "final_all_boxes_top10pct_multilayer_irelief_fixedk32_rbf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit final top10pct multilayer I-Relief fixed-k=32 RBF-SVM on labeled ROIs, "
            "predict all ROIs, and render bbox overlays."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--multilayer-cache-subdir", type=str, default="patch_feature_cache_multilayer_l1to12")
    parser.add_argument("--top-percent", type=float, default=0.10)
    parser.add_argument("--min-patches", type=int, default=1)
    parser.add_argument("--selection-mode", type=str, default="center_in_box", choices=("center_in_box", "overlap"))
    parser.add_argument("--fixed-k", type=int, default=32)
    parser.add_argument("--sigma-quantile", type=float, default=0.5)
    parser.add_argument("--min-sigma", type=float, default=1e-3)
    parser.add_argument("--irelief-max-iter", type=int, default=50)
    parser.add_argument("--irelief-tol", type=float, default=1e-6)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-labels", nargs="*", default=("2D", "3D"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--font-scale", type=float, default=0.55)
    parser.add_argument("--thickness", type=int, default=2)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().lower()


def sanitize_text(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and np.isnan(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def group_name_from_sample(sample: str) -> str:
    sample_path = Path(sample)
    parts = sample_path.parts
    if len(parts) >= 2:
        return parts[-2]
    if len(parts) == 1:
        return parts[0]
    return "ungrouped"


def clip_patch_window(
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    grid_rows: int,
    grid_cols: int,
) -> tuple[int, int, int, int]:
    row_min = max(0, min(row_min, grid_rows - 1))
    col_min = max(0, min(col_min, grid_cols - 1))
    row_max = max(row_min + 1, min(row_max, grid_rows))
    col_max = max(col_min + 1, min(col_max, grid_cols))
    return row_min, row_max, col_min, col_max


def select_from_patch_window(
    score_grid: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    top_percent: float,
    min_patches: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for patch_row in range(row_min, row_max):
        for patch_col in range(col_min, col_max):
            candidates.append((float(score_grid[patch_row, patch_col]), patch_row, patch_col))
    candidates.sort(key=lambda item: item[0], reverse=True)
    num_candidates = len(candidates)
    top_k = max(int(min_patches), int(np.ceil(float(top_percent) * float(num_candidates))))
    top_k = min(top_k, num_candidates)
    return [(patch_row, patch_col) for _, patch_row, patch_col in candidates[:top_k]]


def aggregate_selected_patch_features(
    patch_features_norm: np.ndarray,
    score_grid: np.ndarray,
    grid_shape: tuple[int, int],
    selected_patches: List[tuple[int, int]],
) -> np.ndarray:
    feature_rows: List[np.ndarray] = []
    anomaly_scores: List[float] = []
    for patch_row, patch_col in selected_patches:
        idx = patch_row * grid_shape[1] + patch_col
        feature_rows.append(patch_features_norm[idx])
        anomaly_scores.append(float(score_grid[patch_row, patch_col]))
    feature_matrix = np.stack(feature_rows, axis=0).astype(np.float32)
    anomaly_array = np.array(anomaly_scores, dtype=np.float32)
    if anomaly_array.size == 1:
        weights = np.array([1.0], dtype=np.float32)
    else:
        score_min = float(anomaly_array.min())
        score_max = float(anomaly_array.max())
        if score_max <= score_min:
            weights = np.full(anomaly_array.shape, 1.0 / anomaly_array.size, dtype=np.float32)
        else:
            logits = (anomaly_array - score_min) / max(score_max - score_min, 1e-8)
            logits = logits - float(logits.max())
            exp_logits = np.exp(logits).astype(np.float32)
            weights = (exp_logits / max(float(exp_logits.sum()), 1e-8)).astype(np.float32)
    combined = (feature_matrix * weights[:, None]).sum(axis=0)
    combined_norm = np.linalg.norm(combined)
    if combined_norm <= 1e-8:
        combined = feature_matrix.mean(axis=0)
        combined_norm = np.linalg.norm(combined)
    return (combined / max(float(combined_norm), 1e-8)).astype(np.float32)


def build_base_and_expand1_features(
    roi_table: pd.DataFrame,
    sample_map: dict[str, dict],
    top_percent: float,
    min_patches: int,
    selection_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_rows: list[np.ndarray] = []
    expand1_rows: list[np.ndarray] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], dict]] = {}

    for row in roi_table.itertuples(index=False):
        roi_row = pd.Series(row._asdict())
        sample_name = str(roi_row["sample"]).replace("\\", "/")
        sample_info = sample_map[sample_name]

        if sample_name not in cache:
            features_layers, grid_shape, _, cache_meta = load_multilayer_cache(Path(sample_info["feature_cache_path"]))
            concat_features = concatenated_patch_features(features_layers)
            score_grid = load_patch_scores(sample_info["run_sample"])
            cache[sample_name] = (concat_features, score_grid, grid_shape, cache_meta)

        concat_features, score_grid, grid_shape, cache_meta = cache[sample_name]

        if selection_mode == "overlap":
            selected_base, _, _ = select_roi_patches_overlap(
                roi_row,
                cache_meta,
                score_grid,
                top_percent=float(top_percent),
                min_patches=int(min_patches),
            )
        else:
            selected_base, _, _ = select_roi_patches_center_in_box(
                roi_row,
                cache_meta,
                score_grid,
                top_percent=float(top_percent),
                min_patches=int(min_patches),
            )
        base_feature = aggregate_selected_patch_features(
            concat_features,
            score_grid,
            grid_shape,
            selected_base,
        )
        base_rows.append(base_feature)

        row_min, row_max, col_min, col_max = bbox_patch_window(roi_row, cache_meta)
        row_min, row_max, col_min, col_max = clip_patch_window(
            row_min - 1,
            row_max + 1,
            col_min - 1,
            col_max + 1,
            cache_meta["grid_rows"],
            cache_meta["grid_cols"],
        )
        selected_expand = select_from_patch_window(
            score_grid,
            row_min,
            row_max,
            col_min,
            col_max,
            top_percent=float(top_percent),
            min_patches=int(min_patches),
        )
        expand1_feature = aggregate_selected_patch_features(
            concat_features,
            score_grid,
            grid_shape,
            selected_expand,
        )
        expand1_rows.append(expand1_feature)

    return np.stack(base_rows, axis=0).astype(np.float32), np.stack(expand1_rows, axis=0).astype(np.float32)


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_x1 = max(0, x)
    box_y2 = max(text_h + baseline + 4, y)
    box_y1 = max(0, box_y2 - text_h - baseline - 8)
    box_x2 = min(image.shape[1] - 1, box_x1 + text_w + 8)
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(
        image,
        text,
        (box_x1 + 4, box_y2 - baseline - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def box_color(row: pd.Series) -> tuple[int, int, int]:
    used = bool(int(row["used_for_training"]))
    if not used:
        return (160, 160, 160)

    correct = bool(int(row["correct_labeled"]))
    pred = sanitize_text(row["predicted_label"]).lower()
    if correct:
        return (40, 180, 40)
    if pred == "3d":
        return (0, 80, 220)
    return (0, 140, 255)


def render_sample(
    sample_df: pd.DataFrame,
    output_path: Path,
    font_scale: float,
    thickness: int,
) -> dict[str, object]:
    image_path = Path(str(sample_df.iloc[0]["image_path"]))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    ordered = sample_df.sort_values(["roi_index", "roi_nummer", "x_min", "y_min"]).reset_index(drop=True)
    rendered_rows = 0

    for _, row in ordered.iterrows():
        x_min = int(row["x_min"])
        y_min = int(row["y_min"])
        x_max = int(row["x_max"])
        y_max = int(row["y_max"])
        roi_name = sanitize_text(row.get("roi_nummer"), f"roi{int(row.get('roi_index', 0))}")
        pred = sanitize_text(row.get("predicted_label"), "?").upper()
        prob = float(row.get("predicted_probability", 0.0))
        gt = sanitize_text(row.get("label_display"), "?").upper()
        used = "yes" if int(row.get("used_for_training", 0)) else "no"
        color = box_color(row)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        text = f"{roi_name} pred={pred} {prob*100:.1f}% gt={gt} used={used}"
        label_y = max(18, y_min - 4)
        draw_label(
            image=image,
            text=text,
            x=x_min,
            y=label_y,
            color=color,
            font_scale=font_scale,
            thickness=max(1, thickness - 1),
        )
        rendered_rows += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return {
        "sample": sample_df.iloc[0]["sample"],
        "image_path": str(image_path),
        "output_path": str(output_path),
        "num_boxes": rendered_rows,
        "num_used_labeled_boxes": int(sample_df["used_for_training"].sum()),
        "num_unlabeled_boxes": int((1 - sample_df["used_for_training"]).sum()),
    }


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    roi_metadata_csv = args.roi_metadata_csv.resolve()
    labels_file = args.labels_file.resolve()
    output_dir = args.output_dir.resolve()
    overlays_dir = output_dir / "overlay_images"
    ensure_dir(output_dir)
    ensure_dir(overlays_dir)

    roi_table = load_roi_table(roi_metadata_csv)
    labels_table = load_labels_table(labels_file, list(args.valid_labels) if args.valid_labels else None)
    labeled_rois = prepare_labeled_roi_table(roi_table, labels_table, limit=None).copy()
    labeled_rois["label"] = labeled_rois["label"].astype(str).str.strip()
    labeled_rois = labeled_rois[labeled_rois["label"].isin(list(args.valid_labels))].reset_index(drop=True)
    if labeled_rois.empty:
        raise ValueError("No labeled ROIs found after filtering valid labels.")

    all_rois = roi_table.copy()
    label_lookup = labeled_rois[["roi_uid", "label", "detailed_label"]].copy()
    label_lookup["label_lower"] = label_lookup["label"].map(clean_label)
    all_rois = all_rois.merge(
        label_lookup[["roi_uid", "label", "detailed_label", "label_lower"]],
        on="roi_uid",
        how="left",
    )
    all_rois["label_lower"] = all_rois["label_lower"].fillna("")

    sample_map = build_multilayer_run_context(
        experiment_dir,
        seed=int(args.seed),
        cache_subdir=str(args.multilayer_cache_subdir),
    )
    all_rois["image_path"] = all_rois["sample"].map(lambda sample: sample_map[str(sample).replace("\\", "/")]["image_path"])

    X_base_all, X_expand1_all = build_base_and_expand1_features(
        roi_table=all_rois,
        sample_map=sample_map,
        top_percent=float(args.top_percent),
        min_patches=int(args.min_patches),
        selection_mode=str(args.selection_mode),
    )
    labeled_mask = all_rois["label_lower"].isin({"2d", "3d"}).to_numpy()
    X_train_raw = X_base_all[labeled_mask]
    X_train_expand1_raw = X_expand1_all[labeled_mask]
    y_train_labels = all_rois.loc[labeled_mask, "label_lower"].astype(str).to_numpy()

    features_unit = l2_normalize_rows(X_train_raw)
    sigma = estimate_sigma(
        pairwise_cosine_distance(features_unit),
        quantile=float(args.sigma_quantile),
        min_sigma=float(args.min_sigma),
    )
    weights, trace = fit_irelief_cosine(
        features_unit=features_unit,
        labels=y_train_labels,
        sigma=float(sigma),
        max_iter=int(args.irelief_max_iter),
        tol=float(args.irelief_tol),
    )
    ranked_indices = np.argsort(-weights, kind="stable")
    selected = ranked_indices[: int(args.fixed_k)]

    X_train_weighted = build_weighted_feature_set(features_unit, weights)
    X_train_expand1_weighted = build_weighted_feature_set(l2_normalize_rows(X_train_expand1_raw), weights)
    X_all_weighted = build_weighted_feature_set(l2_normalize_rows(X_base_all), weights)

    label_to_int = {"2d": 0, "3d": 1}
    y_train = np.array([label_to_int[label] for label in y_train_labels], dtype=np.int32)

    X_fit = np.vstack(
        [
            X_train_weighted[:, selected],
            X_train_expand1_weighted[:, selected],
        ]
    ).astype(np.float32)
    y_fit = np.concatenate([y_train, y_train], axis=0)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=args.svm_c,
                    gamma=args.svm_gamma,
                    class_weight=None if args.class_weight == "none" else args.class_weight,
                    probability=True,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    model.fit(X_fit, y_fit)

    X_all_selected = X_all_weighted[:, selected]
    proba_all = model.predict_proba(X_all_selected).astype(np.float32)
    pred_idx = model.predict(X_all_selected).astype(np.int32)
    pred_all = np.where(pred_idx == 1, "3d", "2d")
    pred_prob = np.where(pred_all == "3d", proba_all[:, 1], proba_all[:, 0])

    table = all_rois.copy()
    table["used_for_training"] = labeled_mask.astype(int)
    table["predicted_label"] = pred_all
    table["proba_2d"] = proba_all[:, 0]
    table["proba_3d"] = proba_all[:, 1]
    table["predicted_probability"] = pred_prob
    table["label_display"] = table["label_lower"].replace("", "?")
    table["correct_labeled"] = np.where(
        table["used_for_training"].astype(bool),
        (table["label_lower"] == table["predicted_label"]).astype(int),
        -1,
    )

    manifest_rows: list[dict[str, object]] = []
    for sample, sample_df in table.groupby("sample", sort=True):
        group_name = group_name_from_sample(str(sample))
        bildname = sanitize_text(sample_df.iloc[0].get("bildname"), Path(str(sample)).name)
        output_path = overlays_dir / group_name / bildname
        manifest_rows.append(
            render_sample(
                sample_df=sample_df,
                output_path=output_path,
                font_scale=args.font_scale,
                thickness=args.thickness,
            )
        )

    all_predictions_csv = output_dir / "all_box_predictions.csv"
    manifest_csv = output_dir / "manifest.csv"
    summary_json = output_dir / "summary.json"
    model_info_json = output_dir / "model_info.json"
    selection_csv = output_dir / "selected_topk_features.csv"

    table.to_csv(all_predictions_csv, index=False)
    manifest_df = pd.DataFrame(manifest_rows).sort_values("sample").reset_index(drop=True)
    manifest_df.to_csv(manifest_csv, index=False)
    pd.DataFrame(
        {
            "rank": np.arange(1, len(selected) + 1, dtype=np.int32),
            "feature_index": selected.astype(np.int32),
            "weight": weights[selected].astype(np.float32),
        }
    ).to_csv(selection_csv, index=False)

    summary = {
        "experiment_dir": str(experiment_dir),
        "roi_metadata_csv": str(roi_metadata_csv),
        "labels_file": str(labels_file),
        "output_dir": str(output_dir),
        "overlay_dir": str(overlays_dir),
        "num_total_boxes": int(len(table)),
        "num_labeled_boxes_used_for_training": int(table["used_for_training"].sum()),
        "num_unlabeled_boxes_predicted_only": int((1 - table["used_for_training"]).sum()),
        "num_images": int(manifest_df.shape[0]),
        "all_box_predictions_csv": str(all_predictions_csv),
        "manifest_csv": str(manifest_csv),
        "selected_topk_features_csv": str(selection_csv),
        "sigma": float(sigma),
        "irelief_iterations": int(len(trace)),
        "fixed_k": int(args.fixed_k),
        "selection_mode": str(args.selection_mode),
    }
    model_info = {
        "classifier": "svm_rbf",
        "aggregation": f"top10pct_{args.selection_mode}_softmax_multilayer",
        "augmentation_during_training": "expand1",
        "feature_weighting": "irelief_cosine",
        "fixed_k": int(args.fixed_k),
        "svm_c": float(args.svm_c),
        "svm_gamma": str(args.svm_gamma),
        "class_weight": None if args.class_weight == "none" else args.class_weight,
        "random_state": int(args.random_state),
        "classes": ["2d", "3d"],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    model_info_json.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    print(f"Saved predictions CSV: {all_predictions_csv}")
    print(f"Saved overlays: {overlays_dir}")
    print(f"Saved manifest: {manifest_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
