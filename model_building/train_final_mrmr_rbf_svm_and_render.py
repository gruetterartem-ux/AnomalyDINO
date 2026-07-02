from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler

from extract_labeled_roi_overthreshold_multilayer_maxminmean_features import (
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_LABELS_FILE,
    DEFAULT_MULTILAYER_CACHE_SUBDIR,
    DEFAULT_ROI_METADATA_CSV,
    aggregate_maxminmean_per_layer,
    select_overlap_threshold_patches,
)
from extract_labeled_roi_toppercent_multilayer_softmax_patch_features import (
    build_multilayer_run_context,
    load_labels_table,
    load_multilayer_cache,
    load_patch_scores,
    load_roi_table,
    prepare_labeled_roi_table,
)
from model_building.rbf_svm_utils import build_classifier
from model_building.overlay_render_utils import (
    draw_label,
    group_name_from_sample,
    sanitize_text,
)


DEFAULT_OUTPUT_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "final_all_boxes_overthreshold_maxminmean_mrmr_fixedk384_rbf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit final overlap+over-threshold max/min/mean mRMR fixed-k RBF-SVM on labeled ROIs, "
            "predict all ROIs, and render bbox overlays."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--multilayer-cache-subdir", type=str, default=DEFAULT_MULTILAYER_CACHE_SUBDIR)
    parser.add_argument("--fixed-k", type=int, default=384)
    parser.add_argument("--prefilter-top", type=int, default=4096)
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


def write_json(data: dict, output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip().lower()


def safe_abs_corrcoef(matrix: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    return corr


def greedy_mrmr_rank(relevance: np.ndarray, redundancy_matrix: np.ndarray, max_select: int) -> np.ndarray:
    selected_local: list[int] = []
    selected_mask = np.zeros(relevance.shape[0], dtype=bool)
    redundancy_sum = np.zeros(relevance.shape[0], dtype=np.float32)
    for iteration in range(max_select):
        if iteration == 0:
            scores = relevance.copy()
        else:
            scores = relevance - (redundancy_sum / float(iteration))
        scores[selected_mask] = -np.inf
        best_local = int(np.argmax(scores))
        selected_local.append(best_local)
        selected_mask[best_local] = True
        redundancy_sum += redundancy_matrix[:, best_local]
    return np.asarray(selected_local, dtype=np.int32)


def rank_mrmr_features(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    fixed_k: int,
    prefilter_top: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    selector_scaler = StandardScaler()
    X_scaled = selector_scaler.fit_transform(X_train_raw).astype(np.float32)
    relevance = mutual_info_classif(X_scaled, y_train, discrete_features=False, random_state=int(random_state))
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    prefilter_top = int(min(max(prefilter_top, fixed_k), X_train_raw.shape[1]))
    prefilter_indices = np.argsort(-relevance, kind="stable")[:prefilter_top]
    redundancy = safe_abs_corrcoef(X_scaled[:, prefilter_indices])
    ranked_local = greedy_mrmr_rank(relevance[prefilter_indices], redundancy, fixed_k)
    ranked_indices = prefilter_indices[ranked_local]
    return ranked_indices.astype(np.int32), relevance


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
    import cv2

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


def build_features_for_rois(
    rois: pd.DataFrame,
    sample_map: dict[str, dict],
) -> np.ndarray:
    feature_rows: list[np.ndarray] = []
    sample_cache: dict[str, tuple[np.ndarray, tuple[int, int], list[int], dict, np.ndarray, float]] = {}

    for row in rois.itertuples(index=False):
        roi_row = pd.Series(row._asdict())
        sample_name = str(roi_row["sample"]).replace("\\", "/")
        sample_info = sample_map[sample_name]
        if sample_name not in sample_cache:
            features_layers, grid_shape, layer_indices, cache_meta = load_multilayer_cache(Path(sample_info["feature_cache_path"]))
            score_grid = load_patch_scores(sample_info["run_sample"])
            image_threshold = float(sample_info["image_threshold"])
            sample_cache[sample_name] = (features_layers, grid_shape, layer_indices, cache_meta, score_grid, image_threshold)
        features_layers, grid_shape, _, cache_meta, score_grid, image_threshold = sample_cache[sample_name]

        selected_patches, _, _ = select_overlap_threshold_patches(
            row=roi_row,
            meta=cache_meta,
            anomaly_grid=score_grid,
            image_threshold=image_threshold,
        )
        feature_rows.append(aggregate_maxminmean_per_layer(features_layers, selected_patches, grid_shape))

    return np.stack(feature_rows, axis=0).astype(np.float32)


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
    all_rois["image_path"] = all_rois["sample"].map(
        lambda sample: sample_map[str(sample).replace("\\", "/")]["image_path"]
    )

    X_all = build_features_for_rois(all_rois, sample_map)
    labeled_mask = all_rois["label_lower"].isin({"2d", "3d"}).to_numpy()
    X_train = X_all[labeled_mask]
    y_train_labels = all_rois.loc[labeled_mask, "label_lower"].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    label_encoder.fit(["2d", "3d"])
    y_train = label_encoder.transform(y_train_labels)

    selected, relevance = rank_mrmr_features(
        X_train_raw=X_train,
        y_train=y_train,
        fixed_k=int(args.fixed_k),
        prefilter_top=int(args.prefilter_top),
        random_state=int(args.random_state),
    )

    model = build_classifier(
        c_value=float(args.svm_c),
        gamma=str(args.svm_gamma),
        class_weight=None if args.class_weight == "none" else str(args.class_weight),
        random_state=int(args.random_state),
    )
    model.fit(X_train[:, selected], y_train)

    proba_all = model.predict_proba(X_all[:, selected]).astype(np.float32)
    pred_idx = model.predict(X_all[:, selected]).astype(np.int32)
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
    selection_csv = output_dir / "selected_features.csv"
    selection_npy = output_dir / "selected_feature_indices.npy"
    classifier_joblib = output_dir / "classifier_pipeline.joblib"

    table.to_csv(all_predictions_csv, index=False)
    pd.DataFrame(manifest_rows).sort_values("sample").reset_index(drop=True).to_csv(manifest_csv, index=False)
    pd.DataFrame(
        {
            "rank": np.arange(1, len(selected) + 1, dtype=np.int32),
            "feature_index": selected.astype(np.int32),
            "mutual_information": relevance[selected].astype(np.float32),
        }
    ).to_csv(selection_csv, index=False)
    np.save(selection_npy, selected.astype(np.int32))
    dump(model, classifier_joblib)

    summary = {
        "experiment_dir": str(experiment_dir),
        "roi_metadata_csv": str(roi_metadata_csv),
        "labels_file": str(labels_file),
        "output_dir": str(output_dir),
        "overlay_dir": str(overlays_dir),
        "num_total_boxes": int(len(table)),
        "num_labeled_boxes_used_for_training": int(table["used_for_training"].sum()),
        "num_unlabeled_boxes_predicted_only": int((1 - table["used_for_training"]).sum()),
        "fixed_k": int(args.fixed_k),
        "prefilter_top": int(args.prefilter_top),
        "selector": "mrmr_mi_relevance_abs_corr_redundancy",
        "classifier": "rbf_svm",
        "selection_mode": "overlap",
        "candidate_rule": "overlap_and_score_gt_image_threshold",
        "aggregation_mode": "max_plus_min_plus_mean",
        "classifier_joblib": str(classifier_joblib),
        "selected_features_csv": str(selection_csv),
        "all_box_predictions_csv": str(all_predictions_csv),
        "manifest_csv": str(manifest_csv),
    }
    write_json(summary, summary_json)

    model_info = {
        "selector": "mrmr_mi_relevance_abs_corr_redundancy",
        "fixed_k": int(args.fixed_k),
        "prefilter_top": int(args.prefilter_top),
        "svm_kernel": "rbf",
        "svm_c": float(args.svm_c),
        "svm_gamma": str(args.svm_gamma),
        "class_weight": None if args.class_weight == "none" else str(args.class_weight),
        "selection_mode": "overlap",
        "candidate_rule": "overlap_and_score_gt_image_threshold",
        "aggregation_mode": "max_plus_min_plus_mean",
        "selected_features_csv": str(selection_csv),
        "selected_feature_indices_npy": str(selection_npy),
        "classifier_joblib": str(classifier_joblib),
    }
    write_json(model_info, model_info_json)


if __name__ == "__main__":
    main()
