from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import dump

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
from model_building.boruta_mrmr_prefilter_maxminmean import (
    decode_feature_index,
    format_seconds,
    greedy_mrmr_rank,
    safe_abs_corrcoef,
    write_progress,
)


DEFAULT_OUTPUT_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "final_all_boxes_overthreshold_maxminmean_boruta_prefilter1000_relaxed_rbf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit final overlap+over-threshold max/min/mean Boruta-selected RBF-SVM on labeled ROIs, "
            "predict all ROIs, and render bbox overlays."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--multilayer-cache-subdir", type=str, default=DEFAULT_MULTILAYER_CACHE_SUBDIR)
    parser.add_argument("--prefilter-k", type=int, default=1000)
    parser.add_argument("--mrmr-prefilter-top", type=int, default=4096)
    parser.add_argument("--boruta-max-iter", type=int, default=100)
    parser.add_argument("--boruta-alpha", type=float, default=0.1)
    parser.add_argument("--shadow-percentile", type=float, default=95.0)
    parser.add_argument("--boruta-patience", type=int, default=40)
    parser.add_argument("--rf-n-estimators", type=int, default=256)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-max-features", type=str, default="sqrt")
    parser.add_argument("--rf-class-weight", type=str, default="balanced_subsample")
    parser.add_argument("--rf-n-jobs", type=int, default=1)
    parser.add_argument("--fallback-k", type=int, default=32)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-labels", nargs="*", default=("2D", "3D"))
    parser.add_argument("--font-scale", type=float, default=0.55)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def build_mrmr_prefilter(X_train: np.ndarray, y_train: np.ndarray, prefilter_k: int, mrmr_prefilter_top: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import StandardScaler

    feature_dim = int(X_train.shape[1])
    prefilter_k = int(min(max(1, prefilter_k), feature_dim))
    mrmr_prefilter_top = int(min(max(prefilter_k, mrmr_prefilter_top), feature_dim))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)
    relevance = mutual_info_classif(X_scaled, y_train, discrete_features=False, random_state=int(random_state))
    relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    seed_indices = np.argsort(-relevance, kind="stable")[:mrmr_prefilter_top]
    redundancy = safe_abs_corrcoef(X_scaled[:, seed_indices])
    ranked_local = greedy_mrmr_rank(relevance[seed_indices], redundancy, prefilter_k)
    prefilter_indices = seed_indices[ranked_local]
    return prefilter_indices.astype(np.int32), relevance


def run_boruta_all_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    prefilter_indices: np.ndarray,
    relevance: np.ndarray,
    args: argparse.Namespace,
    progress_file: Path,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    from scipy.stats import binomtest
    from sklearn.ensemble import RandomForestClassifier

    start_time = time.time()
    X_prefilter = X_train[:, prefilter_indices].astype(np.float32)
    n_features = int(X_prefilter.shape[1])
    statuses = np.zeros(n_features, dtype=np.int8)
    hits = np.zeros(n_features, dtype=np.int32)
    real_importance_sum = np.zeros(n_features, dtype=np.float64)
    real_importance_max = np.zeros(n_features, dtype=np.float64)
    iteration_rows: list[dict[str, object]] = []
    no_change_rounds = 0

    for iteration in range(1, int(args.boruta_max_iter) + 1):
        tentative_mask = statuses == 0
        num_tentative = int(tentative_mask.sum())
        num_confirmed = int((statuses == 1).sum())
        num_rejected = int((statuses == -1).sum())
        if num_tentative == 0:
            break

        iter_start = time.time()
        shadow = X_prefilter[:, tentative_mask].copy()
        rng = np.random.default_rng(int(args.random_state) + iteration)
        for column_index in range(shadow.shape[1]):
            rng.shuffle(shadow[:, column_index])

        X_boruta = np.concatenate([X_prefilter, shadow], axis=1)
        rf = RandomForestClassifier(
            n_estimators=int(args.rf_n_estimators),
            max_depth=None if args.rf_max_depth is None else int(args.rf_max_depth),
            min_samples_leaf=int(args.rf_min_samples_leaf),
            max_features=str(args.rf_max_features),
            class_weight=None if args.rf_class_weight == "none" else str(args.rf_class_weight),
            n_jobs=int(args.rf_n_jobs),
            random_state=int(args.random_state) + iteration,
        )
        rf.fit(X_boruta, y_train)
        importances = rf.feature_importances_.astype(np.float64)
        real_importances = importances[:n_features]
        shadow_importances = importances[n_features:]
        threshold = float(np.percentile(shadow_importances, float(args.shadow_percentile)))

        hits += (real_importances > threshold).astype(np.int32)
        real_importance_sum += real_importances
        real_importance_max = np.maximum(real_importance_max, real_importances)

        changed = False
        bonferroni = max(1, num_tentative)
        for feature_idx in np.where(tentative_mask)[0]:
            p_accept = binomtest(int(hits[feature_idx]), iteration, 0.5, alternative="greater").pvalue
            p_reject = binomtest(int(hits[feature_idx]), iteration, 0.5, alternative="less").pvalue
            if p_accept < (float(args.boruta_alpha) / float(bonferroni)):
                statuses[feature_idx] = 1
                changed = True
            elif p_reject < (float(args.boruta_alpha) / float(bonferroni)):
                statuses[feature_idx] = -1
                changed = True
        no_change_rounds = 0 if changed else (no_change_rounds + 1)

        elapsed = time.time() - start_time
        mean_iter = elapsed / float(iteration)
        eta_seconds = mean_iter * float(max(0, int(args.boruta_max_iter) - iteration))
        iteration_rows.append(
            {
                "iteration": int(iteration),
                "shadow_threshold": float(threshold),
                "num_confirmed": int((statuses == 1).sum()),
                "num_rejected": int((statuses == -1).sum()),
                "num_tentative": int((statuses == 0).sum()),
                "iteration_seconds": float(time.time() - iter_start),
                "elapsed_seconds": float(elapsed),
                "eta_seconds": float(eta_seconds),
            }
        )
        write_progress(
            {
                "status": "running",
                "phase": "boruta_final_model",
                "iteration": int(iteration),
                "boruta_max_iter": int(args.boruta_max_iter),
                "num_confirmed": int((statuses == 1).sum()),
                "num_rejected": int((statuses == -1).sum()),
                "num_tentative": int((statuses == 0).sum()),
                "elapsed_seconds": float(elapsed),
                "eta_seconds": float(eta_seconds),
                "eta_human": format_seconds(eta_seconds),
            },
            progress_file,
        )
        if int(args.boruta_patience) > 0 and no_change_rounds >= int(args.boruta_patience):
            break

    mean_importance = real_importance_sum / float(max(1, len(iteration_rows)))
    rows: list[dict[str, object]] = []
    for local_idx, feature_index in enumerate(prefilter_indices):
        status_value = int(statuses[local_idx])
        if status_value == 1:
            status_name = "confirmed"
        elif status_value == -1:
            status_name = "rejected"
        else:
            status_name = "tentative"
        row = {
            "prefilter_rank": int(local_idx + 1),
            "feature_index": int(feature_index),
            "status": status_name,
            "hits": int(hits[local_idx]),
            "hit_rate": float(hits[local_idx] / max(1, len(iteration_rows))),
            "mean_importance": float(mean_importance[local_idx]),
            "max_importance": float(real_importance_max[local_idx]),
            "mutual_information": float(relevance[feature_index]),
        }
        row.update(decode_feature_index(int(feature_index)))
        rows.append(row)

    confirmed = np.asarray([row["feature_index"] for row in rows if row["status"] == "confirmed"], dtype=np.int32)
    if confirmed.size == 0:
        tentative_rows = [row for row in rows if row["status"] == "tentative"]
        tentative_rows.sort(key=lambda row: (float(row["hit_rate"]), float(row["mean_importance"])), reverse=True)
        confirmed = np.asarray([row["feature_index"] for row in tentative_rows[: int(min(max(1, args.fallback_k), len(tentative_rows)))]], dtype=np.int32)
    return confirmed, rows, iteration_rows


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


def render_sample(sample_df: pd.DataFrame, output_path: Path, font_scale: float, thickness: int) -> dict[str, object]:
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


def build_features_for_rois(rois: pd.DataFrame, sample_map: dict[str, dict]) -> np.ndarray:
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
    progress_file = output_dir / "progress.json"
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
    y_train = np.where(y_train_labels == "3d", 1, 0).astype(np.int32)

    prefilter_indices, relevance = build_mrmr_prefilter(
        X_train=X_train,
        y_train=y_train,
        prefilter_k=int(args.prefilter_k),
        mrmr_prefilter_top=int(args.mrmr_prefilter_top),
        random_state=int(args.random_state),
    )
    selected_indices, boruta_rows, iteration_rows = run_boruta_all_data(
        X_train=X_train,
        y_train=y_train,
        prefilter_indices=prefilter_indices,
        relevance=relevance,
        args=args,
        progress_file=progress_file,
    )

    model = build_classifier(
        c_value=float(args.svm_c),
        gamma=str(args.svm_gamma),
        class_weight=None if args.class_weight == "none" else str(args.class_weight),
        random_state=int(args.random_state),
    )
    model.fit(X_train[:, selected_indices], y_train)

    proba_all = model.predict_proba(X_all[:, selected_indices]).astype(np.float32)
    pred_idx = model.predict(X_all[:, selected_indices]).astype(np.int32)
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
    pd.DataFrame(boruta_rows).to_csv(selection_csv, index=False)
    np.save(selection_npy, selected_indices.astype(np.int32))
    pd.DataFrame(iteration_rows).to_csv(output_dir / "boruta_iteration_log.csv", index=False)
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
        "selector": "boruta_shadow_random_forest",
        "prefilter_selector": "mrmr_mi_relevance_abs_corr_redundancy",
        "prefilter_k": int(args.prefilter_k),
        "mrmr_prefilter_top": int(args.mrmr_prefilter_top),
        "selected_feature_count_for_model": int(len(selected_indices)),
        "boruta_max_iter": int(args.boruta_max_iter),
        "boruta_alpha": float(args.boruta_alpha),
        "shadow_percentile": float(args.shadow_percentile),
        "boruta_patience": int(args.boruta_patience),
        "rf_n_estimators": int(args.rf_n_estimators),
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
        "selector": "boruta_shadow_random_forest",
        "prefilter_selector": "mrmr_mi_relevance_abs_corr_redundancy",
        "prefilter_k": int(args.prefilter_k),
        "mrmr_prefilter_top": int(args.mrmr_prefilter_top),
        "selected_feature_count_for_model": int(len(selected_indices)),
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
    write_progress(
        {
            "status": "completed",
            "phase": "done",
            "summary_json": str(summary_json),
            "elapsed_seconds": float(pd.DataFrame(iteration_rows)["elapsed_seconds"].max() if iteration_rows else 0.0),
            "eta_seconds": 0.0,
            "eta_human": "0s",
        },
        progress_file,
    )


if __name__ == "__main__":
    main()
