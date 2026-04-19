from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from component_labeling_app.session_io import load_session
from component_labeling_app.rendering import draw_patch_grid, grid_edges
from component_memory_bank.components import PatchComponent
from component_memory_bank.data_io import RunSample, load_patch_features, load_run_samples
from component_memory_bank.export import write_csv, write_json
from component_memory_bank.inference import (
    classify_components,
    classify_part,
    compute_patch_class_scores,
    summarize_components,
)
from component_memory_bank.memory_bank import (
    ComponentLabelRecord,
    ManualPatchLabelRecord,
    build_memory_banks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render patch-evidence sheets for the component memory-bank kNN logic, "
            "showing which active patches lean toward 2D or 3D."
        )
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--evaluation-group", type=str, default="test/bad")
    parser.add_argument("--k-neighbors", type=int, default=3)
    parser.add_argument("--anomaly-threshold", type=float, default=None)
    parser.add_argument("--tau-s", type=float, default=0.0)
    parser.add_argument("--tau-n", type=int, default=1)
    parser.add_argument("--tau-p", type=float, default=0.0)
    parser.add_argument("--max-patches-per-sheet", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _slug_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _component_label_records(df: pd.DataFrame) -> list[ComponentLabelRecord]:
    labeled = df[df["label"].fillna("").astype(str).isin(["2D", "3D", "skip"])].copy()
    records: list[ComponentLabelRecord] = []
    for row in labeled.itertuples(index=False):
        records.append(
            ComponentLabelRecord(
                object_name=str(row.object_name),
                sample=str(row.sample).replace("\\", "/"),
                component_id=int(row.component_id),
                anomaly_threshold=float(row.anomaly_threshold),
                top_k=int(row.top_k),
                label=str(row.label),
            )
        )
    return records


def _manual_patch_label_records(df: pd.DataFrame) -> list[ManualPatchLabelRecord]:
    if df.empty:
        return []
    labeled = df[df["label"].fillna("").astype(str).isin(["2D", "3D"])].copy()
    records: list[ManualPatchLabelRecord] = []
    for row in labeled.itertuples(index=False):
        records.append(
            ManualPatchLabelRecord(
                object_name=str(row.object_name),
                sample=str(row.sample).replace("\\", "/"),
                row=int(row.row),
                col=int(row.col),
                patch_index=int(row.patch_index),
                anomaly_score=float(row.anomaly_score),
                label=str(row.label),
            )
        )
    return records


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def _nearest_cosine_distance(query_features: np.ndarray, bank_features: np.ndarray) -> np.ndarray:
    query_norm = _normalize_rows(query_features)
    bank_norm = _normalize_rows(bank_features)
    cosine_sim = query_norm @ bank_norm.T
    best_sim = cosine_sim.max(axis=1)
    return 1.0 - best_sim


def _load_display_image(sample: RunSample) -> tuple[np.ndarray, tuple[int, int], int]:
    with np.load(sample.feature_cache_path) as cache_data:
        resized_h, resized_w = [int(v) for v in cache_data["resized_size"].tolist()]
        patch_size = int(np.asarray(cache_data["patch_size"]).reshape(-1)[0])
        grid_rows, grid_cols = [int(v) for v in cache_data["grid_size"].tolist()]
        image_path = Path(str(cache_data["image_path"].tolist()))

    image = Image.open(image_path).convert("RGB")
    image = image.resize((resized_w, resized_h), resample=Image.Resampling.BICUBIC)
    image_np = np.array(image)
    crop_h = grid_rows * patch_size
    crop_w = grid_cols * patch_size
    return image_np[:crop_h, :crop_w].copy(), (grid_rows, grid_cols), patch_size


def _fit_on_canvas(image_rgb: np.ndarray, target_size: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    scale = min(target_size / w, target_size / h)
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((target_size, target_size, 3), 18, dtype=np.uint8)
    y0 = (target_size - out_h) // 2
    x0 = (target_size - out_w) // 2
    canvas[y0 : y0 + out_h, x0 : x0 + out_w] = resized
    return canvas


def _draw_text_lines(
    canvas: np.ndarray,
    lines: Iterable[str],
    start_xy: tuple[int, int],
    line_height: int,
    color: tuple[int, int, int] = (230, 230, 230),
    font_scale: float = 0.55,
) -> None:
    x, y = start_xy
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
        y += line_height


def _render_patch_overlay(
    image_rgb: np.ndarray,
    grid_shape: tuple[int, int],
    patch_rows: list[dict[str, object]],
    component_decisions_by_id: dict[int, object],
    component_summaries_by_id: dict[int, dict[str, object]],
) -> np.ndarray:
    canvas = image_rgb.copy()
    row_edges, col_edges = grid_edges(canvas.shape, grid_shape)
    overlay = canvas.copy()

    for patch in patch_rows:
        row = int(patch["row"])
        col = int(patch["col"])
        y0, y1 = row_edges[row], row_edges[row + 1]
        x0, x1 = col_edges[col], col_edges[col + 1]
        evidence_label = str(patch["evidence_label"])
        if evidence_label == "3D":
            color = (220, 50, 50)
        elif evidence_label == "2D":
            color = (40, 170, 70)
        else:
            color = (220, 180, 40)
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, -1)

    canvas = cv2.addWeighted(canvas, 0.6, overlay, 0.4, 0.0)

    for component_id, decision in component_decisions_by_id.items():
        if not decision.is_3d:
            continue
        component_summary = component_summaries_by_id.get(component_id)
        if component_summary is None:
            continue
        component = component_summary["component"]
        y0, y1 = row_edges[component.bbox_row_min], row_edges[component.bbox_row_max + 1]
        x0, x1 = col_edges[component.bbox_col_min], col_edges[component.bbox_col_max + 1]
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 3)
        cv2.putText(
            canvas,
            f"C{component_id}",
            (x0 + 4, min(y1 - 8, y0 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    return draw_patch_grid(canvas, grid_shape)


def _build_patch_tile(
    image_rgb: np.ndarray,
    grid_shape: tuple[int, int],
    patch_row: dict[str, object],
    tile_size: tuple[int, int] = (260, 300),
) -> np.ndarray:
    tile_w, tile_h = tile_size
    tile = np.full((tile_h, tile_w, 3), 20, dtype=np.uint8)
    row_edges, col_edges = grid_edges(image_rgb.shape, grid_shape)

    row = int(patch_row["row"])
    col = int(patch_row["col"])
    rows, cols = grid_shape
    row0 = max(0, row - 1)
    row1 = min(rows, row + 2)
    col0 = max(0, col - 1)
    col1 = min(cols, col + 2)
    y0, y1 = row_edges[row0], row_edges[row1]
    x0, x1 = col_edges[col0], col_edges[col1]
    crop = image_rgb[y0:y1, x0:x1].copy()

    cy0, cy1 = row_edges[row] - y0, row_edges[row + 1] - y0
    cx0, cx1 = col_edges[col] - x0, col_edges[col + 1] - x0
    evidence_label = str(patch_row["evidence_label"])
    border_color = (220, 50, 50) if evidence_label == "3D" else (40, 170, 70)
    cv2.rectangle(crop, (cx0, cy0), (cx1 - 1, cy1 - 1), (255, 255, 255), 2)

    crop_canvas = _fit_on_canvas(crop, 180)
    cv2.rectangle(crop_canvas, (1, 1), (crop_canvas.shape[1] - 2, crop_canvas.shape[0] - 2), border_color, 3)
    tile[10:190, (tile_w - 180) // 2 : (tile_w + 180) // 2] = crop_canvas

    nearest_label = str(patch_row["nearest_neighbor_label"])
    component_label = str(patch_row["component_predicted_label"])
    lines = [
        f"r{row:02d} c{col:02d} p{int(patch_row['patch_index']):04d}",
        f"ev={evidence_label} nn={nearest_label} comp={component_label}",
        f"a={float(patch_row['anomaly_score']):.4f} c={float(patch_row['margin_c']):.4f}",
        f"z={float(patch_row['weighted_margin_z']):.4f}",
        f"d2={float(patch_row['d_2d_mean']):.4f} d3={float(patch_row['d_3d_mean']):.4f}",
    ]
    _draw_text_lines(tile, lines, start_xy=(12, 215), line_height=18)
    return tile


def _compose_sheet(
    sample_name: str,
    true_part_label: str,
    overlay_rgb: np.ndarray,
    patch_tiles: list[np.ndarray],
    num_active_2d: int,
    num_active_3d: int,
    anomaly_threshold: float,
    k_neighbors: int,
    tau_s: float,
    tau_n: int,
    tau_p: float,
    part_predicted_label: str,
    component_decisions: list[object],
) -> np.ndarray:
    pad = 16
    overlay_panel_h = overlay_rgb.shape[0] + 120
    overlay_panel_w = overlay_rgb.shape[1] + 2 * pad
    overlay_panel = np.full((overlay_panel_h, overlay_panel_w, 3), 15, dtype=np.uint8)
    overlay_panel[90 : 90 + overlay_rgb.shape[0], pad : pad + overlay_rgb.shape[1]] = overlay_rgb
    _draw_text_lines(
        overlay_panel,
        [
            sample_name,
            (
                f"gt={true_part_label or '-'} pred={part_predicted_label} "
                f"active_2D={num_active_2d} active_3D={num_active_3d} comps={len(component_decisions)}"
            ),
            (
                f"k={k_neighbors} thr={anomaly_threshold:.4f} "
                f"tau_s={tau_s:.4f} tau_n={tau_n} tau_p={tau_p:.4f}"
            ),
            "rot=3D Evidenz, gruen=2D Evidenz, weiss=Komponente erfuellt 3D-Regel",
        ],
        start_xy=(pad, 24),
        line_height=20,
        font_scale=0.58,
    )

    if not patch_tiles:
        blank = np.full((260, 420, 3), 20, dtype=np.uint8)
        _draw_text_lines(blank, ["Keine aktiven Patches ueber dem Gating-Threshold."], start_xy=(20, 40), line_height=22)
        patch_tiles = [blank]

    tiles_per_row = max(1, min(4, int(np.ceil(np.sqrt(len(patch_tiles))))))
    tile_h = patch_tiles[0].shape[0]
    tile_w = patch_tiles[0].shape[1]
    rows = int(np.ceil(len(patch_tiles) / tiles_per_row))
    gallery_h = rows * tile_h + (rows + 1) * pad
    gallery_w = tiles_per_row * tile_w + (tiles_per_row + 1) * pad
    gallery = np.full((gallery_h, gallery_w, 3), 15, dtype=np.uint8)
    for idx, tile in enumerate(patch_tiles):
        r = idx // tiles_per_row
        c = idx % tiles_per_row
        y0 = pad + r * (tile_h + pad)
        x0 = pad + c * (tile_w + pad)
        gallery[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

    sheet_h = overlay_panel.shape[0] + pad + gallery.shape[0]
    sheet_w = max(overlay_panel.shape[1], gallery.shape[1])
    sheet = np.full((sheet_h, sheet_w, 3), 10, dtype=np.uint8)
    sheet[: overlay_panel.shape[0], : overlay_panel.shape[1]] = overlay_panel
    sheet[overlay_panel.shape[0] + pad : overlay_panel.shape[0] + pad + gallery.shape[0], : gallery.shape[1]] = gallery
    return sheet


def _category_folder(sample_name: str) -> Path:
    sample_path = Path(sample_name.replace("\\", "/"))
    try:
        relative = sample_path.relative_to(Path("test/bad"))
        parent = relative.parent
        return parent if str(parent) != "." else Path("_root")
    except ValueError:
        return sample_path.parent if str(sample_path.parent) != "." else Path("_root")


def main() -> int:
    args = parse_args()

    context, component_df, part_df, manual_patch_df, run_args, sample_inventory = load_session(args.session_dir)
    samples = load_run_samples(context.experiment_dir, seed=context.seed)
    sample_map = {sample.sample: sample for sample in samples}

    component_records = _component_label_records(component_df)
    manual_records = _manual_patch_label_records(manual_patch_df)
    bundle = build_memory_banks(samples, component_records, manual_records)
    bank_2d = bundle.features_2d.astype(np.float32, copy=False)
    bank_3d = bundle.features_3d.astype(np.float32, copy=False)

    anomaly_threshold = float(args.anomaly_threshold) if args.anomaly_threshold is not None else float(context.anomaly_threshold)
    if args.output_dir is None:
        output_dir = (
            context.session_dir
            / (
                f"patch_evidence_views_k{args.k_neighbors}"
                f"_ts{_slug_float(args.tau_s)}"
                f"_tn{args.tau_n}"
                f"_tp{_slug_float(args.tau_p)}"
            )
        )
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_samples = sample_inventory[sample_inventory["evaluation_group"] == args.evaluation_group].copy()
    candidate_samples.sort_values("sample", inplace=True)

    part_label_map = {
        str(row.sample).replace("\\", "/"): str(row.part_label)
        for row in part_df.itertuples(index=False)
    }

    patch_rows_all: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for sample_name in candidate_samples["sample"].tolist():
        sample = sample_map.get(sample_name)
        if sample is None:
            continue

        patch_features, grid_shape = load_patch_features(sample)
        anomaly_scores = np.load(sample.anomaly_map_path).astype(np.float32)
        if tuple(anomaly_scores.shape) != tuple(grid_shape):
            raise ValueError(
                f"Grid mismatch for {sample.sample!r}: feature cache {grid_shape}, anomaly grid {anomaly_scores.shape}."
            )

        patch_scores = compute_patch_class_scores(
            patch_features=patch_features,
            anomaly_scores=anomaly_scores,
            bank_2d=bank_2d,
            bank_3d=bank_3d,
            k_neighbors=args.k_neighbors,
            anomaly_threshold=anomaly_threshold,
        )
        active_mask = patch_scores["active_mask"]
        active_flat = active_mask.reshape(-1)
        active_features = patch_features[active_flat].astype(np.float32, copy=False)
        d_2d_nn1 = _nearest_cosine_distance(active_features, bank_2d) if active_flat.any() else np.array([], dtype=np.float32)
        d_3d_nn1 = _nearest_cosine_distance(active_features, bank_3d) if active_flat.any() else np.array([], dtype=np.float32)

        component_summaries = summarize_components(
            anomaly_scores=anomaly_scores,
            active_mask=active_mask,
            margin_c=patch_scores["margin_c"],
            weighted_margin_z=patch_scores["weighted_margin_z"],
        )
        component_decisions = classify_components(
            sample=sample.sample,
            component_summaries=component_summaries,
            tau_s=args.tau_s,
            tau_n=args.tau_n,
            tau_p=args.tau_p,
        )
        part_decision = classify_part(sample.sample, component_decisions)

        component_summaries_by_id = {
            int(summary["component"].component_id): summary for summary in component_summaries
        }
        component_decisions_by_id = {
            int(decision.component_id): decision for decision in component_decisions
        }
        component_id_grid = np.zeros(grid_shape, dtype=np.int32)
        component_pred_label_grid = np.zeros(grid_shape, dtype=np.int32)
        for summary in component_summaries:
            component = summary["component"]
            component_id_grid[component.rows, component.cols] = int(component.component_id)
        for decision in component_decisions:
            summary = component_summaries_by_id.get(int(decision.component_id))
            if summary is None:
                continue
            component = summary["component"]
            component_pred_label_grid[component.rows, component.cols] = 1 if decision.is_3d else -1

        active_indices = np.flatnonzero(active_flat)
        patch_rows: list[dict[str, object]] = []
        active_nn_idx = 0
        for patch_index in active_indices:
            row = int(patch_index // grid_shape[1])
            col = int(patch_index % grid_shape[1])
            d2 = float(patch_scores["d_2d"].reshape(-1)[patch_index])
            d3 = float(patch_scores["d_3d"].reshape(-1)[patch_index])
            c_val = float(patch_scores["margin_c"].reshape(-1)[patch_index])
            z_val = float(patch_scores["weighted_margin_z"].reshape(-1)[patch_index])
            evidence_label = "3D" if d3 < d2 else "2D"
            nearest_neighbor_label = "3D" if float(d_3d_nn1[active_nn_idx]) < float(d_2d_nn1[active_nn_idx]) else "2D"
            component_id = int(component_id_grid[row, col])
            component_predicted_label = (
                "3D" if int(component_pred_label_grid[row, col]) > 0 else "2D"
            ) if component_id > 0 else ""
            patch_row = {
                "sample": sample.sample,
                "evaluation_group": sample.evaluation_group,
                "true_part_label": part_label_map.get(sample.sample, ""),
                "predicted_part_label": part_decision.predicted_label,
                "image_score": float(sample.image_score),
                "image_threshold": float(sample.image_threshold),
                "component_id": component_id,
                "component_predicted_label": component_predicted_label,
                "row": row,
                "col": col,
                "patch_index": int(patch_index),
                "anomaly_score": float(anomaly_scores[row, col]),
                "d_2d_mean": d2,
                "d_3d_mean": d3,
                "margin_c": c_val,
                "weighted_margin_z": z_val,
                "evidence_label": evidence_label,
                "d_2d_nn1": float(d_2d_nn1[active_nn_idx]),
                "d_3d_nn1": float(d_3d_nn1[active_nn_idx]),
                "nearest_neighbor_label": nearest_neighbor_label,
            }
            patch_rows.append(patch_row)
            patch_rows_all.append(patch_row)
            active_nn_idx += 1

        patch_rows.sort(key=lambda row: (-float(row["anomaly_score"]), -float(row["weighted_margin_z"])))
        if args.max_patches_per_sheet > 0:
            patch_rows_render = patch_rows[: int(args.max_patches_per_sheet)]
        else:
            patch_rows_render = patch_rows

        display_image_rgb, display_grid_shape, patch_size = _load_display_image(sample)
        if tuple(display_grid_shape) != tuple(grid_shape):
            raise ValueError(
                f"Display grid mismatch for {sample.sample!r}: display {display_grid_shape}, feature grid {grid_shape}."
            )

        overlay_rgb = _render_patch_overlay(
            image_rgb=display_image_rgb,
            grid_shape=grid_shape,
            patch_rows=patch_rows_render,
            component_decisions_by_id=component_decisions_by_id,
            component_summaries_by_id=component_summaries_by_id,
        )
        patch_tiles = [
            _build_patch_tile(display_image_rgb, grid_shape, patch_row) for patch_row in patch_rows_render
        ]
        sheet = _compose_sheet(
            sample_name=sample.sample,
            true_part_label=part_label_map.get(sample.sample, ""),
            overlay_rgb=overlay_rgb,
            patch_tiles=patch_tiles,
            num_active_2d=sum(1 for row in patch_rows if str(row["evidence_label"]) == "2D"),
            num_active_3d=sum(1 for row in patch_rows if str(row["evidence_label"]) == "3D"),
            anomaly_threshold=anomaly_threshold,
            k_neighbors=args.k_neighbors,
            tau_s=args.tau_s,
            tau_n=args.tau_n,
            tau_p=args.tau_p,
            part_predicted_label=part_decision.predicted_label,
            component_decisions=component_decisions,
        )

        category_dir = output_dir / _category_folder(sample.sample)
        category_dir.mkdir(parents=True, exist_ok=True)
        sheet_path = category_dir / f"{Path(sample.sample).stem}__patch_evidence.png"
        overlay_path = category_dir / f"{Path(sample.sample).stem}__overlay.png"
        cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        sample_rows.append(
            {
                "sample": sample.sample,
                "evaluation_group": sample.evaluation_group,
                "true_part_label": part_label_map.get(sample.sample, ""),
                "predicted_part_label": part_decision.predicted_label,
                "image_score": float(sample.image_score),
                "num_active_patches": int(len(patch_rows)),
                "num_2d_evidence_patches": int(sum(1 for row in patch_rows if str(row["evidence_label"]) == "2D")),
                "num_3d_evidence_patches": int(sum(1 for row in patch_rows if str(row["evidence_label"]) == "3D")),
                "num_components": int(len(component_decisions)),
                "num_3d_components": int(sum(1 for decision in component_decisions if decision.is_3d)),
                "sheet_path": str(sheet_path),
                "overlay_path": str(overlay_path),
            }
        )

    write_csv(sample_rows, output_dir / "sample_summary.csv")
    write_csv(patch_rows_all, output_dir / "patch_evidence.csv")
    write_json(
        {
            "session_dir": str(context.session_dir),
            "experiment_dir": str(context.experiment_dir),
            "seed": int(context.seed),
            "evaluation_group": args.evaluation_group,
            "k_neighbors": int(args.k_neighbors),
            "anomaly_threshold": float(anomaly_threshold),
            "tau_s": float(args.tau_s),
            "tau_n": int(args.tau_n),
            "tau_p": float(args.tau_p),
            "num_samples": int(len(sample_rows)),
            "num_active_patches_total": int(len(patch_rows_all)),
            "num_2d_evidence_patches": int(sum(1 for row in patch_rows_all if str(row["evidence_label"]) == "2D")),
            "num_3d_evidence_patches": int(sum(1 for row in patch_rows_all if str(row["evidence_label"]) == "3D")),
            "output_dir": str(output_dir),
        },
        output_dir / "summary.json",
    )

    print(f"Patch-evidence render export written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
