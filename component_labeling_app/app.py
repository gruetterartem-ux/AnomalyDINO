from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from component_memory_bank.components import build_components
from component_labeling_app.rendering import (
    draw_patch_grid,
    prepare_display_image,
    render_manual_patch_overlay,
    render_binary_mask_overlay,
    render_components_overlay,
    render_heatmap_overlay,
    render_selected_component_overlay,
)
from component_labeling_app.session_io import (
    build_memory_bank_export,
    component_progress,
    load_session,
    memory_bank_patch_progress,
    manual_patch_progress,
    part_progress,
    save_component_annotations,
    save_manual_patch_annotations,
    save_part_labels,
)


COMPONENT_LABEL_OPTIONS = ["", "2D", "3D", "skip"]
PART_LABEL_OPTIONS = ["", "2D", "3D", "skip"]


def _parse_session_dir_from_argv() -> str:
    argv = sys.argv[1:]
    if "--session-dir" in argv:
        idx = argv.index("--session-dir")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return ""


@st.cache_data(show_spinner=False)
def _load_session_cached(session_dir_str: str):
    return load_session(Path(session_dir_str))


@st.cache_data(show_spinner=False)
def _prepare_sample_assets(
    image_path: str,
    anomaly_map_path: str,
    smaller_edge_size: int,
    patch_size: int,
    anomaly_threshold: float,
):
    image_rgb = prepare_display_image(Path(image_path), smaller_edge_size, patch_size)
    score_grid = np.load(anomaly_map_path).astype(np.float32)
    active_mask, components = build_components(score_grid, anomaly_threshold)
    return {
        "image_rgb": image_rgb,
        "score_grid": score_grid,
        "active_mask": active_mask,
        "components": components,
        "grid_shape": tuple(score_grid.shape),
    }


def _set_sample_index(sample_list: list[str], sample_value: str) -> None:
    try:
        st.session_state["sample_idx"] = sample_list.index(sample_value)
    except ValueError:
        st.session_state["sample_idx"] = 0
    st.session_state["component_id"] = None


def _goto_prev_sample(sample_list: list[str]) -> None:
    st.session_state["sample_idx"] = max(0, st.session_state.get("sample_idx", 0) - 1)
    st.session_state["component_id"] = None


def _goto_next_sample(sample_list: list[str]) -> None:
    st.session_state["sample_idx"] = min(len(sample_list) - 1, st.session_state.get("sample_idx", 0) + 1)
    st.session_state["component_id"] = None


def _goto_prev_component(component_ids: list[int]) -> None:
    current = st.session_state.get("component_id", component_ids[0])
    idx = max(0, component_ids.index(current) - 1)
    st.session_state["component_id"] = component_ids[idx]


def _goto_next_component(component_ids: list[int]) -> None:
    current = st.session_state.get("component_id", component_ids[0])
    idx = min(len(component_ids) - 1, component_ids.index(current) + 1)
    st.session_state["component_id"] = component_ids[idx]


def _record_history(action: dict[str, Any]) -> None:
    history = st.session_state.setdefault("history", [])
    history.append(action)
    if len(history) > 100:
        del history[0]


def _undo_last_action(context, component_df, part_df, manual_patch_df):
    history = st.session_state.get("history", [])
    if not history:
        return component_df, part_df, manual_patch_df

    action = history.pop()
    if action["type"] == "component":
        mask = (
            (component_df["sample"] == action["sample"])
            & (component_df["component_id"] == action["component_id"])
        )
        component_df.loc[mask, "label"] = action["previous_label"]
        component_df.loc[mask, "top_k"] = action["previous_top_k"]
        component_df.loc[mask, "notes"] = action["previous_notes"]
        save_component_annotations(context, component_df)
    elif action["type"] == "part":
        mask = part_df["sample"] == action["sample"]
        part_df.loc[mask, "part_label"] = action["previous_label"]
        part_df.loc[mask, "part_notes"] = action["previous_notes"]
        save_part_labels(context, part_df)
    elif action["type"] == "manual_patch":
        sample = action["sample"]
        row = action["row"]
        col = action["col"]
        patch_mask = (manual_patch_df["sample"] == sample) & (manual_patch_df["row"] == row) & (manual_patch_df["col"] == col)
        manual_patch_df = manual_patch_df.loc[~patch_mask].copy()
        previous_row = action.get("previous_row")
        if previous_row is not None:
            manual_patch_df = (
                pd.concat([manual_patch_df, pd.DataFrame([previous_row])], ignore_index=True)
                .sort_values(["sample", "row", "col"])
                .reset_index(drop=True)
            )
        save_manual_patch_annotations(context, manual_patch_df)
    elif action["type"] == "manual_patch_batch":
        keys = {(item["sample"], int(item["row"]), int(item["col"])) for item in action["items"]}
        manual_patch_df = _drop_manual_patch_keys(manual_patch_df, keys)
        previous_rows = action.get("previous_rows", [])
        if previous_rows:
            manual_patch_df = (
                pd.concat([manual_patch_df, pd.DataFrame(previous_rows)], ignore_index=True)
                .sort_values(["sample", "row", "col"])
                .reset_index(drop=True)
            )
        save_manual_patch_annotations(context, manual_patch_df)
    return component_df, part_df, manual_patch_df


def _pixel_to_patch(
    image_shape: tuple[int, int, int],
    grid_shape: tuple[int, int],
    click_payload: dict[str, Any],
) -> tuple[int, int] | None:
    if not click_payload:
        return None

    width = int(click_payload.get("width", image_shape[1]))
    height = int(click_payload.get("height", image_shape[0]))
    if width <= 0 or height <= 0:
        return None

    x = float(click_payload.get("x", 0.0)) * image_shape[1] / width
    y = float(click_payload.get("y", 0.0)) * image_shape[0] / height
    x = min(max(x, 0.0), image_shape[1] - 1)
    y = min(max(y, 0.0), image_shape[0] - 1)

    rows, cols = grid_shape
    row_edges = np.linspace(0, image_shape[0], rows + 1).round().astype(int)
    col_edges = np.linspace(0, image_shape[1], cols + 1).round().astype(int)
    row = int(np.searchsorted(row_edges, y, side="right") - 1)
    col = int(np.searchsorted(col_edges, x, side="right") - 1)
    row = min(max(row, 0), rows - 1)
    col = min(max(col, 0), cols - 1)
    return row, col


def _get_pending_patch_selection(current_sample: str) -> list[dict[str, Any]]:
    pending = st.session_state.get("pending_patch_selection", [])
    return [item for item in pending if item.get("sample") == current_sample]


def _set_pending_patch_selection(current_sample: str, items: list[dict[str, Any]]) -> None:
    pending = st.session_state.get("pending_patch_selection", [])
    remaining = [item for item in pending if item.get("sample") != current_sample]
    st.session_state["pending_patch_selection"] = remaining + items


def _drop_manual_patch_keys(
    manual_patch_df: pd.DataFrame,
    keys: set[tuple[str, int, int]],
) -> pd.DataFrame:
    if manual_patch_df.empty or not keys:
        return manual_patch_df.copy()

    key_series = list(
        zip(
            manual_patch_df["sample"].astype(str),
            manual_patch_df["row"].astype(int),
            manual_patch_df["col"].astype(int),
        )
    )
    keep_mask = [key not in keys for key in key_series]
    return manual_patch_df.loc[keep_mask].copy()


def main():
    st.set_page_config(page_title="Component Labeling", layout="wide")
    st.title("Component Memory-Bank Labeling")

    default_session_dir = _parse_session_dir_from_argv()
    session_dir_str = st.sidebar.text_input("Session Directory", value=default_session_dir)
    if not session_dir_str:
        st.info("Gib einen Session-Ordner an, der `component_inventory.csv` und `summary.json` enthält.")
        return

    try:
        context, component_df, part_df, manual_patch_df, run_args, sample_inventory = _load_session_cached(session_dir_str)
    except Exception as exc:
        st.error(f"Session konnte nicht geladen werden: {exc}")
        return

    if "component_df" not in st.session_state:
        st.session_state["component_df"] = component_df
    if "part_df" not in st.session_state:
        st.session_state["part_df"] = part_df
    if "manual_patch_df" not in st.session_state:
        st.session_state["manual_patch_df"] = manual_patch_df
    if "sample_inventory" not in st.session_state:
        st.session_state["sample_inventory"] = sample_inventory
    component_df = st.session_state["component_df"]
    part_df = st.session_state["part_df"]
    manual_patch_df = st.session_state["manual_patch_df"]
    sample_inventory = st.session_state["sample_inventory"]

    bad_only_inventory = sample_inventory[sample_inventory["evaluation_group"] == "test/bad"].copy()
    if not bad_only_inventory.empty:
        allowed_samples = set(bad_only_inventory["sample"].tolist())
        sample_inventory = bad_only_inventory
        component_df = component_df[component_df["sample"].isin(allowed_samples)].copy()
        part_df = part_df[part_df["sample"].isin(allowed_samples)].copy()
        manual_patch_df = manual_patch_df[manual_patch_df["sample"].isin(allowed_samples)].copy()
        st.sidebar.caption("Scope: nur `test/bad`-Bauteile")
    else:
        st.sidebar.caption("Scope: alle Samples")

    sample_list = sorted(sample_inventory["sample"].unique().tolist())
    if not sample_list:
        st.error("Keine Komponenten in der Session gefunden.")
        return

    if "sample_idx" not in st.session_state:
        st.session_state["sample_idx"] = 0
    current_sample = sample_list[st.session_state["sample_idx"]]

    st.sidebar.subheader("Fortschritt")
    st.sidebar.write(component_progress(component_df))
    st.sidebar.write(part_progress(part_df))
    st.sidebar.write({"manual_patches": manual_patch_progress(manual_patch_df)})
    st.sidebar.write({"memory_bank_patches": memory_bank_patch_progress(component_df, manual_patch_df)})
    st.sidebar.write(
        {
            "experiment_dir": str(context.experiment_dir),
            "seed": context.seed,
            "anomaly_threshold": context.anomaly_threshold,
            "resolution": run_args.get("resolution"),
            "model_name": run_args.get("model_name"),
        }
    )

    selected_sample = st.sidebar.selectbox(
        "Bauteil / Sample",
        options=sample_list,
        index=sample_list.index(current_sample),
    )
    if selected_sample != current_sample:
        _set_sample_index(sample_list, selected_sample)
        current_sample = selected_sample

    sample_rows = component_df[component_df["sample"] == current_sample].copy()
    if not sample_rows.empty:
        sample_rows = sample_rows.sort_values(
            by=["component_max_score", "component_mean_score", "component_size", "component_id"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        sample_rows["component_rank"] = np.arange(1, len(sample_rows) + 1)
    else:
        sample_rows["component_rank"] = []
    sample_meta = sample_inventory[sample_inventory["sample"] == current_sample].iloc[0]
    part_row = part_df[part_df["sample"] == current_sample].iloc[0]
    component_ids = sample_rows["component_id"].tolist()
    component_rank_map = dict(zip(sample_rows["component_id"], sample_rows["component_rank"]))
    rank_to_component_id = dict(zip(sample_rows["component_rank"], sample_rows["component_id"]))
    has_components = bool(sample_meta["has_components"])
    if has_components:
        if st.session_state.get("component_id") not in component_ids:
            st.session_state["component_id"] = component_ids[0]
        current_component_id = st.session_state["component_id"]
    else:
        st.session_state["component_id"] = None
        current_component_id = None

    nav1, nav2, nav3, nav4 = st.columns([1, 1, 2, 1])
    with nav1:
        if st.button("Prev Image"):
            _goto_prev_sample(sample_list)
            st.rerun()
    with nav2:
        if st.button("Next Image"):
            _goto_next_sample(sample_list)
            st.rerun()
    with nav3:
        if has_components:
            display_ranks = sample_rows["component_rank"].tolist()
            current_component_rank = component_rank_map[current_component_id]
            selected_component_rank = st.selectbox(
                "Komponente",
                options=display_ranks,
                index=display_ranks.index(current_component_rank),
            )
            selected_component_id = rank_to_component_id[selected_component_rank]
            if selected_component_id != current_component_id:
                st.session_state["component_id"] = selected_component_id
                current_component_id = selected_component_id
        else:
            st.caption("Keine Komponente in diesem Bauteil")
    with nav4:
        if st.button("Undo"):
            component_df, part_df, manual_patch_df = _undo_last_action(context, component_df, part_df, manual_patch_df)
            st.session_state["component_df"] = component_df
            st.session_state["part_df"] = part_df
            st.session_state["manual_patch_df"] = manual_patch_df
            st.rerun()

    navc1, navc2 = st.columns(2)
    with navc1:
        if st.button("Prev Component", disabled=not has_components):
            _goto_prev_component(component_ids)
            st.rerun()
    with navc2:
        if st.button("Next Component", disabled=not has_components):
            _goto_next_component(component_ids)
            st.rerun()

    sample_first = sample_meta
    try:
        with np.load(sample_first["feature_cache_path"]) as cache_data:
            patch_size = int(np.asarray(cache_data["patch_size"]).reshape(-1)[0])
        smaller_edge_size = int(run_args.get("resolution", patch_size * int(sample_first["grid_rows"])))
        assets = _prepare_sample_assets(
            image_path=sample_first["image_path"],
            anomaly_map_path=sample_first["anomaly_map_path"],
            smaller_edge_size=smaller_edge_size,
            patch_size=patch_size,
            anomaly_threshold=float(sample_first["anomaly_threshold"]),
        )
    except Exception as exc:
        st.error(f"Sample-Daten konnten nicht geladen werden: {exc}")
        return

    components_by_id = {component.component_id: component for component in assets["components"]}
    patch_to_component_id = {
        (int(row), int(col)): component.component_id
        for component in assets["components"]
        for row, col in zip(component.rows, component.cols)
    }
    if has_components and current_component_id not in components_by_id:
        st.error(
            f"Komponente {current_component_id} ist unter dem aktuellen Threshold nicht mehr vorhanden. "
            "Bitte Session neu erzeugen."
        )
        return

    current_component = components_by_id[current_component_id] if has_components else None
    current_row = sample_rows[sample_rows["component_id"] == current_component_id].iloc[0] if has_components else None
    current_component_rank = int(component_rank_map[current_component_id]) if has_components else None
    manual_sample_rows = (
        manual_patch_df[manual_patch_df["sample"] == current_sample]
        .sort_values(["row", "col"])
        .reset_index(drop=True)
    )

    header_left, header_right = st.columns([2, 1])
    with header_left:
        st.subheader(current_sample)
        st.write(
            {
                "evaluation_group": sample_first["evaluation_group"],
                "image_score": float(sample_first["image_score"]),
                "num_components": int(sample_first["num_components"]),
                "component_rank": current_component_rank,
                "component_size": int(current_row["component_size"]) if has_components else None,
                "component_max_score": float(current_row["component_max_score"]) if has_components else None,
                "component_mean_score": float(current_row["component_mean_score"]) if has_components else None,
            }
        )
    with header_right:
        st.subheader("Bauteillabel")
        part_label_value = st.selectbox(
            "Part Label",
            options=PART_LABEL_OPTIONS,
            index=PART_LABEL_OPTIONS.index(str(part_row["part_label"])),
            key=f"part_label_{current_sample}",
        )
        part_notes_value = st.text_input(
            "Part Notes",
            value=str(part_row["part_notes"]),
            key=f"part_notes_{current_sample}",
        )
        if st.button("Save Part Label"):
            mask = part_df["sample"] == current_sample
            previous_label = str(part_df.loc[mask, "part_label"].iloc[0])
            previous_notes = str(part_df.loc[mask, "part_notes"].iloc[0])
            _record_history(
                {
                    "type": "part",
                    "sample": current_sample,
                    "previous_label": previous_label,
                    "previous_notes": previous_notes,
                }
            )
            part_df.loc[mask, "part_label"] = part_label_value
            part_df.loc[mask, "part_notes"] = part_notes_value
            save_part_labels(context, part_df)
            st.session_state["part_df"] = part_df
            st.success("Bauteillabel gespeichert.")
            st.rerun()

    label_left, label_right = st.columns([2, 1])
    with label_left:
        st.subheader("Komponentenlabel")
        if has_components:
            current_label = str(current_row["label"])
            current_top_k = int(current_row["top_k"])
            current_notes = str(current_row["notes"])
            max_top_k = min(5, int(current_row["component_size"]))
            component_label_value = st.selectbox(
                "Component Label",
                options=COMPONENT_LABEL_OPTIONS,
                index=COMPONENT_LABEL_OPTIONS.index(current_label),
                key=f"component_label_{current_sample}_{current_component_id}",
            )
            if max_top_k <= 1:
                top_k_value = 1
                st.caption("Top-k Patches: 1 (Komponente besteht nur aus einem Patch)")
            else:
                top_k_value = st.slider(
                    "Top-k Patches",
                    min_value=1,
                    max_value=max_top_k,
                    value=min(current_top_k, max_top_k),
                    key=f"top_k_{current_sample}_{current_component_id}",
                )
            notes_value = st.text_input(
                "Component Notes",
                value=current_notes,
                key=f"component_notes_{current_sample}_{current_component_id}",
            )
            if st.button("Save Component Label"):
                mask = (
                    (component_df["sample"] == current_sample)
                    & (component_df["component_id"] == current_component_id)
                )
                previous = component_df.loc[mask, ["label", "top_k", "notes"]].iloc[0]
                _record_history(
                    {
                        "type": "component",
                        "sample": current_sample,
                        "component_id": current_component_id,
                        "previous_label": str(previous["label"]),
                        "previous_top_k": int(previous["top_k"]),
                        "previous_notes": str(previous["notes"]),
                    }
                )
                component_df.loc[mask, "label"] = component_label_value
                component_df.loc[mask, "top_k"] = int(top_k_value)
                component_df.loc[mask, "notes"] = notes_value
                save_component_annotations(context, component_df)
                st.session_state["component_df"] = component_df
                st.rerun()
        else:
            top_k_value = 1
            st.info("Dieses Bauteil hat unter dem aktuellen Threshold keine Komponente. Hier kannst du nur das Bauteillabel setzen.")
    with label_right:
        st.subheader("Export")
        if st.button("Export Memory Banks"):
            try:
                output_dir = build_memory_bank_export(
                    context,
                    component_annotations=component_df,
                    manual_patch_annotations=manual_patch_df,
                )
                st.success(f"Memory-Bänke exportiert nach: {output_dir}")
            except Exception as exc:
                st.error(f"Export fehlgeschlagen: {exc}")

    vis1, vis2 = st.columns(2)
    with vis1:
        st.image(draw_patch_grid(assets["image_rgb"], assets["grid_shape"]), caption="Original + Patchgitter", use_container_width=True)
        st.image(render_heatmap_overlay(assets["image_rgb"], assets["score_grid"]), caption="Anomaly Heatmap", use_container_width=True)
    with vis2:
        st.image(render_binary_mask_overlay(assets["image_rgb"], assets["active_mask"]), caption="Binäre Anomaliemaske", use_container_width=True)
        st.image(
            render_components_overlay(
                assets["image_rgb"],
                assets["grid_shape"],
                assets["components"],
                selected_component_id=current_component_id,
                component_labels={component_id: str(rank) for component_id, rank in component_rank_map.items()},
            ),
            caption="Komponenten (8er-Nachbarschaft)",
            use_container_width=True,
        )

    base_patch_view = (
        render_selected_component_overlay(
            assets["image_rgb"],
            assets["grid_shape"],
            current_component,
            assets["score_grid"],
            top_k=int(top_k_value),
        )
        if has_components
        else draw_patch_grid(assets["image_rgb"], assets["grid_shape"])
    )
    pending_patch_selection = _get_pending_patch_selection(current_sample)

    interactive_patch_view = render_manual_patch_overlay(
        base_patch_view,
        assets["grid_shape"],
        manual_patches=manual_sample_rows[["row", "col", "label"]].to_dict(orient="records"),
        selected_patches=[(int(item["row"]), int(item["col"])) for item in pending_patch_selection],
    )

    st.subheader("Klickbare Patch-Auswahl")
    click_payload = streamlit_image_coordinates(
        interactive_patch_view,
        key=f"patch_click_{current_sample}_{current_component_id if has_components else 'none'}",
        use_column_width="always",
        cursor="crosshair",
    )
    if click_payload:
        click_token = click_payload.get("unix_time")
        last_click_token_key = f"last_patch_click_{current_sample}"
        if click_token != st.session_state.get(last_click_token_key):
            patch_coords = _pixel_to_patch(interactive_patch_view.shape, assets["grid_shape"], click_payload)
            if patch_coords is not None:
                row, col = patch_coords
                component_id_for_patch = patch_to_component_id.get((row, col), "")
                patch_payload = {
                    "sample": current_sample,
                    "row": int(row),
                    "col": int(col),
                    "patch_index": int(row * assets["grid_shape"][1] + col),
                    "anomaly_score": float(assets["score_grid"][row, col]),
                    "component_id": str(component_id_for_patch) if component_id_for_patch != "" else "",
                    "component_rank": (
                        int(component_rank_map[component_id_for_patch])
                        if component_id_for_patch in component_rank_map
                        else ""
                    ),
                }
                existing_index = next(
                    (
                        idx
                        for idx, item in enumerate(pending_patch_selection)
                        if int(item["row"]) == int(row) and int(item["col"]) == int(col)
                    ),
                    None,
                )
                if existing_index is None:
                    pending_patch_selection = pending_patch_selection + [patch_payload]
                else:
                    pending_patch_selection = [
                        item for idx, item in enumerate(pending_patch_selection) if idx != existing_index
                    ]
                _set_pending_patch_selection(current_sample, pending_patch_selection)
            st.session_state[last_click_token_key] = click_token
            st.rerun()

    st.caption(
        (
            f"Aktuelle Komponente {current_component_rank} mit Top-{int(top_k_value)}-Patches. "
            "Gespeicherte manuelle Patches sind rot/grün markiert, aktuell ausgewählte Patches gelb."
        )
        if has_components
        else "Kein Komponenten-Overlay vorhanden. Du kannst trotzdem einzelne Patches manuell wählen."
    )

    manual_left, manual_right = st.columns([2, 1])
    with manual_left:
        st.subheader("Ausgewählte Patches")
        pending_patch_selection = _get_pending_patch_selection(current_sample)
        if pending_patch_selection:
            selected_df = pd.DataFrame(pending_patch_selection).sort_values(["row", "col"]).reset_index(drop=True)
            st.dataframe(
                selected_df[["row", "col", "patch_index", "anomaly_score", "component_rank", "component_id"]],
                use_container_width=True,
                hide_index=True,
            )

            btn1, btn2, btn3, btn4 = st.columns(4)
            if btn1.button("Save selected as 2D"):
                previous_rows = []
                new_rows = []
                for patch in pending_patch_selection:
                    row = int(patch["row"])
                    col = int(patch["col"])
                    existing_manual_row = manual_sample_rows[
                        (manual_sample_rows["row"] == row) & (manual_sample_rows["col"] == col)
                    ]
                    if not existing_manual_row.empty:
                        previous_rows.append(existing_manual_row.iloc[0].to_dict())
                    new_rows.append(
                        {
                            "object_name": sample_first["object_name"],
                            "sample": current_sample,
                            "evaluation_group": sample_first["evaluation_group"],
                            "image_path": sample_first["image_path"],
                            "anomaly_map_path": sample_first["anomaly_map_path"],
                            "feature_cache_path": sample_first["feature_cache_path"],
                            "grid_rows": int(sample_first["grid_rows"]),
                            "grid_cols": int(sample_first["grid_cols"]),
                            "component_id": patch["component_id"],
                            "row": row,
                            "col": col,
                            "patch_index": int(patch["patch_index"]),
                            "anomaly_score": float(patch["anomaly_score"]),
                            "label": "2D",
                        }
                    )
                _record_history(
                    {
                        "type": "manual_patch_batch",
                        "items": pending_patch_selection,
                        "previous_rows": previous_rows,
                    }
                )
                selected_keys = {(current_sample, int(p["row"]), int(p["col"])) for p in pending_patch_selection}
                manual_patch_df = _drop_manual_patch_keys(manual_patch_df, selected_keys)
                manual_patch_df = pd.concat([manual_patch_df, pd.DataFrame(new_rows)], ignore_index=True)
                save_manual_patch_annotations(context, manual_patch_df)
                st.session_state["manual_patch_df"] = manual_patch_df
                _set_pending_patch_selection(current_sample, [])
                st.rerun()
            if btn2.button("Save selected as 3D"):
                previous_rows = []
                new_rows = []
                for patch in pending_patch_selection:
                    row = int(patch["row"])
                    col = int(patch["col"])
                    existing_manual_row = manual_sample_rows[
                        (manual_sample_rows["row"] == row) & (manual_sample_rows["col"] == col)
                    ]
                    if not existing_manual_row.empty:
                        previous_rows.append(existing_manual_row.iloc[0].to_dict())
                    new_rows.append(
                        {
                            "object_name": sample_first["object_name"],
                            "sample": current_sample,
                            "evaluation_group": sample_first["evaluation_group"],
                            "image_path": sample_first["image_path"],
                            "anomaly_map_path": sample_first["anomaly_map_path"],
                            "feature_cache_path": sample_first["feature_cache_path"],
                            "grid_rows": int(sample_first["grid_rows"]),
                            "grid_cols": int(sample_first["grid_cols"]),
                            "component_id": patch["component_id"],
                            "row": row,
                            "col": col,
                            "patch_index": int(patch["patch_index"]),
                            "anomaly_score": float(patch["anomaly_score"]),
                            "label": "3D",
                        }
                    )
                _record_history(
                    {
                        "type": "manual_patch_batch",
                        "items": pending_patch_selection,
                        "previous_rows": previous_rows,
                    }
                )
                selected_keys = {(current_sample, int(p["row"]), int(p["col"])) for p in pending_patch_selection}
                manual_patch_df = _drop_manual_patch_keys(manual_patch_df, selected_keys)
                manual_patch_df = pd.concat([manual_patch_df, pd.DataFrame(new_rows)], ignore_index=True)
                save_manual_patch_annotations(context, manual_patch_df)
                st.session_state["manual_patch_df"] = manual_patch_df
                _set_pending_patch_selection(current_sample, [])
                st.rerun()
            if btn3.button("Remove saved labels for selected"):
                previous_rows = []
                for patch in pending_patch_selection:
                    row = int(patch["row"])
                    col = int(patch["col"])
                    existing_manual_row = manual_sample_rows[
                        (manual_sample_rows["row"] == row) & (manual_sample_rows["col"] == col)
                    ]
                    if not existing_manual_row.empty:
                        previous_rows.append(existing_manual_row.iloc[0].to_dict())
                if previous_rows:
                    _record_history(
                        {
                            "type": "manual_patch_batch",
                            "items": pending_patch_selection,
                            "previous_rows": previous_rows,
                        }
                    )
                    selected_keys = {(current_sample, int(p["row"]), int(p["col"])) for p in pending_patch_selection}
                    manual_patch_df = _drop_manual_patch_keys(manual_patch_df, selected_keys)
                    save_manual_patch_annotations(context, manual_patch_df)
                    st.session_state["manual_patch_df"] = manual_patch_df
                _set_pending_patch_selection(current_sample, [])
                st.rerun()
            if btn4.button("Clear selection"):
                _set_pending_patch_selection(current_sample, [])
                st.rerun()
        else:
            st.info("Klicke im großen Patchgitter auf einen oder mehrere Patches. Ein weiterer Klick auf denselben Patch wählt ihn wieder ab.")
    with manual_right:
        st.subheader("Manuelle Patches im Bauteil")
        if manual_sample_rows.empty:
            st.caption("Noch keine manuellen Patches gespeichert.")
        else:
            st.dataframe(
                manual_sample_rows[["row", "col", "label", "anomaly_score", "component_id"]],
                use_container_width=True,
                hide_index=True,
            )

    if has_components:
        preview_cols = [
            "component_rank",
            "label",
            "top_k",
            "component_size",
            "component_max_score",
            "component_mean_score",
            "notes",
        ]
        st.subheader("Komponenten des aktuellen Bauteils")
        st.dataframe(sample_rows[preview_cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
