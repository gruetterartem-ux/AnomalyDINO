from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .components import build_components, build_components_from_mask


@dataclass(frozen=True)
class ComponentDecision:
    sample: str
    component_id: int
    score_s: float
    size_n: int
    peak_p: float
    max_margin_c: float
    is_3d: bool


@dataclass(frozen=True)
class PartDecision:
    sample: str
    predicted_label: str
    has_3d_component: bool
    num_components: int
    num_3d_components: int


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def _mean_cosine_knn_distance(query_features: np.ndarray, bank_features: np.ndarray, k_neighbors: int) -> np.ndarray:
    if bank_features.ndim != 2 or bank_features.shape[0] == 0:
        raise ValueError("bank_features must be a non-empty 2D array.")
    if query_features.ndim != 2 or query_features.shape[0] == 0:
        raise ValueError("query_features must be a non-empty 2D array.")

    query_norm = _normalize_rows(query_features)
    bank_norm = _normalize_rows(bank_features)
    cosine_sim = query_norm @ bank_norm.T
    k_eff = min(int(k_neighbors), bank_norm.shape[0])
    if k_eff < 1:
        raise ValueError("k_neighbors must be at least 1.")

    topk_sim = np.partition(cosine_sim, kth=bank_norm.shape[0] - k_eff, axis=1)[:, -k_eff:]
    topk_dist = 1.0 - topk_sim
    return topk_dist.mean(axis=1)


def compute_patch_class_scores(
    patch_features: np.ndarray,
    anomaly_scores: np.ndarray,
    bank_2d: np.ndarray,
    bank_3d: np.ndarray,
    k_neighbors: int,
    anomaly_threshold: float,
) -> Dict[str, np.ndarray]:
    grid_shape = tuple(anomaly_scores.shape)
    flattened_scores = anomaly_scores.reshape(-1)
    active_flat = flattened_scores > float(anomaly_threshold)

    d_2d = np.full(flattened_scores.shape, np.nan, dtype=np.float32)
    d_3d = np.full(flattened_scores.shape, np.nan, dtype=np.float32)
    margin_c = np.zeros(flattened_scores.shape, dtype=np.float32)
    weighted_margin_z = np.zeros(flattened_scores.shape, dtype=np.float32)

    if active_flat.any():
        active_features = patch_features[active_flat].astype(np.float32, copy=False)
        d_2d_active = _mean_cosine_knn_distance(active_features, bank_2d, k_neighbors).astype(np.float32)
        d_3d_active = _mean_cosine_knn_distance(active_features, bank_3d, k_neighbors).astype(np.float32)
        denom = np.maximum(d_2d_active + d_3d_active, 1e-12)
        margin_active = (d_2d_active - d_3d_active) / denom
        weighted_active = flattened_scores[active_flat] * margin_active

        d_2d[active_flat] = d_2d_active
        d_3d[active_flat] = d_3d_active
        margin_c[active_flat] = margin_active
        weighted_margin_z[active_flat] = weighted_active

    return {
        "active_mask": active_flat.reshape(grid_shape),
        "d_2d": d_2d.reshape(grid_shape),
        "d_3d": d_3d.reshape(grid_shape),
        "margin_c": margin_c.reshape(grid_shape),
        "weighted_margin_z": weighted_margin_z.reshape(grid_shape),
    }


def summarize_components(
    anomaly_scores: np.ndarray,
    active_mask: np.ndarray,
    margin_c: np.ndarray,
    weighted_margin_z: np.ndarray,
) -> List[dict]:
    components = build_components_from_mask(active_mask, anomaly_scores)
    summaries = []
    for component in components:
        weights = anomaly_scores[component.rows, component.cols].astype(np.float32)
        component_margins = margin_c[component.rows, component.cols].astype(np.float32)
        component_weighted = weighted_margin_z[component.rows, component.cols].astype(np.float32)
        weight_sum = float(weights.sum())
        score_s = float(np.sum(weights * component_margins) / max(weight_sum, 1e-12))
        peak_p = float(component_weighted.max())
        max_margin_c = float(component_margins.max())
        summaries.append(
            {
                "component": component,
                "score_s": score_s,
                "size_n": int(component.size),
                "peak_p": peak_p,
                "max_margin_c": max_margin_c,
            }
        )
    return summaries


def classify_components(
    sample: str,
    component_summaries: Iterable[dict],
    tau_s: float,
    tau_n: int,
    tau_p: float,
) -> List[ComponentDecision]:
    decisions: List[ComponentDecision] = []
    for summary in component_summaries:
        component = summary["component"]
        is_3d = (
            (summary["score_s"] > float(tau_s) and summary["size_n"] >= int(tau_n))
            or summary["peak_p"] > float(tau_p)
        )
        decisions.append(
            ComponentDecision(
                sample=sample,
                component_id=component.component_id,
                score_s=float(summary["score_s"]),
                size_n=int(summary["size_n"]),
                peak_p=float(summary["peak_p"]),
                max_margin_c=float(summary["max_margin_c"]),
                is_3d=bool(is_3d),
            )
        )
    return decisions


def classify_part(sample: str, component_decisions: Iterable[ComponentDecision]) -> PartDecision:
    component_list = list(component_decisions)
    num_3d = sum(1 for decision in component_list if decision.is_3d)
    has_3d = num_3d > 0
    return PartDecision(
        sample=sample,
        predicted_label="3D" if has_3d else "2D",
        has_3d_component=has_3d,
        num_components=len(component_list),
        num_3d_components=num_3d,
    )
