from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from component_memory_bank.export import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether 2D and 3D memory-bank features are separable in cosine-distance space "
            "and via leave-one-out kNN."
        )
    )
    parser.add_argument("--memory-bank-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    return parser.parse_args()


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def _pairwise_self_distances(features_norm: np.ndarray) -> np.ndarray:
    sim = features_norm @ features_norm.T
    iu = np.triu_indices(sim.shape[0], k=1)
    return (1.0 - sim[iu]).astype(np.float32)


def _pairwise_cross_distances(features_a_norm: np.ndarray, features_b_norm: np.ndarray) -> np.ndarray:
    sim = features_a_norm @ features_b_norm.T
    return (1.0 - sim).reshape(-1).astype(np.float32)


def _describe(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def _common_language_effect(smaller_should_be_better: np.ndarray, comparison: np.ndarray) -> float:
    # Probability that a random value from the first distribution is smaller than a random value from the second.
    u_stat = mannwhitneyu(smaller_should_be_better, comparison, alternative="less").statistic
    denom = float(len(smaller_should_be_better) * len(comparison))
    return float(u_stat / denom)


def _loo_classify_mean_knn(
    features_2d_norm: np.ndarray,
    features_3d_norm: np.ndarray,
    k_neighbors: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    labels_true: list[str] = []
    labels_pred: list[str] = []
    rows: list[dict[str, object]] = []

    sim_22 = features_2d_norm @ features_2d_norm.T
    sim_33 = features_3d_norm @ features_3d_norm.T
    sim_23 = features_2d_norm @ features_3d_norm.T
    sim_32 = sim_23.T

    for idx in range(features_2d_norm.shape[0]):
        same = sim_22[idx].copy()
        same[idx] = -np.inf
        k_same = min(k_neighbors, same.size - 1)
        k_other = min(k_neighbors, sim_23.shape[1])
        same_top = np.partition(same, kth=same.size - k_same)[-k_same:]
        other_top = np.partition(sim_23[idx], kth=sim_23.shape[1] - k_other)[-k_other:]
        d_same = float((1.0 - same_top).mean())
        d_other = float((1.0 - other_top).mean())
        pred = "3D" if d_other < d_same else "2D"
        labels_true.append("2D")
        labels_pred.append(pred)
        rows.append(
            {
                "index_in_class": idx,
                "true_label": "2D",
                "predicted_label": pred,
                "correct": int(pred == "2D"),
                "mean_same_class_distance": d_same,
                "mean_other_class_distance": d_other,
                "distance_margin_same_minus_other": d_same - d_other,
            }
        )

    for idx in range(features_3d_norm.shape[0]):
        same = sim_33[idx].copy()
        same[idx] = -np.inf
        k_same = min(k_neighbors, same.size - 1)
        k_other = min(k_neighbors, sim_32.shape[1])
        same_top = np.partition(same, kth=same.size - k_same)[-k_same:]
        other_top = np.partition(sim_32[idx], kth=sim_32.shape[1] - k_other)[-k_other:]
        d_same = float((1.0 - same_top).mean())
        d_other = float((1.0 - other_top).mean())
        pred = "2D" if d_other < d_same else "3D"
        labels_true.append("3D")
        labels_pred.append(pred)
        rows.append(
            {
                "index_in_class": idx,
                "true_label": "3D",
                "predicted_label": pred,
                "correct": int(pred == "3D"),
                "mean_same_class_distance": d_same,
                "mean_other_class_distance": d_other,
                "distance_margin_same_minus_other": d_same - d_other,
            }
        )

    labels = ["2D", "3D"]
    cm = confusion_matrix(labels_true, labels_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_true, labels_pred, labels=labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels_true, labels_pred, labels=labels, average="macro", zero_division=0
    )
    summary = {
        "k_neighbors": int(k_neighbors),
        "accuracy": float(accuracy_score(labels_true, labels_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "precision_2d": float(precision[0]),
        "recall_2d": float(recall[0]),
        "f1_2d": float(f1[0]),
        "support_2d": int(support[0]),
        "precision_3d": float(precision[1]),
        "recall_3d": float(recall[1]),
        "f1_3d": float(f1[1]),
        "support_3d": int(support[1]),
    }
    return summary, pd.DataFrame(rows)


def _nearest_distance_analysis(features_2d_norm: np.ndarray, features_3d_norm: np.ndarray) -> dict[str, float]:
    sim_22 = features_2d_norm @ features_2d_norm.T
    sim_33 = features_3d_norm @ features_3d_norm.T
    sim_23 = features_2d_norm @ features_3d_norm.T
    sim_32 = sim_23.T

    sim_22 = sim_22.copy()
    np.fill_diagonal(sim_22, -np.inf)
    sim_33 = sim_33.copy()
    np.fill_diagonal(sim_33, -np.inf)

    nearest_same_2d = 1.0 - sim_22.max(axis=1)
    nearest_other_2d = 1.0 - sim_23.max(axis=1)
    nearest_same_3d = 1.0 - sim_33.max(axis=1)
    nearest_other_3d = 1.0 - sim_32.max(axis=1)

    return {
        **_describe(nearest_same_2d.astype(np.float32), "nearest_same_2d"),
        **_describe(nearest_other_2d.astype(np.float32), "nearest_other_2d"),
        **_describe(nearest_same_3d.astype(np.float32), "nearest_same_3d"),
        **_describe(nearest_other_3d.astype(np.float32), "nearest_other_3d"),
        "frac_2d_nearest_same_closer_than_other": float((nearest_same_2d < nearest_other_2d).mean()),
        "frac_3d_nearest_same_closer_than_other": float((nearest_same_3d < nearest_other_3d).mean()),
    }


def _plot_pairwise_histogram(
    d_22: np.ndarray,
    d_33: np.ndarray,
    d_23: np.ndarray,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = 60
    ax.hist(d_22, bins=bins, alpha=0.55, density=True, color="#2E8B57", label="2D-2D")
    ax.hist(d_33, bins=bins, alpha=0.55, density=True, color="#C0392B", label="3D-3D")
    ax.hist(d_23, bins=bins, alpha=0.45, density=True, color="#4169E1", label="2D-3D")
    ax.set_title("Cosine-Distanz: innerhalb vs. zwischen Klassen")
    ax.set_xlabel("Cosine-Distanz")
    ax.set_ylabel("Dichte")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def _plot_loo_bars(loo_df: pd.DataFrame, output_file: Path) -> None:
    grouped = loo_df.groupby("true_label")["correct"].mean().reindex(["2D", "3D"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["2D", "3D"], grouped.values, color=["#2E8B57", "#C0392B"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Leave-one-out Trefferquote pro Klasse")
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y", alpha=0.18)
    for idx, value in enumerate(grouped.values):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    memory_bank_dir = args.memory_bank_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (memory_bank_dir / "separability_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    features_2d = np.load(memory_bank_dir / "2D-memory-bank.npy").astype(np.float32)
    features_3d = np.load(memory_bank_dir / "3D-memory-bank.npy").astype(np.float32)
    meta_file = memory_bank_dir / "selected_patches.csv"
    meta_df = pd.read_csv(meta_file) if meta_file.exists() else pd.DataFrame()

    features_2d_norm = _normalize_rows(features_2d)
    features_3d_norm = _normalize_rows(features_3d)

    d_22 = _pairwise_self_distances(features_2d_norm)
    d_33 = _pairwise_self_distances(features_3d_norm)
    d_23 = _pairwise_cross_distances(features_2d_norm, features_3d_norm)

    pairwise_summary = {
        **_describe(d_22, "dist_2d_to_2d"),
        **_describe(d_33, "dist_3d_to_3d"),
        **_describe(d_23, "dist_2d_to_3d"),
        "mean_2d_to_2d_smaller_than_2d_to_3d": bool(float(d_22.mean()) < float(d_23.mean())),
        "mean_3d_to_3d_smaller_than_3d_to_2d": bool(float(d_33.mean()) < float(d_23.mean())),
        "mannwhitney_2d_within_vs_between_p": float(mannwhitneyu(d_22, d_23, alternative="less").pvalue),
        "mannwhitney_3d_within_vs_between_p": float(mannwhitneyu(d_33, d_23, alternative="less").pvalue),
        "prob_2d_within_smaller_than_between": _common_language_effect(d_22, d_23),
        "prob_3d_within_smaller_than_between": _common_language_effect(d_33, d_23),
    }
    nearest_summary = _nearest_distance_analysis(features_2d_norm, features_3d_norm)

    loo_summaries: list[dict[str, object]] = []
    loo_detail_frames: list[pd.DataFrame] = []
    for k_value in sorted({int(k) for k in args.k_values}):
        if k_value >= features_2d.shape[0] or k_value >= features_3d.shape[0]:
            continue
        summary, loo_df = _loo_classify_mean_knn(features_2d_norm, features_3d_norm, k_value)
        loo_summaries.append(summary)
        loo_df["k_neighbors"] = int(k_value)
        loo_detail_frames.append(loo_df)

    loo_summary_df = pd.DataFrame(loo_summaries).sort_values("macro_f1", ascending=False)
    loo_detail_df = pd.concat(loo_detail_frames, axis=0, ignore_index=True)

    if not meta_df.empty:
        meta_2d = meta_df[meta_df["component_label"].astype(str) == "2D"].copy().reset_index(drop=True)
        meta_3d = meta_df[meta_df["component_label"].astype(str) == "3D"].copy().reset_index(drop=True)
        # Map leave-one-out details for the best k back to patch metadata.
        if not loo_summary_df.empty:
            best_k = int(loo_summary_df.iloc[0]["k_neighbors"])
            best_loo = loo_detail_df[loo_detail_df["k_neighbors"] == best_k].copy()
            best_loo_2d = best_loo[best_loo["true_label"] == "2D"].copy().reset_index(drop=True)
            best_loo_3d = best_loo[best_loo["true_label"] == "3D"].copy().reset_index(drop=True)
            detailed_best = pd.concat(
                [
                    pd.concat([meta_2d, best_loo_2d.drop(columns=["true_label"])], axis=1),
                    pd.concat([meta_3d, best_loo_3d.drop(columns=["true_label"])], axis=1),
                ],
                axis=0,
                ignore_index=True,
            )
            detailed_best.to_csv(output_dir / "leave_one_out_best_k_details.csv", index=False)

    pd.DataFrame([pairwise_summary]).to_csv(output_dir / "pairwise_distance_summary.csv", index=False)
    pd.DataFrame([nearest_summary]).to_csv(output_dir / "nearest_neighbor_distance_summary.csv", index=False)
    loo_summary_df.to_csv(output_dir / "leave_one_out_knn_summary.csv", index=False)
    loo_detail_df.to_csv(output_dir / "leave_one_out_knn_details.csv", index=False)

    _plot_pairwise_histogram(d_22, d_33, d_23, output_dir / "pairwise_distance_histogram.png")
    if not loo_summary_df.empty:
        best_k = int(loo_summary_df.iloc[0]["k_neighbors"])
        best_loo_df = loo_detail_df[loo_detail_df["k_neighbors"] == best_k].copy()
        _plot_loo_bars(best_loo_df, output_dir / "leave_one_out_best_k_accuracy.png")
    else:
        best_k = None

    summary = {
        "memory_bank_dir": str(memory_bank_dir),
        "output_dir": str(output_dir),
        "num_2d_features": int(features_2d.shape[0]),
        "num_3d_features": int(features_3d.shape[0]),
        "feature_dim": int(features_2d.shape[1]),
        "pairwise_distance_test": pairwise_summary,
        "nearest_neighbor_test": nearest_summary,
        "best_leave_one_out_k": int(best_k) if best_k is not None else None,
        "best_leave_one_out_result": loo_summary_df.iloc[0].to_dict() if not loo_summary_df.empty else {},
    }
    write_json(summary, output_dir / "summary.json")

    print(f"Separability analysis written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
