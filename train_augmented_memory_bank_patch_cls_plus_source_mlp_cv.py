from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_FEATURE_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413\component_memory_bank_backend\session_full\memory_bank_export\aug64_t8_cls_plus_source_logreg"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small MLP on combined 64x64 CLS + source patch features "
            "with grouped 5-fold CV."
        )
    )
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 64])
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_csv(rows: List[Dict[str, object]], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    if not rows:
        output_file.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(device_name: str) -> str:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to cpu.")
        return "cpu"
    return device_name


def load_inputs(feature_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    features_file = feature_dir / "augmented_patch_cls_plus_source_features.npy"
    table_file = feature_dir / "augmented_patch_cls_plus_source_feature_table.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Missing features file: {features_file}")
    if not table_file.exists():
        raise FileNotFoundError(f"Missing feature table: {table_file}")

    features = np.load(features_file).astype(np.float32)
    table = pd.read_csv(table_file)
    if len(features) != len(table):
        raise ValueError(f"Length mismatch: {len(features)} features vs {len(table)} rows")
    return features, table


def metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))).tolist(),
        "per_class": {
            class_name: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, class_name in enumerate(class_names)
        },
    }


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.int64)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def evaluate_loader(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(batch_y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    x_val_outer: np.ndarray,
    device: str,
    hidden_dims: list[int],
    dropout: float,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    inner_train_idx, inner_val_idx = next(inner_splitter.split(x_train, y_train, groups_train))

    x_fit = x_train[inner_train_idx]
    y_fit = y_train[inner_train_idx]
    x_es = x_train[inner_val_idx]
    y_es = y_train[inner_val_idx]

    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit).astype(np.float32)
    x_es_scaled = scaler.transform(x_es).astype(np.float32)
    x_val_outer_scaled = scaler.transform(x_val_outer).astype(np.float32)

    train_loader, es_loader = make_loaders(x_fit_scaled, y_fit, x_es_scaled, y_es, batch_size)
    input_dim = int(x_fit_scaled.shape[1])
    model = SmallMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)

    class_counts = np.bincount(y_fit, minlength=2).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_val_f1 = -np.inf
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        es_logits, es_targets = evaluate_loader(model, es_loader, device)
        es_pred = es_logits.argmax(axis=1)
        es_metrics = metrics(es_targets, es_pred, ["2D", "3D"])
        es_f1 = float(es_metrics["macro_f1"])

        if es_f1 > best_val_f1:
            best_val_f1 = es_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("No best model state captured during training.")

    model.load_state_dict(best_state)
    outer_dataset = TensorDataset(torch.from_numpy(x_val_outer_scaled.astype(np.float32)))
    outer_loader = DataLoader(outer_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    all_logits: list[np.ndarray] = []
    with torch.inference_mode():
        for (batch_x,) in outer_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_logits.append(logits.cpu().numpy())
    val_logits = np.concatenate(all_logits, axis=0)
    val_proba = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy().astype(np.float32)
    val_pred = val_logits.argmax(axis=1).astype(np.int32)
    training_info = {
        "best_inner_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch,
        "class_weights": class_weights.tolist(),
    }
    return val_pred, val_proba, training_info


def default_output_dir(feature_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (feature_dir / "mlp_groupcv_results").resolve()


def main() -> int:
    args = parse_args()
    feature_dir = args.feature_dir.resolve()
    output_dir = default_output_dir(feature_dir, args.output_dir)
    ensure_dir(output_dir)

    features, table = load_inputs(feature_dir)
    y_labels = table["label"].astype(str).to_numpy()
    class_names = ["2D", "3D"]
    y = np.array([0 if label == "2D" else 1 for label in y_labels], dtype=np.int32)
    groups = table["group_id"].astype(str).to_numpy()
    device = resolve_device(args.device)

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    oof_pred = np.full(len(y), fill_value=-1, dtype=np.int32)
    oof_proba = np.zeros((len(y), len(class_names)), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(features, y, groups), start=1):
        val_pred, val_proba, training_info = train_one_fold(
            x_train=features[train_idx],
            y_train=y[train_idx],
            groups_train=groups[train_idx],
            x_val_outer=features[val_idx],
            device=device,
            hidden_dims=[int(v) for v in args.hidden_dims],
            dropout=float(args.dropout),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            patience=int(args.patience),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            random_state=int(args.random_state) + fold_idx,
        )

        oof_pred[val_idx] = val_pred
        oof_proba[val_idx] = val_proba
        fold_metrics = metrics(y[val_idx], val_pred, class_names)
        fold_rows.append(
            {
                "fold": fold_idx,
                "num_train_crops": int(len(train_idx)),
                "num_val_crops": int(len(val_idx)),
                "num_train_source_patches": int(pd.Series(groups[train_idx]).nunique()),
                "num_val_source_patches": int(pd.Series(groups[val_idx]).nunique()),
                "accuracy": fold_metrics["accuracy"],
                "macro_precision": fold_metrics["macro_precision"],
                "macro_recall": fold_metrics["macro_recall"],
                "macro_f1": fold_metrics["macro_f1"],
                "f1_2d": fold_metrics["per_class"]["2D"]["f1"],
                "f1_3d": fold_metrics["per_class"]["3D"]["f1"],
                "best_inner_val_macro_f1": float(training_info["best_inner_val_macro_f1"]),
                "best_epoch": int(training_info["best_epoch"]),
            }
        )

    if np.any(oof_pred < 0):
        raise RuntimeError("Some out-of-fold predictions were not filled.")

    overall = metrics(y, oof_pred, class_names)
    oof_rows: list[dict[str, object]] = []
    for idx, row in table.reset_index(drop=True).iterrows():
        item = row.to_dict()
        item["true_label"] = y_labels[idx]
        item["predicted_label"] = class_names[oof_pred[idx]]
        item["correct"] = int(y_labels[idx] == class_names[oof_pred[idx]])
        item["proba_2D"] = float(oof_proba[idx, 0])
        item["proba_3D"] = float(oof_proba[idx, 1])
        oof_rows.append(item)

    write_csv(fold_rows, output_dir / "fold_metrics.csv")
    write_csv(oof_rows, output_dir / "oof_predictions.csv")
    write_json(
        {
            "classifier": "mlp",
            "feature_type": "cls_plus_source_patch",
            "cv_type": "StratifiedGroupKFold",
            "grouping": "source_patch_uid",
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
            "device": device,
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "hidden_dims": [int(v) for v in args.hidden_dims],
            "num_augmented_crops_total": int(len(table)),
            "num_augmented_crops_2d": int((table["label"] == "2D").sum()),
            "num_augmented_crops_3d": int((table["label"] == "3D").sum()),
            "num_source_patches_total": int(table["group_id"].nunique()),
            "num_source_patches_2d": int(table.loc[table["label"] == "2D", "group_id"].nunique()),
            "num_source_patches_3d": int(table.loc[table["label"] == "3D", "group_id"].nunique()),
            "embedding_dim_total": int(features.shape[1]),
            "overall": overall,
            "folds": fold_rows,
        },
        output_dir / "summary.json",
    )

    print(f"MLP grouped CV complete. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
