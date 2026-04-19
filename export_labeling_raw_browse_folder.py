from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a flat browse folder with raw input1 images in the same order "
            "as the component labeling app."
        )
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        action="append",
        dest="source_roots",
        required=True,
        help="Source root that contains per-part subfolders.",
    )
    parser.add_argument(
        "--group",
        default="test/bad",
        help="Evaluation group to export. Default: test/bad",
    )
    return parser.parse_args()


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _select_samples(sample_rows: list[dict[str, str]], group: str) -> list[dict[str, str]]:
    filtered = [row for row in sample_rows if row.get("evaluation_group") == group]
    return sorted(filtered, key=lambda row: row["sample"])


def _find_input1(part_name: str, source_roots: list[Path]) -> tuple[Path, Path] | None:
    for root in source_roots:
        candidate_dir = root / part_name
        if not candidate_dir.is_dir():
            continue

        direct = candidate_dir / "input1.bmp"
        if direct.is_file():
            return root, direct

        matches = sorted(candidate_dir.glob("input1.*"))
        if matches:
            return root, matches[0]
    return None


def main() -> int:
    args = parse_args()
    session_dir = args.session_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_roots = [Path(path).resolve() for path in args.source_roots]

    sample_inventory_path = session_dir / "sample_inventory.csv"
    if not sample_inventory_path.is_file():
        raise FileNotFoundError(f"Missing sample inventory: {sample_inventory_path}")

    sample_rows = _read_rows(sample_inventory_path)
    selected_rows = _select_samples(sample_rows, args.group)
    if not selected_rows:
        raise RuntimeError(f"No samples found for evaluation_group={args.group!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    pad = len(str(len(selected_rows)))
    copied = 0
    missing = 0
    manifest_rows: list[dict[str, str]] = []

    for index, row in enumerate(selected_rows, start=1):
        sample = row["sample"]
        image_filename = Path(sample).name
        part_name = Path(image_filename).stem
        match = _find_input1(part_name, source_roots)

        output_name = f"{index:0{pad}d}__{part_name}__input1.bmp"
        output_path = output_dir / output_name

        if match is None:
            manifest_rows.append(
                {
                    "order_index": str(index),
                    "sample": sample,
                    "evaluation_group": row.get("evaluation_group", ""),
                    "part_name": part_name,
                    "status": "missing",
                    "source_root": "",
                    "source_path": "",
                    "output_path": str(output_path),
                }
            )
            missing += 1
            continue

        source_root, source_path = match
        shutil.copy2(source_path, output_path)
        copied += 1
        manifest_rows.append(
            {
                "order_index": str(index),
                "sample": sample,
                "evaluation_group": row.get("evaluation_group", ""),
                "part_name": part_name,
                "status": "copied",
                "source_root": str(source_root),
                "source_path": str(source_path),
                "output_path": str(output_path),
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "order_index",
                "sample",
                "evaluation_group",
                "part_name",
                "status",
                "source_root",
                "source_path",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"session_dir={session_dir}",
                f"group={args.group}",
                f"total_selected={len(selected_rows)}",
                f"copied={copied}",
                f"missing={missing}",
                f"manifest={manifest_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Created browse folder: {output_dir}")
    print(f"Selected samples: {len(selected_rows)}")
    print(f"Copied: {copied}")
    print(f"Missing: {missing}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
