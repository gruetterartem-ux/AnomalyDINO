from __future__ import annotations

import argparse
from pathlib import Path

from component_labeling_app.session_io import build_memory_bank_export, load_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 2D/3D memory banks from the current labeling session state."
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context, component_df, part_df, manual_patch_df, run_args, sample_inventory = load_session(args.session_dir)
    output_dir = build_memory_bank_export(
        context,
        component_annotations=component_df,
        manual_patch_annotations=manual_patch_df,
    )
    print(f"Exported session memory banks to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
