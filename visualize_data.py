#!/usr/bin/env python
"""Render raw signals or pulse MAT files with the green-gradient style."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.visualize import visualize_tree


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize raw signals or extracted pulses as green gradients."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("raw", "pulse"), required=True)
    parser.add_argument("--limit-per-class", type=int, default=3)
    parser.add_argument("--show-peak-position", action="store_true")
    args = parser.parse_args()
    records = visualize_tree(
        args.input.resolve(),
        args.output.resolve(),
        pulse=args.kind == "pulse",
        limit_per_class=args.limit_per_class,
        show_peak_position=args.show_peak_position,
    )
    print(f"Rendered {len(records)} files to {args.output.resolve()}")
    print(f"Mapping: {args.output.resolve() / 'file_mapping.csv'}")


if __name__ == "__main__":
    main()
