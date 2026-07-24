#!/usr/bin/env python
"""One-command raw MAT -> pulse MAT -> split -> scalogram dataset flow."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from data_pipeline.pulse_extraction import ExtractionConfig, extract_dataset
from data_pipeline.scalogram import convert_dataset
from data_pipeline.split_dataset import split_pulses
from data_pipeline.visualize import visualize_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a PECT-ready scalogram dataset from raw PD MAT signals. "
            "The output keeps intermediate pulses, diagnostics, and provenance."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw MAT root containing surface/, internal/, and corona/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New preparation output root (data is not copied into the repository).",
    )
    parser.add_argument(
        "--classes",
        default="surface,internal,corona",
        help="Comma-separated raw signal classes.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knee-sensitivity", type=float, default=0.4)
    parser.add_argument("--prominence-median-factor", type=float, default=2.8)
    parser.add_argument("--min-snr", type=float, default=7.8)
    parser.add_argument("--notpd-per-source-class", type=int, default=300)
    parser.add_argument(
        "--visualize-limit-per-class",
        type=int,
        default=3,
        help="Green-gradient raw/pulse examples per class or split/class; 0 disables.",
    )
    parser.add_argument(
        "--limit-files-per-class",
        type=int,
        default=None,
        help="Debug/smoke-test mode: process only the first N raw files per class.",
    )
    return parser.parse_args()


def count_by(rows: list[dict[str, object]], *keys: str) -> dict[str, int]:
    counts = Counter("/".join(str(row[key]) for key in keys) for row in rows)
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    classes = tuple(item.strip() for item in args.classes.split(",") if item.strip())
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not classes:
        raise ValueError("--classes cannot be empty")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output root is not empty: {output_root}. "
            "Use a new/empty folder to avoid mixing stale and new files."
        )
    workers = max(1, int(args.workers))

    flat_pulse_root = output_root / "pulses_flat"
    split_pulse_root = output_root / "pulses"
    scalogram_root = output_root / "scalograms"
    raw_viz_root = output_root / "visualizations" / "raw_signals"
    pulse_viz_root = output_root / "visualizations" / "pulses"
    diagnostics_root = output_root / "diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TIM_2026 DATA PREPARATION")
    print("=" * 78)
    print(f"Raw MAT input : {input_root}")
    print(f"Output root   : {output_root}")
    print(f"Classes       : {classes}")
    print(f"Workers       : {workers}")
    print("Flow          : raw MAT -> knee/SNR pulses -> group split -> CWT scalograms")

    raw_visualizations: list[dict[str, object]] = []
    if args.visualize_limit_per_class > 0:
        print("\n[1/5] Rendering raw signal examples (green gradient)...")
        raw_visualizations = visualize_tree(
            input_root,
            raw_viz_root,
            pulse=False,
            limit_per_class=args.visualize_limit_per_class,
        )
    else:
        print("\n[1/5] Raw signal visualization disabled.")

    print("\n[2/5] Extracting pulses with prominence + knee + SNR validation...")
    extraction_config = ExtractionConfig(
        sensitivity=args.knee_sensitivity,
        prominence_median_factor=args.prominence_median_factor,
        min_snr=args.min_snr,
        workers=workers,
        notpd_per_class=max(0, args.notpd_per_source_class),
    )
    pulse_manifest, extraction_diagnostics = extract_dataset(
        input_root,
        flat_pulse_root,
        diagnostics_root,
        classes=classes,
        config=extraction_config,
        limit_per_class=args.limit_files_per_class,
    )
    extraction_error_path = diagnostics_root / "extraction_errors.csv"
    if extraction_error_path.stat().st_size:
        with extraction_error_path.open(newline="", encoding="utf-8") as handle:
            extraction_error_count = sum(1 for _ in csv.DictReader(handle))
    else:
        extraction_error_count = 0
    if extraction_error_count:
        print(
            f"WARNING: {extraction_error_count} raw MAT files failed. "
            f"See {extraction_error_path}"
        )

    print("\n[3/5] Creating leakage-safe train/val/test splits...")
    split_manifest = split_pulses(
        flat_pulse_root,
        split_pulse_root,
        pulse_manifest,
        seed=args.seed,
    )

    pulse_visualizations: list[dict[str, object]] = []
    if args.visualize_limit_per_class > 0:
        print("\n[4/5] Rendering extracted pulse examples (green gradient)...")
        pulse_visualizations = visualize_tree(
            split_pulse_root,
            pulse_viz_root,
            pulse=True,
            limit_per_class=args.visualize_limit_per_class,
            show_peak_position=True,
        )
    else:
        print("\n[4/5] Pulse visualization disabled.")

    print("\n[5/5] Generating CWT scalograms...")
    scalogram_rows, _ = convert_dataset(
        split_pulse_root,
        scalogram_root,
        diagnostics_root,
        workers=workers,
    )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "configuration": {
            "classes": classes,
            "workers": workers,
            "seed": args.seed,
            "knee_sensitivity": args.knee_sensitivity,
            "prominence_median_factor": args.prominence_median_factor,
            "min_snr": args.min_snr,
            "notpd_per_source_class": args.notpd_per_source_class,
            "limit_files_per_class": args.limit_files_per_class,
        },
        "counts": {
            "raw_files_processed": len(extraction_diagnostics),
            "raw_files_failed": extraction_error_count,
            "pulses": len(pulse_manifest),
            "pulses_by_class": count_by(pulse_manifest, "class"),
            "split_pulses": len(split_manifest),
            "split_class_counts": count_by(split_manifest, "split", "class"),
            "scalograms": len(scalogram_rows),
            "raw_visualizations": len(raw_visualizations),
            "pulse_visualizations": len(pulse_visualizations),
        },
        "training_dataset_path": str(scalogram_root),
        "diagnostics": {
            "extraction": str(diagnostics_root / "extraction_diagnostics.csv"),
            "scalogram": str(diagnostics_root / "scalogram_diagnostics.csv"),
            "pulse_manifest": str(flat_pulse_root / "pulse_manifest.csv"),
            "split_manifest": str(split_pulse_root / "split_manifest.csv"),
        },
    }
    summary_path = diagnostics_root / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + "=" * 78)
    print("DATA PREPARATION COMPLETE")
    print("=" * 78)
    print(f"PECT dataset : {scalogram_root}")
    print(f"Pulse MAT    : {split_pulse_root}")
    print(f"Visuals     : {output_root / 'visualizations'}")
    print(f"Summary     : {summary_path}")
    print(f"Counts      : {summary['counts']['split_class_counts']}")


if __name__ == "__main__":
    main()
