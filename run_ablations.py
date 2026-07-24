"""Run the 14 PECT ablations from the original experiment suite."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from tim_2026.ablations import PECT_ABLATIONS


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _samples(value: str) -> list[int | None]:
    parsed: list[int | None] = []
    for token in _csv_list(value):
        parsed.append(None if token.lower() in {"all", "none"} else int(token))
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the clean PECT ablation suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-name", default="knee_aug_split")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ablations"))
    parser.add_argument("--variants", default="all", help="Comma-separated names, or all")
    parser.add_argument("--samples", default="60,160,240,all")
    parser.add_argument("--shots", default="1,5")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--final-test-seed", type=int, default=200042)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def _selected_variants(value: str) -> list[str]:
    available = {spec.name for spec in PECT_ABLATIONS}
    if value.strip().lower() == "all":
        return [spec.name for spec in PECT_ABLATIONS]
    selected = _csv_list(value)
    unknown = sorted(set(selected) - available)
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    return selected


def main() -> None:
    args = build_parser().parse_args()
    variants = _selected_variants(args.variants)
    samples = _samples(args.samples)
    shots = [int(value) for value in _csv_list(args.shots)]
    seeds = [int(value) for value in _csv_list(args.seeds)]
    script = Path(__file__).resolve().with_name("pect.py")

    commands: list[list[str]] = []
    for variant in variants:
        for sample_count in samples:
            for shot in shots:
                for seed in seeds:
                    sample_name = f"{sample_count}samples" if sample_count else "allsamples"
                    run_name = f"{variant}_{args.dataset_name}_{sample_name}_{shot}shot_seed{seed}"
                    metrics_path = args.output_dir / run_name / "metrics.csv"
                    if args.skip_existing and metrics_path.exists():
                        print(f"Skip existing: {metrics_path}")
                        continue
                    command = [
                        sys.executable,
                        str(script),
                        "--dataset-path",
                        str(args.dataset_path),
                        "--dataset-name",
                        args.dataset_name,
                        "--output-dir",
                        str(args.output_dir),
                        "--run-name",
                        run_name,
                        "--variant",
                        variant,
                        "--shot",
                        str(shot),
                        "--seed",
                        str(seed),
                        "--final-test-seed",
                        str(args.final_test_seed),
                        "--gpu",
                        str(args.gpu),
                        "--num-workers",
                        str(args.num_workers),
                    ]
                    if sample_count is not None:
                        command.extend(["--training-samples", str(sample_count)])
                    commands.append(command)

    print(
        f"Planned {len(commands)} run(s): {len(variants)} variant(s) x "
        f"{len(samples)} sample setting(s) x {len(shots)} shot setting(s) x "
        f"{len(seeds)} seed(s)"
    )
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}")
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
