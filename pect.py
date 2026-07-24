"""Canonical PECT train/test entry point."""

from __future__ import annotations

from dataclasses import asdict

from tim_2026.cli import build_parser, config_from_args
from tim_2026.engine import run_experiment


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    config.validate()
    if args.dry_run:
        for section, values in asdict(config).items():
            if isinstance(values, dict):
                print(f"[{section}]")
                for key, value in values.items():
                    print(f"{key}={value}")
            else:
                print(f"{section}={values}")
        return
    run_experiment(config)


if __name__ == "__main__":
    main()
