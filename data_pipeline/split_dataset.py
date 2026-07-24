"""Leakage-safe deterministic train/validation/test split for extracted pulses."""

from __future__ import annotations

import csv
import shutil
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np


SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _allocate_groups(n_groups: int) -> dict[str, int]:
    raw = {name: ratio * n_groups for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remaining = n_groups - sum(counts.values())
    order = sorted(
        SPLIT_RATIOS,
        key=lambda name: (raw[name] - counts[name], SPLIT_RATIOS[name]),
        reverse=True,
    )
    for name in order[:remaining]:
        counts[name] += 1
    if n_groups >= 3:
        for name in ("val", "test"):
            if counts[name] == 0:
                donor = max(counts, key=counts.get)
                counts[donor] -= 1
                counts[name] += 1
    return counts


def split_pulses(
    flat_root: Path,
    output_root: Path,
    manifest: list[dict[str, object]],
    *,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Copy pulses into splits while keeping each raw source file in one split."""
    by_source_class: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in manifest:
        group = (str(row["source_class"]), str(row["source_file"]))
        by_source_class[group[0]].add(group)

    assignment: dict[tuple[str, str], str] = {}
    for class_index, source_class in enumerate(sorted(by_source_class)):
        groups = sorted(by_source_class[source_class])
        rng = np.random.default_rng(seed + class_index)
        rng.shuffle(groups)
        counts = _allocate_groups(len(groups))
        cursor = 0
        for split_name in ("train", "val", "test"):
            selected = groups[cursor : cursor + counts[split_name]]
            assignment.update({group: split_name for group in selected})
            cursor += counts[split_name]

    output_rows: list[dict[str, object]] = []
    for row in manifest:
        class_name = str(row["class"])
        group = (str(row["source_class"]), str(row["source_file"]))
        split_name = assignment[group]
        source = flat_root / str(row["pulse_path"])
        destination = output_root / split_name / class_name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        output_rows.append(
            {
                **row,
                "split": split_name,
                "split_group": f"{group[0]}/{group[1]}",
                "split_seed": seed,
                "split_path": str(Path(split_name) / class_name / source.name),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "split_manifest.csv"
    if output_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    else:
        manifest_path.write_text("", encoding="utf-8")

    leakage_check: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in output_rows:
        leakage_check[
            (str(row["source_class"]), str(row["source_file"]))
        ].add(str(row["split"]))
    leaked = [group for group, splits in leakage_check.items() if len(splits) > 1]
    if leaked:
        raise RuntimeError(f"Source-group leakage detected: {leaked[:5]}")
    return output_rows


def stable_path_seed(path: Path, base_seed: int = 42) -> int:
    """Stable helper retained for downstream reproducibility diagnostics."""
    return int(zlib.crc32(f"{base_seed}|{path.as_posix()}".encode()) & 0xFFFFFFFF)
