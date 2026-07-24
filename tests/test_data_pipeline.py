from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

from data_pipeline.scalogram import generate_scalogram
from data_pipeline.split_dataset import split_pulses


def test_scalogram_has_expected_size(tmp_path: Path) -> None:
    time = np.arange(1280) / 200e6
    centered = time - 1.25e-6
    pulse = np.exp(-((centered / 0.12e-6) ** 2)) * np.cos(
        2 * np.pi * 8e6 * centered
    )
    output = tmp_path / "sample.png"
    stats = generate_scalogram(pulse, time, output)
    with Image.open(output) as image:
        assert image.size == (224, 224)
        assert image.mode == "RGB"
    assert 0.9e6 < stats["freq_min_hz"] < 1.1e6
    assert 16.4e6 < stats["freq_max_hz"] < 16.6e6


def test_split_keeps_raw_source_groups_together(tmp_path: Path) -> None:
    flat_root = tmp_path / "flat"
    split_root = tmp_path / "split"
    manifest: list[dict[str, object]] = []
    time = (np.arange(1280) / 200e6).reshape(-1, 1)
    for class_name in ("surface", "internal", "corona", "notpd"):
        for source_index in range(5):
            for pulse_index in range(2):
                pulse_id = f"{class_name}_{source_index}_{pulse_index}"
                relative = Path(class_name) / f"{pulse_id}.mat"
                (flat_root / class_name).mkdir(parents=True, exist_ok=True)
                sio.savemat(
                    flat_root / relative,
                    {
                        "Trace_3_VOLT": np.zeros((1280, 1)),
                        "Time_s": time,
                    },
                )
                manifest.append(
                    {
                        "pulse_id": pulse_id,
                        "class": class_name,
                        "source_class": class_name,
                        "source_file": f"raw_{source_index}.mat",
                        "source_path": f"/raw/{class_name}/raw_{source_index}.mat",
                        "pulse_path": str(relative),
                    }
                )
    # A raw PD source may also produce a NOTPD window. It must inherit the same
    # split even though its target class differs.
    extra_id = "notpd_from_surface_0"
    extra_relative = Path("notpd") / f"{extra_id}.mat"
    sio.savemat(
        flat_root / extra_relative,
        {"Trace_3_VOLT": np.zeros((1280, 1)), "Time_s": time},
    )
    manifest.append(
        {
            "pulse_id": extra_id,
            "class": "notpd",
            "source_class": "surface",
            "source_file": "raw_0.mat",
            "source_path": "/raw/surface/raw_0.mat",
            "pulse_path": str(extra_relative),
        }
    )
    rows = split_pulses(flat_root, split_root, manifest, seed=42)
    groups: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        key = (str(row["source_class"]), str(row["source_file"]))
        groups.setdefault(key, set()).add(str(row["split"]))
    assert all(len(splits) == 1 for splits in groups.values())
    assert {str(row["split"]) for row in rows} == {"train", "val", "test"}
    with (split_root / "split_manifest.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(manifest)
