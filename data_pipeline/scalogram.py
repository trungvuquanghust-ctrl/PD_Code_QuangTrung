"""CWT scalogram generation matching dataset/scalogram_fewshot.py defaults."""

from __future__ import annotations

import csv
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.io
from scipy.signal import resample as scipy_resample
from tqdm import tqdm

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False
    from PIL import Image


FS = 200e6
WINDOW_SIZE = 1280
FREQ_MIN = 1e6
FREQ_MAX = 16.5e6
N_SCALES = 200
WAVELET = "cmor1.5-1.0"
LOG_GAIN = 200
OUTPUT_SIZE = (224, 224)
COLORMAP = "inferno"


def load_pulse(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat_data = scipy.io.loadmat(file_path)
    if "Voltage" in mat_data and "Time" in mat_data:
        voltage = mat_data["Voltage"].ravel()
        time = mat_data["Time"].ravel()
    elif "Trace_3_VOLT" in mat_data and "Time_s" in mat_data:
        voltage = mat_data["Trace_3_VOLT"].ravel()
        time = mat_data["Time_s"].ravel()
    else:
        keys = [key for key in mat_data if not key.startswith("__")]
        raise ValueError(f"Unknown MAT keys: {keys}")
    if len(voltage) != WINDOW_SIZE:
        duration = float(time[-1] - time[0])
        voltage = scipy_resample(voltage, WINDOW_SIZE)
        time = np.linspace(time[0], time[0] + duration, WINDOW_SIZE)
    return voltage.astype(float, copy=False), time.astype(float, copy=False)


def generate_scales() -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / FS
    center_frequency = pywt.central_frequency(WAVELET)
    scale_max = center_frequency / (FREQ_MIN * dt)
    scale_min = center_frequency / (FREQ_MAX * dt)
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), N_SCALES)
    frequencies = pywt.scale2frequency(WAVELET, scales) / dt
    return scales, frequencies


def generate_scalogram(
    voltage: np.ndarray,
    time: np.ndarray,
    output_path: Path,
) -> dict[str, float]:
    """CWT -> abs -> log1p(200*x) -> per-image normalization -> inferno PNG."""
    scales, frequencies = generate_scales()
    coefficients, _ = pywt.cwt(
        voltage, scales, WAVELET, sampling_period=1.0 / FS
    )
    magnitude = np.abs(coefficients)
    compressed = np.log1p(LOG_GAIN * magnitude)
    value_min = float(compressed.min())
    value_max = float(compressed.max())
    if value_max > value_min:
        normalized = (compressed - value_min) / (value_max - value_min)
    else:
        normalized = np.zeros_like(compressed)

    time_us = (time - time[0]) * 1e6
    grid_time, grid_frequency = np.meshgrid(time_us, frequencies / 1e6)
    figure, axis = plt.subplots(figsize=(3, 3), dpi=100)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.axis("off")
    axis.pcolormesh(
        grid_time,
        grid_frequency,
        normalized,
        shading="gouraud",
        cmap=COLORMAP,
        vmin=0,
        vmax=1,
    )
    axis.set_xlim(time_us.min(), time_us.max())
    axis.set_ylim(frequencies.min() / 1e6, frequencies.max() / 1e6)
    figure.canvas.draw()
    argb = np.frombuffer(figure.canvas.tostring_argb(), dtype=np.uint8)
    argb = argb.reshape(figure.canvas.get_width_height()[::-1] + (4,))
    plt.close(figure)
    red, green, blue = argb[:, :, 1], argb[:, :, 2], argb[:, :, 3]
    bgr = np.stack([blue, green, red], axis=2)
    if HAS_CV2:
        image = cv2.resize(bgr, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
    else:
        rgb = np.stack([red, green, blue], axis=2)
        image = np.asarray(Image.fromarray(rgb).resize(OUTPUT_SIZE, Image.Resampling.LANCZOS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f"{output_path.stem}.tmp.{os.getpid()}{output_path.suffix}"
    )
    if HAS_CV2:
        if not cv2.imwrite(str(temporary), image):
            raise OSError(f"Could not write scalogram: {temporary}")
    else:
        Image.fromarray(image).save(temporary)
    os.replace(temporary, output_path)
    return {
        "cwt_min": float(magnitude.min()),
        "cwt_max": float(magnitude.max()),
        "cwt_mean": float(magnitude.mean()),
        "log_min": value_min,
        "log_max": value_max,
        "normalized_mean": float(normalized.mean()),
        "freq_min_hz": float(frequencies.min()),
        "freq_max_hz": float(frequencies.max()),
    }


def _worker(task: tuple[str, str]) -> tuple[bool, dict[str, object]]:
    source_text, destination_text = task
    source = Path(source_text)
    destination = Path(destination_text)
    try:
        voltage, time = load_pulse(source)
        stats = generate_scalogram(voltage, time, destination)
        return True, {
            "pulse_path": str(source),
            "scalogram_path": str(destination),
            "num_samples": len(voltage),
            **stats,
        }
    except Exception as exc:
        return False, {
            "pulse_path": str(source),
            "scalogram_path": str(destination),
            "error": str(exc),
        }


def convert_dataset(
    pulse_root: Path,
    output_root: Path,
    diagnostics_root: Path,
    *,
    workers: int = 1,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tasks: list[tuple[str, str]] = []
    for split in ("train", "val", "test"):
        split_dir = pulse_root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for source in sorted(class_dir.glob("*.mat")):
                destination = output_root / split / class_dir.name / f"{source.stem}.png"
                tasks.append((str(source), str(destination)))
    if not tasks:
        raise FileNotFoundError(f"No split/class MAT files found under {pulse_root}")
    worker_count = max(1, min(int(workers), len(tasks)))
    if worker_count == 1:
        results = [_worker(task) for task in tqdm(tasks, desc="Generating scalograms")]
    else:
        with Pool(worker_count) as pool:
            results = list(
                tqdm(
                    pool.imap(_worker, tasks),
                    total=len(tasks),
                    desc="Generating scalograms",
                )
            )
    success_rows = [payload for success, payload in results if success]
    error_rows = [payload for success, payload in results if not success]
    diagnostics_root.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(diagnostics_root / "scalogram_diagnostics.csv", success_rows)
    write_csv(diagnostics_root / "scalogram_errors.csv", error_rows)
    if error_rows:
        raise RuntimeError(
            f"{len(error_rows)} scalograms failed; see "
            f"{diagnostics_root / 'scalogram_errors.csv'}"
        )
    return success_rows, error_rows
