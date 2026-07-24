"""Green-gradient waveform visualizations copied from the dataset workspace style."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np
import scipy.io as sio


FIGSIZE_INCHES = (10, 4)
DPI = 150
PEAK_POSITION_US = 1.25
FONT_SIZE_MULTIPLIER = 1.475
FONT_SCALE = 1.25 * FONT_SIZE_MULTIPLIER
AXIS_LABEL_SIZE = 12 * FONT_SCALE
TICK_LABEL_SIZE = 10 * FONT_SCALE
AXIS_LABEL_PAD = 8
Y_TICK_COUNT = 5
Y_MARGIN_RATIO = 0.02
GREEN_GRADIENT_CMAP = "Greens"
GREEN_GRADIENT_RANGE = (0.62, 0.99)
MAX_PLOT_POINTS = 100_000


def natural_sort_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def load_signal(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load either raw (Voltage/Time) or extracted (Trace_3_VOLT/Time_s) MAT."""
    mat_data = sio.loadmat(file_path)
    if "Voltage" in mat_data and "Time" in mat_data:
        voltage = mat_data["Voltage"].ravel()
        time = mat_data["Time"].ravel()
    elif "Trace_3_VOLT" in mat_data and "Time_s" in mat_data:
        voltage = mat_data["Trace_3_VOLT"].ravel()
        time = mat_data["Time_s"].ravel()
    else:
        keys = [key for key in mat_data if not key.startswith("__")]
        raise ValueError(f"Unknown MAT keys in {file_path}: {keys}")
    if voltage.size != time.size or voltage.size == 0:
        raise ValueError(
            f"Invalid voltage/time sizes in {file_path}: {voltage.size}/{time.size}"
        )
    finite = np.isfinite(voltage) & np.isfinite(time)
    voltage = voltage[finite].astype(np.float64, copy=False)
    time = time[finite].astype(np.float64, copy=False)
    if voltage.size == 0:
        raise ValueError(f"No finite samples in {file_path}")
    return voltage, time


def nice_tick_step(required_step: float) -> float:
    if not np.isfinite(required_step) or required_step <= 0:
        return 1e-3
    exponent = np.floor(np.log10(required_step))
    scale = 10.0**exponent
    fraction = required_step / scale
    for nice_fraction in (
        1.0,
        1.2,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
    ):
        if fraction <= nice_fraction:
            return nice_fraction * scale
    return 10.0 * scale


def amplitude_tick_formatter(step: float) -> FuncFormatter:
    decimals = max(0, int(np.ceil(-np.log10(abs(step)))) + 1)
    decimals = min(decimals, 8)

    def format_tick(value: float, _: int) -> str:
        if abs(value) < abs(step) * 1e-8:
            return "0"
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

    return FuncFormatter(format_tick)


def plot_green_gradient_waveform(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    linewidth: float = 0.8,
    alpha: float = 0.95,
) -> None:
    """Draw the exact amplitude-weighted green gradient used in dataset scripts."""
    points = np.column_stack([x_values, y_values]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_strength = np.maximum(np.abs(y_values[:-1]), np.abs(y_values[1:]))
    max_strength = float(np.nanmax(segment_strength)) if segment_strength.size else 0.0
    if not np.isfinite(max_strength) or max_strength <= 0.0:
        segment_strength = np.full(segment_strength.shape, GREEN_GRADIENT_RANGE[0])
    else:
        low, high = GREEN_GRADIENT_RANGE
        segment_strength = low + (high - low) * (segment_strength / max_strength)
    collection = LineCollection(
        segments,
        cmap=plt.get_cmap(GREEN_GRADIENT_CMAP),
        norm=Normalize(vmin=0.0, vmax=1.0),
        linewidths=linewidth,
        alpha=alpha,
    )
    collection.set_array(segment_strength)
    ax.add_collection(collection)


def symmetric_amplitude_axis(
    values: np.ndarray,
) -> tuple[tuple[float, float], np.ndarray, float]:
    max_abs = float(np.nanmax(np.abs(values))) if values.size else 0.0
    if not np.isfinite(max_abs) or max_abs < 1e-12:
        max_abs = 1e-3
    half_steps = (Y_TICK_COUNT - 1) / 2.0
    required_limit = max_abs * (1.0 + Y_MARGIN_RATIO)
    step = nice_tick_step(required_limit / half_steps)
    y_limit = step * half_steps
    ticks = step * np.arange(-half_steps, half_steps + 1)
    return (-float(y_limit), float(y_limit)), ticks, step


def render_waveform(
    voltage: np.ndarray,
    time: np.ndarray,
    output_path: Path,
    *,
    pulse: bool,
    show_peak_position: bool = False,
) -> None:
    """Render a raw waveform (ms) or extracted pulse (us) as PNG and PDF."""
    relative_time = time - time[0]
    time_scale = 1e6 if pulse else 1e3
    unit = "us" if pulse else "ms"
    plot_time = relative_time * time_scale
    plot_voltage = voltage
    if not pulse and voltage.size > MAX_PLOT_POINTS:
        step = max(1, voltage.size // MAX_PLOT_POINTS)
        plot_time = plot_time[::step]
        plot_voltage = voltage[::step]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES, dpi=DPI)
    plot_green_gradient_waveform(ax, plot_time, plot_voltage)
    ax.set_xlabel(f"Time ({unit})", fontsize=AXIS_LABEL_SIZE, labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel("Amplitude (V)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([float(plot_time[0]), float(plot_time[-1])])
    y_limits, y_ticks, y_tick_step = symmetric_amplitude_axis(plot_voltage)
    ax.set_ylim(y_limits)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(amplitude_tick_formatter(y_tick_step))
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
    if pulse and show_peak_position:
        ax.axvline(x=PEAK_POSITION_US, color="r", linestyle="--", alpha=0.5)
    tick_labels = ax.get_xticklabels()
    if tick_labels:
        tick_labels[0].set_ha("left")
        tick_labels[-1].set_ha("right")
    plt.tight_layout()
    fig.savefig(output_path, format="png", dpi=DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def visualize_tree(
    input_root: Path,
    output_root: Path,
    *,
    pulse: bool,
    limit_per_class: int = 3,
    show_peak_position: bool = False,
) -> list[dict[str, object]]:
    """Render a reproducible sample from flat or split/class MAT trees."""
    records: list[dict[str, object]] = []
    split_dirs = [
        path
        for path in input_root.iterdir()
        if path.is_dir() and path.name in {"train", "val", "test"}
    ]
    roots = sorted(split_dirs, key=natural_sort_key) if split_dirs else [input_root]
    for split_root in roots:
        split = split_root.name if split_dirs else ""
        class_dirs = sorted(
            [path for path in split_root.iterdir() if path.is_dir()],
            key=natural_sort_key,
        )
        for class_dir in class_dirs:
            files = sorted(class_dir.glob("*.mat"), key=natural_sort_key)
            if limit_per_class > 0:
                files = files[:limit_per_class]
            for mat_path in files:
                voltage, time = load_signal(mat_path)
                relative = Path(split) / class_dir.name if split else Path(class_dir.name)
                output_path = output_root / relative / f"{mat_path.stem}.png"
                render_waveform(
                    voltage,
                    time,
                    output_path,
                    pulse=pulse,
                    show_peak_position=show_peak_position,
                )
                records.append(
                    {
                        "split": split,
                        "class": class_dir.name,
                        "mat_path": str(mat_path),
                        "png_path": str(output_path),
                        "pdf_path": str(output_path.with_suffix(".pdf")),
                        "num_samples": int(voltage.size),
                        "duration_seconds": float(time[-1] - time[0]),
                        "voltage_min": float(np.min(voltage)),
                        "voltage_max": float(np.max(voltage)),
                    }
                )
    output_root.mkdir(parents=True, exist_ok=True)
    mapping_path = output_root / "file_mapping.csv"
    if records:
        with mapping_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    else:
        mapping_path.write_text("", encoding="utf-8")
    return records
