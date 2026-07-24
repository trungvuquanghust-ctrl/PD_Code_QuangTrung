"""Prominence + knee pulse extraction adapted from dataset/pulse_extraction_knee.py.

The signal-processing constants and decisions intentionally match the research
workspace implementation. This standalone version adds a provenance manifest
and leaves plotting to the green-gradient visualization module.
"""

from __future__ import annotations

import csv
import zlib
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import argrelextrema, find_peaks, peak_prominences
from tqdm import tqdm


FS = 200e6
WINDOW_SIZE = 1280
PEAK_POSITION_US = 1.25
PEAK_POSITION_SAMPLES = int(PEAK_POSITION_US * 1e-6 * FS)
MIN_PEAK_DISTANCE_US = 3.0
MIN_PEAK_DISTANCE_SAMPLES = int(MIN_PEAK_DISTANCE_US * 1e-6 * FS)
PEAK_NEIGHBORHOOD_SAMPLES = 15
PROMINENCE_WLEN = 401
KNEE_SENSITIVITY = 0.4
PROMINENCE_MEDIAN_FACTOR = 2.8
MIN_PULSE_SNR = 7.8

NOTPD_FREQ_LOW_HZ = 2e6
NOTPD_FREQ_HIGH_HZ = 35e6
NOTPD_MIN_HF_RATIO = 0.20
NOTPD_MIN_SPEC_SIM = 0.65
NOTPD_MAX_CREST_FACTOR = 5.2
NOTPD_MAX_PEAK_TO_P95 = 2.8
NOTPD_MAX_AMP_RATIO_TO_PD = 0.45
NOTPD_SCAN_STRIDE = WINDOW_SIZE // 2
NOTPD_MAX_BACKGROUND_SCANS = 320
NOTPD_MAX_PER_FILE = 48
NOTPD_PER_CLASS = 300
NOTPD_RANDOM_SEED = 42

MULTI_PEAK_SECONDARY_RATIO = 0.30
MULTI_PEAK_MIN_SEPARATION_US = 0.3
MULTI_PEAK_MIN_SEPARATION = int(MULTI_PEAK_MIN_SEPARATION_US * 1e-6 * FS)
MULTI_PEAK_DOMINANCE_RATIO = 0.50

CLASS_NAMES = ("surface", "internal", "corona")
CLASS_PARAMS = {
    "surface": {"skip_prominence": True},
    "internal": {"reject_multi_peak": True},
}


@dataclass(frozen=True)
class ExtractionConfig:
    sensitivity: float = KNEE_SENSITIVITY
    prominence_median_factor: float = PROMINENCE_MEDIAN_FACTOR
    min_snr: float = MIN_PULSE_SNR
    workers: int = max(1, cpu_count() - 2)
    notpd_per_class: int = NOTPD_PER_CLASS


def load_mat_file(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat_data = sio.loadmat(file_path)
    if "Voltage" in mat_data and "Time" in mat_data:
        voltage = mat_data["Voltage"].ravel()
        time = mat_data["Time"].ravel()
    elif "Trace_3_VOLT" in mat_data and "Time_s" in mat_data:
        voltage = mat_data["Trace_3_VOLT"].ravel()
        time = mat_data["Time_s"].ravel()
    else:
        keys = [key for key in mat_data if not key.startswith("__")]
        raise ValueError(f"Unknown MAT keys: {keys}")
    if voltage.size != time.size or voltage.size == 0:
        raise ValueError(f"Invalid signal sizes: {voltage.size}/{time.size}")
    return voltage.astype(float, copy=False), time.astype(float, copy=False)


def save_pulse_mat(pulse: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(WINDOW_SIZE) / FS
    sio.savemat(
        output_path,
        {
            "Trace_3_VOLT": np.asarray(pulse).reshape(-1, 1),
            "Time_s": time.reshape(-1, 1),
        },
        do_compression=True,
    )


def find_knee_point(
    values: np.ndarray,
    sensitivity: float = KNEE_SENSITIVITY,
    prominence_median_factor: float = PROMINENCE_MEDIAN_FACTOR,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    if len(values) < 3:
        return (
            values[0] if len(values) else 0.0,
            0,
            values,
            np.array([]),
        )
    sorted_values = np.sort(values)[::-1]
    n = len(sorted_values)
    x = np.arange(n) / (n - 1)
    y_range = sorted_values[0] - sorted_values[-1]
    if y_range < 1e-12:
        return sorted_values[0], 0, sorted_values, np.zeros(n)
    y = (sorted_values - sorted_values[-1]) / y_range
    distances = np.abs(x + y - 1) / np.sqrt(2)
    knee_idx = int(np.argmax(distances))
    if sensitivity < 1.0:
        adjusted_idx = max(int(knee_idx * sensitivity), 0)
    elif sensitivity > 1.0:
        adjusted_idx = min(int(knee_idx * sensitivity), n - 1)
    else:
        adjusted_idx = knee_idx
    knee_value = float(sorted_values[adjusted_idx])
    min_threshold = float(np.median(sorted_values) * prominence_median_factor)
    if knee_value < min_threshold:
        knee_value = min_threshold
        adjusted_idx = int(np.searchsorted(-sorted_values, -knee_value))
    return knee_value, adjusted_idx, sorted_values, distances


def _non_maximum_suppression(
    peaks: np.ndarray, strengths: np.ndarray, signal_length: int
) -> np.ndarray:
    if len(peaks) <= 1:
        return peaks
    order = np.argsort(-strengths)
    kept = np.zeros(len(peaks), dtype=bool)
    suppressed = np.zeros(signal_length, dtype=bool)
    for index in order:
        peak = int(peaks[index])
        if not suppressed[peak]:
            kept[index] = True
            lo = max(0, peak - MIN_PEAK_DISTANCE_SAMPLES + 1)
            hi = min(signal_length, peak + MIN_PEAK_DISTANCE_SAMPLES)
            suppressed[lo:hi] = True
    return np.sort(peaks[kept])


def detect_peaks_knee_method(
    signal: np.ndarray,
    *,
    sensitivity: float,
    prominence_median_factor: float,
    skip_prominence: bool,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, object]]:
    abs_signal = np.abs(signal)
    candidates = argrelextrema(
        abs_signal, np.greater, order=PEAK_NEIGHBORHOOD_SAMPLES
    )[0]
    if len(candidates) == 0:
        return np.array([], dtype=int), 0.0, candidates, {
            "n_candidates": 0,
            "n_accepted": 0,
            "acceptance_rate": 0.0,
        }
    if skip_prominence:
        final = _non_maximum_suppression(
            candidates, abs_signal[candidates], len(signal)
        )
        threshold = 0.0
    else:
        prominences, _, _ = peak_prominences(
            abs_signal, candidates, wlen=PROMINENCE_WLEN
        )
        threshold, _, _, _ = find_knee_point(
            prominences,
            sensitivity=sensitivity,
            prominence_median_factor=prominence_median_factor,
        )
        mask = prominences >= threshold
        final = _non_maximum_suppression(
            candidates[mask], prominences[mask], len(signal)
        )
    return final, threshold, candidates, {
        "n_candidates": int(len(candidates)),
        "n_accepted": int(len(final)),
        "acceptance_rate": float(len(final) / len(candidates)),
        "skip_prominence": bool(skip_prominence),
    }


def extract_window_around_peak(signal: np.ndarray, peak_idx: int) -> np.ndarray | None:
    start = int(peak_idx) - PEAK_POSITION_SAMPLES
    end = start + WINDOW_SIZE
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]


def center_window_baseline(window: np.ndarray) -> tuple[np.ndarray, float]:
    n = len(window)
    quiet = np.concatenate([window[: int(n * 0.10)], window[int(n * 0.70) :]])
    baseline = np.median(quiet)
    centered = window - baseline
    return centered, float(np.std(quiet - baseline))


def validate_and_center_pulse(
    window: np.ndarray, min_snr: float
) -> tuple[np.ndarray | None, float]:
    centered, noise_std = center_window_baseline(window)
    noise_std = max(noise_std, 1e-15)
    snr = float(np.max(np.abs(centered)) / noise_std)
    return (centered if snr >= min_snr else None), snr


def validate_single_peak(centered_window: np.ndarray) -> bool:
    signal_max = float(np.max(np.abs(centered_window)))
    if signal_max < 1e-15:
        return False
    height = MULTI_PEAK_SECONDARY_RATIO * signal_max
    prominence = 0.1 * float(np.std(centered_window))
    positive, _ = find_peaks(
        centered_window,
        height=height,
        distance=MULTI_PEAK_MIN_SEPARATION,
        prominence=prominence,
    )
    negative, _ = find_peaks(
        -centered_window,
        height=height,
        distance=MULTI_PEAK_MIN_SEPARATION,
        prominence=prominence,
    )
    peaks = np.unique(np.sort(np.concatenate([positive, negative])))
    if len(peaks) <= 1:
        return True
    amplitudes = np.abs(centered_window[peaks])
    order = np.argsort(-amplitudes)
    main_index = int(peaks[order[0]])
    main_amplitude = float(amplitudes[order[0]])
    for index in order[1:]:
        ratio = float(amplitudes[index] / main_amplitude)
        separation_us = abs(int(peaks[index]) - main_index) / FS * 1e6
        if (
            ratio >= MULTI_PEAK_DOMINANCE_RATIO
            and separation_us >= MULTI_PEAK_MIN_SEPARATION_US
        ):
            return False
    return True


def _compute_band_profile(window: np.ndarray) -> tuple[np.ndarray | None, float]:
    spectrum = np.abs(np.fft.rfft(np.asarray(window, dtype=float))) ** 2
    if len(spectrum) < 4:
        return None, 0.0
    spectrum[0] = 0.0
    frequencies = np.fft.rfftfreq(len(window), d=1.0 / FS)
    mask = (frequencies >= NOTPD_FREQ_LOW_HZ) & (
        frequencies <= NOTPD_FREQ_HIGH_HZ
    )
    band_power = spectrum[mask]
    total_energy = float(np.sum(spectrum)) + 1e-12
    band_ratio = float(np.sum(band_power) / total_energy)
    norm = float(np.linalg.norm(band_power))
    return (band_power / norm if norm >= 1e-12 else None), band_ratio


def _is_non_spiky_window(window: np.ndarray) -> tuple[bool, float]:
    absolute = np.abs(window)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(window**2))) + 1e-12
    p95 = float(np.percentile(absolute, 95)) + 1e-12
    crest = peak / rms
    return (
        crest <= NOTPD_MAX_CREST_FACTOR and peak / p95 <= NOTPD_MAX_PEAK_TO_P95
    ), crest


def _overlaps(start: int, end: int, ranges: list[tuple[int, int]], margin: int) -> bool:
    return any(start < other_end + margin and end > other_start - margin
               for other_start, other_end in ranges)


def select_frequency_similar_notpd(
    signal: np.ndarray,
    pd_windows: list[np.ndarray],
    extracted_ranges: list[tuple[int, int]],
    rejected_windows: list[np.ndarray],
    seed: int,
) -> tuple[list[np.ndarray], dict[str, int]]:
    profiles: list[np.ndarray] = []
    pd_peak_amplitudes: list[float] = []
    for window in pd_windows:
        profile, _ = _compute_band_profile(window)
        if profile is not None:
            profiles.append(profile)
            pd_peak_amplitudes.append(float(np.max(np.abs(window))))
    empty = {
        "n_notpd_raw": 0,
        "n_notpd_selected": 0,
        "n_notpd_from_rejected_snr": 0,
        "n_notpd_from_background": 0,
    }
    if not profiles:
        return [], empty
    reference = np.mean(np.vstack(profiles), axis=0)
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm < 1e-12:
        return [], empty
    reference /= reference_norm
    amplitude_limit = max(
        float(np.median(pd_peak_amplitudes)) * NOTPD_MAX_AMP_RATIO_TO_PD, 1e-12
    )
    scored: list[tuple[float, np.ndarray, tuple[int, int] | None, str]] = []
    rejected_selected = 0
    for centered in rejected_windows:
        if float(np.max(np.abs(centered))) > amplitude_limit:
            continue
        shape_ok, crest = _is_non_spiky_window(centered)
        profile, hf_ratio = _compute_band_profile(centered)
        if not shape_ok or profile is None or hf_ratio < NOTPD_MIN_HF_RATIO:
            continue
        similarity = float(np.dot(profile, reference))
        if similarity >= NOTPD_MIN_SPEC_SIM:
            score = similarity * (0.5 + hf_ratio) / (1.0 + 0.12 * crest)
            scored.append((score, centered.copy(), None, "rejected"))
            rejected_selected += 1
    starts = np.arange(
        0, max(len(signal) - WINDOW_SIZE + 1, 0), NOTPD_SCAN_STRIDE, dtype=int
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(starts)
    for start in starts[:NOTPD_MAX_BACKGROUND_SCANS]:
        end = int(start + WINDOW_SIZE)
        if _overlaps(int(start), end, extracted_ranges, WINDOW_SIZE // 4):
            continue
        centered, _ = center_window_baseline(signal[start:end])
        if float(np.max(np.abs(centered))) > amplitude_limit:
            continue
        shape_ok, crest = _is_non_spiky_window(centered)
        profile, hf_ratio = _compute_band_profile(centered)
        if not shape_ok or profile is None or hf_ratio < NOTPD_MIN_HF_RATIO:
            continue
        similarity = float(np.dot(profile, reference))
        if similarity >= NOTPD_MIN_SPEC_SIM:
            score = similarity * (0.5 + hf_ratio) / (1.0 + 0.12 * crest)
            scored.append((score, centered.copy(), (int(start), end), "background"))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected: list[np.ndarray] = []
    selected_ranges: list[tuple[int, int]] = []
    background_selected = 0
    for _, window, span, source in scored:
        if span is not None:
            if _overlaps(span[0], span[1], selected_ranges, WINDOW_SIZE // 3):
                continue
            selected_ranges.append(span)
        selected.append(window)
        background_selected += int(source == "background")
        if len(selected) >= NOTPD_MAX_PER_FILE:
            break
    return selected, {
        "n_notpd_raw": len(scored),
        "n_notpd_selected": len(selected),
        "n_notpd_from_rejected_snr": rejected_selected,
        "n_notpd_from_background": background_selected,
    }


def process_file(
    file_path: Path,
    class_name: str,
    config: ExtractionConfig,
) -> dict[str, object]:
    voltage, _ = load_mat_file(file_path)
    params = CLASS_PARAMS.get(class_name, {})
    peaks, threshold, candidates, diagnostics = detect_peaks_knee_method(
        voltage,
        sensitivity=config.sensitivity,
        prominence_median_factor=config.prominence_median_factor,
        skip_prominence=bool(params.get("skip_prominence", False)),
    )
    windows: list[np.ndarray] = []
    rejected_windows: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    rejected_snr = 0
    rejected_multi = 0
    for peak in peaks:
        if any(start <= peak < end for start, end in ranges):
            continue
        raw_window = extract_window_around_peak(voltage, int(peak))
        if raw_window is None:
            continue
        centered, _ = validate_and_center_pulse(raw_window, config.min_snr)
        if centered is None:
            rejected_snr += 1
            rejected_windows.append(center_window_baseline(raw_window)[0])
            continue
        if params.get("reject_multi_peak", False) and not validate_single_peak(centered):
            rejected_multi += 1
            continue
        windows.append(centered)
        start = int(peak) - PEAK_POSITION_SAMPLES
        ranges.append((start, start + WINDOW_SIZE))
    file_seed = int(zlib.crc32(str(file_path).encode("utf-8")) & 0xFFFFFFFF)
    notpd, notpd_info = select_frequency_similar_notpd(
        voltage, windows, ranges, rejected_windows, file_seed
    )
    diagnostics.update(notpd_info)
    diagnostics.update(
        {
            "n_rejected_snr": rejected_snr,
            "n_rejected_multi_peak": rejected_multi,
            "n_pulses": len(windows),
            "threshold": float(threshold),
            "candidate_count": len(candidates),
        }
    )
    return {
        "source_path": str(file_path),
        "source_file": file_path.name,
        "source_stem": file_path.stem,
        "class": class_name,
        "pulses": windows,
        "notpd": notpd,
        "diagnostics": diagnostics,
    }


def _worker(task: tuple[str, str, ExtractionConfig]) -> tuple[bool, object]:
    path, class_name, config = task
    try:
        return True, process_file(Path(path), class_name, config)
    except Exception as exc:
        return False, {"source_path": path, "class": class_name, "error": str(exc)}


def extract_dataset(
    input_root: Path,
    output_root: Path,
    diagnostics_root: Path,
    *,
    classes: tuple[str, ...] = CLASS_NAMES,
    config: ExtractionConfig = ExtractionConfig(),
    limit_per_class: int | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Extract flat pulse MAT files and write provenance plus per-file diagnostics."""
    tasks: list[tuple[str, str, ExtractionConfig]] = []
    for class_name in classes:
        class_dir = input_root / class_name
        if not class_dir.is_dir():
            continue
        paths = sorted(class_dir.glob("*.mat"))
        if limit_per_class is not None:
            paths = paths[:limit_per_class]
        tasks.extend((str(path), class_name, config) for path in paths)
    if not tasks:
        raise FileNotFoundError(
            f"No MAT files found under {input_root}/{{{','.join(classes)}}}"
        )
    workers = max(1, min(config.workers, len(tasks)))
    if workers == 1:
        results = [_worker(task) for task in tqdm(tasks, desc="Extracting pulses")]
    else:
        with Pool(workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_worker, tasks),
                    total=len(tasks),
                    desc="Extracting pulses",
                )
            )

    output_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    counters = {class_name: 0 for class_name in (*classes, "notpd")}
    reservoirs: dict[str, list[tuple[np.ndarray, dict[str, object]]]] = {
        class_name: [] for class_name in classes
    }
    seen = {class_name: 0 for class_name in classes}
    rngs = {
        class_name: np.random.default_rng(NOTPD_RANDOM_SEED + index)
        for index, class_name in enumerate(classes)
    }
    errors: list[dict[str, object]] = []

    for success, payload in results:
        if not success:
            errors.append(payload)  # type: ignore[arg-type]
            continue
        result = payload  # type: ignore[assignment]
        class_name = str(result["class"])
        source_path = str(result["source_path"])
        source_file = str(result["source_file"])
        for pulse in result["pulses"]:
            counters[class_name] += 1
            pulse_id = f"{class_name}_{counters[class_name]:05d}"
            relative_path = Path(class_name) / f"{pulse_id}.mat"
            save_pulse_mat(pulse, output_root / relative_path)
            manifest.append(
                {
                    "pulse_id": pulse_id,
                    "class": class_name,
                    "source_class": class_name,
                    "source_file": source_file,
                    "source_path": source_path,
                    "pulse_path": str(relative_path),
                }
            )
        for window in result["notpd"]:
            seen[class_name] += 1
            item = (
                window,
                {
                    "source_class": class_name,
                    "source_file": source_file,
                    "source_path": source_path,
                },
            )
            reservoir = reservoirs[class_name]
            if len(reservoir) < config.notpd_per_class:
                reservoir.append(item)
            else:
                replacement = int(rngs[class_name].integers(0, seen[class_name]))
                if replacement < config.notpd_per_class:
                    reservoir[replacement] = item
        diagnostic_rows.append(
            {
                "class": class_name,
                "source_file": source_file,
                "source_path": source_path,
                **result["diagnostics"],
            }
        )

    for source_class in classes:
        for window, provenance in reservoirs[source_class]:
            counters["notpd"] += 1
            pulse_id = f"notpd_{counters['notpd']:05d}"
            relative_path = Path("notpd") / f"{pulse_id}.mat"
            save_pulse_mat(window, output_root / relative_path)
            manifest.append(
                {
                    "pulse_id": pulse_id,
                    "class": "notpd",
                    **provenance,
                    "pulse_path": str(relative_path),
                }
            )

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_root / "pulse_manifest.csv", manifest)
    write_csv(diagnostics_root / "extraction_diagnostics.csv", diagnostic_rows)
    write_csv(diagnostics_root / "extraction_errors.csv", errors)
    return manifest, diagnostic_rows
