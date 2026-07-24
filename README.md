# TIM_2026 — PECT for few-shot partial-discharge classification

This repository contains the standalone PECT model and a complete data flow:

```text
raw .mat signal
  -> pulse detection and extraction
  -> leakage-safe train/val/test split
  -> CWT scalogram PNG
  -> PECT training and final test
```

No dataset is bundled. You always pass explicit input and output paths, so the
same commands work with a local folder, an external disk, or a mounted server
volume.

## 1. Install

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux or Git Bash

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Quick installation check:

```bash
python prepare_data.py --help
python pect.py --help
```

## 2. Prepare the dataset from raw MAT signals

The raw input must contain these three folders:

```text
PD_data_27_1_mat/
├── surface/*.mat
├── internal/*.mat
└── corona/*.mat
```

Supported MAT keys are:

- raw signals: `Voltage` and `Time`;
- legacy/extracted signals: `Trace_3_VOLT` and `Time_s`.

Run the complete flow with one command:

```powershell
python prepare_data.py `
  --input "C:\path\to\PD_data_27_1_mat" `
  --output "D:\prepared\pd_27_1" `
  --workers 4
```

Linux/Git Bash:

```bash
python prepare_data.py \
  --input /data/PD_data_27_1_mat \
  --output /data/prepared/pd_27_1 \
  --workers 4
```

`--output` must be a new or empty folder. This prevents a rerun from silently
mixing stale MAT/PNG files with newly generated data.

The output is:

```text
pd_27_1/
├── pulses_flat/                 # extracted pulse MAT files + provenance
├── pulses/
│   ├── train/<class>/*.mat
│   ├── val/<class>/*.mat
│   └── test/<class>/*.mat
├── scalograms/                  # pass this folder to --dataset-path
│   ├── train/<class>/*.png
│   ├── val/<class>/*.png
│   └── test/<class>/*.png
├── visualizations/
│   ├── raw_signals/             # green-gradient full signals
│   └── pulses/                  # green-gradient extracted pulses
└── diagnostics/
    ├── extraction_diagnostics.csv
    ├── extraction_errors.csv
    ├── scalogram_diagnostics.csv
    ├── scalogram_errors.csv
    └── pipeline_summary.json
```

The four output classes are `surface`, `internal`, `corona`, and `notpd`.
`notpd` windows are mined from high-frequency, PD-spectrum-similar background
regions that do not contain a dominant spike.

### What the preparation code preserves

Pulse extraction follows the workspace implementation:

- sampling rate: `200 MHz`;
- pulse window: `1280` samples (`6.4 us`);
- peak position: `1.25 us` (`250` samples);
- local prominence plus knee threshold;
- knee sensitivity: `0.4`;
- prominence median floor: `2.8`;
- minimum pulse SNR: `7.8`;
- internal-class multi-peak rejection.

Scalogram generation also preserves the original settings:

- Complex Morlet `cmor1.5-1.0`;
- `200` log-spaced scales;
- `1–16.5 MHz`;
- `abs(CWT) -> log1p(200*x) -> per-image min-max normalization`;
- `inferno` colormap and `224 x 224` PNG.

The split is deterministic (`--seed 42`) and groups samples by raw source file.
Therefore, pulses extracted from one raw recording cannot appear in more than
one of train, validation, and test.

### Fast smoke test

Before processing every raw file:

```powershell
python prepare_data.py `
  --input "C:\path\to\PD_data_27_1_mat" `
  --output "D:\prepared\pd_27_1_smoke" `
  --limit-files-per-class 1 `
  --visualize-limit-per-class 1 `
  --workers 1
```

The smoke-test split may not contain all four classes in every split because it
uses only one source recording per class. Use the full dataset for training.

### Visualization only

Render raw signals:

```powershell
python visualize_data.py `
  --kind raw `
  --input "C:\path\to\PD_data_27_1_mat" `
  --output "D:\viz\raw" `
  --limit-per-class 5
```

Render extracted pulses:

```powershell
python visualize_data.py `
  --kind pulse `
  --input "D:\prepared\pd_27_1\pulses" `
  --output "D:\viz\pulses" `
  --limit-per-class 5 `
  --show-peak-position
```

Both commands use the same amplitude-weighted green `Greens` gradient as the
dataset workspace scripts and write PNG, PDF, and `file_mapping.csv`.

## 3. Train canonical PECT

Use the generated `scalograms` folder:

```powershell
python pect.py `
  --dataset-path "D:\prepared\pd_27_1\scalograms" `
  --dataset-name pd_27_1 `
  --shot 1 `
  --training-samples 60 `
  --gpu 0
```

For 5-shot, change `--shot 1` to `--shot 5`. To use all training images, omit
`--training-samples`.

Inspect the resolved configuration without training:

```powershell
python pect.py `
  --dataset-path "D:\prepared\pd_27_1\scalograms" `
  --dry-run
```

Test an existing checkpoint:

```powershell
python pect.py `
  --mode test `
  --weights "artifacts\<run-name>\checkpoints\best.pt" `
  --dataset-path "D:\prepared\pd_27_1\scalograms" `
  --shot 1
```

Linux/Git Bash launchers are also available:

```bash
bash scripts/train_pect.sh /data/prepared/pd_27_1/scalograms 1 60 0
bash scripts/test_pect.sh /data/prepared/pd_27_1/scalograms \
  artifacts/<run-name>/checkpoints/best.pt 1 0
```

## 4. Outputs and debugging

Each experiment writes:

```text
artifacts/<run-name>/
├── config.txt
├── history.csv
├── metrics.txt
├── metrics.csv
├── confusion_matrix.{png,pdf}
├── tsne.{png,pdf}
└── checkpoints/best.pt
```

Do not judge preprocessing only from final accuracy:

- `extraction_diagnostics.csv` reports candidates, accepted peaks, rejection
  counts, knee thresholds, and `notpd` selection per raw file;
- `pulse_manifest.csv` maps every pulse to its raw recording;
- `split_manifest.csv` records the split group and seed and allows a direct
  leakage audit;
- `scalogram_diagnostics.csv` reports CWT/log/normalization statistics for every
  output image;
- green-gradient raw and pulse plots allow visual inspection before training.

## 5. PECT architecture and novelty boundary

PECT is not presented as “UOT alone.” Its standalone architecture is the
episode-level combination of:

1. ResNet12 spatial tokens;
2. learned local-token projection;
3. class/query unbalanced optimal transport;
4. threshold-mass local scoring;
5. a small global-prototype residual with weight `0.1`.

The repository includes 14 controlled ablations to test whether those pieces
are effective beyond final accuracy: no-global/global-only, global-weight
sweep, fixed/latent rho, full/partial OT, class-pooled matching, and cost-only
scoring.

This is the code-level contribution boundary relative to a standard global
prototype baseline and simpler local-matching variants. It is not, by itself, a
claim that each component is new in the entire literature; a paper-level
novelty claim still requires a current literature review. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 6. Reproduce and verify

```bash
python -m pytest
python -m compileall -q .
```

Canonical training settings are ResNet12, 4-way 1/5-shot, 100 epochs,
130/150/150 train/validation/test episodes, AdamW with learning rate and weight
decay `5e-4`, UOT `rho=0.8`, `tau_q=tau_c=0.5`, train seed `42`, and final-test
episode seed `200042`.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact model
parity boundary and [`datasets/README.md`](datasets/README.md) for external
dataset placement.
