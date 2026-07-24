# TIM_2026: PECT for Few-Shot Partial-Discharge Classification

This repository is the clean, standalone implementation of **PECT** used in the
TIM 2026 experimental pipeline. It contains one model only: the paper-facing
PECT configuration built from a ResNet12 backbone, local unbalanced optimal
transport (UOT), threshold-mass scoring, and a global residual classifier with
weight `0.1`.

The training protocol and model hyperparameters are preserved from the research
code. W&B, historical outputs, unrelated models, exploratory diagnostics, and
UOT evidence figures are intentionally excluded. The release retains only
essential TXT/CSV metrics, a confusion matrix, and a t-SNE figure.

## Method at a glance

For each few-shot episode, PECT follows:

```text
scalogram -> ResNet12 spatial tokens -> projected local descriptors
           -> class/query UOT matching -> threshold-mass local score
           -> + 0.1 * global prototype residual -> class logits
```

The local class score uses the transported mass and cost produced by UOT. The
global head is a residual correction, not a replacement for local transport in
the canonical configuration.

## Repository layout

```text
TIM_2026/
├── pect.py                     # canonical PECT train/test entry point
├── run_ablations.py            # clean runner for the 14 PECT ablations
├── tim_2026/
│   ├── config.py               # typed paper protocol and model configuration
│   ├── ablations.py            # one-factor ablation definitions
│   ├── data/
│   │   ├── loading.py          # unchanged image loading/normalization semantics
│   │   ├── episodic.py         # deterministic N-way K-shot sampler
│   │   └── pipeline.py         # tensor preparation and balanced subsets
│   ├── engine/runner.py        # train, validation, checkpoint, and final test
│   ├── model/
│   │   ├── pect.py             # small paper-facing PECT builder
│   │   └── _reference/         # exact numerical PECT/OT implementation
│   ├── logging.py              # TXT/CSV-only experiment logging
│   └── visualization.py        # confusion matrix and t-SNE only
├── tests/                      # protocol, model, sampler, and ablation tests
└── docs/                       # architecture and reproducibility notes
```

## Installation

Python 3.10+ and a CUDA-enabled PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is copied byte-for-byte from the workspace's canonical
`requirements_scalogram.txt`. That shared environment file still lists W&B for
compatibility with the wider workspace, but TIM_2026 contains no W&B imports,
initialization, or logging calls.

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset layout

No dataset is included and there is no machine-specific default path.
`--dataset-path` is always required.

The easiest setup is to place or symlink the dataset at
`datasets/scalogram_27_1/`. The dataset can also remain anywhere outside the
repository; simply pass that directory to Python or the Bash launcher. See
[`datasets/README.md`](datasets/README.md) for placement examples.

The required pre-split layout is:

```text
scalogram_27_1/
├── train/
│   ├── surface/
│   ├── internal/
│   ├── corona/
│   └── notpd/
├── val/
│   └── <same four classes>/
└── test/
    └── <same four classes>/
```

Images are converted to RGB, resized to `84 x 84`, and normalized using channel
statistics computed from the training split only. The canonical class order is
`surface`, `internal`, `corona`, `notpd`.

## Canonical PECT experiment

Train and run the final test:

```bash
python pect.py \
  --dataset-path datasets/scalogram_27_1 \
  --dataset-name knee_aug_split \
  --shot 1 \
  --training-samples 60 \
  --gpu 0
```

Use all available training samples by omitting `--training-samples`. For the
5-shot protocol, pass `--shot 5`.

Test an existing checkpoint:

```bash
python pect.py \
  --mode test \
  --weights artifacts/<run-name>/checkpoints/best.pt \
  --dataset-path datasets/scalogram_27_1 \
  --shot 1
```

Inspect the resolved configuration without loading data or training:

```bash
python pect.py --dataset-path datasets/scalogram_27_1 --dry-run
```

## Bash launchers

The `scripts/` directory provides strict Bash wrappers that work from any
current directory and automatically use `.venv/bin/python` when available.

Create the environment:

```bash
bash scripts/setup.sh
```

Train canonical PECT (`DATASET_PATH`, `SHOT`, `SAMPLES`, `GPU`):

```bash
bash scripts/train_pect.sh datasets/scalogram_27_1 1 60 0
bash scripts/train_pect.sh datasets/scalogram_27_1 5 all 1
```

Test a checkpoint:

```bash
bash scripts/test_pect.sh \
  datasets/scalogram_27_1 \
  artifacts/<run-name>/checkpoints/best.pt \
  1 0
```

Run or inspect ablations:

```bash
bash scripts/run_ablations.sh \
  datasets/scalogram_27_1 0 \
  pect_no_global,pect_cost_only \
  60,240 1,5 42 --dry-run
```

Optional environment variables are `PYTHON_BIN`, `DATASET_NAME`, `OUTPUT_DIR`,
`NUM_WORKERS`, `FINAL_TEST_SEED`, and `RUN_NAME` (train/test only). Any trailing
arguments are forwarded to the underlying Python entry point.

## Fixed paper protocol

| Setting | Value |
|---|---:|
| Classes | 4-way |
| Shots | 1 or 5 |
| Queries per class | 1 train / 1 val / 1 test |
| Input | RGB, 84 x 84 |
| Backbone | ResNet12 |
| Token dimension | 128 |
| Epochs | 100 |
| Episodes | 130 train / 150 val / 150 test |
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 5e-4 |
| Scheduler | 5-epoch linear warmup + cosine |
| Minimum LR | 1e-6 |
| Augmentation | Off |
| Label smoothing | 0.0 |
| UOT rho | 0.8 |
| UOT tau_q / tau_c | 0.5 / 0.5 |
| Global residual | residual mode, weight 0.1 |
| Default train seed | 42 |
| Final-test episode seed | 200042 |

Validation episodes use seed `100042 + epoch`, matching the wrapper protocol
in the source repository.

## Ablation suite

Print the full plan without launching jobs:

```bash
python run_ablations.py \
  --dataset-path datasets/scalogram_27_1 \
  --dry-run
```

Run a focused subset:

```bash
python run_ablations.py \
  --dataset-path datasets/scalogram_27_1 \
  --variants pect_no_global,pect_cost_only,pect_full_ot \
  --samples 60,240 \
  --shots 1,5 \
  --seeds 42 \
  --gpu 0
```

The runner contains the same 14 configurations as the original PECT suite:
no-global; global weights `0.05/0.1/0.15/0.2`; global-only; rho
`0.6/0.7/0.9`; latent-rho; full OT; partial OT; class-pooled; and cost-only.
The default cross product is `14 variants x 4 sample settings x 2 shots`.

## Outputs

Each run writes to one self-contained directory:

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

`metrics.txt` and `metrics.csv` contain only essential evaluation information:
test loss, episode accuracy mean/std/95% CI, query accuracy, macro
precision/recall/F1, inference time, and parameter count. A compact
`artifacts/summary.csv` accumulates one row per run.

## Reproducibility checks

```bash
python -m pytest
python -m compileall -q .
```

The model-parity test locks the canonical parameter count and constructor
contract. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact
equivalence boundary.

## Scope

This repository does not include baseline implementations, datasets,
checkpoints, old results, W&B integration, transport audit logs, failure-probe
reports, or UOT matching visualizations. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the standalone PECT boundary
and its distinction from standard global/prototype and local-matching baselines.
