# Dataset placement and generation

TIM_2026 does not distribute raw signals, extracted pulses, or scalograms.
Keep large data outside Git and pass its path explicitly.

## Generate scalograms from raw MAT signals

From the repository root:

```powershell
python prepare_data.py `
  --input "C:\path\to\PD_data_27_1_mat" `
  --output "D:\prepared\pd_27_1" `
  --workers 4
```

The training-ready path is:

```text
D:\prepared\pd_27_1\scalograms
```

Use it directly:

```powershell
python pect.py `
  --dataset-path "D:\prepared\pd_27_1\scalograms" `
  --shot 1 `
  --training-samples 60 `
  --gpu 0
```

See the main [`README.md`](../README.md) for the raw MAT layout, pulse/CWT
settings, visualization commands, and diagnostics.

## Use an already prepared dataset

The required structure is:

```text
scalograms/
├── train/
│   ├── surface/*.png
│   ├── internal/*.png
│   ├── corona/*.png
│   └── notpd/*.png
├── val/
│   └── <same four classes>/*.png
└── test/
    └── <same four classes>/*.png
```

The folder may be anywhere:

```bash
python pect.py --dataset-path /mnt/datasets/pd_27_1/scalograms --shot 1
```

Alternatively, place or link it at `datasets/scalogram_27_1/` and pass that
relative path. Do not commit generated images, MAT files, or checkpoints.
