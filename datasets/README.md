# Dataset placement

The dataset itself is not distributed with TIM_2026.

For the simplest project-local setup, place or link it here:

```text
TIM_2026/
└── datasets/
    └── scalogram_27_1/
        ├── train/
        ├── val/
        └── test/
```

Each split must contain the four class folders:

```text
surface/
internal/
corona/
notpd/
```

Then run from the project root:

```bash
bash scripts/train_pect.sh datasets/scalogram_27_1 1 60 0
```

The data may also remain anywhere outside the repository, including a mounted
server volume. In that case, pass its path explicitly:

```bash
bash scripts/train_pect.sh /mnt/datasets/scalogram_27_1 1 60 0
```

Do not commit images, generated splits, or checkpoints to this repository.
