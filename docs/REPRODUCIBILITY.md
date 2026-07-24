# Reproducibility contract

## Preserved exactly

- RGB loading, train-derived normalization, 84 x 84 resize, and class order.
- Deterministic episodic sampling formula.
- Balanced limited-sample selection with the seed reset per class.
- ResNet12 PECT numerical implementation and constructor defaults.
- UOT rho/tau/fixed-mass settings and threshold-mass scoring.
- Global residual mode and canonical weight `0.1`.
- AdamW, learning rate, weight decay, warmup, cosine schedule, and epoch count.
- 4-way 1/5-shot protocol with 1 query per class.
- 130/150/150 train/validation/test episodes.
- per-epoch training and validation seed semantics and final-test seed `200042`.
- best-checkpoint selection by validation accuracy.

## Intentionally removed

- W&B initialization, logging, summaries, and images.
- baseline model registry and baseline-specific loss branches.
- noise/robust benchmark protocols not part of the requested clean scope.
- transport, threshold, failure-probe, and exploratory diagnostic reports.
- UOT evidence, support-distribution, Q1, and training-curve figures.

## Retained outputs

- training/validation loss and accuracy history in CSV;
- essential final metrics in TXT and CSV;
- confusion matrix and t-SNE in PNG and PDF;
- best model checkpoint.

The reference transport implementation remains isolated under
`tim_2026/model/_reference`. The public constructor in
`tim_2026/model/pect.py` is the supported architecture surface.
