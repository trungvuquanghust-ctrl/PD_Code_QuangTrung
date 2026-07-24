"""Reproducible runtime helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .config import RuntimeConfig


def resolve_device(runtime: RuntimeConfig) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{runtime.gpu_id}")
    return torch.device("cpu")


def configure_runtime(runtime: RuntimeConfig) -> torch.device:
    if runtime.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(runtime.seed)
    np.random.seed(runtime.seed)
    torch.manual_seed(runtime.seed)
    torch.cuda.manual_seed_all(runtime.seed)
    torch.backends.cudnn.deterministic = runtime.deterministic
    torch.backends.cudnn.benchmark = runtime.cudnn_benchmark
    return resolve_device(runtime)
