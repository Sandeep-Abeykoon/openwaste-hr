from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducible experiments.

    Deterministic mode can make experiments slower, but it helps when comparing
    model variants during research.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    except ImportError:
        # Allows non-training utilities to run even if torch is not installed yet.
        return