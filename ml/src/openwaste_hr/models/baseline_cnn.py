from __future__ import annotations

import timm
import torch.nn as nn


def create_baseline_cnn(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    drop_rate: float = 0.0,
) -> nn.Module:
    """
    Create a lightweight CNN baseline model.

    The first OpenWaste-HR baseline uses timm so that the backbone can be
    swapped later without changing the training loop.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    return model