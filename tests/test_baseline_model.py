import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.models.baseline_cnn import create_baseline_cnn  # noqa: E402


def test_create_baseline_cnn_forward_pass():
    model = create_baseline_cnn(
        model_name="mobilenetv3_small_100",
        num_classes=5,
        pretrained=False,
        drop_rate=0.1,
    )

    model.eval()

    inputs = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 5)


def test_create_baseline_cnn_rejects_invalid_num_classes():
    with pytest.raises(ValueError, match="num_classes must be at least 2"):
        create_baseline_cnn(
            model_name="mobilenetv3_small_100",
            num_classes=1,
            pretrained=False,
        )