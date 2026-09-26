import pytest
import torch

from defect_detection.utils.device import get_device


def test_auto_prefers_mps_then_cuda():
    device = get_device("auto")
    if torch.backends.mps.is_available():
        assert device.type == "mps"
    elif torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_cpu_always_available():
    assert get_device("cpu").type == "cpu"


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        get_device("tpu")


def test_unavailable_accelerator_raises_instead_of_falling_back():
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(RuntimeError):
        get_device("cuda")
