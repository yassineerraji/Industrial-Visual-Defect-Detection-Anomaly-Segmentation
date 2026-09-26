import pytest
import torch

from defect_detection.models.unet import ResNetUNet
from defect_detection.training.losses import bce_dice, soft_dice_loss


@pytest.mark.parametrize("shape", [(2, 3, 64, 256), (1, 3, 96, 96)])
def test_unet_output_shape_and_map_range(shape):
    model = ResNetUNet(pretrained=False, decoder_channels=16).eval()
    x = torch.randn(shape)
    assert model(x).shape == (shape[0], 1, *shape[2:])
    amap = model.anomaly_map(x)
    assert amap.min() >= 0 and amap.max() <= 1


def test_unet_rejects_indivisible_input():
    with pytest.raises(ValueError, match="divisible"):
        ResNetUNet(pretrained=False)(torch.randn(1, 3, 50, 64))


def test_soft_dice_loss_extremes():
    target = torch.zeros(1, 1, 4, 4)
    target[..., :2, :2] = 1
    perfect = torch.where(target > 0, 20.0, -20.0)
    assert soft_dice_loss(perfect, target) < 1e-3
    assert soft_dice_loss(-perfect, target) > 0.8


def test_bce_dice_backpropagates():
    model = ResNetUNet(pretrained=False, decoder_channels=8)
    batch = {"image": torch.randn(2, 3, 64, 64), "mask": (torch.rand(2, 1, 64, 64) > 0.95).float()}
    loss = bce_dice(model, batch, torch.device("cpu"))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None
