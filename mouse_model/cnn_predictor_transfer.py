"""
CNN + optional shifter for transfer learning: visual trunk (backbone) and separate readout.

Compatible with legacy checkpoints from monolithic VisualEncoder (encoder.layers.*) where
the last submodule is Linear; backbone maps to encoder.layers.0 .. layers.{READOUT_INDEX-1}.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.transform import get_affine_matrix2d, warp_affine

# Last layer index inside legacy nn.Sequential(VisualEncoder.layers)
_LEGACY_READOUT_INDEX = 13


def size_helper(in_length, kernel_size, padding=0, dilation=1, stride=1):
    res = in_length + 2 * padding - dilation * (kernel_size - 1) - 1
    res /= stride
    res += 1
    return np.floor(res)


class Shifter(nn.Module):
    def __init__(self, input_dim=4, output_dim=3, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.layers(x)
        x0 = (x[..., 0] + self.bias[0]) * 80 / 4
        x1 = (x[..., 1] + self.bias[1]) * 60 / 4
        x2 = (x[..., 2] + self.bias[2]) * 180 / 4
        x = torch.stack([x0, x1, x2], dim=-1)
        x = x.reshape(-1, 1, self.output_dim)
        return x


class VisualEncoderTransfer(nn.Module):
    """Conv backbone + Flatten + Linear readout (output_dim = num_neurons)."""

    def __init__(self, output_dim, input_shape=(60, 80), k1=7, k2=7, k3=7):
        super().__init__()
        self.input_shape = input_shape
        out_shape_0 = size_helper(in_length=input_shape[0], kernel_size=k1, stride=2)
        out_shape_0 = size_helper(in_length=out_shape_0, kernel_size=k2, stride=2)
        out_shape_0 = size_helper(in_length=out_shape_0, kernel_size=k3, stride=2)
        out_shape_1 = size_helper(in_length=input_shape[1], kernel_size=k1, stride=2)
        out_shape_1 = size_helper(in_length=out_shape_1, kernel_size=k2, stride=2)
        out_shape_1 = size_helper(in_length=out_shape_1, kernel_size=k3, stride=2)
        self.output_shape = (int(out_shape_0), int(out_shape_1))
        flat_dim = self.output_shape[0] * self.output_shape[1] * 32

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=k1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=k2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=k3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Flatten(),
        )
        self.readout = nn.Linear(flat_dim, output_dim)

    def forward(self, x):
        return self.readout(self.backbone(x))


class PredictorTransfer(nn.Module):
    def __init__(self, num_neurons: int, use_shifter: bool = False):
        super().__init__()
        self.encoder = VisualEncoderTransfer(output_dim=num_neurons)
        self.softplus = nn.Softplus()
        self.shifter = Shifter()
        self.use_shifter = use_shifter

    def forward(self, images, behav):
        if self.use_shifter:
            bs = images.size()[0]
            behav_shifter = torch.concat(
                (
                    behav[..., 4].unsqueeze(-1),
                    behav[..., 3].unsqueeze(-1),
                    behav[..., 1].unsqueeze(-1),
                    behav[..., 2].unsqueeze(-1),
                ),
                dim=-1,
            )
            shift_param = self.shifter(behav_shifter)
            shift_param = shift_param.reshape(-1, 3)
            scale_param = torch.ones_like(shift_param[..., 0:2]).to(shift_param.device)
            affine_mat = get_affine_matrix2d(
                translations=shift_param[..., 0:2],
                scale=scale_param,
                center=torch.repeat_interleave(
                    torch.tensor([[30, 40]], dtype=torch.float),
                    bs * 1,
                    dim=0,
                ).to(shift_param.device),
                angle=shift_param[..., 2],
            )
            affine_mat = affine_mat[:, :2, :]
            images = warp_affine(images, affine_mat, dsize=(60, 80))
        pred = self.encoder(images)
        pred = self.softplus(pred)
        return pred


def _legacy_backbone_state_dict(state_dict: dict) -> dict:
    """encoder.layers.{0..12}.* -> state dict for nn.Sequential backbone (keys 0..12.*)."""
    out = {}
    prefix = "encoder.layers."
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix) :]
        idx_str, _, tail = rest.partition(".")
        if not idx_str.isdigit():
            continue
        idx = int(idx_str)
        if idx >= _LEGACY_READOUT_INDEX:
            continue
        out[f"{idx_str}.{tail}"] = v
    return out


def load_encoder_backbone_from_checkpoint(
    model: PredictorTransfer,
    state_dict: dict,
) -> None:
    """
    Load weights into model.encoder.backbone only. Accepts:
    - Split format: encoder.backbone.*
    - Legacy format: encoder.layers.0..12.* (skips final Linear)
    """
    if any(k.startswith("encoder.backbone.") for k in state_dict):
        sub = {
            k[len("encoder.backbone.") :]: v
            for k, v in state_dict.items()
            if k.startswith("encoder.backbone.")
        }
        model.encoder.backbone.load_state_dict(sub, strict=True)
        return
    sub = _legacy_backbone_state_dict(state_dict)
    if not sub:
        raise KeyError("No encoder.backbone.* or encoder.layers.* backbone keys found in state_dict")
    model.encoder.backbone.load_state_dict(sub, strict=True)


def freeze_encoder_backbone(model: PredictorTransfer) -> None:
    for p in model.encoder.backbone.parameters():
        p.requires_grad = False


def trainable_parameters(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)
