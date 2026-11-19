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
    model: PredictorTransfer | "PredictorTransferMultiReadout",
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


def freeze_encoder_backbone(model: PredictorTransfer | "PredictorTransferMultiReadout") -> None:
    for p in model.encoder.backbone.parameters():
        p.requires_grad = False


def trainable_parameters(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)


class PredictorTransferMultiReadout(nn.Module):
    """
    Shared CNN backbone + shifter (optional) + one linear readout per session key.
    Session keys are strings (e.g. "0","1","2"). Forward pads outputs to max_neurons
    so batches can mix sessions; use `neuron_mask` for masked loss/metrics.
    """

    def __init__(self, readout_dims: dict[str, int], use_shifter: bool = False):
        super().__init__()
        if not readout_dims:
            raise ValueError("readout_dims must be non-empty")
        self.session_keys = sorted(readout_dims.keys(), key=lambda k: int(k))
        self.readout_dims = {k: int(readout_dims[k]) for k in self.session_keys}
        self.max_neurons = max(self.readout_dims.values())
        self.key_to_sid = {k: i for i, k in enumerate(self.session_keys)}
        self.sid_to_key = {i: k for k, i in self.key_to_sid.items()}

        self.encoder = VisualEncoderTransfer(output_dim=1)
        flat_dim = self.encoder.readout.in_features
        for p in self.encoder.readout.parameters():
            p.requires_grad = False
        self.readouts = nn.ModuleDict(
            {k: nn.Linear(flat_dim, self.readout_dims[k]) for k in self.session_keys}
        )

        self.softplus = nn.Softplus()
        self.shifter = Shifter()
        self.use_shifter = use_shifter

    def sid_tensor_to_keys(self, session_ids: torch.Tensor) -> list[str]:
        return [self.sid_to_key[int(s.item())] for s in session_ids.reshape(-1)]

    def neuron_mask(self, session_ids: torch.Tensor) -> torch.Tensor:
        """[B, max_neurons] mask of valid output neurons per sample."""
        dims = torch.tensor(
            [self.readout_dims[self.sid_to_key[i]] for i in range(len(self.session_keys))],
            device=session_ids.device,
        )
        n = dims[session_ids.reshape(-1).long()]
        cols = torch.arange(self.max_neurons, device=session_ids.device)
        return (cols[None, :] < n[:, None]).to(torch.float32)

    def forward(self, images: torch.Tensor, behav: torch.Tensor, session_ids: torch.Tensor):
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

        feats = self.encoder.backbone(images)
        b = feats.shape[0]
        device = feats.device
        out = torch.zeros(b, self.max_neurons, device=device, dtype=feats.dtype)
        for sid_int in torch.unique(session_ids).tolist():
            sid_int = int(sid_int)
            key = self.sid_to_key[sid_int]
            n = self.readout_dims[key]
            m = session_ids.reshape(-1) == sid_int
            if m.any():
                out[m, :n] = self.readouts[key](feats[m])
        return self.softplus(out), self.neuron_mask(session_ids)
