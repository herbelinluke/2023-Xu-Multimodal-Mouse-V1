"""Per-neuron validation metrics accumulated batch by batch (no full prediction arrays in memory)."""
from __future__ import annotations

import numpy as np
import torch


class StreamingNeuronMetrics:
    """
    Accumulates per-neuron sufficient statistics over validation batches.

    Correlation, R², explained variance and MSE match the earlier per-neuron loops
    (np.corrcoef, sklearn r2_score, 1 - var(res)/var(label)). Bits per spike is the
    Poisson log-likelihood gain over a constant-rate model at each neuron's mean
    validation rate, divided by the neuron's total spike count and ln 2.
    """

    def __init__(self, n_neurons: int, device: torch.device | str = "cpu", eps: float = 1e-12):
        self.n_neurons = int(n_neurons)
        self.device = torch.device(device)
        self.eps = eps
        z = lambda: torch.zeros(self.n_neurons, dtype=torch.float64, device=self.device)
        self.n = 0
        self.sy, self.sp, self.syy, self.spp, self.syp = z(), z(), z(), z(), z()
        self.sylogp = z()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        p = pred.detach().to(self.device, torch.float64).reshape(-1, self.n_neurons)
        y = label.detach().to(self.device, torch.float64).reshape(-1, self.n_neurons)
        self.n += p.shape[0]
        self.sy += y.sum(0)
        self.sp += p.sum(0)
        self.syy += (y * y).sum(0)
        self.spp += (p * p).sum(0)
        self.syp += (y * p).sum(0)
        self.sylogp += (y * torch.log(p.clamp_min(self.eps))).sum(0)

    def compute(self) -> dict:
        n = max(self.n, 1)
        my, mp = self.sy / n, self.sp / n
        var_y = (self.syy / n - my * my).clamp_min(0)
        var_p = (self.spp / n - mp * mp).clamp_min(0)
        cov = self.syp / n - my * mp
        sse = self.syy - 2 * self.syp + self.spp
        valid = (var_y > self.eps) & (var_p > self.eps)

        cor = cov / torch.sqrt(var_y * var_p).clamp_min(self.eps)
        r2 = 1.0 - sse / (var_y * n).clamp_min(self.eps)
        var_res = (sse / n - (my - mp) ** 2).clamp_min(0)
        ev = 1.0 - var_res / var_y.clamp_min(self.eps)
        mse_neuron = sse / n

        ll_gain = self.sylogp - self.sy * torch.log(my.clamp_min(self.eps)) - self.sp + self.sy
        bps = ll_gain / (self.sy.clamp_min(self.eps) * np.log(2.0))
        has_spikes = self.sy > self.eps

        to_np = lambda t: t.cpu().numpy()
        valid_np = to_np(valid)
        cor_n, r2_n, ev_n = (np.where(valid_np, to_np(x), np.nan) for x in (cor, r2, ev))
        bps_n = np.where(valid_np & to_np(has_spikes), to_np(bps), np.nan)

        def mean_or_zero(a):
            return float(np.nanmean(a)) if np.isfinite(a).any() else 0.0

        n_valid = int(valid_np.sum())
        mse_valid = to_np(mse_neuron)[valid_np] if n_valid else to_np(mse_neuron)
        spikes = to_np(self.sy)
        bps_pooled = float(np.nansum(np.where(valid_np, to_np(ll_gain), 0.0)) /
                           max(spikes[valid_np].sum(), self.eps) / np.log(2.0)) if n_valid else 0.0
        return {
            "cor": mean_or_zero(cor_n),
            "r2": mean_or_zero(r2_n),
            "ev": mean_or_zero(ev_n),
            "mse": float(mse_valid.mean()),
            "bits_per_spike": bps_pooled,
            "n_valid": n_valid,
            "n_neurons": self.n_neurons,
            "n_samples": int(self.n),
            "cor_per_neuron": cor_n,
            "r2_per_neuron": r2_n,
            "ev_per_neuron": ev_n,
            "bits_per_spike_per_neuron": bps_n,
            "valid_mask": valid_np,
        }
