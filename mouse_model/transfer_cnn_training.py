"""Training loop for PredictorTransfer with optional frozen CNN backbone (same metrics as mouse CNN notebooks)."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mouse_model.metrics import StreamingNeuronMetrics

from .cnn_predictor_transfer import PredictorTransfer, trainable_parameters


def make_loader(ds, *, batch_size, shuffle, num_workers, collate_fn=None):
    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def count_parameters(model: nn.Module) -> dict:
    return {
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "n_trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def train_cnn_transfer(
    model: PredictorTransfer,
    device: torch.device,
    train_ds,
    val_ds,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    best_train_path: str | None = None,
    best_val_path: str | None = None,
    num_workers: int = 8,
    freeze_encoder_backbone: bool = False,
    patience: int = 5,
    run=None,
):
    """
    Poisson NLL training with validation metrics matching train_cnn_shifter_mouse_vs_sensorium.

    If freeze_encoder_backbone, backbone parameters are frozen and the backbone runs in eval()
    mode during training (BN/Dropout fixed) while the readout (and shifter if enabled) trains.

    run: optional RunLogger. When given, per-epoch metrics, summary and per-neuron arrays are
    written to the run folder, and best_val_path defaults to <run>/best_val.pth.
    Training stops after `patience` + 1 epochs without a new best validation loss.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if run is not None and best_val_path is None:
        best_val_path = run.best_val_path

    train_dataloader = make_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = make_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.Adam(trainable_parameters(model), lr=learning_rate)
    if run is not None:
        run.update_config(n_train=len(train_ds), n_val=len(val_ds), batch_size=batch_size,
                          learning_rate=learning_rate, epochs=epochs, patience=patience, seed=seed,
                          freeze_encoder_backbone=freeze_encoder_backbone, **count_parameters(model))

    best_train_loss = np.inf
    best_val_loss = np.inf
    best_epoch = None
    stop_reason = "max_epochs"
    train_loss_list, val_loss_list = [], []
    val_cor_list, val_r2_list, val_mse_list = [], [], []
    val_poisson_loss_list, val_bits_per_spike_list, val_explained_var_list = [], [], []
    cor_per_neuron_per_epoch, r2_per_neuron_per_epoch, ev_per_neuron_per_epoch = [], [], []
    n_valid_per_epoch = []
    ct = 0
    step = 0
    samples_seen = 0
    t_start = time.time()

    for epoch in range(epochs):
        t_epoch = time.time()
        model.train()
        if freeze_encoder_backbone:
            model.encoder.backbone.eval()

        epoch_train_loss = 0.0
        for (image, behav, spikes) in train_dataloader:
            image = image.to(device, non_blocking=True)
            behav = behav.to(device, non_blocking=True)
            spikes = spikes.to(device, non_blocking=True)
            image = torch.squeeze(image, axis=1)
            pred = model(image, behav)
            loss = nn.functional.poisson_nll_loss(pred, spikes, reduction="mean", log_input=False)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            step += 1
            samples_seen += image.shape[0]

        epoch_train_loss = epoch_train_loss / max(len(train_dataloader), 1)
        train_loss_list.append(epoch_train_loss)

        if epoch_train_loss < best_train_loss:
            best_train_loss = epoch_train_loss
            if best_train_path:
                torch.save(model.state_dict(), best_train_path)
            if len(val_dataloader) == 0 and best_val_path:
                torch.save(model.state_dict(), best_val_path)

        model.eval()
        m = None
        if len(val_dataloader) > 0:
            epoch_val_loss = 0.0
            stream = None
            with torch.no_grad():
                for (image, behav, spikes) in val_dataloader:
                    image = image.to(device, non_blocking=True)
                    behav = behav.to(device, non_blocking=True)
                    spikes = spikes.to(device, non_blocking=True)
                    image = torch.squeeze(image, axis=1)
                    pred = model(image, behav)
                    loss = nn.functional.poisson_nll_loss(pred, spikes, reduction="mean", log_input=False)
                    epoch_val_loss += loss.item()
                    if stream is None:
                        stream = StreamingNeuronMetrics(pred.shape[1], device=device)
                    stream.update(pred, spikes)
            epoch_val_loss = epoch_val_loss / len(val_dataloader)
            m = stream.compute()
            val_cor_list.append(m["cor"])
            val_r2_list.append(m["r2"])
            val_mse_list.append(m["mse"])
            val_poisson_loss_list.append(float(epoch_val_loss))
            val_bits_per_spike_list.append(m["bits_per_spike"])
            val_explained_var_list.append(m["ev"])
            cor_per_neuron_per_epoch.append(m["cor_per_neuron"])
            r2_per_neuron_per_epoch.append(m["r2_per_neuron"])
            ev_per_neuron_per_epoch.append(m["ev_per_neuron"])
            n_valid_per_epoch.append(m["n_valid"])
        else:
            epoch_val_loss = np.inf
            for lst in (val_cor_list, val_r2_list, val_mse_list, val_poisson_loss_list,
                        val_bits_per_spike_list, val_explained_var_list):
                lst.append(float("nan"))
            for lst in (cor_per_neuron_per_epoch, r2_per_neuron_per_epoch, ev_per_neuron_per_epoch,
                        n_valid_per_epoch):
                lst.append(None)

        val_loss_list.append(epoch_val_loss)
        improved = epoch_val_loss < best_val_loss
        if improved:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            ct = 0
            if best_val_path:
                torch.save(model.state_dict(), best_val_path)
            if run is not None and m is not None:
                run.save_per_neuron("best", m, epoch)
        else:
            ct += 1

        epoch_seconds = time.time() - t_epoch
        if run is not None:
            run.log_epoch({
                "epoch": epoch, "step": step, "samples_seen": samples_seen,
                "epoch_seconds": epoch_seconds, "elapsed_seconds": time.time() - t_start,
                "train_loss": epoch_train_loss, "val_loss": epoch_val_loss,
                "val_cor": val_cor_list[-1], "val_r2": val_r2_list[-1], "val_ev": val_explained_var_list[-1],
                "val_mse": val_mse_list[-1], "val_bits_per_spike": val_bits_per_spike_list[-1],
                "n_valid": None if m is None else m["n_valid"], "n_neurons": None if m is None else m["n_neurons"],
                "lr": optimizer.param_groups[0]["lr"],
            })

        if m is not None:
            print(
                "Epoch {} ({:.0f}s) train {:.4f} | val {:.4f} corr {:.4f} R2 {:.4f} EV {:.4f} bits/spike {:.4f} "
                "| valid neurons {}/{}".format(
                    epoch, epoch_seconds, epoch_train_loss, epoch_val_loss, m["cor"], m["r2"], m["ev"],
                    m["bits_per_spike"], m["n_valid"], m["n_neurons"]))
        else:
            print("Epoch {} ({:.0f}s) train {:.4f}".format(epoch, epoch_seconds, epoch_train_loss))

        if len(val_dataloader) > 0 and ct > patience:
            stop_reason = "early_stop"
            print("stop training")
            break

    if run is not None:
        if m is not None:
            run.save_per_neuron("last", m, epoch)
        best = best_epoch if best_epoch is not None else len(val_cor_list) - 1
        run.write_summary({
            "best_epoch": best,
            "selected_by": "val_loss",
            "best_val_loss": best_val_loss,
            "val_cor_at_best": val_cor_list[best] if val_cor_list else None,
            "max_val_cor": float(np.nanmax(val_cor_list)) if val_cor_list else None,
            "epochs_run": len(train_loss_list),
            "steps": step,
            "samples_seen": samples_seen,
            "stop_reason": stop_reason,
            "total_seconds": time.time() - t_start,
        })

    return (
        train_loss_list,
        val_loss_list,
        val_cor_list,
        val_r2_list,
        val_mse_list,
        val_poisson_loss_list,
        val_bits_per_spike_list,
        val_explained_var_list,
        cor_per_neuron_per_epoch,
        r2_per_neuron_per_epoch,
        ev_per_neuron_per_epoch,
        n_valid_per_epoch,
    )
