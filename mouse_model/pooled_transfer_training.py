"""Training loop for PredictorTransferMultiReadout (pooled mice or pooled Sensorium sessions)."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from mouse_model.metrics import StreamingNeuronMetrics

from .cnn_predictor_transfer import PredictorTransferMultiReadout, trainable_parameters
from .transfer_cnn_training import count_parameters, make_loader


def masked_poisson_nll(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Elementwise Poisson NLL with mask; mean over valid elements only."""
    loss = nn.functional.poisson_nll_loss(pred, target, reduction="none", log_input=False)
    m = mask.to(loss.dtype)
    denom = torch.clamp(m.sum(), min=1.0)
    return (loss * m).sum() / denom


def train_cnn_transfer_multi(
    model: PredictorTransferMultiReadout,
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
    collate_fn=None,
    patience: int = 5,
    min_samples_per_epoch: int | None = None,
    select_by: str = "val_loss",
    run=None,
):
    """
    Same return tuple shape as train_cnn_transfer; val correlation is the macro mean
    of per-session mean correlations (each session evaluated on its true neuron count).
    Per-neuron entries are lists with one array per session.
    select_by: "val_loss" or "val_cor" (macro correlation) picks the checkpoint and early stop.
    """
    if select_by not in ("val_loss", "val_cor"):
        raise ValueError("select_by must be 'val_loss' or 'val_cor'")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if run is not None and best_val_path is None:
        best_val_path = run.best_val_path

    train_dataloader = make_loader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, collate_fn=collate_fn,
                                   min_samples_per_epoch=min_samples_per_epoch)
    val_dataloader = make_loader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(trainable_parameters(model), lr=learning_rate)
    if run is not None:
        run.update_config(n_train=len(train_ds), n_val=len(val_ds), batch_size=batch_size,
                          learning_rate=learning_rate, epochs=epochs, patience=patience, seed=seed,
                          freeze_encoder_backbone=freeze_encoder_backbone,
                          min_samples_per_epoch=min_samples_per_epoch, select_by=select_by,
                          readout_dims=model.readout_dims, **count_parameters(model))

    best_train_loss = np.inf
    best_val_loss = np.inf
    best_score = -np.inf
    best_epoch = None
    stop_reason = "max_epochs"
    train_loss_list, val_loss_list = [], []
    val_cor_list, val_r2_list, val_mse_list = [], [], []
    val_poisson_loss_list, val_bits_per_spike_list, val_explained_var_list = [], [], []
    cor_per_neuron_per_epoch, r2_per_neuron_per_epoch, ev_per_neuron_per_epoch = [], [], []
    n_valid_per_epoch = []
    n_sessions = len(model.session_keys)
    best_session_cor = None
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
        for image, behav, spikes, session_ids in train_dataloader:
            image = image.to(device, non_blocking=True)
            behav = behav.to(device, non_blocking=True)
            spikes = spikes.to(device, non_blocking=True)
            session_ids = session_ids.to(device, non_blocking=True)
            image = torch.squeeze(image, axis=1)
            pred, mask = model(image, behav, session_ids)
            loss = masked_poisson_nll(pred, spikes, mask)
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
        session_metrics = None
        if len(val_dataloader) > 0:
            epoch_val_loss = 0.0
            streams = {
                sid: StreamingNeuronMetrics(model.readout_dims[model.sid_to_key[sid]], device=device)
                for sid in range(n_sessions)
            }
            with torch.no_grad():
                for image, behav, spikes, session_ids in val_dataloader:
                    image = image.to(device, non_blocking=True)
                    behav = behav.to(device, non_blocking=True)
                    spikes = spikes.to(device, non_blocking=True)
                    session_ids = session_ids.to(device, non_blocking=True)
                    image = torch.squeeze(image, axis=1)
                    pred, mask = model(image, behav, session_ids)
                    epoch_val_loss += masked_poisson_nll(pred, spikes, mask).item()
                    sid_flat = session_ids.reshape(-1)
                    for sid in torch.unique(sid_flat).tolist():
                        sel = sid_flat == sid
                        n = streams[sid].n_neurons
                        streams[sid].update(pred[sel, :n], spikes[sel, :n])
            epoch_val_loss = epoch_val_loss / len(val_dataloader)

            session_metrics = {sid: s.compute() for sid, s in streams.items() if s.n > 0}
            ms = list(session_metrics.values())
            mses = [x["mse"] for x in ms if np.isfinite(x["mse"])]
            val_cor_list.append(float(np.mean([x["cor"] for x in ms])) if ms else 0.0)
            val_r2_list.append(float(np.mean([x["r2"] for x in ms])) if ms else 0.0)
            val_explained_var_list.append(float(np.mean([x["ev"] for x in ms])) if ms else 0.0)
            val_mse_list.append(float(np.mean(mses)) if mses else float("nan"))
            val_bits_per_spike_list.append(float(np.mean([x["bits_per_spike"] for x in ms])) if ms else 0.0)
            val_poisson_loss_list.append(float(epoch_val_loss))
            cor_per_neuron_per_epoch.append([x["cor_per_neuron"] for x in ms])
            r2_per_neuron_per_epoch.append([x["r2_per_neuron"] for x in ms])
            ev_per_neuron_per_epoch.append([x["ev_per_neuron"] for x in ms])
            n_valid_per_epoch.append(int(sum(x["n_valid"] for x in ms)))
        else:
            epoch_val_loss = np.inf
            for lst in (val_cor_list, val_r2_list, val_mse_list, val_poisson_loss_list,
                        val_bits_per_spike_list, val_explained_var_list):
                lst.append(float("nan"))
            for lst in (cor_per_neuron_per_epoch, r2_per_neuron_per_epoch, ev_per_neuron_per_epoch,
                        n_valid_per_epoch):
                lst.append(None)

        val_loss_list.append(epoch_val_loss)
        best_val_loss = min(best_val_loss, epoch_val_loss)
        score = -epoch_val_loss if select_by == "val_loss" else val_cor_list[-1]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            ct = 0
            if session_metrics:
                best_session_cor = {model.sid_to_key[sid]: sm["cor"] for sid, sm in session_metrics.items()}
            if best_val_path:
                torch.save(model.state_dict(), best_val_path)
            if run is not None and session_metrics:
                for sid, sm in session_metrics.items():
                    run.save_per_neuron(f"best_session{sid}", sm, epoch)
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
                "n_valid": n_valid_per_epoch[-1],
                "n_neurons": int(sum(model.readout_dims.values())),
                "lr": optimizer.param_groups[0]["lr"],
            })

        if session_metrics:
            per_s = " ".join(f"s{sid}={sm['cor']:.3f}" for sid, sm in session_metrics.items())
            print("Epoch {} ({:.0f}s) train {:.4f} | val {:.4f} corr(macro) {:.4f} [{}] bits/spike {:.4f}".format(
                epoch, epoch_seconds, epoch_train_loss, epoch_val_loss, val_cor_list[-1], per_s,
                val_bits_per_spike_list[-1]))
        else:
            print("Epoch {} ({:.0f}s) train {:.4f}".format(epoch, epoch_seconds, epoch_train_loss))

        if len(val_dataloader) > 0 and ct > patience:
            stop_reason = "early_stop"
            print("stop training")
            break

    if run is not None:
        if session_metrics:
            for sid, sm in session_metrics.items():
                run.save_per_neuron(f"last_session{sid}", sm, epoch)
        best = best_epoch if best_epoch is not None else len(val_cor_list) - 1
        run.write_summary({
            "best_epoch": best,
            "selected_by": select_by,
            "best_val_loss": best_val_loss,
            "val_loss_at_best": val_loss_list[best] if val_loss_list else None,
            "val_cor_at_best": val_cor_list[best] if val_cor_list else None,
            "val_cor_per_session_at_best": best_session_cor,
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
