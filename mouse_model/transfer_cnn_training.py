"""Training loop for PredictorTransfer with optional frozen CNN backbone (same metrics as mouse CNN notebooks)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from mouse_model.evaluation import cor_in_time

from .cnn_predictor_transfer import PredictorTransfer, trainable_parameters


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
    best_train_path: str | None,
    best_val_path: str | None,
    num_workers: int = 8,
    freeze_encoder_backbone: bool = False,
):
    """
    Poisson NLL training with validation metrics matching train_cnn_shifter_mouse_vs_sensorium.

    If freeze_encoder_backbone, backbone parameters are frozen and the backbone runs in eval()
    mode during training (BN/Dropout fixed) while the readout (and shifter if enabled) trains.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_dataloader = DataLoader(
        dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    optimizer = torch.optim.Adam(trainable_parameters(model), lr=learning_rate)

    best_train_loss = np.inf
    best_val_loss = np.inf
    train_loss_list = []
    val_loss_list = []
    val_cor_list = []
    val_r2_list = []
    val_mse_list = []
    val_poisson_loss_list = []
    val_bits_per_spike_list = []
    val_explained_var_list = []
    cor_per_neuron_per_epoch = []
    r2_per_neuron_per_epoch = []
    ev_per_neuron_per_epoch = []
    n_valid_per_epoch = []
    ct = 0

    for epoch in range(epochs):
        print("Start epoch", epoch)
        model.train()
        if freeze_encoder_backbone:
            model.encoder.backbone.eval()

        epoch_train_loss = 0.0
        for (image, behav, spikes) in train_dataloader:
            image, behav, spikes = image.to(device), behav.to(device), spikes.to(device)
            image = torch.squeeze(image, axis=1)
            pred = model(image, behav)
            loss = nn.functional.poisson_nll_loss(pred, spikes, reduction="mean", log_input=False)
            epoch_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if freeze_encoder_backbone:
                model.encoder.backbone.eval()

        epoch_train_loss = epoch_train_loss / len(train_dataloader)
        train_loss_list.append(epoch_train_loss)

        if best_train_path and epoch_train_loss < best_train_loss:
            torch.save(model.state_dict(), best_train_path)
            best_train_loss = epoch_train_loss
            if len(val_dataloader) == 0 and best_val_path:
                torch.save(model.state_dict(), best_val_path)

        print("Epoch {} train loss: {:.4f}".format(epoch, epoch_train_loss))

        model.eval()
        epoch_val_loss = 0.0
        pred_val_all = []
        label_val_all = []
        if len(val_dataloader) > 0:
            with torch.no_grad():
                for (image, behav, spikes) in val_dataloader:
                    image, behav, spikes = image.to(device), behav.to(device), spikes.to(device)
                    image = torch.squeeze(image, axis=1)
                    pred = model(image, behav)
                    loss = nn.functional.poisson_nll_loss(pred, spikes, reduction="mean", log_input=False)
                    epoch_val_loss += loss.item()
                    pred_val_all.append(pred.cpu().numpy())
                    label_val_all.append(spikes.cpu().numpy())
            epoch_val_loss = epoch_val_loss / len(val_dataloader)
            pred_val = np.concatenate(pred_val_all, axis=0)
            label_val = np.concatenate(label_val_all, axis=0)
            var_pred = np.var(pred_val, axis=0)
            var_label = np.var(label_val, axis=0)
            eps = 1e-12
            valid = (var_pred > eps) & (var_label > eps)
            n_valid = int(valid.sum())
            num_neurons = pred_val.shape[1]
            cor_array = cor_in_time(pred_val, label_val)
            cor_per_neuron = np.array(cor_array.flatten(), dtype=np.float64)
            cor_per_neuron[~valid] = np.nan
            mean_cor = np.nanmean(cor_per_neuron)
            if np.isnan(mean_cor):
                mean_cor = 0.0
            val_cor_list.append(mean_cor)
            cor_per_neuron_per_epoch.append(cor_per_neuron.copy())
            n_valid_per_epoch.append(n_valid)
            r2_per_neuron = np.array(
                [r2_score(label_val[:, j], pred_val[:, j]) for j in range(num_neurons)],
                dtype=np.float64,
            )
            r2_per_neuron[~valid] = np.nan
            mean_r2 = np.nanmean(r2_per_neuron)
            val_r2_list.append(float(mean_r2) if not np.isnan(mean_r2) else 0.0)
            r2_per_neuron_per_epoch.append(r2_per_neuron.copy())
            if n_valid > 0:
                mse = mean_squared_error(label_val[:, valid], pred_val[:, valid])
            else:
                mse = mean_squared_error(label_val, pred_val)
            val_mse_list.append(float(mse))
            val_poisson_loss_list.append(float(epoch_val_loss))
            val_bits_per_spike_list.append(float(epoch_val_loss / np.log(2)))
            res = label_val - pred_val
            var_res = np.var(res, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ev_per_neuron = np.where(var_label > eps, 1.0 - var_res / var_label, np.nan).astype(
                    np.float64
                )
            ev_per_neuron[~valid] = np.nan
            mean_ev = np.nanmean(ev_per_neuron)
            val_explained_var_list.append(float(mean_ev) if not np.isnan(mean_ev) else 0.0)
            ev_per_neuron_per_epoch.append(ev_per_neuron.copy())
        else:
            epoch_val_loss = np.inf
            val_cor_list.append(float("nan"))
            val_r2_list.append(float("nan"))
            val_mse_list.append(float("nan"))
            val_poisson_loss_list.append(float("nan"))
            val_bits_per_spike_list.append(float("nan"))
            val_explained_var_list.append(float("nan"))
            cor_per_neuron_per_epoch.append(None)
            r2_per_neuron_per_epoch.append(None)
            ev_per_neuron_per_epoch.append(None)
            n_valid_per_epoch.append(None)

        val_loss_list.append(epoch_val_loss)
        if best_val_path and epoch_val_loss < best_val_loss:
            torch.save(model.state_dict(), best_val_path)
            best_val_loss = epoch_val_loss
            ct = 0
        else:
            ct += 1
            if len(val_dataloader) > 0 and ct > 5:
                print("stop training")
                break

        if len(val_dataloader) > 0:
            print(
                "Epoch {} val loss: {:.4f} | corr: {:.4f} R2: {:.4f} MSE: {:.4f} EV: {:.4f} | valid neurons: {} / {}".format(
                    epoch,
                    epoch_val_loss,
                    mean_cor,
                    val_r2_list[-1],
                    val_mse_list[-1],
                    val_explained_var_list[-1],
                    n_valid,
                    num_neurons,
                )
            )
        else:
            print("Epoch {} val loss: {}".format(epoch, epoch_val_loss))
        print("End epoch", epoch)

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
