"""
hipatch_adapter.py
------------------
Adapter tra il nostro formato canonico IMTS e il modello Hi-Patch
(repo `qianlima-lab/Hi-Patch`, vendorizzato in `./vendor/hipatch`).

Hi-Patch espone la stessa identica API di tPatchGNN per il forecasting:

    model.forecasting(
        time_steps_to_predict,   # (B, L_pred)
        X,                       # (B, M, L_in, D)   PATCHED
        truth_time_steps,        # (B, M, L_in, D)   PATCHED
        mask=...,                # (B, M, L_in, D)   PATCHED
    )

e il `compute_all_losses(model, batch_dict)` usa le stesse chiavi:
    observed_data, observed_tp, observed_mask, data_to_predict,
    tp_to_predict, mask_predicted_data.

Quindi possiamo riusare lo stesso patching: la funzione `split_and_patch_batch`
del repo Hi-Patch è funzionalmente identica a quella di tPatchGNN (a meno
di commenti), e il nostro flusso di conversione "flat → patched" funziona
senza modifiche di sostanza.

Differenze rispetto a tPatchGNN:
  - Hi-Patch ha iperparametri extra: `alpha`, `res`, `patch_layer`.
  - `patch_layer` non è libero: si calcola da `npatch` con la funzione
    `layer_of_patches` definita nel training script originale.
  - Hi-Patch richiede `args.task = 'forecasting'` per inizializzare il decoder.
  - Dipendenze extra: torch_geometric, torch_scatter (PyG).

NOTA SU PYTHONPATH: i due repo (tPatchGNN e Hi-Patch) usano entrambi i nomi
di pacchetto top-level `lib` e `model`. Per evitare collisioni se l'utente
importa entrambi gli adapter nello stesso processo, qui inseriamo
`vendor/hipatch` *in testa* al sys.path solo durante l'import di Hi-Patch
e poi lo rimuoviamo, salvando i moduli in cache con nomi univoci.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


# --- Import del vendor di Hi-Patch ------------------------------------------- #
# IMPORTANTE: i due repo (tPatchGNN e Hi-Patch) usano entrambi i nomi
# top-level `lib` e `model`. Per evitare collisioni, gli script di training
# importano UN SOLO modello per processo (`train_tpatchgnn.py` o
# `train_hipatch.py`). Questo modulo aggiunge `vendor/hipatch` in cima al
# sys.path e lascia i moduli `lib`/`model` registrati col loro nome
# originale — necessario perché torch_geometric ispeziona le type
# annotations di `Intra_Inter_Patch_Graph_Layer.message` cercando il modulo
# `model.hipatch` per nome in sys.modules.
#
# Se davvero vuoi import simultaneo dei due, esegui i due training in
# subprocess separati.
_VENDOR_HIPATCH = Path(__file__).resolve().parent / "vendor" / "hipatch"
if str(_VENDOR_HIPATCH) not in sys.path:
    sys.path.insert(0, str(_VENDOR_HIPATCH))

import lib.utils as _hipatch_utils          # noqa: E402
import lib.evaluation as _hipatch_eval      # noqa: E402
from model import hipatch as _hipatch_model_mod  # noqa: E402

split_and_patch_batch = _hipatch_utils.split_and_patch_batch
compute_error = _hipatch_eval.compute_error
Hi_Patch = _hipatch_model_mod.Hi_Patch


# ----------------------------------------------------------------------------- #
# Args holder per Hi-Patch                                                      #
# ----------------------------------------------------------------------------- #
def _layer_of_patches(n_patch: int) -> int:
    """Replica la funzione `layer_of_patches` del training script di Hi-Patch.
    Calcola il numero di layer gerarchici dato il numero di patch."""
    if n_patch <= 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + _layer_of_patches(n_patch // 2)
    return _layer_of_patches(n_patch + 1)


class HiPatchArgs:
    """Container args compatibile con `Hi_Patch(args)`."""

    def __init__(self, **overrides):
        # Iperparametri base
        self.hid_dim = 64
        self.nhead = 1
        self.nlayer = 1
        self.alpha = 1.0
        self.res = 1.0
        self.task = "forecasting"

        # Configurazione finestra (in unità di tp normalizzati)
        self.history = 1.0
        self.pred_window = 1.0
        self.patch_size = 0.125
        self.stride = 0.125

        # Numero canali
        self.ndim = 7

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Override
        for k, v in overrides.items():
            setattr(self, k, v)

        # Calcolati
        self.npatch = int(np.ceil((self.history - self.patch_size) / self.stride)) + 1
        self.patch_layer = _layer_of_patches(self.npatch)
        self.scale_patch_size = self.patch_size / (self.history + self.pred_window)


# ----------------------------------------------------------------------------- #
# Conversione batch flat → batch patched                                        #
# ----------------------------------------------------------------------------- #
def to_hipatch_batch(
    batch: dict[str, torch.Tensor],
    args: HiPatchArgs,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Converte un batch flat nel formato patched richiesto da Hi-Patch."""

    observed_data = batch["observed_data"].to(device)
    observed_mask = batch["observed_mask"].to(device)
    observed_tp_b = batch["observed_tp"].to(device)
    data_to_predict = batch["data_to_predict"].to(device)
    tp_to_predict = batch["tp_to_predict"].to(device)
    mask_pred = batch["mask_predicted_data"].to(device)

    observed_tp = observed_tp_b[0]
    n_observed_tp = observed_tp.shape[0]

    patch_indices = []
    st, ed = 0.0, args.patch_size
    for i in range(args.npatch):
        if i == args.npatch - 1:
            inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
        else:
            inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
        patch_indices.append(inds)
        st += args.stride
        ed += args.stride

    data_dict = {
        "data": observed_data,
        "time_steps": observed_tp,
        "mask": observed_mask,
        "data_to_predict": data_to_predict,
        "tp_to_predict": tp_to_predict,
        "mask_predicted_data": mask_pred,
    }
    return split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)


# ----------------------------------------------------------------------------- #
# Helper: build modello                                                         #
# ----------------------------------------------------------------------------- #
def build_hipatch(args: HiPatchArgs) -> torch.nn.Module:
    """Istanzia Hi-Patch e lo sposta sul device configurato."""
    return Hi_Patch(args).to(args.device)
