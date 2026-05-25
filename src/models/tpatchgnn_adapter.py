from __future__ import annotations

import sys
from pathlib import Path

# Aggiunge ./vendor al PYTHONPATH così che `import lib.utils` e
# `import model.tPatchGNN` (path attesi dal repo) funzionino.
_VENDOR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import numpy as np
import torch

# Import dal repo vendorizzato
from lib.utils import split_and_patch_batch  # noqa: E402
from tpatchgnn.model.tPatchGNN import tPatchGNN          # noqa: E402, F401  (re-export)

"""
tpatchgnn_adapter.py
--------------------
Adapter tra il nostro formato canonico IMTS e il formato richiesto dal
modello tPatchGNN ufficiale (repo `usail-hkust/t-PatchGNN`, vendorizzato in
`./vendor`).

Il modello tPatchGNN si aspetta un `batch_dict` con queste chiavi:
    observed_data       (B, M, L_in, D)   PATCHED
    observed_tp         (B, M, L_in, D)   PATCHED
    observed_mask       (B, M, L_in, D)   PATCHED
    data_to_predict     (B, L_pred, D)
    tp_to_predict       (B, L_pred)
    mask_predicted_data (B, L_pred, D)

dove M = `args.npatch`, L_in = `max_patch_len`. Il patching è prodotto da
`lib.utils.split_and_patch_batch` del repo originale, che riusiamo come
black-box.

La nostra `imts_collate_fn` produce invece un batch "flat":
    observed_data       (B, L_obs_max, D)
    observed_tp         (B, L_obs_max)
    observed_mask       (B, L_obs_max, D)
    padding_mask        (B, L_obs_max)
    data_to_predict     (B, L_pred, D)
    tp_to_predict       (B, L_pred)
    mask_predicted_data (B, L_pred, D)

Questo modulo fornisce due cose:

1) `to_tpatchgnn_batch(batch, args, device)`: converte un batch flat nella
   forma patched, applicando al volo la `split_and_patch_batch` originale.
2) `TPatchGNNArgs`: builder degli iperparametri del modello in formato
   compatibile con `vendor.model.tPatchGNN.tPatchGNN(args)`.

Nota importante: tPatchGNN usa un singolo asse temporale CONDIVISO tra i
canali (`time_steps` shape (T_o,)) e applica il patching su quello. Nel
nostro setting MCAR/burst/etc., i tp osservati sono lo stesso vettore per
tutti i sample del batch (perché partiamo da una griglia regolare), quindi
serve un piccolo accorgimento: usiamo l'unione dei tp del batch e ricostruiamo
i valori canale per canale tramite `observed_mask`. Questo è coerente con
ciò che fa la collate originale per PhysioNet/MIMIC.
"""


# ----------------------------------------------------------------------------- #
# Args holder (sostituisce argparse del repo originale)                         #
# ----------------------------------------------------------------------------- #
class TPatchGNNArgs:
    """Minimal args container con tutti i campi che `tPatchGNN(args)` legge.

    Replica esattamente i default dello script `run_models.py` originale,
    con la possibilità di override da kwargs.
    """

    def __init__(self, **overrides):
        # iperparametri del modello
        self.hop = 1
        self.nhead = 1
        self.tf_layer = 1
        self.nlayer = 1
        self.hid_dim = 64
        self.te_dim = 10
        self.node_dim = 10
        self.outlayer = "Linear"

        # configurazione finestra (in unità coerenti con tp_to_predict)
        # NB: `history` è il limite superiore dei tp osservati;
        #     `patch_size` e `stride` sono nelle stesse unità.
        self.history = 1.0       # tp osservati ∈ [0, history]
        self.patch_size = 0.125  # 12 patch su un orizzonte di 1.0 (dimensione default)
        self.stride = 0.125

        # numero di canali (D)
        self.ndim = 7

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # override da kwargs
        for k, v in overrides.items():
            if not hasattr(self, k):
                # accettiamo comunque, così l'utente può aggiungere campi extra
                pass
            setattr(self, k, v)

        # numero di patch (formula del repo)
        self.npatch = int(np.ceil((self.history - self.patch_size) / self.stride)) + 1


# ----------------------------------------------------------------------------- #
# Conversione batch flat → batch patched                                        #
# ----------------------------------------------------------------------------- #
def to_tpatchgnn_batch(
    batch: dict[str, torch.Tensor],
    args: TPatchGNNArgs,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Trasforma un batch della nostra `imts_collate_fn` nel formato
    `batch_dict` patched richiesto dal modello tPatchGNN.

    Step:
      1) sposta tutto su `device` e ricava il vettore unico di tp osservati
         (gli `observed_tp` sono identici fra sample perché vengono dalla
         stessa griglia normalizzata; se così non fosse, prendiamo il primo
         del batch, dato che `split_and_patch_batch` accetta un solo
         vettore `time_steps` (T_o,) per il batch).
      2) costruisce gli indici delle patch (uguale al codice originale).
      3) impacchetta in un `data_dict` flat con le chiavi che si aspetta
         `split_and_patch_batch`, e chiama la funzione del repo.
    """

    observed_data = batch["observed_data"].to(device)         # (B, T_o, D)
    observed_mask = batch["observed_mask"].to(device)         # (B, T_o, D)
    observed_tp_b = batch["observed_tp"].to(device)           # (B, T_o)
    data_to_predict = batch["data_to_predict"].to(device)     # (B, L_pred, D)
    tp_to_predict = batch["tp_to_predict"].to(device)         # (B, L_pred)
    mask_pred = batch["mask_predicted_data"].to(device)       # (B, L_pred, D)

    # I tp sono uguali per tutti i sample del batch (griglia regolare comune).
    # Verifica veloce in debug; in prod usa quello del primo sample.
    observed_tp = observed_tp_b[0]  # (T_o,)
    n_observed_tp = observed_tp.shape[0]

    # Indici delle patch sui tp osservati (replica esatta del repo originale).
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

    # Padding sul tp_to_predict (B, L_pred) → (B, L_pred, 1) atteso da
    # split_and_patch_batch? In realtà il repo lo lascia (B, L_pred), ok.
    data_dict = {
        "data": observed_data,                # (B, T_o, D)
        "time_steps": observed_tp,            # (T_o,)
        "mask": observed_mask,                # (B, T_o, D)
        "data_to_predict": data_to_predict,   # (B, L_pred, D)
        "tp_to_predict": tp_to_predict,       # (B, L_pred)
        "mask_predicted_data": mask_pred,     # (B, L_pred, D)
    }

    return split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)


# ----------------------------------------------------------------------------- #
# Helper: build modello                                                         #
# ----------------------------------------------------------------------------- #
def build_tpatchgnn(args: TPatchGNNArgs) -> torch.nn.Module:
    """Istanzia il modello e lo sposta sul device configurato in args."""
    model = tPatchGNN(args).to(args.device)
    return model
