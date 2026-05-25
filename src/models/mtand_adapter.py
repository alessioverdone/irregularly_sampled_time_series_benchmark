"""
mtand_adapter.py
----------------
Adapter tra il nostro formato canonico IMTS e il modello mTAND (Multi-Time
Attention Network, Shukla & Marlin 2021), repo `reml-lab/mTAN`, vendorizzato
in `./vendor/mtand`.

Differenza chiave rispetto a tPatchGNN/Hi-Patch: mTAND **non usa patching**.
Lavora direttamente su:
    values     (B, L_obs, D)
    mask       (B, L_obs, D)
    obs_tp     (B, L_obs)            timestamp osservati
    pred_tp    (B, L_pred)           timestamp da predire

L'encoder mTAND `enc_mtan_rnn` mappa la sequenza irregolare su un set di
reference query points (una griglia regolare di tp interna al modello),
producendo un latent `(B, n_ref, latent_dim)` o `(B, n_ref, 2*latent_dim)`
nella variante VAE.

Il decoder mTAND `dec_mtan_rnn` prende il latent e i `pred_tp` e produce
predizioni `(B, L_pred, D)`.

Per fare forecasting (e non interpolation/classification come nei task
originali), avvolgiamo encoder + decoder in un singolo `nn.Module`
`MTANDForecaster` con:
  - `forward(observed_data, observed_mask, observed_tp, tp_to_predict)`
    che ritorna `pred (B, L_pred, D)`;
  - `forecasting(...)` con la stessa signature di tPatchGNN/Hi-Patch
    per coerenza, ma adattata: niente patching.

Versione DETERMINISTICA: l'encoder produce `2*latent_dim` (mean + logvar
per il VAE originale), ma noi scartiamo `logvar` e usiamo solo `mean` come
latent diretto. Questa è la versione standard usata come baseline
forecaster nei paper IMTS recenti (tPatchGNN, Hi-Patch, IMTS-Mixer).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --- Import del modulo mTAND vendorizzato ----------------------------------- #
_VENDOR = Path(__file__).resolve().parent / "vendor" / "mtand"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import mtand.models as mtand_models  # noqa: E402


# ----------------------------------------------------------------------------- #
# Args holder                                                                   #
# ----------------------------------------------------------------------------- #
class MTANDArgs:
    """Container args per `MTANDForecaster`."""

    def __init__(self, **overrides):
        # Iperparametri
        self.ndim = 7                  # numero canali (D)
        self.latent_dim = 16           # dimensione latente (deterministica)
        self.nhidden = 32              # hidden size GRU + attention
        self.embed_time = 64           # dimensione time embedding (deve essere divisibile per num_heads)
        self.num_heads = 1
        self.learn_emb = True          # True = time embedding learnable, False = sinusoidal fisso
        self.n_ref_points = 128        # numero di reference query points (asse interno dell'encoder)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for k, v in overrides.items():
            setattr(self, k, v)


# ----------------------------------------------------------------------------- #
# Wrapper: MTANDForecaster                                                      #
# ----------------------------------------------------------------------------- #
class MTANDForecaster(nn.Module):
    """Wrapper deterministico che usa enc_mtan_rnn + dec_mtan_rnn per
    forecasting puntuale (no VAE).

    Pipeline:
        1) concatena (values, mask) lungo last dim → (B, L_obs, 2*D)
        2) enc_mtan_rnn(input, obs_tp) → (B, n_ref, 2*latent_dim)
        3) prende solo i primi latent_dim (mean del VAE) → (B, n_ref, latent_dim)
        4) dec_mtan_rnn(latent, pred_tp) → (B, L_pred, D)
    """

    def __init__(self, args: MTANDArgs):
        super().__init__()
        self.args = args
        self.dim = args.ndim
        self.latent_dim = args.latent_dim

        # Reference query points (griglia uniforme su [0, 1]). Sono i punti
        # interni su cui l'encoder mTAND fa attention per costruire il latent.
        # NB: nel repo originale `query` è passato come tensor al costruttore.
        ref = torch.linspace(0, 1, args.n_ref_points)
        self.register_buffer("query_ref", ref)

        # Encoder: produce 2*latent_dim (mean + logvar nella variante VAE)
        self.encoder = mtand_models.enc_mtan_rnn(
            input_dim=args.ndim,
            query=self.query_ref,
            latent_dim=args.latent_dim,
            nhidden=args.nhidden,
            embed_time=args.embed_time,
            num_heads=args.num_heads,
            learn_emb=args.learn_emb,
            device=str(args.device),
        )
        # Decoder: prende latent (latent_dim) e ritorna predizioni su pred_tp
        self.decoder = mtand_models.dec_mtan_rnn(
            input_dim=args.ndim,
            query=self.query_ref,
            latent_dim=args.latent_dim,
            nhidden=args.nhidden,
            embed_time=args.embed_time,
            num_heads=args.num_heads,
            learn_emb=args.learn_emb,
            device=str(args.device),
        )

    # --- API forecasting (stessa signature di tPatchGNN/Hi-Patch) -------------#
    def forecasting(
        self,
        time_steps_to_predict: torch.Tensor,   # (B, L_pred)
        X: torch.Tensor,                        # (B, L_obs, D)
        truth_time_steps: torch.Tensor,         # (B, L_obs)  -- 1D per sample
        mask: torch.Tensor,                     # (B, L_obs, D)
    ) -> torch.Tensor:
        """Ritorna pred di shape (1, B, L_pred, D) — il `1` finale serve per
        compatibilità con le metriche `compute_error` di tPatchGNN/Hi-Patch
        che accettano `[n_traj_samples, B, L, D]`."""

        # Encoder vuole (values || mask) di shape (B, L_obs, 2*D)
        enc_input = torch.cat([X, mask], dim=-1)
        # truth_time_steps deve essere (B, L_obs) — già nel formato giusto
        enc_out = self.encoder(enc_input, truth_time_steps)  # (B, n_ref, 2*latent_dim)

        # Versione deterministica: prendo solo la "media"
        latent = enc_out[:, :, : self.latent_dim]            # (B, n_ref, latent_dim)

        # Decoder: latent + pred_tp → (B, L_pred, D)
        pred = self.decoder(latent, time_steps_to_predict)   # (B, L_pred, D)

        return pred.unsqueeze(0)  # (1, B, L_pred, D)


# ----------------------------------------------------------------------------- #
# Conversione batch flat → input mTAND                                          #
# ----------------------------------------------------------------------------- #
def to_mtand_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Per mTAND non c'è patching. Sposta tutto su device e lascia le shape
    della nostra `imts_collate_fn` invariate. Fa un solo cambiamento:
    `observed_tp` viene preso 1D per sample (è (B, L_obs) ed è quello che
    vuole l'encoder)."""

    return {
        "observed_data":       batch["observed_data"].to(device),        # (B, L_obs, D)
        "observed_tp":         batch["observed_tp"].to(device),          # (B, L_obs)
        "observed_mask":       batch["observed_mask"].to(device),        # (B, L_obs, D)
        "data_to_predict":     batch["data_to_predict"].to(device),      # (B, L_pred, D)
        "tp_to_predict":       batch["tp_to_predict"].to(device),        # (B, L_pred)
        "mask_predicted_data": batch["mask_predicted_data"].to(device),  # (B, L_pred, D)
    }


# ----------------------------------------------------------------------------- #
# Helper: build modello                                                         #
# ----------------------------------------------------------------------------- #
def build_mtand(args: MTANDArgs) -> MTANDForecaster:
    """Istanzia MTANDForecaster e lo sposta sul device configurato."""
    model = MTANDForecaster(args).to(args.device)
    return model


# ----------------------------------------------------------------------------- #
# Loss e metriche (replica della logica compute_error di tPatchGNN)             #
# ----------------------------------------------------------------------------- #
def compute_error(
    truth: torch.Tensor,
    pred_y: torch.Tensor,
    mask: torch.Tensor,
    func: str,
    reduce: str,
):
    """Versione locale di `compute_error` (stessa logica di tPatchGNN/Hi-Patch).

    Args:
        truth  : (B, L, D)
        pred_y : (1, B, L, D)  — n_traj_samples=1 dimension prepended dal forecaster
        mask   : (B, L, D)
        func   : "MSE" | "MAE"
        reduce : "mean" | "sum"
    """
    if pred_y.dim() == 3:
        pred_y = pred_y.unsqueeze(0)
    n_traj, B, L, D = pred_y.shape
    truth_rep = truth.repeat(n_traj, 1, 1, 1)
    mask_rep = mask.repeat(n_traj, 1, 1, 1)

    if func == "MSE":
        err = ((truth_rep - pred_y) ** 2) * mask_rep
    elif func == "MAE":
        err = torch.abs(truth_rep - pred_y) * mask_rep
    else:
        raise ValueError(f"func sconosciuta: {func}")

    err_var_sum = err.reshape(-1, D).sum(dim=0)            # (D,)
    mask_count  = mask_rep.reshape(-1, D).sum(dim=0)       # (D,)

    if reduce == "mean":
        err_var_avg = err_var_sum / (mask_count + 1e-8)
        n_avai = torch.count_nonzero(mask_count)
        return err_var_avg.sum() / n_avai
    elif reduce == "sum":
        return err_var_sum, mask_count
    else:
        raise ValueError(f"reduce sconosciuto: {reduce}")
