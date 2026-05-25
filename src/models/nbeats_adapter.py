"""
nbeats_adapter.py
-----------------
N-BEATS: Neural Basis Expansion Analysis for interpretable Time Series Forecasting.
"N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting"
(Oreshkin et al., 2020)

Implementa la variante **generic** (basi apprese, non interpretabili), che è la
formulazione più generale e di solito la più competitiva su task di forecasting
multivariato.

Architettura:
    Ogni *blocco* riceve il residuo dell'ingresso, lo proietta tramite FC layers
    condivisi e produce un backcast (ricostruzione del passato) e un forecast
    (previsione del futuro).  Il residuo viene aggiornato sottraendo il backcast;
    il forecast totale è la somma dei forecast dei singoli blocchi (doubly residual).

    Ogni feature (canale) viene elaborata indipendentemente dagli stessi pesi:
    il batch viene reshapato in (B*D, T) prima delle FC e ripristinato dopo.

Interfaccia:
    forward(x: Tensor[B, seq_len, D]) -> Tensor[B, pred_len, D]
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- #
# Blocco elementare                                                             #
# ----------------------------------------------------------------------------- #
class _NBEATSBlock(nn.Module):
    """Un singolo blocco N-BEATS (generic basis).

    Args:
        seq_len              : lunghezza finestra di input.
        pred_len             : orizzonte di previsione.
        layer_width          : larghezza dei layer FC.
        n_fc_layers          : numero di layer FC (con ReLU).
        expansion_coeff_dim  : dimensione dei coefficienti di espansione θ.
        dropout              : dropout applicato dopo le FC.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        layer_width: int,
        n_fc_layers: int,
        expansion_coeff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Stack di FC condivisi tra tutti i canali
        layers: list[nn.Module] = []
        in_size = seq_len
        for _ in range(n_fc_layers):
            layers += [nn.Linear(in_size, layer_width), nn.ReLU()]
            in_size = layer_width
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*layers)

        # Proiezioni θ → backcast / forecast tramite basi apprese
        self.theta_backcast = nn.Linear(layer_width, expansion_coeff_dim, bias=False)
        self.theta_forecast = nn.Linear(layer_width, expansion_coeff_dim, bias=False)
        self.backcast_basis = nn.Linear(expansion_coeff_dim, seq_len, bias=False)
        self.forecast_basis = nn.Linear(expansion_coeff_dim, pred_len, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, seq_len, D)
        B, T, D = x.shape
        h = x.reshape(B * D, T)             # (B*D, seq_len) — canali indipendenti
        h = self.fc(h)                       # (B*D, layer_width)

        backcast = self.backcast_basis(self.theta_backcast(h))    # (B*D, seq_len)
        forecast = self.forecast_basis(self.theta_forecast(h))    # (B*D, pred_len)

        backcast = backcast.reshape(B, T, D)
        forecast = forecast.reshape(B, self.pred_len, D)
        return backcast, forecast


# ----------------------------------------------------------------------------- #
# Modello completo                                                              #
# ----------------------------------------------------------------------------- #
class NBEATSModule(nn.Module):
    """N-BEATS generic (basi apprese).

    Args:
        seq_len             : lunghezza finestra di input.
        pred_len            : orizzonte di previsione.
        n_features          : numero di feature (canali D).
        n_stacks            : numero di blocchi impilati in serie.
        layer_width         : larghezza di ogni layer FC interno.
        n_fc_layers         : numero di layer FC per blocco.
        expansion_coeff_dim : dimensione dei coefficienti θ (backcast e forecast).
        dropout             : dropout sulle FC (0 = disabilitato).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        n_stacks: int = 30,
        layer_width: int = 256,
        n_fc_layers: int = 4,
        expansion_coeff_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.blocks = nn.ModuleList([
            _NBEATSBlock(
                seq_len=seq_len,
                pred_len=pred_len,
                layer_width=layer_width,
                n_fc_layers=n_fc_layers,
                expansion_coeff_dim=expansion_coeff_dim,
                dropout=dropout,
            )
            for _ in range(n_stacks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, D)
        B, T, D = x.shape
        residual = x
        forecast = torch.zeros(B, self.pred_len, D, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast      # doubly residual link
            forecast = forecast + block_forecast
        return forecast                         # (B, pred_len, D)


def build_nbeats(
    seq_len: int,
    pred_len: int,
    n_features: int,
    n_stacks: int = 30,
    layer_width: int = 256,
    n_fc_layers: int = 4,
    expansion_coeff_dim: int = 32,
    dropout: float = 0.0,
) -> NBEATSModule:
    return NBEATSModule(
        seq_len=seq_len,
        pred_len=pred_len,
        n_features=n_features,
        n_stacks=n_stacks,
        layer_width=layer_width,
        n_fc_layers=n_fc_layers,
        expansion_coeff_dim=expansion_coeff_dim,
        dropout=dropout,
    )
