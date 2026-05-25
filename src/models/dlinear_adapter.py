"""
dlinear_adapter.py
------------------
DLinear: Decomposition-Linear per forecasting su serie temporali regolari.
"Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2022)

Il modello decompone l'input in trend (moving average) e stagionalità,
applica una proiezione lineare indipendente a ciascuna componente e somma
i due contributi per produrre il forecast.

Interfaccia:
    forward(x: Tensor[B, seq_len, D]) -> Tensor[B, pred_len, D]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinearModule(nn.Module):
    """
    Args:
        seq_len     : lunghezza della finestra di input.
        pred_len    : orizzonte di previsione.
        n_features  : numero di feature (canali).
        kernel_size : dimensione del kernel per la moving average (estrazione trend).
        individual  : se True, una coppia di layer lineari per feature;
                      se False (default), layer condivisi tra tutte le feature.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        kernel_size: int = 25,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_size = kernel_size
        self.individual = individual

        if individual:
            self.linear_trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_features)]
            )
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_features)]
            )
        else:
            self.linear_trend = nn.Linear(seq_len, pred_len)
            self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def _moving_avg(self, x: torch.Tensor) -> torch.Tensor:
        """Moving average per estrarre il trend. x: (B, T, D) → (B, T, D)"""
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size // 2
        x_t = x.permute(0, 2, 1)                                        # (B, D, T)
        x_pad = F.pad(x_t, (pad_left, pad_right), mode="replicate")
        trend = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1, padding=0)
        return trend.permute(0, 2, 1)                                    # (B, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, D)
        trend = self._moving_avg(x)
        seasonal = x - trend

        if self.individual:
            D = x.shape[-1]
            trend_out = torch.stack(
                [self.linear_trend[d](trend[..., d]) for d in range(D)], dim=-1
            )
            seasonal_out = torch.stack(
                [self.linear_seasonal[d](seasonal[..., d]) for d in range(D)], dim=-1
            )
        else:
            # (B, T, D) → transpose → (B, D, T) → linear → (B, D, P) → (B, P, D)
            trend_out = self.linear_trend(trend.transpose(1, 2)).transpose(1, 2)
            seasonal_out = self.linear_seasonal(seasonal.transpose(1, 2)).transpose(1, 2)

        return trend_out + seasonal_out  # (B, pred_len, D)


def build_dlinear(
    seq_len: int,
    pred_len: int,
    n_features: int,
    kernel_size: int = 25,
    individual: bool = False,
) -> DLinearModule:
    return DLinearModule(
        seq_len=seq_len,
        pred_len=pred_len,
        n_features=n_features,
        kernel_size=kernel_size,
        individual=individual,
    )
