"""
irregular_datasets.py
---------------------
Wrapper che trasforma i Dataset regolari (ETTh1/2, ETTm1/2, Electricity)
nel formato canonico IMTS (Irregular Multivariate Time Series) usato dal
benchmark di tPatchGNN, Hi-Patch e mTAND.

Per ogni campione il dataset restituisce un dict con i seguenti tensori
(stesso schema di `lib/parse_datasets.py` di tPatchGNN):

    observed_data       (L_obs, D)        valori osservati nella finestra di input
                                          (zero come placeholder dove non osservato)
    observed_tp         (L_obs,)          timestamp normalizzati [0,1] dei tp ritenuti
                                          (unione dei tp osservati su qualunque canale)
    observed_mask       (L_obs, D)        1 se il canale d è osservato a tp[i], else 0
    data_to_predict     (L_pred, D)       valori target sull'orizzonte (fully observed)
    tp_to_predict       (L_pred,)         timestamp normalizzati delle query di forecast
    mask_predicted_data (L_pred, D)       maschera dei target da predire (tutti 1)

Il batching avviene tramite `imts_collate_fn`, che fa zero-pad alla massima
L_obs del batch e produce un campo `padding_mask` aggiuntivo.

Quattro meccanismi di sparsificazione:
  mcar     – Bernoulli indipendente per ogni (t, d): packet loss casuale.
  burst    – Blocchi contigui di timestep mancanti su tutti i canali: guasto di rete.
  periodic – Gap periodici di manutenzione: ogni `period` step, `gap_len` step assenti.
  async    – Ogni canale campiona in modo indipendente (processo di Poisson per canale).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset.datasets import (
    StandardScaler,
    _TimeSeriesCSVDataset,
    Split,
)


MissingnessMechanism = Literal[
    "mcar",         # random packet loss: Bernoulli indipendente per ogni (t, d)
    "burst",        # burst outages: blocchi contigui su tutti i canali (guasto di rete)
    "periodic",     # periodic maintenance: gap regolari ogni `period` step
    "async",        # sensor-async: ogni canale ha il proprio processo di Poisson
]


# ----------------------------------------------------------------------------- #
# Sparsificazione                                                               #
# ----------------------------------------------------------------------------- #
@dataclass
class SparsifyConfig:
    """Configurazione della sparsificazione applicata alla finestra di input."""

    mechanism: MissingnessMechanism = "mcar"
    sparsity: float = 0.3       # frazione di valori (t, d) da rimuovere
    seed: int | None = 0        # seed base per riproducibilità
    burst_len: int = 8          # burst: lunghezza media dei burst (timestep)
    period: int = 24            # periodic: periodo tra i gap di manutenzione (timestep)

    def __post_init__(self) -> None:
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity deve essere in [0, 1), ricevuto {self.sparsity}")
        if self.burst_len < 1:
            raise ValueError(f"burst_len deve essere >= 1, ricevuto {self.burst_len}")
        if self.period < 2:
            raise ValueError(f"period deve essere >= 2, ricevuto {self.period}")


def _sparsify_mcar(
    seq_x: np.ndarray,
    sparsity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Maschera MCAR: ogni (t, d) viene osservato con probabilità (1 - sparsity)."""
    L, D = seq_x.shape
    keep_prob = 1.0 - sparsity
    mask = (rng.random(size=(L, D)) < keep_prob).astype(np.float32)
    return mask


def _sparsify_burst(
    seq_x: np.ndarray,
    sparsity: float,
    rng: np.random.Generator,
    burst_len: int = 8,
) -> np.ndarray:
    """Burst outages: blocchi contigui di timestep mancanti su tutti i canali.

    Usa una catena di Markov a due stati (osservato / mancante) con:
      p_mo = 1 / burst_len           (prob. di uscire dal burst)
      p_om = sparsity * p_mo / (1 - sparsity)  (prob. di entrare in burst)
    La distribuzione stazionaria è esattamente `sparsity`, e la durata media
    di ogni burst è `burst_len` step.  Tutti i canali sono assenti insieme.
    """
    L, D = seq_x.shape
    p_mo = 1.0 / max(1, burst_len)
    p_om = sparsity * p_mo / max(1e-9, 1.0 - sparsity)

    missing = np.zeros(L, dtype=bool)
    state = int(rng.random() < sparsity)  # stato iniziale campionato dalla stazionaria
    for t in range(L):
        if state == 1:
            missing[t] = True
            if rng.random() < p_mo:
                state = 0
        else:
            if rng.random() < p_om:
                state = 1

    mask = np.ones((L, D), dtype=np.float32)
    mask[missing] = 0.0
    return mask


def _sparsify_periodic(
    seq_x: np.ndarray,
    sparsity: float,
    rng: np.random.Generator,
    period: int = 24,
) -> np.ndarray:
    """Gap periodici di manutenzione: ogni `period` step, `gap_len` step sono assenti.

    La durata del gap è derivata dalla sparsity: ``gap_len = round(sparsity * period)``.
    La fase iniziale è randomizzata per evitare artefatti legati all'inizio della finestra.
    Tutti i canali sono simultaneamente assenti durante il gap (downtime globale).
    """
    L, D = seq_x.shape
    gap_len = max(1, round(sparsity * period))
    phase = int(rng.integers(0, period))

    t_idx = np.arange(L)
    missing = (t_idx + phase) % period < gap_len

    mask = np.ones((L, D), dtype=np.float32)
    mask[missing] = 0.0
    return mask


def _sparsify_async(
    seq_x: np.ndarray,
    sparsity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Campionamento asincrono per canale: ogni sensore ha il proprio processo di Poisson.

    Simula sensori eterogenei con frequenze di campionamento diverse e indipendenti.
    Per ogni canale d si generano inter-arrivi geometrici con tasso leggermente variabile
    attorno a (1 - sparsity): i canali sono asincroni perché campionano in tempi diversi
    e con rate leggermente differenti (±20 % di variazione uniforme).
    """
    L, D = seq_x.shape
    mask = np.zeros((L, D), dtype=np.float32)
    keep_prob = 1.0 - sparsity

    if keep_prob <= 0.0:
        return mask

    for d in range(D):
        # Rate leggermente diverso per canale: simula sensori eterogenei
        rate_d = float(np.clip(keep_prob * rng.uniform(0.8, 1.2), 0.05, 0.95))
        # Processo di Poisson: inter-arrivi geometrici con parametro rate_d
        t = int(rng.geometric(p=rate_d)) - 1
        while t < L:
            mask[t, d] = 1.0
            t += int(rng.geometric(p=rate_d))

    return mask


def _generate_mask(
    seq_x: np.ndarray,
    cfg: SparsifyConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dispatcher per i meccanismi di missingness."""
    if cfg.mechanism == "mcar":
        return _sparsify_mcar(seq_x, cfg.sparsity, rng)
    if cfg.mechanism == "burst":
        return _sparsify_burst(seq_x, cfg.sparsity, rng, burst_len=cfg.burst_len)
    if cfg.mechanism == "periodic":
        return _sparsify_periodic(seq_x, cfg.sparsity, rng, period=cfg.period)
    if cfg.mechanism == "async":
        return _sparsify_async(seq_x, cfg.sparsity, rng)
    raise NotImplementedError(f"Meccanismo '{cfg.mechanism}' non supportato.")


# ----------------------------------------------------------------------------- #
# Dataset IMTS                                                                  #
# ----------------------------------------------------------------------------- #
class IrregularTimeSeriesDataset(Dataset):
    """Avvolge un `_TimeSeriesCSVDataset` regolare e produce sample in formato
    canonico IMTS compatibile con tPatchGNN, Hi-Patch e mTAND.

    Solo la **finestra di input** viene sparsificata; l'orizzonte target resta
    fully observed (è il setting standard del paper: si valuta su test pulito).

    Args:
        base_dataset : istanza di un `_TimeSeriesCSVDataset` (ETTh1Dataset, ...)
                       già configurato con seq_len, label_len, pred_len, ecc.
                       NB: questa classe ignora `label_len` e usa solo gli ultimi
                       `pred_len` step come target, in linea col formato IMTS
                       (i modelli IMTS non usano il decoder-start-token).
        sparsify_cfg : configurazione del processo di sparsificazione.
    """

    def __init__(
        self,
        base_dataset: _TimeSeriesCSVDataset,
        sparsify_cfg: SparsifyConfig | None = None,
    ) -> None:
        super().__init__()
        self.base = base_dataset
        self.cfg = sparsify_cfg or SparsifyConfig()

        self.seq_len = base_dataset.seq_len
        self.pred_len = base_dataset.pred_len

        # Timestamp normalizzati in [0, 1] sulla finestra completa input+forecast.
        # Convenzione tPatchGNN: tp continui in float, durata totale arbitraria
        # finché coerente. Usiamo (seq_len + pred_len - 1) come scala.
        total_len = self.seq_len + self.pred_len
        self._tp_full = np.linspace(0.0,
                                    1.0,
                                    num=total_len,
                                    dtype=np.float32,
                                    endpoint=True)

    # --- API Dataset --------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Estrae la coppia (x, y) regolare dal dataset base.
        # x : (seq_len, D)         input fully observed
        # y : (label_len + pred_len, D)  con overlap di label_len step
        x_reg, y_reg = self.base[idx]
        x_reg = x_reg.numpy()
        y_reg = y_reg.numpy()

        # Prendiamo solo gli ultimi pred_len step di y come target IMTS,
        # scartando l'overlap label_len (non rilevante per i modelli IMTS).
        y_target = y_reg[-self.pred_len:]

        # --- Sparsificazione della finestra di input ----------------------- #
        # Seed per-campione deterministico → riproducibilità tra split / run.
        seed = None if self.cfg.seed is None else self.cfg.seed + idx
        rng = np.random.default_rng(seed)
        full_mask = _generate_mask(x_reg, self.cfg, rng)  # (seq_len, D)

        # --- Compressione al formato ragged: tieni solo i tp con almeno   --- #
        # --- un canale osservato.                                          --- #
        any_obs = full_mask.any(axis=1)            # (seq_len,)
        # Edge case: se per caso non c'è alcuna osservazione, forziamo
        # almeno il primo step come visibile (sul canale 0) per evitare
        # batch vuoti — situazione rarissima con sparsity < 0.95.
        if not any_obs.any():
            full_mask[0, 0] = 1.0
            any_obs[0] = True

        observed_data = x_reg[any_obs]              # (L_obs, D)
        observed_mask = full_mask[any_obs]          # (L_obs, D)
        observed_tp   = self._tp_full[:self.seq_len][any_obs]  # (L_obs,)

        # Dove un canale non è osservato a quel tp, azzera il valore
        # (placeholder: tPatchGNN/Hi-Patch ignorano comunque quei valori
        # tramite la observed_mask, ma è buona pratica non lasciare leak).
        observed_data = observed_data * observed_mask

        # --- Target: orizzonte fully observed ------------------------------- #
        tp_to_predict = self._tp_full[self.seq_len:]      # (pred_len,)
        mask_predicted_data = np.ones_like(y_target, dtype=np.float32)

        return {
            "observed_data":       torch.from_numpy(observed_data).float(),
            "observed_tp":         torch.from_numpy(observed_tp).float(),
            "observed_mask":       torch.from_numpy(observed_mask).float(),
            "data_to_predict":     torch.from_numpy(y_target).float(),
            "tp_to_predict":       torch.from_numpy(tp_to_predict).float(),
            "mask_predicted_data": torch.from_numpy(mask_predicted_data).float(),
        }

    # --- Info ---------------------------------------------------------------- #
    @property
    def n_features(self) -> int:
        return self.base.n_features_in

    @property
    def scaler(self) -> StandardScaler:
        return self.base.scaler


# ----------------------------------------------------------------------------- #
# Collate function: zero-pad al massimo L_obs del batch                         #
# ----------------------------------------------------------------------------- #
def imts_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Padda gli elementi del batch alla lunghezza massima `L_obs`.

    Aggiunge il campo `padding_mask` (B, L_obs_max) con 1 sui timestep validi
    e 0 sul padding, come si aspettano i loader di tPatchGNN/Hi-Patch.

    L'orizzonte target ha lunghezza fissa `pred_len` per costruzione, quindi
    non richiede padding.
    """
    B = len(batch)
    D = batch[0]["observed_data"].shape[1]
    L_pred = batch[0]["data_to_predict"].shape[0]

    L_obs_max = max(item["observed_data"].shape[0] for item in batch)

    observed_data = torch.zeros(B, L_obs_max, D)
    observed_tp   = torch.zeros(B, L_obs_max)
    observed_mask = torch.zeros(B, L_obs_max, D)
    padding_mask  = torch.zeros(B, L_obs_max)

    for i, item in enumerate(batch):
        L = item["observed_data"].shape[0]
        observed_data[i, :L] = item["observed_data"]
        observed_tp[i, :L]   = item["observed_tp"]
        observed_mask[i, :L] = item["observed_mask"]
        padding_mask[i, :L]  = 1.0

    data_to_predict     = torch.stack([item["data_to_predict"]     for item in batch])
    tp_to_predict       = torch.stack([item["tp_to_predict"]       for item in batch])
    mask_predicted_data = torch.stack([item["mask_predicted_data"] for item in batch])

    return {
        "observed_data":       observed_data,        # (B, L_obs_max, D)
        "observed_tp":         observed_tp,          # (B, L_obs_max)
        "observed_mask":       observed_mask,        # (B, L_obs_max, D)
        "padding_mask":        padding_mask,         # (B, L_obs_max)
        "data_to_predict":     data_to_predict,      # (B, L_pred, D)
        "tp_to_predict":       tp_to_predict,        # (B, L_pred)
        "mask_predicted_data": mask_predicted_data,  # (B, L_pred, D)
    }


# ----------------------------------------------------------------------------- #
# Helper: costruisce DataLoader IMTS train/val/test condividendo lo scaler      #
# ----------------------------------------------------------------------------- #
def build_irregular_dataloaders(
    dataset_cls: type[_TimeSeriesCSVDataset],
    csv_path: str | Path,
    *,
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 96,
    features: Literal["M", "S", "MS"] = "M",
    target: str | None = None,
    sparsify_cfg: SparsifyConfig | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[Split, DataLoader]:
    """Costruisce i tre DataLoader IMTS (train, val, test).

    - Lo `StandardScaler` è fittato sul training e condiviso con val/test.
    - La sparsificazione è applicata in modo deterministico tramite
      seed(cfg.seed + idx); per ottenere maschere indipendenti tra split,
      passare `sparsify_cfg` con seed diversi per i tre split (oppure
      lasciare il default e accettare che lo stesso indice dia la stessa
      maschera, che è ciò che si vuole quasi sempre per riproducibilità).
    - `label_len` è fissato a 0 perché non serve nei modelli IMTS.
    """
    common = dict(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        features=features,
    )
    if target is not None:
        common["target"] = target

    train_base = dataset_cls(csv_path=csv_path, split="train", **common)
    scaler = train_base.scaler
    val_base = dataset_cls(csv_path=csv_path, split="val", scaler=scaler, **common)
    test_base = dataset_cls(csv_path=csv_path, split="test", scaler=scaler, **common)

    train_ds = IrregularTimeSeriesDataset(train_base, sparsify_cfg)
    val_ds   = IrregularTimeSeriesDataset(val_base,   sparsify_cfg)
    test_ds  = IrregularTimeSeriesDataset(test_base,  sparsify_cfg)

    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
            collate_fn=imts_collate_fn,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
            collate_fn=imts_collate_fn,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
            collate_fn=imts_collate_fn,
        ),
    }


if __name__ == '__main__':
    from src.dataset.datamodule import get_datamodule
    from src.utils.utils import setup_seed
    from src.config import initialize_configuration

    # Params
    run_params = initialize_configuration()
    run_params.irregular_time_series = True
    run_params.irregular_time_series_pattern = 'static'
    run_params.dataset = "etth1"

    setup_seed(run_params.seed)
    print('Configuration settled!')

    # Data
    dataModuleInstance, run_params = get_datamodule(run_params)
    print('Data imported!')
