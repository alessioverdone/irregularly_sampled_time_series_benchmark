from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


"""
Classi `torch.utils.data.Dataset` per i tre dataset, con una classe BASE
(`_TimeSeriesCSVDataset`) che fa il lavoro comune (lettura CSV, normalizzazione,
finestra scorrevole) e tre sottoclassi specifiche con i parametri tipici dei
benchmark di previsione di serie temporali multivariate:

    - ETTh1Dataset       (orario,        7 feature)
    - ETTh2Dataset       (orario,        7 feature)
    - ETTm1Dataset       (15 minuti,     7 feature)
    - ETTm2Dataset       (15 minuti,     7 feature)
    - ElectricityDataset (orario,      321 feature)

Per ogni split (train / val / test) viene anche fornita una funzione
`build_dataloader(...)` che restituisce un `DataLoader` di PyTorch già
configurato.

Il task implementato è la previsione multi-step "seq2seq" tipica:
    - Input  : finestra di lunghezza `seq_len`     (passato)
    - Output : finestra di lunghezza `pred_len`    (futuro da prevedere)
    - Etichetta intermedia di lunghezza `label_len` (overlap, usata da modelli
      tipo Informer/Autoformer come "decoder start token"; può essere ignorata).

Lo split temporale è quello standard dei benchmark:
    ETTh1: train [0 : 12*30*24], val [12*30*24 : 16*30*24], test [16*30*24 : 20*30*24]
    ETTm1: train [0 : 12*30*24*4], val [.. : 16*30*24*4], test [.. : 20*30*24*4]
    Electricity: 70% / 10% / 20% sequenziali.
"""

Split = Literal["train", "val", "test"]


# ----------------------------------------------------------------------------- #
# Helpers                                                                       #
# ----------------------------------------------------------------------------- #
@dataclass
class StandardScaler:
    """Z-score scaler `fit` sul solo set di training."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        # Evita divisioni per zero su feature costanti
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None, "scaler non fittato"
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return x * self.std + self.mean


# ----------------------------------------------------------------------------- #
# Dataset di base                                                               #
# ----------------------------------------------------------------------------- #
class _TimeSeriesCSVDataset(Dataset):
    """
    Dataset base. Le sottoclassi forniscono:
        - `csv_path`
        - `_split_borders()`   : tuple (train_end, val_end, test_end) sugli indici
        - `freq`               : "h", "t" (15-min), ecc. – informativo
    """

    csv_path: Path
    freq: str = "h"

    def __init__(
        self,
        csv_path: str | Path,
        split: Split = "train",
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        target: str | None = None,
        features: Literal["M", "S", "MS"] = "M",
        scale: bool = True,
        scaler: StandardScaler | None = None,
        split_borders: tuple[int, int, int] | None = None,
    ) -> None:
        """
        Args:
            csv_path  : CSV con colonna 'date' + colonne feature.
            split     : "train" | "val" | "test".
            seq_len   : lunghezza finestra di input.
            label_len : lunghezza overlap col target (decoder start, può essere 0).
            pred_len  : lunghezza orizzonte di previsione.
            target    : nome della colonna target (per task univariati / MS).
                        Per ETTh1/ETTm1 il default dei benchmark è "OT".
            features  : "M" multivariato → multivariato,
                        "S" univariato (solo target) → univariato,
                        "MS" multivariato → univariato (predice solo target).
            scale     : se True applica z-score (fit solo su train).
            scaler    : se passato dall'esterno (es. per val/test), viene riusato.
        """
        super().__init__()
        self.csv_path = Path(csv_path)
        self.split = split
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.features = features

        # --- Lettura CSV --------------------------------------------------- #
        df_raw = pd.read_csv(self.csv_path)
        if "date" not in df_raw.columns:
            raise ValueError(f"Il CSV {self.csv_path} non ha colonna 'date'")
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

        cols_data = [c for c in df_raw.columns if c != "date"]
        if target is None:
            target = cols_data[-1]  # default: ultima colonna
        if target not in cols_data:
            raise ValueError(f"target '{target}' non trovato in {cols_data}")
        self.target = target

        # --- Selezione colonne in base a `features` ----------------------- #
        if features == "S":
            cols_use = [target]
        else:  # "M" o "MS"
            cols_use = cols_data
        self.feature_cols = cols_use

        data_array = df_raw[cols_use].values.astype(np.float32)  # (T, F)
        self.dates = df_raw["date"].values

        # --- Split borders ------------------------------------------------- #
        if split_borders is not None:
            _total = len(data_array)
            train_end = min(split_borders[0], _total)
            val_end   = min(split_borders[1], _total)
            test_end  = min(split_borders[2], _total)
        else:
            train_end, val_end, test_end = self._split_borders(len(data_array))
        borders_left = {
            "train": 0,
            "val": train_end - self.seq_len,
            "test": val_end - self.seq_len,
        }
        borders_right = {
            "train": train_end,
            "val": val_end,
            "test": test_end,
        }
        l, r = borders_left[split], borders_right[split]
        if l < 0:
            raise ValueError(
                f"seq_len={seq_len} troppo grande per lo split '{split}': "
                f"borders left={l}"
            )

        # --- Scaling: fit SOLO sui dati di training ----------------------- #
        if scale:
            if scaler is None:
                scaler = StandardScaler().fit(data_array[:train_end])
            data_scaled = scaler.transform(data_array)
        else:
            scaler = scaler or StandardScaler(mean=np.zeros(data_array.shape[1]),
                                              std=np.ones(data_array.shape[1]))
            data_scaled = data_array
        self.scaler = scaler

        # `data_x` è l'input (multivariato), `data_y` è l'output:
        # in modalità "MS" l'output è solo il target.
        self.data_x = data_scaled[l:r]
        if features == "MS":
            target_idx = cols_use.index(target)
            self.data_y = data_scaled[l:r, target_idx : target_idx + 1]
        else:
            self.data_y = data_scaled[l:r]

        # numero di campioni con finestra scorrevole
        self._len = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self._len <= 0:
            raise ValueError(
                f"Split '{split}' troppo corto: "
                f"len(data)={len(self.data_x)}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}"
            )

    # --- Da implementare nelle sottoclassi --------------------------------- #
    def _split_borders(self, total_len: int) -> tuple[int, int, int]:
        raise NotImplementedError

    # --- Interfaccia Dataset ----------------------------------------------- #
    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]                # (seq_len, F_in)
        seq_y = self.data_y[r_begin:r_end]                # (label_len + pred_len, F_out)

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
        )

    # --- Info ---------------------------------------------------------------- #
    @property
    def n_features_in(self) -> int:
        return self.data_x.shape[1]

    @property
    def n_features_out(self) -> int:
        return self.data_y.shape[1]


# ----------------------------------------------------------------------------- #
# Sottoclassi specifiche                                                        #
# ----------------------------------------------------------------------------- #
class ETTh1Dataset(_TimeSeriesCSVDataset):
    """ETTh1: Electricity Transformer Temperature, sample orario (~17 mesi)."""

    freq = "h"

    def __init__(self, csv_path="./data/ETTh1.csv", target="OT", **kwargs):
        super().__init__(csv_path=csv_path, target=target, **kwargs)

    def _split_borders(self, total_len: int) -> tuple[int, int, int]:
        # Convenzione standard ETT: 12 mesi train, 4 val, 4 test → ore
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = val_end + 4 * 30 * 24
        return min(train_end, total_len), min(val_end, total_len), min(test_end, total_len)


class ETTh2Dataset(ETTh1Dataset):
    """ETTh2: stessa struttura di ETTh1 (orario, 7 feature, target 'OT'),
    raccolto da una stazione di trasformazione diversa. Stessi border."""

    def __init__(self, csv_path="./data/ETTh2.csv", target="OT", **kwargs):
        # Salta ETTh1Dataset.__init__ e va direttamente alla base, per non
        # forzare csv_path="./data/ETTh1.csv".
        _TimeSeriesCSVDataset.__init__(
            self, csv_path=csv_path, target=target, **kwargs
        )


class ETTm1Dataset(_TimeSeriesCSVDataset):
    """ETTm1: Electricity Transformer Temperature, sample ogni 15 minuti."""

    freq = "t"  # "minutely"

    def __init__(self, csv_path="./data/ETTm1.csv", target="OT", **kwargs):
        super().__init__(csv_path=csv_path, target=target, **kwargs)

    def _split_borders(self, total_len: int) -> tuple[int, int, int]:
        # Convenzione standard ETT: ×4 perché campionato ogni 15 min
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = val_end + 4 * 30 * 24 * 4
        return min(train_end, total_len), min(val_end, total_len), min(test_end, total_len)


class ETTm2Dataset(ETTm1Dataset):
    """ETTm2: stessa struttura di ETTm1 (15 min, 7 feature, target 'OT'),
    raccolto da una stazione di trasformazione diversa. Stessi border."""

    def __init__(self, csv_path="./data/ETTm2.csv", target="OT", **kwargs):
        _TimeSeriesCSVDataset.__init__(
            self, csv_path=csv_path, target=target, **kwargs
        )


class ElectricityDataset(_TimeSeriesCSVDataset):
    """Electricity (LD2011_2014): consumo orario di 321 clienti."""

    freq = "h"

    def __init__(self, csv_path="./data/electricity.csv", target=None, **kwargs):
        # target=None → la base-class prende l'ultima colonna (cliente "MT_321")
        super().__init__(csv_path=csv_path, target=target, **kwargs)

    def _split_borders(self, total_len: int) -> tuple[int, int, int]:
        # Convenzione benchmark: 70% / 10% / 20% sequenziali
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.8)
        test_end = total_len
        return train_end, val_end, test_end


class SolarDataset(_TimeSeriesCSVDataset):
    """
    Solar-Energy / SolarBenchmark: produzione PV simulata da 137 impianti
    in Alabama, anno 2006, sampling ogni 10 minuti, 52560 step totali.

    Note:
      - Il dataset originale non ha asse temporale: viene ricostruito
        in `download_solar()` partendo dal 2006-01-01 00:00 a passo 10min.
      - Il target di default è l'ultimo impianto ('plant_136'); è una scelta
        arbitraria, modificabile via parametro `target`.
      - Split 70/10/20 sequenziali, come da convenzione dei benchmark
        Informer/Autoformer/PatchTST per "Solar-Energy".
    """

    freq = "10min"

    def __init__(self, csv_path="./data/solar_AL.csv", target=None, **kwargs):
        super().__init__(csv_path=csv_path, target=target, **kwargs)

    def _split_borders(self, total_len: int) -> tuple[int, int, int]:
        # Convenzione benchmark: 70% / 10% / 20% sequenziali
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.8)
        test_end = total_len
        return train_end, val_end, test_end


# ----------------------------------------------------------------------------- #
# Helper per costruire DataLoader train/val/test condividendo lo scaler         #
# ----------------------------------------------------------------------------- #
def build_dataloaders(
    dataset_cls: type[_TimeSeriesCSVDataset],
    csv_path: str | Path,
    *,
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 96,
    features: Literal["M", "S", "MS"] = "M",
    target: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[Split, DataLoader]:
    """
    Costruisce i tre dataset (train, val, test) condividendo lo `StandardScaler`
    fittato sul solo training, e li avvolge in `DataLoader`.

    Ritorna: dict con chiavi "train", "val", "test".
    """
    common = dict(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        features=features,
    )
    if target is not None:
        common["target"] = target

    train_ds = dataset_cls(csv_path=csv_path, split="train", **common)
    scaler = train_ds.scaler
    val_ds = dataset_cls(csv_path=csv_path, split="val", scaler=scaler, **common)
    test_ds = dataset_cls(csv_path=csv_path, split="test", scaler=scaler, **common)

    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
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
    run_params.dataset = "electricity"

    setup_seed(run_params.seed)
    print('Configuration settled!')

    # Data
    dataModuleInstance, run_params = get_datamodule(run_params)
    print('Data imported!')
