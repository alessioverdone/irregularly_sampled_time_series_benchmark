"""
irregular_datasets_static.py
----------------------------
Versione "statica" del dataset IMTS: la sparsificazione viene calcolata
una sola volta sull'intero dataset e salvata su disco, in modo che:

  1. Le maschere non cambiano tra run diversi → esperimenti riproducibili
     senza dover gestire seed per ogni singolo indice.
  2. Il data-loading è più veloce: niente ricalcolo di maschere nel __getitem__.
  3. I dati irregolari si possono ispezionare/visualizzare prima del training.
  4. Run con modelli diversi ma stessa configurazione condividono gli stessi
     dati: basta puntare alla stessa cartella save_dir.

Struttura della cartella di output (una per configurazione):

    <save_dir>/
        dataset.npz    intero dataset con tutte le finestre scorrevoli:
                           x_data  (N, seq_len, D)      input sparsificato
                           x_mask  (N, seq_len, D)      maschera (1=osservato)
                           y_data  (N, pred_len, D_out) target fully-observed
                           tp_obs  (seq_len,)            timestamp norm. input
                           tp_pred (pred_len,)           timestamp norm. target
        config.json    metadati e indici di split:
                           sample_splits   → {train/val/test: [start, end]}
                           timestep_borders → {train_end, val_end, test_end}
                           scaler          → {mean, std}
                           sparsify_cfg    → {mechanism, sparsity, seed}
                           seq_len, pred_len, features, ...

I confini di split rispettano esattamente la stessa convenzione delle classi
_TimeSeriesCSVDataset (overlap di seq_len tra i set adiacenti), quindi i
campioni prodotti sono identici a quelli del dataset dinamico.

API pubblica:
    sparsify_and_save(...)                  genera dataset.npz + config.json
    export_samples_to_csv(...)              esporta campioni in CSV leggibile
    StaticIrregularTimeSeriesDataset        drop-in per IrregularTimeSeriesDataset
    build_static_irregular_dataloaders(...) drop-in per build_irregular_dataloaders
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset.datasets import (
    StandardScaler,
    Split,
)
from src.dataset.irregular_datasets import (
    MissingnessMechanism,  # re-export per comodità
    SparsifyConfig,
    _generate_mask,
    imts_collate_fn,
)

__all__ = [
    "SparsifyConfig",
    "MissingnessMechanism",
    "sparsify_and_save",
    "export_samples_to_csv",
    "StaticIrregularTimeSeriesDataset",
    "build_static_irregular_dataloaders",
]


# ----------------------------------------------------------------------------- #
# Generazione e salvataggio su disco                                            #
# ----------------------------------------------------------------------------- #

def sparsify_and_save(
    csv_path: str | Path,
    save_dir: str | Path,
    split_config: dict,
    *,
    seq_len: int = 96,
    pred_len: int = 96,
    features: Literal["M", "S", "MS"] = "M",
    target: str | None = None,
    sparsify_cfg: SparsifyConfig | None = None,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Trasforma il dataset regolare in irregolare e salva l'intero dataset su disco.

    Genera UN singolo ``dataset.npz`` con tutte le finestre scorrevoli dell'intera
    serie temporale (train + val + test), più un ``config.json`` con i metadati e
    gli indici di campione che separano i tre split.  Run successivi con modelli
    diversi ma stessa configurazione possono riutilizzare questi file senza
    rigenerare le maschere.

    Args:
        csv_path     : percorso al CSV sorgente (da ``data/regular/``).
        save_dir     : cartella di destinazione (viene creata se non esiste).
        split_config : dizionario di configurazione degli split, nel formato
                       ``{"train_end": N, "val_end": M, "test_end": K}``  (assoluto)
                       oppure ``{"train_frac": 0.7, "val_frac": 0.8}``    (frazione).
                       Corrisponde a ``run_params.split_configs[dataset_name]``.
        seq_len      : lunghezza finestra di input.
        pred_len     : lunghezza orizzonte di previsione.
        features     : modalità di selezione feature (``"M"``, ``"S"``, ``"MS"``).
        target       : colonna target (None → ultima colonna).
        sparsify_cfg : configurazione della sparsificazione.
        force        : se True ricalcola anche se i file esistono già.
        verbose      : stampa messaggi di log.

    Returns:
        Path della ``save_dir``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = sparsify_cfg or SparsifyConfig()

    dataset_path = save_dir / "dataset.npz"
    config_path  = save_dir / "config.json"

    if dataset_path.exists() and config_path.exists() and not force:
        if verbose:
            print(
                f"[static] Dataset già presente in {save_dir}, skip "
                f"(usa force=True per rigenerare)."
            )
        return save_dir

    # --- Caricamento CSV completo ------------------------------------------ #
    csv_path = Path(csv_path)
    df_raw = pd.read_csv(csv_path)
    if "date" not in df_raw.columns:
        raise ValueError(f"Il CSV {csv_path} non ha colonna 'date'")
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    cols_data = [c for c in df_raw.columns if c != "date"]
    _target = target if target is not None else cols_data[-1]
    if _target not in cols_data:
        raise ValueError(f"target '{_target}' non trovato in {cols_data}")

    if features == "S":
        cols_use = [_target]
    else:  # "M" o "MS"
        cols_use = cols_data

    data_array = df_raw[cols_use].values.astype(np.float32)  # (total_len, D)
    total_len, D = data_array.shape

    # --- Calcolo bordi di split -------------------------------------------- #
    if "train_end" in split_config:
        train_end = min(split_config["train_end"], total_len)
        val_end   = min(split_config["val_end"],   total_len)
        test_end  = min(split_config["test_end"],  total_len)
    elif "train_frac" in split_config:
        train_end = int(total_len * split_config["train_frac"])
        val_end   = int(total_len * split_config["val_frac"])
        test_end  = total_len
    else:
        raise ValueError(
            f"split_config deve contenere 'train_end'/'val_end'/'test_end' "
            f"o 'train_frac'/'val_frac': {split_config}"
        )

    # --- Scaling: fit SOLO sulla porzione di training ---------------------- #
    scaler     = StandardScaler().fit(data_array[:train_end])
    data_scaled = scaler.transform(data_array)  # (total_len, D)

    # Colonne di output (per modalità MS il target è una sola colonna)
    if features == "MS":
        target_idx = cols_use.index(_target)
        data_out = data_scaled[:, target_idx : target_idx + 1]
    else:
        data_out = data_scaled
    D_out = data_out.shape[1]

    # --- Finestre scorrevoli sull'intero dataset --------------------------- #
    N = total_len - seq_len - pred_len + 1
    if N <= 0:
        raise ValueError(
            f"Dataset troppo corto: total_len={total_len}, "
            f"seq_len={seq_len}, pred_len={pred_len}"
        )

    x_data = np.zeros((N, seq_len, D),     dtype=np.float32)
    x_mask = np.zeros((N, seq_len, D),     dtype=np.float32)
    y_data = np.zeros((N, pred_len, D_out), dtype=np.float32)

    if verbose:
        print(
            f"[static] Calcolo {N} campioni "
            f"(total_len={total_len}, seq_len={seq_len}, pred_len={pred_len})…",
            flush=True,
        )

    for j in range(N):
        x_raw = data_scaled[j : j + seq_len]                        # (seq_len, D)
        y_raw = data_out[j + seq_len : j + seq_len + pred_len]       # (pred_len, D_out)

        seed_j = None if cfg.seed is None else cfg.seed + j
        rng    = np.random.default_rng(seed_j)
        mask   = _generate_mask(x_raw, cfg, rng)                     # (seq_len, D)

        if not mask.any():
            mask[0, 0] = 1.0

        x_data[j] = x_raw * mask
        x_mask[j] = mask
        y_data[j] = y_raw

    # --- Calcolo indici di campione per i tre split ------------------------ #
    # Convenzione identica a _TimeSeriesCSVDataset:
    #   train : finestre j in [0,            train_end - seq_len - pred_len]
    #   val   : finestre j in [train_end - seq_len, val_end - seq_len - pred_len]
    #   test  : finestre j in [val_end   - seq_len, test_end - seq_len - pred_len]
    j_train_end  = max(0, train_end - seq_len - pred_len + 1)           # esclusivo
    j_val_start  = max(0, train_end - seq_len)
    j_val_end    = max(j_val_start, val_end - seq_len - pred_len + 1)   # esclusivo
    j_test_start = max(0, val_end - seq_len)
    j_test_end   = max(j_test_start, test_end - seq_len - pred_len + 1) # esclusivo

    sample_splits = {
        "train": [0,           j_train_end],
        "val":   [j_val_start, j_val_end],
        "test":  [j_test_start, j_test_end],
    }

    # --- Timestamp normalizzati -------------------------------------------- #
    tp_full = np.linspace(0.0, 1.0, num=seq_len + pred_len, dtype=np.float32, endpoint=True)
    tp_obs  = tp_full[:seq_len]
    tp_pred = tp_full[seq_len:]

    # --- Salvataggio -------------------------------------------------------- #
    np.savez_compressed(
        dataset_path,
        x_data=x_data, x_mask=x_mask, y_data=y_data,
        tp_obs=tp_obs, tp_pred=tp_pred,
    )

    config_data = {
        "sample_splits":    sample_splits,
        "timestep_borders": {
            "train_end": int(train_end),
            "val_end":   int(val_end),
            "test_end":  int(test_end),
        },
        "scaler": {
            "mean": scaler.mean.tolist(),
            "std":  scaler.std.tolist(),
        },
        "sparsify_cfg": {
            "mechanism": cfg.mechanism,
            "sparsity":  cfg.sparsity,
            "seed":      cfg.seed,
        },
        "seq_len":         seq_len,
        "pred_len":        pred_len,
        "features":        features,
        "target":          _target,
        "feature_cols":    cols_use,
        "n_features_in":   D,
        "n_features_out":  D_out,
        "total_timesteps": total_len,
        "n_total_samples": N,
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    if verbose:
        sparsity_actual = 1.0 - float(x_mask.mean())
        n_train = j_train_end
        n_val   = j_val_end   - j_val_start
        n_test  = j_test_end  - j_test_start
        print(f"[static] Salvati {N} campioni totali → {save_dir}")
        print(
            f"[static] Campioni per split: "
            f"train={n_train}  val={n_val}  test={n_test}"
        )
        print(f"[static] Sparsità effettiva: {sparsity_actual:.3f}")

    return save_dir


# ----------------------------------------------------------------------------- #
# Esportazione CSV per ispezione visiva                                         #
# ----------------------------------------------------------------------------- #

def export_samples_to_csv(
    npz_path: str | Path,
    csv_path: str | Path,
    *,
    sample_indices: Sequence[int] | None = None,
    max_samples: int = 20,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Esporta campioni selezionati da ``dataset.npz`` in un CSV leggibile.

    Formato "long": una riga per (campione, timestep).
    Per ogni riga: sample_idx, t_idx, t_normalized, role (input/target),
    observed_step, feat_0…feat_{D-1}, mask_0…mask_{D-1}.

    I timestep di input hanno role="input" e le colonne mask riflettono la
    sparsificazione; i timestep di target hanno role="target" e mask sempre 1.

    Args:
        npz_path       : path a ``dataset.npz`` generato da ``sparsify_and_save``.
        csv_path       : path di output per il CSV.
        sample_indices : indici dei campioni da esportare (None = primi max_samples).
        max_samples    : numero massimo di campioni se sample_indices è None.
        feature_names  : nomi delle feature (None = feat_0, feat_1, …).

    Returns:
        DataFrame corrispondente al CSV salvato.
    """
    data = np.load(npz_path)
    x_data  = data["x_data"]   # (N, seq_len, D)
    x_mask  = data["x_mask"]   # (N, seq_len, D)
    y_data  = data["y_data"]   # (N, pred_len, D_out)
    tp_obs  = data["tp_obs"]   # (seq_len,)
    tp_pred = data["tp_pred"]  # (pred_len,)

    N, seq_len, D = x_data.shape
    pred_len, D_out = y_data.shape[1], y_data.shape[2]

    if sample_indices is None:
        sample_indices = list(range(min(max_samples, N)))

    feat_in   = feature_names if feature_names is not None else [f"feat_{d}" for d in range(D)]
    feat_out  = feature_names[:D_out] if feature_names is not None else [f"feat_{d}" for d in range(D_out)]
    mask_names = [f"mask_{d}" for d in range(D)]

    rows = []

    for sidx in sample_indices:
        # ---- Finestra di input -------------------------------------------- #
        for t in range(seq_len):
            row: dict = {
                "sample_idx":    sidx,
                "t_idx":         t,
                "t_normalized":  float(tp_obs[t]),
                "role":          "input",
                "observed_step": int(x_mask[sidx, t].any()),
            }
            for d, fname in enumerate(feat_in):
                row[fname] = float(x_data[sidx, t, d])
            for d, mname in enumerate(mask_names):
                row[mname] = int(x_mask[sidx, t, d])
            rows.append(row)

        # ---- Orizzonte target (fully observed) ----------------------------- #
        for t in range(pred_len):
            row = {
                "sample_idx":    sidx,
                "t_idx":         seq_len + t,
                "t_normalized":  float(tp_pred[t]),
                "role":          "target",
                "observed_step": 1,
            }
            for d, fname in enumerate(feat_out):
                row[fname] = float(y_data[sidx, t, d])
            for d in range(D):
                row[f"mask_{d}"] = 1 if d < D_out else 0
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(
        f"[static] CSV esportato → {csv_path}  "
        f"({len(sample_indices)} campioni × {seq_len + pred_len} step)"
    )
    return df


# ----------------------------------------------------------------------------- #
# Dataset statico: drop-in per IrregularTimeSeriesDataset                       #
# ----------------------------------------------------------------------------- #

class StaticIrregularTimeSeriesDataset(Dataset):
    """Dataset IMTS pre-calcolato su disco.

    Carica ``dataset.npz`` e ``config.json`` generati da ``sparsify_and_save()``
    e serve i campioni del solo split richiesto ("train", "val" o "test").

    Corrisponde 1-a-1 a ``IrregularTimeSeriesDataset``:
    - stesso formato di output di ``__getitem__``
    - stesse proprietà ``n_features`` e ``scaler``
    - compatibile con ``imts_collate_fn``

    La differenza è che le maschere sono fisse e pre-calcolate: risultati
    identici tra run senza dover gestire seed per campione.

    Args:
        dataset_dir : cartella generata da ``sparsify_and_save`` (contiene
                      ``dataset.npz`` e ``config.json``).
        split       : "train" | "val" | "test".
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: Split,
    ) -> None:
        super().__init__()
        dataset_dir = Path(dataset_dir)

        config_path  = dataset_dir / "config.json"
        dataset_path = dataset_dir / "dataset.npz"

        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json non trovato in {dataset_dir}\n"
                "Esegui prima sparsify_and_save() per generare il dataset statico."
            )
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"dataset.npz non trovato in {dataset_dir}\n"
                "Esegui prima sparsify_and_save() per generare il dataset statico."
            )

        with open(config_path) as f:
            config = json.load(f)

        start, end = config["sample_splits"][split]

        data = np.load(dataset_path)
        self._x_data  = data["x_data"][start:end]   # (N_split, seq_len, D)
        self._x_mask  = data["x_mask"][start:end]   # (N_split, seq_len, D)
        self._y_data  = data["y_data"][start:end]   # (N_split, pred_len, D_out)
        self._tp_obs  = data["tp_obs"]               # (seq_len,)
        self._tp_pred = data["tp_pred"]              # (pred_len,)

        self._N, self._seq_len, self._D = self._x_data.shape
        self._pred_len = self._y_data.shape[1]

        sc = config["scaler"]
        self._scaler = StandardScaler(
            mean=np.array(sc["mean"], dtype=np.float32),
            std=np.array(sc["std"],  dtype=np.float32),
        )

    # --- Interfaccia Dataset ------------------------------------------------- #

    def __len__(self) -> int:
        return self._N

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x_data = self._x_data[idx]   # (seq_len, D)
        x_mask = self._x_mask[idx]   # (seq_len, D)
        y_data = self._y_data[idx]   # (pred_len, D_out)

        # Compressione ragged: tieni solo i timestep con almeno un canale osservato
        any_obs       = x_mask.any(axis=1)        # (seq_len,)
        observed_data = x_data[any_obs]            # (L_obs, D)
        observed_mask = x_mask[any_obs]            # (L_obs, D)
        observed_tp   = self._tp_obs[any_obs]      # (L_obs,)

        mask_predicted_data = np.ones_like(y_data, dtype=np.float32)

        return {
            "observed_data":       torch.from_numpy(observed_data).float(),
            "observed_tp":         torch.from_numpy(observed_tp).float(),
            "observed_mask":       torch.from_numpy(observed_mask).float(),
            "data_to_predict":     torch.from_numpy(y_data).float(),
            "tp_to_predict":       torch.from_numpy(self._tp_pred).float(),
            "mask_predicted_data": torch.from_numpy(mask_predicted_data).float(),
        }

    # --- Info ---------------------------------------------------------------- #

    @property
    def n_features(self) -> int:
        """Numero di feature di input (compatibile con IrregularTimeSeriesDataset)."""
        return self._D

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def pred_len(self) -> int:
        return self._pred_len


# ----------------------------------------------------------------------------- #
# Helper: drop-in per build_irregular_dataloaders                               #
# ----------------------------------------------------------------------------- #

def build_static_irregular_dataloaders(
    csv_path: str | Path,
    save_dir: str | Path,
    split_config: dict,
    *,
    seq_len: int = 96,
    pred_len: int = 96,
    features: Literal["M", "S", "MS"] = "M",
    target: str | None = None,
    sparsify_cfg: SparsifyConfig | None = None,
    force: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    verbose: bool = True,
) -> dict[Split, DataLoader]:
    """Drop-in replacement per ``build_irregular_dataloaders`` (versione statica).

    Chiama ``sparsify_and_save()`` se i file non esistono (o ``force=True``),
    poi crea tre viste ``StaticIrregularTimeSeriesDataset`` (una per split) e
    le avvolge in ``DataLoader`` con la stessa ``imts_collate_fn``.

    Args:
        csv_path     : percorso al CSV sorgente (da ``data/regular/``).
        save_dir     : cartella dove salvare/caricare i file pre-calcolati.
                       Usa ``run_params.get_irr_save_dir()`` per ottenere
                       una cartella che identifica univocamente la configurazione.
        split_config : dizionario di configurazione degli split
                       (``run_params.split_configs[dataset_name]``).
        seq_len      : lunghezza finestra input.
        pred_len     : lunghezza orizzonte previsione.
        features     : modalità feature ("M", "S", "MS").
        target       : colonna target.
        sparsify_cfg : configurazione sparsificazione.
        force        : se True rigenera i file anche se già presenti.
        batch_size   : dimensione del batch.
        num_workers  : worker per il DataLoader.
        verbose      : stampa messaggi di log.

    Returns:
        dict con chiavi "train", "val", "test" e DataLoader corrispondenti.
    """
    save_dir = Path(save_dir)

    sparsify_and_save(
        csv_path=csv_path,
        save_dir=save_dir,
        split_config=split_config,
        seq_len=seq_len,
        pred_len=pred_len,
        features=features,
        target=target,
        sparsify_cfg=sparsify_cfg,
        force=force,
        verbose=verbose,
    )

    loaders: dict[Split, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = StaticIrregularTimeSeriesDataset(dataset_dir=save_dir,
                                              split=split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train"),
            collate_fn=imts_collate_fn,
        )

    return loaders


if __name__ == '__main__':
    from src.dataset.datamodule import get_datamodule
    from src.utils.utils import setup_seed
    from src.config import initialize_configuration

    # Params
    run_params = initialize_configuration()
    run_params.irregular_time_series = True
    run_params.mechanism = "async"
    run_params.sparsity = 0.6
    run_params.irregular_time_series_pattern = 'static'
    run_params.dataset = "etth1"

    setup_seed(run_params.seed)
    print('Configuration settled!')

    # Data
    dataModuleInstance, run_params = get_datamodule(run_params)
    print('Data imported!')
