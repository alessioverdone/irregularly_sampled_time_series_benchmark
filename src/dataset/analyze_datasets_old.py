from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from download import download_all
from datasets import (
    ElectricityDataset,
    ETTh1Dataset,
    ETTh2Dataset,
    ETTm1Dataset,
    ETTm2Dataset,
    SolarDataset,
    build_dataloaders,
)

"""
Pipeline end-to-end:
    1) scarica i tre dataset (ETTh1, ETTm1, Electricity) in ./data/
    2) costruisce Dataset + DataLoader (train / val / test) per ognuno
    3) stampa le proprietà di ciascun dataset:
         - shape del CSV grezzo, range temporale, frequenza
         - colonne feature, target
         - dimensioni dei tre split
         - shape di un singolo campione (x, y)
         - shape di un batch dal DataLoader
         - statistiche dello scaler
"""


def _print_section(title: str) -> None:
    line = "═" * 70
    print(f"\n{line}\n  {title}\n{line}")


def describe_dataset(name: str, dataset_cls, csv_path: Path) -> None:
    _print_section(f"Dataset: {name}")

    # --- 1) Info dal CSV grezzo --------------------------------------------- #
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = [c for c in df.columns if c != "date"]
    print(f"  File CSV          : {csv_path}")
    print(f"  Shape CSV grezzo  : {df.shape}  (righe × colonne incluso 'date')")
    print(f"  N. feature        : {len(feature_cols)}")
    if len(feature_cols) <= 10:
        print(f"  Colonne feature   : {feature_cols}")
    else:
        print(
            f"  Colonne feature   : {feature_cols[:5]} … "
            f"{feature_cols[-3:]}  (totale {len(feature_cols)})"
        )
    print(f"  Range temporale   : {df['date'].min()}  →  {df['date'].max()}")
    if len(df) > 1:
        delta = df["date"].iloc[1] - df["date"].iloc[0]
        print(f"  Passo temporale   : {delta}  (frequenza='{dataset_cls.freq}')")
    print(f"  Memoria CSV       : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    # --- 2) Costruzione DataLoader ----------------------------------------- #
    seq_len, label_len, pred_len = 96, 48, 96
    loaders = build_dataloaders(
        dataset_cls,
        csv_path=csv_path,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        features="M",          # multivariato → multivariato
        batch_size=32,
        num_workers=0,
    )

    print(
        f"\n  Configurazione finestra: "
        f"seq_len={seq_len}, label_len={label_len}, pred_len={pred_len}"
    )

    # --- 3) Proprietà di ogni split ---------------------------------------- #
    for split, loader in loaders.items():
        ds = loader.dataset
        x_sample, y_sample = ds[0]
        scaler = ds.scaler
        print(
            f"\n  ── Split '{split}' ────────────────────────────────"
        )
        print(f"    n. campioni (finestre)  : {len(ds):>8d}")
        print(f"    n. batch DataLoader     : {len(loader):>8d}  (batch_size=32)")
        print(f"    target                  : '{ds.target}'")
        print(f"    n_features_in           : {ds.n_features_in}")
        print(f"    n_features_out          : {ds.n_features_out}")
        print(
            f"    shape singolo campione  : "
            f"x={tuple(x_sample.shape)}, y={tuple(y_sample.shape)}"
        )
        # Stats post-normalizzazione (sui dati dello split)
        x_arr = ds.data_x
        print(
            f"    dati split (post-scaling): mean={x_arr.mean():+.3f}, "
            f"std={x_arr.std():.3f}, min={x_arr.min():+.2f}, max={x_arr.max():+.2f}"
        )
        if split == "train" and scaler.mean is not None:
            mean_preview = np.array2string(
                scaler.mean[:5], precision=3, suppress_small=True
            )
            std_preview = np.array2string(
                scaler.std[:5], precision=3, suppress_small=True
            )
            print(f"    scaler mean (prime 5)   : {mean_preview}")
            print(f"    scaler std  (prime 5)   : {std_preview}")

    # --- 4) Test di un batch reale dal DataLoader train ---------------------- #
    train_loader = loaders["train"]
    xb, yb = next(iter(train_loader))
    print(
        f"\n  Esempio di batch dal train DataLoader:\n"
        f"    x batch shape : {tuple(xb.shape)}  dtype={xb.dtype}\n"
        f"    y batch shape : {tuple(yb.shape)}  dtype={yb.dtype}"
    )


def main() -> None:
    _print_section("STEP 1/2 — Download dei dataset")
    paths = download_all("../data")

    _print_section("STEP 2/2 — Costruzione Dataset/DataLoader e proprietà")
    describe_dataset("ETTh1",        ETTh1Dataset,        paths["ETTh1"])
    describe_dataset("ETTh2",        ETTh2Dataset,        paths["ETTh2"])
    describe_dataset("ETTm1",        ETTm1Dataset,        paths["ETTm1"])
    describe_dataset("ETTm2",        ETTm2Dataset,        paths["ETTm2"])
    describe_dataset("Electricity",  ElectricityDataset,  paths["Electricity"])
    describe_dataset("Solar",        SolarDataset,        paths["Solar"])

    _print_section("FATTO ✓")
    print(
        "  I DataLoader sono pronti all'uso per il training.\n"
        "  Ogni iterazione restituisce una tupla (x, y) di tensori float32:\n"
        "    x : (batch, seq_len,            n_features_in)\n"
        "    y : (batch, label_len+pred_len, n_features_out)\n"
    )


if __name__ == "__main__":
    # Solo per riproducibilità
    torch.manual_seed(0)
    np.random.seed(0)
    main()
