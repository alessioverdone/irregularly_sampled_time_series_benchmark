"""
main.py
-------
Pipeline end-to-end:
    1) scarica i dataset (ETTh1, ETTh2, ETTm1, ETTm2, Electricity) in ./data/
    2) costruisce Dataset + DataLoader regolari (train / val / test) per ognuno
    3) costruisce Dataset + DataLoader IMTS in formato canonico
       compatibile con tPatchGNN / Hi-Patch / mTAND, applicando una
       sparsificazione MCAR placeholder
    4) stampa le proprietà di entrambe le versioni (regolare e irregolare)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset.download import download_all
from src.dataset.datasets import (
    ETTh1Dataset,
    ETTh2Dataset,
    ETTm1Dataset,
    ETTm2Dataset,
    build_dataloaders,
)
from irregular_datasets import (
    SparsifyConfig,
    build_irregular_dataloaders,
)


def _print_section(title: str) -> None:
    line = "═" * 70
    print(f"\n{line}\n  {title}\n{line}")


# ----------------------------------------------------------------------------- #
# Descrizione di un dataset regolare                                            #
# ----------------------------------------------------------------------------- #
def describe_dataset(name: str, dataset_cls, csv_path: Path) -> None:
    _print_section(f"Dataset regolare: {name}")

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

    # --- 2) Costruzione DataLoader regolari -------------------------------- #
    seq_len, label_len, pred_len = 96, 48, 96
    loaders = build_dataloaders(
        dataset_cls,
        csv_path=csv_path,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        features="M",
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
        print(f"\n  ── Split '{split}' ────────────────────────────────")
        print(f"    n. campioni (finestre)  : {len(ds):>8d}")
        print(f"    n. batch DataLoader     : {len(loader):>8d}  (batch_size=32)")
        print(f"    target                  : '{ds.target}'")
        print(f"    n_features_in           : {ds.n_features_in}")
        print(f"    n_features_out          : {ds.n_features_out}")
        print(
            f"    shape singolo campione  : "
            f"x={tuple(x_sample.shape)}, y={tuple(y_sample.shape)}"
        )
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

    # --- 4) Test di un batch reale dal DataLoader train -------------------- #
    train_loader = loaders["train"]
    xb, yb = next(iter(train_loader))
    print(
        f"\n  Esempio di batch dal train DataLoader:\n"
        f"    x batch shape : {tuple(xb.shape)}  dtype={xb.dtype}\n"
        f"    y batch shape : {tuple(yb.shape)}  dtype={yb.dtype}"
    )


# ----------------------------------------------------------------------------- #
# Descrizione di un dataset IMTS irregolare                                     #
# ----------------------------------------------------------------------------- #
def describe_irregular_dataset(
    name: str,
    dataset_cls,
    csv_path: Path,
    sparsity: float = 0.3,
) -> None:
    _print_section(f"Dataset IMTS (irregolare): {name}  [sparsity={sparsity}]")

    seq_len, pred_len = 96, 96
    cfg = SparsifyConfig(mechanism="mcar", sparsity=sparsity, seed=0)
    loaders = build_irregular_dataloaders(
        dataset_cls,
        csv_path=csv_path,
        seq_len=seq_len,
        pred_len=pred_len,
        features="M",
        sparsify_cfg=cfg,
        batch_size=32,
        num_workers=0,
    )

    print(
        f"  Configurazione finestra : seq_len={seq_len}, pred_len={pred_len}"
    )
    print(f"  Meccanismo missingness  : '{cfg.mechanism}', sparsity={cfg.sparsity}")
    print(f"  Seed base per maschere  : {cfg.seed}")

    for split, loader in loaders.items():
        ds = loader.dataset
        sample = ds[0]
        print(f"\n  ── Split '{split}' ────────────────────────────────")
        print(f"    n. campioni (finestre)  : {len(ds):>8d}")
        print(f"    n. batch DataLoader     : {len(loader):>8d}  (batch_size=32)")
        print(f"    n_features (D)          : {ds.n_features}")
        print(f"    shape singolo campione (formato canonico IMTS):")
        for key, tensor in sample.items():
            print(f"      {key:20s}: {tuple(tensor.shape)}  dtype={tensor.dtype}")

        # Statistiche di sparsità effettiva
        obs_mask = sample["observed_mask"]
        L_obs = sample["observed_data"].shape[0]
        kept_frac = obs_mask.mean().item()
        print(
            f"    L_obs (tp con ≥1 obs)   : {L_obs} / {seq_len} "
            f"(compressione = {L_obs / seq_len:.1%})"
        )
        print(
            f"    frazione (t,d) osservati: {kept_frac:.1%}  "
            f"(atteso ≈ {1.0 - cfg.sparsity:.1%})"
        )

    # Batch reale dal train: mostra padding a L_obs_max
    train_loader = loaders["train"]
    batch = next(iter(train_loader))
    print(f"\n  Esempio di batch dal train DataLoader (post-collate):")
    for key, tensor in batch.items():
        print(f"    {key:20s}: {tuple(tensor.shape)}  dtype={tensor.dtype}")
    L_obs_max = batch["observed_data"].shape[1]
    avg_valid = batch["padding_mask"].sum(dim=1).float().mean().item()
    print(
        f"    L_obs_max nel batch     : {L_obs_max}  "
        f"(media tp validi per sample = {avg_valid:.1f})"
    )


# ----------------------------------------------------------------------------- #
# Entry-point                                                                   #
# ----------------------------------------------------------------------------- #
def main() -> None:
    _print_section("STEP 1/3 — Download dei dataset")
    paths = download_all("../../data")

    _print_section("STEP 2/3 — Dataset/DataLoader regolari")
    describe_dataset("ETTh1", ETTh1Dataset, paths["ETTh1"])
    describe_dataset("ETTh2", ETTh2Dataset, paths["ETTh2"])
    describe_dataset("ETTm1", ETTm1Dataset, paths["ETTm1"])
    describe_dataset("ETTm2", ETTm2Dataset, paths["ETTm2"])
    # Electricity escluso di default per dimensioni; decommenta se serve.
    # describe_dataset("Electricity", ElectricityDataset, paths["Electricity"])

    _print_section("STEP 3/3 — Dataset/DataLoader IMTS (formato canonico)")
    describe_irregular_dataset("ETTh1", ETTh1Dataset, paths["ETTh1"], sparsity=0.3)
    describe_irregular_dataset("ETTh2", ETTh2Dataset, paths["ETTh2"], sparsity=0.5)
    describe_irregular_dataset("ETTm1", ETTm1Dataset, paths["ETTm1"], sparsity=0.3)
    describe_irregular_dataset("ETTm2", ETTm2Dataset, paths["ETTm2"], sparsity=0.7)

    _print_section("FATTO ✓")
    print(
        "  - I DataLoader regolari restituiscono coppie (x, y) per modelli LTSF.\n"
        "  - I DataLoader IMTS restituiscono dict in formato canonico IMTS:\n"
        "      observed_data       (B, L_obs_max, D)\n"
        "      observed_tp         (B, L_obs_max)\n"
        "      observed_mask       (B, L_obs_max, D)\n"
        "      padding_mask        (B, L_obs_max)\n"
        "      data_to_predict     (B, L_pred,    D)\n"
        "      tp_to_predict       (B, L_pred)\n"
        "      mask_predicted_data (B, L_pred,    D)\n"
        "    Drop-in compatibile con i loader di tPatchGNN, Hi-Patch e mTAND.\n"
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()
