"""
train_tpatchgnn.py
------------------
Addestra il modello tPatchGNN (repo ufficiale, vendorizzato in `./vendor`)
su uno dei dataset ETTh1/h2/m1/m2 sparsificati col nostro protocollo MCAR.

Esempi:
    # ETTh1, sparsity 0.3 MCAR, finestra 96, orizzonte 96
    python train_tpatchgnn.py --dataset etth1 --sparsity 0.3 \
        --seq_len 96 --pred_len 96 --epochs 50

    # Test rapido con epoch ridotte
    python train_tpatchgnn.py --dataset etth1 --epochs 3 --batch_size 16

Lo script:
    1) costruisce i tre DataLoader IMTS (train/val/test) con la sparsificazione
       richiesta;
    2) istanzia tPatchGNN con `args.ndim = D` (numero di canali) e gli iperparametri
       passati da CLI;
    3) gira un training loop con Adam, MSE loss, early stopping su val MSE,
       e salva il checkpoint del best model;
    4) valuta sul test set con MAE/RMSE/MSE.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# Le nostre classi
from src.dataset.datasets import (
    ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset, ElectricityDataset,
)
from src.dataset.irregular_datasets import SparsifyConfig, build_irregular_dataloaders

# Adapter + modello (registra anche `vendor/` nel sys.path)
from tpatchgnn_adapter import TPatchGNNArgs, to_tpatchgnn_batch, build_tpatchgnn

# Utility riusate dal repo originale (loss e metriche di valutazione)
from lib.evaluation import compute_error  # noqa: E402


DATASET_REGISTRY = {
    "etth1":       (ETTh1Dataset,       "../data/ETTh1.csv",       7),
    "etth2":       (ETTh2Dataset,       "../data/ETTh2.csv",       7),
    "ettm1":       (ETTm1Dataset,       "../data/ETTm1.csv",       7),
    "ettm2":       (ETTm2Dataset,       "../data/ETTm2.csv",       7),
    "electricity": (ElectricityDataset, "../data/electricity.csv", 321),
}


# ----------------------------------------------------------------------------- #
# CLI                                                                           #
# ----------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train tPatchGNN su ETT sparsificato")

    # Dataset
    p.add_argument("--dataset", type=str, default="etth1",
                   choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--csv_path", type=str, default=None,
                   help="Override del CSV path (default: ./data/<dataset>.csv)")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)

    # Sparsificazione
    p.add_argument("--mechanism", type=str, default="mcar",
                   choices=["mcar", "burst", "periodic", "async"])
    p.add_argument("--sparsity", type=float, default=0.3)
    p.add_argument("--mask_seed", type=int, default=0)

    # tPatchGNN hyperparams (default come repo originale)
    p.add_argument("--patch_size", type=float, default=0.125,
                   help="Dimensione patch in unità di tp normalizzati [0,1]")
    p.add_argument("--stride", type=float, default=0.125,
                   help="Stride patch in unità di tp normalizzati [0,1]")
    p.add_argument("--hid_dim", type=int, default=64)
    p.add_argument("--te_dim", type=int, default=10)
    p.add_argument("--node_dim", type=int, default=10)
    p.add_argument("--nlayer", type=int, default=1)
    p.add_argument("--nhead", type=int, default=1)
    p.add_argument("--tf_layer", type=int, default=1)
    p.add_argument("--hop", type=int, default=1)
    p.add_argument("--outlayer", type=str, default="Linear")

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)

    # I/O
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--device", type=str, default=None,
                   help="cuda | cpu (default: auto)")

    return p.parse_args()


# ----------------------------------------------------------------------------- #
# Training step e validation                                                    #
# ----------------------------------------------------------------------------- #
def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    args: TPatchGNNArgs,
    device: torch.device,
) -> float:
    """Una epoch di training. Ritorna la loss media (MSE)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        optimizer.zero_grad()
        bd = to_tpatchgnn_batch(batch, args, device)

        pred_y = model.forecasting(
            bd["tp_to_predict"], bd["observed_data"],
            bd["observed_tp"], bd["observed_mask"],
        )
        loss = compute_error(
            bd["data_to_predict"], pred_y,
            mask=bd["mask_predicted_data"], func="MSE", reduce="mean",
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    args: TPatchGNNArgs,
    device: torch.device,
) -> dict[str, float]:
    """Valuta MAE / MSE / RMSE su un dataloader. Replica la logica del repo
    originale: somma errori per variabile e normalizza per il numero di
    osservazioni effettive (utile quando la mask del target ha zeri)."""
    model.eval()
    se_sum = None
    ae_sum = None
    n_obs = None
    for batch in loader:
        bd = to_tpatchgnn_batch(batch, args, device)
        pred_y = model.forecasting(
            bd["tp_to_predict"], bd["observed_data"],
            bd["observed_tp"], bd["observed_mask"],
        )
        se_var, mask_count = compute_error(
            bd["data_to_predict"], pred_y,
            mask=bd["mask_predicted_data"], func="MSE", reduce="sum",
        )
        ae_var, _ = compute_error(
            bd["data_to_predict"], pred_y,
            mask=bd["mask_predicted_data"], func="MAE", reduce="sum",
        )
        if se_sum is None:
            se_sum = se_var.clone()
            ae_sum = ae_var.clone()
            n_obs = mask_count.clone()
        else:
            se_sum += se_var
            ae_sum += ae_var
            n_obs += mask_count

    n_avai = torch.count_nonzero(n_obs)
    mse = ((se_sum / (n_obs + 1e-8)).sum() / n_avai).item()
    mae = ((ae_sum / (n_obs + 1e-8)).sum() / n_avai).item()
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "rmse": rmse, "mae": mae}


# ----------------------------------------------------------------------------- #
# Main                                                                          #
# ----------------------------------------------------------------------------- #
def main() -> None:
    cli = parse_args()

    # Reproducibility
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    # Device
    if cli.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cli.device)
    print(f"[device] {device}")

    # --- Dataset / DataLoader ---------------------------------------------- #
    dataset_cls, default_csv, default_D = DATASET_REGISTRY[cli.dataset]
    csv_path = cli.csv_path or default_csv
    if not Path(csv_path).exists():
        print(f"[error] CSV non trovato: {csv_path}. Esegui prima `python download.py`.")
        sys.exit(1)

    sparsify_cfg = SparsifyConfig(
        mechanism=cli.mechanism,
        sparsity=cli.sparsity,
        seed=cli.mask_seed,
    )
    loaders = build_irregular_dataloaders(
        dataset_cls,
        csv_path=csv_path,
        seq_len=cli.seq_len,
        pred_len=cli.pred_len,
        sparsify_cfg=sparsify_cfg,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
    )
    train_ld = loaders["train"]
    val_ld = loaders["val"]
    test_ld = loaders["test"]
    D = train_ld.dataset.n_features
    print(f"[dataset] {cli.dataset} D={D}  L_in={cli.seq_len}  L_pred={cli.pred_len}")
    print(f"[dataset] sparsity={cli.sparsity} mechanism='{cli.mechanism}'")
    print(f"[dataset] train batches: {len(train_ld)}, val: {len(val_ld)}, test: {len(test_ld)}")

    # --- Modello ----------------------------------------------------------- #
    args = TPatchGNNArgs(
        ndim=D,
        history=1.0,                # tp osservati ∈ [0, 1] dopo normalizzazione
        patch_size=cli.patch_size,
        stride=cli.stride,
        hid_dim=cli.hid_dim,
        te_dim=cli.te_dim,
        node_dim=cli.node_dim,
        nlayer=cli.nlayer,
        nhead=cli.nhead,
        tf_layer=cli.tf_layer,
        hop=cli.hop,
        outlayer=cli.outlayer,
        device=device,
    )
    print(f"[model] tPatchGNN  npatch={args.npatch}  patch_size={args.patch_size}  "
          f"stride={args.stride}  hid_dim={args.hid_dim}")

    model = build_tpatchgnn(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] n_params = {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cli.lr, weight_decay=cli.w_decay)

    # --- Training loop ----------------------------------------------------- #
    ckpt_dir = Path(cli.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"tpatchgnn_{cli.dataset}_s{cli.sparsity}_{cli.mechanism}_seed{cli.seed}.pt"
    ckpt_path = ckpt_dir / ckpt_name

    best_val_mse = float("inf")
    best_epoch = -1
    best_test = None
    epochs_without_improve = 0

    for epoch in range(1, cli.epochs + 1):
        t0 = time.time()
        train_mse = train_one_epoch(model, train_ld, optimizer, args, device)
        val_metrics = evaluate(model, val_ld, args, device)
        elapsed = time.time() - t0

        improved = val_metrics["mse"] < best_val_mse
        if improved:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            epochs_without_improve = 0
            best_test = evaluate(model, test_ld, args, device)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "args": vars(cli),
                "val_metrics": val_metrics,
                "test_metrics": best_test,
            }, ckpt_path)
            tag = " ★"
        else:
            epochs_without_improve += 1
            tag = ""

        print(
            f"[epoch {epoch:03d}] "
            f"train MSE={train_mse:.5f} | "
            f"val MSE={val_metrics['mse']:.5f} MAE={val_metrics['mae']:.5f} | "
            f"test MSE={best_test['mse']:.5f} MAE={best_test['mae']:.5f} "
            f"({elapsed:.1f}s){tag}"
        )

        if epochs_without_improve >= cli.patience:
            print(f"[early stop] {cli.patience} epoch senza miglioramento. "
                  f"Best epoch = {best_epoch}.")
            break

    # --- Riepilogo finale --------------------------------------------------- #
    print("\n" + "═" * 60)
    print(f"  Best epoch        : {best_epoch}")
    print(f"  Val  MSE          : {best_val_mse:.5f}")
    if best_test is not None:
        print(f"  Test MSE / RMSE / MAE : "
              f"{best_test['mse']:.5f} / {best_test['rmse']:.5f} / {best_test['mae']:.5f}")
    print(f"  Checkpoint        : {ckpt_path}")
    print("═" * 60)


if __name__ == "__main__":
    main()
