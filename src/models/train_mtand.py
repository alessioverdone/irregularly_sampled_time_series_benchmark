"""
train_mtand.py
--------------
Addestra il modello mTAND (versione deterministica forecaster, basato sul
repo `reml-lab/mTAN`, vendorizzato in `./vendor/mtand`) su uno dei dataset
ETTh1/h2/m1/m2 sparsificati col nostro protocollo MCAR.

Esempio:
    python train_mtand.py --dataset etth1 --sparsity 0.3 \\
        --seq_len 96 --pred_len 96 --epochs 50

A differenza di tPatchGNN/Hi-Patch, mTAND non fa patching: lavora direttamente
su `(values, mask, tp)`. Niente PyG, niente compilazioni, leggero da
addestrare. Versione deterministica (no VAE/KL): l'encoder produce
2*latent_dim ma usiamo solo la "media".
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from src.dataset.datasets import (
    ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset, ElectricityDataset,
)
from src.dataset.irregular_datasets import SparsifyConfig, build_irregular_dataloaders

from mtand_adapter import (
    MTANDArgs, to_mtand_batch, build_mtand, compute_error,
)


DATASET_REGISTRY = {
    "etth1":       (ETTh1Dataset,       "../data/ETTh1.csv",       7),
    "etth2":       (ETTh2Dataset,       "../data/ETTh2.csv",       7),
    "ettm1":       (ETTm1Dataset,       "../data/ETTm1.csv",       7),
    "ettm2":       (ETTm2Dataset,       "../data/ETTm2.csv",       7),
    "electricity": (ElectricityDataset, "../data/electricity.csv", 321),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train mTAND su ETT sparsificato")
    p.add_argument("--dataset", type=str, default="etth1",
                   choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)

    p.add_argument("--mechanism", type=str, default="mcar",
                   choices=["mcar", "burst", "periodic", "async"])
    p.add_argument("--sparsity", type=float, default=0.3)
    p.add_argument("--mask_seed", type=int, default=0)

    # mTAND hyperparams
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--nhidden", type=int, default=32)
    p.add_argument("--embed_time", type=int, default=64,
                   help="Time embedding dim (deve essere divisibile per num_heads)")
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--n_ref_points", type=int, default=128,
                   help="Numero di reference query points dell'encoder mTAND")
    p.add_argument("--learn_emb", action="store_true", default=True,
                   help="Time embedding learnable (default) vs sinusoidal fisso")
    p.add_argument("--no_learn_emb", dest="learn_emb", action="store_false")

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0; n = 0
    for batch in loader:
        optimizer.zero_grad()
        bd = to_mtand_batch(batch, device)
        pred = model.forecasting(
            bd["tp_to_predict"], bd["observed_data"],
            bd["observed_tp"], bd["observed_mask"],
        )
        loss = compute_error(
            bd["data_to_predict"], pred,
            mask=bd["mask_predicted_data"], func="MSE", reduce="mean",
        )
        loss.backward()
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    se_sum = ae_sum = n_obs = None
    for batch in loader:
        bd = to_mtand_batch(batch, device)
        pred = model.forecasting(
            bd["tp_to_predict"], bd["observed_data"],
            bd["observed_tp"], bd["observed_mask"],
        )
        se_var, mc = compute_error(bd["data_to_predict"], pred,
            mask=bd["mask_predicted_data"], func="MSE", reduce="sum")
        ae_var, _ = compute_error(bd["data_to_predict"], pred,
            mask=bd["mask_predicted_data"], func="MAE", reduce="sum")
        if se_sum is None:
            se_sum, ae_sum, n_obs = se_var.clone(), ae_var.clone(), mc.clone()
        else:
            se_sum += se_var; ae_sum += ae_var; n_obs += mc
    n_avai = torch.count_nonzero(n_obs)
    mse = ((se_sum / (n_obs + 1e-8)).sum() / n_avai).item()
    mae = ((ae_sum / (n_obs + 1e-8)).sum() / n_avai).item()
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mae}


def main() -> None:
    cli = parse_args()
    torch.manual_seed(cli.seed); np.random.seed(cli.seed)

    device = torch.device(cli.device) if cli.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    dataset_cls, default_csv, _ = DATASET_REGISTRY[cli.dataset]
    csv_path = cli.csv_path or default_csv
    if not Path(csv_path).exists():
        print(f"[error] CSV non trovato: {csv_path}. Esegui `python download.py`.")
        sys.exit(1)

    sparsify_cfg = SparsifyConfig(
        mechanism=cli.mechanism, sparsity=cli.sparsity, seed=cli.mask_seed,
    )
    loaders = build_irregular_dataloaders(
        dataset_cls, csv_path=csv_path,
        seq_len=cli.seq_len, pred_len=cli.pred_len,
        sparsify_cfg=sparsify_cfg, batch_size=cli.batch_size,
        num_workers=cli.num_workers,
    )
    train_ld, val_ld, test_ld = loaders["train"], loaders["val"], loaders["test"]
    D = train_ld.dataset.n_features
    print(f"[dataset] {cli.dataset} D={D}  L_in={cli.seq_len}  L_pred={cli.pred_len}")
    print(f"[dataset] sparsity={cli.sparsity} mechanism='{cli.mechanism}'")
    print(f"[dataset] batches: train={len(train_ld)} val={len(val_ld)} test={len(test_ld)}")

    args = MTANDArgs(
        ndim=D,
        latent_dim=cli.latent_dim,
        nhidden=cli.nhidden,
        embed_time=cli.embed_time,
        num_heads=cli.num_heads,
        learn_emb=cli.learn_emb,
        n_ref_points=cli.n_ref_points,
        device=device,
    )
    print(f"[model] mTAND  latent_dim={args.latent_dim}  nhidden={args.nhidden}  "
          f"n_ref={args.n_ref_points}  embed_time={args.embed_time}")

    model = build_mtand(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] n_params = {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cli.lr, weight_decay=cli.w_decay)

    ckpt_dir = Path(cli.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"mtand_{cli.dataset}_s{cli.sparsity}_{cli.mechanism}_seed{cli.seed}.pt"

    best_val = float("inf")
    best_epoch = -1
    best_test = None
    bad = 0
    for epoch in range(1, cli.epochs + 1):
        t0 = time.time()
        train_mse = train_one_epoch(model, train_ld, optimizer, device)
        val = evaluate(model, val_ld, device)
        elapsed = time.time() - t0

        if val["mse"] < best_val:
            best_val = val["mse"]; best_epoch = epoch; bad = 0
            best_test = evaluate(model, test_ld, device)
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "args": vars(cli), "val_metrics": val, "test_metrics": best_test,
            }, ckpt_path)
            tag = " ★"
        else:
            bad += 1; tag = ""

        print(f"[epoch {epoch:03d}] "
              f"train MSE={train_mse:.5f} | "
              f"val MSE={val['mse']:.5f} MAE={val['mae']:.5f} | "
              f"test MSE={best_test['mse']:.5f} MAE={best_test['mae']:.5f} "
              f"({elapsed:.1f}s){tag}")

        if bad >= cli.patience:
            print(f"[early stop] best epoch = {best_epoch}.")
            break

    print("\n" + "═" * 60)
    print(f"  Best epoch        : {best_epoch}")
    print(f"  Val  MSE          : {best_val:.5f}")
    if best_test is not None:
        print(f"  Test MSE/RMSE/MAE : "
              f"{best_test['mse']:.5f} / {best_test['rmse']:.5f} / {best_test['mae']:.5f}")
    print(f"  Checkpoint        : {ckpt_path}")
    print("═" * 60)


if __name__ == "__main__":
    main()
