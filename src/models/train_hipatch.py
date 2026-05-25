"""
train_hipatch.py
----------------
Addestra il modello Hi-Patch (repo `qianlima-lab/Hi-Patch`, vendorizzato in
`./vendor/hipatch`) su uno dei dataset ETTh1/h2/m1/m2 sparsificati col
nostro protocollo MCAR.

Esempio:
    python train_hipatch.py --dataset etth1 --sparsity 0.3 \\
        --seq_len 96 --pred_len 96 --epochs 50

Dipendenze extra rispetto a tPatchGNN: torch_geometric, torch_scatter (PyG).
Su ambienti standard:
    pip install torch_geometric torch_scatter \\
        -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
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

from hipatch_adapter import (
    HiPatchArgs, to_hipatch_batch, build_hipatch, compute_error,
)


DATASET_REGISTRY = {
    "etth1":       (ETTh1Dataset,       "../data/ETTh1.csv",       7),
    "etth2":       (ETTh2Dataset,       "../data/ETTh2.csv",       7),
    "ettm1":       (ETTm1Dataset,       "../data/ETTm1.csv",       7),
    "ettm2":       (ETTm2Dataset,       "../data/ETTm2.csv",       7),
    "electricity": (ElectricityDataset, "../data/electricity.csv", 321),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train Hi-Patch su ETT sparsificato")
    p.add_argument("--dataset", type=str, default="etth1",
                   choices=list(DATASET_REGISTRY.keys()))
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)

    p.add_argument("--mechanism", type=str, default="mcar",
                   choices=["mcar", "burst", "periodic", "async"])
    p.add_argument("--sparsity", type=float, default=0.3)
    p.add_argument("--mask_seed", type=int, default=0)

    p.add_argument("--patch_size", type=float, default=0.125)
    p.add_argument("--stride", type=float, default=0.125)
    p.add_argument("--hid_dim", type=int, default=64)
    p.add_argument("--nlayer", type=int, default=1)
    p.add_argument("--nhead", type=int, default=1)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--res", type=float, default=1.0)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, args, device) -> float:
    model.train()
    total = 0.0; n = 0
    for batch in loader:
        optimizer.zero_grad()
        bd = to_hipatch_batch(batch, args, device)
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
def evaluate(model, loader, args, device) -> dict[str, float]:
    model.eval()
    se_sum = ae_sum = n_obs = None
    for batch in loader:
        bd = to_hipatch_batch(batch, args, device)
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
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

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

    # pred_window relativo (per Hi-Patch.scale_patch_size)
    total_steps = cli.seq_len + cli.pred_len
    pred_window = cli.pred_len / max(1, total_steps - 1)

    args = HiPatchArgs(
        ndim=D, history=1.0, pred_window=pred_window,
        patch_size=cli.patch_size, stride=cli.stride,
        hid_dim=cli.hid_dim, nlayer=cli.nlayer, nhead=cli.nhead,
        alpha=cli.alpha, res=cli.res,
        device=device,
    )
    print(f"[model] Hi-Patch  npatch={args.npatch}  patch_layer={args.patch_layer}  "
          f"patch_size={args.patch_size}  hid_dim={args.hid_dim}")

    model = build_hipatch(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] n_params = {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cli.lr, weight_decay=cli.w_decay)

    ckpt_dir = Path(cli.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"hipatch_{cli.dataset}_s{cli.sparsity}_{cli.mechanism}_seed{cli.seed}.pt"

    best_val = float("inf")
    best_epoch = -1
    best_test = None
    bad = 0
    for epoch in range(1, cli.epochs + 1):
        t0 = time.time()
        train_mse = train_one_epoch(model, train_ld, optimizer, args, device)
        val = evaluate(model, val_ld, args, device)
        elapsed = time.time() - t0

        if val["mse"] < best_val:
            best_val = val["mse"]; best_epoch = epoch; bad = 0
            best_test = evaluate(model, test_ld, args, device)
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
