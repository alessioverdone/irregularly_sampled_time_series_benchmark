# import itertools
# import time
#
# import torch
# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm
# from src.utils.mlflow_utils import log_metrics, log_artifact
#
# _BAR_FMT = "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
#
#
# def _pbar(loader, desc, enable, total=None, **kwargs):
#     return tqdm(
#         loader,
#         desc=desc,
#         total=total,
#         leave=False,
#         disable=not enable,
#         unit="batch",
#         dynamic_ncols=True,
#         bar_format=_BAR_FMT,
#     )
#
#
# def train(training, dataModuleInstance, run_params):
#     # Optimizer
#     training.configure_optimizers()
#
#     # Log
#
#     # Early stopping
#     early_stop_counter = 0
#     log_every = 50
#     early_stop_best = float('inf')
#
#     # Training
#     for epoch in range(run_params.max_epochs):
#         training.model.train()
#         train_metrics_accum = defaultdict(list)
#         start_time = time.time()
#
#         n_train = len(dataModuleInstance["train"])
#         pbar = _pbar(
#             itertools.islice(dataModuleInstance["train"], n_train),
#             desc=f"Epoch {epoch + 1}/{run_params.max_epochs} [train]",
#             enable=run_params.enable_progress_bar,
#             total=n_train,
#             mininterval=1.0,
#             miniters=50,
#         )
#         for step, batch in enumerate(pbar):
#             if batch is not None and isinstance(batch, dict):
#                 batch_dict = {k: v.to(run_params.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#             else:
#                 batch_dict = [v.to(run_params.device) if isinstance(v, torch.Tensor) else v for v in batch]
#
#             metrics = training.training_step(batch_dict)
#             for k, v in metrics.items():
#                 train_metrics_accum[k].append(v)
#
#             if (step + 1) % log_every == 0:
#                 pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in train_metrics_accum.items()})
#
#         avg_train = {k: float(np.mean(v)) for k, v in train_metrics_accum.items()}
#
#         if run_params.enable_progress_bar:
#             print(f"Epoch {epoch + 1}/{run_params.max_epochs} | " +
#                   " | ".join(f"{k}: {v:.4f}" for k, v in avg_train.items()))
#         print(f'Epoch time: {time.time() - start_time}')
#         if (epoch + 1) % run_params.check_val_every_n_epoch == 0:
#             avg_val = validate(training, dataModuleInstance, run_params)
#             training.on_validation_epoch_end(avg_val)
#             training.scheduler.step(avg_val['val_mse'])
#
#             if run_params.enable_progress_bar:
#                 print("  Val | " + " | ".join(f"{k}: {v:.4f}" for k, v in avg_val.items()))
#
#             if run_params.early_stop_callback_flag:
#                 if avg_val['val_mse'] < early_stop_best:
#                     early_stop_best = avg_val['val_mse']
#                     early_stop_counter = 0
#                 else:
#                     early_stop_counter += 1
#                     if early_stop_counter >= run_params.early_stop_patience:
#                         print(f"Early stopping at epoch {epoch + 1}")
#                         break
#
#
# def validate(training, dataModuleInstance, run_params):
#     training.model.eval()
#     val_metrics_accum = defaultdict(list)
#
#     with torch.no_grad():
#         n_val = len(dataModuleInstance["val"])
#         pbar = _pbar(itertools.islice(dataModuleInstance["val"], n_val),
#                      desc="  [val]",
#                      enable=run_params.enable_progress_bar,
#                      total=n_val)
#         for batch in pbar:
#             if batch is not None and isinstance(batch, dict):
#                 batch_dict = {k: v.to(run_params.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#             else:
#                 batch_dict = [v.to(run_params.device) if isinstance(v, torch.Tensor) else v for v in batch]
#
#             metrics = training.validation_step(batch_dict)
#             for k, v in metrics.items():
#                 val_metrics_accum[k].append(v)
#
#             pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in val_metrics_accum.items()})
#
#     return {k: float(np.mean(v)) for k, v in val_metrics_accum.items()}
#
#
# def test(training, dataModuleInstance, run_params):
#     training.model.eval()
#     test_metrics_accum = defaultdict(list)
#
#     with torch.no_grad():
#         n_test = len(dataModuleInstance["test"])
#         pbar = _pbar(
#             itertools.islice(dataModuleInstance["test"], n_test),
#             desc="  [test]",
#             enable=run_params.enable_progress_bar,
#             total=n_test,
#         )
#         pbar.leave = True
#         for batch in pbar:
#             if batch is not None and isinstance(batch, dict):
#                 batch_dict = {k: v.to(run_params.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#             else:
#                 batch_dict = [v.to(run_params.device) if isinstance(v, torch.Tensor) else v for v in batch]
#             metrics = training.test_step(batch_dict)
#             for k, v in metrics.items():
#                 test_metrics_accum[k].append(v)
#
#             pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in test_metrics_accum.items()})
#
#     avg_test = {k: float(np.mean(v)) for k, v in test_metrics_accum.items()}
#
#     print("Test results: " + " | ".join(f"{k}: {v:.4f}" for k, v in [avg_test][0].items()))
#     return [avg_test]

import itertools
import time
import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from src.utils.mlflow_utils import log_metrics, log_artifact

_BAR_FMT = "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"


def _pbar(loader, desc, enable, total=None, **kwargs):
    return tqdm(loader, desc=desc, total=total, leave=False,
                disable=not enable, unit="batch", dynamic_ncols=True,
                bar_format=_BAR_FMT, **kwargs)


def _to_device(batch, device):
    if batch is not None and isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]


def train(training, dataModuleInstance, run_params, trial=None):
    training.configure_optimizers()

    log_every = 50
    early_stop_counter = 0
    early_stop_best = float('inf')
    best_val_mse = float('inf')

    # checkpoint dir
    ckpt_dir = getattr(run_params, "checkpoint_dir", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")

    for epoch in range(run_params.max_epochs):
        training.model.train()
        train_metrics_accum = defaultdict(list)
        start_time = time.time()

        n_train = len(dataModuleInstance["train"])
        pbar = _pbar(
            itertools.islice(dataModuleInstance["train"], n_train),
            desc=f"Epoch {epoch + 1}/{run_params.max_epochs} [train]",
            enable=run_params.enable_progress_bar,
            total=n_train, mininterval=1.0, miniters=50,
        )
        for step, batch in enumerate(pbar):
            batch_dict = _to_device(batch, run_params.device)
            metrics = training.training_step(batch_dict)
            for k, v in metrics.items():
                train_metrics_accum[k].append(v)

            if (step + 1) % log_every == 0:
                pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in train_metrics_accum.items()})

        avg_train = {k: float(np.mean(v)) for k, v in train_metrics_accum.items()}
        epoch_time = time.time() - start_time

        if run_params.enable_progress_bar:
            print(f"Epoch {epoch + 1}/{run_params.max_epochs} | " +
                  " | ".join(f"{k}: {v:.4f}" for k, v in avg_train.items()))
        print(f'Epoch time: {epoch_time}')

        # ── MLflow: log epoch train metrics ─────────────────────────────
        log_metrics(run_params, {**avg_train, "epoch_time": epoch_time}, step=epoch)

        # Validation
        if (epoch + 1) % run_params.check_val_every_n_epoch == 0:
            avg_val = validate(training, dataModuleInstance, run_params)
            training.on_validation_epoch_end(avg_val)
            training.scheduler.step(avg_val['val_mse'])

            if run_params.enable_progress_bar:
                print("  Val | " + " | ".join(f"{k}: {v:.4f}" for k, v in avg_val.items()))

            # ── MLflow: log epoch val metrics ───────────────────────────
            log_metrics(run_params, avg_val, step=epoch)

            # ── MLflow: save & log best checkpoint ──────────────────────
            if avg_val['val_mse'] < best_val_mse:
                best_val_mse = avg_val['val_mse']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': training.model.state_dict(),
                    'optimizer_state_dict': training.optimizer.state_dict(),
                    'val_mse': avg_val['val_mse'],
                }, best_ckpt_path)
                log_artifact(run_params, best_ckpt_path, artifact_path="checkpoints")

            # Early stopping
            if run_params.early_stop_callback_flag:
                if avg_val['val_mse'] < early_stop_best:
                    early_stop_best = avg_val['val_mse']
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= run_params.early_stop_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Optuna per-epoch pruning (trial=None quando chiamato fuori da Fase 1)
            if trial is not None:
                trial.report(avg_val['val_mse'], step=epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

    # ── MLflow: summary best metrics ────────────────────────────────────
    log_metrics(run_params, {
        "best_val_mse":  training.best_mse,
        "best_val_rmse": training.best_rmse,
        "best_val_mae":  training.best_mae,
    })


def validate(training, dataModuleInstance, run_params):
    training.model.eval()
    val_metrics_accum = defaultdict(list)

    with torch.no_grad():
        n_val = len(dataModuleInstance["val"])
        pbar = _pbar(itertools.islice(dataModuleInstance["val"], n_val),
                     desc="  [val]", enable=run_params.enable_progress_bar, total=n_val)
        for batch in pbar:
            batch_dict = _to_device(batch, run_params.device)
            metrics = training.validation_step(batch_dict)
            for k, v in metrics.items():
                val_metrics_accum[k].append(v)
            pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in val_metrics_accum.items()})

    return {k: float(np.mean(v)) for k, v in val_metrics_accum.items()}


def test(training, dataModuleInstance, run_params):
    training.model.eval()
    test_metrics_accum = defaultdict(list)

    with torch.no_grad():
        n_test = len(dataModuleInstance["test"])
        pbar = _pbar(itertools.islice(dataModuleInstance["test"], n_test),
                     desc="  [test]", enable=run_params.enable_progress_bar, total=n_test)
        pbar.leave = True
        for batch in pbar:
            batch_dict = _to_device(batch, run_params.device)
            metrics = training.test_step(batch_dict)
            for k, v in metrics.items():
                test_metrics_accum[k].append(v)
            pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in test_metrics_accum.items()})

    avg_test = {k: float(np.mean(v)) for k, v in test_metrics_accum.items()}
    print("Test results: " + " | ".join(f"{k}: {v:.4f}" for k, v in avg_test.items()))

    # ── MLflow: log test metrics ────────────────────────────────────────
    log_metrics(run_params, avg_test)

    return [avg_test]