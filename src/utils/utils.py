import random
import os
import numpy as np
import torch
from torch import nn

from src.models.dlinear_adapter import build_dlinear
from src.models.hipatch_adapter import HiPatchArgs, build_hipatch
from src.models.mtand_adapter import MTANDArgs, build_mtand
from src.models.nbeats_adapter import build_nbeats
from src.models.tpatchgnn_adapter import TPatchGNNArgs, build_tpatchgnn


def set_params_wrt_dataset(run_params, loaders):
    train_ld, val_ld, test_ld = loaders["train"], loaders["val"], loaders["test"]

    if run_params.irregular_time_series:
        # Hi-patch  (pred_window relativo (per Hi-Patch.scale_patch_size))
        total_steps = run_params.seq_len + run_params.pred_len
        run_params.hipatch_pred_window = run_params.pred_len / max(1, total_steps - 1)

        # tpatch-gnn e mtanD
        run_params.D = train_ld.dataset.n_features
    else:
        # LSTM
        run_params.D = train_ld.dataset.data_x.shape[1]
    return run_params


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(hparams):
    if hparams.model == 'tpatch-gnn':
        args = TPatchGNNArgs(ndim=hparams.D,
                            history=1.0,  # tp osservati ∈ [0, 1] dopo normalizzazione
                            patch_size=hparams.patch_size,
                            stride=hparams.stride,
                            hid_dim=hparams.hid_dim,
                            te_dim=hparams.te_dim,
                            node_dim=hparams.node_dim,
                            nlayer=hparams.nlayer,
                            nhead=hparams.nhead,
                            tf_layer=hparams.tf_layer,
                            hop=hparams.hop,
                            outlayer=hparams.outlayer,
                            device=hparams.device)
        print(f"[model] tPatchGNN  npatch={args.npatch}  patch_size={args.patch_size}  "
              f"stride={args.stride}  hid_dim={args.hid_dim}")

        model = build_tpatchgnn(args)

    elif hparams.model == 'hi-patch':
        args = HiPatchArgs(ndim=hparams.D,
                           history=1.0,
                           pred_window=hparams.hipatch_pred_window,
                           patch_size=hparams.patch_size,
                           stride=hparams.stride,
                           hid_dim=hparams.hid_dim,
                           nlayer=hparams.nlayer,
                           nhead=hparams.nhead,
                           alpha=hparams.alpha,
                           res=hparams.res,
                           device=hparams.device)
        print(f"[model] Hi-Patch  npatch={args.npatch}  patch_layer={args.patch_layer}  "
              f"patch_size={args.patch_size}  hid_dim={args.hid_dim}")

        model = build_hipatch(args)

    elif hparams.model == 'mtand':
        args = MTANDArgs(
            ndim=hparams.D,
            latent_dim=hparams.latent_dim,
            nhidden=hparams.nhidden,
            embed_time=hparams.embed_time,
            num_heads=hparams.num_heads,
            learn_emb=hparams.learn_emb,
            n_ref_points=hparams.n_ref_points,
            device=hparams.device,
        )
        print(f"[model] mTAND  latent_dim={args.latent_dim}  nhidden={args.nhidden}  "
              f"n_ref={args.n_ref_points}  embed_time={args.embed_time}")
        model = build_mtand(args)

    elif hparams.model == 'lstm':
        args = None

        class ExtractLSTMOutput(nn.Module):
            def __init__(self, lstm):
                super().__init__()
                self.lstm = lstm

            def forward(self, x):
                output, _ = self.lstm(x)
                return output

        print(f"[model] LSTM  latent_dim={hparams.latent_dim}")
        model = nn.Sequential(
            ExtractLSTMOutput(nn.LSTM(input_size=hparams.D,
                                      hidden_size=hparams.latent_dim,
                                      num_layers=hparams.num_layers,
                                      bias=True,
                                      dropout=hparams.dropout,
                                      batch_first=True,
                                      bidirectional=False)),
            nn.Linear(hparams.latent_dim, hparams.D)
        )

    elif hparams.model == 'dlinear':
        args = None
        print(
            f"[model] DLinear  kernel_size={hparams.dlinear_kernel_size}"
            f"  individual={hparams.dlinear_individual}"
        )
        model = build_dlinear(
            seq_len=hparams.seq_len,
            pred_len=hparams.pred_len,
            n_features=hparams.D,
            kernel_size=hparams.dlinear_kernel_size,
            individual=hparams.dlinear_individual,
        )

    elif hparams.model == 'nbeats':
        args = None
        print(
            f"[model] N-BEATS  n_stacks={hparams.nbeats_n_stacks}"
            f"  layer_width={hparams.nbeats_layer_width}"
            f"  n_fc_layers={hparams.nbeats_n_fc_layers}"
            f"  expansion_coeff_dim={hparams.nbeats_expansion_coeff_dim}"
        )
        model = build_nbeats(
            seq_len=hparams.seq_len,
            pred_len=hparams.pred_len,
            n_features=hparams.D,
            n_stacks=hparams.nbeats_n_stacks,
            layer_width=hparams.nbeats_layer_width,
            n_fc_layers=hparams.nbeats_n_fc_layers,
            expansion_coeff_dim=hparams.nbeats_expansion_coeff_dim,
            dropout=hparams.dropout,
        )

    else:
        raise Exception('Error in select the model!')

    if hparams.verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] n_params = {n_params:,}")

    model = model.to(hparams.device)
    return model, args


def initialize_log_parameters(cont: int, combo: dict) -> dict:
    METRICS = ['mse', 'rmse', 'mae']
    SPLITS = ['val', 'test']

    # colonne metriche: val_mse_mean, val_mse_std, ...
    metric_keys = [f'{split}_{metric}_{stat}'
                   for split in SPLITS
                   for metric in METRICS
                   for stat in ('mean', 'std')]

    grid_params = {'Run': cont, **combo, **{k: 0. for k in metric_keys}}

    print(' '.join(f'{k}: {v}' for k, v in grid_params.items()))
    return grid_params


def update_seed_metrics(model, res_test, val_results, test_results):
    best_val_mse, best_val_rmse, best_val_mae = model.best_mse, model.best_rmse, model.best_mae

    # Testing
    test_mse = res_test[0]['test_mse']
    test_rmse = res_test[0]['test_rmse']
    test_mae = res_test[0]['test_mae']

    val_results.append([best_val_mse, best_val_rmse, best_val_mae])
    test_results.append([test_mse, test_rmse, test_mae])

    print(f'best_val_mse: {best_val_mse}')
    print(f'best_val_rmse: {best_val_rmse}')
    print(f'best_val_mae: {best_val_mae}')
    print(f'test_mse: {test_mse}')
    print(f'test_rmse {test_rmse}')
    print(f'test_mae: {test_mae}')
    return val_results, test_results


def update_run_metrics(val_results,
                       test_results,
                       grid_params_dict,
                       run_params):
    metrics = ['mse', 'rmse', 'mae']
    splits = ['val', 'test']
    results = {'val':  torch.tensor(val_results),
               'test': torch.tensor(test_results)}

    grid_params_dict.update({f'{split}_{metric}_{stat}': float(getattr(torch, stat)(results[split][:, i]))
                            for split in splits
                            for i, metric in enumerate(metrics)
                            for stat in ('mean', 'std')})

    print(' '.join(f'{k}: {v}' for k, v in grid_params_dict.items()))
    output_string = ' '.join([f'{k}: {v}' for k, v in grid_params_dict.items()])

    if run_params.save_logs:
        os.makedirs(run_params.logs_dir, exist_ok=True)
        with open(os.path.join(run_params.logs_dir, 'log.txt'), 'a') as file:
            print(output_string, file=file)

