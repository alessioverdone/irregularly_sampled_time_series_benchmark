import os
from pathlib import Path
from pprint import pformat
from random import SystemRandom

import numpy as np
import torch
import yaml

from src.dataset.datasets import ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset, ElectricityDataset, SolarDataset


# ------------------------------------------------------------------------------- #
# Split configurations per dataset                                                #
# Modificare qui per cambiare i bordi di train/val/test per tutti i dataset.      #
# "train_end"/"val_end"/"test_end" : indici assoluti di timestep                  #
# "train_frac"/"val_frac"          : frazioni (test_end = total_len)              #
# ------------------------------------------------------------------------------- #
DATASET_SPLIT_CONFIGS: dict[str, dict] = {
    "etth1":       {"train_end": 12 * 30 * 24,     "val_end": 16 * 30 * 24,     "test_end": 20 * 30 * 24},
    "etth2":       {"train_end": 12 * 30 * 24,     "val_end": 16 * 30 * 24,     "test_end": 20 * 30 * 24},
    "ettm1":       {"train_end": 12 * 30 * 24 * 4, "val_end": 16 * 30 * 24 * 4, "test_end": 20 * 30 * 24 * 4},
    "ettm2":       {"train_end": 12 * 30 * 24 * 4, "val_end": 16 * 30 * 24 * 4, "test_end": 20 * 30 * 24 * 4},
    "electricity": {"train_frac": 0.7, "val_frac": 0.8},
    "solar":       {"train_frac": 0.7, "val_frac": 0.8},
}


def compute_split_borders(split_config: dict, total_len: int) -> tuple[int, int, int]:
    """Calcola (train_end, val_end, test_end) come indici assoluti di timestep."""
    if "train_end" in split_config:
        return (
            min(split_config["train_end"], total_len),
            min(split_config["val_end"],   total_len),
            min(split_config["test_end"],  total_len),
        )
    elif "train_frac" in split_config:
        return (
            int(total_len * split_config["train_frac"]),
            int(total_len * split_config["val_frac"]),
            total_len,
        )
    raise ValueError(
        f"split_config deve avere 'train_end'/'val_end'/'test_end' o "
        f"'train_frac'/'val_frac': {split_config}"
    )


class Parameters:
    def __init__(self):
        project_dir = Path(__file__).resolve().parents[1]
        # self.DATASET_REGISTRY = {
        #     "etth1": (ETTh1Dataset, "../data/ETTh1.csv", 7),
        #     "etth2": (ETTh2Dataset, "../data/ETTh2.csv", 7),
        #     "ettm1": (ETTm1Dataset, "../data/ETTm1.csv", 7),
        #     "ettm2": (ETTm2Dataset, "../data/ETTm2.csv", 7),
        #     "electricity": (ElectricityDataset, "../data/electricity.csv", 321),
        # }

        # Path
        self.dataset = "ettm2"  #  ['etth1', 'etth2', 'ettm1', 'ettm2', 'electricity', 'solar']
        self.data_dir = os.path.join(project_dir, 'data')
        self.data_dir_irr = os.path.join(self.data_dir, 'irregular')
        self.registry_dir = os.path.join(project_dir, 'registry')
        self.logs_dir = os.path.join(self.registry_dir, 'logs')
        self.ckpt_dir = os.path.join(self.registry_dir, 'checkpoints')

        self.config_path = os.path.join(project_dir, 'registry', 'configurations')
        self.csv_path = None
        # Fonte dei CSV regolari (usata sia per dataset regolari che per la
        # generazione di quelli irregolari statici).
        self.DATASET_REGISTRY = {
            "etth1":       (ETTh1Dataset,       os.path.join(self.data_dir, 'regular', 'ETTh1.csv'),       7),
            "etth2":       (ETTh2Dataset,       os.path.join(self.data_dir, 'regular', 'ETTh2.csv'),       7),
            "ettm1":       (ETTm1Dataset,       os.path.join(self.data_dir, 'regular', 'ETTm1.csv'),       7),
            "ettm2":       (ETTm2Dataset,       os.path.join(self.data_dir, 'regular', 'ETTm2.csv'),       7),
            "electricity": (ElectricityDataset, os.path.join(self.data_dir, 'regular', 'electricity.csv'), 321),
            "solar":       (SolarDataset,       os.path.join(self.data_dir, 'regular', 'solar_AL.csv'),    137),
        }

        # Split configurations (modificabili qui per cambiare i bordi di split).
        self.split_configs = dict(DATASET_SPLIT_CONFIGS)

        # Main
        self.model = 'dlinear'  # ['mtand', 'hi-patch', 'tpatch-gnn', 'lstm']
        self.irregular_time_series = True
        self.irregular_time_series_pattern = 'static'  # ['static' 'dynamic']
        self.mechanism = "async"  # ['mcar', 'burst', 'periodic', 'async']

        # Data
        self.seq_len = 96
        self.pred_len = 96
        self.label_len = 0
        self.sparsity = 0.5
        self.mask_seed = 0

        # Trainer parameters
        self.accelerator = 'gpu'
        self.data_device = 'cpu'
        self.device = torch.device('cuda' if self.accelerator == 'gpu' and torch.cuda.is_available() else 'cpu')
        self.max_epochs = 5
        self.check_val_every_n_epoch = 2
        self.enable_progress_bar = True
        self.log_every_n_steps = 300
        self.lr = 1e-3
        self.lr_patience = 2
        self.lr_factor = 0.8
        self.w_decay = 0.0
        self.early_stop_callback_flag = False
        self.early_stop_patience = 5
        self.dropout = 0.0
        self.logging = True
        self.save_ckpts = False
        self.save_logs = True
        self.tracking_uri = f"sqlite:///{os.path.join(project_dir, 'mlflow.db')}"
        self.experiment_name = "benchmark_irr_v2.0"
        self.run_name = None  # None → generato automaticamente come "<model>_<dataset>_seed<seed>"
        self.reproducible = True
        self.exp_id = 0
        self.seed = 456  # 42
        self.batch_size = 32
        self.num_workers = 12
        self.prefetch_factor = 32
        self.compile_model = False
        self.verbose = True

        # LSTM
        self.num_layers = 2

        # DLinear
        self.dlinear_kernel_size = 25    # kernel moving average per decomposizione trend
        self.dlinear_individual = False  # True = un layer lineare per feature

        # N-BEATS
        self.nbeats_n_stacks = 30             # blocchi in serie
        self.nbeats_layer_width = 256         # larghezza layer FC
        self.nbeats_n_fc_layers = 4           # layer FC per blocco
        self.nbeats_expansion_coeff_dim = 32  # dim coefficienti θ (backcast/forecast)

        # Hi-patch
        self.patch_size = 0.125
        self.stride = 0.125
        self.hid_dim = 64
        self.nlayer = 1
        self.nhead = 1
        self.alpha = 1.0
        self.res = 1.0
        self.hipatch_pred_window = 0

        # m-tand
        self.latent_dim = 16
        self.nhidden = 32
        self.embed_time = 64
        self.num_heads = 1
        self.n_ref_points = 128
        self.learn_emb = True

        # tPatchGNN
        self.patch_size = 0.125
        self.stride = 0.125
        self.hid_dim = 64
        self.te_dim = 10
        self.node_dim = 10
        self.nlayer = 1
        self.nhead = 1
        self.tf_layer = 1
        self.hop = 1
        self.outlayer = "Linear"


    def get_irr_save_dir(self) -> str:
        """Restituisce la cartella dove salvare/caricare il dataset irregolare statico.

        Struttura: data/irregular/<dataset>/<seq>_<pred>_<mechanism>_sp<sparsity>_seed<seed>/
        La cartella identifica univocamente la configurazione di sparsificazione,
        così run con modelli diversi ma stessa configurazione condividono i dati.
        """
        subdir = (
            f"seq{self.seq_len}_pred{self.pred_len}"
            f"_{self.mechanism}_sp{self.sparsity:.3f}_seed{self.mask_seed}"
        )
        return os.path.join(self.data_dir, 'irregular', self.dataset, subdir)

    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(vars(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'Parameters':
        instance = cls()
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        for key, value in data.items():
            setattr(instance, key, value)
        return instance

    def update(self, combo: dict) -> 'Parameters':
        for key, value in combo.items():
            setattr(self, key, value)
        return self


    def __str__(self) -> str:
        # pformat rende leggibili liste/nidificazioni
        body = pformat(vars(self), sort_dicts=False, width=120)
        return f"{self.__class__.__name__}(\n{body}\n)"


def initialize_configuration(config_file=None):
    run_params = Parameters()
    if config_file is not None:
        config_path = os.path.join(run_params.config_path,
                                   config_file)
        run_params = Parameters.from_yaml(config_path)
        print(f'Loaded configuration named: {config_file}')
    else:
        run_params.experimentID = int(SystemRandom().random() * 100000)
        print(f'Loaded default configuration.')
    print('List of parameters:')
    print(run_params)
    return run_params