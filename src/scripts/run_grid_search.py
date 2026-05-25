import copy
import json
import os
import sys
import itertools
from datetime import datetime

os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0+PTX"  # per nuove GPU

from src.config import Parameters
from src.dataset.datamodule import get_datamodule
from src.train.train import train, test
from src.train.training_module import Training
from src.utils.mlflow_utils import (
    make_loggable_dict, config_hash,
    mlflow_parent_run, mlflow_child_run, log_aggregate_metrics,
)
from src.utils.utils import setup_seed, initialize_log_parameters, update_seed_metrics, update_run_metrics


def build_combinations(search_space: dict) -> list:
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_single_seed(run_params: Parameters) -> tuple:
    if run_params.reproducible:
        setup_seed(run_params.seed)

    data_module_instance, run_params = get_datamodule(run_params)
    training_module = Training(run_params)
    train(training_module, data_module_instance, run_params)
    res_test = test(training_module, data_module_instance, run_params)
    return training_module, res_test


def run_single_combination(combo: dict, cont: int, global_config: dict, seed_list: list):
    base_params = Parameters().update(combo | global_config)
    params_dict = make_loggable_dict(base_params)
    cfg_id      = config_hash(params_dict)

    grid_params_dict = initialize_log_parameters(cont, combo)
    val_results, test_results, seed_metrics = [], [], []

    with mlflow_parent_run(base_params, params_dict, n_seeds=len(seed_list)):
        for seed in seed_list:
            run_params = copy.deepcopy(base_params)
            run_params.seed             = seed
            run_params.mlflow_parent_id = base_params.mlflow_parent_id

            with mlflow_child_run(run_params, seed, params_dict, cfg_id):
                train_module, res_test = run_single_seed(run_params)

            val_results, test_results = update_seed_metrics(train_module, res_test, val_results, test_results)
            seed_metrics.append({
                'val_mse':  train_module.best_mse,
                'val_rmse': train_module.best_rmse,
                'val_mae':  train_module.best_mae,
                **res_test[0],
            })

        log_aggregate_metrics(base_params, seed_metrics)

    update_run_metrics(val_results, test_results, grid_params_dict, run_params)


def main():
    search_space = {'dataset':    ['etth1', 'ettm1'],
                    'model':      ['nbeats', 'lstm', 'dlinear', 'mtand', 'hi-patch', 'tpatch-gnn'],
                    'lr':         [1e-3, 1e-3],
                    'batch_size': [32],
                    'hid_dim':    [32],}
    # self.irregular_time_series = False
    # self.irregular_time_series_pattern = 'static'  # ['static' 'dynamic']
    # self.mechanism = "periodic"  # ['mcar', 'burst', 'periodic', 'async']
    #
    # # Data
    # self.seq_len = 96
    # self.pred_len = 96
    # self.label_len = 0
    # self.sparsity = 0.6
    # self.dropout = 0.0
    # self.w_decay = 0.0

    global_config = {'save_ckpts':               False,
                     'early_stop_callback_flag': True,
                     'save_logs':                True,
                     'reproducible':             True,
                     'logging':                  True,}

    seed_list  = [654, 897, 26]
    log_folder = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    global_config['logs_dir'] = os.path.join(Parameters().logs_dir, log_folder)
    os.makedirs(global_config['logs_dir'], exist_ok=True)

    combinations = build_combinations(search_space)
    print(f'Total combinations: {len(combinations)}')

    # Salva lo spazio di ricerca
    static_run_params = Parameters().update(combinations[0] | global_config)
    static_run_params.to_yaml(os.path.join(global_config['logs_dir'], 'static_run_params.yaml'))
    with open(os.path.join(global_config['logs_dir'], 'search_space_params.json'), "w", encoding="utf-8") as f:
        json.dump(search_space | global_config, f, indent=4, ensure_ascii=False)

    last_run       = 0
    free_error_run = False

    for cont, combo in enumerate(combinations):
        if cont < last_run:
            continue

        print(f'\nRun {cont + 1}/{len(combinations)}')
        if free_error_run:
            try:
                run_single_combination(combo, cont, global_config, seed_list)
            except:
                print('Error: ', sys.exc_info()[0])
        else:
            run_single_combination(combo, cont, global_config, seed_list)


if __name__ == '__main__':
    main()
