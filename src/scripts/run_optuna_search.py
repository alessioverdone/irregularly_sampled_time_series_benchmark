"""
Hyperparameter search con Optuna.
Riusa la stessa infrastruttura di basso livello di run_grid_search.py:
  - stessa struttura MLflow (parent run per combo + child run per seed)
  - stesse funzioni train/test/Training/datamodule
  - nessuna modifica ai file esistenti

Per ogni trial Optuna campiona una combinazione di parametri globali +
parametri specifici del modello selezionato (condizionali su 'model').
Al termine salva i migliori parametri per modello come JSON in logs_dir/best_params/.
"""
import copy
import json
import os
import sys
from datetime import datetime

import numpy as np
import optuna
import mlflow

os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0+PTX"

from src.config import Parameters
from src.dataset.datamodule import get_datamodule
from src.train.train import train, test
from src.train.training_module import Training
from src.utils.mlflow_utils import (
    make_loggable_dict, config_hash,
    mlflow_parent_run, mlflow_child_run, log_aggregate_metrics,
)
from src.utils.utils import setup_seed, initialize_log_parameters, update_seed_metrics, update_run_metrics


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH SPACE — modifica qui per personalizzare la ricerca
# ══════════════════════════════════════════════════════════════════════════════
#
#  Formato di ogni spec:
#    { 'type': 'categorical', 'choices': [...] }
#    { 'type': 'float',  'low': x, 'high': y, 'log': True/False }
#    { 'type': 'int',    'low': x, 'high': y, 'step': n }         (step opzionale)
#
#  I parametri in MODEL_SEARCH_SPACES vengono campionati solo quando il modello
#  corrispondente è selezionato (parametri condizionali supportati da TPE).

GLOBAL_SEARCH_SPACE: dict = {
    'dataset':    {'type': 'categorical', 'choices': ['etth1']},
    'mechanism':      {'type': 'categorical', 'choices': ['mcar', 'burst', 'periodic', 'async']},
    'model':      {'type': 'categorical', 'choices': ['mtand', 'hi-patch', 'tpatch-gnn']},
    'lr':         {'type': 'float',       'low': 5e-5, 'high': 5e-3, 'log': True},
    'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
    # 'w_decay':    {'type': 'float',       'low': 0.0,  'high': 1e-3},
    'seq_len':    {'type': 'categorical', 'choices': [96]},
    'pred_len':   {'type': 'categorical', 'choices': [96]},
}

MODEL_SEARCH_SPACES: dict = {
    'dlinear': {
        'dlinear_kernel_size': {'type': 'int',         'low': 11,  'high': 51, 'step': 2},
        'dlinear_individual':  {'type': 'categorical', 'choices': [True, False]},
    },
    'nbeats': {
        'nbeats_n_stacks':            {'type': 'int',         'low': 10, 'high': 50},
        'nbeats_layer_width':         {'type': 'categorical', 'choices': [64, 128, 256]},
        'nbeats_n_fc_layers':         {'type': 'int',         'low': 2,  'high': 6},
        'nbeats_expansion_coeff_dim': {'type': 'categorical', 'choices': [16, 32, 64]},
        'dropout':                    {'type': 'float',       'low': 0.0, 'high': 0.4},
    },
    'lstm': {
        'latent_dim': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
        'num_layers': {'type': 'int',         'low': 1,  'high': 4},
        'dropout':    {'type': 'float',       'low': 0.0, 'high': 0.4},
    },
    'mtand': {
        'latent_dim':   {'type': 'categorical', 'choices': [8, 16, 32]},
        'nhidden':      {'type': 'categorical', 'choices': [16, 32, 64]},
        'embed_time':   {'type': 'categorical', 'choices': [32, 64, 128]},
        'num_heads':    {'type': 'categorical', 'choices': [1, 2, 4]},
        'n_ref_points': {'type': 'categorical', 'choices': [64, 128, 256]},
    },
    'hi-patch': {
        'hid_dim':    {'type': 'categorical', 'choices': [32, 64, 128]},
        'nlayer':     {'type': 'int',         'low': 1, 'high': 4},
        'nhead':      {'type': 'categorical', 'choices': [1, 2, 4]},
        'patch_size': {'type': 'categorical', 'choices': [0.0625, 0.125, 0.25]},
        'stride':     {'type': 'categorical', 'choices': [0.0625, 0.125, 0.25]},
    },
    'tpatch-gnn': {
        'hid_dim':    {'type': 'categorical', 'choices': [32, 64, 128]},
        'nlayer':     {'type': 'int',         'low': 1, 'high': 4},
        'nhead':      {'type': 'categorical', 'choices': [1, 2, 4]},
        'te_dim':     {'type': 'categorical', 'choices': [5, 10, 20]},
        'node_dim':   {'type': 'categorical', 'choices': [5, 10, 20]},
        'patch_size': {'type': 'categorical', 'choices': [0.0625, 0.125, 0.25]},
        'stride':     {'type': 'categorical', 'choices': [0.0625, 0.125, 0.25]},
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Logica interna — non è necessario modificare sotto per casi d'uso tipici
# ══════════════════════════════════════════════════════════════════════════════

def _suggest_param(trial: optuna.Trial, name: str, spec: dict):
    t = spec['type']
    if t == 'categorical':
        return trial.suggest_categorical(name, spec['choices'])
    if t == 'float':
        return trial.suggest_float(name, spec['low'], spec['high'],
                                   log=spec.get('log', False))
    if t == 'int':
        return trial.suggest_int(name, spec['low'], spec['high'],
                                 step=spec.get('step', 1))
    raise ValueError(f"Tipo di parametro sconosciuto: '{t}'")


def _sample_combo(trial: optuna.Trial) -> dict:
    """Campiona una combinazione di parametri dallo spazio di ricerca.

    I parametri model-specific vengono registrati in Optuna con il prefisso
    '<model>__<name>' (es. 'lstm__latent_dim', 'mtand__latent_dim').
    Questo evita il ValueError "CategoricalDistribution does not support
    dynamic value space" quando lo stesso nome ha choices diverse tra modelli
    (problema tipico con SQLite storage e parametri condizionali).
    Il combo passato a Parameters usa sempre il nome originale (senza prefisso).
    """
    combo = {name: _suggest_param(trial, name, spec)
             for name, spec in GLOBAL_SEARCH_SPACE.items()}
    model = combo['model']
    if model in MODEL_SEARCH_SPACES:
        for name, spec in MODEL_SEARCH_SPACES[model].items():
            optuna_name = f"{model}__{name}"   # nome unico per Optuna/storage
            combo[name] = _suggest_param(trial, optuna_name, spec)
    return combo


def _run_single_seed(run_params) -> tuple:
    """Identica a run_single_seed in run_grid_search.py (duplicata per indipendenza)."""
    if run_params.reproducible:
        setup_seed(run_params.seed)
    data_module_instance, run_params = get_datamodule(run_params)
    training_module = Training(run_params)
    train(training_module, data_module_instance, run_params)
    res_test = test(training_module, data_module_instance, run_params)
    return training_module, res_test


def run_optuna_trial(trial: optuna.Trial, combo: dict, global_config: dict,
                     seed_list: list, study_name: str) -> float:
    """
    Esegue un trial Optuna: lancia più seed per la combinazione data.
    Preserva la struttura MLflow parent/child di run_grid_search.py e aggiunge
    tag optuna_study e trial_number al run padre per filtraggio facile.
    Restituisce val_mse_mean (obiettivo da minimizzare).
    """
    base_params = Parameters().update(combo | global_config)
    params_dict = make_loggable_dict(base_params)
    cfg_id      = config_hash(params_dict)

    grid_params_dict              = initialize_log_parameters(trial.number, combo)
    val_results, test_results     = [], []
    seed_metrics: list[dict]      = []
    run_params                    = base_params  # fallback se seed_list fosse vuota

    with mlflow_parent_run(base_params, params_dict, n_seeds=len(seed_list)) as parent_run:
        if parent_run is not None:
            mlflow.set_tags({'optuna_study': study_name,
                             'trial_number': str(trial.number)})

        for idx, seed in enumerate(seed_list):
            run_params                  = copy.deepcopy(base_params)
            run_params.seed             = seed
            run_params.mlflow_parent_id = base_params.mlflow_parent_id

            with mlflow_child_run(run_params, seed, params_dict, cfg_id):
                train_module, res_test = _run_single_seed(run_params)

            val_results, test_results = update_seed_metrics(
                train_module, res_test, val_results, test_results)
            seed_metrics.append({
                'val_mse':  train_module.best_mse,
                'val_rmse': train_module.best_rmse,
                'val_mae':  train_module.best_mae,
                **res_test[0],
            })

            # Valore intermedio per potenziale pruning (un punto per seed)
            trial.report(train_module.best_mse, step=idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        log_aggregate_metrics(base_params, seed_metrics)

    update_run_metrics(val_results, test_results, grid_params_dict, run_params)

    return float(np.mean([m['val_mse'] for m in seed_metrics]))


def _objective(trial: optuna.Trial, global_config: dict, seed_list: list,
               study_name: str, free_error_run: bool) -> float:
    combo = _sample_combo(trial)
    print(f'\n[Trial {trial.number}] {combo}')

    if free_error_run:
        try:
            return run_optuna_trial(trial, combo, global_config, seed_list, study_name)
        except optuna.TrialPruned:
            raise
        except Exception:
            print('Trial fallito:', sys.exc_info()[0])
            return float('inf')

    return run_optuna_trial(trial, combo, global_config, seed_list, study_name)


def _strip_model_prefix(params: dict, model: str) -> dict:
    """Rimuove il prefisso '<model>__' dai nomi dei parametri model-specific,
    restituendo un dict con i nomi originali (compatibili con Parameters)."""
    prefix = f"{model}__"
    return {(k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in params.items()}


def save_best_params(study: optuna.Study, save_dir: str) -> None:
    """Salva i migliori parametri per modello (e overall) come JSON.
    I nomi dei parametri nei JSON non hanno prefisso e sono direttamente
    utilizzabili con Parameters.update()."""
    os.makedirs(save_dir, exist_ok=True)

    best_per_model: dict[str, optuna.trial.FrozenTrial] = {}
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        model = t.params.get('model', 'unknown')
        if model not in best_per_model or t.value < best_per_model[model].value:
            best_per_model[model] = t

    for model, t in best_per_model.items():
        clean = _strip_model_prefix(t.params, model)
        path = os.path.join(save_dir, f'best_params_{model}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({**clean, 'val_mse_mean': t.value,
                       'trial_number': t.number},
                      f, indent=4, ensure_ascii=False)
        print(f'[{model}] migliori params salvati → {path}  '
              f'(val_mse_mean={t.value:.6f}, trial #{t.number})')

    try:
        best = study.best_trial
    except ValueError:
        return  # nessun trial completato

    model = best.params.get('model', 'unknown')
    clean = _strip_model_prefix(best.params, model)
    path = os.path.join(save_dir, 'best_params_overall.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({**clean, 'val_mse_mean': best.value,
                   'trial_number': best.number},
                  f, indent=4, ensure_ascii=False)
    print(f'Overall best params salvati → {path}  '
          f'(val_mse_mean={best.value:.6f}, trial #{best.number})')


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Impostazioni esecuzione ───────────────────────────────────────────────
    n_trials        = 50
    seed_list       = [654]  # , 897, 26
    free_error_run  = True   # continua anche se un trial lancia un'eccezione

    # URI di storage Optuna (None = in-memory, non persistente).
    # Per persistenza/parallelismo impostare es.:
    #   f"sqlite:///{os.path.join(Parameters().registry_dir, 'optuna.db')}"
    # optuna_storage: str | None = None
    optuna_storage = f"sqlite:///{os.path.join(Parameters().registry_dir, 'optuna.db')}"

    global_config = {
        'save_ckpts':               False,
        'early_stop_callback_flag': True,
        'save_logs':                True,
        'reproducible':             True,
        'logging':                  True,
    }

    study_name = datetime.now().strftime("optuna_%Y-%m-%dT%H-%M-%S")
    global_config['logs_dir'] = os.path.join(Parameters().logs_dir, study_name)
    os.makedirs(global_config['logs_dir'], exist_ok=True)

    # Salva la definizione del search space per riproducibilità
    with open(os.path.join(global_config['logs_dir'], 'search_space.json'),
              'w', encoding='utf-8') as f:
        json.dump({
            'global':       GLOBAL_SEARCH_SPACE,
            'model_params': MODEL_SEARCH_SPACES,
            'seed_list':    seed_list,
            'n_trials':     n_trials,
        }, f, indent=4, ensure_ascii=False, default=str)

    # ── Crea e lancia lo study ────────────────────────────────────────────────
    study = optuna.create_study(
        study_name=study_name,
        storage=optuna_storage,
        load_if_exists=True,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    study.optimize(
        lambda trial: _objective(trial, global_config, seed_list,
                                 study_name, free_error_run),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # ── Salva i migliori parametri ────────────────────────────────────────────
    best_params_dir = os.path.join(global_config['logs_dir'], 'best_params')
    save_best_params(study, best_params_dir)

    n_complete = sum(1 for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE)
    print(f'\n=== Studio Optuna completato: {study_name} ===')
    print(f'Trials completati: {n_complete}/{n_trials}')
    try:
        print(f'Miglior trial: #{study.best_trial.number}')
        print(f'Miglior val_mse_mean: {study.best_value:.6f}')
        print(f'Miglior combinazione: {study.best_params}')
    except ValueError:
        print('Nessun trial completato con successo.')


if __name__ == '__main__':
    main()
