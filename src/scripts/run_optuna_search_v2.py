"""
run_optuna_search_v2.py — Pipeline a 3 fasi per hyperparam search e comparison
================================================================================

  ┌──────────────────────────────────────────────────────────────────────────┐
  │ FASE 1 — Optuna  (veloce: 1 seed, pruning attivo)                        │
  │                                                                          │
  │  for (dataset, model) in DATASETS × MODELS:                             │
  │    study = create_study(name=f"p1_{dataset}__{model}")                  │
  │    study.optimize(n_trials=N_TRIALS_PHASE1)                             │
  │    ──▶  registry/optuna/phase1_best_params/{dataset}__{model}.json      │
  └────────────────────────────────┬─────────────────────────────────────────┘
                                   │  file .json  (unico canale tra le fasi)
  ┌────────────────────────────────▼─────────────────────────────────────────┐
  │ FASE 2 — Grid Fine  (stabile: tutti i seed)                              │
  │                                                                          │
  │  per ogni phase1_best_params/{d}__{m}.json:                             │
  │    griglia ristretta: lr × [0.5, 1.0, 2.0]                              │
  │                       batch vicini in [16, 32, 64]                      │
  │                       w_decay × [0.5, 1.0, 2.0]  (se > 0)              │
  │    run su tutti i seed → sceglie config con val_mse_mean minore         │
  │    ──▶  registry/optuna/phase2_final_config/{dataset}__{model}.json     │
  └────────────────────────────────┬─────────────────────────────────────────┘
                                   │  file .json  (unico canale tra le fasi)
  ┌────────────────────────────────▼─────────────────────────────────────────┐
  │ FASE 3 — Comparison  (multi-seed, config definitiva)                    │
  │                                                                          │
  │  per ogni phase2_final_config/{d}__{m}.json:                           │
  │    esegue il modello con la sua config definitiva                       │
  │    ──▶  registry/optuna/phase3_comparison/comparison_{timestamp}.json   │
  └──────────────────────────────────────────────────────────────────────────┘

  Ogni fase comunica con la successiva SOLO tramite file .json.
  RIAVVIABILITÀ: se crasha la Fase 2, si riparte da Fase 2 senza rifare la 1.
  I file già prodotti vengono saltati automaticamente (check esistenza).

  Nota: ogni studio Optuna di Fase 1 è dedicato a UNA coppia (dataset, model).
  Questo evita il problema dei parametri condizionali con storage SQLite e
  assegna un budget equo a ogni modello.

  Usage:
    python -m src.scripts.run_optuna_search_v2              # tutte e 3 le fasi
    python -m src.scripts.run_optuna_search_v2 --phase 1
    python -m src.scripts.run_optuna_search_v2 --phase 2 3
"""

import argparse
import copy
import itertools
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
from src.utils.utils import (
    setup_seed, initialize_log_parameters,
    update_seed_metrics, update_run_metrics,
)


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH SPACES
#
#  Fase 1: 'dataset' e 'model' NON compaiono qui perché ogni studio Optuna
#  è dedicato a una singola coppia (dataset, model) → spazio pulito e senza
#  parametri condizionali, compatibile con qualsiasi Optuna storage.
#
#  Formato spec:
#    { 'type': 'categorical', 'choices': [...] }
#    { 'type': 'float',       'low': x, 'high': y, 'log': True/False }
#    { 'type': 'int',         'low': x, 'high': y, 'step': n }
# ══════════════════════════════════════════════════════════════════════════════

PHASE1_GLOBAL_SEARCH_SPACE: dict = {
    # 'lr':         {'type': 'float',       'low': 5e-5,  'high': 5e-3, 'log': True},
    'lr':         {'type': 'float', 'low': 5e-5, 'high': 5e-3, 'log': True},
    'batch_size': {'type': 'categorical', 'choices': [32, 64]},
    # 'w_decay':    {'type': 'float',       'low': 0.0,   'high': 1e-3},
    'w_decay':    {'type': 'categorical', 'choices': [0.0]},
    'seq_len':    {'type': 'categorical', 'choices': [96]},
    'pred_len':   {'type': 'categorical', 'choices': [96]},
}

MODEL_SEARCH_SPACES: dict = {
    'dlinear': {
        'dlinear_kernel_size': {'type': 'int', 'low': 11,  'high': 51, 'step': 2},
        'dlinear_individual':  {'type': 'categorical', 'choices': [True, False]},
    },
    'nbeats': {
        'nbeats_n_stacks':            {'type': 'int',         'low': 10, 'high': 50},
        'nbeats_layer_width':         {'type': 'categorical', 'choices': [64, 96]},
        'nbeats_n_fc_layers':         {'type': 'int',         'low': 2,  'high': 6},
        'nbeats_expansion_coeff_dim': {'type': 'categorical', 'choices': [16, 32, 64]},
        'dropout':                    {'type': 'float',       'low': 0.0, 'high': 0.4},
    },
    'lstm': {
        'latent_dim': {'type': 'categorical', 'choices': [16, 32, 64, 96]},
        'num_layers': {'type': 'int',         'low': 1,  'high': 4},
        'dropout':    {'type': 'float',       'low': 0.0, 'high': 0.4},
    },
    'mtand': {
        'latent_dim':   {'type': 'categorical', 'choices': [8, 16, 32]},
        'nhidden':      {'type': 'categorical', 'choices': [16, 32, 64]},
        'embed_time':   {'type': 'categorical', 'choices': [32, 64, 96]},
        'num_heads':    {'type': 'categorical', 'choices': [1, 2, 4]},
        'n_ref_points': {'type': 'categorical', 'choices': [64, 128, 256]},
    },
    'hi-patch': {
        'hid_dim':    {'type': 'categorical', 'choices': [32, 64, 96]},
        'nlayer':     {'type': 'int',         'low': 1, 'high': 4},
        'nhead':      {'type': 'categorical', 'choices': [1, 2, 4]},
        'patch_size': {'type': 'categorical', 'choices': [0.0625, 0.125, 0.25]},
        'stride':     {'type': 'categorical', 'choices': [0.0625, 0.125, 0.25]},
    },
    'tpatch-gnn': {
        'hid_dim':    {'type': 'categorical', 'choices': [32, 64, 96]},
        'nlayer':     {'type': 'int',         'low': 1, 'high': 4},
        'nhead':      {'type': 'categorical', 'choices': [1, 2, 4]},
        'te_dim':     {'type': 'categorical', 'choices': [5, 10, 20]},
        'node_dim':   {'type': 'categorical', 'choices': [5, 10, 20]},
        'patch_size': {'type': 'categorical', 'choices': [0.0625, 0.125, 0.25]},
        'stride':     {'type': 'categorical', 'choices': [0.0625, 0.125, 0.25]},
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE GRIGLIA FINE  (Fase 2)
#
#  ┌────────────────┬────────────────────────────────────────────────────────┐
#  │  Parametro     │  Valori nella griglia fine                             │
#  ├────────────────┼────────────────────────────────────────────────────────┤
#  │  lr            │  [best * 0.5,  best,  best * 2.0]                     │
#  │  w_decay > 0   │  [best * 0.5,  best,  best * 2.0]                     │
#  │  w_decay = 0   │  [0.0]  (non ha senso moltiplicare zero)              │
#  │  batch_size    │  [vicino sinistro, best, vicino destro] in BATCH_LIST  │
#  │  tutti gli     │                                                        │
#  │  altri params  │  fissi al valore ottimale trovato dalla Fase 1         │
#  └────────────────┴────────────────────────────────────────────────────────┘
# ══════════════════════════════════════════════════════════════════════════════

PHASE2_FLOAT_MULTIPLIERS: list[float] = [0.5, 1.0, 2.0]
PHASE2_BATCH_CHOICES:     list[int]   = [32, 64]   # deve coincidere con PHASE1_GLOBAL


# Chiavi di metadato nei .json di fase: NON sono iperparametri di training,
# quindi vengono filtrate prima di costruire il combo per Parameters.update().
_META_KEYS: frozenset = frozenset({
    'val_mse_mean', 'val_mse_std', 'val_rmse_mean', 'val_rmse_std',
    'val_mae_mean', 'val_mae_std', 'test_mse_mean', 'test_mse_std',
    'test_rmse_mean', 'test_rmse_std', 'test_mae_mean', 'test_mae_std',
    'trial_number', 'source_phase1', 'phase', 'n_seeds',
})


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS DI PATH
#
#  Tutta la pipeline scrive sotto registry/optuna/:
#
#    registry/optuna/
#    ├── optuna_studies.db          ← storage SQLite per tutti gli studi P1
#    ├── phase1_best_params/        ← output Fase 1
#    │   └── {dataset}__{model}.json
#    ├── phase2_final_config/       ← output Fase 2
#    │   └── {dataset}__{model}.json
#    └── phase3_comparison/         ← output Fase 3
#        └── comparison_{ts}.json
# ══════════════════════════════════════════════════════════════════════════════

def _optuna_base_dir() -> str:
    return os.path.join(Parameters().registry_dir, 'optuna')


def _phase1_dir() -> str:
    return os.path.join(_optuna_base_dir(), 'phase1_best_params')


def _phase2_dir() -> str:
    return os.path.join(_optuna_base_dir(), 'phase2_final_config')


def _phase3_dir() -> str:
    return os.path.join(_optuna_base_dir(), 'phase3_comparison')


def _optuna_db_uri() -> str:
    """URI SQLite condiviso da tutti gli studi Optuna della pipeline."""
    os.makedirs(_optuna_base_dir(), exist_ok=True)
    return f"sqlite:///{os.path.join(_optuna_base_dir(), 'optuna_studies.db')}"


def _pair_fname(dataset: str, model: str) -> str:
    return f"{dataset}__{model}.json"


def _load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_json(path: str, data: dict | list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)


def _base_global_config(logs_dir: str) -> dict:
    """Config globale condivisa da tutte le fasi (non contiene iperparametri di rete)."""
    os.makedirs(logs_dir, exist_ok=True)
    return {
        'save_ckpts':               False,
        'early_stop_callback_flag': True,
        'save_logs':                True,
        'reproducible':             True,
        'logging':                  True,
        'logs_dir':                 logs_dir,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CORE ENGINE  (condiviso da tutte e 3 le fasi)
#
#  _run_single_seed       → addestra e testa un singolo seed
#  _run_combination       → loop multi-seed con MLflow parent/child
#
#  Tutte le fasi convergono su _run_combination; cambia solo il combo passato:
#    Fase 1: combo campionato da Optuna (trial)
#    Fase 2: combo dalla griglia ristretta attorno ai best_params
#    Fase 3: combo dalla config definitiva di Fase 2
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
    raise ValueError(f"Tipo parametro sconosciuto: '{t}'")


def _run_single_seed(run_params: Parameters) -> tuple:
    if run_params.reproducible:
        setup_seed(run_params.seed)
    data_module_instance, run_params = get_datamodule(run_params)
    training_module = Training(run_params)
    train(training_module, data_module_instance, run_params)
    res_test = test(training_module, data_module_instance, run_params)
    return training_module, res_test


def _run_combination(combo: dict, cont: int, global_config: dict,
                     seed_list: list[int], phase_tag: str = '') -> dict:
    """
    Esegue una combinazione su più seed con tracking MLflow parent/child.

    Struttura MLflow (identica a run_grid_search.py):
      parent run  ← aggregate level, tag pipeline_phase
        └── child run (seed=s1)
        └── child run (seed=s2)
        └── ...

    Restituisce un dict con le metriche aggregate (val/test mse/rmse/mae mean+std).
    Questo dict viene poi salvato nel .json di fase e usato per scegliere la
    config migliore (Fase 2) o per costruire la tabella di confronto (Fase 3).
    """
    base_params = Parameters().update(combo | global_config)
    params_dict = make_loggable_dict(base_params)
    cfg_id      = config_hash(params_dict)

    grid_params_dict          = initialize_log_parameters(cont, combo)
    val_results, test_results = [], []
    seed_metrics: list[dict]  = []
    run_params                = base_params  # fallback se seed_list fosse vuota

    with mlflow_parent_run(base_params, params_dict, n_seeds=len(seed_list)) as parent_run:
        if parent_run is not None and phase_tag:
            mlflow.set_tag('pipeline_phase', phase_tag)

        for seed in seed_list:
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

        log_aggregate_metrics(base_params, seed_metrics)

    update_run_metrics(val_results, test_results, grid_params_dict, run_params)

    n = len(seed_metrics)

    def _mean(key: str) -> float:
        return float(np.mean([m[key] for m in seed_metrics]))

    def _std(key: str) -> float:
        return float(np.std([m[key] for m in seed_metrics], ddof=1) if n > 1 else 0.0)

    return {
        'val_mse_mean':    _mean('val_mse'),    'val_mse_std':    _std('val_mse'),
        'val_rmse_mean':   _mean('val_rmse'),   'val_rmse_std':   _std('val_rmse'),
        'val_mae_mean':    _mean('val_mae'),    'val_mae_std':    _std('val_mae'),
        'test_mse_mean':   _mean('test_mse'),   'test_mse_std':   _std('test_mse'),
        'test_rmse_mean':  _mean('test_rmse'),  'test_rmse_std':  _std('test_rmse'),
        'test_mae_mean':   _mean('test_mae'),   'test_mae_std':   _std('test_mae'),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 1 — OPTUNA SEARCH
#
#  Un studio Optuna per coppia (dataset, model):
#    ┌────────────────────────────────────────────────────────────────────┐
#    │  Campionamento:  TPESampler (seed=42)                             │
#    │  Pruning:        MedianPruner  (n_startup=5, n_warmup=0)         │
#    │                  segnale = val_mse al termine di ogni seed        │
#    │  Obiettivo:      minimizza val_mse_mean                           │
#    │  Storage:        SQLite condiviso (riavviabile)                   │
#    └────────────────────────────────────────────────────────────────────┘
#
#  Skip automatico: se il file .json di output esiste già, la coppia viene
#  saltata (utile se si vuole aggiungere nuovi modelli/dataset senza rifare
#  quelli già completati).
# ══════════════════════════════════════════════════════════════════════════════

def _p1_sample_combo(trial: optuna.Trial, dataset: str, model: str) -> dict:
    """Campiona parametri per la coppia (dataset, model) fissa di questo studio."""
    combo: dict = {'dataset': dataset, 'model': model}
    combo.update({
        name: _suggest_param(trial, name, spec)
        for name, spec in PHASE1_GLOBAL_SEARCH_SPACE.items()
    })
    if model in MODEL_SEARCH_SPACES:
        combo.update({
            name: _suggest_param(trial, name, spec)
            for name, spec in MODEL_SEARCH_SPACES[model].items()
        })
    return combo


def _p1_objective(trial: optuna.Trial, dataset: str, model: str,
                  global_config: dict, seed_list: list[int],
                  study_name: str, free_error_run: bool) -> float:
    combo = _p1_sample_combo(trial, dataset, model)
    print(f'\n  [P1 Trial {trial.number}]  {dataset}__{model}  '
          f'lr={combo["lr"]:.2e}  bs={combo["batch_size"]}  '
          f'wd={combo["w_decay"]:.2e}')

    def _execute() -> float:
        base_params = Parameters().update(combo | global_config)
        params_dict = make_loggable_dict(base_params)
        cfg_id      = config_hash(params_dict)
        val_results, test_results = [], []
        seed_metrics: list[dict]  = []
        run_params                = base_params

        with mlflow_parent_run(base_params, params_dict, n_seeds=len(seed_list)) as parent:
            if parent is not None:
                mlflow.set_tags({'optuna_study':   study_name,
                                 'trial_number':   str(trial.number),
                                 'pipeline_phase': 'phase1_optuna'})

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

                # Segnale al MedianPruner: un punto per seed completato
                trial.report(train_module.best_mse, step=idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            log_aggregate_metrics(base_params, seed_metrics)

        return float(np.mean([m['val_mse'] for m in seed_metrics]))

    if free_error_run:
        try:
            return _execute()
        except optuna.TrialPruned:
            raise
        except Exception:
            print(f'  [P1] Trial fallito: {sys.exc_info()[0]}')
            return float('inf')
    return _execute()


def _p1_save_best(study: optuna.Study, dataset: str, model: str,
                  save_dir: str) -> dict | None:
    """Estrae il best trial completato e lo salva come .json. Ritorna il dict."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f'  [P1] Nessun trial completato per {dataset}__{model}.')
        return None

    best = min(completed, key=lambda t: t.value)
    result = {
        'dataset':      dataset,
        'model':        model,
        **best.params,              # tutti i parametri campionati
        'val_mse_mean': best.value,
        'trial_number': best.number,
    }
    path = os.path.join(save_dir, _pair_fname(dataset, model))
    _save_json(path, result)
    print(f'  [P1] ✓ {dataset}__{model}  val_mse={best.value:.6f}  '
          f'trial #{best.number}  →  {path}')
    return result


def phase1_optuna(
    datasets: list[str],
    models: list[str],
    n_trials: int,
    seed_list: list[int],
    logs_root: str,
    free_error_run: bool = True,
) -> dict[str, dict]:
    """
    Fase 1: uno studio Optuna dedicato per ogni (dataset, model).

    Args:
        datasets:       Dataset da esplorare.
        models:         Modelli da esplorare.
        n_trials:       Trial Optuna per coppia.
        seed_list:      Seed per l'esecuzione (tipicamente 1 solo per velocità).
        logs_root:      Directory radice per i log di questa pipeline run.
        free_error_run: Se True, i trial falliti vengono registrati ma non bloccano.

    Returns:
        Dict keyed by '{dataset}__{model}' con i best_params di ogni coppia.
    """
    print('\n' + '═' * 72)
    print('  FASE 1 — OPTUNA SEARCH')
    print(f'  Coppie: {len(datasets) * len(models)}  '
          f'({len(datasets)} dataset × {len(models)} modelli)')
    print(f'  Trials per coppia: {n_trials}  |  Seed: {seed_list}')
    print('═' * 72)

    save_dir = _phase1_dir()
    os.makedirs(save_dir, exist_ok=True)
    db_uri   = _optuna_db_uri()
    results  = {}

    for dataset in datasets:
        for model in models:
            pair_key = f'{dataset}__{model}'
            out_path = os.path.join(save_dir, _pair_fname(dataset, model))

            # Skip se già completato (riavviabilità)
            if os.path.exists(out_path):
                print(f'\n  [P1] {pair_key} — già completato, skip.')
                results[pair_key] = _load_json(out_path)
                continue

            study_name  = f'p1_{pair_key}'
            pair_logs   = os.path.join(logs_root, 'phase1', pair_key)
            global_cfg  = _base_global_config(pair_logs)

            print(f'\n  [P1] Avvio studio: {study_name}')

            study = optuna.create_study(
                study_name=study_name,
                storage=db_uri,
                load_if_exists=True,
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            )

            study.optimize(
                lambda trial, d=dataset, m=model, gc=global_cfg: _p1_objective(
                    trial, d, m, gc, seed_list, study_name, free_error_run),
                n_trials=n_trials,
                show_progress_bar=True,
            )

            n_ok = sum(1 for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE)
            print(f'  [P1] {pair_key}: {n_ok}/{n_trials} trial completati')

            best = _p1_save_best(study, dataset, model, save_dir)
            if best is not None:
                results[pair_key] = best

    print('\n  [P1] Fase 1 completata.')
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 2 — GRID FINE
#
#  Legge i best_params da Fase 1 e costruisce una griglia ristretta:
#
#    param  │  tipo       │  griglia fine
#    ───────┼─────────────┼──────────────────────────────────────────────
#    lr     │  float log  │  [best*0.5, best, best*2.0]
#    w_dec  │  float      │  [best*0.5, best, best*2.0]  se > 0
#           │             │  [0.0]                       se ≈ 0
#    bs     │  categorical│  {vicino sx, best, vicino dx} in BATCH_LIST
#    altri  │  qualsiasi  │  fissi al valore di Fase 1
#
#  Combinazioni totali (caso tipico): 3 × 3 × 3 = 27 (se w_decay > 0)
#                                     3 × 3 × 1 =  9 (se w_decay = 0)
#
#  La combinazione con il minore val_mse_mean viene salvata come final_config.
# ══════════════════════════════════════════════════════════════════════════════

def _build_fine_grid(best_params: dict) -> list[dict]:
    """
    Costruisce la lista di combo per la griglia fine di Fase 2.
    I parametri non presenti nella griglia rimangono fissi a best_params.
    """
    # Rimuove i metadati non-parametro (val_mse_mean, trial_number, ecc.)
    base = {k: v for k, v in best_params.items() if k not in _META_KEYS}

    # lr: 3 valori centrati sul best
    lr_vals = [base['lr'] * m for m in PHASE2_FLOAT_MULTIPLIERS]

    # batch_size: vicini nella lista ufficiale (±1 step)
    bs = base.get('batch_size', 32)
    if bs in PHASE2_BATCH_CHOICES:
        idx = PHASE2_BATCH_CHOICES.index(bs)
    else:
        idx = 1  # fallback al centro
    bs_vals = sorted(set([
        PHASE2_BATCH_CHOICES[max(0, idx - 1)],
        bs,
        PHASE2_BATCH_CHOICES[min(len(PHASE2_BATCH_CHOICES) - 1, idx + 1)],
    ]))

    # w_decay: 3 valori se > 0, altrimenti rimane 0
    wd = base.get('w_decay', 0.0)
    wd_vals = ([wd * m for m in PHASE2_FLOAT_MULTIPLIERS]
               if wd > 1e-9 else [0.0])

    combos: list[dict] = []
    for lr, batch, wdecay in itertools.product(lr_vals, bs_vals, wd_vals):
        c = dict(base)
        c['lr']         = lr
        c['batch_size'] = batch
        c['w_decay']    = wdecay
        combos.append(c)

    return combos


def phase2_grid(
    datasets: list[str],
    models: list[str],
    seed_list: list[int],
    logs_root: str,
    free_error_run: bool = True,
) -> dict[str, dict]:
    """
    Fase 2: griglia fine attorno ai best_params di Fase 1, con tutti i seed.

    Args:
        datasets:       Dataset da processare.
        models:         Modelli da processare.
        seed_list:      Seed per la stabilità (tipicamente 3 seed).
        logs_root:      Directory radice per i log di questa pipeline run.
        free_error_run: Se True, le combinazioni fallite vengono saltate.

    Returns:
        Dict keyed by '{dataset}__{model}' con la final_config di ogni coppia.
    """
    print('\n' + '═' * 72)
    print('  FASE 2 — GRID FINE')
    print(f'  Seed: {seed_list}')
    print('═' * 72)

    p1_dir   = _phase1_dir()
    save_dir = _phase2_dir()
    os.makedirs(save_dir, exist_ok=True)
    results  = {}

    for dataset in datasets:
        for model in models:
            pair_key = f'{dataset}__{model}'
            out_path = os.path.join(save_dir, _pair_fname(dataset, model))

            # Skip se già completato
            if os.path.exists(out_path):
                print(f'\n  [P2] {pair_key} — già completato, skip.')
                results[pair_key] = _load_json(out_path)
                continue

            # Richiede il file di Fase 1
            p1_path = os.path.join(p1_dir, _pair_fname(dataset, model))
            if not os.path.exists(p1_path):
                print(f'\n  [P2] {pair_key} — file Fase 1 mancante, skip.')
                continue

            best_p1 = _load_json(p1_path)
            combos  = _build_fine_grid(best_p1)
            print(f'\n  [P2] {pair_key}  combinazioni da valutare: {len(combos)}')
            print(f'       best lr da P1: {best_p1["lr"]:.2e}  '
                  f'bs: {best_p1["batch_size"]}  '
                  f'wd: {best_p1.get("w_decay", 0):.2e}')

            pair_logs       = os.path.join(logs_root, 'phase2', pair_key)
            global_cfg      = _base_global_config(pair_logs)
            best_val_mse    = float('inf')
            best_result     = None

            for cont, combo in enumerate(combos):
                print(f'\n  [P2] {pair_key}  run {cont + 1}/{len(combos)}  '
                      f'lr={combo["lr"]:.2e}  bs={combo["batch_size"]}  '
                      f'wd={combo["w_decay"]:.2e}')
                try:
                    agg = _run_combination(combo, cont, global_cfg,
                                           seed_list, phase_tag='phase2_grid')
                except Exception:
                    if free_error_run:
                        print(f'  [P2] Combinazione fallita: {sys.exc_info()[0]}')
                        continue
                    raise

                if agg['val_mse_mean'] < best_val_mse:
                    best_val_mse = agg['val_mse_mean']
                    best_result  = {
                        **combo,
                        **agg,
                        'source_phase1': _pair_fname(dataset, model),
                    }

            if best_result is None:
                print(f'  [P2] {pair_key} — nessuna combinazione riuscita.')
                continue

            _save_json(out_path, best_result)
            print(f'  [P2] ✓ {pair_key}  val_mse_mean={best_val_mse:.6f}  →  {out_path}')
            results[pair_key] = best_result

    print('\n  [P2] Fase 2 completata.')
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 3 — COMPARISON
#
#  Legge tutti i phase2_final_config/{d}__{m}.json e lancia ogni modello con
#  la sua config definitiva (multi-seed). Produce la tabella di confronto:
#
#    Dataset       Model         val_mse   val_rmse  test_mse  test_mae
#    ─────────────────────────────────────────────────────────────────────
#    etth1         dlinear      0.381234  0.617441  0.391234  0.412345
#    etth1         lstm         0.412345  0.641823  0.421234  0.431234
#    ...
#    ettm1         dlinear      0.291234  0.539441  0.301234  0.312345
#    ...
#
#  Non ha logica di skip: ogni esecuzione di Fase 3 produce un nuovo file
#  comparison_{timestamp}.json (utile per confrontare run con semi diversi).
# ══════════════════════════════════════════════════════════════════════════════

def _print_comparison_table(rows: list[dict]) -> None:
    """Stampa una tabella ASCII ordinata per dataset poi val_mse_mean."""
    if not rows:
        print('  Nessun risultato da mostrare.')
        return

    rows_s = sorted(rows, key=lambda r: (r.get('dataset', ''),
                                          r.get('val_mse_mean', float('inf'))))
    w = {'ds': 14, 'mo': 12, 'v': 12, 'vr': 12, 't': 12, 'tm': 12}
    hdr = (f"{'Dataset':<{w['ds']}}  {'Model':<{w['mo']}}  "
           f"{'val_mse':>{w['v']}}  {'val_rmse':>{w['vr']}}  "
           f"{'test_mse':>{w['t']}}  {'test_mae':>{w['tm']}}")
    sep = '─' * len(hdr)

    print(f'\n{sep}\n{hdr}\n{sep}')
    prev_ds = None
    for r in rows_s:
        if r.get('dataset') != prev_ds and prev_ds is not None:
            print()
        prev_ds = r.get('dataset')
        print(f"{r.get('dataset', '?'):<{w['ds']}}  "
              f"{r.get('model', '?'):<{w['mo']}}  "
              f"{r.get('val_mse_mean',  float('nan')):>{w['v']}.6f}  "
              f"{r.get('val_rmse_mean', float('nan')):>{w['vr']}.6f}  "
              f"{r.get('test_mse_mean', float('nan')):>{w['t']}.6f}  "
              f"{r.get('test_mae_mean', float('nan')):>{w['tm']}.6f}")
    print(sep)


def phase3_comparison(
    datasets: list[str],
    models: list[str],
    seed_list: list[int],
    logs_root: str,
    free_error_run: bool = True,
) -> list[dict]:
    """
    Fase 3: esegue ogni modello con la sua config definitiva (da Fase 2)
    e produce la tabella di confronto.

    Args:
        datasets:       Dataset da comparare.
        models:         Modelli da comparare.
        seed_list:      Seed per la valutazione finale.
        logs_root:      Directory radice per i log di questa pipeline run.
        free_error_run: Se True, i modelli falliti vengono saltati.

    Returns:
        Lista di dict, uno per coppia (dataset, model), con metriche aggregate.
    """
    print('\n' + '═' * 72)
    print('  FASE 3 — COMPARISON')
    print(f'  Seed: {seed_list}')
    print('═' * 72)

    p2_dir   = _phase2_dir()
    save_dir = _phase3_dir()
    os.makedirs(save_dir, exist_ok=True)
    all_rows: list[dict] = []
    cont = 0

    for dataset in datasets:
        for model in models:
            pair_key = f'{dataset}__{model}'
            p2_path  = os.path.join(p2_dir, _pair_fname(dataset, model))

            if not os.path.exists(p2_path):
                print(f'\n  [P3] {pair_key} — file Fase 2 mancante, skip.')
                continue

            final_cfg  = _load_json(p2_path)
            combo      = {k: v for k, v in final_cfg.items() if k not in _META_KEYS}
            pair_logs  = os.path.join(logs_root, 'phase3', pair_key)
            global_cfg = _base_global_config(pair_logs)

            print(f'\n  [P3] {pair_key}  '
                  f'lr={combo.get("lr", "?"):.2e}  '
                  f'bs={combo.get("batch_size", "?")}')

            try:
                agg = _run_combination(combo, cont, global_cfg,
                                       seed_list, phase_tag='phase3_comparison')
                cont += 1
                all_rows.append({**combo, **agg})
            except Exception:
                if free_error_run:
                    print(f'  [P3] {pair_key} fallito: {sys.exc_info()[0]}')
                else:
                    raise

    timestamp   = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    output_path = os.path.join(save_dir, f'comparison_{timestamp}.json')
    _save_json(output_path, all_rows)
    print(f'\n  [P3] Comparison salvata  →  {output_path}')

    _print_comparison_table(all_rows)
    print('\n  [P3] Fase 3 completata.')
    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
#
#  Lancia le tre fasi in sequenza passando solo parametri scalari e liste.
#  Ogni fase è richiamabile individualmente tramite --phase N.
#
#  Flusso delle directory:
#
#    registry/
#    ├── logs/
#    │   └── pipeline_v2_{timestamp}/   ← log.txt + static_params per ogni run
#    │       ├── phase1/{d}__{m}/
#    │       ├── phase2/{d}__{m}/
#    │       └── phase3/{d}__{m}/
#    └── optuna/                        ← output permanente della pipeline
#        ├── optuna_studies.db
#        ├── phase1_best_params/
#        ├── phase2_final_config/
#        └── phase3_comparison/
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    datasets: list[str],
    models: list[str],
    seed_list_phase1: list[int],
    seed_list_phase23: list[int],
    n_trials_phase1: int,
    phases: tuple[int, ...] = (1, 2, 3),
    free_error_run: bool = True,
) -> None:
    """
    Orchestratore della pipeline a 3 fasi.

    Args:
        datasets:          Lista dei dataset da processare.
        models:            Lista dei modelli da processare.
        seed_list_phase1:  Seed usati in Fase 1 (tipicamente [1 seed] per velocità).
        seed_list_phase23: Seed usati in Fase 2 e 3 (tipicamente 3 seed per stabilità).
        n_trials_phase1:   Numero di trial Optuna per coppia (dataset, model) in Fase 1.
        phases:            Fasi da eseguire, es. (1, 2, 3) oppure (2, 3).
        free_error_run:    Se True, le eccezioni non bloccano la pipeline.
    """
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    logs_root = os.path.join(Parameters().logs_dir, f'pipeline_v2_{timestamp}')
    os.makedirs(logs_root, exist_ok=True)

    # Manifest: salva la configurazione di questa pipeline run per riproducibilità
    _save_json(os.path.join(logs_root, 'pipeline_manifest.json'), {
        'timestamp':         timestamp,
        'datasets':          datasets,
        'models':            models,
        'seed_list_phase1':  seed_list_phase1,
        'seed_list_phase23': seed_list_phase23,
        'n_trials_phase1':   n_trials_phase1,
        'phases':            list(phases),
        'free_error_run':    free_error_run,
        'phase1_dir':        _phase1_dir(),
        'phase2_dir':        _phase2_dir(),
        'phase3_dir':        _phase3_dir(),
    })

    print(f'\n{"═" * 72}')
    print(f'  PIPELINE V2  —  run: {timestamp}')
    print(f'  Fasi da eseguire:  {phases}')
    print(f'  Datasets:  {datasets}')
    print(f'  Modelli:   {models}')
    print(f'  Seed P1:   {seed_list_phase1}')
    print(f'  Seed P2/3: {seed_list_phase23}')
    print(f'{"═" * 72}')

    if 1 in phases:
        phase1_optuna(
            datasets=datasets,
            models=models,
            n_trials=n_trials_phase1,
            seed_list=seed_list_phase1,
            logs_root=logs_root,
            free_error_run=free_error_run,
        )

    if 2 in phases:
        phase2_grid(
            datasets=datasets,
            models=models,
            seed_list=seed_list_phase23,
            logs_root=logs_root,
            free_error_run=free_error_run,
        )

    if 3 in phases:
        phase3_comparison(
            datasets=datasets,
            models=models,
            seed_list=seed_list_phase23,
            logs_root=logs_root,
            free_error_run=free_error_run,
        )

    print(f'\n{"═" * 72}')
    print(f'  PIPELINE COMPLETATA  —  {timestamp}')
    print(f'  Output permanente in:  {_optuna_base_dir()}')
    print(f'  Log training in:       {logs_root}')
    print(f'{"═" * 72}\n')


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pipeline a 3 fasi: Optuna Search → Grid Fine → Comparison')
    parser.add_argument(
        '--phase', type=int, nargs='+', default=[1, 2, 3],
        choices=[1, 2, 3],
        help='Fasi da eseguire (default: 1 2 3). Esempio: --phase 2 3')
    args = parser.parse_args()

    # ── Configurazione principale ─────────────────────────────────────────────
    # Modifica questi valori per personalizzare la pipeline.

    DATASETS          = ['etth1']
    MODELS            = ['mtand', 'hi-patch', 'tpatch-gnn'] # 'nbeats', 'lstm', 'dlinear'

    # Fase 1: 1 solo seed per massimizzare la velocità di esplorazione
    SEED_LIST_PHASE1  = [654]

    # Fase 2 e 3: più seed per stabilità e intervalli di confidenza
    SEED_LIST_PHASE23 = [654, 897, 26]

    # Numero di trial Optuna per coppia (dataset, model) in Fase 1.
    # Con 2 dataset e 6 modelli → 12 studi × 30 trial = 360 trial totali.
    N_TRIALS_PHASE1   = 30

    # Se True, le eccezioni in un trial/run vengono loggiate ma non bloccano.
    FREE_ERROR_RUN    = True

    run_pipeline(
        datasets=DATASETS,
        models=MODELS,
        seed_list_phase1=SEED_LIST_PHASE1,
        seed_list_phase23=SEED_LIST_PHASE23,
        n_trials_phase1=N_TRIALS_PHASE1,
        phases=tuple(args.phase),
        free_error_run=FREE_ERROR_RUN,
    )



if __name__ == '__main__':
    main()
