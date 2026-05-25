# src/utils/mlflow_utils.py
import hashlib
import os
import json
import tempfile
import contextlib
from dataclasses import asdict, is_dataclass
import numpy as np
import mlflow


def _params_to_dict(p):
    if is_dataclass(p):                   return asdict(p)
    if hasattr(p, "model_dump"):          return p.model_dump()         # pydantic v2
    if hasattr(p, "to_container"):                                       # OmegaConf
        from omegaconf import OmegaConf
        return OmegaConf.to_container(p, resolve=True)
    if hasattr(p, "__dict__"):            return vars(p)
    return dict(p)


def _flatten(d, parent_key=""):
    out = {}
    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict): out.update(_flatten(v, key))
        else: out[key] = v
    return out


_SKIP_PARAMS = frozenset({'DATASET_REGISTRY', 'split_configs'})


def make_loggable_dict(run_params) -> dict:
    """Extracts a flat, MLflow-safe dict of params, skipping registry/config dicts."""
    raw = _params_to_dict(run_params)
    out = {}
    for k, v in raw.items():
        if k in _SKIP_PARAMS or callable(v) or isinstance(v, dict):
            continue
        out[k] = str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
    return out


def config_hash(params_dict: dict) -> str:
    """6-char hex hash that uniquely identifies a hyperparameter configuration."""
    return hashlib.md5(
        json.dumps(params_dict, sort_keys=True, default=str).encode()
    ).hexdigest()[:6]


def _aggregate_seed_metrics(seed_results: list) -> dict:
    if not seed_results:
        return {}
    out = {}
    for key in seed_results[0]:
        vals = np.array([r[key] for r in seed_results], dtype=float)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


@contextlib.contextmanager
def mlflow_session(run_params):
    """
    Apre un run MLflow se run_params.logging == True.
    Altrimenti è un no-op completo: il resto del codice non si accorge di nulla.
    Inietta run_params.mlflow_run_id (None se disattivato).
    """
    if not getattr(run_params, "logging", False):
        run_params.mlflow_run_id = None
        yield None
        return

    mlflow.set_tracking_uri(getattr(run_params, "tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(getattr(run_params, "experiment_name", "default"))

    params_dict = _params_to_dict(run_params)
    model   = getattr(run_params, "model",   "unknown")
    dataset = getattr(run_params, "dataset", "unknown")
    seed    = getattr(run_params, "seed",    0)
    run_name = (getattr(run_params, "run_name", None)
                or f"{model}_{dataset}_seed{seed}")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"model": model, "dataset": dataset})
        mlflow.log_params(_flatten(params_dict))

        # Config completa come artifact
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(params_dict, f, indent=2, default=str)
            path = f.name
        mlflow.log_artifact(path, artifact_path="config")
        os.unlink(path)

        run_params.mlflow_run_id = run.info.run_id
        yield run


@contextlib.contextmanager
def mlflow_parent_run(run_params, params_dict: dict, n_seeds: int):
    """
    Context manager for the aggregate (parent) MLflow run of one configuration.
    Tags with level='aggregate'. Logs params and config as artifact.
    Sets run_params.mlflow_parent_id and run_params.mlflow_run_id.
    """
    if not getattr(run_params, "logging", False):
        run_params.mlflow_parent_id = None
        run_params.mlflow_run_id    = None
        yield None
        return

    mlflow.set_tracking_uri(getattr(run_params, "tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(getattr(run_params, "experiment_name", "default"))

    model   = getattr(run_params, "model",   "unknown")
    dataset = getattr(run_params, "dataset", "unknown")
    cfg_id  = config_hash(params_dict)
    run_name = f"{model}__{dataset}__cfg{cfg_id}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "model":     model,
            "dataset":   dataset,
            "config_id": cfg_id,
            "level":     "aggregate",
            "n_seeds":   str(n_seeds),
        })
        mlflow.log_params(params_dict)
        mlflow.log_dict(params_dict, "config/config.json")
        run_params.mlflow_run_id    = run.info.run_id
        run_params.mlflow_parent_id = run.info.run_id
        yield run


@contextlib.contextmanager
def mlflow_child_run(run_params, seed: int, params_dict: dict, cfg_id: str):
    """
    Context manager for a single-seed nested (child) MLflow run.
    Must be called inside an active mlflow_parent_run context.
    Tags with level='single_seed'. Replicates params for per-seed filtering.
    """
    if not getattr(run_params, "logging", False):
        yield None
        return

    model     = getattr(run_params, "model",   "unknown")
    dataset   = getattr(run_params, "dataset", "unknown")
    parent_id = getattr(run_params, "mlflow_parent_id", "")
    run_name  = f"{model}__{dataset}__cfg{cfg_id}__seed{seed}"

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.set_tags({
            "model":         model,
            "dataset":       dataset,
            "config_id":     cfg_id,
            "level":         "single_seed",
            "seed":          str(seed),
            "parent_run_id": parent_id,
        })
        mlflow.log_params({**params_dict, "seed": seed})
        run_params.mlflow_run_id = run.info.run_id
        yield run


def log_aggregate_metrics(run_params, seed_results: list) -> dict:
    """
    Logs mean±std metrics to the currently active (parent) MLflow run.
    Returns the aggregated metrics dict.
    """
    if not getattr(run_params, "logging", False):
        return {}
    agg = _aggregate_seed_metrics(seed_results)
    if agg:
        mlflow.log_metrics(agg)
    return agg


def log_metrics(run_params, metrics: dict, step: int | None = None):
    if not getattr(run_params, "logging", False): return
    mlflow.log_metrics(metrics, step=step)


def log_artifact(run_params, path: str, artifact_path: str | None = None):
    if not getattr(run_params, "logging", False): return
    mlflow.log_artifact(path, artifact_path=artifact_path)
