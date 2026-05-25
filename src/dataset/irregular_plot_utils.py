
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Split = Literal["train", "val", "test"]
ViewMode = Literal["zero", "observed", "both"]
MergePolicy = Literal["first", "last", "mean"]
CollapseMode = Literal["representative", "union"]
RepresentativePolicy = Literal["latest", "earliest", "center"]


def load_static_irregular_npz(
    dataset_dir: str | Path | None = None,
    *,
    npz_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Carica dataset.npz + config.json salvati dal generatore statico.

    Puoi usarla in due modi:
        load_static_irregular_npz("../../data/irregular/etth1/...")
    oppure:
        load_static_irregular_npz(npz_path="dataset.npz", config_path="config.json")

    Returns
    -------
    arrays : dict
        Contiene x_data, x_mask, y_data, tp_obs, tp_pred.
    config : dict
        Config JSON con sample_splits, feature_cols, scaler, ecc.
    """
    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir)
        npz_path = dataset_dir / "dataset.npz"
        config_path = dataset_dir / "config.json"

    if npz_path is None or config_path is None:
        raise ValueError("Passa dataset_dir oppure entrambi npz_path e config_path.")

    npz_path = Path(npz_path)
    config_path = Path(config_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"File NPZ non trovato: {npz_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"File config JSON non trovato: {config_path}")

    with np.load(npz_path, allow_pickle=True) as db:
        arrays = {k: db[k] for k in db.files}

    with open(config_path, "r") as f:
        config = json.load(f)

    _validate_static_irregular_arrays(arrays, config)
    return arrays, config


def _validate_static_irregular_arrays(arrays: dict[str, np.ndarray], config: dict) -> None:
    required = {"x_data", "x_mask", "y_data", "tp_obs", "tp_pred"}
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"dataset.npz non contiene le chiavi richieste: {sorted(missing)}")

    x_data = arrays["x_data"]
    x_mask = arrays["x_mask"]
    y_data = arrays["y_data"]

    if x_data.ndim != 3:
        raise ValueError(f"x_data deve avere shape (N, seq_len, D), trovato {x_data.shape}")
    if x_mask.shape != x_data.shape:
        raise ValueError(f"x_mask.shape={x_mask.shape} diverso da x_data.shape={x_data.shape}")
    if y_data.ndim != 3:
        raise ValueError(f"y_data deve avere shape (N, pred_len, D_out), trovato {y_data.shape}")

    N, seq_len, D = x_data.shape
    if y_data.shape[0] != N:
        raise ValueError(f"x_data e y_data hanno N diverso: {N} vs {y_data.shape[0]}")
    if arrays["tp_obs"].shape[0] != seq_len:
        raise ValueError("tp_obs non coerente con seq_len")
    if arrays["tp_pred"].shape[0] != y_data.shape[1]:
        raise ValueError("tp_pred non coerente con pred_len")

    if "sample_splits" not in config:
        raise KeyError("config.json non contiene 'sample_splits'")
    if "feature_cols" not in config:
        raise KeyError("config.json non contiene 'feature_cols'")
    if len(config["feature_cols"]) != D:
        raise ValueError(
            f"feature_cols ha lunghezza {len(config['feature_cols'])}, ma x_data ha D={D}"
        )


def _split_sample_range(config: dict, split: Split) -> tuple[int, int]:
    if split not in config["sample_splits"]:
        raise KeyError(f"Split '{split}' non presente in config['sample_splits']")
    start, end = map(int, config["sample_splits"][split])
    if end <= start:
        raise ValueError(f"Split '{split}' vuoto o invalido: [{start}, {end})")
    return start, end


def _feature_indices(
    feature_cols: Sequence[str],
    features: Sequence[str | int] | str | int | None,
) -> tuple[list[int], list[str]]:
    """
    Normalizza la selezione feature.

    features:
      - None       -> tutte le feature
      - "OT"       -> una feature per nome
      - 6          -> una feature per indice
      - ["OT", 0]  -> lista mista
    """
    if features is None:
        idx = list(range(len(feature_cols)))
    else:
        if isinstance(features, (str, int)):
            features = [features]

        idx = []
        for f in features:
            if isinstance(f, str):
                if f not in feature_cols:
                    raise KeyError(f"Feature '{f}' non trovata in {list(feature_cols)}")
                idx.append(feature_cols.index(f))
            else:
                f = int(f)
                if not 0 <= f < len(feature_cols):
                    raise IndexError(f"Indice feature {f} fuori range [0, {len(feature_cols)})")
                idx.append(f)

    names = [feature_cols[i] for i in idx]
    return idx, names


def _inverse_scale_values(
    values: np.ndarray,
    config: dict,
    feature_idx: Sequence[int],
) -> np.ndarray:
    """
    Inversa dello StandardScaler solo sulle feature richieste.

    values shape: (..., len(feature_idx))
    """
    scaler = config.get("scaler")
    if scaler is None:
        raise KeyError("inverse_scale=True richiede config['scaler']")

    mean = np.asarray(scaler["mean"], dtype=np.float32)[list(feature_idx)]
    std = np.asarray(scaler["std"], dtype=np.float32)[list(feature_idx)]

    return values * std.reshape((1,) * (values.ndim - 1) + (-1,)) + mean.reshape(
        (1,) * (values.ndim - 1) + (-1,)
    )


def _load_regular_csv_values(
    csv_path: str | Path,
    config: dict,
    feature_idx: Sequence[int],
    *,
    inverse_scale: bool,
    regular_already_scaled: bool = False,
) -> np.ndarray:
    """
    Legge il CSV regolare sorgente e restituisce le feature richieste.

    Convenzione di scala:
      - regular_already_scaled=False, inverse_scale=True  -> CSV raw/originale.
      - regular_already_scaled=False, inverse_scale=False -> scala il CSV con
        lo stesso StandardScaler salvato in config, così l'overlay è confrontabile
        con x_data/y_data.
      - regular_already_scaled=True, inverse_scale=False  -> CSV già scalato,
        quindi non viene trasformato.
      - regular_already_scaled=True, inverse_scale=True   -> riporta il CSV
        scalato alla scala originale con config["scaler"].

    Returns
    -------
    regular : np.ndarray
        Array shape (T_regular, len(feature_idx)).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV regolare non trovato: {csv_path}")

    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    feature_cols = list(config["feature_cols"])
    selected_cols = [feature_cols[i] for i in feature_idx]
    missing = [c for c in selected_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Il CSV regolare non contiene le colonne richieste: {missing}. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    regular = df[selected_cols].to_numpy(dtype=np.float32)

    total_timesteps = config.get("total_timesteps")
    if total_timesteps is not None and len(regular) < int(total_timesteps):
        raise ValueError(
            f"CSV regolare troppo corto: len={len(regular)}, "
            f"config['total_timesteps']={total_timesteps}"
        )

    scaler = config.get("scaler")
    if scaler is None and (not inverse_scale or regular_already_scaled):
        raise KeyError("La trasformazione di scala del CSV richiede config['scaler'].")

    if scaler is not None:
        mean = np.asarray(scaler["mean"], dtype=np.float32)[list(feature_idx)]
        std = np.asarray(scaler["std"], dtype=np.float32)[list(feature_idx)]

        if regular_already_scaled and inverse_scale:
            # CSV già nello spazio standardizzato -> scala originale.
            regular = regular * std.reshape(1, -1) + mean.reshape(1, -1)
        elif (not regular_already_scaled) and (not inverse_scale):
            # CSV raw/originale -> spazio standardizzato, come x_data/y_data.
            regular = (regular - mean.reshape(1, -1)) / std.reshape(1, -1)
        # Altri casi:
        #   raw + inverse_scale=True     -> già nella scala originale
        #   scaled + inverse_scale=False -> già nella scala standardizzata

    return regular


def _regular_values_at_times(
    regular_values: np.ndarray,
    time_abs: np.ndarray,
) -> np.ndarray:
    """Estrae regular_values[time_abs] con controllo esplicito dei limiti."""
    if len(time_abs) == 0:
        return np.empty((0, regular_values.shape[1]), dtype=regular_values.dtype)

    t_min = int(np.nanmin(time_abs))
    t_max = int(np.nanmax(time_abs))
    if t_min < 0 or t_max >= len(regular_values):
        raise IndexError(
            f"Richiesti timestep regolari [{t_min}, {t_max}], "
            f"ma il CSV ha lunghezza {len(regular_values)}."
        )
    return regular_values[time_abs.astype(np.int64)]


def reconstruct_split_timeline(
    arrays: dict[str, np.ndarray],
    config: dict,
    *,
    split: Split = "train",
    features: Sequence[str | int] | str | int | None = None,
    collapse: CollapseMode = "representative",
    representative: RepresentativePolicy = "latest",
    merge_policy: MergePolicy = "first",
    inverse_scale: bool = False,
) -> dict[str, np.ndarray | list[str] | tuple[int, int]]:
    """
    Ricostruisce una timeline 2D a partire da finestre salvate per-sample.

    Mappatura corretta:
        sample globale j  -> finestra x_data[j]
        local timestep t  -> timestep globale j + t

    Attenzione:
        Il generatore salva maschere per finestra. Quindi lo stesso timestep
        globale può apparire in più sample e può essere osservato in alcune
        finestre ma non in altre. Per una visualizzazione globale la scelta
        tecnicamente più sensata è collapse="representative": una sola
        occorrenza per timestep-feature, così la sparsità non viene cancellata
        dall'overlap tra finestre. collapse="union" è disponibile solo se vuoi
        sapere se quel timestep-feature è stato osservato almeno una volta in
        una qualunque finestra.

    Parameters
    ----------
    split:
        "train", "val" o "test", letto da config["sample_splits"].
    features:
        Feature da plottare/ricostruire. None = tutte.
    collapse:
        - "representative": prende una sola occorrenza per timestep globale.
        - "union": collassa tutte le finestre sovrapposte; tende a densificare.
    representative:
        Usato solo con collapse="representative".
        - "latest"  : per timestep t usa il sample più recente che contiene t.
                      Nell'interno dello split equivale quasi a local_t=0.
        - "earliest": usa il primo sample che contiene t.
        - "center"  : usa il sample in cui t è circa al centro della finestra.
    merge_policy:
        Usato solo con collapse="union".
        - "first": tiene la prima osservazione incontrata per timestep-feature.
        - "last" : sovrascrive con l'ultima osservazione incontrata.
        - "mean" : media tutte le osservazioni disponibili.
    inverse_scale:
        Se True riporta le osservazioni nella scala originale usando config["scaler"].

    Returns
    -------
    out : dict
        time_abs:
            Timestep globali assoluti.
        values_zero:
            Matrice (T_split, F) con missing = 0.
        values_nan:
            Matrice (T_split, F) con missing = np.nan.
        mask:
            Bool array (T_split, F), True se osservato almeno una volta.
        counts:
            Numero di finestre in cui quel timestep-feature è stato osservato.
        feature_names:
            Nomi feature selezionate.
        sample_range:
            Range sample globale [start, end).
    """
    x_data = arrays["x_data"]
    x_mask = arrays["x_mask"].astype(bool)

    sample_start, sample_end = _split_sample_range(config, split)
    if sample_end > x_data.shape[0]:
        raise ValueError(
            f"sample_splits[{split}]={sample_start, sample_end} eccede N={x_data.shape[0]}"
        )

    seq_len = x_data.shape[1]
    feature_idx, feature_names = _feature_indices(config["feature_cols"], features)
    F = len(feature_idx)

    # La timeline coperta dagli input dello split è:
    #   primo sample: sample_start ... sample_start + seq_len - 1
    #   ultimo sample: sample_end-1 ... sample_end-1 + seq_len - 1
    t0 = sample_start
    t1_excl = sample_end + seq_len - 1
    T = t1_excl - t0

    # np.take evita la sorpresa di NumPy con advanced indexing: con una lista
    # di feature, x_data[j, :, [f]] può diventare shape (1, seq_len) invece di
    # (seq_len, 1). Qui vogliamo sempre (N_split, seq_len, F).
    x_sel = np.take(x_data[sample_start:sample_end], feature_idx, axis=2)
    m_sel = np.take(x_mask[sample_start:sample_end], feature_idx, axis=2)

    if inverse_scale:
        x_sel = _inverse_scale_values(x_sel, config, feature_idx)

    values_out = np.zeros((T, F), dtype=np.float64)
    counts = np.zeros((T, F), dtype=np.int32)

    if collapse == "representative":
        # Una sola occorrenza per timestep globale.
        # Questo evita che l'overlap di seq_len finestre renda la serie quasi densa.
        for rel_t, global_t in enumerate(range(t0, t1_excl)):
            if representative == "latest":
                # sample più recente che contiene global_t
                global_sample = min(global_t, sample_end - 1)
            elif representative == "earliest":
                # sample più vecchio che contiene global_t
                global_sample = max(sample_start, global_t - seq_len + 1)
            elif representative == "center":
                # sample tale che global_t cada circa al centro della finestra
                global_sample = global_t - seq_len // 2
                global_sample = min(max(global_sample, sample_start), sample_end - 1)
            else:
                raise ValueError(f"representative non valida: {representative}")

            local_t = global_t - global_sample
            if not 0 <= local_t < seq_len:
                # Non dovrebbe succedere se le formule sopra sono corrette.
                continue

            local_sample = global_sample - sample_start
            obs = m_sel[local_sample, local_t]  # (F,)
            vals = x_sel[local_sample, local_t] # (F,)

            values_out[rel_t] = np.where(obs, vals, 0.0)
            counts[rel_t] = obs.astype(np.int32)

        mask = counts > 0

    elif collapse == "union":
        # Collassa tutte le occorrenze osservate nello stesso timestep-feature.
        # Utile per debug, ma con finestre molto sovrapposte può nascondere la sparsità.
        values_sum = np.zeros((T, F), dtype=np.float64)

        for local_sample, global_sample in enumerate(range(sample_start, sample_end)):
            # posizioni assolute della finestra, shiftate in coordinate relative [0, T)
            rows = np.arange(global_sample, global_sample + seq_len) - t0

            obs = m_sel[local_sample]  # (seq_len, F)
            vals = x_sel[local_sample] # (seq_len, F)

            if merge_policy == "mean":
                values_sum[rows] += np.where(obs, vals, 0.0)
                counts[rows] += obs.astype(np.int32)

            elif merge_policy == "first":
                for f in range(F):
                    idx_t = rows[obs[:, f] & (counts[rows, f] == 0)]
                    # idx_t sono indici relativi dentro values_out
                    values_out[idx_t, f] = vals[idx_t - rows[0], f]
                    counts[idx_t, f] = 1

            elif merge_policy == "last":
                for f in range(F):
                    idx_local = np.flatnonzero(obs[:, f])
                    idx_t = rows[idx_local]
                    values_out[idx_t, f] = vals[idx_local, f]
                    counts[idx_t, f] += 1

            else:
                raise ValueError(f"merge_policy non valida: {merge_policy}")

        mask = counts > 0
        if merge_policy == "mean":
            values_out = np.zeros_like(values_sum, dtype=np.float64)
            values_out[mask] = values_sum[mask] / counts[mask]

    else:
        raise ValueError(f"collapse non valida: {collapse}")

    values_zero = values_out.astype(np.float32, copy=True)
    values_nan = values_out.astype(np.float32, copy=True)
    values_nan[~mask] = np.nan

    return {
        "time_abs": np.arange(t0, t1_excl, dtype=np.int64),
        "values_zero": values_zero,
        "values_nan": values_nan,
        "mask": mask,
        "counts": counts,
        "feature_names": feature_names,
        "feature_idx": np.asarray(feature_idx, dtype=np.int64),
        "sample_range": (sample_start, sample_end),
        "timestep_range": (t0, t1_excl),
    }


def _limit_time_range(
    time: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    *,
    max_points: int | None,
    t_min: int | None,
    t_max: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.ones_like(time, dtype=bool)
    if t_min is not None:
        keep &= time >= int(t_min)
    if t_max is not None:
        keep &= time < int(t_max)

    time = time[keep]
    values = values[keep]
    mask = mask[keep]

    if max_points is not None and len(time) > max_points:
        time = time[:max_points]
        values = values[:max_points]
        mask = mask[:max_points]

    return time, values, mask


def plot_split_timeline(
    arrays: dict[str, np.ndarray],
    config: dict,
    *,
    split: Split = "train",
    features: Sequence[str | int] | str | int | None = None,
    view: ViewMode = "both",
    collapse: CollapseMode = "representative",
    representative: RepresentativePolicy = "latest",
    merge_policy: MergePolicy = "first",
    inverse_scale: bool = False,
    max_points: int | None = 1000,
    t_min: int | None = None,
    t_max: int | None = None,
    marker_size: float = 14.0,
    linewidth: float = 1.2,
    figsize_per_feature: tuple[float, float] = (14.0, 2.4),
    plot_regular: bool = False,
    regular_csv_path: str | Path | None = None,
    regular_alpha: float = 0.35,
    regular_linewidth: float | None = None,
    regular_already_scaled: bool = False,
    show: bool = True,
) -> list[plt.Figure]:
    """
    Plotta la timeline ricostruita dello split.

    view="zero":
        linea continua su values_zero, cioè missing = 0.
    view="observed":
        linea con marker solo sui punti osservati; i missing non esistono.
        La linea collega osservazioni anche a timestep non adiacenti.
    view="both":
        produce due figure: zero-filled e observed-only.

    collapse="representative" è il default consigliato: preserva la sparsità
    scegliendo una sola occorrenza per timestep globale. collapse="union" è
    utile per debug ma tende a rendere tutto osservato a causa delle finestre
    sovrapposte.

    plot_regular=True:
        sovrappone, con linea più chiara, la serie regolare letta da
        regular_csv_path. Se inverse_scale=False e regular_already_scaled=False,
        il CSV viene scalato con config["scaler"] per stare nella stessa scala
        di x_data. Il CSV deve essere esattamente quello usato per generare
        dataset.npz, altrimenti l'overlay sarà diverso.
    """
    rec = reconstruct_split_timeline(
        arrays,
        config,
        split=split,
        features=features,
        collapse=collapse,
        representative=representative,
        merge_policy=merge_policy,
        inverse_scale=inverse_scale,
    )

    time = rec["time_abs"]
    feature_names = rec["feature_names"]
    feature_idx = rec["feature_idx"]

    regular_values_full = None
    if plot_regular:
        if regular_csv_path is None:
            raise ValueError(
                "plot_regular=True richiede regular_csv_path, es. '../../data/regular/ETTh1.csv'."
            )
        regular_values_full = _load_regular_csv_values(
            regular_csv_path,
            config,
            feature_idx,
            inverse_scale=inverse_scale,
            regular_already_scaled=regular_already_scaled,
        )

    views = ["zero", "observed"] if view == "both" else [view]
    figures: list[plt.Figure] = []

    for v in views:
        values_key = "values_zero" if v == "zero" else "values_nan"
        values = rec[values_key]
        mask = rec["mask"]

        time_i, values_i, mask_i = _limit_time_range(
            time,
            values,
            mask,
            max_points=max_points,
            t_min=t_min,
            t_max=t_max,
        )
        regular_i = (
            _regular_values_at_times(regular_values_full, time_i)
            if regular_values_full is not None
            else None
        )

        n_feat = len(feature_names)
        fig_h = max(2.4, figsize_per_feature[1] * n_feat)
        fig, axes = plt.subplots(
            n_feat,
            1,
            figsize=(figsize_per_feature[0], fig_h),
            sharex=True,
            squeeze=False,
        )
        axes = axes.ravel()

        for f, ax in enumerate(axes):
            if regular_i is not None:
                ax.plot(
                    time_i,
                    regular_i[:, f],
                    color="C0",
                    alpha=regular_alpha,
                    linewidth=regular_linewidth or linewidth * 1.15,
                    label="serie regolare",
                    zorder=1,
                )

            if v == "zero":
                ax.plot(
                    time_i,
                    values_i[:, f],
                    color="C0",
                    linewidth=linewidth,
                    label="irregolare, missing=0",
                    zorder=2,
                )
                ax.set_title(f"{split} | {feature_names[f]} | missing = 0")
            else:
                obs_f = mask_i[:, f]
                ax.plot(
                    time_i[obs_f],
                    values_i[obs_f, f],
                    color="C0",
                    linestyle="-",
                    linewidth=linewidth,
                    label="irregolare osservato",
                    zorder=3,
                )
                ax.scatter(
                    time_i[obs_f],
                    values_i[obs_f, f],
                    color="C0",
                    s=marker_size,
                    zorder=4,
                )
                ax.set_title(
                    f"{split} | {feature_names[f]} | observed-only nodes "
                    f"({int(obs_f.sum())}/{len(obs_f)} timestep osservati)"
                )

            ax.set_ylabel(feature_names[f])
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")

        axes[-1].set_xlabel("timestep globale")
        fig.tight_layout()
        figures.append(fig)

    if show:
        plt.show()

    return figures


def _target_feature_names(config: dict, y_data: np.ndarray) -> list[str]:
    """
    Nomi delle feature target.

    Caso M/S:
        D_out == D_in -> target names = feature_cols.
    Caso MS:
        D_out == 1 -> target name = config["target"].
    """
    feature_cols = list(config["feature_cols"])
    D_out = y_data.shape[2]

    if D_out == len(feature_cols):
        return feature_cols
    if D_out == 1:
        return [config.get("target", "target")]
    return [f"target_{i}" for i in range(D_out)]


def _sample_feature_indices_for_y(
    config: dict,
    arrays: dict[str, np.ndarray],
    selected_x_feature_names: Sequence[str],
) -> list[int | None]:
    """
    Mappa le feature selezionate in input verso y_data.
    Per MS, y_data contiene solo il target: le altre feature ritornano None.
    """
    y_names = _target_feature_names(config, arrays["y_data"])
    out: list[int | None] = []
    for name in selected_x_feature_names:
        out.append(y_names.index(name) if name in y_names else None)
    return out


def plot_random_samples(
    arrays: dict[str, np.ndarray],
    config: dict,
    *,
    split: Split = "train",
    n_samples: int = 3,
    features: Sequence[str | int] | str | int | None = None,
    view: ViewMode = "both",
    seed: int | None = 0,
    sample_indices: Sequence[int] | None = None,
    inverse_scale: bool = False,
    include_target: bool = False,
    marker_size: float = 18.0,
    linewidth: float = 1.2,
    figsize_per_feature: tuple[float, float] = (13.0, 2.5),
    plot_regular: bool = False,
    regular_csv_path: str | Path | None = None,
    regular_alpha: float = 0.35,
    regular_linewidth: float | None = None,
    regular_already_scaled: bool = False,
    show: bool = True,
) -> list[plt.Figure]:
    """
    Plotta sample singoli scelti a caso dallo split.

    Se sample_indices è passato, usa quegli indici GLOBALI di sample.
    Altrimenti estrae n_samples indici globali dentro config["sample_splits"][split].

    Per ogni sample:
        x timeline: global_sample ... global_sample + seq_len - 1
        y timeline: global_sample + seq_len ... global_sample + seq_len + pred_len - 1

    view="zero":
        x_data plottato con missing = 0.
    view="observed":
        solo nodi osservati secondo x_mask, collegati anche se non adiacenti.
        La linea di x viene disegnata esplicitamente tra i nodi osservati.
    view="both":
        due figure per ogni sample.

    plot_regular=True:
        sovrappone la serie regolare letta da regular_csv_path.
        L'intervallo x è disegnato in blu chiaro, l'intervallo y/futuro in
        arancione chiaro quando include_target=True. Se inverse_scale=False e
        regular_already_scaled=False, il CSV viene prima scalato con
        config["scaler"]. Il CSV deve essere esattamente quello usato per
        generare dataset.npz, altrimenti l'overlay sarà diverso.
    """
    x_data = arrays["x_data"]
    x_mask = arrays["x_mask"].astype(bool)
    y_data = arrays["y_data"]

    sample_start, sample_end = _split_sample_range(config, split)

    if sample_indices is None:
        rng = np.random.default_rng(seed)
        n_available = sample_end - sample_start
        n_pick = min(int(n_samples), n_available)
        sample_indices = rng.choice(np.arange(sample_start, sample_end), size=n_pick, replace=False)
        sample_indices = sorted(map(int, sample_indices))
    else:
        sample_indices = [int(s) for s in sample_indices]
        bad = [s for s in sample_indices if not sample_start <= s < sample_end]
        if bad:
            raise ValueError(
                f"Alcuni sample_indices non appartengono allo split '{split}' "
                f"[{sample_start}, {sample_end}): {bad}"
            )

    x_feature_idx, x_feature_names = _feature_indices(config["feature_cols"], features)
    y_feature_idx = _sample_feature_indices_for_y(config, arrays, x_feature_names)

    regular_values_full = None
    if plot_regular:
        if regular_csv_path is None:
            raise ValueError(
                "plot_regular=True richiede regular_csv_path, es. '../../data/regular/ETTh1.csv'."
            )
        regular_values_full = _load_regular_csv_values(
            regular_csv_path,
            config,
            x_feature_idx,
            inverse_scale=inverse_scale,
            regular_already_scaled=regular_already_scaled,
        )

    views = ["zero", "observed"] if view == "both" else [view]
    figures: list[plt.Figure] = []

    for global_sample in sample_indices:
        # IMPORTANTE: usare np.take, non x_data[global_sample, :, x_feature_idx].
        # Con advanced indexing NumPy può restituire (F, seq_len) quando F=1,
        # facendo sembrare che x abbia un solo timestep. Qui forziamo sempre
        # shape (seq_len, F).
        x = np.take(x_data[global_sample], x_feature_idx, axis=1).astype(np.float32)
        m = np.take(x_mask[global_sample], x_feature_idx, axis=1)

        if inverse_scale:
            x = _inverse_scale_values(x, config, x_feature_idx)

        seq_len = x.shape[0]
        pred_len = y_data.shape[1]
        t_x = global_sample + np.arange(seq_len)
        t_y = global_sample + seq_len + np.arange(pred_len)

        regular_x = (
            _regular_values_at_times(regular_values_full, t_x)
            if regular_values_full is not None
            else None
        )
        regular_y = (
            _regular_values_at_times(regular_values_full, t_y)
            if regular_values_full is not None and include_target
            else None
        )

        for v in views:
            n_feat = len(x_feature_names)
            fig_h = max(2.6, figsize_per_feature[1] * n_feat)
            fig, axes = plt.subplots(
                n_feat,
                1,
                figsize=(figsize_per_feature[0], fig_h),
                sharex=True,
                squeeze=False,
            )
            axes = axes.ravel()

            for f, ax in enumerate(axes):
                if regular_x is not None:
                    ax.plot(
                        t_x,
                        regular_x[:, f],
                        color="C0",
                        alpha=regular_alpha,
                        linewidth=regular_linewidth or linewidth * 1.15,
                        label="x regolare",
                        zorder=1,
                    )

                if v == "zero":
                    ax.plot(
                        t_x,
                        x[:, f],
                        color="C0",
                        linestyle="-",
                        linewidth=linewidth,
                        label="x irregolare, missing=0",
                        zorder=3,
                    )
                    title_mode = "missing = 0"
                else:
                    obs_f = m[:, f]
                    # Linea esplicita tra i soli nodi osservati: i missing non
                    # vengono plottati, ma i punti osservati non adiacenti sono
                    # collegati come nella timeline ricostruita.
                    ax.plot(
                        t_x[obs_f],
                        x[obs_f, f],
                        color="C0",
                        linestyle="-",
                        linewidth=linewidth,
                        label="x irregolare osservato",
                        zorder=4,
                    )
                    ax.scatter(
                        t_x[obs_f],
                        x[obs_f, f],
                        color="C0",
                        s=marker_size,
                        zorder=5,
                    )
                    title_mode = "observed-only nodes"

                if include_target and regular_y is not None and y_feature_idx[f] is not None:
                    ax.plot(
                        t_y,
                        regular_y[:, f],
                        color="C1",
                        alpha=regular_alpha,
                        linewidth=regular_linewidth or linewidth * 1.15,
                        label="y regolare",
                        zorder=1,
                    )

                if include_target and y_feature_idx[f] is not None:
                    y_f = y_data[global_sample, :, y_feature_idx[f]].astype(np.float32)
                    if inverse_scale:
                        # y usa lo stesso scaler se D_out==D; nel caso MS il target è una feature originaria.
                        target_name = x_feature_names[f]
                        target_idx_in_input = config["feature_cols"].index(target_name)
                        y_f = y_f * np.asarray(config["scaler"]["std"])[target_idx_in_input] \
                            + np.asarray(config["scaler"]["mean"])[target_idx_in_input]

                    ax.plot(
                        t_y,
                        y_f,
                        color="C1",
                        marker=".",
                        markersize=marker_size / 5,
                        linestyle="-",
                        linewidth=linewidth,
                        label="y_data target",
                        zorder=3,
                    )
                    ax.axvline(t_y[0], linestyle="--", linewidth=1.0, alpha=0.7)

                ax.set_title(
                    f"{split} | sample globale {global_sample} | "
                    f"{x_feature_names[f]} | {title_mode}"
                )
                ax.set_ylabel(x_feature_names[f])
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best")

            axes[-1].set_xlabel("timestep globale")
            fig.tight_layout()
            figures.append(fig)

    if show:
        plt.show()

    return figures


def check_regular_alignment(
    arrays: dict[str, np.ndarray],
    config: dict,
    *,
    regular_csv_path: str | Path,
    split: Split = "train",
    features: Sequence[str | int] | str | int | None = None,
    inverse_scale: bool = False,
    regular_already_scaled: bool = False,
    n_checks: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Controlla numericamente che il CSV regolare sia allineato al dataset.npz.

    Confronta punti osservati di x_data[j, local_t, feature] con il CSV al
    timestep globale j + local_t. Se il CSV è quello usato per generare il file
    statico, gli errori devono essere ~0, salvo piccole differenze float.

    Returns
    -------
    DataFrame con colonne:
        sample_idx, local_t, global_t, feature, x_saved, regular, abs_error
    """
    x_data = arrays["x_data"]
    x_mask = arrays["x_mask"].astype(bool)
    sample_start, sample_end = _split_sample_range(config, split)
    feature_idx, feature_names = _feature_indices(config["feature_cols"], features)

    regular = _load_regular_csv_values(
        regular_csv_path,
        config,
        feature_idx,
        inverse_scale=inverse_scale,
        regular_already_scaled=regular_already_scaled,
    )

    rng = np.random.default_rng(seed)
    candidates = []
    for j in range(sample_start, sample_end):
        m_j = np.take(x_mask[j], feature_idx, axis=1)
        local_t, local_f = np.where(m_j)
        if len(local_t) == 0:
            continue
        for lt, lf in zip(local_t, local_f):
            candidates.append((j, int(lt), int(lf)))

    if not candidates:
        raise ValueError(f"Nessun punto osservato trovato nello split '{split}'.")

    n = min(int(n_checks), len(candidates))
    chosen = rng.choice(len(candidates), size=n, replace=False)

    rows = []
    for idx in chosen:
        j, lt, lf = candidates[int(idx)]
        f_global = feature_idx[lf]
        x_val = float(x_data[j, lt, f_global])
        if inverse_scale:
            mean = float(config["scaler"]["mean"][f_global])
            std = float(config["scaler"]["std"][f_global])
            x_val = x_val * std + mean

        gt = j + lt
        reg_val = float(regular[gt, lf])
        rows.append(
            {
                "sample_idx": j,
                "local_t": lt,
                "global_t": gt,
                "feature": feature_names[lf],
                "x_saved": x_val,
                "regular": reg_val,
                "abs_error": abs(x_val - reg_val),
            }
        )

    df = pd.DataFrame(rows).sort_values("abs_error", ascending=False).reset_index(drop=True)
    print(
        "Alignment check | "
        f"max_abs_error={df['abs_error'].max():.6g}, "
        f"mean_abs_error={df['abs_error'].mean():.6g}, "
        f"n={len(df)}"
    )
    return df


# -------------------------------------------------------------------------
# Esempio d'uso: equivalente al tuo snippet iniziale
# -------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = "etth1"
    dataset_param = "seq96_pred96_async_sp0.600_seed0"

    dataset_dir = Path(f"../../data/irregular/{dataset}/{dataset_param}")
    arrays, config = load_static_irregular_npz(dataset_dir)

    print("Data loaded!")
    print("x_data:", arrays["x_data"].shape)
    print("x_mask:", arrays["x_mask"].shape)
    print("y_data:", arrays["y_data"].shape)
    print("sample_splits:", config["sample_splits"])

    # 1) Timeline ricostruita dello split, solo target OT, prime 1000 posizioni.
    plot_split_timeline(
        arrays,
        config,
        split="train",
        features=["OT"],
        view="both",
        merge_policy="first",
        inverse_scale=False,
        max_points=1000,
        plot_regular=True,
        regular_csv_path="../../data/regular/ETTh1.csv",
    )

    # 2) Tre sample casuali dello split test, con x irregolare + y target.
    plot_random_samples(
        arrays,
        config,
        split="test",
        n_samples=3,
        features=["OT","LULL", "HUFL", "HULL", "MUFL"],
        view="both",
        seed=42,
        inverse_scale=False,
        include_target=True,
        plot_regular=True,
        regular_csv_path="../../data/regular/ETTh1.csv",
    )