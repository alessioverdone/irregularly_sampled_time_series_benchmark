"""
irregular_plot_utils_v2.py
--------------------------
Plot di comparazione visiva tra serie temporali sparsificate con meccanismi diversi.

Dato un dataset, una sparsità e un campione di interesse, genera n dataset irregolari
statici (uno per meccanismo di sparsificazione) nella cartella
data/irregular/<dataset>/temp/, poi crea una figura con le n versioni della stessa
finestra temporale (stesso sample_idx, stesso canale).

Usa le funzioni di irregular_plot_utils.py dove possibile.

Uso standalone:
    python -m src.dataset.irregular_plot_utils_v2
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.dataset.irregular_plot_utils import (
    load_static_irregular_npz,
    _feature_indices,
    _inverse_scale_values,
    _load_regular_csv_values,
    _regular_values_at_times,
    _split_sample_range,
)
from src.dataset.irregular_datasets_static import sparsify_and_save
from src.dataset.irregular_datasets import SparsifyConfig, MissingnessMechanism

Split = Literal["train", "val", "test"]
OutputFormat = Literal["png", "pdf"]


# ─────────────────────────────────────────────────────────────────────────── #
# Funzioni di supporto                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _temp_save_dir(
    data_dir_irr: str | Path,
    dataset: str,
    seq_len: int,
    pred_len: int,
    mechanism: str,
    sparsity: float,
    seed: int,
) -> Path:
    """Cartella temp/ per un dato meccanismo di sparsificazione."""
    subdir = f"seq{seq_len}_pred{pred_len}_{mechanism}_sp{sparsity:.3f}_seed{seed}"
    return Path(data_dir_irr) / dataset / "temp" / subdir


# ─────────────────────────────────────────────────────────────────────────── #
# API pubblica                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def generate_mechanism_datasets(
    csv_path: str | Path,
    data_dir_irr: str | Path,
    dataset: str,
    split_config: dict,
    mechanisms: Sequence[MissingnessMechanism],
    sparsity: float,
    *,
    seq_len: int = 96,
    pred_len: int = 96,
    seed: int = 0,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Genera (o riutilizza) un dataset irregolare statico per ogni meccanismo,
    salvando nella cartella data/irregular/<dataset>/temp/.

    Parameters
    ----------
    csv_path     : CSV sorgente regolare (es. data/regular/ETTh1.csv)
    data_dir_irr : cartella radice dei dataset irregolari (data/irregular/)
    dataset      : nome del dataset (es. "etth1")
    split_config : configurazione degli split (da DATASET_SPLIT_CONFIGS)
    mechanisms   : lista di meccanismi da generare
    sparsity     : sparsità target [0, 1)
    seq_len      : lunghezza finestra di input
    pred_len     : lunghezza orizzonte di previsione
    seed         : seed per la generazione delle maschere
    force        : se True rigenera anche se i file esistono già
    verbose      : stampa messaggi di log

    Returns
    -------
    dict  mechanism -> Path della save_dir corrispondente
    """
    paths: dict[str, Path] = {}
    for mech in mechanisms:
        save_dir = _temp_save_dir(
            data_dir_irr, dataset, seq_len, pred_len, mech, sparsity, seed
        )
        cfg = SparsifyConfig(mechanism=mech, sparsity=sparsity, seed=seed)
        sparsify_and_save(
            csv_path=csv_path,
            save_dir=save_dir,
            split_config=split_config,
            seq_len=seq_len,
            pred_len=pred_len,
            sparsify_cfg=cfg,
            force=force,
            verbose=verbose,
        )
        paths[mech] = save_dir
    return paths


def plot_mechanism_comparison(
    dataset_paths: dict[str, Path],
    sample_idx: int,
    channel: int | str,
    *,
    split: Split = "train",
    inverse_scale: bool = False,
    view: Literal["observed", "zero", "both"] = "observed",
    plot_regular: bool = False,
    regular_csv_path: str | Path | None = None,
    regular_already_scaled: bool = False,
    regular_alpha: float = 0.35,
    marker_size: float = 18.0,
    linewidth: float = 1.2,
    figsize: tuple[float, float] | None = None,
    output_format: OutputFormat = "png",
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Crea una figura di comparazione tra meccanismi di sparsificazione diversi.

    Per ogni meccanismo in dataset_paths:
      - carica il dataset irregolare statico dalla save_dir corrispondente
      - estrae il sample `sample_idx` dello split `split`
      - plotta il canale `channel` in un subplot separato

    Layout: n_meccanismi righe × n_view colonne (1 se view!="both", 2 se "both").

    Parameters
    ----------
    dataset_paths : dict mechanism -> Path della save_dir
    sample_idx    : indice GLOBALE del sample (deve appartenere a `split` in
                    tutti i dataset)
    channel       : feature da plottare (nome stringa o indice intero)
    split         : "train" | "val" | "test"
    inverse_scale : se True, riporta alla scala originale con config["scaler"]
    view          : "observed" (solo nodi osservati con scatter + linea),
                    "zero" (missing sostituiti con 0),
                    "both" (due colonne: zero e observed)
    plot_regular  : sovrappone la serie regolare come riferimento grigio
    regular_csv_path : CSV sorgente (richiesto se plot_regular=True)
    output_format : "png" | "pdf"
    output_path   : percorso di output; None = non salva su file.
                    Se il suffisso non è .png/.pdf viene sostituito con
                    output_format.
    show          : se True chiama plt.show()

    Returns
    -------
    fig : plt.Figure
    """
    fontsize_title = 15
    fontsize_ylabel = 12
    fontsize_xlabel = 12
    fontsize_legend = 10

    mechanisms = list(dataset_paths.keys())
    n_mech = len(mechanisms)
    if n_mech == 0:
        raise ValueError("dataset_paths è vuoto: nessun meccanismo da plottare.")

    views = ["zero", "observed"] if view == "both" else [view]
    n_views = len(views)

    fig_w, fig_h = figsize or (13.0 * n_views, 3.2 * n_mech + 1.0)
    fig, axes = plt.subplots(
        # n_mech, n_views,
        2,2,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    feat_name_global: str | None = None

    for row_idx, mech in enumerate(mechanisms):
        save_dir = dataset_paths[mech]
        arrays, config = load_static_irregular_npz(save_dir)

        feature_cols = list(config["feature_cols"])
        feat_idx, feat_names = _feature_indices(feature_cols, channel)
        fi = feat_idx[0]
        feat_name = feat_names[0]
        if feat_name_global is None:
            feat_name_global = feat_name

        sample_start, sample_end = _split_sample_range(config, split)
        if not (sample_start <= sample_idx < sample_end):
            raise ValueError(
                f"sample_idx={sample_idx} non appartiene allo split '{split}' "
                f"[{sample_start}, {sample_end}) per il meccanismo '{mech}'."
            )

        seq_len = int(config["seq_len"])

        x = arrays["x_data"][sample_idx, :, fi].astype(np.float32)  # (seq_len,)
        m = arrays["x_mask"][sample_idx, :, fi].astype(bool)         # (seq_len,)

        if inverse_scale:
            x = _inverse_scale_values(x[:, np.newaxis], config, [fi])[:, 0]

        t_x = sample_idx + np.arange(seq_len)
        obs_rate = float(m.sum()) / len(m)
        color = palette[row_idx % len(palette)]

        regular_vals: np.ndarray | None = None
        if plot_regular:
            if regular_csv_path is None:
                raise ValueError("plot_regular=True richiede regular_csv_path.")
            reg_full = _load_regular_csv_values(
                regular_csv_path, config, [fi],
                inverse_scale=inverse_scale,
                regular_already_scaled=regular_already_scaled,
            )
            regular_vals = _regular_values_at_times(reg_full, t_x)[:, 0]

        for col_idx, v in enumerate(views):
            # ax = axes[row_idx, col_idx]
            ax = axes.flatten()[row_idx]

            if regular_vals is not None:
                ax.plot(
                    t_x, regular_vals,
                    color="gray", alpha=regular_alpha,
                    linewidth=linewidth * 1.1,
                    label="Regular Time Series", zorder=1,
                                    )

            if v == "zero":
                ax.plot(
                    t_x, x,
                    color=color, linewidth=linewidth,
                    label=f"{mech} | missing=0", zorder=2,
                )
                view_label = "missing = 0"
            else:  # "observed"
                ax.plot(
                    t_x[m], x[m],
                    color=color, linestyle="-", linewidth=linewidth,
                    label="Irregular Time Series", zorder=3,
                    # label=f"{mech} | osservato", zorder=3,
                )
                ax.scatter(t_x[m], x[m], color=color, s=marker_size, zorder=4)
                view_label = "observed-only"
                # view_label = " "

            target_sparsity = config["sparsify_cfg"]["sparsity"]
            # ax.set_title(
            #     f"[{mech.upper()}]  |  {view_label}  "
            #     f"|  sparsity={target_sparsity:.0%}  obs={obs_rate:.1%}",
            #     fontsize=9,
            # )


            ax.set_title(
                f"{mech.upper()} sparsity={target_sparsity:.0%}  obs={obs_rate:.1%}",
                fontsize=fontsize_title,
            )
            ax.set_ylabel(feat_name, fontsize=fontsize_ylabel)
            # ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=fontsize_legend)

            if row_idx == n_mech - 1:
                ax.set_xlabel("timestep globale", fontsize=fontsize_xlabel)

    dataset_name = list(dataset_paths.values())[0].parents[1].name
    # fig.suptitle(
    #     f"Confronto meccanismi di sparsificazione  |  dataset={dataset_name}  "
    #     f"channel={feat_name_global}  sample={sample_idx}  split={split}",
    #     fontsize=11, fontweight="bold",
    # )
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() not in (".png", ".pdf"):
            out = out.with_suffix(f".{output_format}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[v2] Figura salvata → {out}")

    if show:
        plt.show()

    return fig


def run_mechanism_comparison(
    csv_path: str | Path,
    data_dir_irr: str | Path,
    dataset: str,
    split_config: dict,
    mechanisms: Sequence[MissingnessMechanism],
    sparsity: float,
    sample_idx: int,
    channel: int | str,
    *,
    seq_len: int = 96,
    pred_len: int = 96,
    split: Split = "train",
    seed: int = 0,
    inverse_scale: bool = False,
    view: Literal["observed", "zero", "both"] = "observed",
    plot_regular: bool = False,
    regular_already_scaled: bool = False,
    regular_alpha: float = 0.35,
    marker_size: float = 18.0,
    linewidth: float = 1.2,
    figsize: tuple[float, float] | None = None,
    output_format: OutputFormat = "png",
    output_path: str | Path | None = None,
    force: bool = False,
    verbose: bool = True,
    show: bool = True,
) -> plt.Figure:
    """
    Funzione tutto-in-uno: genera i dataset irregolari per ogni meccanismo,
    poi crea e (opzionalmente) salva la figura di comparazione.

    Combina generate_mechanism_datasets() + plot_mechanism_comparison().

    Parameters
    ----------
    csv_path      : CSV sorgente regolare
    data_dir_irr  : cartella data/irregular/
    dataset       : nome del dataset (es. "etth1")
    split_config  : dict di configurazione degli split
    mechanisms    : lista di meccanismi da confrontare
    sparsity      : sparsità target
    sample_idx    : indice GLOBALE del sample da plottare
    channel       : canale da plottare (nome stringa o indice intero)
    seq_len       : lunghezza finestra input
    pred_len      : lunghezza orizzonte previsione
    split         : split di riferimento per il sample_idx
    seed          : seed maschere
    inverse_scale : riporta alla scala originale
    view          : "observed" | "zero" | "both"
    plot_regular  : sovrappone la serie regolare (usa csv_path come sorgente)
    output_format : "png" | "pdf"
    output_path   : percorso di output (None = non salva)
    force         : rigenera i dataset anche se già presenti
    verbose       : stampa messaggi di log
    show          : chiama plt.show()

    Returns
    -------
    fig : plt.Figure
    """
    regular_csv_path: Path | None = Path(csv_path) if plot_regular else None

    print(f"[v2] Generazione dataset per meccanismi: {list(mechanisms)}")
    dataset_paths = generate_mechanism_datasets(
        csv_path=csv_path,
        data_dir_irr=data_dir_irr,
        dataset=dataset,
        split_config=split_config,
        mechanisms=mechanisms,
        sparsity=sparsity,
        seq_len=seq_len,
        pred_len=pred_len,
        seed=seed,
        force=force,
        verbose=verbose,
    )

    print("[v2] Creazione plot comparativo...")
    return plot_mechanism_comparison(
        dataset_paths=dataset_paths,
        sample_idx=sample_idx,
        channel=channel,
        split=split,
        inverse_scale=inverse_scale,
        view=view,
        plot_regular=plot_regular,
        regular_csv_path=regular_csv_path,
        regular_already_scaled=regular_already_scaled,
        regular_alpha=regular_alpha,
        marker_size=marker_size,
        linewidth=linewidth,
        figsize=figsize,
        output_format=output_format,
        output_path=output_path,
        show=show,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    from src.config import DATASET_SPLIT_CONFIGS

    # ── Parametri utente ───────────────────────────────────────────────── #
    RAND = random.randint(0,100000000)
    DATASET       = "etth1"
    SPARSITY      = 0.25
    MECHANISMS    = ["mcar", "burst", "periodic", "async"]
    SAMPLE_IDX    = 100      # indice GLOBALE del sample; deve appartenere a SPLIT
    CHANNEL       = "OT"     # nome stringa (es. "OT") o indice intero (es. 0)
    SPLIT         = "train"  # "train" | "val" | "test"
    SEQ_LEN       = 96
    PRED_LEN      = 96
    SEED          = 0
    INVERSE_SCALE = False
    VIEW          = "observed"  # "observed" | "zero" | "both"
    PLOT_REGULAR  = True        # sovrappone la serie regolare come riferimento
    OUTPUT_FORMAT = "pdf"       # "png" | "pdf"
    OUTPUT_PATH   = f"/home/user/Scrivania/PhD/EEEIC/MIO/plots/comparison_{DATASET}_{CHANNEL}_{str(SPARSITY).split('.')[-1]}_{SAMPLE_IDX}_{SEQ_LEN}_{PRED_LEN}_{RAND}"        # None = auto; es. "/path/to/comparison.pdf"
    SHOW          = True
    FORCE_REGEN   = False       # True = rigenera i dataset anche se esistono già
    FIGSIZE       = (30,10)
    LINEWIDTH     = 3
    MARKERSIZE    = 120
    # ────────────────────────────────────────────────────────────────────── #
    print(OUTPUT_PATH)
    _project_dir  = Path(__file__).resolve().parents[2]
    _data_dir     = _project_dir / "data"
    _data_dir_irr = _data_dir / "irregular"

    _csv_registry: dict[str, Path] = {
        "etth1":       _data_dir / "regular" / "ETTh1.csv",
        "etth2":       _data_dir / "regular" / "ETTh2.csv",
        "ettm1":       _data_dir / "regular" / "ETTm1.csv",
        "ettm2":       _data_dir / "regular" / "ETTm2.csv",
        "electricity": _data_dir / "regular" / "electricity.csv",
        "solar":       _data_dir / "regular" / "solar_AL.csv",
    }

    _csv_path     = _csv_registry[DATASET]
    _split_config = DATASET_SPLIT_CONFIGS[DATASET]

    if OUTPUT_PATH is None:
        _mech_str   = "_".join(MECHANISMS)
        OUTPUT_PATH = (
            _data_dir_irr / DATASET
            / f"comparison_{DATASET}_{_mech_str}"
            f"_sp{SPARSITY:.0%}_sample{SAMPLE_IDX}_ch{CHANNEL}.{OUTPUT_FORMAT}"
        )

    run_mechanism_comparison(
        csv_path=_csv_path,
        data_dir_irr=_data_dir_irr,
        dataset=DATASET,
        split_config=_split_config,
        mechanisms=MECHANISMS,
        sparsity=SPARSITY,
        sample_idx=SAMPLE_IDX,
        channel=CHANNEL,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        split=SPLIT,
        seed=SEED,
        inverse_scale=INVERSE_SCALE,
        view=VIEW,
        plot_regular=PLOT_REGULAR,
        output_format=OUTPUT_FORMAT,
        output_path=OUTPUT_PATH,
        force=FORCE_REGEN,
        verbose=True,
        show=SHOW,
        figsize=FIGSIZE,
        linewidth=LINEWIDTH,
        marker_size=MARKERSIZE
    )
