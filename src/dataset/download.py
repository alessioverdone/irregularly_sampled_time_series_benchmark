from __future__ import annotations

import io
import os
import gzip
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

"""
download.py
-----------
Scarica i dataset di time-series ETTh1, ETTm1 ed Electricity (LD2011_2014)
e li salva nella cartella ./data/ in formato CSV.

- ETTh1 / ETTm1: dal repository ufficiale https://github.com/zhouhaoyi/ETDataset
- Electricity:   dall'archivio UCI (LD2011_2014.txt.zip), poi convertito a CSV
                 con la stessa struttura usata dai benchmark (Informer, Autoformer,
                 PatchTST, ecc.): 321 clienti come colonne + colonna 'date'.
"""


# ----------------------------------------------------------------------------- #
# Configurazione URL                                                            #
# ----------------------------------------------------------------------------- #
ETTH1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)
ETTH2_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
)
ETTM1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
)
ETTM2_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv"
)
ELECTRICITY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/"
    "LD2011_2014.txt.zip"
)

# --- Fotovoltaico --------------------------------------------------------- #
# SolarBenchmark (a.k.a. "Solar-Energy"): 137 impianti PV simulati in Alabama,
# anno 2006, campionamento ogni 10 minuti. Equivalente al dataset usato in
# Informer/Autoformer/PatchTST. Il file è T×N float separati da virgola, senza
# header e senza colonna data: la data va ricostruita.
SOLAR_URL = (
    "https://github.com/TorchSpatiotemporal/multivariate-time-series-data/"
    "blob/master/solar-energy/solar_AL.txt.gz?raw=true"
)

# PvUS (NREL): produzione PV simulata da ~5000 impianti USA, 2006, 5 minuti
# (in tsl ricampionati a 10 minuti). MOLTO PESANTE (alcuni GB), opt-in.
# Il pacchetto raw è ospitato come release del repo tsl.
PVUS_URL = (
    "https://github.com/TorchSpatiotemporal/tsl/releases/download/"
    "0.9.1/pv_us.zip"
)


# ----------------------------------------------------------------------------- #
# Utility                                                                       #
# ----------------------------------------------------------------------------- #
def _download_bytes(url: str) -> bytes:
    """Scarica un URL in memoria (con User-Agent per evitare 403)."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def _save_bytes(data: bytes, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


# ----------------------------------------------------------------------------- #
# Download dei singoli dataset                                                  #
# ----------------------------------------------------------------------------- #
def download_etth1(data_dir: Path) -> Path:
    """Scarica ETTh1.csv (sample orario, 7 colonne + date)."""
    out = data_dir / "ETTh1.csv"
    if out.exists():
        print(f"[ETTh1] già presente: {out}")
        return out
    print(f"[ETTh1] download da {ETTH1_URL}")
    _save_bytes(_download_bytes(ETTH1_URL), out)
    print(f"[ETTh1] salvato in {out}")
    return out


def download_etth2(data_dir: Path) -> Path:
    """Scarica ETTh2.csv (sample orario, 7 colonne + date)."""
    out = data_dir / "ETTh2.csv"
    if out.exists():
        print(f"[ETTh2] già presente: {out}")
        return out
    print(f"[ETTh2] download da {ETTH2_URL}")
    _save_bytes(_download_bytes(ETTH2_URL), out)
    print(f"[ETTh2] salvato in {out}")
    return out


def download_ettm1(data_dir: Path) -> Path:
    """Scarica ETTm1.csv (sample 15 min, 7 colonne + date)."""
    out = data_dir / "ETTm1.csv"
    if out.exists():
        print(f"[ETTm1] già presente: {out}")
        return out
    print(f"[ETTm1] download da {ETTM1_URL}")
    _save_bytes(_download_bytes(ETTM1_URL), out)
    print(f"[ETTm1] salvato in {out}")
    return out


def download_ettm2(data_dir: Path) -> Path:
    """Scarica ETTm2.csv (sample 15 min, 7 colonne + date)."""
    out = data_dir / "ETTm2.csv"
    if out.exists():
        print(f"[ETTm2] già presente: {out}")
        return out
    print(f"[ETTm2] download da {ETTM2_URL}")
    _save_bytes(_download_bytes(ETTM2_URL), out)
    print(f"[ETTm2] salvato in {out}")
    return out


def download_electricity(data_dir: Path) -> Path:
    """
    Scarica e processa il dataset Electricity (LD2011_2014).
    Ritorna il path di un CSV con colonna 'date' + 321 colonne cliente,
    risampleato a frequenza oraria (somma sui 4 quarti d'ora), come da
    pre-processing standard nei lavori Informer/Autoformer/PatchTST.
    """
    out = data_dir / "electricity.csv"
    if out.exists():
        print(f"[Electricity] già presente: {out}")
        return out

    print(f"[Electricity] download da {ELECTRICITY_URL}")
    raw = _download_bytes(ELECTRICITY_URL)

    print("[Electricity] estrazione zip e parsing CSV…")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # Il file all'interno si chiama "LD2011_2014.txt"
        inner_name = next(n for n in zf.namelist() if n.endswith(".txt"))
        with zf.open(inner_name) as fh:
            df = pd.read_csv(
                fh,
                sep=";",
                decimal=",",
                index_col=0,
                parse_dates=True,
                low_memory=False,
            )

    # I dati grezzi sono a 15 minuti in kW (energia su 15min * 4 = kWh/h
    # nelle convenzioni UCI). Per uniformarci al pre-processing standard
    # dei benchmark, risamplamo a 1H sommando i 4 quarti d'ora.
    df = df.astype("float32")
    df_hourly = df.resample("1h").sum()

    # I primi clienti hanno solo zeri all'inizio (entrarono in rete dopo).
    # Manteniamo tutti i 321 clienti come fanno i benchmark; nessun filtro.
    df_hourly = df_hourly.reset_index().rename(columns={"index": "date"})
    # Assicuriamoci che la colonna data si chiami 'date'
    if df_hourly.columns[0] != "date":
        df_hourly = df_hourly.rename(columns={df_hourly.columns[0]: "date"})

    df_hourly.to_csv(out, index=False)
    print(
        f"[Electricity] salvato in {out} "
        f"(shape={df_hourly.shape}, freq=1h)"
    )
    return out


def download_solar(data_dir: Path) -> Path:
    """
    Scarica il dataset SolarBenchmark (a.k.a. "Solar-Energy"):
    137 impianti PV simulati in Alabama, anno 2006, ogni 10 minuti, 52560 step.

    Il file raw `solar_AL.txt.gz` non ha né header né colonna data. Qui:
      - decomprimiamo,
      - leggiamo le 137 colonne float,
      - ricostruiamo l'asse temporale (2006-01-01 00:00 a passo di 10 min),
      - salviamo come `solar_AL.csv` con colonna 'date' + 137 colonne 'plant_000…plant_136',
        coerenti con lo schema usato per gli altri dataset del progetto.
    """
    out = data_dir / "solar_AL.csv"
    if out.exists():
        print(f"[Solar] già presente: {out}")
        return out

    print(f"[Solar] download da {SOLAR_URL}")
    raw = _download_bytes(SOLAR_URL)

    print("[Solar] decompressione e parsing…")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        df = pd.read_csv(gz, header=None, dtype=np.float32)

    n_steps, n_plants = df.shape
    df.columns = [f"plant_{i:03d}" for i in range(n_plants)]
    # Asse temporale: 2006, sample @ 10 min
    dates = pd.date_range("2006-01-01", periods=n_steps, freq="10min")
    df.insert(0, "date", dates)

    df.to_csv(out, index=False)
    print(
        f"[Solar] salvato in {out} "
        f"(shape={df.shape}, freq=10min, anno=2006)"
    )
    return out


def download_pvus(data_dir: Path, skip_if_large: bool = True) -> Path | None:
    """
    Scarica il dataset PvUS (NREL): ~5000 impianti PV simulati USA, 2006.

    ATTENZIONE: archivio pesante (qualche GB). Per default questa funzione NON
    lo scarica se `skip_if_large=True` (comportamento di default in `download_all`),
    ma puoi forzarla con `skip_if_large=False`.

    NB: il formato raw di PvUS è strutturato per zona (east/west) con file
    separati per metadata e serie temporali. Per mantenere l'API uniforme col
    resto del pipeline (un singolo CSV con colonna `date` + colonne impianto),
    qui ci limitiamo a salvare l'archivio zip estratto in `data_dir/pv_us/`,
    senza creare un mega-CSV unificato. La classe `PvUSDataset` (vedi
    datasets.py) leggerà il file giusto al volo.
    """
    target_dir = data_dir / "pv_us"
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[PvUS] già presente: {target_dir}")
        return target_dir

    if skip_if_large:
        print(
            "[PvUS] SKIPPED: dataset pesante (qualche GB). "
            "Chiama esplicitamente download_pvus(data_dir, skip_if_large=False) "
            "per scaricarlo."
        )
        return None

    print(f"[PvUS] download da {PVUS_URL} (può richiedere diversi minuti)…")
    raw = _download_bytes(PVUS_URL)

    print("[PvUS] estrazione zip…")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(target_dir)
    print(f"[PvUS] estratto in {target_dir}")
    return target_dir


# ----------------------------------------------------------------------------- #
# Entry-point                                                                   #
# ----------------------------------------------------------------------------- #
def download_all(
    data_dir: str | os.PathLike = "./data",
    include_pvus: bool = False,
) -> dict[str, Path]:
    """
    Scarica tutti i dataset principali. Ritorna un dict nome → path.

    Args:
        data_dir     : cartella di destinazione.
        include_pvus : se True scarica anche PvUS (~GB). Default False.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "ETTh1": download_etth1(data_dir),
        "ETTh2": download_etth2(data_dir),
        "ETTm1": download_ettm1(data_dir),
        "ETTm2": download_ettm2(data_dir),
        "Electricity": download_electricity(data_dir),
        "Solar": download_solar(data_dir),
    }
    if include_pvus:
        pvus_path = download_pvus(data_dir, skip_if_large=False)
        if pvus_path is not None:
            paths["PvUS"] = pvus_path
    return paths


if __name__ == "__main__":
    download_all("../../data", include_pvus=False)
