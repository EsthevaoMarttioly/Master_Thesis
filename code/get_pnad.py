#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Construct a Matched Panel of PNAD Continua households using Osorio (2022).
# ---------------------------------------------------------------------------
#=

# ---- Packages --------------------------------------------------------------
import os
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from contextlib import redirect_stdout

from pynad import auxiliares, copia_local, converter, metadados, paineis


# ---------------------------------------------------------------------------
# Helpers
_PARQUET, _CSV = [False, False], [True, False]


def make_noninteractive():
    # Silence pynad's y/n prompts, ENTER pauses and screen clears.
    for mod in (copia_local, converter, metadados, paineis, auxiliares):
        if hasattr(mod, "continuar"): mod.continuar = lambda *a, **k: True
        if hasattr(mod, "pausar"):    mod.pausar    = lambda *a, **k: None
        if hasattr(mod, "cli_main"):  mod.cli_main  = lambda *a, **k: None


def _quiet():
    # Swallow pynad's verbose progress so we can print one line per quarter.
    return redirect_stdout(open(os.devnull, "w"))


def _keep_file(f, years):
    # Keep the dictionary and the quarter zips (of "years", or all if years is None).
    p = f["path"]
    if not p.startswith("trimestral"): return False
    if p == "trimestral/Documentacao": return f["name"].startswith("Dicionario_e_input")
    if years is None: return p.split("/")[-1].isdigit()
    return p.split("/")[-1] in {str(y) for y in years}


def panels_covering_year(year):
    # Pids with at least one visit in "year": 4 started in year-1, 4 in year.
    return sorted([(year - 1) * 10 + q for q in range(1, 5)] +
                  [year * 10 + q for q in range(1, 5)])


def _year_of(name):
    # Year of a quarterly file: 'PNADC_012025.zip' -> 2025.
    return int(Path(name).name.split("_")[1][2:6])


# ---------------------------------------------------------------------------
# 1. Download
def download(folder, years):
    if not Path(folder, copia_local.IDFILE).is_file(): copia_local.register(folder)
    remote = copia_local.list_remote_files()
    if not remote: raise ConnectionError("Cannot reach IBGE FTP (ports 20000-21000).")
    local = copia_local.list_local_files(folder)
    for f in [f for f in remote if _keep_file(f, years) and f not in local]:
        with _quiet(): copia_local.download_manager([f], folder)
        print(f"[down] {f['name']} done")
    copia_local.register(folder)


# ---------------------------------------------------------------------------
# 2. Metadata + Convert to parquet + Identify panels to csv
def build(folder, years, only):
    with _quiet(): metadados.generate(folder)

    updates = converter.verify(folder, _PARQUET)[0]
    for u in [u for u in updates if _year_of(u[0]) in set(years)]:
        with _quiet(): converter.conversion_manager(folder, [u], _PARQUET)
        print(f"[conv] {Path(u[0]).name} done")

    panels = paineis.verify(folder, _CSV)[0]
    if only is not None:
        only = set(only)
        skipped = sorted(only - set(paineis.available(folder)))
        if skipped: print(f"[panel] not released yet, skipped: {skipped}")
        panels = {p: v for p, v in panels.items() if p in only}
    for pid, pan in panels.items():
        with _quiet(): paineis.build_manager(folder, {pid: pan}, _CSV)
        print(f"[panel] {pid} done")
    return sorted(panels)


# ---------------------------------------------------------------------------
# 3. Collect Outputs
def collect(folder, years):
    raw, final = Path(folder, "raw"), Path(folder, "final")
    tri = Path(folder, copia_local.COPIA_LOCAL, "trimestral")

    # final/ : the matched panel csv (copied from pynad's paineis/csv)
    if final.is_dir():
        for csv in Path(folder, paineis.PAINEISCSV).glob(f"{paineis.FILESTUB}*.csv"):
            (final / csv.name).write_bytes(csv.read_bytes())

    # dictionary + column layout kept in data/pnad/ (used by the R survey code)
    dz = next((tri / "Documentacao").glob("Dicionario_e_input*.zip"), None)
    if dz:
        with ZipFile(dz) as z: z.extractall(folder)

    # raw/ : the unzipped fixed-width quarter .txt
    if raw.is_dir():
        yrs = [p.name for p in tri.glob("*") if p.name.isdigit()] if years is None \
              else [str(y) for y in years]
        for y in yrs:
            ydir = tri / y
            for zp in sorted(ydir.glob("PNADC_*.zip")) if ydir.is_dir() else []:
                with ZipFile(zp) as z:
                    inner = z.namelist()[0]              # PNADC_012025.txt
                    if (raw / Path(inner).name).exists(): continue
                    with z.open(inner) as s, open(raw / Path(inner).name, "wb") as d:
                        while (c := s.read(1 << 20)): d.write(c)
                print(f"[raw] {Path(inner).name} done")


# ---------------------------------------------------------------------------
def run_all(folder, target_year=2025, do_download=True):
    # Build the matched panel for a target calendar year.
    make_noninteractive()
    years = range(target_year - 1, target_year + 2)
    if do_download: download(folder, years)
    built = build(folder, years, panels_covering_year(target_year))
    collect(folder, years)
    print(f"done — panels {built} in final/, raw txt in raw/")
    return built


# ---------------------------------------------------------------------------
# Loading
def load_panel(folder, pid):
    # One matched panel as a DataFrame.
    return pd.read_csv(Path(folder, "final", f"{paineis.FILESTUB}{pid}.csv"))


def load_year_panels(folder, year):
    # {pid: DataFrame} for every built panel covering "year".
    out = {}
    for pid in panels_covering_year(year):
        p = Path(folder, "final", f"{paineis.FILESTUB}{pid}.csv")
        if p.is_file(): out[pid] = pd.read_csv(p)
    return out


# ---------------------------------------------------------------------------
# Running
if __name__ == "__main__":
    # download("data/pnad/", None)
    run_all("data/pnad/", target_year=2025)
    # run_all("data/pnad/", target_year=2025, do_download=False)
    # panels = load_year_panels("data/pnad/", 2025)   # {pid: DataFrame}