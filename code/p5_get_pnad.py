#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Construct a Matched Panel of PNAD Continua households using Osorio (2022).
# ---------------------------------------------------------------------------
#=

# ---- Packages --------------------------------------------------------------
import os, time
import pandas as pd
from shutil import rmtree
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
    # Keep the dictionary, the deflators and the quarter zips (of "years", or all).
    p = f["path"]
    if not p.startswith("trimestral"): return False
    if p == "trimestral/Documentacao":
        return f["name"].startswith(("Dicionario_e_input", "Deflatores"))
    if years is None: return p.split("/")[-1].isdigit()
    return p.split("/")[-1] in {str(y) for y in years}


def panels_covering_year(year):
    # Pids with at least one visit in "year": 4 started in year-1, 4 in year.
    return sorted([(year - 1) * 10 + q for q in range(1, 5)] +
                  [year * 10 + q for q in range(1, 5)])


# ---------------------------------------------------------------------------
# 1. Download
def download(folder, years, redownload=False):
    if not Path(folder, copia_local.IDFILE).is_file(): copia_local.register(folder)
    remote = copia_local.list_remote_files()
    if not remote: raise ConnectionError("Cannot reach IBGE FTP (ports 20000-21000).")
    local = copia_local.list_local_files(folder)
    for f in remote:
        if not _keep_file(f, years): continue
        if not redownload and f in local: continue    # skip files already on disk
        with _quiet(): copia_local.download_manager([f], folder)
        print(f"[down] {f['name']} done")
    copia_local.register(folder)


# ---------------------------------------------------------------------------
# 2. Metadata + Convert to parquet + Identify panels to csv
def build(folder, years, only):
    with _quiet(): metadados.generate(folder)

    updates = converter.verify(folder, _PARQUET)[0]
    yrs = None if years is None else set(years)  # None -> convert every quarter
    for u in [u for u in updates if yrs is None
              or int(Path(u[0]).name.split("_")[1][2:6]) in yrs]:  # name -> year
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

    # dictionary + column layout + deflator kept in data/pnad/ (used by R survey code)
    for zp in ("Dicionario_e_input*.zip", "Deflatores.zip"):
        dz = next((tri / "Documentacao").glob(zp), None)
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
def run_all(folder, target_year=2025, redownload=False):
    # Build the matched panel for a target year. target_year=None -> all years/panels.
    start = time.time()
    make_noninteractive()
    if target_year is None:
        years, pids = None, None                    # every quarter, every panel
    else:
        years = range(target_year - 1, target_year + 2)
        pids = panels_covering_year(target_year)
        if not redownload:
            pids = [p for p in pids if not
                    Path(folder, "final", f"{paineis.FILESTUB}{p}.csv").is_file()]
            if not pids:
                print("Skipping download, panels already in final/."); return []
    download(folder, years, redownload)
    built = build(folder, years, pids)
    collect(folder, years)
    for name in (copia_local.COPIA_LOCAL, converter.MICRO, paineis.PAINEIS,
                 metadados.META, "pynad"):    # Removes pynad's scaffoding.
        rmtree(Path(folder, name), ignore_errors=True)
    print(f"Done in {(time.time()-start)/60:.1f}min ({(time.time()-start)/3600:.1f})h.")
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
    run_all("data/pnad/", None, False)
    # panels = load_year_panels("data/pnad/", 2025)   # {pid: DataFrame}