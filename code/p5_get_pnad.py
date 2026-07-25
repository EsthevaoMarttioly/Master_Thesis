#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Construct a Matched Panel of PNAD Continua households using Osorio (2022).
#----------------------------------------------------------------------------
#=

# ---- Packages --------------------------------------------------------------
import os, time
import pandas as pd
from shutil import rmtree
from pathlib import Path
from zipfile import ZipFile
from contextlib import redirect_stdout

from pynad import auxiliares, copia_local, converter, metadados, paineis

_PARQUET, _CSV = [False, False], [True, False]   # pynad (make_csv, zip_csv) flags


# ---------------------------------------------------------------------------
# Helpers
def make_noninteractive():
    # Silence pynad's y/n prompts, ENTER pauses and screen clears.
    for mod in (copia_local, converter, metadados, paineis, auxiliares):
        if hasattr(mod, "continuar"): mod.continuar = lambda *a, **k: True
        if hasattr(mod, "pausar"):    mod.pausar    = lambda *a, **k: None
        if hasattr(mod, "cli_main"):  mod.cli_main  = lambda *a, **k: None


def _quiet():
    return redirect_stdout(open(os.devnull, "w"))   # silence pynad's progress


def _qi(name):
    # Absolute quarter index of a data file: 'PNADC_012025.zip' -> 2025*4 + 0.
    return int(name[8:12]) * 4 + int(name[6:8]) - 1


def _is_quarter(f):
    # A quarterly microdata file (path is a year folder, name is PNADC_QQYYYY).
    return f["path"].split("/")[-1].isdigit() and f["name"].startswith("PNADC_")


def _panel_qis(pid):
    start = pid // 10 * 4 + pid % 10 - 1       # pid = year*10 + quarter
    return set(range(start, start + 5))


def panels_covering_year(year):
    # Pids with a visit in "year": the 4 started in year-1 and the 4 in year.
    return [(year - 1) * 10 + q for q in range(1, 5)] + [year * 10 + q for q in range(1, 5)]


def buildable_pids(target_year, available):
    # Complete panels (all 5 quarters on the server) covering a year.
    if target_year is None:
        candidates = {i // 4 * 10 + i % 4 + 1 for i in available}
    else:
        candidates = set(panels_covering_year(target_year))
    return sorted(p for p in candidates if _panel_qis(p) <= available)


# ---------------------------------------------------------------------------
# 1. Download
def _wanted(f, need):
    # A needed quarter file, or the dictionary/deflator documents.
    if f["path"] == "trimestral/Documentacao":
        return f["name"].startswith(("Dicionario_e_input", "Deflatores"))
    return _is_quarter(f) and _qi(f["name"]) in need


def download(folder, remote, need, redownload=False):
    if not Path(folder, copia_local.IDFILE).is_file(): copia_local.register(folder)
    local = copia_local.list_local_files(folder)
    for f in remote:
        if not _wanted(f, need): continue
        if not redownload and f in local: continue    # already on disk
        with _quiet(): copia_local.download_manager([f], folder)
        print(f"[down] {f['name']} done")
    copia_local.register(folder)


# ---------------------------------------------------------------------------
# 2. Convert quarters, then identify panels
def build(folder, need, pids):
    with _quiet(): metadados.generate(folder)
    for u in converter.verify(folder, _PARQUET)[0]:
        if _qi(Path(u[0]).name) not in need: continue
        with _quiet(): converter.conversion_manager(folder, [u], _PARQUET)
        print(f"[conv] {Path(u[0]).name} done")
    pids = set(pids)
    panels = {p: v for p, v in paineis.verify(folder, _CSV)[0].items() if p in pids}
    for pid, pan in panels.items():
        with _quiet(): paineis.build_manager(folder, {pid: pan}, _CSV)
    return sorted(panels)


# ---------------------------------------------------------------------------
# 3. Collect Outputs
def collect(folder, need, pids):
    tri = Path(folder, copia_local.COPIA_LOCAL, "trimestral")

    # final/ : the matched panel csv of each panel we built
    for pid in pids:
        src = Path(folder, paineis.PAINEISCSV, f"{paineis.FILESTUB}{pid}.csv")
        (Path(folder, "final") / src.name).write_bytes(src.read_bytes())

    # data/pnad/ : dictionary, column layout and deflator
    for pattern in ("Dicionario_e_input*.zip", "Deflatores.zip"):
        z = next((tri / "Documentacao").glob(pattern), None)
        if z:
            with ZipFile(z) as zf: zf.extractall(folder)

    # raw/ : the unzipped fixed-width .txt of the quarters we used
    for ydir in tri.glob("*"):
        for zp in sorted(ydir.glob("PNADC_*.zip")) if ydir.name.isdigit() else []:
            if _qi(zp.name) not in need: continue
            with ZipFile(zp) as zf:
                inner = zf.namelist()[0]                      # PNADC_012025.txt
                out = Path(folder, "raw") / inner
                if out.exists(): continue
                with zf.open(inner) as s, open(out, "wb") as d:
                    while (c := s.read(1 << 20)): d.write(c)
            print(f"[raw] {inner} done")


# ---------------------------------------------------------------------------
def run_all(folder, target_year=2025, redownload=False):
    # Download the quarters, build every panels covering target_year.
    start = time.time()
    make_noninteractive()
    remote = copia_local.list_remote_files(); print("\n")
    if not remote: raise ConnectionError("Cannot reach IBGE FTP (ports 20000-21000).")

    available = {_qi(f["name"]) for f in remote if _is_quarter(f)}
    pids = buildable_pids(target_year, available)
    if not redownload:
        pids = [p for p in pids
                if not Path(folder, "final", f"{paineis.FILESTUB}{p}.csv").is_file()]
    if not pids:
        print("Nothing to do - panels already in final/."); return []

    need = set().union(*(_panel_qis(p) for p in pids))    # exact quarters to fetch
    download(folder, remote, need, redownload)
    built = build(folder, need, pids)
    collect(folder, need, built)
    for name in (copia_local.COPIA_LOCAL, converter.MICRO, paineis.PAINEIS,
                 metadados.META, "pynad"):
        rmtree(Path(folder, name), ignore_errors=True)     # drop pynad scaffolding
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
    run_all("data/pnad/", None)
    panels = load_year_panels("data/pnad/", 2025)   # {pid: DataFrame}
    df = load_panel("data/pnad/", 20251)

df
panels[20251]