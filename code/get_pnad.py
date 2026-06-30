#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Osorio (2022) Method for Matching PNAD's Panels.
# import run_pynad as rp
# rp.run_all("./pnadc", prefs="parquet")
# tbl = rp.load_table("./pnadc", 20254)       # pyarrow Table
# df  = tbl.to_pandas()
# ---------------------------------------------------------------------------
#=

# Import Packages
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from itertools import chain
from json import loads, dumps

from pynad import auxiliares, copia_local, converter, metadados, paineis


_PREFS = {
    "parquet"        : (False, False),      # Only Parquet
    "parquet+csvzip" : (True, True),        # Parquet + ZIP CSV
    "parquet+csv"    : (True, False),       # Parquet + Raw CSV (huge)
}


# ---------------------------------------------------------------------------
# Make Pynad non-interactive
def make_noninteractive():
    # Disable all blocking CLI prompts in Pynad.
    for mod in (copia_local, converter, metadados, paineis):
        if hasattr(mod, "continuar"): mod.continuar = lambda *a, **k: True
        if hasattr(mod, "pausar"):    mod.pausar    = lambda *a, **k: None
        if hasattr(mod, "cli_main"):  mod.cli_main  = lambda *a, **k: None
    auxiliares.continuar = lambda *a, **k: True
    auxiliares.pausar    = lambda *a, **k: None



def _write_prefs(folder, prefs='parquet'):
    # Persist the csv preferences to MPFILE so library and disk agree.
    Path(folder, copia_local.MPFILE).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(folder, copia_local.MPFILE), "w", encoding="utf-8") as f:
        f.write(dumps(list(prefs)))



# Keep just doc and files with an year
_SHARED_DOCS = frozenset({"Dicionario_e_input_PNADC_trimestral.zip"})

def _year_filter(files, years):
    # Keep files whose name contains any target year, plus shared docs.
    strs = {str(y) for y in years}
    return [f for f in files if f["name"] in _SHARED_DOCS
            or any(s in f["name"] for s in strs)]



def selective_sync(folder, years):
    """
    Download only cross-section files for the target calendar years.

    IMPORTANT: this bypasses copia_local.sync() entirely.
    Why: sync() computes excluir = local_files - remote_filtered, which would
    DELETE every file outside the year filter (e.g. all your pre-existing 2012–
    2023 files).  By calling download_manager() directly we only ADD files,
    never remove anything.

    years : e.g. [2024, 2025, 2026]
            Rule of thumb — include the year of the LAST VISIT of your
            youngest target panel:
              pid 20244 (2024Q4 → 2025Q4)  →  years = [2024, 2025]
              pid 20251 (2025Q1 → 2026Q1)  →  years = [2024, 2025, 2026]
    """
    print(f"Connecting to IBGE (year filter: {years})...")
    all_files = copia_local.list_remote_files()
    if not all_files:
        raise ConnectionError(
            "Cannot reach IBGE. Check network and FTP ports 20000-21000.")

    needed = _year_filter(all_files, years)
    local  = copia_local.list_local_files(folder)
    baixar = [f for f in needed if f not in local]

    n_total = len(all_files)
    n_need  = len(needed)
    n_dl    = len(baixar)
    sz_dl   = sum(f["size"] for f in baixar) / 1e9

    print(f"  Total IBGE files : {n_total}")
    print(f"  After year filter: {n_need}  ({n_need/n_total*100:.0f}%)")
    print(f"  To download      : {n_dl}  ({sz_dl:.1f} GB)")

    if baixar:
        copia_local.download_manager(baixar, folder)   # calls register() internally
    else:
        copia_local.register(folder)
        print("  All target files already present.")


def _resolve_pids(only):
    # Accept ints, range objects
    if only is None:
        return None
    return sorted({int(p) for item in only
                   for p in (item if isinstance(item, range) else [item])})



# ---------------------------------------------------------------------------
# Pipeline stages
def setup(folder, prefs="parquet", do_sync=True):
    # Creates a local folder copy or verify an existing one
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if Path(folder, copia_local.IDFILE).is_file():
        print(f"Existing local copy found.")
    else:
        copia_local.register(folder)
        print(f"Initialized a new local copy.")

    # Clone IBGE's microdata into 'originais/'
    if do_sync:
        copia_local.sync(folder)
        print("Sync: IBGE files downloaded.")
    else:
        print("Sync: SKIPPED.")

    # Parse the .xls variable into json
    metadados.generate(folder)
    print("Metadata: Parsed xls into json.")

    # Converte microdata to parquet (and optional csv)
    p = _PREFS[prefs]
    _write_prefs(folder, p)
    converter.sync_microdados(folder, list(p))
    print(f'Convert: Converted microdata to {prefs}.')



def build_panels(folder, prefs="parquet", only=None):
    """ Separate panels and identify households and individuals
    ----------
    only : optional list of panel ids (pids) (e.g. [20244, 20254]) to build.
           Already-built panels are preserved. Missing pids are skipped with a warning.
    """
    p  = _PREFS[prefs]
    folder = Path(folder)
    _write_prefs(folder, p)
    only = _resolve_pids(only)

    updates, excluir, regprq = paineis.verify(folder, list(p))

    if only is not None:
        complete = set(paineis.available(folder).keys())
        missing  = sorted(set(only) - complete)
        if missing:
            print(f"WARNING: panels not yet complete: {missing}")
        updates = {pid: panel for pid, panel in updates.items() if pid in only}

    # Drop orphans and rewrite registry (mirrors sync_paineis bookkeeping)
    for arq in excluir:
        arq.unlink()
    with open(Path(folder, paineis.PAINEISPRQ, metadados.METALIST),
              "w", encoding="utf-8") as f:
        if regprq:
            f.write("\n".join(dumps(r) for r in regprq) + "\n")

    if not updates:
        print("Panels: already up to date.")
        return []

    built = sorted(updates.keys())
    paineis.build_manager(folder, updates, list(p))
    print(f"Panels: Construct {len(built)} panel(s): {built}.")
    return built


# ---------------------------------------------------------------------------
# Run the complete pipeline
def run_all(folder, prefs="parquet", years=None, only=None, do_sync=True):
    make_noninteractive()
    setup(folder, prefs, do_sync)
    if years:
        selective_sync(folder, years)
    else:
        copia_local.sync(folder)
    built = build_panels(folder, prefs, only=only)
    print("Pipeline finished.")
    return built


# ---------------------------------------------------------------------------
# Load Panel as Table
def load_panel(folder, pid: int):
    # Load a built panel as a pyarrow Table (.to_pandas() for pandas).
    path = Path(folder, paineis.PAINEISPRQ, f"{paineis.FILESTUB}{pid}.parquet")
    if not path.is_file():
        raise FileNotFoundError(f"Panel {pid} not found. Run build_panels().")
    return pq.read_table(path)


def load_panel_metadata(folder, pid: int):
    # Load the json metadata (variable labels, categories) for a panel.
    path = Path(folder, paineis.PAINEISMET, f"{paineis.METASTUB}{pid}.json")
    with open(path, encoding="utf-8") as f:
        return loads(f.read())



# ---------------------------------------------------------------------------
# Run everything
run_all("data/pnad/", "parquet", do_sync = True, years=[2024,2025],
        only = list(range(20241, 20245)) + list(range(20251, 20254)))



