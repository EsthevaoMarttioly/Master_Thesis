# ---------------------------------------------------------------------------
# get_pnad.py
# ---------------------------------------------------------------------------
# Build a MATCHED (longitudinal) panel of PNAD Continua households using the
# method of Osorio (2022), "Sobre a montagem e a identificacao dos paineis da
# PNAD Continua" (IPEA, Mercado de Trabalho n.73), as implemented by his
# `pynad` package (v3.0.3).
#
# WHAT THIS SCRIPT DOES
# ---------------------
# `pynad` is normally an interactive terminal menu. This module drives its
# internal functions programmatically and NON-interactively, and adds two
# things the menu does not offer:
#   1. a YEAR FILTER, so you only download / convert the quarters you need
#      (instead of the full 2012-today history, > 25 GB);
#   2. an automatic selection of the PANELS that cover a target calendar year,
#      including the "border" panels that begin the year before and finish the
#      year after.
#
# THE PIPELINE (same 4 stages as the pynad menu, in the correct order)
#   1. setup            -> create / verify the local copy folder
#   2. selective_sync   -> download ONLY the target-year cross-section files
#   3. metadata         -> parse the IBGE variable dictionary (.xls -> .json)
#   4. convert          -> cross-section fixed-width .txt  ->  .parquet
#   5. build_panels     -> separate panels + identify households/individuals,
#                          writing one wide "individual-record" file per panel
#
# WHY A MATCHED PANEL NEEDS THREE YEARS
#   A PNADC panel visits the same dwellings 5 times, one quarter apart, so it
#   spans ~15 months and straddles calendar years. To observe every household
#   that appears in 2025 across ALL of its quarters you need:
#       - panels that STARTED in 2024 (their last visits fall in 2025)  -> left border
#       - panels that START   in 2025                                    -> core
#       - the 2026 quarters that COMPLETE the panels started in 2025     -> right border
#   Hence years = (2024, 2025, 2026) for target_year = 2025.
#
# TYPICAL USE
#   import get_pnad as gp
#   gp.run_all("data/pnad/", target_year=2025)          # download+build
#   tbl = gp.load_panel("data/pnad/", 20251)            # a single panel (pyarrow)
#   df  = tbl.to_pandas()
#   panels = gp.load_year_panels("data/pnad/", 2025)    # every 2025 panel -> dict
#
# OUTPUT
#   data/pnad/paineis/parquet/pnadc.microdados.painel.<pid>.parquet
#   data/pnad/paineis/metadados/pnadc.variaveis.painel.<pid>.json
# where <pid> = AAAAT (year*10 + quarter of the panel's FIRST visit), e.g.
# 20251 = panel first visited in 2025-Q1. In each file, a variable measured in
# every visit appears as v3001_1 ... v3001_5 (one column per visit), and the
# individual is identified by (upa, v1008, pidgrp, pidind).
# ---------------------------------------------------------------------------

from json import loads, dumps
from pathlib import Path

import pyarrow.parquet as pq

# pynad internals (v3.0.3) - we call the engine functions directly and skip
# the interactive menu wrappers.
from pynad import auxiliares, copia_local, converter, metadados, paineis


# ---------------------------------------------------------------------------
# CSV / Parquet output preferences (pynad stores these in a config file).
#   (make_csv, zip_csv)
# Parquet is ALWAYS produced; the flags only control the intermediate CSVs.
# "parquet" is the right choice for analysis: smallest footprint, fastest load.
# ---------------------------------------------------------------------------
_PREFS = {
    "parquet":        (False, False),   # parquet only  (recommended)
    "parquet+csvzip": (True,  True),    # parquet + zipped csv
    "parquet+csv":    (True,  False),   # parquet + raw csv (very large)
}


# ---------------------------------------------------------------------------
# Make pynad non-interactive
# ---------------------------------------------------------------------------
def make_noninteractive():
    """Replace pynad's blocking terminal prompts with no-ops.

    pynad's engine functions call continuar() (a y/n prompt), pausar()
    ("press ENTER") and cli_main() (ANSI screen clear). We overwrite the
    references that were imported into EACH module's namespace, plus the
    originals in `auxiliares`, so nothing ever waits for keyboard input.
    """
    for mod in (copia_local, converter, metadados, paineis, auxiliares):
        if hasattr(mod, "continuar"):
            mod.continuar = lambda *a, **k: True     # always "yes"
        if hasattr(mod, "pausar"):
            mod.pausar = lambda *a, **k: None         # never wait
        if hasattr(mod, "cli_main"):
            mod.cli_main = lambda *a, **k: None       # don't clear the screen


# ---------------------------------------------------------------------------
# Preferences file helper
# ---------------------------------------------------------------------------
def _write_prefs(folder, prefs):
    """Persist the (make_csv, zip_csv) tuple so pynad and disk stay in sync."""
    mp = Path(folder, copia_local.MPFILE)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(dumps(list(prefs)))


# ---------------------------------------------------------------------------
# YEAR FILTER  (which remote files to keep)
# ---------------------------------------------------------------------------
# pynad describes every remote file by a dict {"path", "name", "size"} where
# `path` is the folder relative to Microdados, e.g.:
#     "trimestral/2025"          -> quarterly microdata zips  (PNADC_012025.zip)
#     "trimestral/Documentacao"  -> variable dictionary, deflators, etc.
#     "anual/..."                -> annual datasets (NOT needed for the panel)
#
# For a matched panel we only need the TRIMESTRAL (quarterly) product:
#   * every quarterly data zip whose year is in `years`, and
#   * the whole Documentacao folder (the dictionary is required to read them).
# Everything else - the entire `anual` tree, projection archives, root PDFs -
# is skipped. This is what keeps the download minimal.
# ---------------------------------------------------------------------------
def _keep_file(f, years):
    """True if remote file `f` is a trimestral file we need for `years`."""
    path = f["path"]
    if not path.startswith("trimestral"):
        return False                      # skip the whole 'anual' tree
    # keep the documentation folder (holds the required variable dictionary)
    if path == "trimestral/Documentacao":
        return True
    # keep data folders whose last segment is a target year: "trimestral/2025"
    last = path.split("/")[-1]
    return last in {str(y) for y in years}


def selective_sync(folder, years):
    """Download ONLY the trimestral cross-section files for `years`.

    This deliberately bypasses copia_local.sync(). pynad's own sync() computes
    `excluir = local_files - remote_filtered` and DELETES the difference; with a
    year filter that would wipe every quarter you already have outside the
    filter. By calling download_manager() ourselves we only ever ADD files.

    Already-present files are skipped (compared by name AND size, so IBGE
    patches - e.g. PNADC_022024_20260324.zip - are re-fetched when they change).
    """
    print(f"[sync] contacting IBGE FTP (year filter: {sorted(years)}) ...")
    remote = copia_local.list_remote_files()
    if not remote:
        raise ConnectionError(
            "Cannot reach IBGE FTP. Check the connection and that ports "
            "20000-21000 are open.")

    needed = [f for f in remote if _keep_file(f, years)]
    local = copia_local.list_local_files(folder)
    to_get = [f for f in needed if f not in local]

    gb = sum(f["size"] for f in to_get) / 1e9
    print(f"[sync] remote files total : {len(remote)}")
    print(f"[sync] kept by year filter : {len(needed)}")
    print(f"[sync] to download         : {len(to_get)} ({gb:.1f} GB)")

    if to_get:
        copia_local.download_manager(to_get, folder)   # also calls register()
    else:
        copia_local.register(folder)                   # refresh the file list
        print("[sync] all target files already present.")


# ---------------------------------------------------------------------------
# Panel-id helpers
# ---------------------------------------------------------------------------
# A panel id (pid) is AAAAT = year*10 + quarter of the FIRST visit.
# Its 5 visits fall on 5 consecutive quarters, so it spans this quarter and the
# following four. A panel therefore "covers" a calendar year if any of its 5
# visits lands in that year.
# ---------------------------------------------------------------------------
def panels_covering_year(year):
    """Return the pids of every panel that has at least one visit in `year`.

    Those are the panels first visited in `year` (4 of them) and in the previous
    year (4 of them, whose later visits reach `year`) - eight pids in total:
        year-1: Q1..Q4   (left-border panels, e.g. 20241..20244 for 2025)
        year  : Q1..Q4   (core + right-border panels, 20251..20254)
    Panels needing quarters IBGE has not released yet are filtered out later by
    build_panels(), which only builds panels whose 5 visits are all available.
    """
    pids = [(year - 1) * 10 + q for q in range(1, 5)]
    pids += [year * 10 + q for q in range(1, 5)]
    return sorted(pids)


def _trimestral_year(datazip_path):
    """Extract the 4-digit year from a quarterly data-file path/name.

    Names look like 'PNADC_012025.zip' or 'PNADC_022024_20260324.zip' -> the
    year is characters 2:6 of the QQYYYY token after 'PNADC_'.
    """
    token = Path(datazip_path).name.split("_")[1]   # e.g. "012025"
    return int(token[2:6])                          # e.g. 2025


# ---------------------------------------------------------------------------
# STAGE 1 - setup
# ---------------------------------------------------------------------------
def setup(folder):
    """Create the local-copy folder (or verify an existing one)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if Path(folder, copia_local.IDFILE).is_file():
        print(f"[setup] existing local copy found at {folder}")
    else:
        copia_local.register(folder)                # creates originais/ + index
        print(f"[setup] initialised a new local copy at {folder}")


# ---------------------------------------------------------------------------
# STAGE 3 - metadata
# ---------------------------------------------------------------------------
def metadata(folder):
    """Parse the IBGE .xls variable dictionary into the .json pynad needs.

    Cheap (reads one spreadsheet); regenerates metadados/ from scratch. Only the
    trimestral dictionary is used here because we never downloaded annual data.
    """
    metadados.generate(folder)
    print("[meta] variable dictionary parsed -> metadados/*.json")


# ---------------------------------------------------------------------------
# STAGE 4 - convert  (fixed-width .txt  ->  .parquet), filtered by year
# ---------------------------------------------------------------------------
def convert(folder, prefs="parquet", years=None):
    """Convert the quarterly microdata to parquet, but ONLY for `years`.

    This mirrors converter.sync_microdados()'s bookkeeping, except the list of
    files to convert is filtered to the target years - so we do not spend hours
    unpacking quarters (2012-2023) that no requested panel uses.
    """
    p = list(_PREFS[prefs])
    _write_prefs(folder, p)

    updates, excluir, regcsv, regprq = converter.verify(folder, p)

    # keep only the quarterly files whose year is requested
    if years is not None:
        yrs = {int(y) for y in years}
        updates = [u for u in updates if _trimestral_year(u[0]) in yrs]

    # drop orphan outputs and rewrite the two registries (as sync_microdados does)
    for arq in excluir:
        arq.unlink()
    with open(Path(folder, converter.MICROCSV, metadados.METALIST),
              "w", encoding="utf-8") as f:
        if regcsv:
            f.write("\n".join(dumps(r) for r in regcsv) + "\n")
    with open(Path(folder, converter.MICROPRQ, metadados.METALIST),
              "w", encoding="utf-8") as f:
        if regprq:
            f.write("\n".join(dumps(r) for r in regprq) + "\n")

    if updates:
        print(f"[conv] converting {len(updates)} quarterly file(s) to parquet ...")
        converter.conversion_manager(folder, updates, p)
    else:
        print("[conv] parquet files already up to date.")

    # in parquet-only mode the intermediate csv folder is disposable
    if p[0] is False and Path(folder, converter.MICROCSV).is_dir():
        from shutil import rmtree
        rmtree(Path(folder, converter.MICROCSV))
    print("[conv] done.")


# ---------------------------------------------------------------------------
# STAGE 5 - build panels  (separate panels + identify households/individuals)
# ---------------------------------------------------------------------------
def build_panels(folder, prefs="parquet", only=None):
    """Mount and identify panels; write one wide individual-record file each.

    only : iterable of pids to build (default: all complete panels). Panels that
           are not yet complete (a needed quarter is missing from IBGE) are
           reported and skipped. Already-built panels are left untouched.
    """
    p = list(_PREFS[prefs])
    _write_prefs(folder, p)
    only = None if only is None else {int(x) for x in only}

    updates, excluir, regprq = paineis.verify(folder, p)

    if only is not None:
        complete = set(paineis.available(folder).keys())
        missing = sorted(only - complete)
        if missing:
            print(f"[panel] not yet complete (skipped): {missing}")
        updates = {pid: pan for pid, pan in updates.items() if pid in only}

    # mirror sync_paineis() bookkeeping: drop orphans, rewrite the registry
    for arq in excluir:
        arq.unlink()
    with open(Path(folder, paineis.PAINEISPRQ, metadados.METALIST),
              "w", encoding="utf-8") as f:
        if regprq:
            f.write("\n".join(dumps(r) for r in regprq) + "\n")

    if not updates:
        print("[panel] nothing to build (already up to date).")
        return []

    built = sorted(updates.keys())
    print(f"[panel] building {len(built)} panel(s): {built} ...")
    paineis.build_manager(folder, updates, p)
    print(f"[panel] done -> paineis/parquet/")
    return built


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_all(folder,
            target_year=2025,
            years=None,
            prefs="parquet",
            do_download=True,
            only=None):
    """Run the whole matched-panel pipeline for a target calendar year.

    Parameters
    ----------
    folder       : local-copy folder (created if absent), e.g. "data/pnad/".
    target_year  : the year you want to study (default 2025). Determines which
                   panels are built via panels_covering_year().
    years        : quarters to download/convert. Default = target_year-1,
                   target_year, target_year+1 (left border, core, right border).
    prefs        : "parquet" (recommended), "parquet+csvzip" or "parquet+csv".
    do_download  : set False to (re)build from files already on disk.
    only         : override the automatic panel selection with an explicit list.
    """
    if years is None:
        years = (target_year - 1, target_year, target_year + 1)
    if only is None:
        only = panels_covering_year(target_year)

    make_noninteractive()
    setup(folder)
    if do_download:
        selective_sync(folder, years)
    else:
        print("[sync] SKIPPED (do_download=False).")
    metadata(folder)
    convert(folder, prefs, years=years)
    built = build_panels(folder, prefs, only=only)
    print(f"\nPipeline finished. Panels built for {target_year}: {built}")
    return built


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_panel(folder, pid):
    """Load one built panel as a pyarrow Table (.to_pandas() -> DataFrame)."""
    path = Path(folder, paineis.PAINEISPRQ, f"{paineis.FILESTUB}{pid}.parquet")
    if not path.is_file():
        raise FileNotFoundError(
            f"Panel {pid} not found at {path}. Run build_panels() first.")
    return pq.read_table(path)


def load_panel_metadata(folder, pid):
    """Load the json variable metadata (labels, categories) for one panel."""
    path = Path(folder, paineis.PAINEISMET, f"{paineis.METASTUB}{pid}.json")
    with open(path, encoding="utf-8") as f:
        return loads(f.read())


def load_year_panels(folder, year):
    """Load every BUILT panel that covers `year` -> {pid: pyarrow.Table}.

    Handy for calibration: iterate the households seen in `year` across all the
    panels (left-border, core, right-border) that observed them.
    """
    out = {}
    for pid in panels_covering_year(year):
        path = Path(folder, paineis.PAINEISPRQ, f"{paineis.FILESTUB}{pid}.parquet")
        if path.is_file():
            out[pid] = pq.read_table(path)
    return out


# ---------------------------------------------------------------------------
# Run when executed as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Main interest: 2025. Downloads/uses 2024 (left border), 2025 (core) and
    # 2026 (right border) quarters, and builds every panel that touches 2025.
    # Panels whose completing quarters IBGE has not released yet are skipped
    # automatically - rerun later to pick them up.
    run_all("data/pnad/", target_year=2025)
