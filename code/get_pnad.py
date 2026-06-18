#=
#----------------------------------------------------------------------------
# DESCRIPTION
# Osorio (2022) Method for Matching PNAD's Panels
# ---------------------------------------------------------------------------
#=
"""
    1. sync       -> clone/refresh IBGE's `Microdados` tree into `originais/`
    2. metadata   -> parse the .xls dictionaries into json
    3. convert    -> fixed-width .txt  ->  csv  ->  parquet
    4. panels     -> separate panels, identify households & individuals,
                     reshape to one wide record per individual (vars _1.._5)

You were right: Pynad fetches the data itself. The `sync` stage lists IBGE's
FTP server and downloads the ZIP archives (each ZIP holds the fixed-width .txt)
into `<folder>/originais/...`. Pynad's converter consumes those ZIPs, not loose
.txt files, so dropping bare .txt into the folder will not work. `sync` is
idempotent: it compares name+size and only pulls what is missing, so re-running
it is cheap. If you already downloaded IBGE's *Microdados* directory tree
yourself, place it verbatim under `<folder>/originais/` and call `setup()` once;
`sync` will then see the files as already present.

REQUIREMENTS
------------
    pip install pynad            # pulls pyarrow and xlrd
Network: outbound FTP to ftp.ibge.gov.br (control port 21, passive 20000-21000)
for listing, plus HTTPS to https://ftp.ibge.gov.br for the downloads.
Disk: tens of GB for parquet; ~200 GB if you also keep uncompressed csv.

USAGE (CLI)
-----------
    # full run into ./pnadc, parquet only, all complete panels
    python run_pynad.py --folder ./pnadc

    # only build two specific panels (faster), keep a zipped csv copy too
    python run_pynad.py --folder ./pnadc --only 20162 20171 --prefs parquet+csvzip

    # skip the (slow) IBGE sync because the local copy is already current
    python run_pynad.py --folder ./pnadc --no-sync

    # just list the complete panels available in the converted microdata
    python run_pynad.py --folder ./pnadc --list

USAGE (as a library)
--------------------
    import run_pynad as rp
    rp.run_all("./pnadc", prefs="parquet")                # everything
    tbl = rp.load_panel("./pnadc", 20162)                 # pyarrow Table
    df  = tbl.to_pandas()                                 # -> pandas

Panel id (pid) = year*10 + quarter of the FIRST visit. 20162 == 2016 Q2.
"""

from __future__ import annotations

import argparse
import sys
import time
from json import loads
from pathlib import Path
import pynad


_PREFS = {
    "parquet"        : (False, False),      # Only Parquet
    "parquet+csvzip" : (True, True),        # Parquet + ZIP CSV
    "parquet+csv"    : (True, False),       # Parquet + Raw CSV (huge)
}



# ---------------------------------------------------------------------------
# Make Pynad non-interactive
def make_noninteractive() -> None:
    """Neutralise the blocking CLI helpers in every Pynad module.

    Pynad uses `from .auxiliares import continuar, pausar, cli_main`, so each
    name is rebound inside the importing module's namespace. We overwrite it
    there: `continuar()` always proceeds, `pausar()` is a no-op, and
    `cli_main()` no longer clears the terminal (progress prints stay linear).
    """
    for module in (pynad.copia_local, pynad.converter, pynad.metadados, pynad.paineis):
        if hasattr(module, "continuar"):
            module.continuar = lambda *a, **k: True
        if hasattr(module, "pausar"):
            module.pausar = lambda *a, **k: None
        if hasattr(module, "cli_main"):
            module.cli_main = lambda *a, **k: None
    # Also silence the source helpers, in case anything calls them directly.
    pynad.auxiliares.continuar = lambda *a, **k: True
    pynad.auxiliares.pausar = lambda *a, **k: None


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_prefs(folder: str | Path, prefs) -> None:
    # Persist the csv preferences to MPFILE so library and disk agree.
    from json import dumps
    Path(folder, pynad.copia_local.MPFILE).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(folder, pynad.copia_local.MPFILE), "w", encoding="utf-8") as tgt:
        tgt.write(dumps(list(prefs)))


# ---------------------------------------------------------------------------
# Pipeline stages
def setup(folder: str | Path) -> None:
    """Register a local copy (creates `originais/` and the `pynad/` control
    files) or verify an existing one. Safe to call repeatedly."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if Path(folder, pynad.copia_local.IDFILE).is_file():
        data = pynad.copia_local.verify(folder)
        intact = data[2] if data else False
        _log(f"Existing local copy found (last update {data[1]}; "
             f"intact={intact}).")
    else:
        pynad.copia_local.register(folder)
        _log(f"Initialised a new local copy at: {folder}")


def sync_ibge(folder: str | Path) -> None:
    """Stage 1 - clone/refresh IBGE's microdata into `originais/`.

    Downloads only missing/changed files (compared by name+size)."""
    _log("Stage 1/4  sync: contacting IBGE and downloading missing files...")
    pynad.copia_local.sync(folder)
    _log("Stage 1/4  sync: done.")


def build_metadata(folder: str | Path) -> None:
    """Stage 2 - parse the .xls variable dictionaries into json."""
    _log("Stage 2/4  metadata: parsing variable dictionaries...")
    pynad.metadados.generate(folder)
    _log("Stage 2/4  metadata: done.")


def convert_microdata(folder: str | Path, prefs="parquet") -> None:
    """Stage 3 - convert fixed-width microdata to parquet (and optional csv)."""
    prefs = _PREFS[prefs]
    _log(f"Stage 3/4  convert: fixed-width -> parquet (prefs={tuple(prefs)})...")
    _write_prefs(folder, prefs)
    pynad.converter.sync_microdados(folder, list(prefs))
    _log("Stage 3/4  convert: done.")


def list_available_panels(folder: str | Path) -> list[int]:
    """Return the pids of every COMPLETE panel present in the converted
    microdata (a panel needs all five quarterly cross-sections)."""
    if not Path(folder, pynad.converter.MICROPRQ, pynad.metadados.METALIST).is_file():
        raise RuntimeError(
            "No converted microdata found. Run stages 1-3 first "
            "(sync -> metadata -> convert).")
    pids = sorted(pynad.paineis.available(folder).keys())
    return pids


def build_panels(folder: str | Path, prefs="parquet",
                 only: list[int] | None = None) -> list[int]:
    """Stage 4 - separate panels and identify households & individuals
    (Osorio 2022), then reshape to one wide record per individual.

    Parameters
    ----------
    only : optional list of pids (e.g. [20162, 20171]) to build just those
           panels instead of every complete one. Already-built panels are
           preserved; requested-but-missing pids are skipped with a warning.

    Returns the list of pids actually built in this call.
    """
    prefs = _PREFS[prefs]
    folder = Path(folder)
    _write_prefs(folder, prefs)
    prefs = list(prefs)

    # verify() creates the panel folders, prunes orphans, and returns the
    # set of panels still to build plus the surviving registry.
    updates, excluir, regprq = pynad.paineis.verify(folder, prefs)

    if only is not None:
        want = set(int(p) for p in only)
        all_complete = set(pynad.paineis.available(folder).keys())
        missing = sorted(want - all_complete)
        if missing:
            _log(f"WARNING: requested panels not complete/available and "
                 f"skipped: {missing}")
        updates = {pid: panel for pid, panel in updates.items()
                   if pid in want}

    # Replicate sync_paineis bookkeeping (drop orphans, rewrite registry).
    from json import dumps
    for arq in excluir:
        arq.unlink()
    with open(Path(folder, pynad.paineis.PAINEISPRQ, pynad.metadados.METALIST),
              "w", encoding="utf-8") as tgt:
        if regprq:
            tgt.write("\n".join(dumps(reg) for reg in regprq) + "\n")

    if not updates:
        _log("Stage 4/4  panels: nothing to build (already up to date).")
        return []

    built = sorted(updates.keys())
    _log(f"Stage 4/4  panels: building {len(built)} panel(s): {built}")
    pynad.paineis.build_manager(folder, updates, prefs)
    _log("Stage 4/4  panels: done.")
    return built


def run_all(folder: str | Path, prefs="parquet",
            only: list[int] | None = None, do_sync: bool = True) -> list[int]:
    """Run the complete pipeline and return the pids built in this call."""
    prefs = _PREFS[prefs]
    make_noninteractive()
    setup(folder)
    if do_sync:
        sync_ibge(folder)
    else:
        _log("Stage 1/4  sync: SKIPPED (--no-sync).")
    build_metadata(folder)
    convert_microdata(folder, prefs)
    built = build_panels(folder, prefs, only=only)
    _log("Pipeline finished.")
    return built


# --------------------------------------------------------------------------- #
# Convenience loaders for analysis
# --------------------------------------------------------------------------- #
def panel_path(folder: str | Path, pid: int) -> Path:
    return Path(folder, pynad.paineis.PAINEISPRQ, f"{pynad.paineis.FILESTUB}{pid}.parquet")


def load_panel(folder: str | Path, pid: int):
    """Load a built panel as a pyarrow Table (use .to_pandas() for pandas)."""
    import pyarrow.parquet as pq
    path = panel_path(folder, pid)
    if not path.is_file():
        raise FileNotFoundError(
            f"Panel {pid} not found at {path}. Build it first with "
            f"build_panels(folder, only=[{pid}]).")
    return pq.read_table(path)


def load_panel_metadata(folder: str | Path, pid: int) -> dict:
    """Load the json metadata (variable labels, categories) for a panel."""
    path = Path(folder, pynad.paineis.PAINEISMET, f"{pynad.paineis.METASTUB}{pid}.json")
    with open(path, encoding="utf-8") as src:
        return loads(src.read())


def pid_to_label(pid: int) -> str:
    """20162 -> '2016 Q2'."""
    return f"{pid // 10} Q{pid % 10}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified non-interactive driver for Pynad (Osorio 2022 "
                    "PNADC panels).")
    parser.add_argument("--folder", required=True,
                        help="Local copy folder (created if absent).")
    parser.add_argument("--prefs", default="parquet", choices=list(_PREFS),
                        help="Output formats (default: parquet only).")
    parser.add_argument("--only", nargs="*", type=int, default=None,
                        metavar="PID",
                        help="Build only these panel ids, e.g. 20162 20171.")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip the IBGE download/sync stage.")
    parser.add_argument("--list", action="store_true",
                        help="List complete panels in the converted microdata "
                             "and exit (no building).")
    args = parser.parse_args(argv)

    make_noninteractive()

    if args.list:
        setup(args.folder)
        try:
            pids = list_available_panels(args.folder)
        except RuntimeError as err:
            _log(str(err))
            return 1
        _log(f"{len(pids)} complete panel(s) available:")
        for pid in pids:
            print(f"  {pid}  ({pid_to_label(pid)})")
        return 0

    run_all(args.folder, prefs=_PREFS[args.prefs],
            only=args.only, do_sync=not args.no_sync)
    return 0


if __name__ == "__main__":
    sys.exit(_main())