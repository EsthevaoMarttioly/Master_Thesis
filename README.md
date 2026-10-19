# Master's Thesis

Author: **Esthevao Marttioly Lopes Martins**

Program: MSc Economics - FGV EESP

Advisor: Bernardo Guimarães

This repository contains the code and results for the **Master's Thesis** by Esthevão Marttioly from FGV EESP.

All code is written with reproducibility and defensive programming in mind.

## Project Structure

```
├── .venv                                # Package version lockfile
├── code/
│   ├── main.py                          # Master script - runs the main model
│   ├── p1_household.py                  # Household's block
│   ├── p2_other_blocks.py               # Firm, Fiscal, Monetary, and Mkt Clearing
│   ├── p3_pnad_matching.py              # PNAD: Panel Matching
│   ├── p4_analysis.R                    # PNAD: Calibration Data
│   ├── p5_calibration.py                # Parameters
│   ├── p6_solve.py                      # Solve Steady State, Dynamics, and SMM
│   ├── p7_results.py                    # Graphics and Tables
│   └── p8_robustness.py                 # Resolve Steady State with different values
├── data/
│   ├── pnad/                            # PNAD Data
│   ├── final/                           # Calibration Data
│   ├── visdata3-*.csv                   # CadÚnico Data
│   └── lorenz_nw_scf_2019.raw           # SCF Data for US Lorenz Curve, from Auclert et al. (2025)
├── output/
│   ├── figures/                         # Figures
│   ├── smm/                             # Internal Calibration runs
│   └── tables/                          # Tables in LaTeX
├── paper/
│   ├── biblio.bib                       # References in bibtex
│   ├── project.tex                      # Paper in LaTeX
│   └── project.pdf                      # Final Paper
├── project.RProj                        # R Project for downloading data
├── requirements.txt                     # pip install -r requirements.txt
└── README.md
```


## Computational Environment

The analysis was conducted using Python version 3.12.10 and R version 4.6.0 (2026-06-09) on a Windows 11 system.


## Data

The PNAD microdata is not in the repository, because of IBGE terms of use.
To download it, run `code/p3_pnad_matching.py`. At the bottom of the file, call

```python
run_all("data/pnad/", target_year=2025, redownload=True)
```

After that, `code/p4_analysis.R` downloads the annual file and create all output data files.
If you would like the full history, set `run_all(, target_year=None)` in `p3_pnad_matching.py`
and then change each switch at the top of `p4_analysis.R` to `TRUE`. With all three `FALSE`,
the script rebuilds only the annual targets, which are what the model reads.


## Running the project

To reproduce the analysis:

* Open the project's folder as a project.
* Open the file: code/main.py.
* In the Terminal, type "pip install -r requirements.txt" on bash (Ctrl+Shift+' to open).
* Run the script: code/main.py.

To change any calibration parameter outside the SMM, set in `p5_calibration.py`.
To run the SMM estimation, follow the instructions at the bottom of `p6_solve.py`.