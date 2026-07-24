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
│   ├── main.py                          # Master script - runs everything
│   ├── p1_household_block.py            # Household's block
│   ├── p2_other_blocks.py               # Firm, Fiscal, Monetary, and Mkt Clearing
│   ├── p3_parameters.py                 # Calibration Parameters
│   ├── p4_results.py                    # Graphics and Tables
│   ├── p5_get_pnad.py                   # PNAD: Panel Matching
│   └── p6_get_pnad.R                    # PNAD: Calibration Data
├── data/
│   ├── pnad/                            # PNAD Data
│   ├── lorenz_nw_scf_2019.raw           # SCF Data for US Lorenz Curve
│   └── data.csv                         # Not yet
├── output/
│   ├── figures/                         # Figures
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

The analysis was conducted using Python version 3.14 and R version 4.6.0 (2026-06-09) on a Windows 11 system.

## Running the project

To reproduce the analysis:

* Open the project's folder as a project.
* Open the file: code/main.py.
* In the Terminal, type "pip install -r requirements.txt" on bash (Ctrl+Shift+' to open).
* Run the script: code/main.py.
