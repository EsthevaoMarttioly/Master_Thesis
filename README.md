# Master's Thesis

Author: **Esthevao Marttioly Lopes Martins**
Program: MSc Economics — FGV EESP

This repository contains the code and results for the **Master's Thesis** by Esthevão Marttioly from FGV EESP.

All code is written with reproducibility and defensive programming in mind.

## Project Structure

```
├── .venv                                # Package version lockfile
├── code/
│   ├── main.py                          # Master script - runs everything
│   ├── household_block.py               # Household's heterogenous block
│   ├── other_blocks.py                  # Firm, Fiscal, Monetary, and Mkt Clearing
│   └── parameters.R                     # Calibration and Estimation Parameters
├── data/
│   └── thesis_data.csv
├── output/
├── EsthevaoMarttioly_Thesis.pdf         # Final submitted report
├── requirements.txt                     # pip install -r requirements.txt
└── README.md
```


## Computational Environment

Open the 'Visual Studio Code' to run Python.

If you do not have either VS Code or Python, install it.
* To install python language _https://www.python.org_.
* To install VSCode _https://code.visualstudio.com_.
* Important to install python inside VSCode as well.

The analysis was conducted using Python version 3.14 (2026-04-21) on a Windows 11 system.

## Running the project

To reproduce the analysis:

* Open the file: code/main.py (with VS Code)
* Open the project's folder in VS Code.
* In the Terminal, type "pip install -r requirements.txt" on bash (Ctrl+Shift+' to open)
* Run the script: code/main.py.