#### Thesis - Targeted vs Universal Fiscal Policy ####
## Author:  Esthevao Marttioly  |  EESP-FGV  |  2026
## Advisor: Bernardo Guimarães
#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# This program solves an one-asset HANK model with 3-state unemployment.
#
# Solution method: Sequence-Space Jacobian (Auclert et al. 2021).
# https://github.com/shade-econ/sequence-jacobian
# ---------------------------------------------------------------------------
#=
# Write this in the terminal this to install packages
# pip install -r requirements.txt


# Import Packages
import random
import time
import numpy as np

from sequence_jacobian import create_model


# Set a seed for future replications
random.seed(20260415)


# Import parameters
from code.parameters import calibration


# Import blocks
from code.household_block import hh
from code.other_blocks import (firm, nkpc_ss, price_nkpc, wage_nkpc,
                               monetary, fiscal, mkt_clearing)


# Import results
from code.results import (print_ss_summary, plot_consumption_policy,
                          plot_wealth_distribution, plot_impc_profiles,
                          plot_irfs, plot_irf_comparison,
                          compute_multipliers, print_multipliers)


# ---------------------------------------------------------------------------
# Assemble Model - Steady State

blocks_ss = [hh, firm, nkpc_ss, monetary, fiscal, mkt_clearing]
hank_ss   = create_model(blocks_ss, name="HANK - Steady State")

print(f"\nModel inputs (SS):  {hank_ss.inputs}")
print(f"Model outputs (SS): {hank_ss.outputs}")


# Target asset market + goods market clearing
targets_ss = {
    'asset_mkt': 0,
#     'goods_mkt': 0,
    'labor_mkt': 0,    # adjust Z to have L = Y/Z = 1-U-V
#     'nkpc': 0,       # solved by nkpc_ss directly
#     'nkwpc': 0,
}

# Unknown values to be estimated
unknowns_ss = {
    'Z' : 1.0,          # productivity - solved by labor market
#     'w' : 0.7,          # real wage - solved by NKPC
    'beta_high': 0.97,  # patient's discount factor - solved by asset market
}

# Steady State
start = time.time()
ss0 = hank_ss.solve_steady_state(calibration, unknowns = unknowns_ss,
                                 targets = targets_ss, solver = 'hybr')
print(f"Steady State solved in {time.time()-start:.1f}s")    # 6.4 seconds on my laptop



# ---------------------------------------------------------------------------
# Assemble Model - Dynamics

blocks = [hh, firm, price_nkpc, wage_nkpc, monetary, fiscal, mkt_clearing]
hank   = create_model(blocks, name="HANK - Dynamics")

print(f"\nModel inputs (Dynamics):  {hank.inputs}")
print(f"Model outputs (Dynamics): {hank.outputs}")


# Verify ss0 is also a valid Steady State
ss = hank.steady_state(ss0)
for k in ss0.keys():
    assert np.all(np.isclose(ss[k], ss0[k])), f"SS mismatch at key {k}"
print("Steady State reached in dynamics DAG.")



# ---------------------------------------------------------------------------
# Steady State Diagnostics
print_ss_summary(ss, calibration)


# ---------------------------------------------------------------------------
# Figures: steady-state distribution and policy functions
lorenz_scf_raw = np.loadtxt('data/lorenz_nw_scf_2019.raw', delimiter=',')

plot_consumption_policy(ss, calibration, patient_type='P', savepath='output/consumption_policy.png')
plot_consumption_policy(ss, calibration, patient_type='I', savepath='output/consumption_policy_impatient.png')

plot_wealth_distribution(ss, lorenz_scf_raw, savepath='output/wealth_distribution.png')


# ---------------------------------------------------------------------------
# GE Jacobians
# Unknowns: {Y, pi, w}   Targets: {goods_mkt, nkpc, nkwpc}
# Inputs:   {b, Tr, Z}
# ---------------------------------------------------------------------------
T = 100

unknowns_dyn = ['Y', 'pi', 'w']
targets_dyn  = ['goods_mkt', 'nkpc', 'nkwpc']
inputs_dyn   = ['b', 'Tr', 'Z']

G = hank.solve_jacobian(ss, unknowns_dyn, targets_dyn, inputs_dyn, T=T)


# ---------------------------------------------------------------------------
# PE Jacobian (hold r, w, Div fixed — isolate direct HH response)
# ---------------------------------------------------------------------------
G_hh = hh.jacobian(ss, inputs=['b', 'Tr', 'r', 'w'], T=T)

plot_impc_profiles(G_hh, T_plot=30, savepath='output/impc_profiles.png')


# ---------------------------------------------------------------------------
# IRFs: Employment-targeted (b shock) vs Needy-targeted (Tr shock)
# AR(1) shock with rho = 0.4, size = 1%
# ---------------------------------------------------------------------------
rho_sh = 0.40
db  = 0.01 * rho_sh ** np.arange(T)
dTr = 0.01 * rho_sh ** np.arange(T)

irf_b  = {v: G[v]['b']  @ db  for v in ['Y', 'C', 'U', 'V', 'w', 'pi']}
irf_Tr = {v: G[v]['Tr'] @ dTr for v in ['Y', 'C', 'U', 'V', 'w', 'pi']}

plot_irfs(irf_b, T_plot=30,
          title='GE IRFs: Employment-Targeted (b shock)',
          savepath='output/irf_b.png')

plot_irf_comparison({'Benefit $b$ (U-targeted)': irf_b,
                     'Safety-net $Tr$ (V-targeted)': irf_Tr},
                     variables=['Y', 'C'], T_plot=30, savepath='output/irf_comparison.png')


# ---------------------------------------------------------------------------
# Fiscal multipliers summary table

mults = compute_multipliers(G, {'b': db, 'Tr': dTr})
print_multipliers(mults)

