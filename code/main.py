#### Thesis - Conditional Cash Transfers: A HANK Approach ####
## Author:  Esthevao Marttioly  |  EESP-FGV  |  2026
## Advisor: Bernardo Guimarães
#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# This program solves an one-asset HANK model with 3-state formality-status.
# ---------------------------------------------------------------------------
#=
# Write this in the terminal to install packages
# pip install -r requirements.txt


# ---- Packages --------------------------------------------------------------
import random
import numpy as np
from sequence_jacobian import create_model

random.seed(20260415)


# Import parameters
from code.parameters import *
from code.other_blocks import *
from code.results import *
from code.household_block import hh, solve_ss


# ---------------------------------------------------------------------------
# Steady State
hank_ss = create_model([hh, firm_formal, firm_informal, informal_wage,
                        nkpc_ss, union_ss, monetary, fiscal, mkt_clearing])

print(f"\nModel inputs (SS):  {hank_ss.inputs}")
print(f"Model outputs (SS): {hank_ss.outputs}")


ss0 = solve_ss(hank_ss, calibration, unknowns, targets, verbose=True)



# ---------------------------------------------------------------------------
# Dynamics
hank = create_model([hh, firm_formal, firm_informal, informal_wage,
                     phillips_curve, monetary, fiscal, mkt_clearing])

print(f"\nModel inputs (Dynamics):  {hank.inputs}")
print(f"Model outputs (Dynamics): {hank.outputs}")


# Verify ss0 is also a valid Steady State
ss = hank.steady_state(ss0)
for k in ss0.keys() - {'union_ss','sig','delta_F','delta_I','pi_F','pi_I'}:
    assert np.all(np.isclose(ss[k], ss0[k])), f"SS mismatch at key {k}"
print("Steady State reached in dynamics DAG.")



# ---------------------------------------------------------------------------
# Steady State Diagnostics

print_ss_summary(ss, ['Y', 'Y_I', 'C_GHH', 'C', 'beta_high', 'A', 'B',
                      'psi', 'w', 'w_I', 'Z', 'F', 'I', 'U', 'BF', 'Div',
                      'G', 'asset_mkt', 'goods_mkt', 'labor_mkt', 'wage_nkpc'])



# ---------------------------------------------------------------------------
# Informal Worked Hours Statistics

print(f"\nAverage informal hours = {ss.internals['household']['h_I'][:7].mean()}")
print(f"Std of informal hours = {ss.internals['household']['h_I'].std()}")



# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# 1. Steady State Distribution and Policy Functions
lorenz_scf_raw = np.loadtxt('data/lorenz_nw_scf_2019.raw', delimiter=',')

plot_consumption_policy(ss, calibration, savepath='output/consumption_policy.png')
plot_wealth_distribution(ss, lorenz_scf_raw, n_bins=30, savepath='output/wealth_distribution.png')



# 2. General Equilibrium Jacobians
T = 100
unknowns_dyn = ['Y', 'pi', 'w']
targets_dyn  = ['goods_mkt', 'nkpc', 'wage_nkpc']
inputs_dyn   = ['Tr']

G = hank.solve_jacobian(ss, unknowns_dyn, targets_dyn, inputs_dyn, T=T)


# 3. Partial Equilibrium Jacobians
G_hh = hh.jacobian(ss, inputs=['Tr', 'r'], T=T)

plot_impc_profiles(G_hh, T_plot=30, savepath='output/impc_profiles.png')



# 4. Impulse Response Functions
# AR(1) shock with rho = 0.4, size = 1%
k = 4              # 1 year antecipation horizon
dTr0   = 0.01
rho_sh = 0.40

dTr_mit  = dTr0 * rho_sh ** np.arange(T)                # MIT Shock
dTr_ant  = np.concatenate([np.zeros(k), dTr_mit[:-k]])  # Antecipated Shock: Perfect Foresight
dTr_perm = dTr0 * np.ones(T)                            # Permanent Shock

variables = ['Y', 'C', 'L', 'BF', 'pi', 'w']
irf_mit  = {v: G[v]['Tr'] @ dTr_mit  for v in variables}
irf_ant  = {v: G[v]['Tr'] @ dTr_ant  for v in variables}
irf_perm = {v: G[v]['Tr'] @ dTr_perm for v in variables}


plot_irfs(irf_mit, T_plot=30,
          title='GE IRFs: Conditional Transfer Targeted (Tr shock)',
          savepath='output/irf_Tr.png')

plot_irf_comparison({'MIT shock (surprise)'      : irf_mit,
                     f'Anticipated ({k}q ahead)' : irf_ant},
                     variables=['Y', 'C', 'pi', 'w'],
                     T_plot=30, savepath='output/irf_comparison.png')



# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
# Fiscal multipliers summary table
mults = compute_multipliers(G, {'Tr': dTr_mit})
print_multipliers(mults)


