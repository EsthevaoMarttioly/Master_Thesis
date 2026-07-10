#### Thesis - Conditional Cash Transfers: A HANK Approach ####
## Author:  Esthevao Marttioly  |  EESP-FGV  |  2026
## Advisor: Bernardo Guimarães
#=
# ---------------------------------------------------------------------------
# DESCRIPTION
# This program solves an one-asset HANK model with endogenous informality.
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
from code.household_block import hh, solve_ss, solve_dyn


# ---------------------------------------------------------------------------
# Steady State
hank_ss = create_model([hh, firm_formal, firm_informal, informal_wage,
                        nkpc_ss, union_ss, monetary, fiscal, mkt_clearing])

print(f"\nModel inputs (SS):  {hank_ss.inputs}")
print(f"Model outputs (SS): {hank_ss.outputs}")


ss = solve_ss(hank_ss, calibration, unknowns, targets, verbose=True)


# Steady State Diagnostics
print_ss_summary(ss, ['Y', 'Y_I', 'C_GHH', 'C', 'beta_high', 'A', 'B',
                      'psi', 'w', 'w_I', 'Z', 'F', 'I', 'U', 'BF', 'Div',
                      'G', 'asset_mkt', 'goods_mkt', 'labor_mkt', 'wage_nkpc'])


# ---------------------------------------------------------------------------
# No-BF Counterfactual
ss_nobf = solve_ss(hank_ss, {**calibration, 'Tr': 0.0}, unknowns, targets, verbose=True)


compare_bf_ss(ss, ss_nobf, savepath='output/tables/ss_comparison.tex')

consumption_by_state(ss, savepath='output/figures/consump_by_state.png')
formality_by_wealth(ss, savepath='output/figures/formality_wealth.png')
welfare_by_type(ss, ss_nobf, calibration, savepath='output/figures/welfare_type.png')


# Re-solves for different Tr (very slow)
# plot_bf_sweep(lambda cal: solve_ss(hank_ss, cal, unknowns, targets, verbose=False),
#               calibration, np.linspace(0.0, 0.20, 6), savepath='output/figures/bf_sweep.png')


# ---------------------------------------------------------------------------
# Steady State Distribution and Policy Functions
lorenz_scf_raw = np.loadtxt('data/lorenz_nw_scf_2019.raw', delimiter=',')

plot_consumption_policy(ss, calibration, savepath='output/figures/consump_policy.png')
plot_wealth_distribution(ss, lorenz_scf_raw, n_bins=30, savepath='output/figures/wealth_distribution.png')


# ---------------------------------------------------------------------------
# Dynamics
hank = create_model([hh, firm_formal, firm_informal, informal_wage,
                     phillips_curve, monetary, fiscal, mkt_clearing])

print(f"\nModel inputs (Dynamics):  {hank.inputs}")
print(f"Model outputs (Dynamics): {hank.outputs}")


dyn = hank.steady_state(ss)

# Verify ss0 is also a valid Steady State
for k in dyn.keys():
    assert np.all(np.isclose(dyn[k], ss[k], atol=1e-4)), f"SS mismatch at key {k}"
print("Steady State reached in dynamics DAG.")


# ---------------------------------------------------------------------------
# Partial Equilibrium Jacobians
T = 100
G_hh = hh.jacobian(dyn, inputs=['Tr', 'r'], T=T)
plot_impc(G_hh, T_plot=30, savepath='output/figures/impc.png')


# General Equilibrium Jacobians
# IRFs:   AR(1) shock,      with   rho = 0.4,  size = 1%,  1 year antecipation
unknowns_dyn = ['Y', 'pi', 'w']
targets_dyn  = ['goods_mkt', 'nkpc', 'wage_nkpc']   # gov_budget, unknown B
variables    = ['C', 'Y', 'L', 'I', 'U', 'BF', 'pi', 'w']


dTr0 = 0.01; rho_Tr = 0.40; k = 4
dTr  = dTr0 * rho_Tr ** np.arange(T)                # MIT Shock
dTr_ant  = np.concatenate([np.zeros(k), dTr[:-k]])  # Antecipated Shock: Perfect Foresight
dTr_perm = dTr0 * np.ones(T)                        # Permanent Shock


# Build IRFs
irf_insu = solve_dyn(hank, ss, unknowns_dyn, targets_dyn,
                     dTr, calibration, variables, moving=False, verbose=True)

irf_full = solve_dyn(hank, ss, unknowns_dyn, targets_dyn,
                     dTr, calibration, variables, moving=True, verbose=True)

irf_ant  = solve_dyn(hank, ss, unknowns_dyn, targets_dyn,
                     dTr_ant, calibration, variables, moving=True, verbose=True)

irf_comp = {v: irf_full[v] - irf_insu[v] for v in variables}


# Graphics
plot_channel_decomposition(irf_insu, irf_full, ['C','I','U','BF','pi','w'],
                           savepath='output/figures/irf_decomposition.png')  # Adicionar partial equilibrium


plot_irf({'MIT Shock': irf_full, 'Antecipated Shock': irf_ant},
         variables=['C','I','U','BF','pi','w'], T_plot=30,
         savepath='output/figures/irf_antecipated.png')


# ---------------------------------------------------------------------------
# Fiscal Multipliers Summary Table
cumulative_response_table(irf_insu, irf_full, var='C',
                          savepath='output/tables/response_consump.tex')




